#!/usr/bin/env python3
"""Run the frozen Spring v3.1 + FFS seed-42 experiment matrix.

This entry point is intentionally separate from the historical S0--S6
screening runner.  It freezes the common Spring domain before launching any
cache or optimizer work and refuses to reuse artifacts that do not match that
domain.  F7 remains a machine-readable blocked optional arm until a native
v3.1 Stage-C implementation exists.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cache_dataset import sha256_file  # noqa: E402
from data.endpoint_selection import load_endpoint_index  # noqa: E402
from data.manifest import load_manifest  # noqa: E402
from data.spring import SPRING_GT_COMPONENT, SPRING_GT_TARGET_TYPE  # noqa: E402
from data.stereo_calibration import (  # noqa: E402
    load_rectified_calibration_sidecar,
)
from utils.checkpoint import repository_git_hash  # noqa: E402


SCHEMA_VERSION = 1
COMPONENT = "spring-v3.1-ffs-seed42-orchestrator"
PROTOCOL = "spring_seed42_common_fixed384_v1"
SEED = 42
ARM_ORDER = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
RUNNABLE_ARMS = ARM_ORDER[:-1]
VALIDATION_SEQUENCES = (
    "0005",
    "0010",
    "0015",
    "0021",
    "0023",
    "0030",
    "0032",
    "0047",
)
EXPECTED_ALL_RECORDS = 5_000
EXPECTED_TRAIN_RECORDS = 3_650
EXPECTED_VALIDATION_RECORDS = 1_350
ENDPOINT_WARMUP_FRAMES = 6
EXPECTED_ENDPOINT_COUNT = 1_302
EXPECTED_ENDPOINT_ID_SHA256 = (
    "aa6ba30295b8d5ab0e1b4326a14fae61f9c8ec42641801cd8442097bc3ab5b57"
)
FFS_UPSTREAM_COMMIT = "a290ba04c1b3ad1ec41a33974a157b2917b624d4"
FFS_CHECKPOINT_SHA256 = (
    "98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692"
)
VGGT_UPSTREAM_COMMIT = "282ec70363edeff59424bf43731658092fba3d37"
VGGT_CHECKPOINT_SHA256 = (
    "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
)
CROP_SIZE_HW = (384, 768)
CROP_ORIGIN_XY = (576, 348)
CACHE_IDENTITY_FIELDS = frozenset(
    {
        "component",
        "upstream_commit",
        "checkpoint_sha256",
        "torch_version",
        "cuda_version",
        "config_sha256",
    }
)
SPRING_TARGET = {
    "type": SPRING_GT_TARGET_TYPE,
    "cache_component": SPRING_GT_COMPONENT,
    "paper_gt": True,
    "paper_accuracy": False,
    "synthetic_ground_truth": True,
}

ARM_MODELS = {
    "F0": "full-resolution FFS",
    "F1": "half-resolution FFS + bilinear",
    "F2": "half-resolution FFS + v3.1 T1",
    "F3": "half-resolution FFS + v2/K=2 T3 control, GT pose",
    "F4": "half-resolution FFS + v3.1 T3, GT pose",
    "F5": "F4 + VGGT depth prior, GT pose",
    "F6": "F5 + VGGT pose",
    "F7": "F6 + Stage C",
}
ARM_POSE_SOURCES = {
    "F0": "none",
    "F1": "none",
    "F2": "gt",
    "F3": "gt",
    "F4": "gt",
    "F5": "gt",
    "F6": "vggt",
    "F7": "vggt",
}
ARM_PRIMARY_METHODS = {
    "F0": "FFS_full",
    "F1": "FFS_half_bilinear",
    "F2": "T1",
    "F3": "T3",
    "F4": "T3",
    "F5": "T3_VGGT",
    "F6": "T3_VGGT",
    "F7": None,
}

ARM_CONFIGS = {
    arm: f"configs/spring_v3_1/{arm}.yaml" for arm in ARM_ORDER
}
F3_INITIALIZER_CONFIG = "configs/spring_v3_1/F3_stage_a_control.yaml"

TRAINING_CONFIG_CONTRACTS: dict[str, dict[str, Any]] = {
    "F2": {
        "stage": "spatial",
        "sequence_length": 1,
        "derived_contract": "calibrated_stereo_v2",
        "pose_source": "gt",
        "use_vggt_pose": False,
        "use_vggt_depth": False,
        "top_k": 4,
        "steps": 5_000,
    },
    "F3_stage_a_control": {
        "stage": "spatial",
        "sequence_length": 1,
        "derived_contract": "legacy_v1",
        "pose_source": "gt",
        "use_vggt_pose": False,
        "use_vggt_depth": False,
        "top_k": 2,
        "steps": 5_000,
    },
    "F3": {
        "stage": "temporal",
        "sequence_length": 3,
        "derived_contract": "legacy_v1",
        "pose_source": "gt",
        "use_vggt_pose": False,
        "use_vggt_depth": False,
        "top_k": 2,
        "steps": 15_000,
    },
    "F4": {
        "stage": "temporal",
        "sequence_length": 3,
        "derived_contract": "calibrated_stereo_v2",
        "pose_source": "gt",
        "use_vggt_pose": False,
        "use_vggt_depth": False,
        "top_k": 4,
        "steps": 15_000,
    },
    "F5": {
        "stage": "temporal",
        "sequence_length": 3,
        "derived_contract": "calibrated_stereo_v2",
        "pose_source": "gt",
        "use_vggt_pose": False,
        "use_vggt_depth": True,
        "top_k": 4,
        "steps": 15_000,
    },
    "F6": {
        "stage": "temporal",
        "sequence_length": 3,
        "derived_contract": "calibrated_stereo_v2",
        "pose_source": "vggt",
        "use_vggt_pose": True,
        "use_vggt_depth": True,
        "top_k": 4,
        "steps": 15_000,
    },
}


class SpringV31Error(RuntimeError):
    """A fail-closed protocol, lineage, or subprocess error."""


@dataclass(frozen=True, slots=True)
class Paths:
    project_root: Path
    spring_root: Path
    output_root: Path
    all_manifest: Path
    train_manifest: Path
    validation_manifest: Path
    split_receipt: Path
    endpoint_index: Path
    pixel_audit: Path
    train_calibration: Path
    validation_calibration: Path
    train_half_observation: Path
    validation_half_observation: Path
    validation_full_observation: Path
    train_ground_truth: Path
    validation_ground_truth: Path
    train_vggt: Path
    validation_vggt: Path
    train_legacy_derived: Path
    validation_legacy_derived: Path
    train_calibrated_derived: Path
    validation_calibrated_derived: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Paths":
        project_root = args.project_root.expanduser().resolve()
        output_root = args.output_root.expanduser().resolve()
        manifests = output_root / "manifests"
        cache = output_root / "cache"
        calibration = output_root / "calibration"
        return cls(
            project_root=project_root,
            spring_root=args.spring_root.expanduser().resolve(),
            output_root=output_root,
            all_manifest=manifests / "all.jsonl",
            train_manifest=manifests / "train.jsonl",
            validation_manifest=manifests / "validation.jsonl",
            split_receipt=manifests / "split_receipt.json",
            endpoint_index=manifests / "common_fixed384_endpoints.json",
            pixel_audit=output_root / "audits" / "pixel_rectification.json",
            train_calibration=calibration / "train.jsonl",
            validation_calibration=calibration / "validation.jsonl",
            train_half_observation=cache / "train" / "observation",
            validation_half_observation=cache / "validation" / "observation",
            validation_full_observation=(
                cache / "validation" / "observation_full_resolution"
            ),
            train_ground_truth=cache / "train" / "teacher",
            validation_ground_truth=cache / "validation" / "teacher",
            train_vggt=cache / "train" / "vggt",
            validation_vggt=cache / "validation" / "vggt",
            train_legacy_derived=cache / "train" / "derived_f3_v2_gt_pose_no_depth",
            validation_legacy_derived=(
                cache / "validation" / "derived_f3_v2_gt_pose_no_depth"
            ),
            train_calibrated_derived=(
                cache / "train" / "derived_v31_calibrated_vggt"
            ),
            validation_calibrated_derived=(
                cache / "validation" / "derived_v31_calibrated_vggt"
            ),
        )

    def arm_train(self, arm: str) -> Path:
        return self.output_root / "arms" / arm / "train"

    def arm_eval(self, arm: str) -> Path:
        return self.output_root / "arms" / arm / "eval"

    @property
    def f3_initializer_train(self) -> Path:
        return self.output_root / "arms" / "F3" / "stage_a_control"


@dataclass(frozen=True, slots=True)
class Job:
    name: str
    command: tuple[str, ...]
    expected_output: Path
    group: str
    gpu: bool = True


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpringV31Error(f"cannot read {name}: {path}") from exc
    if not isinstance(value, Mapping):
        raise SpringV31Error(f"{name} must be a JSON object: {path}")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_finite_number(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (not positive or number > 0.0)


def _same_resolved_path(value: object, expected: Path) -> bool:
    return (
        isinstance(value, str)
        and Path(value).expanduser().resolve() == expected.expanduser().resolve()
    )


def _metric_evidence(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpringV31Error(f"metric {name} is missing or malformed")
    metric_value = value.get("value")
    numerator = value.get("numerator")
    count = value.get("count")
    if (
        value.get("valid") is not True
        or not _is_finite_number(metric_value)
        or not _is_finite_number(numerator)
        or not _is_positive_int(count)
        or not math.isclose(
            float(metric_value),
            float(numerator) / int(count),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise SpringV31Error(f"metric {name} has invalid numerator/count evidence")
    return {
        "value": float(metric_value),
        "numerator": float(numerator),
        "count": int(count),
        "valid": True,
    }


def _validate_cache_identity(
    identity: Mapping[str, Any],
    name: str,
    *,
    component: str,
    expected_commit: str | None,
    expected_checkpoint: str | None,
) -> None:
    if set(identity) != CACHE_IDENTITY_FIELDS:
        raise SpringV31Error(f"{name} cache identity fields differ")
    for field in CACHE_IDENTITY_FIELDS:
        value = identity.get(field)
        if field.endswith("_sha256") and not _is_sha256(value):
            raise SpringV31Error(f"{name} cache identity {field} is malformed")
        if field == "upstream_commit" and (
            not isinstance(value, str) or not value
        ):
            raise SpringV31Error(f"{name} cache identity upstream_commit is malformed")
    if identity.get("component") != component:
        raise SpringV31Error(
            f"{name} cache component differs: {identity.get('component')!r}"
        )
    if expected_commit is not None and identity.get("upstream_commit") != expected_commit:
        raise SpringV31Error(f"{name} cache upstream commit differs")
    if expected_checkpoint is not None and identity.get("checkpoint_sha256") != expected_checkpoint:
        raise SpringV31Error(f"{name} cache checkpoint SHA differs")


def _command_text(command: Sequence[str]) -> str:
    return shlex.join([str(value) for value in command])


def _parse_arms(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ARM_ORDER
    requested: set[str] = set()
    for value in values:
        for item in value.replace(",", " ").split():
            arm = item.strip().upper()
            if arm == "ALL":
                requested.update(ARM_ORDER)
            elif arm not in ARM_ORDER:
                raise SpringV31Error(f"unknown arm: {item!r}")
            else:
                requested.add(arm)
    return tuple(arm for arm in ARM_ORDER if arm in requested)


def _parse_devices(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise SpringV31Error("--devices must not be empty")
    if items == ("cpu",):
        return items
    if "cpu" in items or any(not item.isdigit() for item in items):
        raise SpringV31Error("--devices must be 'cpu' or comma-separated CUDA indices")
    if len(set(items)) != len(items):
        raise SpringV31Error("--devices contains duplicates")
    return items


def _device_argument(devices: Sequence[str]) -> str:
    return "cpu" if devices == ("cpu",) else "cuda"


def _path_identity(path: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.exists(),
    }
    if path.is_file():
        identity.update({"size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return identity


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_checkout_state(path: Path, name: str) -> dict[str, Any]:
    if not path.is_dir():
        raise SpringV31Error(f"{name} checkout is missing: {path}")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError as exc:
        raise SpringV31Error(f"cannot inspect {name} checkout: {path}") from exc
    return {"path": str(path.resolve()), "commit": head, "dirty_paths": dirty}


def _backbone_protocol(paths: Paths, args: argparse.Namespace, arms: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if any(arm in RUNNABLE_ARMS for arm in arms):
        checkpoint = args.ffs_checkpoint.resolve()
        if not checkpoint.is_file():
            raise SpringV31Error(f"FFS checkpoint is missing: {checkpoint}")
        checkpoint_sha = sha256_file(checkpoint)
        checkout = _git_checkout_state(args.ffs_repo.resolve(), "Fast-FoundationStereo")
        if checkpoint_sha != FFS_CHECKPOINT_SHA256:
            raise SpringV31Error(
                f"FFS checkpoint SHA-256 differs: expected {FFS_CHECKPOINT_SHA256}, "
                f"got {checkpoint_sha}"
            )
        if checkout["commit"] != FFS_UPSTREAM_COMMIT or checkout["dirty_paths"]:
            raise SpringV31Error(
                "Fast-FoundationStereo must be the pinned clean checkout "
                f"{FFS_UPSTREAM_COMMIT}"
            )
        result["ffs"] = {
            "checkpoint": _path_identity(checkpoint),
            "checkout": checkout,
        }
    if any(arm in ("F4", "F5", "F6") for arm in arms):
        checkpoint = args.vggt_checkpoint.resolve()
        if not checkpoint.is_file():
            raise SpringV31Error(f"VGGT checkpoint is missing: {checkpoint}")
        checkpoint_sha = sha256_file(checkpoint)
        checkout = _git_checkout_state(args.vggt_repo.resolve(), "VGGT-Omega")
        if checkpoint_sha != VGGT_CHECKPOINT_SHA256:
            raise SpringV31Error(
                f"VGGT checkpoint SHA-256 differs: expected {VGGT_CHECKPOINT_SHA256}, "
                f"got {checkpoint_sha}"
            )
        if checkout["commit"] != VGGT_UPSTREAM_COMMIT or checkout["dirty_paths"]:
            raise SpringV31Error(
                f"VGGT-Omega must be the pinned clean checkout {VGGT_UPSTREAM_COMMIT}"
            )
        result["vggt"] = {
            "checkpoint": _path_identity(checkpoint),
            "checkout": checkout,
        }
    return result


def _config_protocol(paths: Paths) -> dict[str, Any]:
    """Resolve every trainable arm and freeze the intentional differences."""

    if paths.project_root != PROJECT_ROOT.resolve():
        raise SpringV31Error(
            "--project-root must identify the checkout containing this runner; "
            "cross-checkout imports would make source lineage ambiguous"
        )
    from train import (  # Imported lazily so ``--help`` remains lightweight.
        resolve_config,
        supervision_target_from_config,
        temporal_pose_source_from_config,
        training_stage,
    )

    result: dict[str, Any] = {}
    for name, expected in TRAINING_CONFIG_CONTRACTS.items():
        relative = F3_INITIALIZER_CONFIG if name == "F3_stage_a_control" else ARM_CONFIGS[name]
        path = (paths.project_root / relative).resolve()
        config = resolve_config(path)
        stage = training_stage(config)
        pose_source = temporal_pose_source_from_config(config)
        supervision = supervision_target_from_config(config)
        observed = {
            "seed": int(config.seed),
            "arm": config.get("arm"),
            "stage": stage,
            "sequence_length": int(config.data.sequence_length),
            "derived_contract": str(config.data.derived_contract),
            "source_dataset": str(config.data.source_dataset),
            "pose_source": pose_source,
            "use_vggt_pose": bool(config.model.use_vggt_pose),
            "use_vggt_depth": bool(config.model.use_vggt_depth),
            "top_k": int(config.temporal_history_v2.top_k),
            "steps": int(
                config.train.steps_spatial if stage == "spatial" else config.train.steps
            ),
            "supervision_component": supervision.cache_component,
            "paper_ground_truth": supervision.paper_ground_truth,
            "synthetic_ground_truth": supervision.synthetic_ground_truth,
        }
        required = {
            **expected,
            "seed": SEED,
            "arm": None if name == "F3_stage_a_control" else name,
            "source_dataset": "Spring-v2",
            "supervision_component": "spring-ground-truth",
            "paper_ground_truth": True,
            "synthetic_ground_truth": True,
        }
        differences = {
            key: {"expected": value, "actual": observed.get(key)}
            for key, value in required.items()
            if observed.get(key) != value
        }
        is_v31 = name != "F3" and name != "F3_stage_a_control"
        calibration = config.calibration_conditioning_v3
        v31_contracts = {
            "pixel_center_contract": calibration.get("pixel_center_contract"),
            "measurement_ownership": config.get("measurement_ownership_v3_1"),
            "candidate_fusion": config.get("temporal_candidate_fusion_v3_1"),
        }
        if is_v31:
            if (
                v31_contracts["pixel_center_contract"]
                != "align_corners_false_half_pixel_v3_1"
                or not bool(v31_contracts["measurement_ownership"].enabled)
                or v31_contracts["measurement_ownership"].protocol_version
                != "lr_center_projection_bounded_subpixel_v3_1"
                or not bool(v31_contracts["candidate_fusion"].enabled)
                or v31_contracts["candidate_fusion"].protocol_version
                != "current_conditioned_age_phase_diverse_v3_1"
            ):
                differences["v3_1_contracts"] = {
                    "expected": "half-pixel + LR ownership + current-conditioned fusion",
                    "actual": str(v31_contracts),
                }
        if differences:
            raise SpringV31Error(f"{name} config differs from the frozen arm: {differences}")
        result[name] = {"path": str(path), "sha256": sha256_file(path), **observed}

    # F0/F1 are consumed by the baseline evaluator, not train.py. Their YAMLs
    # still form public evidence and must freeze the FFS resolution contract.
    from omegaconf import OmegaConf

    for name, scale, component, prediction in (
        ("F0", 1, "ffs-observation-full-resolution", "direct_full_resolution_ffs"),
        ("F1", 2, "ffs-observation", "align_corners_false_bilinear"),
    ):
        path = (paths.project_root / ARM_CONFIGS[name]).resolve()
        config = OmegaConf.load(path)
        observed = {
            "seed": int(config.seed),
            "arm": str(config.arm),
            "stage": str(config.stage),
            "scale": int(config.ffs.spatial_scale),
            "component": str(config.ffs.cache_component),
            "iterations": int(config.ffs.iterations),
            "max_disp": int(config.ffs.max_disp),
            "right_left_check": bool(config.ffs.right_left_check),
            "prediction": str(config.evaluation.prediction),
            "endpoint_protocol": str(config.evaluation.endpoint_protocol),
            "crop_size_hw": list(config.evaluation.crop_hr),
            "crop_origin_xy": list(config.evaluation.crop_origin_hr_xy),
        }
        required = {
            "seed": SEED,
            "arm": name,
            "stage": "baseline",
            "scale": scale,
            "component": component,
            "iterations": 4,
            "max_disp": 384 if scale == 1 else 192,
            "right_left_check": True,
            "prediction": prediction,
            "endpoint_protocol": PROTOCOL,
            "crop_size_hw": list(CROP_SIZE_HW),
            "crop_origin_xy": list(CROP_ORIGIN_XY),
        }
        differences = {
            key: {"expected": value, "actual": observed.get(key)}
            for key, value in required.items()
            if observed.get(key) != value
        }
        if differences:
            raise SpringV31Error(f"{name} config differs from the frozen arm: {differences}")
        result[name] = {"path": str(path), "sha256": sha256_file(path), **observed}
    f7_path = (paths.project_root / ARM_CONFIGS["F7"]).resolve()
    f7 = OmegaConf.load(f7_path)
    if (
        int(f7.seed) != SEED
        or str(f7.arm) != "F7"
        or str(f7.stage) != "optional_blocked"
        or str(f7.depends_on) != "F6"
        or str(f7.required_contract.pixel_center_contract)
        != "align_corners_false_half_pixel_v3_1"
        or str(f7.required_contract.measurement_ownership)
        != "lr_center_projection_bounded_subpixel_v3_1"
        or str(f7.required_contract.temporal_candidate_fusion)
        != "current_conditioned_age_phase_diverse_v3_1"
    ):
        raise SpringV31Error("F7 optional-blocked contract differs")
    result["F7"] = {
        "path": str(f7_path),
        "sha256": sha256_file(f7_path),
        "seed": SEED,
        "arm": "F7",
        "stage": "optional_blocked",
        "depends_on": "F6",
    }
    return {name: result[name] for name in ARM_ORDER if name in result} | {
        "F3_stage_a_control": result["F3_stage_a_control"]
    }


def _source_snapshot(paths: Paths) -> dict[str, Any]:
    relative_files = {
        "train.py",
        "eval.py",
        "src/data/spring.py",
        "src/data/stereo_calibration.py",
        "src/data/training_dataset.py",
        "src/data/temporal_training_dataset.py",
        "src/evaluation.py",
        "src/utils/checkpoint.py",
        "tools/run_spring_v3_1_ffs.py",
        "tools/build_spring_manifest.py",
        "tools/build_spring_sequence_split.py",
        "tools/build_spring_endpoint_index.py",
        "tools/cache_spring_ffs.py",
        "tools/cache_spring_gt.py",
        "tools/cache_vggt.py",
        "tools/build_spring_gt_geometry.py",
        "tools/derive_geometry_manifest.py",
        "tools/derive_geometry_cache.py",
        "tools/eval_spring_baseline.py",
        "tools/audit_epipolar_rectification.py",
        "tools/build_stereo_calibration.py",
        *ARM_CONFIGS.values(),
        F3_INITIALIZER_CONFIG,
    }
    for directory in (paths.project_root / "src/models", paths.project_root / "src/geometry"):
        relative_files.update(
            str(path.relative_to(paths.project_root)) for path in directory.rglob("*.py")
        )
    pending_configs = [value for value in relative_files if value.endswith(".yaml")]
    while pending_configs:
        relative = pending_configs.pop()
        config = paths.project_root / relative
        if not config.is_file():
            continue
        for line in config.read_text(encoding="utf-8").splitlines():
            if not line.strip().startswith("defaults_from:"):
                continue
            inherited = line.split(":", 1)[1].strip().split("#", 1)[0].strip()
            if inherited and inherited not in relative_files:
                relative_files.add(inherited)
                pending_configs.append(inherited)
    records: dict[str, Any] = {}
    for relative in sorted(relative_files):
        path = (paths.project_root / relative).resolve()
        if not path.is_file():
            raise SpringV31Error(f"source snapshot file is missing: {path}")
        records[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths.project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty_paths = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=paths.project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "git_head": git_head,
        "git_clean": not git_dirty_paths,
        "git_dirty_paths": git_dirty_paths,
        "files": records,
        "files_sha256": digest,
    }


def _manifest_protocol(paths: Paths) -> dict[str, Any]:
    for path in (paths.all_manifest, paths.train_manifest, paths.validation_manifest):
        if not path.is_file():
            raise SpringV31Error(f"manifest is missing: {path}")
    all_records = load_manifest(paths.all_manifest)
    train_records = load_manifest(paths.train_manifest)
    validation_records = load_manifest(paths.validation_manifest)
    expected_counts = (
        ("all", len(all_records), EXPECTED_ALL_RECORDS),
        ("train", len(train_records), EXPECTED_TRAIN_RECORDS),
        ("validation", len(validation_records), EXPECTED_VALIDATION_RECORDS),
    )
    for name, actual, expected in expected_counts:
        if actual != expected:
            raise SpringV31Error(
                f"{name} manifest count differs: expected {expected}, got {actual}"
            )
    train_sequences = {record.sequence_id for record in train_records}
    validation_sequences = {record.sequence_id for record in validation_records}
    if tuple(sorted(validation_sequences)) != VALIDATION_SEQUENCES:
        raise SpringV31Error(
            "validation sequence domain differs: "
            f"expected={list(VALIDATION_SEQUENCES)}, got={sorted(validation_sequences)}"
        )
    if train_sequences & validation_sequences:
        raise SpringV31Error("train and validation sequences overlap")
    expected_train = [
        record.to_dict()
        for record in all_records
        if record.sequence_id not in set(VALIDATION_SEQUENCES)
    ]
    expected_validation = [
        record.to_dict()
        for record in all_records
        if record.sequence_id in set(VALIDATION_SEQUENCES)
    ]
    if [record.to_dict() for record in train_records] != expected_train:
        raise SpringV31Error("train manifest is not the exact sequence partition of all.jsonl")
    if [record.to_dict() for record in validation_records] != expected_validation:
        raise SpringV31Error(
            "validation manifest is not the exact sequence partition of all.jsonl"
        )
    for name, records in (
        ("all", all_records),
        ("train", train_records),
        ("validation", validation_records),
    ):
        for record in records:
            if record.timestamp != float(record.frame_id - 1):
                raise SpringV31Error(
                    f"{name} manifest is not frame-index timestamped at "
                    f"{record.sequence_id}/{record.frame_id}"
                )
            if record.extras.get("timestamp_source") != "frame_index":
                raise SpringV31Error(
                    f"{name} manifest timestamp lineage is not frame_index"
                )
            if str(record.extras.get("dataset", "")).lower() != "spring":
                raise SpringV31Error(f"{name} manifest contains a non-Spring record")
    split = _strict_json(paths.split_receipt, "sequence split receipt")
    expected_split = {
        "schema_version": 1,
        "source_manifest": str(paths.all_manifest.resolve()),
        "source_manifest_sha256": sha256_file(paths.all_manifest),
        "train_manifest": str(paths.train_manifest.resolve()),
        "train_manifest_sha256": sha256_file(paths.train_manifest),
        "validation_manifest": str(paths.validation_manifest.resolve()),
        "validation_manifest_sha256": sha256_file(paths.validation_manifest),
        "train_records": EXPECTED_TRAIN_RECORDS,
        "validation_records": EXPECTED_VALIDATION_RECORDS,
        "validation_sequences": list(VALIDATION_SEQUENCES),
        "sequence_disjoint": True,
    }
    split_differences = {
        key: {"expected": value, "actual": split.get(key)}
        for key, value in expected_split.items()
        if split.get(key) != value
    }
    if split_differences:
        raise SpringV31Error(
            f"sequence split receipt differs from the frozen partition: {split_differences}"
        )
    return {
        "all": _path_identity(paths.all_manifest),
        "train": _path_identity(paths.train_manifest),
        "validation": _path_identity(paths.validation_manifest),
        "counts": {
            "all": len(all_records),
            "train": len(train_records),
            "validation": len(validation_records),
        },
        "train_sequences": sorted(train_sequences),
        "validation_sequences": sorted(validation_sequences),
        "sequence_disjoint": True,
        "timestamp_source": "frame_index",
        "split_receipt": _path_identity(paths.split_receipt),
    }


def _endpoint_protocol(paths: Paths) -> dict[str, Any]:
    selection = load_endpoint_index(
        paths.endpoint_index, manifest_path=paths.validation_manifest
    )
    if selection.count != EXPECTED_ENDPOINT_COUNT:
        raise SpringV31Error(
            "common endpoint count differs: "
            f"expected {EXPECTED_ENDPOINT_COUNT}, got {selection.count}"
        )
    if selection.entries_sha256 != EXPECTED_ENDPOINT_ID_SHA256:
        raise SpringV31Error(
            "common endpoint identity differs: "
            f"expected {EXPECTED_ENDPOINT_ID_SHA256}, got {selection.entries_sha256}"
        )
    records = load_manifest(paths.validation_manifest)
    positions: dict[str, int] = {}
    selected = set(selection.manifest_indices)
    expected: list[int] = []
    for index, record in enumerate(records):
        position = positions.get(record.sequence_id, 0)
        positions[record.sequence_id] = position + 1
        if position >= ENDPOINT_WARMUP_FRAMES:
            expected.append(index)
    if tuple(expected) != selection.manifest_indices or selected != set(expected):
        raise SpringV31Error("endpoint index is not the exact per-sequence warmup=6 domain")
    receipt_path = paths.endpoint_index.with_suffix(".receipt.json")
    receipt = _strict_json(receipt_path, "endpoint receipt")
    protocol = receipt.get("protocol")
    source = receipt.get("input")
    output = receipt.get("output")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("component") != "spring-common-endpoint-index-builder"
        or receipt.get("status") != "PASS"
        or not isinstance(protocol, Mapping)
        or protocol.get("sequence_warmup") != ENDPOINT_WARMUP_FRAMES
        or protocol.get("student_sequence_length") != 3
        or protocol.get("vggt_context_pairs") != 5
        or not isinstance(source, Mapping)
        or not isinstance(source.get("manifest"), Mapping)
        or source["manifest"].get("path") != str(paths.validation_manifest.resolve())
        or source["manifest"].get("sha256") != sha256_file(paths.validation_manifest)
        or not isinstance(output, Mapping)
        or output.get("path") != str(paths.endpoint_index.resolve())
        or output.get("file_sha256") != sha256_file(paths.endpoint_index)
        or output.get("endpoint_id_sha256") != EXPECTED_ENDPOINT_ID_SHA256
        or output.get("endpoint_count") != EXPECTED_ENDPOINT_COUNT
    ):
        raise SpringV31Error("endpoint receipt does not bind the frozen common domain")
    return {
        "path": str(selection.path),
        "file_sha256": selection.file_sha256,
        "manifest_sha256": selection.manifest_sha256,
        "endpoint_id_sha256": selection.entries_sha256,
        "count": selection.count,
        "warmup_frames_per_sequence": ENDPOINT_WARMUP_FRAMES,
        "receipt": _path_identity(receipt_path),
    }


def _audit_protocol(paths: Paths) -> dict[str, Any]:
    audit = _strict_json(paths.pixel_audit, "pixel rectification audit")
    if (
        audit.get("schema_version") != 1
        or audit.get("component") != "pixel-level-epipolar-rectification-audit"
        or audit.get("status") != "PASS"
        or audit.get("published_contract")
        != "audited_same_row_rectified_pixels_v1"
    ):
        raise SpringV31Error("pixel rectification audit did not publish the required contract")
    manifests = audit.get("manifests")
    if not isinstance(manifests, Mapping):
        raise SpringV31Error("pixel audit manifest bindings are missing")
    for name, path in (
        ("train", paths.train_manifest),
        ("validation", paths.validation_manifest),
    ):
        entry = manifests.get(name)
        if (
            not isinstance(entry, Mapping)
            or Path(str(entry.get("path", ""))).expanduser().resolve() != path.resolve()
            or entry.get("sha256") != sha256_file(path)
            or entry.get("record_count") != len(load_manifest(path))
        ):
            raise SpringV31Error(f"pixel audit does not bind the exact {name} manifest")
    expected_config = {
        "samples_per_sequence": 32,
        "seed": SEED,
        "sift_nfeatures": 4096,
        "sift_contrast_threshold": 0.02,
        "ratio_threshold": 0.75,
        "min_horizontal_disparity_px": 0.25,
        "max_horizontal_disparity_px": 512.0,
        "broad_vertical_prefilter_px": 32.0,
        "ransac_reprojection_threshold_px": 1.0,
        "ransac_confidence": 0.999,
        "ransac_max_iterations": 10_000,
    }
    expected_thresholds = {
        "min_ratio_matches_per_frame": 64,
        "min_plausible_matches_per_frame": 48,
        "min_ransac_inliers_per_frame": 32,
        "min_frame_coverage_fraction": 0.95,
        "max_abs_median_dy_px": 1.25,
        "max_p95_abs_dy_px": 3.0,
    }
    if audit.get("config") != expected_config or audit.get("thresholds") != expected_thresholds:
        raise SpringV31Error("pixel audit sampling/configuration thresholds differ")
    checks = audit.get("threshold_checks")
    if (
        not isinstance(checks, list)
        or not checks
        or any(not isinstance(check, Mapping) or check.get("passed") is not True for check in checks)
    ):
        raise SpringV31Error("pixel audit contains a missing or failed threshold check")
    global_checks = {
        str(check.get("metric")): check
        for check in checks
        if isinstance(check, Mapping) and check.get("scope") == "global"
    }
    expected_global = {
        "frame_coverage_fraction": (">=", 0.95),
        "abs_median_dy_right_minus_left_px": ("<=", 1.25),
        "p95_abs_dy_right_minus_left_px": ("<=", 3.0),
    }
    for metric, (operator, threshold) in expected_global.items():
        check = global_checks.get(metric)
        if (
            not isinstance(check, Mapping)
            or check.get("operator") != operator
            or check.get("threshold") != threshold
            or check.get("passed") is not True
        ):
            raise SpringV31Error(f"pixel audit global threshold differs: {metric}")
    return _path_identity(paths.pixel_audit)


def _calibration_protocol(paths: Paths) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, sidecar, manifest in (
        ("train", paths.train_calibration, paths.train_manifest),
        ("validation", paths.validation_calibration, paths.validation_manifest),
    ):
        index = load_rectified_calibration_sidecar(
            sidecar,
            receipt_path=sidecar.with_suffix(".receipt.json"),
            expected_manifest_path=manifest,
        )
        if len(index.records) != len(load_manifest(manifest)):
            raise SpringV31Error(f"{name} calibration sidecar has incomplete coverage")
        if index.pixel_audit_path != paths.pixel_audit.resolve():
            raise SpringV31Error(f"{name} calibration sidecar uses a different pixel audit")
        result[name] = {
            "sidecar": _path_identity(sidecar),
            "receipt": _path_identity(sidecar.with_suffix(".receipt.json")),
            "records": len(index.records),
            "component": "rectified-stereo-calibration",
            "contract_version": "stored_rectified_virtual_cameras_v1",
        }
    return result


def verify_protocol(paths: Paths) -> dict[str, Any]:
    result = {
        "schema_version": SCHEMA_VERSION,
        "component": "spring-v3.1-ffs-common-domain",
        "status": "PASS",
        "protocol": PROTOCOL,
        "seed": SEED,
        "manifests": _manifest_protocol(paths),
        "endpoints": _endpoint_protocol(paths),
        "pixel_audit": _audit_protocol(paths),
        "calibration": _calibration_protocol(paths),
        "crop": {
            "coordinate_space": "model_hr_image",
            "size_hw": list(CROP_SIZE_HW),
            "origin_xy": list(CROP_ORIGIN_XY),
        },
    }
    _atomic_json(paths.output_root / "protocol_receipt.json", result)
    return result


def _run_direct(
    command: Sequence[str], *, cwd: Path, log_path: Path, dry_run: bool
) -> Mapping[str, Any]:
    record: dict[str, Any] = {
        "command": [str(value) for value in command],
        "command_shell": _command_text(command),
        "cwd": str(cwd),
        "log_path": str(log_path),
        "started_at": _utc_now(),
        "status": "PLANNED" if dry_run else "RUNNING",
    }
    if dry_run:
        return record
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            [str(value) for value in command],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    record.update(
        {
            "status": "COMPLETE" if process.returncode == 0 else "FAILED",
            "returncode": int(process.returncode),
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": _utc_now(),
            "log_sha256": sha256_file(log_path),
        }
    )
    if process.returncode:
        raise SpringV31Error(
            f"command failed with exit {process.returncode}: {_command_text(command)}; "
            f"see {log_path}"
        )
    return record


def prepare(paths: Paths, args: argparse.Namespace) -> dict[str, Any]:
    python = str(args.python)
    commands: list[tuple[str, list[str], Path]] = []
    if not paths.all_manifest.is_file():
        commands.append(
            (
                "build_manifest",
                [
                    python,
                    str(paths.project_root / "tools/build_spring_manifest.py"),
                    "--spring-root",
                    str(paths.spring_root),
                    "--split",
                    "train",
                    "--output",
                    str(paths.all_manifest),
                ],
                paths.output_root / "logs/prepare/build_manifest.log",
            )
        )
    if (
        not paths.train_manifest.is_file()
        or not paths.validation_manifest.is_file()
        or not paths.split_receipt.is_file()
    ):
        commands.append(
            (
                "split_manifest",
                [
                    python,
                    str(paths.project_root / "tools/build_spring_sequence_split.py"),
                    "--input",
                    str(paths.all_manifest),
                    "--output-dir",
                    str(paths.train_manifest.parent),
                    "--validation-sequences",
                    ",".join(VALIDATION_SEQUENCES),
                ],
                paths.output_root / "logs/prepare/split_manifest.log",
            )
        )
    if (
        not paths.endpoint_index.is_file()
        or not paths.endpoint_index.with_suffix(".receipt.json").is_file()
    ):
        commands.append(
            (
                "build_endpoint_index",
                [
                    python,
                    str(paths.project_root / "tools/build_spring_endpoint_index.py"),
                    "--manifest",
                    str(paths.validation_manifest),
                    "--output",
                    str(paths.endpoint_index),
                    "--sequence-warmup",
                    str(ENDPOINT_WARMUP_FRAMES),
                ],
                paths.output_root / "logs/prepare/build_endpoint_index.log",
            )
        )
    if not paths.pixel_audit.is_file():
        commands.append(
            (
                "pixel_rectification_audit",
                [
                    python,
                    str(paths.project_root / "tools/audit_epipolar_rectification.py"),
                    "--train-manifest",
                    str(paths.train_manifest),
                    "--validation-manifest",
                    str(paths.validation_manifest),
                    "--json-out",
                    str(paths.pixel_audit),
                    "--samples-per-sequence",
                    str(args.audit_samples_per_sequence),
                    "--seed",
                    str(SEED),
                ],
                paths.output_root / "logs/prepare/pixel_rectification_audit.log",
            )
        )
    records: dict[str, Any] = {}
    for name, command, log in commands:
        records[name] = _run_direct(
            command, cwd=paths.project_root, log_path=log, dry_run=args.dry_run
        )
        if args.dry_run:
            continue
    if args.dry_run:
        return records
    _manifest_protocol(paths)
    _endpoint_protocol(paths)
    _audit_protocol(paths)
    for name, manifest, sidecar in (
        ("train_calibration", paths.train_manifest, paths.train_calibration),
        (
            "validation_calibration",
            paths.validation_manifest,
            paths.validation_calibration,
        ),
    ):
        if sidecar.is_file() and sidecar.with_suffix(".receipt.json").is_file():
            continue
        command = [
            python,
            str(paths.project_root / "tools/build_stereo_calibration.py"),
            "--manifest",
            str(manifest),
            "--pixel-audit",
            str(paths.pixel_audit),
            "--output",
            str(sidecar),
        ]
        records[name] = _run_direct(
            command,
            cwd=paths.project_root,
            log_path=paths.output_root / f"logs/prepare/{name}.log",
            dry_run=False,
        )
    records["protocol"] = verify_protocol(paths)
    return records


def _cache_jobs(paths: Paths, args: argparse.Namespace, arms: Sequence[str]) -> list[Job]:
    python = str(args.python)
    device = _device_argument(args.devices)
    jobs: list[Job] = []
    need_half_train = any(arm in arms for arm in ("F2", "F3", "F4", "F5", "F6"))
    need_half_validation = any(
        arm in arms for arm in ("F1", "F2", "F3", "F4", "F5", "F6")
    )
    need_gt = any(arm in arms for arm in ("F2", "F3", "F4", "F5", "F6"))
    need_vggt = any(arm in arms for arm in ("F4", "F5", "F6"))
    ffs_base = [
        python,
        str(paths.project_root / "tools/cache_spring_ffs.py"),
        "--checkpoint",
        str(args.ffs_checkpoint),
        "--repo",
        str(args.ffs_repo),
        "--role",
        "observation",
        "--device",
        device,
        "--right-left-check",
        "--missing-normalize",
        "error",
    ]
    if "F0" in arms:
        jobs.append(
            Job(
                "cache_validation_full_ffs",
                tuple(
                    ffs_base
                    + [
                        "--manifest",
                        str(paths.validation_manifest),
                        "--output",
                        str(paths.validation_full_observation.parent),
                        "--scale",
                        "1",
                        "--iterations",
                        "4",
                        "--max-disp",
                        "384",
                    ]
                ),
                paths.validation_full_observation / "run_receipt.json",
                "cache",
            )
        )
    if need_half_train or need_half_validation:
        half_splits = []
        if need_half_train:
            half_splits.append(
                ("train", paths.train_manifest, paths.train_half_observation)
            )
        if need_half_validation:
            half_splits.append(
                (
                    "validation",
                    paths.validation_manifest,
                    paths.validation_half_observation,
                )
            )
        for split, manifest, root in half_splits:
            jobs.append(
                Job(
                    f"cache_{split}_half_ffs",
                    tuple(
                        ffs_base
                        + [
                            "--manifest",
                            str(manifest),
                            "--output",
                            str(root.parent),
                            "--scale",
                            "2",
                            "--iterations",
                            "4",
                            "--max-disp",
                            "192",
                        ]
                    ),
                    root / "run_receipt.json",
                    "cache",
                )
            )
    if need_gt:
        for split, manifest, root in (
            ("train", paths.train_manifest, paths.train_ground_truth),
            ("validation", paths.validation_manifest, paths.validation_ground_truth),
        ):
            jobs.append(
                Job(
                    f"cache_{split}_spring_gt",
                    (
                        python,
                        str(paths.project_root / "tools/cache_spring_gt.py"),
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(root.parent),
                        "--cache-dtype",
                        "float16",
                    ),
                    root / "run_receipt.json",
                    "cache",
                    gpu=False,
                )
            )
    if need_vggt:
        for split, manifest, root in (
            ("train", paths.train_manifest, paths.train_vggt),
            ("validation", paths.validation_manifest, paths.validation_vggt),
        ):
            jobs.append(
                Job(
                    f"cache_{split}_vggt",
                    (
                        python,
                        str(paths.project_root / "tools/cache_vggt.py"),
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(root),
                        "--checkpoint",
                        str(args.vggt_checkpoint),
                        "--repo",
                        str(args.vggt_repo),
                        "--context-pairs",
                        "5",
                        "--causal",
                        "--input-mode",
                        "balanced",
                        "--image-resolution",
                        "512",
                        "--output-grid",
                        "540",
                        "960",
                        "--device",
                        device,
                    ),
                    root / "run_receipt.json",
                    "cache",
                )
            )
    return jobs


def _geometry_jobs(paths: Paths, args: argparse.Namespace, arms: Sequence[str]) -> list[Job]:
    python = str(args.python)
    jobs: list[Job] = []
    need_legacy = "F3" in arms
    need_calibrated = any(arm in arms for arm in ("F4", "F5", "F6"))
    if need_legacy:
        for split, manifest, ffs, output in (
            (
                "train",
                paths.train_manifest,
                paths.train_half_observation,
                paths.train_legacy_derived,
            ),
            (
                "validation",
                paths.validation_manifest,
                paths.validation_half_observation,
                paths.validation_legacy_derived,
            ),
        ):
            jobs.append(
                Job(
                    f"derive_{split}_legacy_v2_control",
                    (
                        python,
                        str(paths.project_root / "tools/build_spring_gt_geometry.py"),
                        "--manifest",
                        str(manifest),
                        "--observation-root",
                        str(ffs),
                        "--output",
                        str(output),
                        "--sequence-warmup",
                        "4",
                    ),
                    output / "run_receipt.json",
                    "geometry",
                    gpu=False,
                )
            )
    if need_calibrated:
        for split, vggt, ffs, output, calibration in (
            (
                "train",
                paths.train_vggt,
                paths.train_half_observation,
                paths.train_calibrated_derived,
                paths.train_calibration,
            ),
            (
                "validation",
                paths.validation_vggt,
                paths.validation_half_observation,
                paths.validation_calibrated_derived,
                paths.validation_calibration,
            ),
        ):
            jobs.append(
                Job(
                    f"derive_{split}_calibrated_v3_1",
                    (
                        python,
                        str(paths.project_root / "tools/derive_geometry_manifest.py"),
                        "--vggt-root",
                        str(vggt),
                        "--ffs-root",
                        str(ffs),
                        "--output",
                        str(output),
                        "--cache-dtype",
                        "float32",
                        "--rectified-calibration-sidecar",
                        str(calibration),
                        "--rectified-calibration-receipt",
                        str(calibration.with_suffix(".receipt.json")),
                    ),
                    output / "run_receipt.json",
                    "geometry",
                    gpu=False,
                )
            )
    return jobs


def _resume_or_init(
    command: list[str], *, output: Path, initializer: Path | None
) -> list[str]:
    latest = output / "latest.pt"
    if latest.is_file():
        command.extend(("--resume", str(latest)))
    elif initializer is not None:
        command.extend(("--init-from", str(initializer)))
    return command


def _train_job(
    paths: Paths,
    args: argparse.Namespace,
    *,
    name: str,
    config: Path,
    output: Path,
    initializer: Path | None = None,
    derived: Path | None = None,
    calibration: bool = False,
) -> Job:
    command = [
        str(args.python),
        str(paths.project_root / "train.py"),
        "--config",
        str(config),
        "--manifest",
        str(paths.train_manifest),
        "--observation-cache-root",
        str(paths.train_half_observation),
        "--teacher-cache-root",
        str(paths.train_ground_truth),
        "--output-dir",
        str(output),
        "--device",
        _device_argument(args.devices),
    ]
    if derived is not None:
        command.extend(("--derived-cache-root", str(derived)))
    if calibration:
        command.extend(("--calibration-sidecar", str(paths.train_calibration)))
    command = _resume_or_init(command, output=output, initializer=initializer)
    command.append("seed=42")
    return Job(name, tuple(command), output / "final.pt", "train")


def _initializer_jobs(paths: Paths, args: argparse.Namespace, arms: Sequence[str]) -> list[Job]:
    jobs: list[Job] = []
    if any(arm in arms for arm in ("F2", "F4", "F5", "F6")):
        jobs.append(
            _train_job(
                paths,
                args,
                name="train_F2",
                config=paths.project_root / ARM_CONFIGS["F2"],
                output=paths.arm_train("F2"),
                calibration=True,
            )
        )
    if "F3" in arms:
        jobs.append(
            _train_job(
                paths,
                args,
                name="train_F3_stage_a_control",
                config=paths.project_root / F3_INITIALIZER_CONFIG,
                output=paths.f3_initializer_train,
            )
        )
    return jobs


def _temporal_train_jobs(
    paths: Paths, args: argparse.Namespace, arms: Sequence[str]
) -> list[Job]:
    jobs: list[Job] = []
    if "F3" in arms:
        jobs.append(
            _train_job(
                paths,
                args,
                name="train_F3",
                config=paths.project_root / ARM_CONFIGS["F3"],
                output=paths.arm_train("F3"),
                initializer=paths.f3_initializer_train / "final.pt",
                derived=paths.train_legacy_derived,
            )
        )
    for arm in ("F4", "F5", "F6"):
        if arm not in arms:
            continue
        jobs.append(
            _train_job(
                paths,
                args,
                name=f"train_{arm}",
                config=paths.project_root / ARM_CONFIGS[arm],
                output=paths.arm_train(arm),
                initializer=paths.arm_train("F2") / "final.pt",
                derived=paths.train_calibrated_derived,
                calibration=True,
            )
        )
    return jobs


def _baseline_eval_job(paths: Paths, args: argparse.Namespace, arm: str) -> Job:
    mode = "full" if arm == "F0" else "half"
    observation = (
        paths.validation_full_observation
        if arm == "F0"
        else paths.validation_half_observation
    )
    output = paths.arm_eval(arm)
    return Job(
        f"eval_{arm}",
        (
            str(args.python),
            str(paths.project_root / "tools/eval_spring_baseline.py"),
            "--manifest",
            str(paths.validation_manifest),
            "--observation-cache-root",
            str(observation),
            "--output",
            str(output),
            "--seed",
            str(SEED),
            "--mode",
            mode,
            "--spring-endpoint-index-list",
            str(paths.endpoint_index),
            "--crop-mode",
            "fixed",
            "--crop-origin",
            str(CROP_ORIGIN_XY[0]),
            str(CROP_ORIGIN_XY[1]),
            "--crop-size",
            str(CROP_SIZE_HW[0]),
            str(CROP_SIZE_HW[1]),
        ),
        output / "metrics.json",
        "eval",
        gpu=False,
    )


def _model_eval_job(paths: Paths, args: argparse.Namespace, arm: str) -> Job:
    output = paths.arm_eval(arm)
    command = [
        str(args.python),
        str(paths.project_root / "eval.py"),
        "--config",
        str(paths.project_root / ARM_CONFIGS[arm]),
        "--checkpoint",
        str(paths.arm_train(arm) / "final.pt"),
        "--manifest",
        str(paths.validation_manifest),
        "--observation-cache-root",
        str(paths.validation_half_observation),
        "--teacher-cache-root",
        str(paths.validation_ground_truth),
        "--output-dir",
        str(output),
        "--device",
        _device_argument(args.devices),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.eval_num_workers),
        "--spring-endpoint-index-list",
        str(paths.endpoint_index),
        "--crop-mode",
        "fixed",
        "--crop-origin",
        str(CROP_ORIGIN_XY[0]),
        str(CROP_ORIGIN_XY[1]),
        "--spring-native-metrics",
    ]
    if arm in ("F3", "F4", "F5", "F6"):
        if arm == "F3":
            derived = paths.validation_legacy_derived
            spatial = paths.f3_initializer_train / "final.pt"
        else:
            derived = paths.validation_calibrated_derived
            spatial = paths.arm_train("F2") / "final.pt"
        command.extend(
            (
                "--derived-cache-root",
                str(derived),
                "--spatial-checkpoint",
                str(spatial),
            )
        )
    if arm in ("F2", "F4", "F5", "F6"):
        command.extend(("--calibration-sidecar", str(paths.validation_calibration)))
    command.append("seed=42")
    return Job(f"eval_{arm}", tuple(command), output / "metrics.json", "eval")


def _eval_jobs(paths: Paths, args: argparse.Namespace, arms: Sequence[str]) -> list[Job]:
    jobs: list[Job] = []
    for arm in RUNNABLE_ARMS:
        if arm not in arms:
            continue
        if arm in ("F0", "F1"):
            jobs.append(_baseline_eval_job(paths, args, arm))
        else:
            jobs.append(_model_eval_job(paths, args, arm))
    return jobs


def _command_option(command: Sequence[str], name: str) -> str | None:
    try:
        index = command.index(name)
    except ValueError:
        return None
    if index + 1 >= len(command):
        raise SpringV31Error(f"job command option has no value: {name}")
    return str(command[index + 1])


def _strict_jsonl(path: Path, name: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise SpringV31Error(f"blank {name} row: {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise SpringV31Error(
                        f"{name} row is not an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise SpringV31Error(f"cannot read {name}: {path}") from exc
    if not rows:
        raise SpringV31Error(f"{name} is empty: {path}")
    return rows


def _expected_cache_records(
    manifest: Path, *, sequence_warmup: int
) -> list[tuple[int, str, int]]:
    positions: dict[str, int] = {}
    expected: list[tuple[int, str, int]] = []
    for index, record in enumerate(load_manifest(manifest)):
        position = positions.get(record.sequence_id, 0)
        positions[record.sequence_id] = position + 1
        if position >= sequence_warmup:
            expected.append((index, record.sequence_id, int(record.frame_id)))
    return expected


def _validate_cache_inventory(
    root: Path,
    manifest: Path,
    *,
    sequence_warmup: int,
    expected_identity: Mapping[str, Any] | None = None,
    require_record_sha: bool = False,
    validate_payloads: bool = False,
    strict_rows: bool = False,
    payload_kind: str = "raw",
) -> dict[str, Any]:
    """Validate ordered inventory rows, files, digests, and payload identity."""
    inventory = root / "cache_manifest.jsonl"
    rows = _strict_jsonl(inventory, "cache manifest")
    expected = _expected_cache_records(manifest, sequence_warmup=sequence_warmup)
    if len(rows) != len(expected):
        raise SpringV31Error(
            f"cache inventory count differs at {root}: expected {len(expected)}, got {len(rows)}"
        )
    declared_paths: list[Path] = []
    actual_keys: list[tuple[int, str, int]] = []
    statuses: list[str] = []
    expected_identity_dict = (
        None if expected_identity is None else dict(expected_identity)
    )
    for selection_index, (row, expected_key) in enumerate(zip(rows, expected, strict=True)):
        manifest_index, sequence_id, frame_id = expected_key
        has_target_manifest_index = "target_manifest_index" in row
        row_manifest_index = row.get(
            "target_manifest_index",
            # Non-window caches canonically use selection_index as both the
            # inventory order and manifest index. Windowed caches must publish
            # the separate target_manifest_index below.
            row.get("selection_index") if not sequence_warmup else None,
        )
        if sequence_warmup and not has_target_manifest_index:
            raise SpringV31Error(
                "canonical cache row lacks target_manifest_index: "
                f"{inventory}:{selection_index + 1}"
            )
        if strict_rows and (
            not _is_nonnegative_int(row.get("selection_index"))
            or int(row["selection_index"]) != selection_index
        ):
            raise SpringV31Error(
                f"cache inventory selection_index differs at row {selection_index}"
            )
        if not _is_nonnegative_int(row_manifest_index):
            raise SpringV31Error(
                f"cache inventory manifest index is malformed at row {selection_index}"
            )
        actual_key = (
            int(row_manifest_index),
            str(row.get("sequence_id", "")),
            int(row.get("frame_id"))
            if _is_nonnegative_int(row.get("frame_id"))
            else -1,
        )
        if actual_key != expected_key:
            raise SpringV31Error(
                f"cache inventory record differs at row {selection_index}: "
                f"expected={expected_key}, got={actual_key}"
            )
        raw_path = row.get("cache_path", row.get("vggt_cache_path"))
        if not isinstance(raw_path, str):
            raise SpringV31Error(f"cache inventory path is missing at row {selection_index}")
        cache_path = Path(raw_path).expanduser().resolve()
        canonical_path = (root / sequence_id / f"{frame_id}.pt").resolve()
        if cache_path != canonical_path or not cache_path.is_file():
            raise SpringV31Error(
                f"cache inventory path is missing or non-canonical: {cache_path}"
            )
        status = row.get("status")
        if strict_rows and status is not None and status not in {"written", "reused"}:
            raise SpringV31Error(
                f"cache inventory status is invalid at row {selection_index}: {status!r}"
            )
        if status is not None and status not in {"written", "reused"}:
            # Even compatibility-mode callers must not silently accept an
            # explicitly invalid producer status.
            raise SpringV31Error(
                f"cache inventory status is invalid at row {selection_index}: {status!r}"
            )
        if status is not None:
            statuses.append(str(status))
        recorded_sha = row.get("cache_sha256")
        if require_record_sha and not _is_sha256(recorded_sha):
            raise SpringV31Error(
                f"cache inventory record lacks canonical cache_sha256: {cache_path}"
            )
        if recorded_sha is not None:
            if not _is_sha256(recorded_sha):
                raise SpringV31Error(f"cache record SHA-256 is malformed: {cache_path}")
            if recorded_sha != sha256_file(cache_path):
                raise SpringV31Error(f"cache record SHA-256 differs: {cache_path}")
        if validate_payloads:
            try:
                import torch

                payload = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
            except Exception as exc:  # torch exposes multiple loader exception types
                raise SpringV31Error(
                    f"cache payload cannot be safely loaded: {cache_path}"
                ) from exc
            if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
                raise SpringV31Error(f"cache payload schema differs: {cache_path}")
            payload_identity = payload.get("identity")
            if not isinstance(payload_identity, Mapping):
                raise SpringV31Error(f"cache payload identity is missing: {cache_path}")
            if payload_kind == "raw":
                if set(payload_identity) != CACHE_IDENTITY_FIELDS:
                    raise SpringV31Error(f"cache payload identity fields differ: {cache_path}")
                for key, value in payload_identity.items():
                    if key.endswith("_sha256") and not _is_sha256(value):
                        raise SpringV31Error(
                            f"cache payload identity {key} is malformed: {cache_path}"
                        )
                    if key == "upstream_commit" and not _is_git_hash(value):
                        raise SpringV31Error(
                            f"cache payload upstream commit is malformed: {cache_path}"
                        )
            if expected_identity_dict is not None and dict(payload_identity) != expected_identity_dict:
                raise SpringV31Error(
                    f"cache payload identity differs from receipt: {cache_path}"
                )
            if not isinstance(payload.get("metadata"), Mapping) or not isinstance(
                payload.get("tensors"), Mapping
            ):
                raise SpringV31Error(f"cache payload metadata/tensors are malformed: {cache_path}")
        actual_keys.append(actual_key)
        declared_paths.append(cache_path)
    actual_paths = sorted(path.resolve() for path in root.rglob("*.pt"))
    if set(actual_paths) != set(declared_paths) or len(actual_paths) != len(declared_paths):
        extra = sorted(str(path) for path in set(actual_paths) - set(declared_paths))
        missing = sorted(str(path) for path in set(declared_paths) - set(actual_paths))
        raise SpringV31Error(
            f"cache directory/inventory .pt set differs at {root}: "
            f"extra={extra[:8]}, missing={missing[:8]}"
        )
    return {
        "root": str(root.resolve()),
        "inventory": _path_identity(inventory),
        "inventory_path": str(inventory.resolve()),
        "inventory_sha256": sha256_file(inventory),
        "records": len(rows),
        "sequence_warmup": sequence_warmup,
        "selection_indices": [key[0] for key in actual_keys],
        "statuses": statuses,
        "canonical_pt_set": True,
    }


def _validate_cache_job(job: Job) -> dict[str, Any]:
    root = job.expected_output.parent.resolve()
    manifest_value = _command_option(job.command, "--manifest")
    if manifest_value is None:
        raise SpringV31Error(f"cache job lacks --manifest: {job.name}")
    manifest = Path(manifest_value).expanduser().resolve()
    receipt = _strict_json(job.expected_output, f"{job.name} receipt")
    identity = receipt.get("identity")
    config = receipt.get("config")
    if (
        receipt.get("schema_version") != 1
        or not _same_resolved_path(receipt.get("manifest"), manifest)
        or receipt.get("manifest_sha256") != sha256_file(manifest)
    ):
        raise SpringV31Error(f"{job.name} receipt manifest/schema lineage differs")
    if not isinstance(identity, Mapping) or not isinstance(config, Mapping):
        raise SpringV31Error(f"{job.name} receipt lacks cache identity/config")
    warmup = 4 if job.name.startswith("cache_") and job.name.endswith("_vggt") else 0
    expected = _expected_cache_records(manifest, sequence_warmup=warmup)
    count_field = "selected_windows" if warmup else "selected_records"
    selected = receipt.get(count_field)
    written = receipt.get("written_records")
    reused = receipt.get("reused_records")
    if (
        not _is_nonnegative_int(selected)
        or not _is_nonnegative_int(written)
        or not _is_nonnegative_int(reused)
        or selected != len(expected)
        or written + reused != selected
        or not _is_finite_number(receipt.get("elapsed_seconds"), positive=True)
    ):
        raise SpringV31Error(f"{job.name} receipt counts/runtime are malformed")
    if "half_ffs" in job.name:
        required = ("ffs-observation", 2, "half", 4, 192)
    elif "full_ffs" in job.name:
        required = ("ffs-observation-full-resolution", 1, "full", 4, 384)
    elif "spring_gt" in job.name:
        required = ("spring-ground-truth", None, None, None, None)
    elif job.name.endswith("_vggt"):
        required = ("vggt-omega", None, None, None, None)
        if (
            config.get("context_pairs") != 5
            or config.get("causal") is not True
            or config.get("output_grid_hw") != [540, 960]
        ):
            raise SpringV31Error(f"{job.name} VGGT inference config differs")
    else:
        raise SpringV31Error(f"unknown cache job: {job.name}")
    component, scale, resolution, iterations, max_disp = required
    expected_commit = None
    expected_checkpoint = None
    if component.startswith("ffs-observation"):
        expected_commit = FFS_UPSTREAM_COMMIT
        expected_checkpoint = FFS_CHECKPOINT_SHA256
    elif component == "vggt-omega":
        expected_commit = VGGT_UPSTREAM_COMMIT
        expected_checkpoint = VGGT_CHECKPOINT_SHA256
    _validate_cache_identity(
        identity,
        job.name,
        component=component,
        expected_commit=expected_commit,
        expected_checkpoint=expected_checkpoint,
    )
    if scale is not None and (
        config.get("scale") != scale
        or config.get("resolution_mode") != resolution
        or config.get("iterations") != iterations
        or config.get("max_disp") != max_disp
        or config.get("right_left_check") is not True
    ):
        raise SpringV31Error(f"{job.name} FFS inference config differs")
    if component.startswith("ffs-observation"):
        expected_config = {
            "role": "observation",
            "scale": scale,
            "resolution_mode": resolution,
            "iterations": iterations,
            "max_disp": max_disp,
            "max_disp_hr_equivalent_px": 384,
            "max_disp_input_grid_px": max_disp,
            "max_disp_policy": "matched_physical_search_range_384_hr_px",
            "right_left_check": True,
            "missing_normalize": "error",
            "cache_dtype": "float16",
            "checkpoint_sha256": FFS_CHECKPOINT_SHA256,
            "upstream_commit": FFS_UPSTREAM_COMMIT,
        }
        for key, expected_value in expected_config.items():
            if config.get(key) != expected_value:
                raise SpringV31Error(
                    f"{job.name} FFS config field {key} differs"
                )
    elif component == "vggt-omega":
        expected_view_order = [
            label
            for time_label in ("t-4", "t-3", "t-2", "t-1", "t")
            for label in (f"L[{time_label}]", f"R[{time_label}]")
        ]
        expected_config = {
            "causal": True,
            "context_pairs": 5,
            "current_left_view_index": 8,
            "input_mode": "balanced",
            "image_resolution": 512,
            "cache_dtype": "float16",
            "output_grid_hw": [540, 960],
            "view_order": expected_view_order,
        }
        for key, expected_value in expected_config.items():
            if config.get(key) != expected_value:
                raise SpringV31Error(
                    f"{job.name} VGGT config field {key} differs"
                )
        if (
            not _is_nonnegative_int(receipt.get("available_windows"))
            or receipt.get("available_windows") != selected
        ):
            raise SpringV31Error(f"{job.name} VGGT available/selected coverage differs")
    else:  # Spring GT cache
        if (
            config.get("cache_dtype") != "float16"
            or config.get("resolution") != "image"
            or config.get("disparity_unit") != "full_hd_pixels"
            or config.get("disparity_value_scaling") != 1
            or config.get("sampling") != "dsp5[::2,::2]"
            or config.get("target_type") != SPRING_GT_TARGET_TYPE
            or config.get("supervision_source") != "Spring_GT"
            or receipt.get("target_type") != SPRING_GT_TARGET_TYPE
            or receipt.get("paper_ground_truth") is not True
            or receipt.get("synthetic_ground_truth") is not True
        ):
            raise SpringV31Error(f"{job.name} Spring GT target/config differs")
    if (
        receipt.get("cache_manifest")
        != str((root / "cache_manifest.jsonl").resolve())
        or receipt.get("cache_manifest_sha256")
        != sha256_file(root / "cache_manifest.jsonl")
    ):
        raise SpringV31Error(f"{job.name} receipt is not bound to its canonical inventory")
    inventory_evidence = _validate_cache_inventory(
        root,
        manifest,
        sequence_warmup=warmup,
        expected_identity=identity,
        require_record_sha=True,
        validate_payloads=True,
        strict_rows=True,
        payload_kind="raw",
    )
    return {
        "receipt": _path_identity(job.expected_output),
        "identity": dict(identity),
        "config": dict(config),
        "selected": int(selected),
        "written": int(written),
        "reused": int(reused),
        "inventory": inventory_evidence,
    }


def _validate_geometry_job(job: Job) -> dict[str, Any]:
    root = job.expected_output.parent.resolve()
    manifest_value = _command_option(job.command, "--manifest")
    if manifest_value is None:
        vggt_value = _command_option(job.command, "--vggt-root")
        if vggt_value is None:
            raise SpringV31Error(f"geometry job lacks its source manifest: {job.name}")
        raw_receipt = _strict_json(
            Path(vggt_value).expanduser().resolve() / "run_receipt.json",
            f"{job.name} raw VGGT receipt",
        )
        manifest_value = raw_receipt.get("manifest")
    if not isinstance(manifest_value, str):
        raise SpringV31Error(f"geometry job has no manifest lineage: {job.name}")
    manifest = Path(manifest_value).expanduser().resolve()
    receipt = _strict_json(job.expected_output, f"{job.name} receipt")
    config = receipt.get("config")
    counts = receipt.get("counts")
    selection = receipt.get("selection")
    if (
        receipt.get("manifest_sha256") != sha256_file(manifest)
        or not isinstance(config, Mapping)
        or not isinstance(counts, Mapping)
        or not isinstance(selection, Mapping)
    ):
        raise SpringV31Error(f"{job.name} derived receipt lineage is malformed")
    expected = _expected_cache_records(manifest, sequence_warmup=4)
    if counts.get("selected") != len(expected) or selection.get("selected_windows") != len(expected):
        raise SpringV31Error(f"{job.name} derived cache has incomplete coverage")
    if "legacy_v2_control" in job.name:
        if (
            receipt.get("schema_version") != 1
            or receipt.get("component") != "vggt-ffs-derived-geometry-batch"
            or config.get("pose_source") != "Spring_GT_pose"
            or config.get("depth_prior_source") != "disabled_zero_fill"
            or config.get("sequence_warmup") != 4
            or config.get("rectified_stereo_calibration") is not None
        ):
            raise SpringV31Error(f"{job.name} is not the F3 GT-pose/no-depth control")
    else:
        calibration = config.get("rectified_stereo_calibration")
        receipt_value = _command_option(job.command, "--rectified-calibration-receipt")
        if (
            receipt.get("schema_version") != 2
            or receipt.get("component")
            != "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch"
            or not isinstance(calibration, Mapping)
            or receipt_value is None
            or calibration.get("receipt_sha256")
            != sha256_file(Path(receipt_value).expanduser().resolve())
        ):
            raise SpringV31Error(f"{job.name} calibrated geometry lineage differs")
    return {
        "receipt": _path_identity(job.expected_output),
        "inventory": _validate_cache_inventory(root, manifest, sequence_warmup=4),
    }


def _validate_training_job(job: Job) -> dict[str, Any]:
    checkpoint = job.expected_output.resolve()
    summary_path = checkpoint.parent / "run_summary.json"
    if not checkpoint.is_file():
        raise SpringV31Error(f"training checkpoint is missing: {checkpoint}")
    summary = _strict_json(summary_path, f"{job.name} training summary")
    final = summary.get("final_checkpoint")
    expected_stage = "temporal" if job.name in {"train_F3", "train_F4", "train_F5", "train_F6"} else "spatial"
    if (
        summary.get("status") != "TRAINING_COMPLETE"
        or summary.get("stage") != expected_stage
        or not isinstance(final, Mapping)
        or Path(str(final.get("path", ""))).expanduser().resolve() != checkpoint
        or final.get("sha256") != sha256_file(checkpoint)
    ):
        raise SpringV31Error(f"{job.name} training completion lineage differs")
    import torch
    from utils.checkpoint import config_fingerprint

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_config = payload.get("config") if isinstance(payload, Mapping) else None
    if not isinstance(saved_config, Mapping):
        raise SpringV31Error(f"{job.name} checkpoint lacks its resolved config")
    config_sha = hashlib.sha256(config_fingerprint(saved_config).encode("utf-8")).hexdigest()
    if config_sha != summary.get("config_fingerprint"):
        raise SpringV31Error(f"{job.name} summary/checkpoint config fingerprint differs")
    data = saved_config.get("data")
    model = saved_config.get("model")
    train = saved_config.get("train")
    if not all(isinstance(value, Mapping) for value in (data, model, train)):
        raise SpringV31Error(f"{job.name} checkpoint config sections are malformed")
    config_name = Path(str(_command_option(job.command, "--config"))).stem
    contract = TRAINING_CONFIG_CONTRACTS[config_name]
    expected_paths = {
        "manifest_path": _command_option(job.command, "--manifest"),
        "observation_cache_root": _command_option(job.command, "--observation-cache-root"),
        "teacher_cache_root": _command_option(job.command, "--teacher-cache-root"),
        "derived_geometry_cache_root": _command_option(job.command, "--derived-cache-root"),
        "calibration_sidecar_path": _command_option(job.command, "--calibration-sidecar"),
    }
    for key, value in expected_paths.items():
        actual = data.get(key)
        if value is None:
            if actual is not None:
                raise SpringV31Error(f"{job.name} checkpoint unexpectedly binds data.{key}")
        elif Path(str(actual)).expanduser().resolve() != Path(value).expanduser().resolve():
            raise SpringV31Error(f"{job.name} checkpoint data.{key} path differs")
    if (
        saved_config.get("seed") != SEED
        or saved_config.get("arm") != (None if config_name == "F3_stage_a_control" else config_name)
        or data.get("temporal_pose_source") != contract["pose_source"]
        or model.get("use_vggt_pose") is not contract["use_vggt_pose"]
        or model.get("use_vggt_depth") is not contract["use_vggt_depth"]
        or payload.get("step") != contract["steps"]
        or summary.get("steps") != contract["steps"]
    ):
        raise SpringV31Error(f"{job.name} checkpoint arm/seed/schedule lineage differs")
    if expected_stage == "temporal":
        output_root = checkpoint.parents[3]
        expected_initializer = (
            output_root / "arms" / "F3" / "stage_a_control" / "final.pt"
            if config_name == "F3"
            else output_root / "arms" / "F2" / "train" / "final.pt"
        )
    else:
        expected_initializer = None
    if expected_initializer is not None and (
        Path(str(train.get("initialization_checkpoint", ""))).expanduser().resolve()
        != expected_initializer.resolve()
        or train.get("initialization_checkpoint_sha256")
        != sha256_file(expected_initializer)
    ):
        raise SpringV31Error(f"{job.name} Stage-A initializer lineage differs")
    return {
        "checkpoint": _path_identity(checkpoint),
        "summary": _path_identity(summary_path),
        "config_fingerprint": config_sha,
    }


def _validate_job_output(job: Job, paths: Paths) -> dict[str, Any]:
    if job.name.startswith("cache_"):
        return _validate_cache_job(job)
    if job.name.startswith("derive_"):
        return _validate_geometry_job(job)
    if job.name.startswith("train_"):
        return _validate_training_job(job)
    if job.name.startswith("eval_"):
        return _verify_eval_result(paths, job.name.removeprefix("eval_"))
    raise SpringV31Error(f"job has no output validator: {job.name}")


def _output_valid(job: Job, paths: Paths) -> bool:
    try:
        _validate_job_output(job, paths)
    except (OSError, ValueError, TypeError, KeyError, SpringV31Error):
        return False
    return True


def _run_job(
    job: Job,
    *,
    paths: Paths,
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    receipt_path = paths.output_root / "jobs" / job.name / "receipt.json"
    log_path = paths.output_root / "jobs" / job.name / "process.log"
    reuse_evidence: Mapping[str, Any] | None = None
    if not dry_run:
        try:
            reuse_evidence = _validate_job_output(job, paths)
        except (OSError, ValueError, TypeError, KeyError, SpringV31Error):
            reuse_evidence = None
    if reuse_evidence is not None:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "component": f"{COMPONENT}-job",
            "status": "REUSED_VERIFIED_OUTPUT",
            "name": job.name,
            "group": job.group,
            "expected_output": _path_identity(job.expected_output),
            "artifact_validation": dict(reuse_evidence),
            "finished_at": _utc_now(),
        }
        _atomic_json(receipt_path, receipt)
        return receipt
    environment = os.environ.copy()
    if device != "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = device
    started_at = _utc_now()
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "component": f"{COMPONENT}-job",
        "status": "PLANNED" if dry_run else "RUNNING",
        "name": job.name,
        "group": job.group,
        "gpu_job": job.gpu,
        "physical_device": device,
        "command": list(job.command),
        "command_shell": _command_text(job.command),
        "cwd": str(paths.project_root),
        "log_path": str(log_path),
        "expected_output_path": str(job.expected_output),
        "started_at": started_at,
    }
    if dry_run:
        _atomic_json(receipt_path, receipt)
        return receipt
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            list(job.command),
            cwd=paths.project_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    artifact_validation: Mapping[str, Any] | None = None
    validation_error: str | None = None
    if process.returncode == 0:
        try:
            artifact_validation = _validate_job_output(job, paths)
        except (OSError, ValueError, TypeError, KeyError, SpringV31Error) as exc:
            validation_error = str(exc)
    success = process.returncode == 0 and artifact_validation is not None
    receipt.update(
        {
            "status": "COMPLETE" if success else "FAILED",
            "returncode": int(process.returncode),
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": _utc_now(),
            "log_sha256": sha256_file(log_path),
            "expected_output": _path_identity(job.expected_output),
            "artifact_validation": (
                None if artifact_validation is None else dict(artifact_validation)
            ),
            "validation_error": validation_error,
        }
    )
    _atomic_json(receipt_path, receipt)
    if not success:
        raise SpringV31Error(
            f"job {job.name} failed; command={_command_text(job.command)}; log={log_path}"
        )
    return receipt


def _run_parallel(
    jobs: Sequence[Job],
    *,
    paths: Paths,
    devices: Sequence[str],
    dry_run: bool,
) -> dict[str, Any]:
    if not jobs:
        return {}
    device_queue = list(devices)
    lock = threading.Lock()

    def run_one(index_job: tuple[int, Job]) -> tuple[str, Mapping[str, Any]]:
        index, job = index_job
        if job.gpu:
            with lock:
                device = device_queue.pop(0)
            try:
                return job.name, _run_job(
                    job, paths=paths, device=device, dry_run=dry_run
                )
            finally:
                with lock:
                    device_queue.append(device)
        # CPU/I/O jobs do not consume the CUDA queue.  Give each a stable
        # receipt device while limiting total concurrency with the same pool.
        return job.name, _run_job(job, paths=paths, device="cpu", dry_run=dry_run)

    max_workers = max(1, len(devices))
    results: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, indexed): indexed[1].name
            for indexed in enumerate(jobs)
        }
        failures: list[BaseException] = []
        for future in concurrent.futures.as_completed(futures):
            try:
                name, receipt = future.result()
                results[name] = dict(receipt)
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise SpringV31Error(
                f"{len(failures)} job(s) failed; first failure: {failures[0]}"
            ) from failures[0]
    return results


def _write_f7_blocked(paths: Paths) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "component": "spring-v3.1-ffs-arm-status",
        "status": "OPTIONAL_BLOCKED",
        "arm": "F7",
        "optional": True,
        "depends_on": "F6",
        "reason": (
            "The repository Stage-C implementation is the legacy v2 unroll and "
            "does not implement v3.1 half-pixel transport, LR-centre measurement "
            "ownership, and current-conditioned candidate fusion."
        ),
        "required_contracts": {
            "pixel_center": "align_corners_false_half_pixel_v3_1",
            "measurement_ownership": "lr_center_projection_bounded_subpixel_v3_1",
            "candidate_fusion": "current_conditioned_age_phase_diverse_v3_1",
        },
    }
    _atomic_json(paths.output_root / "arms/F7/status.json", payload)
    return payload


def _assert_training_dependencies(paths: Paths, arms: Sequence[str]) -> None:
    if any(arm in arms for arm in ("F4", "F5", "F6")):
        checkpoint = paths.arm_train("F2") / "final.pt"
        if not checkpoint.is_file():
            raise SpringV31Error(f"F4--F6 require the unique F2 initializer: {checkpoint}")
    if "F3" in arms:
        checkpoint = paths.f3_initializer_train / "final.pt"
        if not checkpoint.is_file():
            raise SpringV31Error(f"F3 requires its v2/K=2 Stage-A initializer: {checkpoint}")


def _verify_eval_result(paths: Paths, arm: str) -> dict[str, Any]:
    metrics_path = paths.arm_eval(arm) / "metrics.json"
    report = _strict_json(metrics_path, f"{arm} metrics")
    target = report.get("target")
    if not isinstance(target, Mapping):
        raise SpringV31Error(f"{arm} metrics do not identify their target")
    if arm in ("F0", "F1"):
        import torch

        mode = "full" if arm == "F0" else "half"
        scale = 1 if arm == "F0" else 2
        cache_component = (
            "ffs-observation-full-resolution"
            if arm == "F0"
            else "ffs-observation"
        )
        reconstruction = (
            "identity" if arm == "F0" else "bilinear_align_corners_false"
        )
        expected_target = {
            "type": SPRING_GT_TARGET_TYPE,
            "component": SPRING_GT_COMPONENT,
            "paper_gt": True,
            "synthetic_ground_truth": True,
            "paper_accuracy": False,
            "disparity_unit": "full_hd_pixels",
            "resolution": "image",
        }
        expected_resolution = {
            "scale": scale,
            "cache_component": cache_component,
            "reconstruction": reconstruction,
            "max_disp_hr_equivalent_px": 384,
        }
        expected_evaluator = {
            "git_hash": repository_git_hash(paths.project_root),
            "eval_py_sha256": sha256_file(
                paths.project_root / "tools/eval_spring_baseline.py"
            ),
            "evaluation_module_sha256": sha256_file(
                paths.project_root / "src/metrics/spring_arms.py"
            ),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
        }
        elapsed_seconds = report.get("elapsed_seconds")
        if (
            report.get("schema_version") != 2
            or report.get("status") != "SCREENING_ONLY"
            or report.get("arm") != arm
            or report.get("seed") != SEED
            or report.get("baseline_mode") != mode
            or report.get("resolution_contract") != expected_resolution
            or dict(target) != expected_target
            or report.get("pose_source") != "none"
            or report.get("temporal_metrics")
            != {
                "rigid_temporal_residual_error": None,
                "non_rigid_temporal_residual_error": None,
            }
            or report.get("evaluator") != expected_evaluator
            or report.get("device") != "cpu"
            or isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or float(elapsed_seconds) <= 0.0
        ):
            raise SpringV31Error(f"{arm} baseline identity/target differs")
        observation_root = (
            paths.validation_full_observation
            if arm == "F0"
            else paths.validation_half_observation
        )
        observation_receipt_path = observation_root / "run_receipt.json"
        observation_receipt = _strict_json(
            observation_receipt_path, f"{arm} observation receipt"
        )
        observation_identity = observation_receipt.get("identity")
        observation_config = observation_receipt.get("config")
        observation_lineage = report.get("observation_lineage")
        expected_observation_lineage = {
            "receipt_path": str(observation_receipt_path.resolve()),
            "receipt_sha256": sha256_file(observation_receipt_path),
            "identity": (
                dict(observation_identity)
                if isinstance(observation_identity, Mapping)
                else None
            ),
            "config": (
                dict(observation_config)
                if isinstance(observation_config, Mapping)
                else None
            ),
        }
        if (
            report.get("observation_root") != str(observation_root.resolve())
            or not isinstance(observation_identity, Mapping)
            or not isinstance(observation_config, Mapping)
            or observation_identity.get("component") != cache_component
            or observation_config.get("role") != "observation"
            or observation_config.get("scale") != scale
            or observation_config.get("resolution_mode") != mode
            or observation_config.get("max_disp_hr_equivalent_px") != 384
            or observation_lineage != expected_observation_lineage
        ):
            raise SpringV31Error(f"{arm} observation cache lineage differs")
        manifest = report.get("manifest")
        if (
            not isinstance(manifest, Mapping)
            or Path(str(manifest.get("path", ""))).expanduser().resolve()
            != paths.validation_manifest.resolve()
            or manifest.get("sha256") != sha256_file(paths.validation_manifest)
        ):
            raise SpringV31Error(f"{arm} metrics use a different validation manifest")
        selection = report.get("selection")
        endpoint = (
            selection.get("endpoint_index_list")
            if isinstance(selection, Mapping)
            else None
        )
        crop_mode = selection.get("crop_mode") if isinstance(selection, Mapping) else None
        crop_origin = (
            selection.get("crop_origin_xy") if isinstance(selection, Mapping) else None
        )
        crop_size = selection.get("crop_size_hw") if isinstance(selection, Mapping) else None
        expected_evaluation_lineage = {
            "manifest_sha256": sha256_file(paths.validation_manifest),
            "endpoint_id_sha256": EXPECTED_ENDPOINT_ID_SHA256,
            "endpoint_count": EXPECTED_ENDPOINT_COUNT,
            "crop_mode": "fixed",
            "crop_hr_xywh": [
                CROP_ORIGIN_XY[0],
                CROP_ORIGIN_XY[1],
                CROP_SIZE_HW[1],
                CROP_SIZE_HW[0],
            ],
            "baseline_mode": mode,
            "cache_identity": dict(observation_identity),
        }
        if (
            report.get("evaluation_lineage") != expected_evaluation_lineage
            or report.get("evaluation_lineage_sha256")
            != _canonical_sha256(expected_evaluation_lineage)
        ):
            raise SpringV31Error(f"{arm} evaluation lineage differs")
    else:
        contract = TRAINING_CONFIG_CONTRACTS[arm]
        resolved = report.get("resolved_config")
        data = resolved.get("data") if isinstance(resolved, Mapping) else None
        model = resolved.get("model") if isinstance(resolved, Mapping) else None
        eval_config = resolved.get("eval") if isinstance(resolved, Mapping) else None
        if (
            report.get("schema_version") != 1
            or target.get("paper_gt") is not True
            or target.get("cache_component") != "spring-ground-truth"
            or not isinstance(resolved, Mapping)
            or resolved.get("arm") != arm
            or resolved.get("seed") != SEED
            or not isinstance(data, Mapping)
            or data.get("temporal_pose_source") != contract["pose_source"]
            or not isinstance(model, Mapping)
            or model.get("use_vggt_pose") is not contract["use_vggt_pose"]
            or model.get("use_vggt_depth") is not contract["use_vggt_depth"]
        ):
            raise SpringV31Error(f"{arm} resolved evaluation config differs from its arm")
        if (
            Path(str(report.get("manifest_path", ""))).expanduser().resolve()
            != paths.validation_manifest.resolve()
        ):
            raise SpringV31Error(f"{arm} metrics use a different validation manifest")
        endpoint = report.get("endpoint_selection")
        crop_mode = report.get("crop_mode")
        crop_origin = (
            eval_config.get("fixed_crop_origin_hr_xy")
            if isinstance(eval_config, Mapping)
            else None
        )
        crop_size = report.get("hr_crop")
        checkpoint = report.get("checkpoint")
        expected_checkpoint = paths.arm_train(arm) / "final.pt"
        if (
            not isinstance(checkpoint, Mapping)
            or Path(str(checkpoint.get("path", ""))).expanduser().resolve()
            != expected_checkpoint.resolve()
            or checkpoint.get("checkpoint_sha256") != sha256_file(expected_checkpoint)
        ):
            raise SpringV31Error(f"{arm} metrics do not bind the requested checkpoint")
    if not isinstance(endpoint, Mapping):
        raise SpringV31Error(f"{arm} metrics lack endpoint lineage")
    if (
        Path(str(endpoint.get("path", ""))).expanduser().resolve()
        != paths.endpoint_index.resolve()
        or endpoint.get("file_sha256") != sha256_file(paths.endpoint_index)
        or Path(str(endpoint.get("manifest_path", ""))).expanduser().resolve()
        != paths.validation_manifest.resolve()
        or endpoint.get("manifest_sha256") != sha256_file(paths.validation_manifest)
        or endpoint.get("endpoint_count") != EXPECTED_ENDPOINT_COUNT
        or endpoint.get("endpoint_id_sha256") != EXPECTED_ENDPOINT_ID_SHA256
    ):
        raise SpringV31Error(f"{arm} endpoint artifact lineage differs")
    endpoint_sha = endpoint.get("evaluated_endpoint_id_sha256")
    if endpoint_sha != EXPECTED_ENDPOINT_ID_SHA256:
        raise SpringV31Error(
            f"{arm} endpoint identity differs: expected {EXPECTED_ENDPOINT_ID_SHA256}, "
            f"got {endpoint_sha}"
        )
    evaluated_count = endpoint.get("evaluated_endpoint_count")
    if evaluated_count != EXPECTED_ENDPOINT_COUNT:
        raise SpringV31Error(f"{arm} evaluated {evaluated_count} endpoints")
    if (
        crop_mode != "fixed"
        or list(crop_origin or ()) != list(CROP_ORIGIN_XY)
        or list(crop_size or ()) != list(CROP_SIZE_HW)
    ):
        raise SpringV31Error(f"{arm} metrics use a different fixed crop")
    evaluated = report.get("records_evaluated", report.get("selection", {}).get("records"))
    if evaluated != EXPECTED_ENDPOINT_COUNT:
        raise SpringV31Error(f"{arm} report record count differs: {evaluated!r}")
    return _path_identity(metrics_path)


def _matrix_receipt(
    paths: Paths,
    *,
    arms: Sequence[str],
    source_snapshot: Mapping[str, Any],
    config_protocol: Mapping[str, Any],
    backbone_protocol: Mapping[str, Any],
    job_results: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    initializer = str(paths.arm_train("F2") / "final.pt")
    rows: dict[str, Any] = {
        "F0": {"model": "full-resolution FFS", "initializer": None},
        "F1": {"model": "half-resolution FFS + bilinear", "initializer": None},
        "F2": {"model": "half-resolution FFS + v3.1 T1", "initializer": None},
        "F3": {
            "model": "half-resolution FFS + v2/K=2 T3 control, GT pose",
            "initializer": str(paths.f3_initializer_train / "final.pt"),
        },
        "F4": {"model": "half-resolution FFS + v3.1 T3, GT pose", "initializer": initializer},
        "F5": {"model": "F4 + VGGT depth prior, GT pose", "initializer": initializer},
        "F6": {"model": "F5 + VGGT pose", "initializer": initializer},
        "F7": {
            "model": "F6 + Stage C",
            "initializer": None,
            "status": "OPTIONAL_BLOCKED",
        },
    }
    runnable_complete = True
    for arm, row in rows.items():
        row["selected"] = arm in arms
        row["config"] = str((paths.project_root / ARM_CONFIGS[arm]).resolve())
        if arm not in arms:
            row["status"] = "NOT_SELECTED"
            continue
        if arm == "F7":
            continue
        metrics = paths.arm_eval(arm) / "metrics.json"
        if dry_run:
            row["status"] = "PLANNED"
        elif metrics.is_file():
            try:
                row["evaluation"] = _verify_eval_result(paths, arm)
                row["status"] = "COMPLETE"
            except SpringV31Error as exc:
                row["status"] = "INVALID_ARTIFACT"
                row["reason"] = str(exc)
                runnable_complete = False
        else:
            row["status"] = "PENDING"
            runnable_complete = False
    if dry_run:
        matrix_status = "PLANNED"
    elif runnable_complete:
        matrix_status = (
            "COMPLETE_WITH_OPTIONAL_F7_BLOCKED" if "F7" in arms else "COMPLETE"
        )
    else:
        matrix_status = "PHASE_COMPLETE_EVIDENCE_PENDING"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": matrix_status,
        "seed": SEED,
        "protocol": PROTOCOL,
        "selected_arms": list(arms),
        "source_snapshot": dict(source_snapshot),
        "config_protocol": dict(config_protocol),
        "backbone_protocol": dict(backbone_protocol),
        "shared_calibrated_derived_cache": {
            "arms": [arm for arm in ("F4", "F5", "F6") if arm in arms],
            "train": str(paths.train_calibrated_derived),
            "validation": str(paths.validation_calibrated_derived),
        },
        "arms": rows,
        "jobs": dict(job_results),
        "finished_at": _utc_now(),
    }
    _atomic_json(paths.output_root / "matrix_receipt.json", payload)
    return payload


def _plan(paths: Paths, args: argparse.Namespace, arms: Sequence[str]) -> dict[str, Any]:
    def record(job: Job) -> dict[str, Any]:
        return {
            "name": job.name,
            "command": list(job.command),
            "command_shell": _command_text(job.command),
            "expected_output": str(job.expected_output),
            "group": job.group,
            "gpu": job.gpu,
        }

    return {
        "cache": [record(job) for job in _cache_jobs(paths, args, arms)],
        "geometry": [record(job) for job in _geometry_jobs(paths, args, arms)],
        "initializers": [record(job) for job in _initializer_jobs(paths, args, arms)],
        "temporal_training": [
            record(job) for job in _temporal_train_jobs(paths, args, arms)
        ],
        "evaluation": [record(job) for job in _eval_jobs(paths, args, arms)],
    }


def run(args: argparse.Namespace) -> int:
    if args.seed != SEED:
        raise SpringV31Error(f"this protocol only permits seed {SEED}")
    args.devices = _parse_devices(args.devices)
    args.project_root = args.project_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.spring_root = args.spring_root.expanduser().resolve()
    args.ffs_checkpoint = args.ffs_checkpoint.expanduser().resolve()
    args.ffs_repo = args.ffs_repo.expanduser().resolve()
    args.vggt_checkpoint = args.vggt_checkpoint.expanduser().resolve()
    args.vggt_repo = args.vggt_repo.expanduser().resolve()
    paths = Paths.from_args(args)
    arms = _parse_arms(args.arms)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = paths.output_root / "runner.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpringV31Error(f"another runner holds {lock_path}") from exc
        config_protocol = _config_protocol(paths)
        source_snapshot = _source_snapshot(paths)
        if args.phase in {"train", "all"} and not args.dry_run and not source_snapshot["git_clean"]:
            raise SpringV31Error(
                "formal training requires a clean committed source tree; dirty paths: "
                f"{source_snapshot['git_dirty_paths'][:20]}"
            )
        backbone_protocol = (
            _backbone_protocol(paths, args, arms)
            if args.phase in {"cache", "train", "eval", "all"}
            else {}
        )
        _write_f7_blocked(paths)
        plan = _plan(paths, args, arms)
        _atomic_json(
            paths.output_root / "execution_plan.json",
            {
                "schema_version": SCHEMA_VERSION,
                "component": COMPONENT,
                "seed": SEED,
                "protocol": PROTOCOL,
                "phase": args.phase,
                "dry_run": bool(args.dry_run),
                "selected_arms": list(arms),
                "devices": list(args.devices),
                "config_protocol": config_protocol,
                "backbone_protocol": backbone_protocol,
                "plan": plan,
            },
        )
        results: dict[str, Any] = {}
        phases = (
            ("prepare", "cache", "geometry", "train", "eval")
            if args.phase == "all"
            else (args.phase,)
        )
        if "prepare" in phases:
            results["prepare"] = prepare(paths, args)
        elif not args.dry_run:
            results["protocol"] = verify_protocol(paths)
        if args.dry_run:
            receipt = _matrix_receipt(
                paths,
                arms=arms,
                source_snapshot=source_snapshot,
                config_protocol=config_protocol,
                backbone_protocol=backbone_protocol,
                job_results=results,
                dry_run=True,
            )
            print(json.dumps({"status": receipt["status"], "output": str(paths.output_root)}))
            return 0
        if "cache" in phases:
            results["cache"] = _run_parallel(
                _cache_jobs(paths, args, arms),
                paths=paths,
                devices=args.devices,
                dry_run=False,
            )
        if "geometry" in phases:
            results["geometry"] = _run_parallel(
                _geometry_jobs(paths, args, arms),
                paths=paths,
                devices=args.devices,
                dry_run=False,
            )
        if "train" in phases:
            results["initializers"] = _run_parallel(
                _initializer_jobs(paths, args, arms),
                paths=paths,
                devices=args.devices,
                dry_run=False,
            )
            _assert_training_dependencies(paths, arms)
            results["temporal_training"] = _run_parallel(
                _temporal_train_jobs(paths, args, arms),
                paths=paths,
                devices=args.devices,
                dry_run=False,
            )
        if "eval" in phases:
            _assert_training_dependencies(paths, arms)
            results["evaluation"] = _run_parallel(
                _eval_jobs(paths, args, arms),
                paths=paths,
                devices=args.devices,
                dry_run=False,
            )
        receipt = _matrix_receipt(
            paths,
            arms=arms,
            source_snapshot=source_snapshot,
            config_protocol=config_protocol,
            backbone_protocol=backbone_protocol,
            job_results=results,
            dry_run=False,
        )
        print(json.dumps({"status": receipt["status"], "output": str(paths.output_root)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--spring-root",
        type=Path,
        default=PROJECT_ROOT.parent / "spring_dataset" / "spring",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "spring_v3_1_seed42",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--phase",
        choices=("prepare", "cache", "geometry", "train", "eval", "all"),
        default="all",
    )
    parser.add_argument("--arm", "--arms", dest="arms", action="append")
    parser.add_argument(
        "--devices",
        default="0,1,2,3",
        help="comma-separated physical CUDA indices, or cpu",
    )
    parser.add_argument("--audit-samples-per-sequence", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument(
        "--ffs-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/ffs/20-30-48/model_best_bp2_serialize.pth",
    )
    parser.add_argument(
        "--ffs-repo",
        type=Path,
        default=PROJECT_ROOT / "third_party/Fast-FoundationStereo",
    )
    parser.add_argument(
        "--vggt-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/vggt/vggt_omega_1b_512.pt",
    )
    parser.add_argument(
        "--vggt-repo",
        type=Path,
        default=PROJECT_ROOT / "third_party/vggt-omega",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
