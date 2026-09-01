from __future__ import annotations

import copy
import csv
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import eval_epipolar
from data.manifest import ManifestRecord, write_manifest
from evaluation import MethodMetricAccumulator, compute_sample_metrics
from models.epipolar_refiner import HREpipolarRefiner
from train_epipolar import PSEUDO_GT_SUPERVISION, resolve_epipolar_config
from utils.checkpoint import CheckpointMismatchError, capture_rng_state


def _config() -> dict[str, object]:
    config = resolve_epipolar_config(
        Path(__file__).parents[1] / "configs" / "epipolar_x2.yaml"
    )
    # DictConfig's nested objects need an explicit plain conversion.
    from omegaconf import OmegaConf

    plain = OmegaConf.to_container(config, resolve=True)
    assert isinstance(plain, dict)
    return plain


def _formal_training_runtime() -> dict[str, object]:
    return {
        "device": "cuda",
        "device_type": "cuda",
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_capability": [12, 0],
        "torch_version": "2.7.0+cu128",
        "cuda_version": "12.8",
        "cuda_available": True,
        "bf16_supported": True,
        "autocast_enabled": True,
        "autocast_dtype": "torch.bfloat16",
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "strict_determinism_eligible": True,
        "formal_cuda_bf16_eligible": True,
    }


def _checkpoint(path: Path, *, include_model: bool = True) -> dict[str, object]:
    config = _config()
    refiner = HREpipolarRefiner()
    parameter_ids = list(range(len(list(refiner.parameters()))))
    rectification_audit = {
        "path": "/audit.json",
        "sha256": eval_epipolar.FORMAL_RECTIFICATION_AUDIT_SHA256,
        "schema_version": 1,
        "component": "pixel-level-epipolar-rectification-audit",
        "status": "PASS",
        "contract_version": eval_epipolar.EPIPOLAR_GEOMETRY_CONTRACT["version"],
        "manifest_sha256": {"train": "c" * 64, "validation": "d" * 64},
        "algorithm": {},
        "thresholds": {},
        "counts": {
            "sampled_frames": 96,
            "covered_frames": 96,
            "ratio_matches": 98_095,
            "ransac_inliers": 71_436,
        },
        "pixel_evidence": {},
        "metadata_vs_pixels": {},
        "sample_identity_sha256": "e" * 64,
    }
    runtime_file_records = [{"path": "train.py", "sha256": "f" * 64}]
    runtime_bundle_sha = eval_epipolar.hashlib.sha256(
        eval_epipolar.json.dumps(
            {"git_head": "a" * 40, "files": runtime_file_records},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    runtime_source_bundle = {
        "schema_version": 1,
        "git_head": "a" * 40,
        "relevant_paths_clean": True,
        "git_scopes": ["train.py"],
        "files": runtime_file_records,
        "bundle_sha256": runtime_bundle_sha,
    }
    config["data"]["epipolar_rectification_audit_path"] = "/audit.json"
    config["data"]["epipolar_rectification_audit"] = rectification_audit
    payload: dict[str, object] = {
        "schema_version": 1,
        "component": eval_epipolar.STAGE_C_COMPONENT,
        "model_component": eval_epipolar.STAGE_C_MODEL_COMPONENT,
        "optimizer": {
            "state": {
                parameter_id: {
                    "step": torch.tensor(3.0),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
                for parameter_id, parameter in zip(
                    parameter_ids, refiner.parameters(), strict=True
                )
            },
            "param_groups": [
                {
                    "params": parameter_ids,
                    "initial_lr": float(config["train"]["learning_rate"]),
                    "lr": float(config["train"]["learning_rate"])
                    * eval_epipolar.learning_rate_multiplier(
                        3,
                        total_steps=int(config["train"]["steps_epipolar"]),
                        warmup_steps=int(config["train"]["warmup_steps"]),
                    ),
                    "weight_decay": float(config["train"]["weight_decay"]),
                    "eps": 1e-8,
                    "betas": (0.9, 0.999),
                    "amsgrad": False,
                    "maximize": False,
                }
            ],
        },
        "scheduler": {
            "last_epoch": 3,
            "_step_count": 4,
            "base_lrs": [float(config["train"]["learning_rate"])],
            "_last_lr": [
                float(config["train"]["learning_rate"])
                * eval_epipolar.learning_rate_multiplier(
                    3,
                    total_steps=int(config["train"]["steps_epipolar"]),
                    warmup_steps=int(config["train"]["warmup_steps"]),
                )
            ],
        },
        "scaler": {},
        "step": 3,
        "config": config,
        "git_hash": "a" * 40,
        "rng_states": capture_rng_state(),
        "base_checkpoint": {"path": "/base.pt", "sha256": "b" * 64, "step": 2},
        "base_lineage": {"lineage": "exact"},
        "raw_lineage": {"raw_vggt_identity": {"component": "vggt-omega"}},
        "geometry_contract": eval_epipolar.EPIPOLAR_GEOMETRY_CONTRACT,
        "rectification_audit": rectification_audit,
        "runtime_source_bundle": runtime_source_bundle,
        "training_runtime": _formal_training_runtime(),
        "supervision": PSEUDO_GT_SUPERVISION,
        "parameter_count": refiner.trainable_parameter_count,
        "trainable_refiner_parameter_count": refiner.trainable_parameter_count,
    }
    if include_model:
        payload["model"] = refiner.state_dict()
    torch.save(payload, path)
    return payload


def test_stage_c_checkpoint_reconstructs_exact_refiner(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage_c.pt"
    _checkpoint(checkpoint)

    refiner, metadata = eval_epipolar.load_stage_c_checkpoint(
        checkpoint, evaluation_config=_config()
    )

    assert refiner.trainable_parameter_count == 69_905
    assert metadata["parameter_count"] == 69_905
    assert metadata["step"] == 3
    assert metadata["base_checkpoint"]["sha256"] == "b" * 64
    assert metadata["supervision"] == PSEUDO_GT_SUPERVISION
    assert metadata["training_state_receipt"]["optimizer_steps_consistent"]
    assert metadata["training_runtime_receipt"]["eligible"]


def test_stage_c_checkpoint_requires_a_typed_training_runtime(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stage_c.pt"
    payload = _checkpoint(checkpoint)
    payload.pop("training_runtime")
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointMismatchError, match="training_runtime"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )

    payload = _checkpoint(checkpoint)
    payload["training_runtime"].pop("deterministic_algorithms_enabled")  # type: ignore[union-attr]
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointMismatchError, match="legacy|missing|malformed"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )

    payload = _checkpoint(checkpoint)
    payload["training_runtime"]["formal_cuda_bf16_eligible"] = False  # type: ignore[index]
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointMismatchError, match="eligibility flag"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deterministic_algorithms_enabled", False),
        ("deterministic_algorithms_warn_only", True),
        ("cublas_workspace_config", ":16:8"),
        ("cudnn_deterministic", False),
        ("cudnn_benchmark", True),
    ],
)
def test_stage_c_checkpoint_rejects_non_strict_determinism(
    tmp_path: Path, field: str, value: object
) -> None:
    checkpoint = tmp_path / "stage_c.pt"
    payload = _checkpoint(checkpoint)
    runtime = payload["training_runtime"]
    assert isinstance(runtime, dict)
    runtime[field] = value
    runtime["strict_determinism_eligible"] = False
    runtime["formal_cuda_bf16_eligible"] = False
    torch.save(payload, checkpoint)

    with pytest.raises(CheckpointMismatchError, match="legacy/ineligible"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )


def test_stage_c_checkpoint_rejects_legacy_refiner_key_and_missing_model(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    payload = _checkpoint(checkpoint, include_model=False)
    payload["refiner"] = HREpipolarRefiner().state_dict()
    torch.save(payload, checkpoint)

    with pytest.raises(CheckpointMismatchError, match="model"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )


def test_stage_c_checkpoint_rejects_legacy_geometry_and_malformed_base(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stage_c.pt"
    payload = _checkpoint(checkpoint)
    payload.pop("geometry_contract")
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointMismatchError, match="geometry_contract"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )

    payload = _checkpoint(checkpoint)
    payload["base_checkpoint"] = {
        "path": "",
        "sha256": "b" * 64,
        "step": 2,
    }
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointMismatchError, match="path"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )


def test_stage_c_checkpoint_rejects_architecture_and_parameter_drift(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stage_c.pt"
    payload = _checkpoint(checkpoint)
    payload["parameter_count"] = 69_904
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointMismatchError, match="parameter count"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )

    payload = _checkpoint(checkpoint)
    changed = _config()
    changed["model"]["epipolar_offsets_hr_px"] = [-2, 0, 2]  # type: ignore[index]
    with pytest.raises(CheckpointMismatchError, match="architecture/search"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=changed
        )

    payload = _checkpoint(checkpoint)
    payload["config"]["data"]["sequence_length"] = 1  # type: ignore[index]
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointMismatchError, match="causal/geometry"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )


def test_stage_c_checkpoint_rejects_inconsistent_optimizer_scheduler_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stage_c.pt"
    payload = _checkpoint(checkpoint)
    payload["scheduler"]["last_epoch"] = 2  # type: ignore[index]
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointMismatchError, match="scheduler progress"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )

    payload = _checkpoint(checkpoint)
    payload["optimizer"] = {}
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointMismatchError, match="AdamW state"):
        eval_epipolar.load_stage_c_checkpoint(
            checkpoint, evaluation_config=_config()
        )

def test_paired_refinement_metrics_use_one_identical_domain() -> None:
    target = torch.tensor([[[[10.0, 10.0, 10.0, 10.0]]]])
    base = torch.tensor([[[[12.0, 11.0, 10.0, 8.0]]]])
    refined = torch.tensor([[[[11.0, 12.0, 10.0, 9.0]]]])
    trusted = torch.tensor([[[[True, True, False, True]]]])

    metrics = eval_epipolar.paired_refinement_metrics(
        base, refined, target, trusted
    )

    # Improvements are +1, -1, +1 on the same three selected pixels.
    assert metrics["paired_epe_improvement_hr_px"].count == 3
    assert metrics["paired_epe_improvement_hr_px"].value == pytest.approx(1 / 3)
    assert metrics["paired_refined_better_rate"].value == pytest.approx(2 / 3)
    assert metrics["paired_refined_worse_rate"].value == pytest.approx(1 / 3)
    assert metrics["paired_unchanged_rate"].value == pytest.approx(0.0)
    assert metrics["paired_finite_coverage_rate"].value == pytest.approx(1.0)
    assert metrics["paired_nonfinite_rate"].value == pytest.approx(0.0)


def test_paired_refinement_nonfinite_does_not_shrink_domain() -> None:
    target = torch.ones((1, 1, 1, 2))
    base = target.clone()
    refined = torch.tensor([[[[1.0, float("nan")]]]])
    metrics = eval_epipolar.paired_refinement_metrics(
        base, refined, target, torch.ones_like(target, dtype=torch.bool)
    )
    improvement = metrics["paired_epe_improvement_hr_px"]
    assert not improvement.valid
    assert improvement.count == 2
    assert metrics["paired_finite_coverage_rate"].value == pytest.approx(0.5)
    assert metrics["paired_nonfinite_rate"].value == pytest.approx(0.5)
    for name in (
        "paired_refined_better_rate",
        "paired_refined_worse_rate",
        "paired_unchanged_rate",
    ):
        assert not metrics[name].valid
        assert metrics[name].count == 2


def test_empty_paired_and_finite_statistics_are_null_not_zero() -> None:
    value = torch.ones((1, 1, 2, 2))
    metrics = eval_epipolar.paired_refinement_metrics(
        value,
        value,
        value,
        torch.zeros_like(value, dtype=torch.bool),
    )
    assert all(not result.valid and result.count == 0 for result in metrics.values())

    accumulator = eval_epipolar.FiniteStatisticsAccumulator()
    accumulator.update(value, torch.zeros_like(value, dtype=torch.bool))
    assert accumulator.finalize().to_dict() == {
        "count": 0,
        "mean": None,
        "minimum": None,
        "maximum": None,
        "valid": False,
    }


def test_nonfinite_frozen_base_state_is_rejected() -> None:
    module = torch.nn.Linear(2, 1)
    eval_epipolar.validate_finite_module_state(module, label="base")
    with torch.no_grad():
        module.weight[0, 0] = float("nan")
    with pytest.raises(CheckpointMismatchError, match="non-finite"):
        eval_epipolar.validate_finite_module_state(module, label="base")


def test_runtime_source_bundle_is_bound_to_checkpoint_commit(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )
    source = tmp_path / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "runtime.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "runtime"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    records = [
        {
            "path": "runtime.py",
            "sha256": eval_epipolar.sha256_file(source),
        }
    ]
    encoded = eval_epipolar.json.dumps(
        {"git_head": commit, "files": records},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    bundle = {
        "schema_version": 1,
        "git_head": commit,
        "relevant_paths_clean": True,
        "git_scopes": ["runtime.py"],
        "files": records,
        "bundle_sha256": eval_epipolar.hashlib.sha256(encoded).hexdigest(),
    }
    result = eval_epipolar.validate_runtime_source_bundle(
        bundle,
        checkpoint_git_hash=commit,
        project_root=tmp_path,
        expected_scopes=("runtime.py",),
        expected_paths=("runtime.py",),
    )
    assert result["all_byte_identical"]

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(CheckpointMismatchError, match="differs"):
        eval_epipolar.validate_runtime_source_bundle(
            bundle,
            checkpoint_git_hash=commit,
            project_root=tmp_path,
            expected_scopes=("runtime.py",),
            expected_paths=("runtime.py",),
        )


def test_formal_coverage_is_exactly_244_240_238() -> None:
    coverage = {
        "manifest_records": 244,
        "derived_endpoint_records": 240,
        "evaluable_t3_windows": 238,
    }
    assert eval_epipolar.require_formal_stage_c_coverage(
        coverage,
        manifest_sha256=eval_epipolar.FORMAL_VALIDATION_MANIFEST_SHA256,
    ) == coverage
    for field in coverage:
        changed = dict(coverage)
        changed[field] -= 1
        with pytest.raises(ValueError, match="244/240/238"):
            eval_epipolar.require_formal_stage_c_coverage(
                changed,
                manifest_sha256=eval_epipolar.FORMAL_VALIDATION_MANIFEST_SHA256,
            )
    with pytest.raises(ValueError, match="not the bound"):
        eval_epipolar.require_formal_stage_c_coverage(
            coverage,
            manifest_sha256="0" * 64,
        )


def test_intermediate_checkpoints_are_acceptance_ineligible() -> None:
    stage_c = {"step": 5000, "config": {"train": {"steps_epipolar": 5000}}}
    base = {
        "step": 15000,
        "training_config": {
            "train": {"steps": 15000, "steps_temporal": 15000}
        },
    }
    assert eval_epipolar.checkpoint_completion_status(stage_c, base)[
        "all_complete"
    ]
    stage_c["step"] = 1
    status = eval_epipolar.checkpoint_completion_status(stage_c, base)
    assert not status["all_complete"]
    assert not status["stage_c"]["complete"]

    stage_c["config"]["train"]["steps_epipolar"] = 1
    status = eval_epipolar.checkpoint_completion_status(stage_c, base)
    assert status["stage_c"]["execution_complete"]
    assert not status["stage_c"]["canonical_schedule"]
    assert not status["all_complete"]

    stage_c["step"] = 5000
    stage_c["config"]["train"]["steps_epipolar"] = 5000
    base["step"] = 1
    base["training_config"]["train"]["steps"] = 1
    status = eval_epipolar.checkpoint_completion_status(stage_c, base)
    assert status["stage_b_base"]["execution_complete"]
    assert not status["stage_b_base"]["canonical_schedule"]
    assert not status["all_complete"]

    base["training_config"]["train"]["steps_temporal"] = 1
    status = eval_epipolar.checkpoint_completion_status(stage_c, base)
    assert status["stage_b_base"]["execution_complete"]
    assert not status["stage_b_base"]["canonical_schedule"]
    assert not status["all_complete"]


def test_formal_crop_must_match_checkpoint_and_384x768() -> None:
    stage = {
        "config": {"data": {"hr_crop": [384, 768], "crop_mode": "random"}}
    }
    evaluation = {"data": {"hr_crop": [384, 768], "crop_mode": "fixed"}}
    result = eval_epipolar.validate_formal_crop_contract(
        stage, evaluation, limited_smoke=False
    )
    assert result["eligible"]

    evaluation["data"]["hr_crop"] = [32, 64]
    with pytest.raises(CheckpointMismatchError, match=r"\[384,768\]"):
        eval_epipolar.validate_formal_crop_contract(
            stage, evaluation, limited_smoke=False
        )
    result = eval_epipolar.validate_formal_crop_contract(
        stage, evaluation, limited_smoke=True
    )
    assert not result["eligible"]


def test_cpu_execution_is_explicitly_smoke_only() -> None:
    eval_epipolar.seed_everything(42, deterministic=True)
    train = {
        "precision": "bf16",
        "optimizer": "adamw",
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "warmup_steps": 500,
        "micro_batch_size": 2,
        "grad_accumulation": 4,
        "correction_regularizer_weight": 0.01,
    }
    stage = {
        "config": {"train": dict(train)},
        "training_runtime_receipt": eval_epipolar.validate_stage_c_training_runtime(
            _formal_training_runtime()
        ),
    }
    evaluation = {"train": dict(train)}
    result = eval_epipolar.validate_formal_execution_contract(
        stage,
        evaluation,
        device=torch.device("cpu"),
        limited_smoke=True,
    )
    assert not result["eligible"]
    assert result["autocast_dtype"] is None
    with pytest.raises(CheckpointMismatchError, match="CUDA device"):
        eval_epipolar.validate_formal_execution_contract(
            stage,
            evaluation,
            device=torch.device("cpu"),
            limited_smoke=False,
        )


def test_cpu_trained_checkpoint_is_never_acceptance_eligible() -> None:
    train = {
        "precision": "bf16",
        "optimizer": "adamw",
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "warmup_steps": 500,
        "micro_batch_size": 2,
        "grad_accumulation": 4,
        "correction_regularizer_weight": 0.01,
    }
    cpu_runtime = {
        **_formal_training_runtime(),
        "device": "cpu",
        "device_type": "cpu",
        "device_name": None,
        "device_capability": None,
        "bf16_supported": False,
        "autocast_enabled": False,
        "autocast_dtype": None,
        "formal_cuda_bf16_eligible": False,
    }
    receipt = eval_epipolar.validate_stage_c_training_runtime(cpu_runtime)
    assert not receipt["eligible"]
    stage = {
        "config": {"train": dict(train)},
        "training_runtime_receipt": receipt,
    }
    evaluation = {"train": dict(train)}
    limited = eval_epipolar.validate_formal_execution_contract(
        stage,
        evaluation,
        device=torch.device("cpu"),
        limited_smoke=True,
    )
    assert not limited["recorded_training_eligible"]
    assert not limited["eligible"]


def test_recorded_training_lineage_must_match_all_recomputed_fields() -> None:
    metadata = {
        "base_lineage": {"policy": "exact"},
        "raw_lineage": {"receipt_sha256": "a" * 64},
    }
    eval_epipolar.validate_recorded_stage_c_training_lineage(
        metadata,
        recomputed_base_lineage={"policy": "exact"},
        recomputed_raw_lineage={"receipt_sha256": "a" * 64},
    )
    with pytest.raises(CheckpointMismatchError, match="raw/derived"):
        eval_epipolar.validate_recorded_stage_c_training_lineage(
            metadata,
            recomputed_base_lineage={"policy": "exact"},
            recomputed_raw_lineage={"receipt_sha256": "b" * 64},
        )


def test_rectification_audit_binding_accepts_sha_identical_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text("{}\n", encoding="utf-8")
    train_sha = eval_epipolar.sha256_file(train_manifest)
    validation_sha = "d" * 64
    current_path = tmp_path / "relocated_audit.json"
    current_path.write_text("{}\n", encoding="utf-8")
    current = {
        "path": str(current_path),
        "sha256": eval_epipolar.FORMAL_RECTIFICATION_AUDIT_SHA256,
        "schema_version": 1,
        "component": "pixel-level-epipolar-rectification-audit",
        "status": "PASS",
        "contract_version": eval_epipolar.EPIPOLAR_GEOMETRY_CONTRACT["version"],
        "manifest_sha256": {"train": train_sha, "validation": validation_sha},
        "algorithm": {"feature": "SIFT"},
        "thresholds": {"strict": True},
        "counts": {
            "sampled_frames": 96,
            "covered_frames": 96,
            "ratio_matches": 98_095,
            "ransac_inliers": 71_436,
        },
        "pixel_evidence": {"coverage_fraction": 1.0},
        "metadata_vs_pixels": {"conclusion": "inconsistent"},
        "sample_identity_sha256": "e" * 64,
    }
    recorded = {**current, "path": "/recorded/audit.json"}
    stage = {
        "config": {
            "data": {
                "manifest_path": str(train_manifest),
                "epipolar_rectification_audit_path": recorded["path"],
                "epipolar_rectification_audit": recorded,
            }
        },
        "rectification_audit": recorded,
    }
    monkeypatch.setattr(
        eval_epipolar,
        "_validated_rectification_audit",
        lambda path, expected_train_manifest_sha256: current,
    )
    result = eval_epipolar.validate_rectification_audit_binding(
        stage,
        receipt_path=current_path,
        validation_manifest_sha256=validation_sha,
    )
    assert result["checkpoint_recorded_path"] == "/recorded/audit.json"
    assert result["current_verified_path"] == str(current_path)

    changed = copy.deepcopy(current)
    changed["counts"]["sampled_frames"] = 95
    monkeypatch.setattr(
        eval_epipolar,
        "_validated_rectification_audit",
        lambda path, expected_train_manifest_sha256: changed,
    )
    with pytest.raises(CheckpointMismatchError, match="differs"):
        eval_epipolar.validate_rectification_audit_binding(
            stage,
            receipt_path=current_path,
            validation_manifest_sha256=validation_sha,
        )


def test_epipolar_batch_rechecks_causality_and_same_crop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        eval_epipolar,
        "validate_temporal_batch_causality",
        lambda batch: calls.append(batch),
    )
    crop = {"x": 0, "y": 0, "width": 8, "height": 4, "spatial_scale": 2}
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    Image.new("RGB", (8, 4)).save(left_path)
    Image.new("RGB", (8, 4)).save(right_path)
    right_sha256 = eval_epipolar.sha256_file(right_path)
    record = {
        "left_path": str(left_path),
        "right_path": str(right_path),
        "image_size_wh": [8, 4],
        "K": [[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        "K_right": [[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
    }
    batch = {
        "rgb_hr_sequence": torch.zeros((1, 3, 3, 4, 8)),
        "rgb_right_hr": torch.zeros((1, 3, 4, 8)),
        "K_hr_sequence": torch.tensor(record["K"]).repeat(1, 3, 1, 1),
        "K_right_hr": torch.tensor(record["K_right"]).unsqueeze(0),
        "right_intrinsics_source": ["manifest.K_right"],
        "epipolar_right_row_scale": torch.ones(1),
        "epipolar_right_row_offset_hr_px": torch.zeros(1),
        "epipolar_right_row_mapping_source": [
            eval_epipolar.EPIPOLAR_GEOMETRY_CONTRACT["version"]
        ],
        "epipolar_crop_hr_px": [crop],
        "sequence_id": ["sequence"],
        "frame_ids": torch.tensor([[0, 1, 2]]),
        "manifest_indices": torch.tensor([[0, 1, 2]]),
        "identity_metadata": [
            {
                "crop_hr_px": dict(crop),
                "manifest_path": str(tmp_path / "manifest.jsonl"),
                "per_time_ffs": [
                    {},
                    {},
                    {
                        "manifest_record": record,
                        "source_sha256": {"right": right_sha256},
                    },
                ],
            }
        ],
        "right_path": [str(right_path)],
        "right_sha256": [right_sha256],
    }
    rows = eval_epipolar.validate_epipolar_batch_causality(batch)
    assert rows == [(2, "sequence", 2, right_sha256)]
    assert calls == [batch]

    batch["epipolar_crop_hr_px"] = [{**crop, "x": 2}]
    with pytest.raises(ValueError, match="temporal HR crop"):
        eval_epipolar.validate_epipolar_batch_causality(batch)

    batch["epipolar_crop_hr_px"] = [crop]
    Image.new("RGB", (10, 4)).save(right_path)
    changed_sha256 = eval_epipolar.sha256_file(right_path)
    batch["right_sha256"] = [changed_sha256]
    batch["identity_metadata"][0]["per_time_ffs"][-1]["source_sha256"][
        "right"
    ] = changed_sha256
    with pytest.raises(ValueError, match="original dimensions"):
        eval_epipolar.validate_epipolar_batch_causality(batch)

    Image.new("RGB", (8, 4)).save(right_path)
    restored_sha256 = eval_epipolar.sha256_file(right_path)
    batch["right_sha256"] = [restored_sha256]
    batch["identity_metadata"][0]["per_time_ffs"][-1]["source_sha256"][
        "right"
    ] = restored_sha256
    record["K_right"][1][2] = 7.4
    batch["K_right_hr"][0, 1, 2] = 7.4
    eval_epipolar.validate_epipolar_batch_causality(batch)

    record["K_right"][0][0] = 2.0
    batch["K_right_hr"][0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="equal left/right fx and cx"):
        eval_epipolar.validate_epipolar_batch_causality(batch)

    record["K_right"][0][0] = 1.0
    batch["K_right_hr"][0, 0, 0] = 1.0
    batch["right_sha256"] = ["0" * 64]
    with pytest.raises(ValueError, match="SHA-256 differs"):
        eval_epipolar.validate_epipolar_batch_causality(batch)


def _record(sequence_id: str, frame_id: int) -> ManifestRecord:
    return ManifestRecord(
        sequence_id=sequence_id,
        frame_id=frame_id,
        timestamp=float(frame_id),
        left_path=f"left-{frame_id}.png",
        right_path=f"right-{frame_id}.png",
        K=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        baseline_m=0.1,
    )


def test_stage_c_lineage_requires_disjoint_videos_and_exact_base(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.jsonl"
    write_manifest(train_manifest, [_record("train-video", 0)])
    paths = {
        "manifest_path": str(train_manifest),
        "observation_cache_root": "/train/observation",
        "teacher_cache_root": "/train/teacher",
        "derived_geometry_cache_root": "/train/derived",
    }
    stage_metadata = {
        "base_checkpoint": {"sha256": "b" * 64, "step": 2},
        "base_lineage": {"lineage": "exact"},
        "raw_lineage": {"raw_vggt_identity": {"id": "omega"}},
        "config": {"data": paths},
    }
    base_metadata = {
        "checkpoint_sha256": "b" * 64,
        "step": 2,
        "training_config": {"data": copy.deepcopy(paths)},
    }
    validation = SimpleNamespace(records=[_record("heldout-video", 1)])
    holdout = {
        "sequence_overlap": [],
        "evaluation_raw_vggt": {"identity": {"id": "omega"}},
    }

    result = eval_epipolar._validate_stage_c_and_base_lineage(
        stage_c_metadata=stage_metadata,
        base_metadata=base_metadata,
        recomputed_base_lineage={"lineage": "exact"},
        validation_dataset=validation,
        holdout_lineage=holdout,
    )
    assert result["sequence_overlap"] == []
    assert result["validation_sequences"] == ["heldout-video"]

    validation.records = [_record("train-video", 2)]
    with pytest.raises(CheckpointMismatchError, match="overlap"):
        eval_epipolar._validate_stage_c_and_base_lineage(
            stage_c_metadata=stage_metadata,
            base_metadata=base_metadata,
            recomputed_base_lineage={"lineage": "exact"},
            validation_dataset=validation,
            holdout_lineage=holdout,
        )


def test_metric_rows_and_visualizations_cover_base_and_refined(tmp_path: Path) -> None:
    shape = (1, 1, 4, 6)
    target = torch.ones(shape) * 5
    trusted = torch.ones(shape, dtype=torch.bool)
    confidence = torch.ones(shape)
    valid = torch.ones(shape, dtype=torch.bool)
    base = target + 1
    refined = target + 0.25
    methods = {}
    for name, prediction in (
        ("T3_VGGT_base", base),
        ("T3_VGGT_epipolar", refined),
    ):
        accumulator = MethodMetricAccumulator()
        accumulator.update(
            compute_sample_metrics(
                prediction,
                target,
                target_trusted_mask=trusted,
                ffs_confidence_hr=confidence,
                ffs_valid_mask_hr=valid,
                ffs_trusted_mask_hr=trusted,
            )
        )
        methods[name] = {
            metric: result.to_dict()
            for metric, result in accumulator.finalize().items()
        }
    csv_path = tmp_path / "metrics.csv"
    eval_epipolar._write_csv(csv_path, methods)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["method"] for row in rows] == [
        "T3_VGGT_base",
        "T3_VGGT_epipolar",
    ]

    eval_epipolar._save_visualization(
        tmp_path / "visualizations",
        sample_name="sample",
        rgb_left_hr=torch.rand(3, 4, 6),
        rgb_right_hr=torch.rand(3, 4, 6),
        base_disparity_hr_px=base[0],
        refined_disparity_hr_px=refined[0],
        correction_hr_px=(refined - base)[0],
        confidence=torch.ones((1, 4, 6)) * 0.8,
        target_disparity_hr_px=target[0],
        target_trusted_mask=trusted[0],
        candidate_valid_mask=torch.ones_like(trusted[0]),
        correction_limit_hr_px=2.0,
    )
    generated = {path.name for path in (tmp_path / "visualizations/sample").iterdir()}
    assert generated == {
        "rgb_left.png",
        "rgb_right.png",
        "base_disparity_hr_px.png",
        "refined_disparity_hr_px.png",
        "target_disparity_hr_px.png",
        "correction_hr_px.png",
        "base_absolute_error_hr_px.png",
        "refined_absolute_error_hr_px.png",
        "epipolar_confidence.png",
        "target_trusted_mask.png",
        "candidate_valid_mask.png",
        "visualization_metadata.json",
    }
    metadata = eval_epipolar.json.loads(
        (tmp_path / "visualizations/sample/visualization_metadata.json").read_text()
    )
    assert metadata["units"] == "HR pixels"
    assert metadata["shared_disparity_display_range_hr_px"] == [5.0, 5.0]


def test_cli_requires_explicit_validation_paths_and_labels_limit_smoke() -> None:
    parser = eval_epipolar.build_parser()
    args = parser.parse_args(
        [
            "--config",
            "config.yaml",
            "--checkpoint",
            "stage-c.pt",
            "--manifest",
            "val.jsonl",
            "--observation-cache-root",
            "observation",
            "--teacher-cache-root",
            "teacher",
            "--derived-cache-root",
            "derived",
            "--rectification-audit",
            "rectification.json",
            "--output",
            "output",
            "--limit",
            "1",
        ]
    )
    assert args.limit == 1
    assert args.manifest == Path("val.jsonl")
