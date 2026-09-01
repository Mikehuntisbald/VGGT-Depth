from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from geometry.epipolar import EPIPOLAR_GEOMETRY_CONTRACT
from models.epipolar_refiner import HREpipolarRefiner
from tools.audit_epipolar_training_run import (
    EpipolarTrainingAuditError,
    _expected_runtime_paths,
    _learning_rate_multiplier,
    _runtime_source_contract,
    _validate_source_bundle,
    audit_epipolar_training_run,
    main,
)


BASE_LR = 2.0e-4
WARMUP_STEPS = 500
TOTAL_STEPS = 5_000
CONTROLLED_ROLE = "CONTROLLED_D025_STAGE_C_ABLATION"
POSITIVITY_PROTOCOL = "d025_stage_c_physical_positivity_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _allow_legacy_fixture_bundle_for_controlled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep synthetic runs usable until the shared 55-file producer is committed."""

    def validate(
        value: object,
        *,
        git_hash: str,
        controlled_ablation: bool,
        high_vram: bool = False,
    ) -> dict:
        assert controlled_ablation
        result = _validate_source_bundle(
            value,
            git_hash=git_hash,
            controlled_ablation=False,
        )
        if high_vram:
            result["file_count"] = 56
        return result

    monkeypatch.setattr(
        "tools.audit_epipolar_training_run._validate_source_bundle",
        validate,
    )

    def recompute(training_audit: Path, *args: object) -> dict:
        del args
        path = Path(training_audit).resolve().parent / (
            "d025_final_controlled_audit.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        "tools.audit_epipolar_training_run.audit_d025_evaluation",
        recompute,
    )


def _canonical_config_sha(config: dict) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_receipt() -> dict:
    return {
        "device": "cuda:0",
        "device_type": "cuda",
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_capability": [12, 0],
        "torch_version": str(torch.__version__),
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


def _source_bundle(repository: Path) -> tuple[str, dict]:
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _, git_scopes, _ = _runtime_source_contract(
        git_hash, controlled_ablation=False
    )
    records = []
    for relative in _expected_runtime_paths(
        git_hash=git_hash, controlled_ablation=False
    ):
        payload = subprocess.run(
            ["git", "show", f"{git_hash}:{relative}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        records.append(
            {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}
        )
    encoded = json.dumps(
        {"git_head": git_hash, "files": records},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return git_hash, {
        "schema_version": 1,
        "git_head": git_hash,
        "relevant_paths_clean": True,
        "git_scopes": list(git_scopes),
        "files": records,
        "bundle_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _rectification_receipt(root: Path) -> dict:
    train_sha = "1" * 64
    validation_sha = "2" * 64
    sample_sha = "3" * 64
    receipt = {
        "schema_version": 1,
        "component": "pixel-level-epipolar-rectification-audit",
        "status": "PASS",
        "published_contract": EPIPOLAR_GEOMETRY_CONTRACT["version"],
        "manifests": {
            "train": {"sha256": train_sha},
            "validation": {"sha256": validation_sha},
            "train_validation_sequence_disjoint": True,
        },
        "algorithm": {"feature": "SIFT"},
        "thresholds": {"max_abs_median_dy_px": 1.25},
        "sampling": {"sample_identity_sha256": sample_sha},
        "threshold_checks": [{"name": "pixel-contract", "passed": True}],
        "global": {
            "sampled_frames": 96,
            "covered_frames": 96,
            "coverage_fraction": 1.0,
            "counts": {"ratio_matches": 98_095, "ransac_inliers": 71_436},
            "dy_right_minus_left_px": {
                "absolute": {"p95": 2.02},
                "signed": {"p50": -0.07},
            },
        },
        "metadata_vs_pixels": {
            "conclusion": "INCONSISTENT_WITH_AUDITED_PIXEL_COORDINATES"
        },
    }
    path = root / "rectification.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "schema_version": 1,
        "component": receipt["component"],
        "status": "PASS",
        "contract_version": receipt["published_contract"],
        "manifest_sha256": {"train": train_sha, "validation": validation_sha},
        "algorithm": receipt["algorithm"],
        "thresholds": receipt["thresholds"],
        "counts": {
            "sampled_frames": 96,
            "covered_frames": 96,
            "ratio_matches": 98_095,
            "ransac_inliers": 71_436,
        },
        "pixel_evidence": {
            "coverage_fraction": 1.0,
            "median_right_y_minus_left_y_px": -0.07,
            "p95_abs_right_y_minus_left_y_px": 2.02,
        },
        "metadata_vs_pixels": receipt["metadata_vs_pixels"],
        "sample_identity_sha256": sample_sha,
    }


def _config(
    rectification: dict,
    output: Path,
    *,
    controlled_ablation: bool = False,
    high_vram: bool = False,
) -> dict:
    config = {
        "experiment": "ffs_omega_tsr_epipolar_x2",
        "seed": 42,
        "data": {
            "scale": 2,
            "hr_crop": [384, 768],
            "crop_mode": "random",
            "sequence_length": 3,
            "vggt_context_pairs": 5,
            "epipolar_rectification_audit_path": rectification["path"],
            "epipolar_rectification_audit": rectification,
        },
        "model": {
            "use_history": True,
            "use_vggt_pose": True,
            "epipolar_refinement": True,
            "epipolar_offsets_hr_px": [-2, -1, 0, 1, 2],
            "epipolar_feature_channels": 32,
            "epipolar_correlation_groups": 8,
            "epipolar_head_channels": 48,
            "epipolar_correction_limit_hr_px": 2.0,
            "epipolar_confidence_temperature": 1.0,
            "epipolar_vertical_geometry": EPIPOLAR_GEOMETRY_CONTRACT["version"],
        },
        "train": {
            "stage": "epipolar",
            "steps": 15_000,
            "steps_temporal": 15_000,
            "steps_epipolar": TOTAL_STEPS,
            "precision": "bf16",
            "optimizer": "adamw",
            "learning_rate": BASE_LR,
            "weight_decay": 1.0e-4,
            "warmup_steps": WARMUP_STEPS,
            "micro_batch_size": 2,
            "grad_accumulation": 4,
            "effective_batch_size": 8,
            "checkpoint_interval": 500,
            "log_interval": 1,
            "correction_regularizer_weight": 0.01,
            "compile_model": False,
            "epipolar_output_dir": str(output.resolve()),
        },
    }
    if controlled_ablation:
        config.update(
            {
                "experiment": "ffs_omega_tsr_epipolar_x2_d025_positivity",
                "ablation_protocol": {
                    "name": "stage_c_physical_positivity_from_passing_d025",
                    "required_base": "full_stage_b_d025_15000_and_holdout_pass",
                    "canonical_stage_c_replacement": False,
                },
                "positivity_ablation": {
                    "enabled": True,
                    "sanitize_invalid_sources": True,
                    "lower_bound_hr_px": 0.0,
                    "lr_negative_penalty_weight": 0.10,
                    "raw_negative_penalty_weight": 0.01,
                },
                "stage_c_positivity_ablation": {
                    "enabled": True,
                    "protocol_version": POSITIVITY_PROTOCOL,
                    "requires_passing_d025_base": True,
                    "correction_lower_bound_hr_px": 0.0,
                    "pre_lower_bound_negative_penalty_weight": 0.10,
                    "d025_training_audit_path": "/receipts/d025_training_audit.json",
                    "d025_evaluation_audit_path": (
                        str(
                            (
                                output.parent
                                / "d025_final_controlled_audit.json"
                            ).resolve()
                        )
                    ),
                },
            }
        )
    if high_vram:
        assert controlled_ablation
        config["stage_c_high_vram"] = {
            "enabled": True,
            "protocol_version": "d025_stage_c_high_vram_cuda_preflight_v1",
            "requires_cuda_memory_preflight": True,
            "preflight_receipt_path": str(
                (output.parent / "stage_c_high_vram_preflight.json").resolve()
            ),
            "minimum_headroom_bytes": 2 * 1024**3,
            "oom_fallback": {
                "micro_batch_size": 2,
                "grad_accumulation": 4,
                "effective_batch_size": 8,
            },
        }
        config["train"].update(
            {
                "micro_batch_size": 4,
                "grad_accumulation": 2,
                "effective_batch_size": 8,
            }
        )
    return config


def _refiner() -> HREpipolarRefiner:
    return HREpipolarRefiner(
        feature_channels=32,
        correlation_groups=8,
        candidate_offsets_hr_px=(-2, -1, 0, 1, 2),
        correction_limit_hr_px=2.0,
        confidence_temperature=1.0,
        head_channels=48,
    )


def _optimizer_and_scheduler_state(
    model: HREpipolarRefiner, *, step: int
) -> tuple[dict, dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1.0e-4)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)
    optimizer.step()
    expected_lr = BASE_LR * _learning_rate_multiplier(
        step, total_steps=TOTAL_STEPS, warmup_steps=WARMUP_STEPS
    )
    for state in optimizer.state.values():
        state["step"] = torch.tensor(float(step))
    optimizer.param_groups[0]["lr"] = expected_lr
    optimizer.param_groups[0]["initial_lr"] = BASE_LR
    scheduler = {
        "base_lrs": [BASE_LR],
        "last_epoch": step,
        "_step_count": step + 1,
        "_is_initial": False,
        "_get_lr_called_within_step": False,
        "_last_lr": [expected_lr],
        "lr_lambdas": [None],
    }
    return optimizer.state_dict(), scheduler


def _base_checkpoint(
    root: Path, *, controlled_ablation: bool = False
) -> tuple[Path, str]:
    path = root / "stage_b_final.pt"
    config = {
        "train": {
            "stage": "temporal",
            "steps": 15_000,
            "steps_temporal": 15_000,
        }
    }
    if controlled_ablation:
        config.update(
            {
                "ablation_protocol": {
                    "name": "full_stage_b_rerun_from_final_stage_a",
                    "required_updates": 15_000,
                    "stage_b_warm_start": "forbidden",
                },
                "positivity_ablation": {
                    "enabled": True,
                    "sanitize_invalid_sources": True,
                    "lower_bound_hr_px": 0.0,
                    "lr_negative_penalty_weight": 0.10,
                    "raw_negative_penalty_weight": 0.01,
                },
            }
        )
    torch.save(
        {
            "schema_version": 1,
            "step": 15_000,
            "config": config,
        },
        path,
    )
    return path.resolve(), _sha256(path)


def _rng_states() -> dict:
    return {
        "python": random.Random(42).getstate(),
        "numpy": np.random.RandomState(42).get_state(),
        "torch_cpu": torch.Generator().manual_seed(42).get_state(),
        "torch_cuda": [torch.arange(64, dtype=torch.uint8)],
    }


def _loss(*, controlled_ablation: bool = False) -> dict:
    value = {
        "total": 0.2,
        "disparity": 0.19,
        "correction_regularizer": 1.0,
        "valid_pixel_count": 4096,
    }
    if controlled_ablation:
        value["positivity_penalty"] = 0.001
    return value


def _d025_prerequisite(base_checkpoint: dict, formal_audit_path: Path) -> dict:
    root = formal_audit_path.parent
    d025_eval = root / "d025_eval"
    canonical_eval = root / "canonical_eval"
    d025_eval.mkdir(exist_ok=True)
    canonical_eval.mkdir(exist_ok=True)
    artifact_paths = {
        "d025_training_audit": root / "d025_training_audit.json",
        "d025_preflight": root / "d025_preflight.json",
        "d025_metrics": d025_eval / "metrics.json",
        "d025_metrics_csv": d025_eval / "metrics.csv",
        "canonical_stage_b_report": root / "canonical_stage_b_report.json",
        "canonical_metrics": canonical_eval / "metrics.json",
        "canonical_metrics_csv": canonical_eval / "metrics.csv",
    }
    for path in artifact_paths.values():
        path.write_text("{}\n", encoding="utf-8")

    def identity(path: Path) -> dict[str, str]:
        return {"path": str(path.resolve()), "sha256": _sha256(path)}

    formal_audit = {
        "schema_version": 1,
        "component": "d025-positivity-final-evaluation-audit",
        "status": "D025_FINAL_CONTROLLED_COMPARISON_PASS",
        "read_only": True,
        "final_gate": {
            "eligible": True,
            "result": "PASS",
            "limited_or_intermediate_cannot_pass": True,
        },
        "artifacts": {
            name: identity(path) for name, path in artifact_paths.items()
        },
    }
    formal_audit_path.write_text(
        json.dumps(formal_audit, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "component": "d025-stage-b-prerequisite-for-stage-c-positivity",
        "status": "PASS",
        "protocol_version": POSITIVITY_PROTOCOL,
        "base_checkpoint": dict(base_checkpoint),
        "formal_evaluation_audit": {
            "path": str(formal_audit_path.resolve()),
            "sha256": _sha256(formal_audit_path),
            "component": "d025-positivity-final-evaluation-audit",
            "status": "D025_FINAL_CONTROLLED_COMPARISON_PASS",
            "final_gate": {
                "eligible": True,
                "result": "PASS",
                "limited_or_intermediate_cannot_pass": True,
            },
        },
        "canonical_stage_c_replacement": False,
    }


def _build_run(
    root: Path,
    *,
    complete: bool,
    checkpoint_step: int | None = None,
    log_steps: int | None = None,
    reset_elapsed_at: int | None = None,
    controlled_ablation: bool = False,
    high_vram: bool = False,
) -> dict:
    root.mkdir(parents=True)
    checkpoint_step = TOTAL_STEPS if checkpoint_step is None else checkpoint_step
    log_steps = checkpoint_step if log_steps is None else log_steps
    rectification = _rectification_receipt(root.parent)
    config = _config(
        rectification,
        root,
        controlled_ablation=controlled_ablation,
        high_vram=high_vram,
    )
    model = _refiner()
    assert model.trainable_parameter_count == 69_905
    optimizer, scheduler = _optimizer_and_scheduler_state(model, step=checkpoint_step)
    base_path, base_sha = _base_checkpoint(
        root.parent,
        controlled_ablation=controlled_ablation,
    )
    git_hash, source_bundle = _source_bundle(Path(__file__).parents[1])
    runtime = _runtime_receipt()
    base_checkpoint = {
        "path": str(base_path),
        "sha256": base_sha,
        "step": 15_000,
    }
    d025_prerequisite = (
        _d025_prerequisite(
            base_checkpoint,
            root.parent / "d025_final_controlled_audit.json",
        )
        if controlled_ablation
        else None
    )
    high_vram_preflight = None
    if high_vram:
        assert d025_prerequisite is not None
        from train_epipolar import (
            build_stage_c_high_vram_preflight_receipt,
            validate_stage_c_high_vram_preflight_receipt,
        )

        receipt_path = Path(config["stage_c_high_vram"]["preflight_receipt_path"])
        receipt = build_stage_c_high_vram_preflight_receipt(
            config=config,
            base_checkpoint=base_checkpoint,
            d025_prerequisite=d025_prerequisite,
            runtime_source_bundle=source_bundle,
            training_runtime=runtime,
            peak_cuda_allocated_bytes=18 * 1024**3,
            peak_cuda_reserved_bytes=20 * 1024**3,
            cuda_free_before_bytes=28 * 1024**3,
            cuda_free_after_bytes=11 * 1024**3,
            cuda_total_bytes=32 * 1024**3,
            completed_micro_steps=2,
            gradient_norm=0.5,
            parameters_finite_after_step=True,
            loss=_loss(controlled_ablation=True),
        )
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        high_vram_preflight = validate_stage_c_high_vram_preflight_receipt(
            path=receipt_path,
            config=config,
            base_checkpoint=base_checkpoint,
            d025_prerequisite=d025_prerequisite,
            runtime_source_bundle=source_bundle,
            training_runtime=runtime,
        )
    base_completion = {
        "actual_step": 15_000,
        "configured_steps": 15_000,
        "declared_temporal_steps": 15_000,
        "required_steps": 15_000,
        "canonical_required_steps": 15_000,
        "complete": True,
        "required_for_this_run": True,
    }
    batches_per_epoch = 100
    accumulation = 2 if high_vram else 4
    micro_steps = checkpoint_step * accumulation
    epoch, offset = divmod(micro_steps, batches_per_epoch)
    completion = {
        "actual_step": checkpoint_step,
        "configured_steps": TOTAL_STEPS,
        "execution_complete": checkpoint_step == TOTAL_STEPS,
        "canonical_schedule": True,
        "base_complete": True,
        "cuda_bf16_eligible": True,
        "strict_determinism_eligible": True,
        "formal_training_complete": (
            checkpoint_step == TOTAL_STEPS and not controlled_ablation
        ),
    }
    if controlled_ablation:
        completion.update(
            {
                "controlled_ablation_training_complete": (
                    checkpoint_step == TOTAL_STEPS
                ),
                "canonical_stage_c_replacement": False,
            }
        )
    if high_vram:
        completion["high_vram_preflight_passed"] = True
    payload = {
        "schema_version": 1,
        "component": "ffs-omega-tsr-epipolar-stage-c",
        "model_component": "hr_epipolar_refiner",
        "model": model.state_dict(),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": {},
        "step": checkpoint_step,
        "config": config,
        "git_hash": git_hash,
        "rng_states": _rng_states(),
        "data_cursor": {
            "completed_micro_steps": micro_steps,
            "batches_per_epoch": batches_per_epoch,
            "epoch": epoch,
            "batch_offset_in_epoch": offset,
            "grad_accumulation": accumulation,
            "drop_last": True,
        },
        "base_checkpoint": base_checkpoint,
        "base_lineage": {"valid": True},
        "raw_lineage": {"valid": True},
        "base_completion": base_completion,
        "geometry_contract": EPIPOLAR_GEOMETRY_CONTRACT,
        "rectification_audit": rectification,
        "runtime_source_bundle": source_bundle,
        "training_runtime": runtime,
        "supervision": {"paper_ground_truth": False},
        "parameter_count": 69_905,
        "trainable_refiner_parameter_count": 69_905,
        "loss": _loss(controlled_ablation=controlled_ablation),
        "elapsed_seconds": float(checkpoint_step),
        "completion": completion,
    }
    if controlled_ablation:
        payload["experiment_role"] = CONTROLLED_ROLE
        payload["d025_prerequisite"] = d025_prerequisite
    if high_vram:
        payload["high_vram_preflight"] = high_vram_preflight
    torch.save(payload, root / "latest.pt")
    rows = []
    for step in range(1, log_steps + 1):
        elapsed = float(step)
        if reset_elapsed_at is not None and step >= reset_elapsed_at:
            elapsed = float(step - reset_elapsed_at + 1)
        rows.append(
            {
                "step": step,
                "stage": "epipolar",
                "learning_rate": BASE_LR
                * _learning_rate_multiplier(
                    step, total_steps=TOTAL_STEPS, warmup_steps=WARMUP_STEPS
                ),
                "gradient_norm": 0.5,
                "elapsed_seconds": elapsed,
                "loss": _loss(controlled_ablation=controlled_ablation),
            }
        )
    (root / "train.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    if complete:
        torch.save(payload, root / "final.pt")
        run_steps = 2_500
        segment_elapsed = 2_500.0
        summary = {
            "schema_version": 1,
            "component": "ffs-omega-tsr-epipolar-training-run",
            "status": "TRAINING_COMPLETE",
            "stage": "epipolar",
            "steps": TOTAL_STEPS,
            "configured_steps": TOTAL_STEPS,
            "run_steps": run_steps,
            "elapsed_seconds": float(TOTAL_STEPS),
            "segment_elapsed_seconds": segment_elapsed,
            "segment_steps_per_second": run_steps / segment_elapsed,
            "git_hash": git_hash,
            "config_sha256": _canonical_config_sha(config),
            "training_runtime": runtime,
            "base_checkpoint": base_checkpoint,
            "base_completion": base_completion,
            "runtime_source_bundle_sha256": source_bundle["bundle_sha256"],
            "formal_training_complete": not controlled_ablation,
            "final_checkpoint": {
                "path": str((root / "final.pt").resolve()),
                "sha256": _sha256(root / "final.pt"),
            },
            "latest_checkpoint": {
                "path": str((root / "latest.pt").resolve()),
                "sha256": _sha256(root / "latest.pt"),
            },
            "training_log": {
                "path": str((root / "train.jsonl").resolve()),
                "sha256": _sha256(root / "train.jsonl"),
            },
            "peak_cuda_allocated_bytes": 1_000,
            "peak_cuda_reserved_bytes": 2_000,
        }
        if controlled_ablation:
            summary.update(
                {
                    "experiment_role": CONTROLLED_ROLE,
                    "d025_prerequisite": d025_prerequisite,
                    "controlled_ablation_training_complete": True,
                }
            )
        if high_vram:
            summary["high_vram_preflight"] = high_vram_preflight
        (root / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return payload


def test_complete_formal_run_is_read_only_and_validates_all_receipts(
    tmp_path: Path,
) -> None:
    run = tmp_path / "complete"
    _build_run(run, complete=True)
    mtimes = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()}

    report = audit_epipolar_training_run(run)

    assert report["status"] == "PASS"
    assert report["training_status"] == "TRAINING_COMPLETE"
    assert report["safe_load"]["torch_weights_only"] is True
    assert report["checkpoint_validation"]["runtime_source_bundle"]["file_count"] == 52
    assert report["checkpoint_validation"]["training_runtime"][
        "formal_cuda_bf16_eligible"
    ] is True
    assert report["checkpoint_validation"]["base_checkpoint"]["step"] == 15_000
    assert report["checkpoint_validation"]["completion"][
        "formal_training_complete"
    ] is True
    assert report["checkpoint_validation"]["stage_c_positivity_ablation"] == {
        "enabled": False,
        "experiment_role": "CANONICAL_STAGE_C",
        "d025_prerequisite": None,
    }
    assert report["log_validation"]["loss_schema"] == {
        "stage_c_positivity_ablation_enabled": False,
        "terms": ["correction_regularizer", "disparity", "total"],
        "valid_pixel_count": True,
    }
    assert report["completion"]["formal_training_complete"] is True
    assert (
        report["completion"]["controlled_ablation_training_complete"] is False
    )
    assert report["resume"]["detected"] is True
    assert report["resume"]["final_segment_start_step"] == 2_500
    assert {path: path.stat().st_mtime_ns for path in mtimes} == mtimes


def test_runtime_source_contract_preserves_legacy_52_and_requires_complete_55(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.audit_epipolar_training_run._git_tree_declares_runtime_additions",
        lambda git_hash: (False, False, False),
    )
    _, legacy_scopes, legacy_count = _runtime_source_contract(
        "a" * 40, controlled_ablation=False
    )
    assert legacy_count == 52
    assert "tools/audit_d025_evaluation.py" not in legacy_scopes

    monkeypatch.setattr(
        "tools.audit_epipolar_training_run._git_tree_declares_runtime_additions",
        lambda git_hash: (True, True, True),
    )
    monkeypatch.setattr(
        "tools.audit_epipolar_training_run._git_tree_contains",
        lambda git_hash, relative: True,
    )
    _, canonical_scopes, canonical_count = _runtime_source_contract(
        "b" * 40, controlled_ablation=False
    )
    assert canonical_count == 52
    assert "tools/audit_d025_evaluation.py" not in canonical_scopes
    _, controlled_scopes, controlled_count = _runtime_source_contract(
        "b" * 40, controlled_ablation=True
    )
    assert controlled_count == 55
    assert "tools/audit_d025_evaluation.py" in controlled_scopes
    _, high_scopes, high_count = _runtime_source_contract(
        "b" * 40,
        controlled_ablation=True,
        high_vram=True,
    )
    assert high_count == 56
    assert (
        "configs/ablations/d025_stage_c_positivity_high_vram.yaml"
        in high_scopes
    )

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(
        "tools.audit_epipolar_training_run._runtime_source_contract",
        lambda candidate, *, controlled_ablation, high_vram=False: ((), (), 52),
    )
    with pytest.raises(EpipolarTrainingAuditError, match="wrong role-specific"):
        _validate_source_bundle(
            {},
            git_hash=git_hash,
            controlled_ablation=True,
        )

    monkeypatch.setattr(
        "tools.audit_epipolar_training_run._git_tree_declares_runtime_additions",
        lambda git_hash: (True, False, True),
    )
    with pytest.raises(EpipolarTrainingAuditError, match="declares only part"):
        _runtime_source_contract("c" * 40, controlled_ablation=True)


def test_controlled_positivity_run_has_isolated_loss_and_completion_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_legacy_fixture_bundle_for_controlled(monkeypatch)
    run = tmp_path / "controlled_complete"
    _build_run(run, complete=True, controlled_ablation=True)

    report = audit_epipolar_training_run(run)

    assert report["status"] == "PASS"
    assert report["training_status"] == "TRAINING_COMPLETE"
    assert report["experiment_role"] == CONTROLLED_ROLE
    positivity = report["checkpoint_validation"][
        "stage_c_positivity_ablation"
    ]
    assert positivity["enabled"] is True
    assert positivity["experiment_role"] == CONTROLLED_ROLE
    assert positivity["d025_prerequisite"]["status"] == "PASS"
    assert report["log_validation"]["loss_schema"] == {
        "stage_c_positivity_ablation_enabled": True,
        "terms": [
            "correction_regularizer",
            "disparity",
            "positivity_penalty",
            "total",
        ],
        "valid_pixel_count": True,
    }
    checkpoint_completion = report["checkpoint_validation"]["completion"]
    assert checkpoint_completion["formal_training_complete"] is False
    assert checkpoint_completion["controlled_ablation_training_complete"] is True
    assert checkpoint_completion["canonical_stage_c_replacement"] is False
    assert report["completion"]["formal_training_complete"] is False
    assert report["completion"]["controlled_ablation_training_complete"] is True
    assert report["completion"]["summary"]["experiment_role"] == CONTROLLED_ROLE


def test_high_vram_controlled_run_requires_and_audits_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_legacy_fixture_bundle_for_controlled(monkeypatch)
    run = tmp_path / "controlled_high_vram_complete"
    _build_run(
        run,
        complete=True,
        controlled_ablation=True,
        high_vram=True,
    )

    report = audit_epipolar_training_run(run)

    high = report["checkpoint_validation"]["stage_c_high_vram"]
    assert high["enabled"] is True
    assert high["preflight"]["status"] == "PREFLIGHT_PASS"
    assert high["preflight"]["memory"]["headroom_bytes"] == 12 * 1024**3
    assert report["checkpoint_validation"]["completion"][
        "high_vram_preflight_passed"
    ] is True
    assert report["completion"]["summary"]["high_vram"] is True


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_controlled_training_audit_rejects_missing_or_tampered_formal_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _allow_legacy_fixture_bundle_for_controlled(monkeypatch)
    run = tmp_path / f"controlled_formal_audit_{mutation}"
    _build_run(run, complete=False, controlled_ablation=True)
    formal_audit = tmp_path / "d025_final_controlled_audit.json"
    if mutation == "missing":
        formal_audit.unlink()
    else:
        formal_audit.write_text(
            formal_audit.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )

    with pytest.raises(
        EpipolarTrainingAuditError,
        match="missing|content SHA-256 differs",
    ):
        audit_epipolar_training_run(run)


def test_canonical_and_controlled_loss_schemas_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical_extra_loss"
    _build_run(
        canonical,
        complete=False,
        checkpoint_step=500,
        log_steps=500,
    )
    canonical_log = canonical / "train.jsonl"
    canonical_rows = [
        json.loads(line) for line in canonical_log.read_text(encoding="utf-8").splitlines()
    ]
    canonical_rows[0]["loss"]["positivity_penalty"] = 0.0
    canonical_log.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in canonical_rows),
        encoding="utf-8",
    )
    with pytest.raises(EpipolarTrainingAuditError, match="loss schema differs"):
        audit_epipolar_training_run(canonical)

    _allow_legacy_fixture_bundle_for_controlled(monkeypatch)
    controlled = tmp_path / "controlled_missing_loss"
    _build_run(
        controlled,
        complete=False,
        checkpoint_step=500,
        log_steps=500,
        controlled_ablation=True,
    )
    controlled_log = controlled / "train.jsonl"
    controlled_rows = [
        json.loads(line) for line in controlled_log.read_text(encoding="utf-8").splitlines()
    ]
    del controlled_rows[0]["loss"]["positivity_penalty"]
    controlled_log.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in controlled_rows),
        encoding="utf-8",
    )
    with pytest.raises(EpipolarTrainingAuditError, match="loss schema differs"):
        audit_epipolar_training_run(controlled)


@pytest.mark.parametrize("penalty", [-0.1, float("nan")])
def test_controlled_positivity_penalty_must_be_finite_and_nonnegative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    penalty: float,
) -> None:
    _allow_legacy_fixture_bundle_for_controlled(monkeypatch)
    run = tmp_path / "controlled_bad_penalty"
    _build_run(
        run,
        complete=False,
        checkpoint_step=500,
        log_steps=500,
        controlled_ablation=True,
    )
    log_path = run / "train.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["loss"]["positivity_penalty"] = penalty
    log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    expected = "non-finite constant" if math.isnan(penalty) else "positivity_penalty is negative"
    with pytest.raises(EpipolarTrainingAuditError, match=expected):
        audit_epipolar_training_run(run)


@pytest.mark.parametrize(
    ("tamper", "expected"),
    [
        ("experiment_role", "controlled-ablation role"),
        ("prerequisite_status", "controlled PASS"),
        ("formal_audit_status", "formal evaluation PASS audit"),
        ("formal_audit_gate", "formal evaluation audit final gate"),
        ("formal_completion", "completion receipt is inconsistent"),
        ("controlled_completion", "completion receipt is inconsistent"),
    ],
)
def test_controlled_checkpoint_identity_and_completion_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    expected: str,
) -> None:
    _allow_legacy_fixture_bundle_for_controlled(monkeypatch)
    run = tmp_path / f"controlled_{tamper}"
    _build_run(
        run,
        complete=False,
        checkpoint_step=500,
        log_steps=500,
        controlled_ablation=True,
    )
    checkpoint = run / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if tamper == "experiment_role":
        payload["experiment_role"] = "CANONICAL_STAGE_C"
    elif tamper == "prerequisite_status":
        payload["d025_prerequisite"]["status"] = "FAIL"
    elif tamper == "formal_audit_status":
        payload["d025_prerequisite"]["formal_evaluation_audit"][
            "status"
        ] = "D025_FINAL_CONTROLLED_COMPARISON_FAIL"
    elif tamper == "formal_audit_gate":
        payload["d025_prerequisite"]["formal_evaluation_audit"]["final_gate"][
            "result"
        ] = "FAIL"
    elif tamper == "formal_completion":
        payload["completion"]["formal_training_complete"] = True
    else:
        payload["completion"]["controlled_ablation_training_complete"] = True
    torch.save(payload, checkpoint)

    with pytest.raises(EpipolarTrainingAuditError, match=expected):
        audit_epipolar_training_run(run)


def test_in_progress_run_allows_checkpoint_interval_lag_and_detects_resume(
    tmp_path: Path,
) -> None:
    run = tmp_path / "in_progress"
    _build_run(
        run,
        complete=False,
        checkpoint_step=500,
        log_steps=1_000,
        reset_elapsed_at=501,
    )

    report = audit_epipolar_training_run(run)

    assert report["status"] == "IN_PROGRESS"
    assert report["completion"]["receipt_present"] is False
    assert report["log_validation"]["latest_checkpoint_lag_steps"] == 500
    assert report["log_validation"]["maximum_allowed_lag_steps"] == 500
    assert report["resume"]["detected"] is True
    assert report["resume"]["elapsed_reset_boundaries"] == [501]


def test_tampered_nonfinite_checkpoint_state_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "tampered"
    _build_run(run, complete=False, checkpoint_step=500, log_steps=500)
    path = run / "latest.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    first = next(iter(payload["model"]))
    payload["model"][first].reshape(-1)[0] = float("nan")
    torch.save(payload, path)

    with pytest.raises(EpipolarTrainingAuditError, match="non-finite"):
        audit_epipolar_training_run(run)

    completed = tmp_path / "tampered_summary"
    _build_run(completed, complete=True)
    summary_path = completed / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["training_log"]["sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with pytest.raises(EpipolarTrainingAuditError, match="training_log SHA-256"):
        audit_epipolar_training_run(completed)


def test_in_progress_torn_tail_is_ignored_but_not_counted(tmp_path: Path) -> None:
    run = tmp_path / "torn_tail"
    _build_run(run, complete=False, checkpoint_step=500, log_steps=750)
    with (run / "train.jsonl").open("ab") as handle:
        handle.write(b'{"step": 751, "stage": "epipolar"')

    report = audit_epipolar_training_run(run)

    assert report["status"] == "IN_PROGRESS"
    assert report["log_validation"]["records"] == 750
    assert len(report["log_validation"]["warnings"]) == 1
    assert "unterminated" in report["log_validation"]["warnings"][0]

    completed = tmp_path / "completed_torn_tail"
    _build_run(completed, complete=True)
    with (completed / "train.jsonl").open("ab") as handle:
        handle.write(b'{"step": 5001')
    with pytest.raises(EpipolarTrainingAuditError, match="incomplete final line"):
        audit_epipolar_training_run(completed)


def test_cli_refuses_json_output_inside_audited_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _build_run(run, complete=False, checkpoint_step=500, log_steps=500)

    exit_code = main(
        ["--output-dir", str(run), "--json-out", str(run / "audit.json")]
    )

    assert exit_code == 2
    assert not (run / "audit.json").exists()
