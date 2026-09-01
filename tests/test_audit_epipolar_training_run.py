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
    audit_epipolar_training_run,
    main,
)


GIT_SCOPES = [
    "train_epipolar.py",
    "eval_epipolar.py",
    "train.py",
    "eval.py",
    "configs/epipolar_x2.yaml",
    "configs/temporal_x2.yaml",
    "configs/mvp_x2.yaml",
    "pyproject.toml",
    "src",
]
BASE_LR = 2.0e-4
WARMUP_STEPS = 500
TOTAL_STEPS = 5_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    records = []
    for relative in _expected_runtime_paths():
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
        "git_scopes": GIT_SCOPES,
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


def _config(rectification: dict, output: Path) -> dict:
    return {
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


def _base_checkpoint(root: Path) -> tuple[Path, str]:
    path = root / "stage_b_final.pt"
    torch.save(
        {
            "schema_version": 1,
            "step": 15_000,
            "config": {
                "train": {
                    "stage": "temporal",
                    "steps": 15_000,
                    "steps_temporal": 15_000,
                }
            },
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


def _loss() -> dict:
    return {
        "total": 0.2,
        "disparity": 0.19,
        "correction_regularizer": 1.0,
        "valid_pixel_count": 4096,
    }


def _build_run(
    root: Path,
    *,
    complete: bool,
    checkpoint_step: int | None = None,
    log_steps: int | None = None,
    reset_elapsed_at: int | None = None,
) -> dict:
    root.mkdir(parents=True)
    checkpoint_step = TOTAL_STEPS if checkpoint_step is None else checkpoint_step
    log_steps = checkpoint_step if log_steps is None else log_steps
    rectification = _rectification_receipt(root.parent)
    config = _config(rectification, root)
    model = _refiner()
    assert model.trainable_parameter_count == 69_905
    optimizer, scheduler = _optimizer_and_scheduler_state(model, step=checkpoint_step)
    base_path, base_sha = _base_checkpoint(root.parent)
    git_hash, source_bundle = _source_bundle(Path(__file__).parents[1])
    runtime = _runtime_receipt()
    base_checkpoint = {
        "path": str(base_path),
        "sha256": base_sha,
        "step": 15_000,
    }
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
    micro_steps = checkpoint_step * 4
    epoch, offset = divmod(micro_steps, batches_per_epoch)
    completion = {
        "actual_step": checkpoint_step,
        "configured_steps": TOTAL_STEPS,
        "execution_complete": checkpoint_step == TOTAL_STEPS,
        "canonical_schedule": True,
        "base_complete": True,
        "cuda_bf16_eligible": True,
        "strict_determinism_eligible": True,
        "formal_training_complete": checkpoint_step == TOTAL_STEPS,
    }
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
            "grad_accumulation": 4,
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
        "loss": _loss(),
        "elapsed_seconds": float(checkpoint_step),
        "completion": completion,
    }
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
                "loss": _loss(),
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
            "formal_training_complete": True,
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
    assert report["resume"]["detected"] is True
    assert report["resume"]["final_segment_start_step"] == 2_500
    assert {path: path.stat().st_mtime_ns for path in mtimes} == mtimes


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
