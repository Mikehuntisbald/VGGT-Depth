from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from tools.audit_training_run import (
    AuditExpectations,
    TrainingAuditError,
    audit_training_run,
    main,
)
from utils.checkpoint import config_fingerprint, save_training_checkpoint


GIT_HASH = "a" * 40
LOSS_TERMS = (
    "disparity",
    "epipolar",
    "gate_regularizer",
    "gradient",
    "measurement",
    "temporal",
    "total",
    "uncertainty_nll",
)


def _multiplier(update_index: int, *, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and update_index < warmup_steps:
        return float(update_index + 1) / float(warmup_steps)
    decay_updates = total_steps - warmup_steps
    if decay_updates <= 1:
        return 1.0
    progress = min(max(update_index - warmup_steps, 0), decay_updates - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress / (decay_updates - 1)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_run(root: Path, *, complete: bool, gradient_norm: float = 2.5) -> dict:
    root.mkdir(parents=True)
    total_steps = 4
    checkpoint_step = total_steps if complete else 2
    log_steps = total_steps if complete else 3
    base_lr = 2.0e-4
    warmup = 1
    config = {
        "data": {"sequence_length": 1},
        "train": {
            "stage": "spatial",
            "steps_spatial": total_steps,
            "steps": 8,
            "learning_rate": base_lr,
            "warmup_steps": warmup,
            "checkpoint_interval": 2,
            "gradient_clip": 1.0,
        },
    }
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda index: _multiplier(
            index, total_steps=total_steps, warmup_steps=warmup
        ),
    )
    learning_rates: list[float] = []
    for _ in range(checkpoint_step):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_training_checkpoint(
        root / "latest.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=checkpoint_step,
        config=config,
        git_hash=GIT_HASH,
        parameter_count=parameter_count,
    )
    if complete:
        save_training_checkpoint(
            root / "final.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            completed_step=checkpoint_step,
            config=config,
            git_hash=GIT_HASH,
            parameter_count=parameter_count,
        )
    records = []
    for step in range(1, log_steps + 1):
        learning_rate = base_lr * _multiplier(
            step, total_steps=total_steps, warmup_steps=warmup
        )
        records.append(
            {
                "step": step,
                "stage": "spatial",
                "learning_rate": learning_rate,
                "gradient_norm": gradient_norm if step == 2 else 0.5,
                "elapsed_seconds": float(step),
                "loss": {name: 0.1 * step for name in LOSS_TERMS},
            }
        )
    (root / "train.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    if complete:
        fingerprint = hashlib.sha256(config_fingerprint(config).encode("utf-8")).hexdigest()
        summary = {
            "stage": "spatial",
            "status": "TRAINING_COMPLETE",
            "steps": total_steps,
            "run_steps": total_steps,
            "elapsed_seconds": 5.0,
            "steps_per_second": total_steps / 5.0,
            "device": "cpu",
            "device_name": None,
            "torch_version": str(torch.__version__),
            "cuda_version": None,
            "git_hash": GIT_HASH,
            "config_fingerprint": fingerprint,
            "final_checkpoint": {
                "path": str((root / "final.pt").resolve()),
                "sha256": _sha256(root / "final.pt"),
            },
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
        }
        (root / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return config


def test_completed_run_passes_full_audit_and_reports_preclip_gradient(tmp_path: Path) -> None:
    run = tmp_path / "run"
    config = _build_run(run, complete=True)
    fingerprint = hashlib.sha256(config_fingerprint(config).encode("utf-8")).hexdigest()
    mtimes = {path.name: path.stat().st_mtime_ns for path in run.iterdir()}

    report = audit_training_run(
        run,
        expectations=AuditExpectations(
            stage="spatial",
            steps=4,
            git_hash=GIT_HASH,
            config_fingerprint=fingerprint,
            checkpoint_sha256=_sha256(run / "final.pt"),
        ),
        rolling_window=2,
    )

    assert report["status"] == "PASS"
    assert report["training_status"] == "TRAINING_COMPLETE"
    assert report["logged_steps"] == 4
    assert report["validation"]["learning_rate_schedule_exact"] is True
    gradient = report["statistics"]["gradient_norm_pre_clip"]
    assert gradient["above_configured_clip_count"] == 1
    assert gradient["rolling"]["window_size"] == 2
    assert {path.name: path.stat().st_mtime_ns for path in run.iterdir()} == mtimes


def test_missing_completion_receipt_is_explicitly_in_progress(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _build_run(run, complete=False)

    report = audit_training_run(run)

    assert report["status"] == "IN_PROGRESS"
    assert report["training_status"] == "IN_PROGRESS"
    assert report["logged_steps"] == 3
    assert report["latest_checkpoint_step"] == 2
    assert report["files"]["final_checkpoint"] is None
    assert report["validation"]["completion_receipt_valid"] is False


def test_discontinuous_steps_and_nonfinite_values_are_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _build_run(run, complete=True)
    records = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    records[2]["step"] = 4
    (run / "train.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(TrainingAuditError, match="not continuous"):
        audit_training_run(run)

    _build_run(tmp_path / "nonfinite", complete=True)
    nonfinite_path = tmp_path / "nonfinite" / "train.jsonl"
    nonfinite = nonfinite_path.read_text().replace('"gradient_norm": 0.5', '"gradient_norm": NaN', 1)
    nonfinite_path.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(TrainingAuditError, match="non-finite"):
        audit_training_run(tmp_path / "nonfinite")


def test_false_completion_and_summary_sha_mismatch_are_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _build_run(run, complete=True)
    summary_path = run / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["final_checkpoint"]["sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(TrainingAuditError, match="SHA mismatch"):
        audit_training_run(run)


def test_cli_refuses_to_write_audit_inside_training_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _build_run(run, complete=True)

    exit_code = main(
        [
            "--output-dir",
            str(run),
            "--json-out",
            str(run / "audit.json"),
        ]
    )

    assert exit_code == 2
    assert not (run / "audit.json").exists()
