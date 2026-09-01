from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.reconcile_training_log import (
    inspect_log_against_checkpoint,
    reconcile_training_log,
)


def _line(step: int) -> bytes:
    return (json.dumps({"loss": 1.0 / step, "step": step}) + "\n").encode()


def _run(tmp_path: Path, *, checkpoint_step: int, log: bytes) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    torch.save({"step": checkpoint_step}, root / "latest.pt")
    (root / "train.jsonl").write_bytes(log)
    return root


def test_inspection_identifies_complete_and_partial_suffix() -> None:
    value = _line(1) + _line(2) + _line(3) + b'{"step":4'
    report = inspect_log_against_checkpoint(value, checkpoint_step=2)
    assert report["aligned"] is False
    assert report["complete_records"] == 3
    assert report["extra_complete_records"] == 1
    assert report["partial_suffix_bytes"] == len(b'{"step":4')
    assert report["safe_prefix"] == _line(1) + _line(2)


def test_apply_preserves_original_and_reconciles_exactly(tmp_path: Path) -> None:
    original = _line(1) + _line(2) + _line(3) + b'{"step":4'
    root = _run(tmp_path, checkpoint_step=2, log=original)
    backup = tmp_path / "archive" / "interrupted.jsonl"

    dry = reconcile_training_log(root)
    assert dry["status"] == "NEEDS_RECONCILIATION"
    assert dry["mutation_performed"] is False
    assert (root / "train.jsonl").read_bytes() == original

    applied = reconcile_training_log(
        root,
        apply=True,
        confirm_training_stopped=True,
        backup_path=backup,
    )
    assert applied["status"] == "RECONCILED"
    assert applied["mutation_performed"] is True
    assert backup.read_bytes() == original
    assert (root / "train.jsonl").read_bytes() == _line(1) + _line(2)


def test_apply_requires_explicit_stopped_training_confirmation(tmp_path: Path) -> None:
    root = _run(tmp_path, checkpoint_step=1, log=_line(1) + _line(2))
    with pytest.raises(RuntimeError, match="confirm-training-stopped"):
        reconcile_training_log(root, apply=True)


def test_log_behind_checkpoint_is_rejected(tmp_path: Path) -> None:
    root = _run(tmp_path, checkpoint_step=2, log=_line(1))
    with pytest.raises(ValueError, match="behind the checkpoint"):
        reconcile_training_log(root)


def test_malformed_or_noncontinuous_committed_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="malformed complete JSON"):
        inspect_log_against_checkpoint(_line(1) + b"not-json\n", checkpoint_step=2)
    with pytest.raises(ValueError, match="not continuous"):
        inspect_log_against_checkpoint(_line(1) + _line(3), checkpoint_step=2)
