#!/usr/bin/env python3
"""Recoverably align ``train.jsonl`` to an atomic training checkpoint.

This tool exists for the crash window where the append-only log can be ahead
of ``latest.pt`` or end in a partial JSON record.  It never changes a model
checkpoint.  Mutation requires both ``--apply`` and an explicit confirmation
that the trainer has stopped, and the original log is preserved byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    return step


def inspect_log_against_checkpoint(
    log_bytes: bytes,
    *,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Return the exact safe prefix and reconciliation diagnostics.

    The first ``checkpoint_step`` records must be strict, newline-terminated
    JSON objects with one-based continuous ``step`` values.  A log shorter than
    the checkpoint cannot be repaired from the available information and is
    rejected rather than padded or fabricated.
    """

    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise TypeError("checkpoint_step must be an integer")
    if checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative")

    newline_ends = [index + 1 for index, byte in enumerate(log_bytes) if byte == 10]
    complete_records: list[Mapping[str, Any]] = []
    start = 0
    for line_number, end in enumerate(newline_ends, start=1):
        raw_line = log_bytes[start:end]
        start = end
        try:
            decoded = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"train log has malformed complete JSON at line {line_number}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ValueError(f"train log line {line_number} is not a JSON object")
        complete_records.append(decoded)

    if len(complete_records) < checkpoint_step:
        raise ValueError(
            "train log is behind the checkpoint and cannot be reconstructed: "
            f"records={len(complete_records)}, checkpoint_step={checkpoint_step}"
        )
    for index, record in enumerate(complete_records[:checkpoint_step], start=1):
        value = record.get("step")
        if value != index:
            raise ValueError(
                "train log is not continuous through the checkpoint: "
                f"line={index}, recorded_step={value!r}"
            )

    prefix_end = 0 if checkpoint_step == 0 else newline_ends[checkpoint_step - 1]
    safe_prefix = log_bytes[:prefix_end]
    suffix = log_bytes[prefix_end:]
    partial_suffix = log_bytes[start:]
    extra_complete = len(complete_records) - checkpoint_step
    return {
        "checkpoint_step": checkpoint_step,
        "complete_records": len(complete_records),
        "extra_complete_records": extra_complete,
        "partial_suffix_bytes": len(partial_suffix),
        "discard_bytes": len(suffix),
        "aligned": len(suffix) == 0,
        "safe_prefix": safe_prefix,
        "original_sha256": _sha256_bytes(log_bytes),
        "reconciled_sha256": _sha256_bytes(safe_prefix),
    }


def _atomic_write(path: Path, value: bytes, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def reconcile_training_log(
    output_dir: str | Path,
    *,
    checkpoint_name: str = "latest.pt",
    log_name: str = "train.jsonl",
    apply: bool = False,
    confirm_training_stopped: bool = False,
    backup_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect or recoverably reconcile a training log to its checkpoint."""

    root = Path(output_dir).expanduser().resolve()
    checkpoint = root / checkpoint_name
    log = root / log_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    if not log.is_file():
        raise FileNotFoundError(f"training log is missing: {log}")
    if apply and not confirm_training_stopped:
        raise RuntimeError("--apply requires --confirm-training-stopped")

    checkpoint_hash = _sha256_file(checkpoint)
    checkpoint_step = _checkpoint_step(checkpoint)
    stat_before = log.stat()
    original = log.read_bytes()
    inspection = inspect_log_against_checkpoint(
        original, checkpoint_step=checkpoint_step
    )
    safe_prefix = inspection.pop("safe_prefix")
    assert isinstance(safe_prefix, bytes)

    report: dict[str, Any] = {
        "schema_version": 1,
        "component": "training-log-checkpoint-reconciliation",
        "mode": "APPLY" if apply else "DRY_RUN",
        "status": "ALREADY_ALIGNED" if inspection["aligned"] else "NEEDS_RECONCILIATION",
        "output_dir": str(root),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            "step": checkpoint_step,
        },
        "log": {
            "path": str(log),
            **inspection,
        },
        "backup": None,
        "mutation_performed": False,
    }
    if not apply or inspection["aligned"]:
        return report

    current_stat = log.stat()
    if (
        current_stat.st_size != stat_before.st_size
        or current_stat.st_mtime_ns != stat_before.st_mtime_ns
        or log.read_bytes() != original
    ):
        raise RuntimeError("training log changed during inspection; refusing mutation")
    if _sha256_file(checkpoint) != checkpoint_hash:
        raise RuntimeError("checkpoint changed during inspection; refusing mutation")

    if backup_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = root / f"{log.stem}.pre_reconcile_step{checkpoint_step}.{stamp}{log.suffix}"
    else:
        backup = Path(backup_path).expanduser().resolve()
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(log, backup)
    with backup.open("rb") as handle:
        os.fsync(handle.fileno())
    if _sha256_file(backup) != inspection["original_sha256"]:
        raise RuntimeError("backup hash differs from the original log")

    _atomic_write(log, safe_prefix, mode=stat_before.st_mode & 0o777)
    if _sha256_file(log) != inspection["reconciled_sha256"]:
        raise RuntimeError("reconciled log hash verification failed")
    report["status"] = "RECONCILED"
    report["backup"] = {
        "path": str(backup),
        "sha256": inspection["original_sha256"],
    }
    report["mutation_performed"] = True
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or recoverably align train.jsonl to latest.pt."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="latest.pt")
    parser.add_argument("--log-name", default="train.jsonl")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-training-stopped", action="store_true")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = reconcile_training_log(
        args.output_dir,
        checkpoint_name=args.checkpoint_name,
        log_name=args.log_name,
        apply=args.apply,
        confirm_training_stopped=args.confirm_training_stopped,
        backup_path=args.backup,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out is not None:
        destination = args.json_out.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(destination, encoded.encode("utf-8"), mode=0o664)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
