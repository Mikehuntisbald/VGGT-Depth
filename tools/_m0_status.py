"""Shared M0 receipt helpers with no third-party dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


class Status(str, Enum):
    PASS = "PASS"
    PASS_WITH_FALLBACK = "PASS_WITH_FALLBACK"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


EXIT_CODE = {
    Status.PASS: 0,
    Status.PASS_WITH_FALLBACK: 0,
    Status.FAIL: 1,
    Status.BLOCKED: 3,
    Status.NOT_RUN: 3,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def base_receipt(kind: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "created_at_utc": utc_now(),
        "hostname": platform.node(),
        "status": Status.NOT_RUN,
        "checks": [],
    }


def add_check(
    receipt: dict[str, Any],
    name: str,
    status: Status,
    detail: Any = None,
) -> None:
    check: dict[str, Any] = {"name": name, "status": status.value}
    if detail is not None:
        check["detail"] = detail
    receipt["checks"].append(check)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def command_output(command: Sequence[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "returncode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def git_snapshot(repo: Path) -> dict[str, Any]:
    head = command_output(["git", "rev-parse", "HEAD"], cwd=repo)
    remote = command_output(["git", "remote", "get-url", "origin"], cwd=repo)
    status = command_output(["git", "status", "--porcelain"], cwd=repo)
    return {
        "path": str(repo.resolve()),
        "head": head.get("stdout"),
        "remote": remote.get("stdout"),
        "dirty": bool(status.get("stdout")),
        "status_porcelain": status.get("stdout", "").splitlines(),
    }


def finalize(receipt: dict[str, Any], status: Status, output: Path) -> int:
    receipt["status"] = status.value
    atomic_write_json(output, receipt)
    print(f"{receipt['kind']}: {status.value}")
    print(f"receipt: {output.resolve()}")
    return EXIT_CODE[status]
