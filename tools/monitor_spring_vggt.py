#!/usr/bin/env python3
"""Wait for CUDA capacity and launch pending Spring arm screening.

The monitor is intentionally fail-safe: it never stops or reconfigures other
processes. In the default CUDA mode it launches the requested runner only
after every visible GPU has the configured free-memory floor. An explicit CPU
mode launches immediately as a bounded-screening fallback. Resource
observations and the exact launch command are appended as JSON lines for
auditability.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def snapshot() -> list[dict[str, int]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    result: list[dict[str, int]] = []
    for line in raw.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            result.append(
                {
                    "index": int(fields[0]),
                    "free_mib": int(fields[1]),
                    "total_mib": int(fields[2]),
                }
            )
        except ValueError:
            continue
    return result


def append_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": now(), **event}, sort_keys=True) + "\n")
        handle.flush()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--rectification-audit",
        type=Path,
        required=True,
        help="Spring-specific pixel rectification receipt used by S6",
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help=(
            "runner/cache device; CUDA waits for the free-memory floor, while "
            "CPU launches immediately as an explicit bounded-screening fallback"
        ),
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=7,
        help=(
            "bounded records per split passed to the Spring runner; keep this "
            "at least 7 for complete causal Stage-C screening coverage"
        ),
    )
    parser.add_argument(
        "--arm",
        dest="arms",
        action="append",
        default=None,
        help=(
            "Spring arm to launch after CUDA capacity is available; repeat for "
            "multiple arms (default: S4 S5 S6)"
        ),
    )
    parser.add_argument(
        "--final-report-json",
        type=Path,
        help="optional consolidated report path written after the runner exits",
    )
    parser.add_argument(
        "--final-report-md",
        type=Path,
        help="optional consolidated Markdown report path written after the runner exits",
    )
    parser.add_argument("--once", action="store_true", help="record one snapshot and exit")
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or args.min_free_mib <= 0
        or args.steps <= 0
        or args.limit <= 0
    ):
        raise ValueError("poll-seconds, min-free-mib, steps, and limit must be positive")

    arms = [str(value).strip().upper() for value in (args.arms or ["S4", "S5", "S6"])]
    if not arms or any(value not in {f"S{index}" for index in range(7)} for value in arms):
        raise ValueError("--arm values must be one or more of S0..S6")

    project = args.project_root.expanduser().resolve()
    train_manifest = args.train_manifest.expanduser().resolve()
    validation_manifest = args.validation_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    rectification_audit = args.rectification_audit.expanduser().resolve()
    log_path = args.log.expanduser().resolve()
    report_suffix = output_root.name
    if report_suffix.startswith("spring_seed42_"):
        report_suffix = report_suffix.removeprefix("spring_seed42_")
    final_report_json = (
        args.final_report_json.expanduser().resolve()
        if args.final_report_json is not None
        else project / "reports" / f"spring_seed42_screening_{report_suffix}.json"
    )
    final_report_md = (
        args.final_report_md.expanduser().resolve()
        if args.final_report_md is not None
        else project / "reports" / f"spring_seed42_screening_{report_suffix}.md"
    )
    command = [
        str(args.python_executable),
        str(project / "tools" / "run_spring_arms.py"),
        "--project-root",
        str(project),
        "--train-manifest",
        str(train_manifest),
        "--validation-manifest",
        str(validation_manifest),
        "--output-root",
        str(output_root),
        "--cache-root",
        str(cache_root),
        "--device",
        str(args.device),
        "--rectification-audit",
        str(rectification_audit),
        "--steps",
        str(args.steps),
        "--limit",
        str(args.limit),
        *sum((["--arm", value] for value in arms), []),
        "--keep-going",
    ]

    while True:
        gpus = snapshot()
        append_log(
            log_path,
            {
                "event": "gpu_snapshot",
                "gpus": gpus,
                "min_free_mib": args.min_free_mib,
            },
        )
        if args.once:
            return 0
        ready = (
            True
            if args.device == "cpu"
            else bool(gpus) and all(item["free_mib"] >= args.min_free_mib for item in gpus)
        )
        if ready:
            append_log(
                log_path,
                {
                    "event": "resource_policy",
                    "device": str(args.device),
                    "cpu_fallback": bool(args.device == "cpu"),
                },
            )
            append_log(log_path, {"event": "launch", "command": command})
            output_root.mkdir(parents=True, exist_ok=True)
            with (log_path.parent / "spring_s4_s5_runner.log").open(
                "a", encoding="utf-8"
            ) as handle:
                process = subprocess.run(
                    command,
                    cwd=str(project),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            append_log(
                log_path,
                {"event": "runner_exit", "returncode": int(process.returncode)},
            )
            # Compose the final contract automatically when the runner has
            # produced the S0 artifact. The report utility preserves null for
            # unavailable Spring-specific fields and records any blocked
            # VGGT arms from the runner receipt.
            s0_metrics = output_root / "arms" / "S0" / "eval" / "metrics.json"
            blocked_receipt = output_root / "spring_seed42_summary.json"
            if s0_metrics.is_file() and blocked_receipt.is_file():
                report_command = [
                    str(args.python_executable),
                    str(project / "tools" / "compose_spring_screening_report.py"),
                    "--output-json",
                    str(final_report_json),
                    "--output-md",
                    str(final_report_md),
                    "--s0",
                    str(s0_metrics),
                    "--arm-root",
                    str(output_root / "arms"),
                    "--blocked",
                    str(blocked_receipt),
                    "--manifest",
                    str(validation_manifest),
                ]
                report_log = log_path.parent / "compose_spring_screening_report.log"
                with report_log.open("a", encoding="utf-8") as report_handle:
                    report_process = subprocess.run(
                        report_command,
                        cwd=str(project),
                        stdin=subprocess.DEVNULL,
                        stdout=report_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                        text=True,
                    )
                append_log(
                    log_path,
                    {
                        "event": "report_compose",
                        "command": report_command,
                        "returncode": int(report_process.returncode),
                        "output_json": str(final_report_json),
                        "output_md": str(final_report_md),
                    },
                )
                if process.returncode == 0 and report_process.returncode != 0:
                    return int(report_process.returncode)
            return int(process.returncode)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
