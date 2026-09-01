#!/usr/bin/env python3
"""Read-only audit for a bounded Spring Stage-C metrics report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.spring_stage_c import (  # noqa: E402
    SpringStageCError,
    audit_spring_screening_report,
)
from data.cache_dataset import sha256_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a bounded Spring Stage-C screening report."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-adapter", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    metrics = args.metrics.expanduser().resolve()
    try:
        report: Any = json.loads(metrics.read_text(encoding="utf-8"))
        audit = audit_spring_screening_report(
            report,
            metrics_path=metrics,
            validation_manifest=args.validation_manifest,
            checkpoint=args.checkpoint,
            train_adapter_path=args.train_adapter,
        )
    except (OSError, json.JSONDecodeError, SpringStageCError, ValueError) as exc:
        payload = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    audit["auditor"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
