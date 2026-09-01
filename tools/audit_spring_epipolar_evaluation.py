#!/usr/bin/env python3
"""Compatibility name for the Spring Stage-C screening audit CLI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_spring_epipolar import build_parser, main, run

__all__ = ["build_parser", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
