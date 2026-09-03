"""Keep legacy bare test-helper imports bound to this repository."""

from __future__ import annotations

import sys
from pathlib import Path


_TEST_ROOT = Path(__file__).resolve().parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))
