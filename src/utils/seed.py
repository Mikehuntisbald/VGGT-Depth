"""Deterministic random-seed helpers for training and DataLoader workers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


STRICT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_strict_determinism() -> None:
    """Enable the fail-closed PyTorch/CUDA determinism contract.

    ``CUBLAS_WORKSPACE_CONFIG`` is installed before any CUDA seeding call so a
    newly started producer process cannot initialize cuBLAS under an
    unrecorded workspace policy.  Unsupported nondeterministic operations are
    errors, never warnings.
    """

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = STRICT_CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deterministic_runtime_state() -> dict[str, bool | str | None]:
    """Return the actual process-global determinism settings for a receipt."""

    return {
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def strict_determinism_enabled() -> bool:
    """Whether every required setting matches the formal strict contract."""

    state = deterministic_runtime_state()
    return bool(
        state["deterministic_algorithms_enabled"] is True
        and state["deterministic_algorithms_warn_only"] is False
        and state["cublas_workspace_config"] == STRICT_CUBLAS_WORKSPACE_CONFIG
        and state["cudnn_deterministic"] is True
        and state["cudnn_benchmark"] is False
    )


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch in the current process.

    Args:
        seed: Non-negative integer experiment seed.
        deterministic: Request deterministic PyTorch algorithms.  CUDA BLAS is
            configured before kernels are launched so supported matrix
            multiplications also follow the deterministic contract.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if deterministic:
        configure_strict_determinism()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_data_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's per-worker initial seed."""

    if isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0:
        raise ValueError("worker_id must be a non-negative integer")
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = [
    "STRICT_CUBLAS_WORKSPACE_CONFIG",
    "configure_strict_determinism",
    "deterministic_runtime_state",
    "seed_data_worker",
    "seed_everything",
    "strict_determinism_enabled",
]
