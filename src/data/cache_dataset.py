"""Versioned on-disk cache records shared by isolated backbone environments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


CACHE_SCHEMA_VERSION = 1


class CacheMismatchError(RuntimeError):
    """Raised when an existing cache does not match the requested identity."""


@dataclass(frozen=True)
class CacheIdentity:
    """Fields that must match before a cache entry may be reused."""

    component: str
    upstream_commit: str
    checkpoint_sha256: str
    torch_version: str
    cuda_version: str | None
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable identity mapping."""

        # ``torch.__version__`` is a ``TorchVersion`` (a ``str`` subclass) in
        # recent PyTorch releases.  ``torch.save`` preserves that subclass,
        # which is intentionally rejected by ``torch.load(weights_only=True)``.
        # Normalize every string-like provenance field to a plain built-in
        # string so cache records remain safely loadable across environments.
        values = asdict(self)
        for key in (
            "component",
            "upstream_commit",
            "checkpoint_sha256",
            "torch_version",
            "config_sha256",
        ):
            values[key] = str(values[key])
        if values["cuda_version"] is not None:
            values["cuda_version"] = str(values["cuda_version"])
        return values


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash a mapping after deterministic compact JSON serialization."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache_identity(
    actual: Mapping[str, Any],
    expected: CacheIdentity,
) -> None:
    """Reject any identity mismatch instead of silently reusing stale data."""

    expected_mapping = expected.to_dict()
    differences = {
        key: {"expected": expected_value, "actual": actual.get(key)}
        for key, expected_value in expected_mapping.items()
        if actual.get(key) != expected_value
    }
    if differences:
        raise CacheMismatchError(
            "cache identity mismatch: "
            + json.dumps(differences, sort_keys=True, separators=(",", ":"))
        )


def save_cache_record(
    path: Path,
    *,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    identity: CacheIdentity,
) -> None:
    """Atomically save CPU tensors and provenance metadata to ``path``.

    Tensors are detached and moved to CPU. Callers choose their cache dtype
    before invoking this function; integer and boolean tensors are preserved.
    """

    normalized_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"cache tensor {name!r} is not a torch.Tensor")
        normalized_tensors[name] = tensor.detach().cpu().contiguous()

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity": identity.to_dict(),
        "metadata": dict(metadata),
        "tensors": normalized_tensors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_cache_record(
    path: Path,
    *,
    expected_identity: CacheIdentity | None = None,
) -> dict[str, Any]:
    """Load a cache record and optionally enforce its complete identity."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise CacheMismatchError("cache payload is not a mapping")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheMismatchError(
            f"cache schema mismatch: expected {CACHE_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("identity"), dict):
        raise CacheMismatchError("cache identity is missing or malformed")
    if not isinstance(payload.get("metadata"), dict):
        raise CacheMismatchError("cache metadata is missing or malformed")
    if not isinstance(payload.get("tensors"), dict):
        raise CacheMismatchError("cache tensors are missing or malformed")
    if expected_identity is not None:
        validate_cache_identity(payload["identity"], expected_identity)
    return payload
