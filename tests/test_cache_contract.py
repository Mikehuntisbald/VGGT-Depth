from pathlib import Path

import pytest
import torch

from data.cache_dataset import (
    CACHE_SCHEMA_VERSION,
    CacheIdentity,
    CacheMismatchError,
    canonical_json_sha256,
    load_cache_record,
    save_cache_record,
)


def identity(config_hash: str = "config-a") -> CacheIdentity:
    return CacheIdentity(
        component="ffs-observation",
        upstream_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        torch_version="2.10.0+cu128",
        cuda_version="12.8",
        config_sha256=config_hash,
    )


def test_canonical_config_hash_is_order_independent() -> None:
    assert canonical_json_sha256({"scale": 2, "iters": 4}) == canonical_json_sha256(
        {"iters": 4, "scale": 2}
    )


def test_cache_round_trip_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "seq" / "frame.pt"
    save_cache_record(
        path,
        tensors={
            "disparity_hr_px": torch.arange(6, dtype=torch.float16).reshape(2, 3),
            "valid_mask": torch.tensor([[True, False, True]]),
        },
        metadata={"frame_id": 12, "units": {"disparity_hr_px": "HR pixels"}},
        identity=identity(),
    )
    payload = load_cache_record(path, expected_identity=identity())
    assert payload["schema_version"] == CACHE_SCHEMA_VERSION
    assert payload["metadata"]["frame_id"] == 12
    assert payload["tensors"]["disparity_hr_px"].dtype == torch.float16
    assert payload["tensors"]["valid_mask"].dtype == torch.bool


def test_cache_identity_normalizes_torch_version_for_safe_loading(tmp_path: Path) -> None:
    path = tmp_path / "torch-version.pt"
    torch_identity = CacheIdentity(
        component="ffs-observation",
        upstream_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        config_sha256="config-a",
    )
    save_cache_record(
        path,
        tensors={"disparity_hr_px": torch.ones(1)},
        metadata={},
        identity=torch_identity,
    )

    payload = load_cache_record(path, expected_identity=torch_identity)
    assert type(payload["identity"]["torch_version"]) is str


def test_cache_identity_mismatch_is_not_silent(tmp_path: Path) -> None:
    path = tmp_path / "frame.pt"
    save_cache_record(
        path,
        tensors={"disparity_hr_px": torch.ones(1)},
        metadata={},
        identity=identity(),
    )
    with pytest.raises(CacheMismatchError, match="config_sha256"):
        load_cache_record(path, expected_identity=identity("config-b"))
