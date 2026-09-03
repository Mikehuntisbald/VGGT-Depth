from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.cache_dataset import CacheIdentity, save_cache_record, sha256_file
from data.endpoint_selection import write_endpoint_index
from data.manifest import ManifestRecord, write_manifest
from tools import eval_spring_baseline as baseline


def _manifest(tmp_path: Path, count: int = 2) -> Path:
    records = [
        ManifestRecord(
            sequence_id="0001",
            frame_id=index + 1,
            timestamp=float(index),
            left_path=f"left_{index + 1}.png",
            right_path=f"right_{index + 1}.png",
            K=((100.0, 0.0, 4.0), (0.0, 100.0, 4.0), (0.0, 0.0, 1.0)),
            baseline_m=0.065,
            gt_disparity_path=f"spring/train/0001/disp1_left/disp1_left_{index + 1:04d}.dsp5",
            extras={"dataset": "spring"},
        )
        for index in range(count)
    ]
    path = tmp_path / "validation.jsonl"
    write_manifest(path, records)
    return path


def _cache(
    tmp_path: Path,
    manifest: Path,
    *,
    mode: str,
    frame_id: int,
    disparity: torch.Tensor,
) -> Path:
    spec = baseline.BASELINE_MODES[mode]
    root = tmp_path / f"cache_{mode}"
    identity = CacheIdentity(
        component=spec.cache_component,
        upstream_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        torch_version="test",
        cuda_version=None,
        config_sha256="c" * 64,
    )
    config = {
        "role": "observation",
        "scale": spec.scale,
        "resolution_mode": spec.name,
        "max_disp_hr_equivalent_px": 384,
    }
    save_cache_record(
        root / "0001" / f"{frame_id}.pt",
        tensors={
            "observation_disparity_hr_px": disparity,
            "observation_trusted_mask": torch.ones_like(disparity, dtype=torch.bool),
        },
        metadata={"config": config},
        identity=identity,
    )
    receipt = {
        "schema_version": 1,
        "identity": identity.to_dict(),
        "config": config,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "selected_records": 1,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_baseline_modes_enforce_expected_grid_and_reconstruction() -> None:
    half = torch.full((1, 4, 4), 2.0)
    pred, trusted = baseline._observation_on_image_grid(
        half,
        torch.ones_like(half, dtype=torch.bool),
        target_hw=(8, 8),
        mode=baseline.BASELINE_MODES["half"],
    )
    assert pred.shape == (8, 8)
    assert trusted.shape == (8, 8)
    assert np.all(pred == 2.0)

    full = torch.full((1, 8, 8), 3.0)
    pred, _ = baseline._observation_on_image_grid(
        full,
        torch.ones_like(full, dtype=torch.bool),
        target_hw=(8, 8),
        mode=baseline.BASELINE_MODES["full"],
    )
    assert np.all(pred == 3.0)
    with pytest.raises(ValueError, match="full-resolution cache grid"):
        baseline._observation_on_image_grid(
            half,
            torch.ones_like(half, dtype=torch.bool),
            target_hw=(8, 8),
            mode=baseline.BASELINE_MODES["full"],
        )


def test_fixed_crop_uses_xy_origin_and_hw_size() -> None:
    crop = baseline._crop_xywh(
        crop_mode="fixed",
        crop_origin_xy=(2, 1),
        crop_size_hw=(3, 4),
        image_hw=(8, 10),
    )
    assert crop == (2, 1, 4, 3)
    image = np.arange(80).reshape(8, 10)
    assert np.array_equal(baseline._crop_array(image, crop), image[1:4, 2:6])


def test_global_metric_aggregation_uses_pixel_denominators() -> None:
    rows = [
        {"overall_epe": 1.0, "valid_count": 1, "image_pixel_count": 1},
        {"overall_epe": 3.0, "valid_count": 3, "image_pixel_count": 3},
    ]
    aggregate = baseline._aggregate(rows)
    assert aggregate["overall_epe"] == 2.5
    assert aggregate["numerators"]["overall_epe"] == 10.0
    assert aggregate["denominators"]["overall_epe"] == 4


@pytest.mark.parametrize(
    ("mode", "arm", "reconstruction", "cache_hw"),
    (
        ("full", "F0", "identity", (8, 8)),
        ("half", "F1", "bilinear_align_corners_false", (4, 4)),
    ),
)
def test_evaluate_binds_endpoint_crop_cache_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    arm: str,
    reconstruction: str,
    cache_hw: tuple[int, int],
) -> None:
    manifest = _manifest(tmp_path)
    endpoint = write_endpoint_index(
        tmp_path / "common.json",
        manifest_path=manifest,
        manifest_indices=[1],
    )
    cache = _cache(
        tmp_path,
        manifest,
        mode=mode,
        frame_id=2,
        disparity=torch.full((1, 1, *cache_hw), 2.0),
    )
    monkeypatch.setattr(
        baseline,
        "load_spring_disparity",
        lambda *_args, **_kwargs: np.full((8, 8), 2.0, dtype=np.float32),
    )

    report = baseline.evaluate(
        manifest_path=manifest,
        observation_root=cache,
        output_dir=tmp_path / "eval",
        mode=mode,
        endpoint_index_list=endpoint.path,
        crop_mode="fixed",
        crop_origin=(2, 2),
        crop_size=(4, 4),
    )

    assert report["arm"] == arm
    assert report["status"] == "SCREENING_ONLY"
    assert report["target"] == {
        "type": "spring_v2_disp1_ground_truth",
        "component": "spring-ground-truth",
        "paper_gt": True,
        "synthetic_ground_truth": True,
        "paper_accuracy": False,
        "disparity_unit": "full_hd_pixels",
        "resolution": "image",
    }
    assert report["evaluator"] == {
        "git_hash": baseline.repository_git_hash(baseline.PROJECT_ROOT),
        "eval_py_sha256": sha256_file(Path(baseline.__file__).resolve()),
        "evaluation_module_sha256": sha256_file(
            baseline.SRC_ROOT / "metrics/spring_arms.py"
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    assert report["device"] == "cpu"
    assert report["elapsed_seconds"] > 0.0
    assert any("does not claim official" in note for note in report["notes"])
    assert report["resolution_contract"]["reconstruction"] == reconstruction
    assert report["selection"]["records"] == 1
    assert report["selection"]["first_manifest_index"] == 1
    assert report["selection"]["crop_origin_xy"] == [2, 2]
    assert report["selection"]["crop_size_hw"] == [4, 4]
    assert (
        report["selection"]["endpoint_index_list"]["endpoint_id_sha256"]
        == endpoint.entries_sha256
    )
    assert report["metrics"]["overall_epe"] == 0.0
    assert report["metrics"]["valid_count"] == 16
    assert report["per_record"][0]["manifest_index"] == 1


def test_evaluator_rejects_full_mode_with_half_cache_receipt(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, count=1)
    cache = _cache(
        tmp_path,
        manifest,
        mode="half",
        frame_id=1,
        disparity=torch.ones((1, 1, 4, 4)),
    )
    with pytest.raises(ValueError, match="component does not match"):
        baseline._load_cache_lineage(
            cache,
            manifest_path=manifest,
            mode=baseline.BASELINE_MODES["full"],
        )
