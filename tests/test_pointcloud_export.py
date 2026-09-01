from __future__ import annotations

import pytest
import torch

from metrics.pointcloud import export_colored_point_cloud_ply


def _intrinsics() -> torch.Tensor:
    return torch.tensor(
        [[4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    )


def test_export_colored_point_cloud_ply_writes_calibrated_rgb_vertices(
    tmp_path,
) -> None:
    output = tmp_path / "cloud.ply"
    result = export_colored_point_cloud_ply(
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[[10, 20, 30], [40, 50, 60]]], dtype=torch.uint8),
        _intrinsics(),
        baseline_m=0.5,
        output_path=output,
    )

    assert result.path == output
    assert result.point_count == 2
    assert output.read_text(encoding="ascii").splitlines() == [
        "ply",
        "format ascii 1.0",
        "comment camera_frame left; coordinates_m",
        "element vertex 2",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "0 0 1 10 20 30",
        "0.5 0 2 40 50 60",
    ]


def test_export_ply_masks_invalid_disparity_confidence_and_depth(tmp_path) -> None:
    output = tmp_path / "filtered.ply"
    disparity = torch.tensor([[2.0, 0.0, float("nan")], [1.0, 4.0, 2.0]])
    confidence = torch.tensor([[0.9, 0.9, 0.9], [float("nan"), 0.8, 0.8]])
    color_rgb = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [float("nan"), 0.0, 0.0]],
        ]
    )

    result = export_colored_point_cloud_ply(
        disparity,
        color_rgb,
        _intrinsics(),
        baseline_m=0.5,
        output_path=output,
        confidence=confidence,
        min_confidence=0.7,
        min_depth_m=0.75,
        max_depth_m=1.5,
    )

    assert result.point_count == 1
    lines = output.read_text(encoding="ascii").splitlines()
    assert "element vertex 1" in lines
    assert lines[-1] == "0 0 1 255 0 0"


def test_export_ply_requires_single_frame_and_explicit_confidence_for_threshold(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        export_colored_point_cloud_ply(
            torch.ones((2, 1, 1)),
            torch.zeros((1, 1, 3), dtype=torch.uint8),
            _intrinsics(),
            baseline_m=0.5,
            output_path=tmp_path / "multi.ply",
        )
    with pytest.raises(ValueError, match="requires confidence"):
        export_colored_point_cloud_ply(
            torch.ones((1, 1)),
            torch.zeros((1, 1, 3), dtype=torch.uint8),
            _intrinsics(),
            baseline_m=0.5,
            output_path=tmp_path / "threshold.ply",
            min_confidence=0.8,
        )


def test_export_ply_rejects_ambiguous_float_rgb_and_bad_depth_range(tmp_path) -> None:
    kwargs = dict(
        disparity_hr_px=torch.ones((1, 1)),
        K_hr_px=_intrinsics(),
        baseline_m=0.5,
        output_path=tmp_path / "invalid.ply",
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        export_colored_point_cloud_ply(
            color_rgb=torch.full((1, 1, 3), 255.0), **kwargs
        )
    with pytest.raises(ValueError, match="less than or equal"):
        export_colored_point_cloud_ply(
            color_rgb=torch.zeros((1, 1, 3), dtype=torch.uint8),
            min_depth_m=2.0,
            max_depth_m=1.0,
            **kwargs,
        )
