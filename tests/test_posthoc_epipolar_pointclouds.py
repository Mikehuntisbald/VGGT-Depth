from __future__ import annotations

from pathlib import Path
from types import ModuleType

import torch

from tools import export_epipolar_pointclouds_posthoc as posthoc


def test_posthoc_callback_exports_calibrated_base_and_refined_ply_from_captured_endpoint(
    tmp_path: Path,
) -> None:
    evaluator = ModuleType("synthetic_frozen_evaluator")

    def validate_epipolar_batch_causality(batch: dict[str, object]) -> list[object]:
        del batch
        return []

    def save_visualization(
        root: Path,
        *,
        sample_name: str,
        rgb_left_hr: torch.Tensor,
        rgb_right_hr: torch.Tensor,
        base_disparity_hr_px: torch.Tensor,
        refined_disparity_hr_px: torch.Tensor,
        correction_hr_px: torch.Tensor,
        confidence: torch.Tensor,
        target_disparity_hr_px: torch.Tensor,
        target_trusted_mask: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        correction_limit_hr_px: float,
        provenance: dict[str, object],
    ) -> None:
        del (
            rgb_left_hr,
            rgb_right_hr,
            base_disparity_hr_px,
            refined_disparity_hr_px,
            correction_hr_px,
            confidence,
            target_disparity_hr_px,
            target_trusted_mask,
            candidate_valid_mask,
            correction_limit_hr_px,
            provenance,
        )
        (root / sample_name).mkdir(parents=True, exist_ok=True)

    evaluator.validate_epipolar_batch_causality = validate_epipolar_batch_causality
    evaluator._save_visualization = save_visualization
    callback = posthoc.install_posthoc_ply_callback(evaluator)

    K_hr_px = torch.tensor(
        [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]]
    ).repeat(1, 3, 1, 1)
    evaluator.validate_epipolar_batch_causality(
        {
            "K_hr_sequence": K_hr_px,
            "baseline_m_sequence": torch.tensor([[0.1, 0.1, 0.1]]),
            "sequence_id": ["seq/one"],
            "frame_ids": torch.tensor([[10, 11, 12]]),
            "manifest_indices": torch.tensor([[20, 21, 22]]),
        }
    )
    base = torch.tensor([[[2.0, 0.0, float("nan")], [-1.0, 4.0, 2.0]]])
    refined = torch.ones((1, 2, 3)) * 2.0
    evaluator._save_visualization(
        tmp_path / "visualizations",
        sample_name="0000_seq_one_12",
        rgb_left_hr=torch.zeros((3, 2, 3)),
        rgb_right_hr=torch.zeros((3, 2, 3)),
        base_disparity_hr_px=base,
        refined_disparity_hr_px=refined,
        correction_hr_px=refined - base,
        confidence=torch.ones((1, 2, 3)),
        target_disparity_hr_px=torch.ones((1, 2, 3)),
        target_trusted_mask=torch.ones((1, 2, 3), dtype=torch.bool),
        candidate_valid_mask=torch.ones((1, 2, 3), dtype=torch.bool),
        correction_limit_hr_px=2.0,
        provenance={"manifest_index": 22, "sequence_id": "seq/one", "frame_id": 12},
    )

    directory = tmp_path / "visualizations/0000_seq_one_12"
    base_ply = (directory / "base_point_cloud_camera_frame.ply").read_text(
        encoding="ascii"
    )
    refined_ply = (directory / "refined_point_cloud_camera_frame.ply").read_text(
        encoding="ascii"
    )
    assert "comment camera_frame left; coordinates_m" in base_ply
    assert "element vertex 3" in base_ply
    assert "element vertex 6" in refined_ply
    assert callback.records == [
        {
            "sample_name": "0000_seq_one_12",
            "endpoint": {
                "manifest_index": 22,
                "sequence_id": "seq/one",
                "frame_id": 12,
                "K_hr_px": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                "baseline_m": 0.10000000149011612,
            },
            "coordinate_frame": "left_camera_frame",
            "coordinate_units": "meters",
            "base": {
                "path": str(directory / "base_point_cloud_camera_frame.ply"),
                "point_count": 3,
            },
            "refined": {
                "path": str(directory / "refined_point_cloud_camera_frame.ply"),
                "point_count": 6,
            },
        }
    ]
    assert callback.captured == {}
