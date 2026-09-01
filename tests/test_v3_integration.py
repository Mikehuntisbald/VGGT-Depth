from pathlib import Path

import pytest
import torch

import eval as eval_cli
import train
from data.cache_dataset import CacheMismatchError
from data.temporal_training_dataset import (
    CALIBRATED_DERIVED_ALGORITHM,
    _validate_derived_lineage,
)
from models.ffs_omega_tsr import count_trainable_parameters
from models.epipolar_refiner import HREpipolarRefiner


def _resolved(path: str):
    return train.resolve_config(
        path,
        ["data.calibration_sidecar_path=/tmp/calibration.jsonl"],
    )


def test_v3_ablation_models_are_parameter_matched_and_v2_is_unchanged() -> None:
    counts = {
        count_trainable_parameters(train.build_model(_resolved(path)))
        for path in (
            "configs/ablations/v3_a0_control.yaml",
            "configs/ablations/v3_a1_rays.yaml",
            "configs/ablations/v3_a2_stereo_pose.yaml",
            "configs/ablations/v3_a3_rays_stereo_pose.yaml",
            "configs/ablations/v3_b0_temporal_pose_off.yaml",
            "configs/ablations/v3_b1_temporal_pose_on.yaml",
        )
    }
    assert len(counts) == 1
    assert next(iter(counts)) < 12_000_000
    legacy = train.resolve_config("configs/mvp_x2_v2.yaml")
    contract = train.calibration_conditioning_v3_from_config(legacy)
    assert not contract.enabled
    assert train.build_model(legacy).calibration_conditioner is None


def _static_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    intrinsics = torch.tensor(
        [[100.0, 0.0, 3.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    ).repeat(batch_size, 1, 1)
    baseline = torch.full((batch_size,), 0.1)
    stereo = torch.eye(4).repeat(batch_size, 1, 1)
    stereo[:, 0, 3] = -baseline
    return {
        "rgb_hr": torch.zeros(batch_size, 3, 8, 12),
        "K_hr": intrinsics,
        "baseline_m": baseline,
        "T_right_rectified_from_left_rectified_m": stereo,
    }


def test_spatial_calibration_kwargs_follow_ablation_switches() -> None:
    batch = _static_batch()
    a0 = train.calibration_conditioning_v3_from_config(
        _resolved("configs/ablations/v3_a0_control.yaml")
    )
    assert train.calibration_model_kwargs_spatial(batch, a0) == {}

    a1 = train.calibration_conditioning_v3_from_config(
        _resolved("configs/ablations/v3_a1_rays.yaml")
    )
    assert set(train.calibration_model_kwargs_spatial(batch, a1)) == {
        "K_left_hr_px"
    }

    a3 = train.calibration_conditioning_v3_from_config(
        _resolved("configs/ablations/v3_a3_rays_stereo_pose.yaml")
    )
    kwargs = train.calibration_model_kwargs_spatial(batch, a3)
    assert set(kwargs) == {
        "K_left_hr_px",
        "baseline_m",
        "T_right_rectified_from_left_rectified_m",
    }
    assert kwargs["T_right_rectified_from_left_rectified_m"].shape == (2, 4, 4)


def _temporal_batch() -> dict[str, torch.Tensor]:
    static = _static_batch(batch_size=1)
    extrinsics = torch.zeros(1, 3, 10, 3, 4)
    extrinsics[..., :3, :3] = torch.eye(3)
    for endpoint in range(3):
        for pair in range(5):
            extrinsics[:, endpoint, 2 * pair, 0, 3] = -float(pair)
            extrinsics[:, endpoint, 2 * pair + 1, 0, 3] = -float(pair) - 0.1
    return {
        "rgb_hr_sequence": static["rgb_hr"].unsqueeze(1).repeat(1, 3, 1, 1, 1),
        "K_hr_sequence": static["K_hr"].unsqueeze(1).repeat(1, 3, 1, 1),
        "baseline_m_sequence": static["baseline_m"].unsqueeze(1).repeat(1, 3),
        "T_right_rectified_from_left_rectified_m_sequence": static[
            "T_right_rectified_from_left_rectified_m"
        ].unsqueeze(1).repeat(1, 3, 1, 1),
        "vggt_extrinsics_camera_from_world_metric_sequence": extrinsics,
        "temporal_pose_valid_sequence": torch.ones(1, 3, dtype=torch.bool),
    }


def test_temporal_calibration_kwargs_expose_only_existing_causal_ages() -> None:
    contract = train.calibration_conditioning_v3_from_config(
        _resolved("configs/ablations/v3_b1_temporal_pose_on.yaml")
    )
    batch = _temporal_batch()
    first = train.calibration_model_kwargs_temporal(
        batch, time_index=0, contract=contract
    )
    last = train.calibration_model_kwargs_temporal(
        batch, time_index=2, contract=contract
    )
    assert first["temporal_pose_valid"].tolist() == [[False, False]]
    assert last["temporal_pose_valid"].tolist() == [[True, True]]
    assert last["T_current_from_history_m"].shape == (1, 2, 4, 4)


def test_t1_endpoint_forces_temporal_pose_conditioning_invalid() -> None:
    contract = train.calibration_conditioning_v3_from_config(
        _resolved("configs/temporal_x2_v3.yaml")
    )
    assert contract.use_temporal_pose
    kwargs = eval_cli._calibration_model_kwargs_t1_from_temporal_batch(
        _temporal_batch(), time_index=2, contract=contract
    )

    assert kwargs["temporal_pose_valid"].tolist() == [[False, False]]
    identity = torch.eye(4, dtype=torch.float32)
    torch.testing.assert_close(
        kwargs["T_current_from_history_m"][0, 0], identity
    )
    torch.testing.assert_close(
        kwargs["T_current_from_history_m"][0, 1], identity
    )


def test_v3_config_fails_closed_on_contract_or_sidecar_mismatch() -> None:
    missing = train.resolve_config("configs/mvp_x2_v3.yaml")
    with pytest.raises(ValueError, match="calibration_sidecar_path"):
        train.validate_stage_a_config(missing)
    wrong = _resolved("configs/mvp_x2_v3.yaml")
    wrong.data.derived_contract = "legacy_v1"
    with pytest.raises(ValueError, match="calibrated_stereo_v2"):
        train.validate_stage_a_config(wrong)


def test_calibrated_per_record_lineage_rejects_sidecar_hash_drift() -> None:
    cache_lineage = {
        "component": "vggt-ffs-derived-geometry-calibrated-stereo-v2",
        "contract_version": "stored_rectified_virtual_cameras_v1",
        "sidecar_sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
        "pixel_audit_sha256": "c" * 64,
    }
    payload = {
        "metadata": {
            "config": {
                "algorithm": CALIBRATED_DERIVED_ALGORITHM,
                "extrinsics_convention": "camera-from-world",
                "previous_left_view_index": 6,
                "current_left_view_index": 8,
                "invalid_temporal_pose_policy": (
                    "zero-filled with false validity tensor"
                ),
                "rectified_stereo_calibration": {
                    **cache_lineage,
                    "sidecar_sha256": "d" * 64,
                },
            }
        }
    }

    with pytest.raises(CacheMismatchError, match="calibration lineage mismatch"):
        _validate_derived_lineage(
            payload,
            record=object(),
            observation_cache_path=Path("/not/reached"),
            observation_cache_sha256="e" * 64,
            cache_path=Path("derived.pt"),
            derived_contract="calibrated_stereo_v2",
            expected_calibration_lineage=cache_lineage,
        )


def test_tiny_positive_disparity_uses_the_same_explicit_metric_formula() -> None:
    disparity_hr_px = torch.tensor([[[[torch.finfo(torch.float32).eps / 2]]]])
    intrinsics_hr = torch.tensor(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    baseline_m = torch.tensor([0.1])

    depth_m, valid = train._metric_depth_from_hr_disparity(
        disparity_hr_px,
        intrinsics_hr=intrinsics_hr,
        baseline_m=baseline_m,
    )

    assert valid.item() is True
    torch.testing.assert_close(
        depth_m,
        intrinsics_hr[:, 0, 0].reshape(1, 1, 1, 1)
        * baseline_m.reshape(1, 1, 1, 1)
        / disparity_hr_px,
    )


def test_stage_c_same_checkpoint_and_input_is_exact_after_v3_instantiation() -> None:
    torch.manual_seed(20260901)
    reference_refiner = HREpipolarRefiner(base_aware_noop_v2=True).eval()
    checkpoint_state = {
        name: value.detach().clone()
        for name, value in reference_refiner.state_dict().items()
    }
    rgb_left = torch.rand(1, 3, 7, 11)
    rgb_right = torch.rand(1, 3, 7, 11)
    base_disparity_hr_px = 0.25 + torch.rand(1, 1, 7, 11)
    with torch.no_grad():
        expected = reference_refiner(
            rgb_left, rgb_right, base_disparity_hr_px
        )

    # Constructing the complete v3 graph must not mutate or replace the
    # separately owned Stage-C module/checkpoint contract.
    _ = train.build_model(_resolved("configs/temporal_x2_v3.yaml"))
    reloaded_refiner = HREpipolarRefiner(base_aware_noop_v2=True).eval()
    reloaded_refiner.load_state_dict(checkpoint_state, strict=True)
    with torch.no_grad():
        actual = reloaded_refiner(rgb_left, rgb_right, base_disparity_hr_px)

    for name in (
        "corrected_disparity_hr_px",
        "correction_hr_px",
        "correlation",
        "candidate_valid_mask",
        "confidence",
        "no_op_probability",
        "no_op_mask",
        "output_valid_mask",
    ):
        expected_value = getattr(expected, name)
        actual_value = getattr(actual, name)
        assert expected_value is not None and actual_value is not None
        torch.testing.assert_close(actual_value, expected_value, atol=0.0, rtol=0.0)
