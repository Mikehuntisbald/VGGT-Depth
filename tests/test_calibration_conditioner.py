from __future__ import annotations

import pytest
import torch

from models.calibration_conditioner import (
    CalibrationConditionerV3,
    dense_unit_rays_from_K_hr,
)
from models.ffs_omega_tsr import FFSOmegaTSR


def _base_model_inputs(
    *, batch_size: int = 2, height_lr: int = 4, width_lr: int = 6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    rgb_hr = torch.rand(
        batch_size,
        3,
        2 * height_lr,
        2 * width_lr,
        generator=generator,
    )
    disparity = 2.0 + torch.rand(
        batch_size, 1, height_lr, width_lr, generator=generator
    )
    confidence = torch.rand(
        batch_size, 1, height_lr, width_lr, generator=generator
    )
    return rgb_hr, disparity, confidence


def _calibration_inputs(
    *,
    batch_size: int = 2,
    baseline_value_m: float = 0.1,
    translation_scale: float = 1.0,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    K = torch.tensor(
        [[120.0, 0.0, 5.5], [0.0, 118.0, 3.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).repeat(batch_size, 1, 1)
    baseline = torch.full(
        (batch_size,),
        baseline_value_m * translation_scale,
        dtype=torch.float32,
        device=device,
    )
    static = torch.zeros(
        batch_size, 3, 4, dtype=torch.float32, device=device
    )
    static[:, :3, :3] = torch.eye(3, dtype=torch.float32, device=device)
    static[:, 0, 3] = -baseline

    temporal = torch.zeros(
        batch_size, 2, 3, 4, dtype=torch.float32, device=device
    )
    temporal[:, :, :3, :3] = torch.eye(
        3, dtype=torch.float32, device=device
    )
    temporal[:, 0, 0, 3] = 0.25 * baseline
    temporal[:, 1, 2, 3] = -0.5 * baseline
    temporal_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=device)
    return {
        "K_left_hr_px": K,
        "baseline_m": baseline,
        "E_right_camera_from_left_camera_m": static,
        "E_current_camera_from_history_camera_m": temporal,
        "temporal_pose_valid_mask": temporal_mask,
    }


def test_dense_unit_rays_are_fp32_unit_length_and_crop_equivariant() -> None:
    K_full = torch.tensor(
        [[[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    full = dense_unit_rays_from_K_hr(
        K_full, height_lr=4, width_lr=4, spatial_scale=2
    )
    assert full.dtype == torch.float32
    torch.testing.assert_close(
        torch.linalg.vector_norm(full, dim=1), torch.ones(1, 4, 4)
    )
    torch.testing.assert_close(full[0, :, 2, 2], torch.tensor([0.0, 0.0, 1.0]))

    # Crop the HR image at (2,2): the cropped LR origin corresponds to (1,1)
    # in the full LR grid under the repository's no-half-pixel convention.
    K_crop = K_full.clone()
    K_crop[:, 0, 2] -= 2.0
    K_crop[:, 1, 2] -= 2.0
    cropped = dense_unit_rays_from_K_hr(
        K_crop, height_lr=2, width_lr=2, spatial_scale=2
    )
    torch.testing.assert_close(cropped, full[:, :, 1:3, 1:3])


def test_conditioner_is_zero_initialized_and_has_no_metric_scale_path() -> None:
    torch.manual_seed(12)
    conditioner = CalibrationConditionerV3().eval()
    reference = torch.randn(2, 64, 4, 6)
    calibration = _calibration_inputs()

    with torch.no_grad():
        zero_residual = conditioner(reference, **calibration)
    torch.testing.assert_close(zero_residual, torch.zeros_like(zero_residual))

    # Open the zero-initialized residual adapter so this comparison exercises
    # all conditioning branches. Scaling every metric translation and baseline
    # together must leave static direction and dynamic ||t||/B unchanged.
    with torch.no_grad():
        conditioner.output_adapter.weight.normal_(mean=0.0, std=0.02)
        residual = conditioner(reference, **calibration)
        scaled = conditioner(
            reference,
            **_calibration_inputs(translation_scale=2.0),
        )
    torch.testing.assert_close(residual, scaled, atol=2e-6, rtol=2e-6)
    assert not any(
        token in name
        for name in conditioner.state_dict()
        for token in ("scale_token", "scale_head", "metric_scaling_factor")
    )


def test_masked_temporal_pose_slot_cannot_leak_into_conditioning() -> None:
    torch.manual_seed(13)
    conditioner = CalibrationConditionerV3().eval()
    with torch.no_grad():
        conditioner.output_adapter.weight.normal_(mean=0.0, std=0.02)
    reference = torch.randn(2, 64, 4, 6)
    calibration = _calibration_inputs()
    calibration["temporal_pose_valid_mask"][:, 1] = False

    changed = {name: value.clone() for name, value in calibration.items()}
    changed_dynamic = changed["E_current_camera_from_history_camera_m"]
    changed_dynamic[:, 1] = 37.0
    with torch.no_grad():
        expected = conditioner(reference, **calibration)
        actual = conditioner(reference, **changed)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_conditioner_rejects_bad_calibration_and_enforces_switches() -> None:
    reference = torch.randn(2, 64, 4, 6)
    calibration = _calibration_inputs()
    conditioner = CalibrationConditionerV3()

    mismatch = {name: value.clone() for name, value in calibration.items()}
    mismatch["baseline_m"] *= 2.0
    with pytest.raises(ValueError, match="translation norm must match baseline"):
        conditioner(reference, **mismatch)

    bad_rotation = {name: value.clone() for name, value in calibration.items()}
    bad_rotation["E_right_camera_from_left_camera_m"][:, 0, 0] = 2.0
    with pytest.raises(ValueError, match="proper orthonormal"):
        conditioner(reference, **bad_rotation)

    bad_mask = {name: value.clone() for name, value in calibration.items()}
    bad_mask["temporal_pose_valid_mask"] = torch.ones(2, 2)
    with pytest.raises(TypeError, match="dtype torch.bool"):
        conditioner(reference, **bad_mask)

    ray_only = CalibrationConditionerV3(
        use_rays=True, use_stereo_pose=False, use_temporal_pose=False
    )
    with torch.no_grad():
        output = ray_only(reference, K_left_hr_px=calibration["K_left_hr_px"])
    torch.testing.assert_close(output, torch.zeros_like(output))
    with pytest.raises(ValueError, match="pose inputs are disabled"):
        ray_only(
            reference,
            K_left_hr_px=calibration["K_left_hr_px"],
            E_right_camera_from_left_camera_m=calibration[
                "E_right_camera_from_left_camera_m"
            ],
        )


def test_conditioner_parameters_receive_finite_gradients_after_adapter_opens() -> None:
    torch.manual_seed(14)
    conditioner = CalibrationConditionerV3().train()
    with torch.no_grad():
        conditioner.output_adapter.weight.normal_(mean=0.0, std=0.02)
    reference = torch.randn(2, 64, 4, 6, requires_grad=True)
    calibration = _calibration_inputs()
    target = torch.randn_like(reference)

    residual = conditioner(reference, **calibration)
    (residual * target).mean().backward()

    parameters = (
        conditioner.ray_encoder[0][0].weight,
        conditioner.stereo_pose_encoder[0].weight,
        conditioner.temporal_pose_encoder[0].weight,
        conditioner.output_adapter.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert parameter.grad.abs().sum().item() > 0.0


def test_v3_disabled_preserves_exact_v2_state_and_behavior() -> None:
    torch.manual_seed(15)
    default = FFSOmegaTSR().eval()
    torch.manual_seed(15)
    explicit_disabled = FFSOmegaTSR(calibration_conditioning_v3=False).eval()

    assert tuple(default.state_dict()) == tuple(explicit_disabled.state_dict())
    for name, value in default.state_dict().items():
        torch.testing.assert_close(
            value, explicit_disabled.state_dict()[name], atol=0.0, rtol=0.0
        )
    assert not any(name.startswith("calibration_conditioner") for name in default.state_dict())

    inputs = _base_model_inputs()
    with torch.no_grad():
        expected = default(*inputs)
        actual = explicit_disabled(*inputs)
    for name in (
        "disparity_hr_px",
        "disparity_raw_hr_px",
        "source_weights",
        "log_variance",
        "uncertainty",
    ):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), atol=0.0, rtol=0.0
        )

    with pytest.raises(ValueError, match="calibration_conditioning_v3=True"):
        default(*inputs, K_left_hr_px=_calibration_inputs()["K_left_hr_px"])


def test_v3_zero_init_preserves_shared_model_output_and_budget() -> None:
    torch.manual_seed(16)
    legacy = FFSOmegaTSR().eval()
    torch.manual_seed(16)
    v3 = FFSOmegaTSR(calibration_conditioning_v3=True).eval()
    calibration = _calibration_inputs()

    legacy_state = legacy.state_dict()
    v3_state = v3.state_dict()
    for name, value in legacy_state.items():
        torch.testing.assert_close(value, v3_state[name], atol=0.0, rtol=0.0)
    assert v3.trainable_parameter_count > legacy.trainable_parameter_count
    assert v3.trainable_parameter_count < 12_000_000
    assert not any(
        token in name
        for name in v3_state
        for token in ("scale_token", "scale_head", "metric_scaling_factor")
    )

    inputs = _base_model_inputs()
    with torch.no_grad():
        expected = legacy(*inputs)
        actual = v3(*inputs, **calibration)
    for name in (
        "disparity_hr_px",
        "disparity_raw_hr_px",
        "source_weights",
        "log_variance",
        "uncertainty",
    ):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), atol=0.0, rtol=0.0
        )
    assert actual.disparity_hr_px.shape == (2, 1, 8, 12)
    assert all(torch.isfinite(state).all() for state in actual.hidden_state)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="native CUDA BF16 is unavailable",
)
def test_calibration_conditioner_cuda_bf16_forward_backward() -> None:
    device = torch.device("cuda")
    torch.manual_seed(17)
    conditioner = CalibrationConditionerV3().to(device).train()
    with torch.no_grad():
        conditioner.output_adapter.weight.normal_(mean=0.0, std=0.02)
    reference = torch.randn(2, 64, 4, 6, device=device, requires_grad=True)
    calibration = _calibration_inputs(device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        residual = conditioner(reference, **calibration)
        loss = residual.square().mean()
    loss.backward()

    assert residual.dtype == torch.bfloat16
    assert bool(torch.isfinite(residual).all())
    assert conditioner.output_adapter.weight.grad is not None
    assert bool(torch.isfinite(conditioner.output_adapter.weight.grad).all())
