from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from models.current_conditioned_history import (  # noqa: E402
    CurrentConditionedHistoryOutput,
    CurrentConditionedTopKAttention,
)


def _inputs(
    *,
    batch: int = 1,
    candidates: int = 3,
    height: int = 2,
    width: int = 3,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(42)
    scalar_shape = (batch, candidates, height, width)
    transforms = torch.eye(4, dtype=torch.float32, device=device).reshape(
        1, 1, 4, 4
    ).repeat(batch, 2, 1, 1)
    transforms[:, 0, 0, 3] = 0.10
    transforms[:, 1, 1, 3] = 0.20
    age = torch.ones(scalar_shape, dtype=torch.float32, device=device)
    if candidates > 1:
        age[:, 1::2] = 2.0
    return {
        "rgb_feature_lr": torch.rand(
            (batch, 96, height, width), generator=generator, device=device
        ),
        "geometry_feature_lr": torch.rand(
            (batch, 64, height, width), generator=generator, device=device
        ),
        "candidate_feature": torch.rand(
            (batch, candidates, 32, height, width),
            generator=generator,
            device=device,
        ),
        "disparity_hr_px": 2.0
        + 8.0 * torch.rand(scalar_shape, generator=generator, device=device),
        "depth_m": 1.0
        + 4.0 * torch.rand(scalar_shape, generator=generator, device=device),
        "confidence": 0.2
        + 0.8 * torch.rand(scalar_shape, generator=generator, device=device),
        "fractional_phase_grid_px": 2.0
        * torch.rand(
            (batch, candidates, 2, height, width),
            generator=generator,
            device=device,
        )
        - 1.0,
        "temporal_age_frames": age,
        "z_aware_prior_weights": 0.1
        + torch.rand(scalar_shape, generator=generator, device=device),
        "pose_quality": torch.rand(scalar_shape, generator=generator, device=device),
        "depth_layer_index": torch.zeros(
            scalar_shape, dtype=torch.float32, device=device
        ),
        "valid_mask": torch.ones(scalar_shape, dtype=torch.bool, device=device),
        "front_surface_mask": torch.ones(
            scalar_shape, dtype=torch.bool, device=device
        ),
        "context_only_mask": torch.zeros(
            scalar_shape, dtype=torch.bool, device=device
        ),
        "current_ffs_disparity_hr_px": torch.full(
            (batch, 1, height, width), 6.0, device=device
        ),
        "current_ffs_confidence": torch.full(
            (batch, 1, height, width), 0.8, device=device
        ),
        "T_current_from_history_m": transforms,
        "baseline_m": torch.full(
            (batch,), 0.10, dtype=torch.float32, device=device
        ),
        "temporal_pose_valid": torch.ones(
            (batch, 2), dtype=torch.bool, device=device
        ),
    }


def _normalized_prior(
    prior: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    masked = torch.where(mask, prior, torch.zeros_like(prior))
    denominator = masked.sum(dim=1, keepdim=True)
    return torch.where(
        denominator > 0,
        masked / denominator.clamp_min(torch.finfo(masked.dtype).tiny),
        torch.zeros_like(masked),
    )


def test_zero_initialized_scores_reproduce_prior_inside_each_mask() -> None:
    model = CurrentConditionedTopKAttention().eval()
    inputs = _inputs(candidates=4, height=2, width=2)
    inputs["front_surface_mask"][:, 2:] = False
    inputs["context_only_mask"][:, 1] = True
    inputs["front_surface_mask"][:, 1] = False
    inputs["valid_mask"][:, 3, 0, 0] = False

    output = model(**inputs)

    assert isinstance(output, CurrentConditionedHistoryOutput)
    metric_mask = (
        inputs["valid_mask"]
        & inputs["front_surface_mask"]
        & ~inputs["context_only_mask"]
    )
    context_mask = inputs["valid_mask"]
    torch.testing.assert_close(
        output.metric_weights,
        _normalized_prior(inputs["z_aware_prior_weights"], metric_mask),
    )
    torch.testing.assert_close(
        output.context_weights,
        _normalized_prior(inputs["z_aware_prior_weights"], context_mask),
    )
    expected_disparity = (
        output.metric_weights * inputs["disparity_hr_px"]
    ).sum(dim=1, keepdim=True)
    torch.testing.assert_close(output.metric_disparity_hr_px, expected_disparity)


def test_back_layer_is_context_only_and_never_changes_metric_disparity() -> None:
    model = CurrentConditionedTopKAttention().eval()
    inputs = _inputs(candidates=2, height=1, width=1)
    inputs["front_surface_mask"] = torch.tensor([[[[True]], [[False]]]])
    inputs["context_only_mask"] = torch.tensor([[[[False]], [[True]]]])
    inputs["z_aware_prior_weights"] = torch.tensor([[[[0.25]], [[0.75]]]])
    inputs["disparity_hr_px"] = torch.tensor([[[[8.0]], [[80.0]]]])
    inputs["candidate_feature"].zero_()
    inputs["candidate_feature"][:, 0] = 1.0
    inputs["candidate_feature"][:, 1] = 5.0

    with_back = model(**inputs)
    without_back_inputs = dict(inputs)
    without_back_inputs["valid_mask"] = inputs["valid_mask"].clone()
    without_back_inputs["valid_mask"][:, 1] = False
    without_back = model(**without_back_inputs)

    torch.testing.assert_close(
        with_back.metric_disparity_hr_px, torch.tensor([[[[8.0]]]])
    )
    torch.testing.assert_close(
        with_back.metric_disparity_hr_px, without_back.metric_disparity_hr_px
    )
    assert with_back.metric_weights[:, 1].eq(0).all()
    assert not torch.allclose(with_back.context_feature, without_back.context_feature)
    torch.testing.assert_close(
        with_back.context_feature,
        torch.full_like(with_back.context_feature, 4.0),
    )


def test_current_conditioned_score_head_gets_finite_nonzero_gradient() -> None:
    model = CurrentConditionedTopKAttention().train()
    inputs = _inputs(candidates=3, height=2, width=3)
    inputs["disparity_hr_px"][:, 0] = 2.0
    inputs["disparity_hr_px"][:, 1] = 8.0
    inputs["disparity_hr_px"][:, 2] = 14.0
    output = model(**inputs)
    loss = (
        output.metric_disparity_hr_px.square().mean()
        + output.context_feature.square().mean()
    )
    loss.backward()

    final = model.score_head[-1]
    assert isinstance(final, torch.nn.Conv2d)
    assert final.weight.grad is not None
    assert bool(torch.isfinite(final.weight.grad).all())
    assert bool((final.weight.grad.abs().sum(dim=(1, 2, 3)) > 0).all())


def test_age_factorized_motion_descriptor_and_invalid_pose_are_exact() -> None:
    model = CurrentConditionedTopKAttention().eval()
    inputs = _inputs(candidates=4, height=1, width=1)
    inputs["temporal_age_frames"] = torch.tensor(
        [[[[1.0]], [[2.0]], [[3.0]], [[2.0]]]]
    )
    transforms = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(
        1, 2, 1, 1
    )
    transforms[:, 0, :3, 3] = torch.tensor([0.20, 0.0, 0.0])
    transforms[:, 1, :3, 3] = torch.tensor([0.0, 0.30, 0.0])
    inputs["T_current_from_history_m"] = transforms
    inputs["baseline_m"] = torch.tensor([0.10], dtype=torch.float32)
    inputs["temporal_pose_valid"] = torch.tensor([[True, False]])

    output = model(**inputs)

    expected_age_one = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, math.log(3.0)]
    )
    torch.testing.assert_close(
        output.candidate_pose_descriptor[0, 0, :, 0, 0], expected_age_one
    )
    assert output.candidate_pose_descriptor[0, 1].eq(0).all()
    assert output.candidate_pose_descriptor[0, 2].eq(0).all()
    assert output.candidate_pose_descriptor[0, 3].eq(0).all()


def test_all_invalid_is_exactly_fail_closed() -> None:
    model = CurrentConditionedTopKAttention().eval()
    inputs = _inputs(candidates=3, height=2, width=2)
    inputs["valid_mask"].zero_()

    output = model(**inputs)

    assert not bool(output.metric_valid_mask.any())
    assert not bool(output.context_valid_mask.any())
    for value in (
        output.metric_disparity_hr_px,
        output.metric_confidence,
        output.context_feature,
        output.metric_weights,
        output.context_weights,
        output.candidate_pose_descriptor,
    ):
        assert value.eq(0).all()
        assert bool(torch.isfinite(value).all())


def test_nonfinite_candidate_measurements_are_invalidated_without_poisoning() -> None:
    model = CurrentConditionedTopKAttention().eval()
    inputs = _inputs(candidates=3, height=1, width=2)
    inputs["disparity_hr_px"][:, 0, 0, 0] = float("nan")
    inputs["depth_m"][:, 1, 0, 1] = float("inf")
    inputs["confidence"][:, 2, 0, 0] = float("nan")
    inputs["z_aware_prior_weights"][:, 2, 0, 1] = float("inf")

    output = model(**inputs)

    assert output.metric_weights[0, 0, 0, 0].item() == 0.0
    assert output.context_weights[0, 1, 0, 1].item() == 0.0
    assert output.metric_weights[0, 2, 0, 0].item() == 0.0
    assert output.context_weights[0, 2, 0, 1].item() == 0.0
    for value in (
        output.metric_disparity_hr_px,
        output.metric_confidence,
        output.context_feature,
        output.metric_weights,
        output.context_weights,
    ):
        assert bool(torch.isfinite(value).all())


@pytest.mark.parametrize(
    ("field", "replacement", "exception", "match"),
    [
        ("rgb_feature_lr", torch.zeros(1, 95, 2, 3), ValueError, "rgb_feature"),
        (
            "candidate_feature",
            torch.zeros(1, 3, 31, 2, 3),
            ValueError,
            "candidate_feature",
        ),
        ("valid_mask", torch.ones(1, 3, 2, 3), TypeError, "valid_mask"),
        (
            "fractional_phase_grid_px",
            torch.zeros(1, 3, 1, 2, 3),
            ValueError,
            "fractional_phase",
        ),
        (
            "T_current_from_history_m",
            torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
            TypeError,
            "float32",
        ),
        ("baseline_m", torch.tensor([0.0]), ValueError, "positive"),
    ],
)
def test_shape_and_dtype_contracts_are_rejected_explicitly(
    field: str,
    replacement: torch.Tensor,
    exception: type[Exception],
    match: str,
) -> None:
    model = CurrentConditionedTopKAttention()
    inputs = _inputs(candidates=3, height=2, width=3)
    inputs[field] = replacement
    with pytest.raises(exception, match=match):
        model(**inputs)


def test_feature_dtype_and_nonfinite_contracts_are_rejected() -> None:
    model = CurrentConditionedTopKAttention()
    integer_candidate = _inputs(candidates=2)
    integer_candidate["candidate_feature"] = torch.zeros(
        (1, 2, 32, 2, 3), dtype=torch.int64
    )
    with pytest.raises(TypeError, match="candidate_feature.*floating"):
        model(**integer_candidate)

    nonfinite_rgb = _inputs(candidates=2)
    nonfinite_rgb["rgb_feature_lr"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="rgb_feature_lr.*finite"):
        model(**nonfinite_rgb)

    nonfinite_candidate = _inputs(candidates=2)
    nonfinite_candidate["candidate_feature"][0, 1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="candidate_feature.*finite"):
        model(**nonfinite_candidate)


def test_device_mismatch_is_rejected_before_tensor_arithmetic() -> None:
    model = CurrentConditionedTopKAttention()
    inputs = _inputs(candidates=2)
    inputs["candidate_feature"] = torch.empty(
        (1, 2, 32, 2, 3), device="meta"
    )
    with pytest.raises(ValueError, match="device"):
        model(**inputs)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_cuda_bf16_forward_backward_is_finite() -> None:
    device = torch.device("cuda")
    model = CurrentConditionedTopKAttention().to(device).train()
    inputs = _inputs(candidates=4, height=3, width=4, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**inputs)
        loss = output.metric_disparity_hr_px.mean() + output.context_feature.mean()
    loss.backward()

    for value in (
        output.metric_disparity_hr_px,
        output.metric_confidence,
        output.context_feature,
        output.metric_weights,
        output.context_weights,
    ):
        assert bool(torch.isfinite(value).all())
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
