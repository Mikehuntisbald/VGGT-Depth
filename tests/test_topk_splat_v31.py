from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from geometry.topk_splat import (
    TOPK_DIVERSITY_V31_CONTRACT,
    TopKSplatResult,
    merge_topk_splat_results,
    merge_topk_splat_results_v31,
    topk_diversity_diagnostics,
    topk_z_aware_splat,
)


def _candidate_result(
    *,
    age: float,
    depth_m: list[float],
    disparity_hr_px: list[float] | None = None,
    phase_u: list[float] | None = None,
    hidden_base: torch.Tensor | None = None,
) -> TopKSplatResult:
    """Create a shape-valid splat result with controlled 1-pixel candidates."""

    top_k = len(depth_m)
    scalar = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    intrinsics = torch.eye(3, dtype=torch.float32)
    base = topk_z_aware_splat(
        scalar,
        scalar,
        scalar,
        intrinsics,
        torch.eye(4),
        torch.eye(4),
        top_k=top_k,
        temporal_age_frames=age,
        previous_hidden_feature=(
            torch.ones((1, 1, 1, 1)) if hidden_base is not None else None
        ),
    )
    depth = torch.tensor(depth_m, dtype=torch.float32).reshape(1, top_k, 1, 1)
    valid = torch.isfinite(depth) & (depth > 0)
    depth = torch.where(valid, depth, torch.zeros_like(depth))
    disparity_values = disparity_hr_px or [10.0 / value for value in depth_m]
    disparity = torch.tensor(disparity_values, dtype=torch.float32).reshape(
        1, top_k, 1, 1
    )
    phase_values = phase_u or [0.0] * top_k
    phase = torch.zeros((1, top_k, 2, 1, 1), dtype=torch.float32)
    phase[0, :, 0, 0, 0] = torch.tensor(phase_values)
    source_index = torch.arange(top_k).reshape(1, top_k, 1, 1)
    warped_hidden = None
    weighted_hidden = None
    if hidden_base is not None:
        if hidden_base.shape != (1, top_k, 1, 1, 1):
            raise ValueError("hidden_base must have shape [1,K,1,1,1]")
        warped_hidden = hidden_base
        weighted_hidden = hidden_base[:, 0]
    return replace(
        base,
        disparity_hr_px=disparity,
        depth_m=depth,
        confidence=valid.to(dtype=torch.float32),
        temporal_age_frames=valid.to(dtype=torch.float32) * age,
        valid_mask=valid,
        visibility_mask=valid & (source_index == 0),
        collision_mask=valid & (valid.sum(dim=1, keepdim=True) > 1),
        source_visibility_mask=valid,
        source_collision_mask=torch.zeros_like(valid),
        footprint_weight=valid.to(dtype=torch.float32),
        projected_uv_grid_px=phase,
        fractional_offset_grid_px=phase,
        source_uv_grid_px=torch.zeros_like(phase),
        source_sequence_index=torch.where(
            valid, torch.zeros_like(source_index), torch.full_like(source_index, -1)
        ),
        source_linear_index=torch.where(
            valid, source_index, torch.full_like(source_index, -1)
        ),
        candidate_count=valid.sum(dim=1, keepdim=True),
        z_aware_weights=valid.to(dtype=torch.float32)
        / valid.sum(dim=1, keepdim=True).clamp_min(1),
        aggregate_valid_mask=valid.any(dim=1, keepdim=True),
        weighted_disparity_hr_px=disparity[:, :1],
        weighted_depth_m=depth[:, :1],
        weighted_confidence=valid[:, :1].to(dtype=torch.float32),
        weighted_fractional_offset_grid_px=phase[:, 0],
        weighted_temporal_age_frames=valid[:, :1].to(dtype=torch.float32) * age,
        warped_hidden_feature=warped_hidden,
        weighted_hidden_feature=weighted_hidden,
    )


def _assert_result_equal(left: TopKSplatResult, right: TopKSplatResult) -> None:
    for name in TopKSplatResult.__dataclass_fields__:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None:
            assert right_value is None
        elif left_value.dtype == torch.bool:
            assert torch.equal(left_value, right_value), name
        else:
            torch.testing.assert_close(left_value, right_value, msg=name)


def test_default_merge_remains_exact_global_depth_v2() -> None:
    age1 = _candidate_result(age=1, depth_m=[1.0, 1.1, 1.2, 1.3])
    age2 = _candidate_result(age=2, depth_m=[0.9, 1.4, 1.5, 1.6])
    implicit = merge_topk_splat_results([age1, age2], top_k=4)
    explicit = merge_topk_splat_results(
        [age1, age2], top_k=4, selection_contract="global_depth_v2"
    )
    _assert_result_equal(implicit, explicit)
    assert implicit.front_surface_mask is None


def test_v31_quota_keeps_two_per_age_and_guarantees_age2_front_survival() -> None:
    age1 = _candidate_result(
        age=1,
        depth_m=[1.000, 1.010, 1.020, 1.030],
        phase_u=[0.0, 0.1, 0.2, 0.3],
    )
    age2 = _candidate_result(
        age=2,
        depth_m=[1.015, 1.025, 1.035, 1.045],
        phase_u=[0.5, 0.6, 0.7, 0.8],
    )
    merged = merge_topk_splat_results_v31(
        [age1, age2], top_k=4, per_age_quota=2, surface_depth_gap_m=0.05
    )

    assert merged.source_sequence_index[0, :, 0, 0].tolist().count(0) == 2
    assert merged.source_sequence_index[0, :, 0, 0].tolist().count(1) == 2
    assert 2.0 in merged.temporal_age_frames[0, :, 0, 0].tolist()
    assert bool(merged.age2_depth_consistent_available_mask.item())
    assert merged.depth_m[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert bool(merged.visibility_mask[0, 0, 0, 0])


def test_v31_phase_penalty_rejects_redundant_same_age_candidate() -> None:
    age1 = _candidate_result(
        age=1,
        depth_m=[1.000, 1.001, 1.002, 1.003],
        phase_u=[0.00, 0.01, 0.50, -0.99],
    )
    merged = merge_topk_splat_results_v31(
        [age1],
        top_k=4,
        per_age_quota=2,
        surface_depth_gap_m=0.05,
        phase_redundancy_sigma_grid_px=0.125,
        phase_redundancy_penalty=1.0,
    )

    assert merged.valid_mask.sum().item() == 2
    # -0.99 and +0.01 are the same canonical phase, so the complementary 0.5
    # candidate survives even though it is slightly deeper.
    assert merged.source_linear_index[0, :, 0, 0].tolist() == [0, 2, -1, -1]
    torch.testing.assert_close(
        merged.fractional_offset_grid_px[0, :2, 0, 0, 0],
        torch.tensor([0.0, 0.5]),
    )


def test_v31_back_layers_are_context_only_and_never_metric_averaged() -> None:
    hidden1 = torch.tensor([1.0, 100.0, 0.0, 0.0]).reshape(1, 4, 1, 1, 1)
    hidden2 = torch.tensor([3.0, 200.0, 0.0, 0.0]).reshape(1, 4, 1, 1, 1)
    hidden1.requires_grad_()
    hidden2.requires_grad_()
    age1 = _candidate_result(
        age=1,
        depth_m=[1.00, 2.00, 0.0, 0.0],
        disparity_hr_px=[10.0, 5.0, 0.0, 0.0],
        hidden_base=hidden1,
    )
    age2 = _candidate_result(
        age=2,
        depth_m=[1.01, 2.10, 0.0, 0.0],
        disparity_hr_px=[9.9, 4.8, 0.0, 0.0],
        hidden_base=hidden2,
    )
    merged = merge_topk_splat_results_v31(
        [age1, age2], top_k=4, per_age_quota=2, surface_depth_gap_m=0.05
    )

    assert merged.depth_layer_index[0, :, 0, 0].tolist() == [0, 0, 1, 1]
    assert merged.front_surface_mask[0, :, 0, 0].tolist() == [True, True, False, False]
    assert merged.context_only_mask[0, :, 0, 0].tolist() == [False, False, True, True]
    torch.testing.assert_close(
        merged.z_aware_weights[0, 2:, 0, 0], torch.zeros(2)
    )
    assert 9.9 <= merged.weighted_disparity_hr_px.item() <= 10.0
    assert merged.weighted_hidden_feature.item() < 4.0
    merged.weighted_hidden_feature.sum().backward()
    assert hidden1.grad is not None and hidden2.grad is not None
    assert hidden1.grad[0, 0].item() > 0
    assert hidden2.grad[0, 0].item() > 0
    assert hidden1.grad[0, 1].item() == 0
    assert hidden2.grad[0, 1].item() == 0


def test_v31_diagnostics_report_required_diversity_and_reference_epe() -> None:
    age1 = _candidate_result(
        age=1,
        depth_m=[1.00, 1.02, 2.00, 2.10],
        disparity_hr_px=[10.0, 9.8, 5.0, 4.8],
        phase_u=[0.0, 0.2, 0.0, 0.1],
    )
    age2 = _candidate_result(
        age=2,
        depth_m=[1.01, 1.03, 2.20, 2.30],
        disparity_hr_px=[11.0, 10.8, 4.5, 4.3],
        phase_u=[0.5, 0.7, 0.5, 0.6],
    )
    merged = merge_topk_splat_results_v31(
        [age1, age2], top_k=4, per_age_quota=2, surface_depth_gap_m=0.05
    )
    diagnostics = topk_diversity_diagnostics(
        merged, reference_disparity_hr_px=torch.tensor([[[[10.5]]]])
    )

    assert diagnostics.valid_target_count.item() == 1
    assert diagnostics.unique_age_fraction.item() == pytest.approx(1.0)
    assert diagnostics.age2_survival_rate.item() == pytest.approx(1.0)
    assert diagnostics.fractional_phase_variance.item() > 0
    assert diagnostics.topk_weight_entropy.item() > 0
    assert diagnostics.candidate_depth_spread_m.item() > 0
    assert diagnostics.rank0_disparity_epe_hr_px is not None
    assert diagnostics.weighted_disparity_epe_hr_px is not None
    assert diagnostics.weighted_minus_rank0_epe_hr_px is not None
    for value in (
        diagnostics.rank0_disparity_epe_hr_px,
        diagnostics.weighted_disparity_epe_hr_px,
        diagnostics.weighted_minus_rank0_epe_hr_px,
    ):
        assert bool(torch.isfinite(value))


def test_v31_fails_closed_for_bad_contract_and_diagnostics_on_v2() -> None:
    age1 = _candidate_result(age=1, depth_m=[1.0, 1.1, 1.2, 1.3])
    age2 = _candidate_result(age=2, depth_m=[1.0, 1.1, 1.2, 1.3])
    with pytest.raises(ValueError, match="selection_contract"):
        merge_topk_splat_results(
            [age1, age2], top_k=4, selection_contract="implicit_v3"
        )
    with pytest.raises(ValueError, match="per_age_quota"):
        merge_topk_splat_results(
            [age1, age2],
            top_k=4,
            selection_contract=TOPK_DIVERSITY_V31_CONTRACT,
            per_age_quota=3,
        )
    v2 = merge_topk_splat_results([age1, age2], top_k=4)
    with pytest.raises(ValueError, match="v3.1"):
        topk_diversity_diagnostics(v2)
