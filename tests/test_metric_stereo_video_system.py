from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn
import torch.nn.functional as F

from backbones.trainable_stereo import TrainableFastFoundationStereo  # noqa: E402
from backbones.trainable_vggt_omega import TrainableVGGTOmega  # noqa: E402
from models.metric_stereo_video_geometry import (  # noqa: E402
    CausalMetricStereoVideoGeometry,
)
from models.metric_stereo_video_system import (  # noqa: E402
    MetricStereoVideoSystem,
    _assert_tensor_condition,
    left_right_stereo_consistency,
    vggt_unbounded_confidence_to_probability,
)


class _DummyStereoFeature(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=1)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        return [F.avg_pool2d(self.projection(value), 4)]


class _DummyStereo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = _DummyStereoFeature()
        self.disparity_lr_px = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        iters: int,
        test_mode: bool,
        low_memory: bool,
        optimize_build_volume: str,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del right, test_mode, low_memory, optimize_build_volume
        _features = self.feature(torch.cat((left, left), dim=0))
        final = torch.ones_like(left[:, :1]) * self.disparity_lr_px
        predictions = [final * float(index + 1) / iters for index in range(iters)]
        return F.avg_pool2d(final, 4), predictions


class _CountingStereoWrapper(TrainableFastFoundationStereo):
    def __init__(self) -> None:
        super().__init__(
            _DummyStereo(), iterations=2, max_disp=192, predict_right=True
        )
        self.forward_calls = 0

    def forward(self, left_rgb: torch.Tensor, right_rgb: torch.Tensor):
        self.forward_calls += 1
        return super().forward(left_rgb, right_rgb)


class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 4


class _DummyAggregator(nn.Module):
    patch_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.projection = nn.Linear(3, 8)
        self.calls = 0

    def forward(
        self, images: torch.Tensor
    ) -> tuple[list[torch.Tensor], int]:
        self.calls += 1
        batch, frames, _, height, width = images.shape
        pooled = F.avg_pool2d(
            images.reshape(batch * frames, 3, height, width), 4
        ).reshape(batch, frames, 3, -1)
        patches = self.projection(pooled.transpose(2, 3))
        camera = patches.new_zeros(batch, frames, 1, patches.shape[-1])
        tokens = torch.cat((camera, patches), dim=2)
        return [tokens for _ in range(24)], 1


class _DummyDenseHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depth_scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(
        self,
        layers: list[torch.Tensor | None],
        *,
        images: torch.Tensor,
        patch_token_start: int,
        frames_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del layers, patch_token_start, frames_chunk_size
        self.calls += 1
        depth = (1.0 + images.mean(dim=2)) * self.depth_scale
        confidence = torch.full_like(depth, 2.0)
        return depth.unsqueeze(-1), confidence


class _DummyVGGT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = _DummyAggregator()
        self.dense_head = _DummyDenseHead()


class _RecordingGeometry(CausalMetricStereoVideoGeometry):
    def __init__(self) -> None:
        super().__init__(
            stereo_feature_channels=8,
            vggt_feature_channels=8,
            hidden_channels=16,
            residual_blocks=1,
            minimum_gauge_overlap=4,
        )
        self.vggt_contexts: list[tuple[int, ...]] = []
        self.vggt_confidence_sums: list[float] = []

    def forward_step(self, frame, state=None):
        self.vggt_contexts.append(frame.vggt_features.context_time_indices)
        self.vggt_confidence_sums.append(
            float(frame.vggt_features.confidence.detach().sum())
        )
        return super().forward_step(frame, state)


def _batch(*, frames: int = 3, padded: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    rgb = torch.rand(1, frames, 2, 3, 64, 64, generator=generator)
    K_one = torch.tensor(
        [[20.0, 0.0, 31.5], [0.0, 20.0, 31.5], [0.0, 0.0, 1.0]]
    )
    K = K_one.reshape(1, 1, 1, 3, 3).repeat(1, frames, 2, 1, 1)
    stereo_transform = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, frames, 1, 1)
    stereo_transform[:, :, 0, 3] = -0.1
    temporal_transform = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        1, frames, 1, 1
    )
    time_valid = torch.ones(1, frames, dtype=torch.bool)
    if padded:
        time_valid[:, 0] = False
        K[:, 0] = 0
    temporal_valid = torch.ones(1, frames, dtype=torch.bool)
    temporal_valid[:, 0] = False
    target_time = torch.zeros(1, frames, dtype=torch.bool)
    target_time[:, -1] = True
    return {
        "rgb": rgb,
        "K": K,
        "T_right_from_left": stereo_transform,
        "T_current_from_previous": temporal_transform,
        "T_current_from_previous_valid": temporal_valid,
        "time_valid_mask": time_valid,
        "target_time_mask": target_time,
        "frame_ids": torch.arange(1, frames + 1).reshape(1, frames),
        "timestamps": torch.arange(frames, dtype=torch.float64).reshape(1, frames),
        "manifest_indices": torch.arange(frames).reshape(1, frames),
        "baseline_m": torch.full((1, frames), 0.1),
    }


def _system() -> tuple[
    MetricStereoVideoSystem,
    _CountingStereoWrapper,
    TrainableVGGTOmega,
    _RecordingGeometry,
]:
    stereo = _CountingStereoWrapper()
    vggt = TrainableVGGTOmega(_DummyVGGT(), geometry_channels=8)
    geometry = _RecordingGeometry()
    return MetricStereoVideoSystem(stereo, vggt, geometry), stereo, vggt, geometry


def test_lr_consistency_uses_its_own_grid_units() -> None:
    left = torch.full((1, 1, 2, 5), 0.5)
    right = torch.full_like(left, 0.5)

    result = left_right_stereo_consistency(left, right)

    assert result.valid_left_mask[0, 0, :, 0].eq(False).all()
    assert result.valid_left_mask[0, 0, :, 1:].all()
    torch.testing.assert_close(
        result.error_px[result.valid_left_mask],
        torch.zeros_like(result.error_px[result.valid_left_mask]),
    )
    torch.testing.assert_close(
        result.confidence_left[result.valid_left_mask],
        torch.ones_like(result.confidence_left[result.valid_left_mask]),
    )


def test_vggt_score_conversion_recovers_sigmoid_probability() -> None:
    score = torch.tensor([1.0, 2.0, 5.0, float("nan")])
    probability = vggt_unbounded_confidence_to_probability(score)
    torch.testing.assert_close(probability, torch.tensor([0.0, 0.5, 0.8, 0.0]))


def test_tensor_condition_preserves_cpu_value_error() -> None:
    _assert_tensor_condition(torch.tensor(True), "valid")
    with pytest.raises(ValueError, match="invalid batch"):
        _assert_tensor_condition(torch.tensor(False), "invalid batch")


def test_tensor_condition_routes_cuda_to_async_assert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    calls: list[tuple[torch.Tensor, str]] = []
    monkeypatch.setattr(
        torch,
        "_assert_async",
        lambda condition, message: calls.append((condition, message)),
    )
    with FakeTensorMode():
        false_cuda_condition = torch.zeros((), dtype=torch.bool, device="cuda")
        _assert_tensor_condition(false_cuda_condition, "async invalid batch")

    assert len(calls) == 1
    assert calls[0][0] is false_cuda_condition
    assert calls[0][1] == "async invalid batch"


def test_system_runs_each_backbone_once_and_never_injects_endpoint_into_history() -> None:
    system, stereo, vggt, geometry = _system()
    system.eval()

    with torch.no_grad():
        output = system(_batch())

    assert stereo.forward_calls == 1
    assert vggt.model.aggregator.calls == 1
    assert vggt.model.dense_head.calls == 1
    assert geometry.vggt_contexts == [(0,), (1,), (0, 1, 2)]
    assert geometry.vggt_confidence_sums[:2] == [0.0, 0.0]
    assert geometry.vggt_confidence_sums[2] > 0.0

    assert output.inverse_depth_m_inv.shape == (1, 1, 64, 64)
    assert output.disparity_right_px is not None
    assert output.disparity_right_px.shape == (1, 1, 64, 64)
    # Dummy FFS predicts 0.5 LR pixels; the integration owns the x2 conversion.
    torch.testing.assert_close(
        output.stereo.disparity_left_hr_px_lr_grid,
        torch.ones_like(output.stereo.disparity_left_hr_px_lr_grid),
    )
    torch.testing.assert_close(
        output.disparity_right_px, torch.ones_like(output.disparity_right_px)
    )
    assert len(output.ffs_iteration_disparities_left_hr_px_lr_grid) == 2
    torch.testing.assert_close(
        output.ffs_iteration_disparities_left_hr_px_lr_grid[0],
        torch.full_like(output.stereo.disparity_left_hr_px_lr_grid, 0.5),
    )
    torch.testing.assert_close(
        output.ffs_iteration_disparities_left_hr_px_lr_grid[1],
        torch.ones_like(output.stereo.disparity_left_hr_px_lr_grid),
    )
    assert bool(output.gauge.valid_mask.all())
    predictions = output.as_loss_predictions()
    assert set(predictions) == {
        "inverse_depth_m_inv",
        "inverse_depth_pyramid_m_inv",
        "disparity_left_px",
        "disparity_right_px",
        "valid_logits",
        "log_variance",
    }
    assert predictions["inverse_depth_pyramid_m_inv"][0].shape == (
        1,
        1,
        32,
        32,
    )


def test_system_debug_phase_tracing_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("METRIC_STEREO_DEBUG_PHASES", raising=False)
    monkeypatch.setenv("RANK", "5")
    system, *_ = _system()
    system.eval()

    with torch.no_grad():
        system(_batch(frames=2))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_system_debug_phase_tracing_reports_rank_and_forward_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("METRIC_STEREO_DEBUG_PHASES", "1")
    monkeypatch.setenv("RANK", "5")
    system, *_ = _system()
    system.eval()

    with torch.no_grad():
        system(_batch(frames=2))

    assert capsys.readouterr().out.splitlines() == [
        "[rank=5 phase=system_forward_start]",
        "[rank=5 phase=batch_validation_end]",
        "[rank=5 phase=stereo_resize_end]",
        "[rank=5 phase=stereo_start]",
        "[rank=5 phase=stereo_end]",
        "[rank=5 phase=vggt_start]",
        "[rank=5 phase=vggt_end]",
        "[rank=5 phase=geometry_forward_step_start time_index=0]",
        "[rank=5 phase=geometry_forward_step_end time_index=0]",
        "[rank=5 phase=geometry_forward_step_start time_index=1]",
        "[rank=5 phase=geometry_forward_step_end time_index=1]",
    ]


def test_system_keeps_gradient_paths_to_both_online_backbones() -> None:
    system, stereo, vggt, _geometry = _system()
    system.train()

    output = system(_batch(frames=2))
    loss = (
        output.inverse_depth_m_inv.mean()
        + output.valid_logits.square().mean()
        + output.disparity_right_px.mean()  # type: ignore[union-attr]
    )
    loss.backward()

    assert stereo.model.disparity_lr_px.grad is not None
    assert stereo.model.feature.projection.weight.grad is not None
    assert vggt.model.aggregator.projection.weight.grad is not None
    assert vggt.model.dense_head.depth_scale.grad is not None
    assert bool(torch.isfinite(vggt.model.dense_head.depth_scale.grad))


def test_system_rejects_left_padding_without_a_vggt_attention_mask() -> None:
    system, _stereo, _vggt, _geometry = _system()

    with pytest.raises(ValueError, match="length-bucketed"):
        system(_batch(padded=True))


def test_system_rejects_noncanonical_rectification_for_disparity_output() -> None:
    system, _stereo, _vggt, _geometry = _system()
    unequal_intrinsics = _batch()
    unequal_intrinsics["K"][:, :, 1, 0, 2] += 1.0
    with pytest.raises(ValueError, match="matching rectified"):
        system(unequal_intrinsics)

    vertical_baseline = _batch()
    vertical_baseline["T_right_from_left"][:, :, 1, 3] = 0.01
    with pytest.raises(ValueError, match="x-only"):
        system(vertical_baseline)


def test_system_rejects_noncausal_endpoint_or_provenance() -> None:
    system, _stereo, _vggt, _geometry = _system()
    wrong_target = _batch()
    wrong_target["target_time_mask"][:] = False
    wrong_target["target_time_mask"][:, -2] = True
    with pytest.raises(ValueError, match="only the final frame"):
        system(wrong_target)

    nonmonotonic = _batch()
    nonmonotonic["timestamps"][:, -1] = nonmonotonic["timestamps"][:, -2]
    with pytest.raises(ValueError, match="strictly increasing"):
        system(nonmonotonic)


def test_component_ablation_flags_disable_only_the_requested_paths() -> None:
    system, _stereo, _vggt, geometry = _system()
    system.enable_vggt_features = False
    geometry.enable_vggt_gauge = False
    geometry.enable_temporal_memory = False
    system.eval()
    with torch.no_grad():
        output = system(_batch(frames=3))
    assert output.gauge.valid_mask.eq(False).all()
    assert output.endpoint.temporal.used_history is False


def test_visibility_gate_ablation_retains_zbuffer_diagnostics() -> None:
    system, _stereo, _vggt, geometry = _system()
    geometry.visibility_aware_gating = False
    system.eval()
    with torch.no_grad():
        output = system(_batch(frames=3))
    assert output.endpoint.temporal.used_history is True
    assert output.endpoint.temporal.zbuffer_visible_mask.shape == (
        *output.endpoint.state.inverse_depth_m_inv.shape,
    )
