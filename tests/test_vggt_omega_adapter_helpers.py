from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from backbones.vggt_omega_adapter import (
    VGGTOmegaAdapter,
    balanced_target_shape,
    center_crop_supported_aspect_ratio,
    max_size_target_shape,
    preprocess_coordinate_transforms,
    preprocess_vggt_omega_images,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_pattern(path: Path, *, width: int, height: int) -> None:
    y, x = np.meshgrid(
        np.arange(height, dtype=np.uint8),
        np.arange(width, dtype=np.uint8),
        indexing="ij",
    )
    image = np.stack((x, y, x ^ y), axis=-1)
    Image.fromarray(image, mode="RGB").save(path)


def _load_upstream_preprocessor() -> Any:
    """Load only upstream load_fn.py, avoiding its model/package imports."""

    path = (
        PROJECT_ROOT
        / "third_party"
        / "vggt-omega"
        / "vggt_omega"
        / "utils"
        / "load_fn.py"
    )
    spec = importlib.util.spec_from_file_location("pinned_vggt_omega_load_fn", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_shape_helpers_match_pinned_upstream() -> None:
    upstream = _load_upstream_preprocessor()
    for aspect_ratio in (0.5, 0.75, 1.0, 1.5, 2.0):
        assert balanced_target_shape(aspect_ratio, 512, 16) == upstream._balanced_target_shape(
            aspect_ratio, 512, 16
        )
        assert max_size_target_shape(aspect_ratio, 512, 16) == upstream._max_size_target_shape(
            aspect_ratio, 512, 16
        )


def test_extreme_aspect_crop_resize_group_pad_and_transforms(tmp_path: Path) -> None:
    landscape = tmp_path / "landscape.png"
    portrait = tmp_path / "portrait.png"
    _write_pattern(landscape, width=100, height=20)
    _write_pattern(portrait, width=20, height=100)

    assert center_crop_supported_aspect_ratio(100, 20) == (30, 0, 70, 20)
    assert center_crop_supported_aspect_ratio(20, 100) == (0, 30, 20, 70)

    with pytest.warns(UserWarning, match="padding to a common size"):
        result = preprocess_vggt_omega_images(
            [landscape, portrait],
            mode="balanced",
            image_resolution=32,
            patch_size=16,
        )

    assert result.images.shape == (2, 3, 48, 48)
    landscape_meta, portrait_meta = result.metadata
    assert landscape_meta.original_size_hw == (20, 100)
    assert landscape_meta.crop_xyxy == (30, 0, 70, 20)
    assert landscape_meta.cropped_size_hw == (20, 40)
    assert landscape_meta.resized_size_hw == (16, 48)
    assert landscape_meta.resize_scale_xy == pytest.approx((1.2, 0.8))
    assert landscape_meta.padding_lrtb == (0, 0, 16, 16)
    assert portrait_meta.crop_xyxy == (0, 30, 20, 70)
    assert portrait_meta.resized_size_hw == (48, 16)
    assert portrait_meta.padding_lrtb == (16, 16, 0, 0)

    # Group padding is white, not black or replicated image content.
    torch.testing.assert_close(result.images[0, :, :16], torch.ones(3, 16, 48))
    torch.testing.assert_close(result.images[0, :, 32:], torch.ones(3, 16, 48))
    torch.testing.assert_close(result.images[1, :, :, :16], torch.ones(3, 48, 16))
    torch.testing.assert_close(result.images[1, :, :, 32:], torch.ones(3, 48, 16))

    original_to_model = np.asarray(landscape_meta.original_to_model_3x3)
    model_to_original = np.asarray(landscape_meta.model_to_original_3x3)
    np.testing.assert_allclose(model_to_original @ original_to_model, np.eye(3))
    np.testing.assert_allclose(
        original_to_model @ np.asarray((30.0, 0.0, 1.0)),
        np.asarray((0.0, 16.0, 1.0)),
    )
    np.testing.assert_allclose(
        original_to_model @ np.asarray((70.0, 20.0, 1.0)),
        np.asarray((48.0, 32.0, 1.0)),
    )


def test_preprocessor_pixels_match_pinned_upstream(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_pattern(image_a, width=90, height=30)
    _write_pattern(image_b, width=30, height=75)

    upstream = _load_upstream_preprocessor()
    paths = [str(image_a), str(image_b)]
    with pytest.warns(UserWarning):
        ours = preprocess_vggt_omega_images(
            paths, mode="max_size", image_resolution=64, patch_size=16
        ).images
    with pytest.warns(UserWarning):
        expected = upstream.load_and_preprocess_images(
            paths, mode="max_size", image_resolution=64, patch_size=16
        )
    torch.testing.assert_close(ours, expected, rtol=0.0, atol=0.0)


def test_coordinate_transform_handles_odd_symmetric_padding() -> None:
    original_to_model, model_to_original = preprocess_coordinate_transforms(
        (3, 5, 13, 25),
        (40, 30),
        (2, 3, 4, 5),
    )
    forward = np.asarray(original_to_model)
    inverse = np.asarray(model_to_original)
    np.testing.assert_allclose(inverse @ forward, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        forward @ np.asarray((3.0, 5.0, 1.0)),
        np.asarray((2.0, 4.0, 1.0)),
    )


class _FakeVGGTOmega(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.grad_enabled_during_call: bool | None = None
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        self.grad_enabled_during_call = torch.is_grad_enabled()
        self.input_shape = tuple(images.shape)
        sequence_length, _, height, width = images.shape
        pose_enc = torch.zeros(1, sequence_length, 9, device=images.device)
        pose_enc[..., 3] = 1.0
        return {
            "depth": torch.full(
                (1, sequence_length, height, width, 1),
                2.0,
                device=images.device,
            ),
            # Deliberately greater than one: this must not be probability-clamped.
            "depth_conf": torch.full(
                (1, sequence_length, height, width),
                3.5,
                device=images.device,
            ),
            "pose_enc": pose_enc,
            "camera_and_register_tokens": torch.arange(
                sequence_length * 17 * 8,
                dtype=torch.float32,
                device=images.device,
            ).reshape(1, sequence_length, 17, 8),
        }


def _fake_pose_decoder(
    pose_enc: Tensor,
    image_size_hw: tuple[int, int],
    *,
    build_intrinsics: bool,
) -> tuple[Tensor, Tensor]:
    assert build_intrinsics
    batch, sequence_length, _ = pose_enc.shape
    height, width = image_size_hw
    extrinsics = torch.zeros(
        batch, sequence_length, 3, 4, dtype=pose_enc.dtype, device=pose_enc.device
    )
    extrinsics[..., :3, :3] = torch.eye(3, device=pose_enc.device)
    intrinsics_pred = torch.zeros(
        batch, sequence_length, 3, 3, dtype=pose_enc.dtype, device=pose_enc.device
    )
    intrinsics_pred[..., 0, 0] = width
    intrinsics_pred[..., 1, 1] = height
    intrinsics_pred[..., 0, 2] = width / 2
    intrinsics_pred[..., 1, 2] = height / 2
    intrinsics_pred[..., 2, 2] = 1
    return extrinsics, intrinsics_pred


def test_adapter_freezes_splits_tokens_and_preserves_calibrated_k(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(10):
        path = tmp_path / f"{index:02d}.png"
        _write_pattern(path, width=32, height=32)
        paths.append(path)

    model = _FakeVGGTOmega()
    adapter = VGGTOmegaAdapter(
        model,
        pose_decoder=_fake_pose_decoder,
        image_resolution=32,
        patch_size=16,
        context_pairs=5,
    )
    adapter.train(True)
    calibrated_k = torch.tensor(
        [[20.0, 0.0, 15.0], [0.0, 21.0, 14.0], [0.0, 0.0, 1.0]]
    ).repeat(10, 1, 1)
    output = adapter(paths, intrinsics_calibrated_original=calibrated_k)

    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.grad_enabled_during_call is False
    assert model.input_shape == (10, 3, 32, 32)
    assert output.depth.shape == (1, 10, 32, 32, 1)
    assert output.depth_conf.min().item() == 3.5
    assert output.depth_conf.max().item() == 3.5
    assert output.camera_tokens.shape == (1, 10, 1, 8)
    assert output.register_tokens.shape == (1, 10, 16, 8)
    torch.testing.assert_close(
        output.camera_and_register_tokens,
        torch.arange(10 * 17 * 8, dtype=torch.float32).reshape(1, 10, 17, 8),
    )
    assert output.metadata["extrinsics_convention"] == "OpenCV camera-from-world [R|t]"
    assert "not probability" in output.metadata["depth_conf_semantics"]
    assert "caller input order only" in output.metadata["causality_enforcement"]
    assert output.metadata["intrinsics_pred_usage"].startswith("diagnostic only")
    torch.testing.assert_close(
        output.intrinsics_calibrated_original,
        calibrated_k.unsqueeze(0),
    )
    # Identity preprocessing means calibrated K is unchanged, and it is kept
    # distinct from the intentionally different predicted diagnostic K.
    torch.testing.assert_close(
        output.intrinsics_calibrated_model,
        calibrated_k.unsqueeze(0),
    )
    assert not torch.equal(
        output.intrinsics_calibrated_model,
        output.intrinsics_pred,
    )


def test_adapter_rejects_non_ten_image_default_context(tmp_path: Path) -> None:
    image = tmp_path / "one.png"
    _write_pattern(image, width=32, height=32)
    adapter = VGGTOmegaAdapter(
        _FakeVGGTOmega(),
        pose_decoder=_fake_pose_decoder,
        image_resolution=32,
    )
    with pytest.raises(ValueError, match="exactly 10 images"):
        adapter([image] * 8)
