"""Causal calibration inputs for the opt-in architecture-v3 conditioner."""

from __future__ import annotations

import torch
from torch import Tensor


TEMPORAL_CONDITIONING_AGES = (1, 2)
CURRENT_LEFT_VIEW_INDEX = 8


def _identity_camera_transform_3x4(
    batch_size: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    value = torch.zeros(batch_size, 3, 4, device=device, dtype=dtype)
    value[:, :3, :3] = torch.eye(3, device=device, dtype=dtype)
    return value


def _homogeneous(extrinsics_3x4: Tensor) -> Tensor:
    batch_size = extrinsics_3x4.shape[0]
    value = torch.zeros(
        batch_size,
        4,
        4,
        device=extrinsics_3x4.device,
        dtype=extrinsics_3x4.dtype,
    )
    value[:, :3] = extrinsics_3x4
    value[:, 3, 3] = 1.0
    return value


def temporal_conditioning_transforms(
    extrinsics_camera_from_world_metric: Tensor,
    pose_valid: Tensor,
    *,
    student_time_index: int,
) -> tuple[Tensor, Tensor]:
    """Return age-1/age-2 history-camera to current-camera transforms.

    Args:
        extrinsics_camera_from_world_metric: Current VGGT window poses
            ``[B,10,3,4]`` in metres. Views are ordered
            ``L[t-4],R[t-4],...,L[t],R[t]``.
        pose_valid: Strict window gate ``[B]``.
        student_time_index: Causal student index in ``{0,1,2}``. An age is
            exposed only when the corresponding student memory exists.

    Returns:
        ``(T_current_from_history_m, valid_mask)`` with shapes
        ``[B,2,4,4]`` and ``[B,2]``. Invalid slots contain identity, never a
        singular zero transform. Geometry is detached and computed in FP32.
    """

    if not isinstance(extrinsics_camera_from_world_metric, Tensor) or not (
        extrinsics_camera_from_world_metric.is_floating_point()
    ):
        raise TypeError("extrinsics_camera_from_world_metric must be floating point")
    if extrinsics_camera_from_world_metric.ndim != 4 or tuple(
        extrinsics_camera_from_world_metric.shape[1:]
    ) != (10, 3, 4):
        raise ValueError(
            "extrinsics_camera_from_world_metric must have shape [B,10,3,4]"
        )
    if not isinstance(pose_valid, Tensor) or pose_valid.dtype != torch.bool:
        raise TypeError("pose_valid must be a bool Tensor")
    batch_size = extrinsics_camera_from_world_metric.shape[0]
    if tuple(pose_valid.shape) != (batch_size,):
        raise ValueError("pose_valid must have shape [B]")
    if pose_valid.device != extrinsics_camera_from_world_metric.device:
        raise ValueError("pose_valid and extrinsics must share a device")
    if (
        isinstance(student_time_index, bool)
        or not isinstance(student_time_index, int)
        or student_time_index not in (0, 1, 2)
    ):
        raise ValueError("student_time_index must be 0, 1, or 2")

    # This helper is called inside the BF16 training autocast region.  Merely
    # converting inputs to float32 is insufficient because autocast would cast
    # the matrix products back to BF16.  Keep the entire pose composition in a
    # real FP32 island before the model validates/encodes it.
    with torch.autocast(
        device_type=extrinsics_camera_from_world_metric.device.type,
        enabled=False,
    ):
        extrinsics = extrinsics_camera_from_world_metric.detach().to(torch.float32)
        if not bool(torch.isfinite(extrinsics).all()):
            raise ValueError("temporal extrinsics contain NaN or infinity")
        identity = _identity_camera_transform_3x4(
            batch_size, device=extrinsics.device, dtype=extrinsics.dtype
        )
        window_valid = pose_valid.detach()
        current = torch.where(
            window_valid.reshape(batch_size, 1, 1),
            extrinsics[:, CURRENT_LEFT_VIEW_INDEX],
            identity,
        )
        current_h = _homogeneous(current)
        identity_h = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype)
        identity_h = identity_h.expand(batch_size, -1, -1)
        outputs: list[Tensor] = []
        masks: list[Tensor] = []
        for age in TEMPORAL_CONDITIONING_AGES:
            slot_valid = window_valid & (student_time_index >= age)
            history_index = CURRENT_LEFT_VIEW_INDEX - 2 * age
            history = torch.where(
                slot_valid.reshape(batch_size, 1, 1),
                extrinsics[:, history_index],
                identity,
            )
            relative = current_h @ torch.linalg.inv(_homogeneous(history))
            # Matrix inversion in FP32 can leave a few-epsilon residue in the
            # homogeneous bottom row (especially for large world
            # translations).  The mathematical camera transform is rigid;
            # canonicalize that row before the strict conditioner validation
            # rather than making otherwise valid GT/VGGT poses fail
            # nondeterministically at ~1e-6.
            relative = relative.clone()
            relative[:, 3] = relative.new_tensor((0.0, 0.0, 0.0, 1.0))
            outputs.append(
                torch.where(
                    slot_valid.reshape(batch_size, 1, 1),
                    relative,
                    identity_h,
                )
            )
            masks.append(slot_valid)
        return torch.stack(outputs, dim=1), torch.stack(masks, dim=1)


def rectified_stereo_transform_4x4(
    transform_right_from_left_m: Tensor,
) -> Tensor:
    """Validate the exact FP32 ``[B,4,4]`` v3 rig input."""

    if not isinstance(transform_right_from_left_m, Tensor) or not (
        transform_right_from_left_m.is_floating_point()
    ):
        raise TypeError("rectified stereo transform must be floating point")
    if transform_right_from_left_m.dtype != torch.float32:
        raise TypeError("rectified stereo transform must have dtype torch.float32")
    if transform_right_from_left_m.ndim != 3 or tuple(
        transform_right_from_left_m.shape[-2:]
    ) != (4, 4):
        raise ValueError("rectified stereo transform must have shape [B,4,4]")
    value = transform_right_from_left_m.detach()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("rectified stereo transform contains NaN or infinity")
    expected = value.new_tensor((0.0, 0.0, 0.0, 1.0))
    if not bool(
        torch.isclose(
            value[:, 3], expected.expand(value.shape[0], -1), atol=1e-6, rtol=0.0
        ).all()
    ):
        raise ValueError("rectified stereo transform has a malformed homogeneous row")
    return value.contiguous()


__all__ = [
    "TEMPORAL_CONDITIONING_AGES",
    "rectified_stereo_transform_4x4",
    "temporal_conditioning_transforms",
]
