"""Spatial and causal evaluation primitives with safe metric aggregation.

The reference target is the trusted subset of an HR FFS teacher cache, so
results are pseudo-GT engineering measurements rather than paper accuracy.
Stage-B helpers enforce endpoint-only causal T=3 lineage and HR z-buffer
temporal domains. All disparities passed here are expressed in HR pixels.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from metrics.boundary import boundary_epe
from metrics.disparity import (
    MetricResult,
    disparity_metrics,
    invalid_negative_nan_rate,
    invalid_region_completeness,
    low_confidence_region_epe,
)
from metrics.temporal import (
    legacy_temporal_disparity_error,
    temporal_residual_error,
)
from data.cache_dataset import sha256_file
from utils.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointMismatchError


PSEUDO_GT_LABEL = "trusted_hr_ffs_teacher_pseudo_gt"
V31_PIXEL_CENTER_CONTRACT = "align_corners_false_half_pixel_v3_1"
V31_BEHAVIOR_CONFIG_SECTIONS = (
    "measurement_ownership_v3_1",
    "temporal_candidate_fusion_v3_1",
)
POINT_TO_PLANE_NOT_AVAILABLE = {
    "status": "NOT_AVAILABLE",
    "reason": "target point normals and explicit correspondences are unavailable",
}


def physical_disparity_clamp_min_zero(disparity_hr_px: Tensor) -> Tensor:
    """Map finite negative disparity to zero without filling invalid holes.

    NaN and infinities retain their IEEE semantics. Zero remains an invalid
    disparity in completeness/output-validity metrics; no epsilon is added.
    """

    if not isinstance(disparity_hr_px, Tensor) or not disparity_hr_px.is_floating_point():
        raise TypeError("disparity_hr_px must be a floating-point torch.Tensor")
    finite_negative = torch.isfinite(disparity_hr_px) & (disparity_hr_px < 0.0)
    return torch.where(
        finite_negative,
        torch.zeros_like(disparity_hr_px),
        disparity_hr_px,
    )


@dataclass(frozen=True, slots=True)
class AggregateMetric:
    """A JSON-safe dataset reduction.

    ``value`` and ``numerator`` are ``None`` for an empty domain or when a
    selected non-finite value makes a mean undefined.  ``count`` still records
    the complete selected denominator in the latter case.
    """

    value: float | None
    numerator: float | None
    count: int
    valid: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "value": self.value,
            "numerator": self.numerator,
            "count": self.count,
            "valid": self.valid,
        }


@dataclass(slots=True)
class MetricAccumulator:
    """Combine metric numerators and counts without averaging image means."""

    numerator: float = 0.0
    count: int = 0
    invalid_selected_count: int = 0

    def update(self, result: MetricResult) -> None:
        if not isinstance(result, MetricResult):
            raise TypeError("result must be MetricResult")
        if result.count < 0:
            raise ValueError("metric count must be non-negative")
        if result.valid:
            if result.count <= 0:
                raise ValueError("a valid metric must have a positive count")
            if not torch.isfinite(torch.tensor(result.numerator)):
                raise ValueError("a valid metric must have a finite numerator")
            self.numerator += float(result.numerator)
            self.count += int(result.count)
        elif result.count > 0:
            # The metric implementation uses this state for selected invalid
            # predictions.  Do not silently discard that part of the domain.
            self.invalid_selected_count += int(result.count)

    def finalize(self) -> AggregateMetric:
        total_count = self.count + self.invalid_selected_count
        if total_count == 0:
            return AggregateMetric(None, None, 0, False)
        if self.invalid_selected_count:
            return AggregateMetric(None, None, total_count, False)
        return AggregateMetric(
            value=self.numerator / self.count,
            numerator=self.numerator,
            count=self.count,
            valid=True,
        )


def aggregate_metric_results(results: Iterable[MetricResult]) -> AggregateMetric:
    """Pure convenience wrapper around :class:`MetricAccumulator`."""

    accumulator = MetricAccumulator()
    for result in results:
        accumulator.update(result)
    return accumulator.finalize()


@dataclass(slots=True)
class MethodMetricAccumulator:
    """Named metric accumulators for one evaluated method."""

    metrics: dict[str, MetricAccumulator] = field(default_factory=dict)

    def update(self, sample_metrics: Mapping[str, MetricResult]) -> None:
        for name, result in sample_metrics.items():
            self.metrics.setdefault(name, MetricAccumulator()).update(result)

    def finalize(self) -> dict[str, AggregateMetric]:
        return {name: accumulator.finalize() for name, accumulator in self.metrics.items()}


def upsample_ffs_inputs_to_hr(
    disparity_ffs_hr_px_lr_grid: Tensor,
    confidence_ffs_lr: Tensor,
    valid_ffs_lr: Tensor,
    trusted_ffs_lr: Tensor,
    *,
    output_size_hw: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Upsample LR-grid FFS fields while preserving their semantics.

    Disparity already has HR-pixel units and is bilinearly interpolated without
    an additional scale multiplier. Confidence is also bilinear. Boolean masks
    are nearest-neighbor sampled so validity is never invented by interpolation.
    """

    reference_shape = disparity_ffs_hr_px_lr_grid.shape
    if disparity_ffs_hr_px_lr_grid.ndim != 4 or reference_shape[1] != 1:
        raise ValueError("disparity_ffs_hr_px_lr_grid must be [B,1,H,W]")
    for name, tensor in (
        ("confidence_ffs_lr", confidence_ffs_lr),
        ("valid_ffs_lr", valid_ffs_lr),
        ("trusted_ffs_lr", trusted_ffs_lr),
    ):
        if tensor.shape != reference_shape:
            raise ValueError(f"{name} must have shape {tuple(reference_shape)}")
    safe_disparity = torch.nan_to_num(
        disparity_ffs_hr_px_lr_grid, nan=0.0, posinf=0.0, neginf=0.0
    )
    disparity_hr_px = functional.interpolate(
        safe_disparity,
        size=output_size_hw,
        mode="bilinear",
        align_corners=False,
    )
    confidence_hr = functional.interpolate(
        torch.nan_to_num(confidence_ffs_lr, nan=0.0, posinf=0.0, neginf=0.0),
        size=output_size_hw,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    def nearest_mask(mask: Tensor) -> Tensor:
        return functional.interpolate(
            mask.to(dtype=torch.float32), size=output_size_hw, mode="nearest"
        ).to(dtype=torch.bool)

    return (
        disparity_hr_px,
        confidence_hr,
        nearest_mask(valid_ffs_lr),
        nearest_mask(trusted_ffs_lr),
    )


def compute_sample_metrics(
    prediction_disparity_hr_px: Tensor,
    target_disparity_hr_px: Tensor,
    *,
    target_trusted_mask: Tensor,
    ffs_confidence_hr: Tensor,
    ffs_valid_mask_hr: Tensor,
    ffs_trusted_mask_hr: Tensor,
    low_confidence_threshold: float = 0.8,
    boundary_gradient_threshold_px: float = 1.0,
    boundary_radius_px: int = 1,
) -> dict[str, MetricResult]:
    """Compute one endpoint's spatial metrics on explicit HR-grid domains."""

    if prediction_disparity_hr_px.shape != target_disparity_hr_px.shape:
        raise ValueError("prediction and target disparity shapes must match")
    expected_shape = prediction_disparity_hr_px.shape
    for name, value in (
        ("target_trusted_mask", target_trusted_mask),
        ("ffs_confidence_hr", ffs_confidence_hr),
        ("ffs_valid_mask_hr", ffs_valid_mask_hr),
        ("ffs_trusted_mask_hr", ffs_trusted_mask_hr),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")

    trusted_target = target_trusted_mask.to(dtype=torch.bool)
    core = disparity_metrics(
        prediction_disparity_hr_px,
        target_disparity_hr_px,
        valid_mask=trusted_target,
    )
    validity = invalid_negative_nan_rate(prediction_disparity_hr_px)
    # Boundary construction itself must not inspect untrusted teacher values;
    # merely intersecting the resulting band afterward would allow an
    # untrusted neighbor to create a boundary on a trusted pixel.
    trusted_boundary_target = torch.where(
        trusted_target,
        target_disparity_hr_px,
        torch.full_like(target_disparity_hr_px, float("nan")),
    )
    # Trusted-region degradation is computed dataset-wide from this exact EPE
    # for the baseline and candidate, rather than averaging per-image ratios.
    trusted_ffs_target = trusted_target & ffs_trusted_mask_hr.to(dtype=torch.bool)
    return {
        "epe_px": core.epe_px,
        "bad_1": core.bad_1,
        "bad_2": core.bad_2,
        "boundary_epe_px": boundary_epe(
            prediction_disparity_hr_px,
            trusted_boundary_target,
            valid_mask=trusted_target,
            gradient_threshold_px=boundary_gradient_threshold_px,
            radius_px=boundary_radius_px,
        ),
        "low_confidence_epe_px": low_confidence_region_epe(
            prediction_disparity_hr_px,
            target_disparity_hr_px,
            ffs_confidence_hr,
            confidence_threshold=low_confidence_threshold,
            valid_mask=trusted_target,
        ),
        "invalid_region_completeness": invalid_region_completeness(
            prediction_disparity_hr_px,
            ~ffs_valid_mask_hr.to(dtype=torch.bool),
            eligible_mask=trusted_target,
        ),
        "trusted_region_epe_px": disparity_metrics(
            prediction_disparity_hr_px,
            target_disparity_hr_px,
            valid_mask=trusted_ffs_target,
        ).epe_px,
        "output_invalid_rate": validity.invalid,
        "output_negative_rate": validity.negative,
        "output_nan_rate": validity.nan,
        "output_infinite_rate": validity.infinite,
        "output_zero_rate": validity.zero,
    }


def aggregate_metric_change(
    baseline: Mapping[str, AggregateMetric],
    candidate: Mapping[str, AggregateMetric],
    metric_name: str,
) -> dict[str, Any]:
    """Compare one metric after global numerator/count aggregation."""

    base = baseline[metric_name]
    cand = candidate[metric_name]
    valid = base.valid and cand.valid
    absolute = cand.value - base.value if valid else None  # type: ignore[operator]
    relative_valid = valid and base.value is not None and base.value != 0.0
    relative = (
        100.0 * absolute / base.value  # type: ignore[operator]
        if relative_valid and absolute is not None
        else None
    )
    return {
        "metric": metric_name,
        "baseline": base.to_dict(),
        "candidate": cand.to_dict(),
        "absolute_change": absolute,
        "relative_change_percent": relative,
        "valid": valid,
        "relative_valid": relative_valid,
    }


def comparison_from_aggregates(
    baseline: Mapping[str, AggregateMetric],
    candidate: Mapping[str, AggregateMetric],
) -> dict[str, Any]:
    """Compute aggregate-only spatial go/no-go comparison values."""

    return {
        "trusted_region_degradation": aggregate_metric_change(
            baseline, candidate, "trusted_region_epe_px"
        ),
        "low_confidence_epe_change": aggregate_metric_change(
            baseline, candidate, "low_confidence_epe_px"
        ),
        "invalid_region_completeness_change": aggregate_metric_change(
            baseline,
            candidate,
            "invalid_region_completeness"
        ),
    }


def load_model_for_evaluation(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    expected_parameter_count: int,
    require_full_training_state: bool = False,
) -> dict[str, Any]:
    """Strictly load the model member of a local training checkpoint.

    The training resume loader also requires an optimizer and scheduler, which
    evaluation intentionally does not create.  This loader applies the same
    schema and parameter-count checks before a strict model-state load.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointMismatchError("checkpoint payload is not a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointMismatchError(
            "checkpoint schema mismatch: expected "
            f"{CHECKPOINT_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    for key in ("model", "parameter_count", "step", "config", "git_hash"):
        if key not in payload:
            raise CheckpointMismatchError(f"checkpoint field is missing: {key}")
    if require_full_training_state:
        required_training_fields = {"optimizer", "scheduler", "scaler", "rng_states"}
        missing_training_fields = sorted(required_training_fields.difference(payload))
        if missing_training_fields:
            raise CheckpointMismatchError(
                "formal training checkpoint fields are missing: "
                f"{missing_training_fields}"
            )
    if payload["parameter_count"] != expected_parameter_count:
        raise CheckpointMismatchError(
            "model parameter count mismatch: expected "
            f"{expected_parameter_count}, got {payload['parameter_count']}"
        )
    completed_step = payload["step"]
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or completed_step < 0
    ):
        raise CheckpointMismatchError("checkpoint step is malformed")
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (TypeError, RuntimeError) as exc:
        raise CheckpointMismatchError(
            f"checkpoint model state is incompatible: {exc}"
        ) from exc
    return {
        "path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "step": completed_step,
        "parameter_count": int(payload["parameter_count"]),
        "git_hash": str(payload["git_hash"]),
        "training_config": payload["config"],
    }


def _required_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointMismatchError(f"{name} is missing or malformed")
    return value


def _strict_config_mapping_fingerprint(
    value: Mapping[str, Any], name: str
) -> str:
    """Canonical JSON preserving numeric types for exact behavior matching."""

    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointMismatchError(
            f"{name} must contain only finite JSON values"
        ) from exc


def _identity_mapping(value: object, name: str) -> dict[str, Any]:
    mapping = _required_mapping(value, name)
    fields = {
        "component",
        "upstream_commit",
        "checkpoint_sha256",
        "torch_version",
        "cuda_version",
        "config_sha256",
    }
    if set(mapping) != fields:
        raise CheckpointMismatchError(
            f"{name} fields must be exactly {sorted(fields)}, got {sorted(mapping)}"
        )
    return dict(mapping)


def validate_checkpoint_lineage(
    checkpoint_metadata: Mapping[str, Any],
    *,
    required_stage: str,
    observation_cache_identity: Mapping[str, Any],
    teacher_cache_identity: Mapping[str, Any],
    derived_cache_lineage: Mapping[str, Any] | None = None,
    evaluation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate checkpoint stage and cache compatibility for evaluation.

    Train and validation manifests are intentionally disjoint, so a temporal
    checkpoint's derived-cache receipt hash must *not* equal the validation
    receipt.  Compatibility instead requires the exact frozen FFS identities
    and the exact derived-geometry policy/configuration to match.
    """

    if required_stage not in {"spatial", "temporal"}:
        raise ValueError("required_stage must be spatial or temporal")
    config = _required_mapping(
        checkpoint_metadata.get("training_config"), "checkpoint training config"
    )
    data = _required_mapping(config.get("data"), "checkpoint data config")
    train = _required_mapping(config.get("train"), "checkpoint train config")
    model = _required_mapping(config.get("model"), "checkpoint model config")
    vggt = _required_mapping(config.get("vggt"), "checkpoint VGGT config")
    expected_sequence_length = 1 if required_stage == "spatial" else 3
    if data.get("sequence_length") != expected_sequence_length:
        raise CheckpointMismatchError(
            f"{required_stage} checkpoint must have sequence_length="
            f"{expected_sequence_length}, got {data.get('sequence_length')!r}"
        )
    if str(train.get("stage", "spatial")).lower() != required_stage:
        raise CheckpointMismatchError(
            f"checkpoint train.stage is not {required_stage!r}: "
            f"{train.get('stage')!r}"
        )
    saved_observation = _identity_mapping(
        data.get("observation_cache_identity"),
        "checkpoint observation cache identity",
    )
    saved_teacher = _identity_mapping(
        data.get("teacher_cache_identity"), "checkpoint teacher cache identity"
    )
    current_observation = _identity_mapping(
        observation_cache_identity, "evaluation observation cache identity"
    )
    current_teacher = _identity_mapping(
        teacher_cache_identity, "evaluation teacher cache identity"
    )
    if saved_observation != current_observation:
        raise CheckpointMismatchError(
            "evaluation observation cache identity differs from checkpoint lineage"
        )
    if saved_teacher != current_teacher:
        raise CheckpointMismatchError(
            "evaluation teacher cache identity differs from checkpoint lineage"
        )

    saved_calibration = config.get("calibration_conditioning_v3")
    if saved_calibration is None:
        saved_calibration = {
            "enabled": False,
            "protocol_version": "disabled",
            "use_rays": False,
            "use_stereo_pose": False,
            "use_temporal_pose": False,
        }
    saved_calibration = _required_mapping(
        saved_calibration, "checkpoint calibration-v3 config"
    )
    calibration_enabled = saved_calibration.get("enabled") is True
    expected_derived_contract = (
        "calibrated_stereo_v2" if calibration_enabled else "legacy_v1"
    )
    if data.get("derived_contract", "legacy_v1") != expected_derived_contract:
        raise CheckpointMismatchError(
            "checkpoint calibration and derived-cache contracts disagree"
        )
    pixel_center_contract = saved_calibration.get("pixel_center_contract")
    if pixel_center_contract is not None and (
        pixel_center_contract != V31_PIXEL_CENTER_CONTRACT
    ):
        raise CheckpointMismatchError(
            "checkpoint calibration has an unsupported pixel-center contract"
        )
    if (
        pixel_center_contract == V31_PIXEL_CENTER_CONTRACT
        and evaluation_config is None
    ):
        raise CheckpointMismatchError(
            "v3.1 lineage validation requires the resolved evaluation config"
        )
    if evaluation_config is not None:
        current_config = _required_mapping(
            evaluation_config, "resolved evaluation config"
        )
        current_calibration = _required_mapping(
            current_config.get("calibration_conditioning_v3"),
            "evaluation calibration-v3 config",
        )
        if dict(current_calibration) != dict(saved_calibration):
            raise CheckpointMismatchError(
                "evaluation calibration conditioning differs from checkpoint"
            )
        if pixel_center_contract == V31_PIXEL_CENTER_CONTRACT:
            for section_name in V31_BEHAVIOR_CONFIG_SECTIONS:
                saved_section = _required_mapping(
                    config.get(section_name),
                    f"checkpoint {section_name} config",
                )
                current_section = _required_mapping(
                    current_config.get(section_name),
                    f"evaluation {section_name} config",
                )
                if _strict_config_mapping_fingerprint(
                    current_section,
                    f"evaluation {section_name} config",
                ) != _strict_config_mapping_fingerprint(
                    saved_section,
                    f"checkpoint {section_name} config",
                ):
                    raise CheckpointMismatchError(
                        "evaluation v3.1 behavior config differs from checkpoint: "
                        f"{section_name}"
                    )
        current_data = _required_mapping(
            current_config.get("data"), "evaluation data config"
        )
        if current_data.get("derived_contract") != expected_derived_contract:
            raise CheckpointMismatchError(
                "evaluation derived-cache contract differs from checkpoint"
            )
        if calibration_enabled:
            saved_sidecar = _required_mapping(
                data.get("calibration_sidecar_lineage"),
                "checkpoint calibration sidecar lineage",
            )
            current_sidecar = _required_mapping(
                current_data.get("calibration_sidecar_lineage"),
                "evaluation calibration sidecar lineage",
            )
            for name in ("component", "contract_version", "pixel_audit_sha256"):
                if saved_sidecar.get(name) != current_sidecar.get(name):
                    raise CheckpointMismatchError(
                        f"evaluation calibration lineage differs for {name}"
                    )

    result: dict[str, Any] = {
        "stage": required_stage,
        "source_sequence_length": expected_sequence_length,
        "observation_cache_identity": saved_observation,
        "teacher_cache_identity": saved_teacher,
        "calibration_conditioning_v3": dict(saved_calibration),
    }
    if required_stage == "spatial":
        if bool(model.get("use_history", False)) or bool(
            model.get("use_vggt_pose", False)
        ):
            raise CheckpointMismatchError(
                "spatial checkpoint unexpectedly enables temporal history/pose"
            )
        return result

    if data.get("vggt_context_pairs") != 5:
        raise CheckpointMismatchError("temporal checkpoint must use five VGGT pairs")
    if not bool(vggt.get("causal")):
        raise CheckpointMismatchError("temporal checkpoint is not causal")
    temporal_pose_source = str(data.get("temporal_pose_source", "vggt")).strip().lower()
    aliases = {"ground_truth": "gt", "ground-truth": "gt", "manifest": "gt"}
    temporal_pose_source = aliases.get(temporal_pose_source, temporal_pose_source)
    if temporal_pose_source not in {"vggt", "gt"}:
        raise CheckpointMismatchError(
            f"checkpoint temporal_pose_source is unsupported: {temporal_pose_source!r}"
        )
    if not bool(model.get("use_history")):
        raise CheckpointMismatchError("temporal checkpoint must enable history")
    # The selected transport pose is controlled by data.temporal_pose_source.
    # GT-pose arms intentionally leave the VGGT-pose branch disabled.
    if temporal_pose_source == "vggt" and not bool(model.get("use_vggt_pose")):
        raise CheckpointMismatchError(
            "temporal checkpoint with temporal_pose_source='vggt' must enable VGGT pose"
        )
    result["temporal_pose_source"] = temporal_pose_source
    if bool(model.get("epipolar_refinement")):
        raise CheckpointMismatchError(
            "Stage-B checkpoint must not enable Stage-C epipolar refinement"
        )
    if str(train.get("init_from_stage", "")).lower() != "spatial":
        raise CheckpointMismatchError(
            "temporal checkpoint is not initialized from the spatial stage"
        )
    if train.get("history_detach") is not True:
        raise CheckpointMismatchError(
            "temporal checkpoint does not use detached prediction history"
        )
    initialization_path = train.get("initialization_checkpoint")
    initialization_sha256 = train.get("initialization_checkpoint_sha256")
    if not isinstance(initialization_path, str) or not initialization_path:
        raise CheckpointMismatchError(
            "temporal checkpoint has no Stage-A initialization path"
        )
    if (
        not isinstance(initialization_sha256, str)
        or len(initialization_sha256) != 64
        or any(character not in "0123456789abcdef" for character in initialization_sha256)
    ):
        raise CheckpointMismatchError(
            "temporal checkpoint has no valid Stage-A initialization SHA-256"
        )
    saved_derived = _required_mapping(
        data.get("derived_cache_lineage"), "checkpoint derived-cache lineage"
    )
    current_derived = _required_mapping(
        derived_cache_lineage, "evaluation derived-cache lineage"
    )
    expected_derived_component = (
        "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch"
        if calibration_enabled
        else "vggt-ffs-derived-geometry-batch"
    )
    if saved_derived.get("component") != expected_derived_component:
        raise CheckpointMismatchError(
            "checkpoint derived-cache component is incompatible"
        )
    if current_derived.get("component") != saved_derived.get("component"):
        raise CheckpointMismatchError(
            "evaluation derived-cache component differs from checkpoint lineage"
        )
    saved_policy = _required_mapping(
        saved_derived.get("config"), "checkpoint derived-cache policy"
    )
    current_policy = _required_mapping(
        current_derived.get("config"), "evaluation derived-cache policy"
    )
    def comparable_policy(value: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(value)
        # A Spring GT-pose override records the exact source VGGT receipt and
        # cache-manifest hashes in its config.  Those hashes intentionally
        # differ between the train and validation split (each split has its
        # own derived cache), but they do not change the geometry policy or
        # model behavior.  Keep them in the receipt/provenance while omitting
        # them from the cross-split policy equality check.
        for split_specific_key in (
            "source_derived_receipt_sha256",
            "source_derived_manifest_sha256",
        ):
            result.pop(split_specific_key, None)
        calibration = result.get("rectified_stereo_calibration")
        if isinstance(calibration, Mapping):
            result["rectified_stereo_calibration"] = {
                name: calibration.get(name)
                for name in (
                    "component",
                    "contract_version",
                    "pixel_audit_sha256",
                )
            }
        return result

    if comparable_policy(saved_policy) != comparable_policy(current_policy):
        raise CheckpointMismatchError(
            "evaluation derived-geometry policy differs from checkpoint lineage"
        )
    active_inference_fields = {
        "data.scale": (data, "scale"),
        "data.sequence_length": (data, "sequence_length"),
        "data.vggt_context_pairs": (data, "vggt_context_pairs"),
        "model.rgb_channels": (model, "rgb_channels"),
        "model.geometry_channels": (model, "geometry_channels"),
        "model.hidden_dim": (model, "hidden_dim"),
        "model.gru_layers": (model, "gru_layers"),
        "model.residual_limit_hr_px": (model, "residual_limit_hr_px"),
        "model.convex_scale": (model, "convex_scale"),
        "model.use_history": (model, "use_history"),
        "model.use_vggt_pose": (model, "use_vggt_pose"),
        "vggt.causal": (vggt, "causal"),
        "train.history_detach": (train, "history_detach"),
        "train.photometric_temperature": (train, "photometric_temperature"),
        "train.disparity_temperature_hr_px": (
            train,
            "disparity_temperature_hr_px",
        ),
        "train.history_conflict_hr_px": (train, "history_conflict_hr_px"),
        "train.temporal_photometric_threshold": (
            train,
            "temporal_photometric_threshold",
        ),
        "train.temporal_geometry_threshold_hr_px": (
            train,
            "temporal_geometry_threshold_hr_px",
        ),
    }
    if evaluation_config is None:
        raise CheckpointMismatchError(
            "temporal lineage validation requires the resolved evaluation config"
        )
    current_config = _required_mapping(
        evaluation_config, "resolved evaluation config"
    )
    for dotted_name, (saved_section, field_name) in active_inference_fields.items():
        section_name, _ = dotted_name.split(".", maxsplit=1)
        current_section = _required_mapping(
            current_config.get(section_name),
            f"evaluation {section_name} config",
        )
        saved_value = saved_section.get(field_name)
        current_value = current_section.get(field_name)
        if saved_value != current_value:
            raise CheckpointMismatchError(
                f"inference config mismatch for {dotted_name}: checkpoint "
                f"{saved_value!r}, evaluation {current_value!r}"
            )
    result.update(
        {
            "derived_geometry_policy": dict(saved_policy),
            "stage_a_initialization_path": initialization_path,
            "stage_a_initialization_sha256": initialization_sha256,
        }
    )
    return result


def validate_spatial_checkpoint_binding(
    spatial_checkpoint_metadata: Mapping[str, Any],
    temporal_checkpoint_lineage: Mapping[str, Any],
) -> None:
    """Require the T1 evaluator to use the exact Stage-A initialization."""

    expected = temporal_checkpoint_lineage.get("stage_a_initialization_sha256")
    actual = spatial_checkpoint_metadata.get("checkpoint_sha256")
    if not isinstance(expected, str) or actual != expected:
        raise CheckpointMismatchError(
            "T1 spatial checkpoint SHA-256 does not match the temporal "
            "checkpoint's Stage-A initialization lineage"
        )


def validate_temporal_batch_causality(batch: Mapping[str, Any]) -> dict[str, int]:
    """Defense-in-depth checks for three endpoint-derived causal frames.

    The dataset already performs these validations while loading each record.
    Evaluation repeats the cheap metadata checks before moving a batch to the
    accelerator, making accidental future-frame or mixed-crop evaluation an
    explicit failure rather than an undocumented metric change.
    """

    frame_ids = batch.get("frame_ids")
    timestamps = batch.get("timestamps")
    manifest_indices = batch.get("manifest_indices")
    metadata = batch.get("identity_metadata")
    pose_valid = batch.get("temporal_pose_valid_sequence")
    pose_quality = batch.get("temporal_pose_quality_score_sequence")
    prior_valid = batch.get("static_prior_valid_sequence")
    if not all(
        isinstance(value, Tensor)
        for value in (
            frame_ids,
            timestamps,
            manifest_indices,
            pose_valid,
            prior_valid,
        )
    ):
        raise ValueError("temporal batch causal tensors are missing")
    if frame_ids.ndim != 2 or frame_ids.shape[1] != 3:
        raise ValueError("temporal frame_ids must have shape [B,3]")
    expected_shape = frame_ids.shape
    for name, value in (
        ("timestamps", timestamps),
        ("manifest_indices", manifest_indices),
        ("temporal_pose_valid_sequence", pose_valid),
        ("static_prior_valid_sequence", prior_valid),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")
    if pose_quality is not None:
        if not isinstance(pose_quality, Tensor) or pose_quality.shape != expected_shape:
            raise ValueError(
                "temporal_pose_quality_score_sequence must have shape "
                f"{tuple(expected_shape)}"
            )
        if (
            not pose_quality.is_floating_point()
            or not bool(torch.isfinite(pose_quality).all().item())
            or bool(((pose_quality < 0) | (pose_quality > 1)).any().item())
            or bool((pose_valid & (pose_quality <= 0)).any().item())
            or bool(((~pose_valid) & (pose_quality != 0)).any().item())
        ):
            raise ValueError(
                "temporal pose quality must be in (0,1] for valid poses and "
                "zero for rejected poses"
            )
    if not bool((timestamps[:, 1:] > timestamps[:, :-1]).all().item()):
        raise ValueError("temporal batch contains non-increasing/future timestamps")
    if not bool(torch.isfinite(timestamps).all().item()):
        raise ValueError("temporal batch timestamps must be finite")
    if not bool((frame_ids[:, 1:] > frame_ids[:, :-1]).all().item()):
        raise ValueError("temporal frame IDs must be strictly increasing")
    if not bool((manifest_indices[:, 1:] > manifest_indices[:, :-1]).all().item()):
        raise ValueError("temporal batch manifest indices are not causal")
    if not isinstance(metadata, list) or len(metadata) != frame_ids.shape[0]:
        raise ValueError("temporal identity_metadata does not match batch size")

    for batch_index, item in enumerate(metadata):
        if not isinstance(item, Mapping):
            raise ValueError("temporal identity metadata item is malformed")
        student_indices = item.get("student_manifest_indices")
        vggt_indices = item.get("vggt_context_manifest_indices")
        endpoint = item.get("endpoint_manifest_index")
        expected_student = [int(value) for value in manifest_indices[batch_index].tolist()]
        if student_indices != expected_student or endpoint != expected_student[-1]:
            raise ValueError("student window metadata does not match batch endpoint")
        if (
            not isinstance(vggt_indices, list)
            or len(vggt_indices) != 5
            or vggt_indices[-1] != endpoint
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > endpoint
                for value in vggt_indices
            )
            or any(
                right <= left
                for left, right in zip(vggt_indices, vggt_indices[1:])
            )
        ):
            raise ValueError("VGGT metadata is not a five-pair causal context")
        per_time_ffs = item.get("per_time_ffs")
        per_time_derived = item.get("per_time_derived")
        if not isinstance(per_time_ffs, list) or len(per_time_ffs) != 3:
            raise ValueError("each temporal window needs three FFS endpoint records")
        if not isinstance(per_time_derived, list) or len(per_time_derived) != 3:
            raise ValueError("each temporal window needs three derived endpoint records")
        shared_crop = item.get("crop_hr_px")
        sequence_ids = batch.get("sequence_id")
        if (
            not isinstance(sequence_ids, list)
            or len(sequence_ids) != frame_ids.shape[0]
            or item.get("sequence_id") != sequence_ids[batch_index]
        ):
            raise ValueError("temporal window sequence metadata is inconsistent")
        if (
            not isinstance(shared_crop, Mapping)
            or shared_crop.get("spatial_scale") != 2
            or any(
                not isinstance(shared_crop.get(name), int)
                or int(shared_crop[name]) % 2
                for name in ("x", "y", "width", "height")
            )
        ):
            raise ValueError("temporal window crop is not a valid x2-aligned crop")
        for time_index in range(3):
            ffs_item = per_time_ffs[time_index]
            derived_item = per_time_derived[time_index]
            if not isinstance(ffs_item, Mapping) or not isinstance(
                derived_item, Mapping
            ):
                raise ValueError("per-time causal lineage entry is malformed")
            record = ffs_item.get("manifest_record")
            if not isinstance(record, Mapping):
                raise ValueError("per-time FFS manifest record is missing")
            expected_frame_id = int(frame_ids[batch_index, time_index].item())
            expected_timestamp = float(timestamps[batch_index, time_index].item())
            if record.get("frame_id") != expected_frame_id or record.get(
                "timestamp"
            ) != expected_timestamp:
                raise ValueError("per-time FFS record does not match causal frame")
            if record.get("sequence_id") != sequence_ids[batch_index]:
                raise ValueError("per-time FFS record crosses a sequence boundary")
            if ffs_item.get("crop_hr_px") != shared_crop:
                raise ValueError("T=3 frames do not share one fixed HR crop")
            cache_path = derived_item.get("cache_path")
            if not isinstance(cache_path, str) or Path(cache_path).stem != str(
                expected_frame_id
            ):
                raise ValueError("derived geometry is not tied to its frame endpoint")
            if not isinstance(derived_item.get("pose_valid"), bool) or bool(
                derived_item["pose_valid"]
            ) != bool(
                pose_valid[batch_index, time_index].item()
            ):
                raise ValueError("derived pose validity differs from batch tensor")
            if not isinstance(
                derived_item.get("static_prior_valid"), bool
            ) or bool(derived_item["static_prior_valid"]) != bool(
                prior_valid[batch_index, time_index].item()
            ):
                raise ValueError("derived prior validity differs from batch tensor")
    return {"batch_size": int(frame_ids.shape[0]), "frames_per_window": 3}


def hr_temporal_safe_mask(
    reference_disparity_hr_px: Tensor,
    *,
    visibility_mask_hr: Tensor,
    static_mask_hr: Tensor,
    collision_mask_hr: Tensor,
    geometry_consistent_mask_hr: Tensor,
    valid_history_hr: Tensor,
) -> Tensor:
    """Build the strict HR z-buffer visible/static evaluation domain."""

    expected_shape = reference_disparity_hr_px.shape
    for name, value in (
        ("visibility_mask_hr", visibility_mask_hr),
        ("static_mask_hr", static_mask_hr),
        ("collision_mask_hr", collision_mask_hr),
        ("geometry_consistent_mask_hr", geometry_consistent_mask_hr),
        ("valid_history_hr", valid_history_hr),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")
    return (
        visibility_mask_hr.to(dtype=torch.bool)
        & static_mask_hr.to(dtype=torch.bool)
        & ~collision_mask_hr.to(dtype=torch.bool)
        & geometry_consistent_mask_hr.to(dtype=torch.bool)
        & valid_history_hr.to(dtype=torch.bool)
    )


def hr_temporal_metric(
    current_disparity_hr_px: Tensor,
    warped_history_disparity_hr_px: Tensor,
    *,
    visibility_mask_hr: Tensor,
    static_mask_hr: Tensor,
    collision_mask_hr: Tensor,
    geometry_consistent_mask_hr: Tensor,
    valid_history_hr: Tensor,
) -> MetricResult:
    """Legacy current/history difference on the strict HR safe domain.

    This function preserves the historical evaluator contract.  It is not the
    v2 TEPE because it has no teacher/GT temporal residual and consequently
    penalizes genuine temporal disparity change.
    """

    if warped_history_disparity_hr_px.shape != current_disparity_hr_px.shape:
        raise ValueError("warped_history_disparity_hr_px shape mismatch")
    safe_mask = hr_temporal_safe_mask(
        current_disparity_hr_px,
        visibility_mask_hr=visibility_mask_hr,
        static_mask_hr=static_mask_hr,
        collision_mask_hr=collision_mask_hr,
        geometry_consistent_mask_hr=geometry_consistent_mask_hr,
        valid_history_hr=valid_history_hr,
    )
    return legacy_temporal_disparity_error(
        current_disparity_hr_px,
        warped_history_disparity_hr_px,
        safe_mask=safe_mask,
    )


def hr_temporal_residual_metric(
    current_prediction_disparity_hr_px: Tensor,
    warped_previous_prediction_disparity_hr_px: Tensor,
    current_reference_disparity_hr_px: Tensor,
    warped_previous_reference_disparity_hr_px: Tensor,
    *,
    visibility_mask_hr: Tensor,
    static_mask_hr: Tensor,
    collision_mask_hr: Tensor,
    geometry_consistent_mask_hr: Tensor,
    valid_prediction_history_hr: Tensor,
    current_reference_valid_mask_hr: Tensor,
    warped_previous_reference_valid_mask_hr: Tensor,
    paired_domain_mask_hr: Tensor | None = None,
) -> MetricResult:
    """V2 teacher/GT temporal-residual error on an explicit HR domain.

    All disparity arguments are ``[B,1,H,W]`` in HR pixels.  The previous
    prediction and reference must already be reprojected into the current
    camera using their respective z-buffer transports.  ``paired_domain_mask``
    optionally intersects otherwise method-native masks, enabling a fair T1
    versus T3 comparison with exactly the same denominator.
    """

    if warped_previous_prediction_disparity_hr_px.shape != (
        current_prediction_disparity_hr_px.shape
    ):
        raise ValueError("warped previous prediction disparity shape mismatch")
    safe_mask = hr_temporal_safe_mask(
        current_prediction_disparity_hr_px,
        visibility_mask_hr=visibility_mask_hr,
        static_mask_hr=static_mask_hr,
        collision_mask_hr=collision_mask_hr,
        geometry_consistent_mask_hr=geometry_consistent_mask_hr,
        valid_history_hr=valid_prediction_history_hr,
    )
    if paired_domain_mask_hr is not None:
        if paired_domain_mask_hr.shape != safe_mask.shape:
            raise ValueError(
                "paired_domain_mask_hr must have shape "
                f"{tuple(safe_mask.shape)}"
            )
        safe_mask &= paired_domain_mask_hr.to(dtype=torch.bool)
    return temporal_residual_error(
        current_prediction_disparity_hr_px,
        warped_previous_prediction_disparity_hr_px,
        current_reference_disparity_hr_px,
        warped_previous_reference_disparity_hr_px,
        safe_mask=safe_mask,
        current_reference_valid_mask=current_reference_valid_mask_hr,
        warped_previous_reference_valid_mask=(
            warped_previous_reference_valid_mask_hr
        ),
    )


__all__ = [
    "AggregateMetric",
    "MethodMetricAccumulator",
    "MetricAccumulator",
    "POINT_TO_PLANE_NOT_AVAILABLE",
    "PSEUDO_GT_LABEL",
    "aggregate_metric_results",
    "aggregate_metric_change",
    "comparison_from_aggregates",
    "compute_sample_metrics",
    "hr_temporal_metric",
    "hr_temporal_residual_metric",
    "hr_temporal_safe_mask",
    "load_model_for_evaluation",
    "physical_disparity_clamp_min_zero",
    "upsample_ffs_inputs_to_hr",
    "validate_checkpoint_lineage",
    "validate_spatial_checkpoint_binding",
    "validate_temporal_batch_causality",
]
