from __future__ import annotations

import math
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from evaluation import (
    AggregateMetric,
    PSEUDO_GT_LABEL,
    aggregate_metric_results,
    comparison_from_aggregates,
    compute_sample_metrics,
    hr_temporal_residual_metric,
    hr_temporal_safe_mask,
    load_model_for_evaluation,
    physical_disparity_clamp_min_zero,
    upsample_ffs_inputs_to_hr,
    validate_checkpoint_lineage,
    validate_temporal_batch_causality,
)
import eval as eval_cli
import train
from data.cache_dataset import sha256_file
from data.manifest import ManifestRecord
from data.training_dataset import build_causal_windows
from metrics.disparity import MetricResult
from test_training_dataset import _identity, _make_cached_example
from utils.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointMismatchError
from models.ffs_omega_tsr import ModelOutput


def test_aggregate_uses_global_numerator_and_count_not_image_means() -> None:
    result = aggregate_metric_results(
        (
            MetricResult(value=1.0, numerator=1.0, count=1, valid=True),
            MetricResult(value=3.0, numerator=9.0, count=3, valid=True),
        )
    )

    assert result.valid
    assert result.numerator == pytest.approx(10.0)
    assert result.count == 4
    assert result.value == pytest.approx(2.5)


def test_aggregate_empty_and_selected_invalid_are_never_zero() -> None:
    empty = aggregate_metric_results((MetricResult.invalid(),))
    assert empty == AggregateMetric(value=None, numerator=None, count=0, valid=False)

    invalid_selected = aggregate_metric_results(
        (
            MetricResult(value=2.0, numerator=4.0, count=2, valid=True),
            MetricResult.invalid(count=3),
        )
    )
    assert not invalid_selected.valid
    assert invalid_selected.count == 5
    assert invalid_selected.value is None
    assert invalid_selected.numerator is None


def test_ffs_upsampling_preserves_hr_units_and_uses_nearest_masks() -> None:
    disparity_hr_px_lr_grid = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    confidence_lr = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    valid_lr = torch.tensor([[[[True, False], [False, True]]]])
    trusted_lr = torch.tensor([[[[False, True], [True, False]]]])

    disparity, confidence, valid, trusted = upsample_ffs_inputs_to_hr(
        disparity_hr_px_lr_grid,
        confidence_lr,
        valid_lr,
        trusted_lr,
        output_size_hw=(4, 4),
    )

    # HR-pixel disparity is interpolated, not multiplied by x2 a second time.
    assert disparity[0, 0, 0, 0].item() == pytest.approx(2.0)
    assert disparity[0, 0, -1, -1].item() == pytest.approx(8.0)
    assert 0.0 < confidence[0, 0, 1, 1].item() < 1.0
    assert valid.dtype == torch.bool and trusted.dtype == torch.bool
    assert valid[0, 0, :2, :2].all()
    assert not valid[0, 0, :2, 2:].any()
    assert trusted[0, 0, :2, 2:].all()
    assert not trusted[0, 0, :2, :2].any()


def test_compute_sample_metrics_uses_trusted_pseudo_gt_domains() -> None:
    target = torch.ones((1, 1, 2, 3)) * 10.0
    prediction = target.clone()
    prediction[..., 0, 0] = 12.0
    target_trusted = torch.tensor([[[[True, True, False], [False, False, False]]]])
    confidence = torch.tensor([[[[0.2, 0.9, 0.1], [0.1, 0.1, 0.1]]]])
    valid_ffs = torch.tensor([[[[True, True, True], [True, True, True]]]])
    trusted_ffs = torch.tensor([[[[True, False, True], [True, True, True]]]])

    metrics = compute_sample_metrics(
        prediction,
        target,
        target_trusted_mask=target_trusted,
        ffs_confidence_hr=confidence,
        ffs_valid_mask_hr=valid_ffs,
        ffs_trusted_mask_hr=trusted_ffs,
        boundary_radius_px=0,
    )

    assert metrics["epe_px"].count == 2
    assert metrics["epe_px"].value == pytest.approx(1.0)
    assert metrics["low_confidence_epe_px"].count == 1
    assert metrics["low_confidence_epe_px"].value == pytest.approx(2.0)
    assert metrics["trusted_region_epe_px"].count == 1
    # No FFS-invalid pixel overlaps trusted pseudo-GT: null domain, not zero.
    assert not metrics["invalid_region_completeness"].valid
    assert metrics["invalid_region_completeness"].count == 0


def _failure_payload() -> dict[str, torch.Tensor]:
    scalar = torch.ones((1, 2, 2), dtype=torch.float32)
    return {
        "rgb_hr": torch.full((3, 2, 2), 0.5),
        "K_hr_px": torch.eye(3),
        "baseline_m": torch.tensor(0.1),
        "baseline_hr_px": scalar.clone(),
        "output_hr_px": scalar.clone(),
        "target_hr_px": scalar.clone(),
        "target_trusted_mask": torch.ones_like(scalar, dtype=torch.bool),
        "source_weights_lr": torch.full((3, 1, 1), 1.0 / 3.0),
        "uncertainty_hr": scalar.clone(),
        "vggt_disparity_hr_px": scalar.clone(),
        "vggt_valid_mask_hr": torch.ones_like(scalar, dtype=torch.bool),
        "history_disparity_hr_px": scalar.clone(),
        "history_valid_mask_hr": torch.ones_like(scalar, dtype=torch.bool),
        "vggt_off_output_hr_px": scalar.clone(),
        "no_vggt_output_hr_px": scalar.clone(),
    }


def test_t3_failure_metrics_are_per_sample_and_keep_metric_receipts() -> None:
    target = torch.tensor([[[[10.0, 20.0]]]])
    prediction = torch.tensor([[[[-1.0, 25.0]]]])
    metrics = eval_cli._t3_failure_sample_metrics(
        prediction_hr_px=prediction,
        target_hr_px=target,
        target_trusted_mask=torch.ones_like(target, dtype=torch.bool),
        ffs_confidence_hr=torch.zeros_like(target),
        ffs_valid_mask_hr=torch.ones_like(target, dtype=torch.bool),
        ffs_trusted_mask_hr=torch.ones_like(target, dtype=torch.bool),
        history_hr_px=torch.tensor([[[[9.0, 22.0]]]]),
        strict_temporal_safe_mask=torch.tensor([[[[True, False]]]]),
        low_confidence_threshold=0.8,
        boundary_gradient_threshold_px=1.0,
        boundary_radius_px=0,
    )
    assert metrics["raw_negative_rate"] == MetricResult(0.5, 1.0, 2, True)
    assert metrics["low_confidence_epe_px"] == MetricResult(8.0, 16.0, 2, True)
    assert metrics["boundary_epe_px"] == MetricResult(8.0, 16.0, 2, True)
    assert metrics["strict_temporal_error_px"] == MetricResult(10.0, 10.0, 1, True)


def test_t3_failure_metrics_use_teacher_residual_name_and_bias_cancellation() -> None:
    target = torch.tensor([[[[10.0, 20.0]]]])
    prediction = target + 4.0
    warped_target = torch.tensor([[[[9.0, 18.0]]]])
    warped_prediction = warped_target + 4.0
    valid = torch.ones_like(target, dtype=torch.bool)
    metrics = eval_cli._t3_failure_sample_metrics(
        prediction_hr_px=prediction,
        target_hr_px=target,
        target_trusted_mask=valid,
        ffs_confidence_hr=torch.zeros_like(target),
        ffs_valid_mask_hr=valid,
        ffs_trusted_mask_hr=valid,
        history_hr_px=warped_prediction,
        strict_temporal_safe_mask=valid,
        warped_reference_history_hr_px=warped_target,
        warped_reference_valid_mask=valid,
        low_confidence_threshold=0.8,
        boundary_gradient_threshold_px=1.0,
        boundary_radius_px=0,
    )
    assert "strict_temporal_error_px" not in metrics
    assert metrics["strict_temporal_residual_error_px"] == MetricResult(
        0.0, 0.0, 2, True
    )


def test_temporal_metric_contract_separates_v2_primary_from_legacy() -> None:
    v2 = eval_cli._temporal_metric_contract(temporal_metric_v2=True)
    legacy = eval_cli._temporal_metric_contract(temporal_metric_v2=False)
    assert v2["primary"] is True
    assert v2["paired_field"] == "temporal_residual_error_paired_px"
    assert v2["legacy_fields_emitted"] is False
    assert legacy["primary"] is False
    assert legacy["paired_field"] == "temporal_disparity_error_paired_px"
    assert "not teacher/GT" in legacy["warning"]


def test_v31_per_record_writer_keeps_all_topk_keys_and_null_invalid(
    tmp_path: Path,
) -> None:
    invalid_name = eval_cli.TOPK_V31_DIAGNOSTIC_NAMES[-1]
    metrics = {
        "ordinary_valid": MetricResult(2.0, 2.0, 1, True),
        "ordinary_invalid": MetricResult.invalid(),
        **{
            f"topk_{name}": (
                MetricResult.invalid()
                if name == invalid_name
                else MetricResult(0.5, 0.5, 1, True)
            )
            for name in eval_cli.TOPK_V31_DIAGNOSTIC_NAMES
        },
    }

    serialized = eval_cli._per_record_metric_values(
        metrics,
        require_topk_v31=True,
    )

    assert serialized["ordinary_valid"] == pytest.approx(2.0)
    assert "ordinary_invalid" not in serialized
    assert {
        name for name in serialized if name.startswith("topk_")
    } == {
        f"topk_{name}" for name in eval_cli.TOPK_V31_DIAGNOSTIC_NAMES
    }
    assert serialized[f"topk_{invalid_name}"] is None
    assert all(
        serialized[f"topk_{name}"] == pytest.approx(0.5)
        for name in eval_cli.TOPK_V31_DIAGNOSTIC_NAMES
        if name != invalid_name
    )
    path = tmp_path / "per_record.jsonl"
    eval_cli._write_jsonl_atomic(path, [{"metrics": serialized}])
    raw = path.read_text(encoding="utf-8")
    assert f'"topk_{invalid_name}": null' in raw
    decoded = json.loads(raw)
    assert decoded["metrics"][f"topk_{invalid_name}"] is None


def test_v31_per_record_writer_fails_closed_on_missing_topk_key() -> None:
    metrics = {
        f"topk_{name}": MetricResult(0.5, 0.5, 1, True)
        for name in eval_cli.TOPK_V31_DIAGNOSTIC_NAMES[:-1]
    }

    with pytest.raises(RuntimeError, match="per-record diagnostics are missing"):
        eval_cli._per_record_metric_values(
            metrics,
            require_topk_v31=True,
        )

    legacy = eval_cli._per_record_metric_values(
        {
            "ordinary_valid": MetricResult(1.0, 1.0, 1, True),
            "ordinary_invalid": MetricResult.invalid(),
        },
        require_topk_v31=False,
    )
    assert legacy == {"ordinary_valid": 1.0}


def test_v2_metric_uses_teacher_correspondence_carried_prediction() -> None:
    transport = _fake_transport(99.0)
    valid = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    reference = train.ReferenceTemporalWarp(
        disparity_hr_px=torch.full((1, 1, 4, 4), 10.0),
        prediction_disparity_hr_px=torch.full((1, 1, 4, 4), 14.0),
        valid_mask_hr=valid,
        visibility_mask_hr=valid,
        collision_mask_hr=torch.zeros_like(valid),
    )
    result = eval_cli._temporal_residual_metric_for_transport(
        torch.full((1, 1, 4, 4), 15.0),
        transport,
        current_reference_hr_px=torch.full((1, 1, 4, 4), 11.0),
        current_reference_valid_hr=valid,
        reference_transport=reference,
    )
    # Prediction and teacher residuals are both +1. If the unrelated method
    # transport value 99 were used, this would be a large non-zero error.
    assert result == MetricResult(0.0, 0.0, 16, True)


def test_explicit_validity_completion_metrics_use_global_confusion_counts() -> None:
    target = torch.tensor([[[[4.0, 5.0, 0.0, 7.0]]]])
    teacher_valid = torch.tensor([[[[True, True, False, True]]]])
    ffs_valid = torch.tensor([[[[True, False, False, False]]]])
    output_valid = torch.tensor([[[[True, False, True, True]]]])
    completion = torch.tensor([[[[False, False, True, True]]]])
    disparity = torch.ones_like(target)
    output = ModelOutput(
        disparity_hr_px=disparity,
        disparity_raw_hr_px=disparity,
        source_weights=torch.zeros((1, 3, 1, 2)),
        log_variance=torch.zeros_like(disparity),
        uncertainty=torch.ones_like(disparity),
        hidden_state=(torch.ones((1, 1, 1, 2)),),
        anchor_gate=torch.ones_like(disparity),
        source_valid_mask=torch.ones((1, 3, 1, 2), dtype=torch.bool),
        valid_probability=torch.tensor([[[[0.9, 0.4, 0.6, 0.8]]]]),
        completion_probability=torch.tensor([[[[0.0, 0.4, 0.6, 0.8]]]]),
        output_valid_mask=output_valid,
        completion_mask=completion,
    )

    metrics = eval_cli._explicit_validity_completion_metrics(
        output,
        target_disparity_hr_px=target,
        teacher_valid_mask_hr=teacher_valid,
        ffs_valid_mask_hr=ffs_valid,
    )

    assert metrics["explicit_valid_precision"] == MetricResult(
        2.0 / 3.0, 2.0, 3, True
    )
    assert metrics["explicit_valid_recall"] == MetricResult(
        2.0 / 3.0, 2.0, 3, True
    )
    assert metrics["explicit_valid_f1"] == MetricResult(2.0 / 3.0, 4.0, 6, True)
    assert metrics["ffs_hole_completion_precision"] == MetricResult(
        0.5, 1.0, 2, True
    )
    assert metrics["ffs_hole_completion_recall"] == MetricResult(
        0.5, 1.0, 2, True
    )
    assert metrics["ffs_hole_completion_f1"] == MetricResult(0.5, 2.0, 4, True)
    assert metrics["explicit_valid_brier"].count == 4
    assert metrics["ffs_hole_completion_brier"].count == 3

    unavailable = eval_cli._explicit_validity_completion_metrics(
        None,
        target_disparity_hr_px=target,
        teacher_valid_mask_hr=teacher_valid,
        ffs_valid_mask_hr=ffs_valid,
    )
    assert all(not metric.valid and metric.count == 0 for metric in unavailable.values())


def test_failure_collector_uses_stable_ties_writes_selection_and_bounds_cpu(
    tmp_path: Path,
) -> None:
    collector = eval_cli.FailureSampleCollector(
        samples_per_criterion=2, cpu_limit_bytes=100_000
    )
    payload_factory_calls = 0

    def counted_payload_factory() -> dict[str, torch.Tensor]:
        nonlocal payload_factory_calls
        payload_factory_calls += 1
        return _failure_payload()

    for sequence_id, frame_id, manifest_index in (
        ("seq-z", 10, 8),
        ("seq-a", 20, 9),
        ("seq-low", 30, 10),
    ):
        collector.consider(
            "raw_negative_rate",
            MetricResult(
                value=0.5 if sequence_id != "seq-low" else 0.25,
                numerator=2.0 if sequence_id != "seq-low" else 1.0,
                count=4,
                valid=True,
            ),
            sequence_id=sequence_id,
            frame_id=frame_id,
            manifest_index=manifest_index,
            timestamp=float(frame_id),
            payload_factory=counted_payload_factory,
        )
    # The low-scoring third item never enters top-k, so its visualization
    # tensors are not copied to CPU.
    assert payload_factory_calls == 2
    report = collector.write(
        tmp_path / "failures",
        checkpoint={
            "path": "/checkpoint.pt",
            "checkpoint_sha256": "c" * 64,
            "step": 15_000,
            "git_hash": "a" * 40,
        },
        evaluator={
            "git_hash": "b" * 40,
            "eval_py_sha256": "d" * 64,
            "evaluation_module_sha256": "e" * 64,
        },
    )
    assert report["criteria"]["raw_negative_rate"]["selected_samples"] == 2
    selection_path = tmp_path / "failures" / "raw_negative_rate" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert [item["sequence_id"] for item in selection["selected"]] == ["seq-a", "seq-z"]
    assert all(
        set(item["metric"]) == {"value", "numerator", "count", "valid"}
        for item in selection["selected"]
    )
    assert all(item["checkpoint_sha256"] == "c" * 64 for item in selection["selected"])
    assert all(
        item["evaluator_eval_py_sha256"] == "d" * 64
        and item["evaluator_evaluation_module_sha256"] == "e" * 64
        for item in selection["selected"]
    )
    assert (tmp_path / "failures" / "raw_negative_rate" / "0001_seq-a_20_9").is_dir()
    assert (tmp_path / "failures" / "boundary_epe_px" / "selection.json").is_file()

    bounded = eval_cli.FailureSampleCollector(
        samples_per_criterion=1, cpu_limit_bytes=1
    )
    with pytest.raises(MemoryError, match="CPU tensor limit"):
        bounded.consider(
            "raw_negative_rate",
            MetricResult(1.0, 1.0, 1, True),
            sequence_id="seq",
            frame_id=1,
            manifest_index=1,
            timestamp=1.0,
            payload_factory=_failure_payload,
        )


def test_comparison_is_derived_from_dataset_aggregates() -> None:
    baseline = {
        "trusted_region_epe_px": AggregateMetric(1.0, 10.0, 10, True),
        "low_confidence_epe_px": AggregateMetric(2.0, 20.0, 10, True),
        "invalid_region_completeness": AggregateMetric(0.4, 4.0, 10, True),
    }
    candidate = {
        "trusted_region_epe_px": AggregateMetric(1.02, 10.2, 10, True),
        "low_confidence_epe_px": AggregateMetric(1.6, 16.0, 10, True),
        "invalid_region_completeness": AggregateMetric(0.5, 5.0, 10, True),
    }

    comparison = comparison_from_aggregates(baseline, candidate)

    assert comparison["trusted_region_degradation"]["relative_change_percent"] == pytest.approx(2.0)
    assert comparison["low_confidence_epe_change"][
        "relative_change_percent"
    ] == pytest.approx(-20.0)
    assert comparison["invalid_region_completeness_change"][
        "relative_change_percent"
    ] == pytest.approx(25.0)


def test_evaluation_checkpoint_loader_is_strict_and_checks_parameter_count(
    tmp_path: Path,
) -> None:
    source = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    count = sum(parameter.numel() for parameter in source.parameters())
    checkpoint = tmp_path / "stage_a.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": source.state_dict(),
            "parameter_count": count,
            "step": 7,
            "config": {"stage": "A"},
            "git_hash": "a" * 40,
        },
        checkpoint,
    )

    metadata = load_model_for_evaluation(
        checkpoint, target, expected_parameter_count=count
    )
    assert metadata["step"] == 7
    assert metadata["parameter_count"] == count
    torch.testing.assert_close(source.weight, target.weight)
    torch.testing.assert_close(source.bias, target.bias)
    assert PSEUDO_GT_LABEL in "trusted_hr_ffs_teacher_pseudo_gt"

    with pytest.raises(CheckpointMismatchError, match="parameter count mismatch"):
        load_model_for_evaluation(
            checkpoint, nn.Linear(2, 1), expected_parameter_count=count + 1
        )


def test_empty_metric_result_nan_does_not_leak_into_json_aggregate() -> None:
    raw = MetricResult.invalid()
    assert math.isnan(raw.value)
    aggregate = aggregate_metric_results((raw,))
    assert aggregate.to_dict() == {
        "value": None,
        "numerator": None,
        "count": 0,
        "valid": False,
    }


def _cache_identity_dict(component: str) -> dict[str, object]:
    return _identity(component).to_dict()


def _temporal_checkpoint_config() -> dict[str, object]:
    config = copy.deepcopy(train.DEFAULT_CONFIG)
    config["data"].update(  # type: ignore[union-attr]
        {
            "sequence_length": 3,
            "vggt_context_pairs": 5,
            "observation_cache_identity": _cache_identity_dict(
                "ffs-observation"
            ),
            "teacher_cache_identity": _cache_identity_dict("ffs-teacher"),
            "derived_cache_lineage": {
                "component": "vggt-ffs-derived-geometry-batch",
                "config": {"algorithm": "strict"},
                "derived_cache_root": "/cache/train/derived",
            },
        }
    )
    config["model"].update(  # type: ignore[union-attr]
        {"use_history": True, "use_vggt_pose": True}
    )
    config["train"].update(  # type: ignore[union-attr]
        {
            "stage": "temporal",
            "init_from_stage": "spatial",
            "history_detach": True,
            "initialization_checkpoint": "/checkpoints/stage_a.pt",
            "initialization_checkpoint_sha256": "c" * 64,
        }
    )
    return config


def _temporal_v31_checkpoint_config() -> dict[str, object]:
    config = _temporal_checkpoint_config()
    config["calibration_conditioning_v3"] = {
        "enabled": True,
        "protocol_version": train.CALIBRATION_CONDITIONING_V3_PROTOCOL,
        "pixel_center_contract": (
            train.ALIGN_CORNERS_FALSE_PIXEL_CENTER_CONTRACT
        ),
        "use_rays": True,
        "use_stereo_pose": True,
        "use_temporal_pose": False,
    }
    config["measurement_ownership_v3_1"] = {
        "enabled": True,
        "protocol_version": train.MEASUREMENT_OWNERSHIP_V31_PROTOCOL,
        "minimum_subpixel_residual_hr_px": 1.0,
        "maximum_subpixel_residual_hr_px": 8.0,
        "boundary_relative_scale": 0.10,
    }
    config["temporal_candidate_fusion_v3_1"] = {
        "enabled": True,
        "protocol_version": train.TEMPORAL_CANDIDATE_FUSION_V31_PROTOCOL,
        "per_age_quota": 2,
        "surface_depth_gap_m": 0.05,
        "surface_relative_depth_gap": 0.05,
        "phase_redundancy_sigma_grid_px": 0.125,
        "phase_redundancy_penalty": 0.25,
    }
    data = config["data"]
    assert isinstance(data, dict)
    data["derived_contract"] = "calibrated_stereo_v2"
    data["calibration_sidecar_lineage"] = {
        "component": "rectified-stereo-calibration",
        "contract_version": "stored_rectified_virtual_cameras_v1",
        "pixel_audit_sha256": "a" * 64,
    }
    derived = data["derived_cache_lineage"]
    assert isinstance(derived, dict)
    derived["component"] = (
        "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch"
    )
    return config


def test_temporal_checkpoint_lineage_checks_stage_policy_and_active_config() -> None:
    config = _temporal_checkpoint_config()
    metadata = {"training_config": config}
    current_derived = {
        "component": "vggt-ffs-derived-geometry-batch",
        "config": {"algorithm": "strict"},
    }
    result = validate_checkpoint_lineage(
        metadata,
        required_stage="temporal",
        observation_cache_identity=_cache_identity_dict("ffs-observation"),
        teacher_cache_identity=_cache_identity_dict("ffs-teacher"),
        derived_cache_lineage=current_derived,
        evaluation_config=config,
    )
    assert result["stage_a_initialization_sha256"] == "c" * 64

    wrong = copy.deepcopy(config)
    wrong["train"]["history_detach"] = False  # type: ignore[index]
    with pytest.raises(CheckpointMismatchError, match="detached"):
        validate_checkpoint_lineage(
            {"training_config": wrong},
            required_stage="temporal",
            observation_cache_identity=_cache_identity_dict("ffs-observation"),
            teacher_cache_identity=_cache_identity_dict("ffs-teacher"),
            derived_cache_lineage=current_derived,
            evaluation_config=config,
        )

    changed_eval = copy.deepcopy(config)
    changed_eval["train"]["history_conflict_hr_px"] = 9.0  # type: ignore[index]
    with pytest.raises(CheckpointMismatchError, match="history_conflict"):
        validate_checkpoint_lineage(
            metadata,
            required_stage="temporal",
            observation_cache_identity=_cache_identity_dict("ffs-observation"),
            teacher_cache_identity=_cache_identity_dict("ffs-teacher"),
            derived_cache_lineage=current_derived,
            evaluation_config=changed_eval,
        )


@pytest.mark.parametrize(
    ("section_name", "field_name", "replacement"),
    (
        ("measurement_ownership_v3_1", "boundary_relative_scale", 0.2),
        ("temporal_candidate_fusion_v3_1", "per_age_quota", 1),
        ("temporal_candidate_fusion_v3_1", "per_age_quota", 2.0),
        (
            "temporal_candidate_fusion_v3_1",
            "phase_redundancy_penalty",
            0.5,
        ),
    ),
)
def test_v31_checkpoint_lineage_requires_exact_behavior_config(
    section_name: str,
    field_name: str,
    replacement: object,
) -> None:
    config = _temporal_v31_checkpoint_config()
    current_derived = {
        "component": "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch",
        "config": {"algorithm": "strict"},
    }
    result = validate_checkpoint_lineage(
        {"training_config": config},
        required_stage="temporal",
        observation_cache_identity=_cache_identity_dict("ffs-observation"),
        teacher_cache_identity=_cache_identity_dict("ffs-teacher"),
        derived_cache_lineage=current_derived,
        evaluation_config=copy.deepcopy(config),
    )
    assert result["calibration_conditioning_v3"]["pixel_center_contract"] == (
        train.ALIGN_CORNERS_FALSE_PIXEL_CENTER_CONTRACT
    )

    changed_eval = copy.deepcopy(config)
    section = changed_eval[section_name]
    assert isinstance(section, dict)
    section[field_name] = replacement
    with pytest.raises(
        CheckpointMismatchError,
        match=rf"v3\.1 behavior config differs.*{section_name}",
    ):
        validate_checkpoint_lineage(
            {"training_config": config},
            required_stage="temporal",
            observation_cache_identity=_cache_identity_dict(
                "ffs-observation"
            ),
            teacher_cache_identity=_cache_identity_dict("ffs-teacher"),
            derived_cache_lineage=current_derived,
            evaluation_config=changed_eval,
        )


@pytest.mark.parametrize(
    "missing_from",
    ("checkpoint", "evaluation"),
)
def test_v31_checkpoint_lineage_rejects_missing_behavior_section(
    missing_from: str,
) -> None:
    checkpoint_config = _temporal_v31_checkpoint_config()
    evaluation_config = copy.deepcopy(checkpoint_config)
    target = checkpoint_config if missing_from == "checkpoint" else evaluation_config
    target.pop("measurement_ownership_v3_1")
    current_derived = {
        "component": "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch",
        "config": {"algorithm": "strict"},
    }

    with pytest.raises(
        CheckpointMismatchError,
        match=rf"{missing_from} measurement_ownership_v3_1 config",
    ):
        validate_checkpoint_lineage(
            {"training_config": checkpoint_config},
            required_stage="temporal",
            observation_cache_identity=_cache_identity_dict(
                "ffs-observation"
            ),
            teacher_cache_identity=_cache_identity_dict("ffs-teacher"),
            derived_cache_lineage=current_derived,
            evaluation_config=evaluation_config,
        )


def test_checkpoint_lineage_rejects_unknown_pixel_center_contract() -> None:
    config = _temporal_v31_checkpoint_config()
    calibration = config["calibration_conditioning_v3"]
    assert isinstance(calibration, dict)
    calibration["pixel_center_contract"] = "unknown"
    current_derived = {
        "component": "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch",
        "config": {"algorithm": "strict"},
    }

    with pytest.raises(
        CheckpointMismatchError,
        match="unsupported pixel-center contract",
    ):
        validate_checkpoint_lineage(
            {"training_config": config},
            required_stage="temporal",
            observation_cache_identity=_cache_identity_dict(
                "ffs-observation"
            ),
            teacher_cache_identity=_cache_identity_dict("ffs-teacher"),
            derived_cache_lineage=current_derived,
            evaluation_config=copy.deepcopy(config),
        )


def test_v31_checkpoint_lineage_requires_evaluation_config() -> None:
    config = _temporal_v31_checkpoint_config()
    current_derived = {
        "component": "vggt-ffs-derived-geometry-calibrated-stereo-v2-batch",
        "config": {"algorithm": "strict"},
    }

    with pytest.raises(
        CheckpointMismatchError,
        match="v3.1 lineage validation requires.*evaluation config",
    ):
        validate_checkpoint_lineage(
            {"training_config": config},
            required_stage="temporal",
            observation_cache_identity=_cache_identity_dict(
                "ffs-observation"
            ),
            teacher_cache_identity=_cache_identity_dict("ffs-teacher"),
            derived_cache_lineage=current_derived,
            evaluation_config=None,
        )


def _valid_temporal_causality_batch() -> dict[str, object]:
    frame_ids = [10, 20, 30]
    timestamps = [1.0, 2.0, 3.0]
    crop = {"x": 0, "y": 2, "width": 8, "height": 4, "spatial_scale": 2}
    per_time_ffs = [
        {
            "manifest_record": {
                "sequence_id": "seq",
                "frame_id": frame_id,
                "timestamp": timestamp,
            },
            "crop_hr_px": crop,
        }
        for frame_id, timestamp in zip(frame_ids, timestamps, strict=True)
    ]
    per_time_derived = [
        {
            "cache_path": f"/cache/seq/{frame_id}.pt",
            "pose_valid": True,
            "static_prior_valid": True,
        }
        for frame_id in frame_ids
    ]
    return {
        "frame_ids": torch.tensor([frame_ids]),
        "timestamps": torch.tensor([timestamps], dtype=torch.float64),
        "manifest_indices": torch.tensor([[4, 5, 6]]),
        "temporal_pose_valid_sequence": torch.ones((1, 3), dtype=torch.bool),
        "static_prior_valid_sequence": torch.ones((1, 3), dtype=torch.bool),
        "sequence_id": ["seq"],
        "identity_metadata": [
            {
                "sequence_id": "seq",
                "student_manifest_indices": [4, 5, 6],
                "vggt_context_manifest_indices": [2, 3, 4, 5, 6],
                "endpoint_manifest_index": 6,
                "crop_hr_px": crop,
                "per_time_ffs": per_time_ffs,
                "per_time_derived": per_time_derived,
            }
        ],
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["timestamps"].__setitem__((0, 2), float("inf")), "finite"),
        (lambda value: value["frame_ids"].__setitem__((0, 1), 10), "frame IDs"),
        (
            lambda value: value["identity_metadata"][0].__setitem__(
                "vggt_context_manifest_indices", [2, 4, 3, 5, 6]
            ),
            "VGGT metadata",
        ),
        (
            lambda value: value["identity_metadata"][0]["per_time_ffs"][1][
                "manifest_record"
            ].__setitem__("sequence_id", "future-seq"),
            "sequence boundary",
        ),
    ),
)
def test_temporal_causality_rejects_future_or_mixed_metadata(
    mutation: object, message: str
) -> None:
    batch = _valid_temporal_causality_batch()
    assert validate_temporal_batch_causality(batch) == {
        "batch_size": 1,
        "frames_per_window": 3,
    }
    mutation(batch)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        validate_temporal_batch_causality(batch)


def test_hr_temporal_safe_mask_is_exact_visible_static_intersection() -> None:
    reference = torch.ones((1, 1, 1, 4))
    mask = hr_temporal_safe_mask(
        reference,
        visibility_mask_hr=torch.tensor([[[[True, True, True, True]]]]),
        static_mask_hr=torch.tensor([[[[True, True, True, False]]]]),
        collision_mask_hr=torch.tensor([[[[False, True, False, False]]]]),
        geometry_consistent_mask_hr=torch.tensor(
            [[[[True, True, False, True]]]]
        ),
        valid_history_hr=torch.tensor([[[[True, True, True, True]]]]),
    )
    assert torch.equal(mask, torch.tensor([[[[True, False, False, False]]]]))


def test_hr_temporal_residual_metric_uses_warp_and_paired_masks() -> None:
    # Identity-warp values are supplied explicitly: teacher residual is +1;
    # prediction residual is +1, +3, +1, +1.  The paired mask removes the last
    # pixel and the teacher-history validity removes the third.
    current_prediction = torch.tensor([[[[15.0, 17.0, 15.0, 15.0]]]])
    warped_previous_prediction = torch.full_like(current_prediction, 14.0)
    current_teacher = torch.full_like(current_prediction, 11.0)
    warped_previous_teacher = torch.full_like(current_prediction, 10.0)
    all_valid = torch.ones_like(current_prediction, dtype=torch.bool)
    teacher_history_valid = all_valid.clone()
    teacher_history_valid[..., 2] = False
    paired = all_valid.clone()
    paired[..., 3] = False

    result = hr_temporal_residual_metric(
        current_prediction,
        warped_previous_prediction,
        current_teacher,
        warped_previous_teacher,
        visibility_mask_hr=all_valid,
        static_mask_hr=all_valid,
        collision_mask_hr=torch.zeros_like(all_valid),
        geometry_consistent_mask_hr=all_valid,
        valid_prediction_history_hr=all_valid,
        current_reference_valid_mask_hr=all_valid,
        warped_previous_reference_valid_mask_hr=teacher_history_valid,
        paired_domain_mask_hr=paired,
    )

    assert result == MetricResult(value=1.0, numerator=2.0, count=2, valid=True)


def test_physical_clamp_uses_zero_not_epsilon_and_preserves_nonfinite() -> None:
    value = torch.tensor(
        [-3.0, -0.0, 2.0, float("nan"), float("inf"), float("-inf")]
    )
    result = physical_disparity_clamp_min_zero(value)
    assert result[0].item() == 0.0
    assert result[1].item() == 0.0
    assert result[2].item() == 2.0
    assert torch.isnan(result[3])
    assert torch.isposinf(result[4])
    assert torch.isneginf(result[5])


def _sign_health_output() -> ModelOutput:
    source_mix = torch.tensor([[[[-2.0, float("nan"), 0.2, 1.0]]]])
    post_lr = torch.tensor([[[[-1.0, -0.5, 0.2, float("inf")]]]])
    post_convex = torch.tensor(
        [
            [
                [
                    [-1.0, -1.0, float("nan"), float("nan"), 0.1, 0.1, 1.0, 1.0],
                    [-1.0, -1.0, float("nan"), float("nan"), 0.1, 0.1, 1.0, 1.0],
                ]
            ]
        ]
    )
    raw = post_convex + 0.5
    final = torch.tensor(
        [
            [
                [
                    [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, float("-inf"), 1.0],
                    [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, float("-inf"), 1.0],
                ]
            ]
        ]
    )
    return ModelOutput(
        disparity_hr_px=final,
        disparity_raw_hr_px=raw,
        source_weights=torch.ones((1, 3, 1, 4)) / 3.0,
        log_variance=torch.zeros_like(final),
        uncertainty=torch.ones_like(final),
        hidden_state=(),
        anchor_gate=torch.ones_like(final),
        source_valid_mask=torch.ones((1, 3, 1, 4), dtype=torch.bool),
        disparity_source_mix_hr_px_lr_grid=source_mix,
        disparity_post_lr_residual_hr_px_lr_grid=post_lr,
        disparity_post_convex_hr_px=post_convex,
    )


def test_t3_vggt_sign_health_counts_native_stages_and_exhaustive_strata() -> None:
    accumulator = eval_cli.T3VGGTSignHealthAccumulator()
    bilinear = torch.tensor(
        [
            [
                [
                    [-1.0, -1.0, 0.1, 0.1, 0.25, 0.25, 1.0, 1.0],
                    [-1.0, -1.0, 0.1, 0.1, 0.25, 0.25, 1.0, 1.0],
                ]
            ]
        ]
    )
    ffs_lr = torch.tensor([[[[True, True, False, False]]]])
    history_lr = torch.tensor([[[[True, False, True, False]]]])
    accumulator.update(
        _sign_health_output(),
        bilinear_disparity_hr_px=bilinear,
        ffs_valid_lr=ffs_lr,
        ffs_valid_hr=torch.nn.functional.interpolate(
            ffs_lr.float(), size=(2, 8), mode="nearest"
        ).bool(),
        history_valid_lr=history_lr,
        history_valid_hr=torch.nn.functional.interpolate(
            history_lr.float(), size=(2, 8), mode="nearest"
        ).bool(),
        pose_valid=torch.tensor([True]),
    )
    report = accumulator.finalize()

    assert report["method"] == "T3_VGGT"
    assert report["records_accumulated"] == 1
    source = report["stages"]["source_mix_lr"]
    assert source["native_grid"] == "LR"
    assert source["diagnostic_domain_count"] == 4
    all_pixels = source["strata"]["all_pixels"]
    assert all_pixels == {
        "diagnostic_domain_count": 4,
        "finite_count": 3,
        "negative_count": 1,
        "nonfinite_count": 1,
        "negative_rate_over_diagnostic_domain": pytest.approx(0.25),
        "negative_rate_over_finite": pytest.approx(1.0 / 3.0),
        "nonfinite_rate_over_diagnostic_domain": pytest.approx(0.25),
    }
    assert source["strata"]["bilinear_disparity_lt_0_hr_px"][
        "negative_count"
    ] == 1
    assert source["strata"]["bilinear_disparity_ge_0_lt_0_25_hr_px"][
        "nonfinite_count"
    ] == 1
    assert source["strata"]["ffs_valid"]["diagnostic_domain_count"] == 2
    assert source["strata"]["history_valid"]["negative_count"] == 1
    assert source["strata"]["pose_valid"]["diagnostic_domain_count"] == 4
    assert source["strata"]["pose_invalid"]["diagnostic_domain_count"] == 0

    for stage in report["stages"].values():
        strata = stage["strata"]
        all_count = strata["all_pixels"]["diagnostic_domain_count"]
        assert sum(
            strata[name]["diagnostic_domain_count"]
            for name in report["partition_contract"]["bilinear_disparity"]
        ) == all_count
        assert (
            strata["ffs_valid"]["diagnostic_domain_count"]
            + strata["ffs_invalid"]["diagnostic_domain_count"]
            == all_count
        )
        assert (
            strata["history_valid"]["diagnostic_domain_count"]
            + strata["history_invalid"]["diagnostic_domain_count"]
            == all_count
        )
        assert (
            strata["pose_valid"]["diagnostic_domain_count"]
            + strata["pose_invalid"]["diagnostic_domain_count"]
            == all_count
        )

    final = report["stages"]["post_anchor_final"]["strata"]["all_pixels"]
    assert final["diagnostic_domain_count"] == 16
    assert final["negative_count"] == 2
    assert final["nonfinite_count"] == 2


def test_visualization_writes_exact_finite_final_negative_mask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written: dict[str, object] = {}
    monkeypatch.setattr(
        eval_cli,
        "save_rgb_uint8",
        lambda path, image: written.__setitem__(Path(path).name, image),
    )
    output = torch.tensor([[[-1.0, 0.0], [float("-inf"), 2.0]]])
    eval_cli._save_visualization(
        tmp_path,
        sample_name="sample",
        rgb_hr=torch.zeros((3, 2, 2)),
        K_hr_px=torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        baseline_m=torch.tensor(0.1),
        baseline_hr_px=torch.ones((1, 2, 2)),
        output_hr_px=output,
        target_hr_px=torch.ones((1, 2, 2)),
        target_trusted_mask=torch.ones((1, 2, 2), dtype=torch.bool),
        source_weights_lr=torch.ones((3, 1, 1)) / 3.0,
        uncertainty_hr=torch.ones((1, 2, 2)),
    )

    mask = written["final_negative_mask.png"]
    assert getattr(mask, "shape") == (2, 2, 3)
    assert mask[0, 0].tolist() == [255, 255, 255]  # type: ignore[index]
    assert mask[0, 1].tolist() == [0, 0, 0]  # type: ignore[index]
    # -inf is non-finite and belongs only in the JSON non-finite counter.
    assert mask[1, 0].tolist() == [0, 0, 0]  # type: ignore[index]
    point_cloud = tmp_path / "sample" / "point_cloud_camera_frame.ply"
    assert point_cloud.is_file()
    assert "element vertex 1" in point_cloud.read_text(encoding="ascii")


def _temporal_flicker_inputs() -> dict[str, torch.Tensor]:
    rgb = torch.tensor(
        [
            [[0.0, 0.25, 0.5, 0.75], [1.0, 0.75, 0.5, 0.25]],
            [[0.0, 0.25, 0.5, 0.75], [1.0, 0.75, 0.5, 0.25]],
            [[0.0, 0.25, 0.5, 0.75], [1.0, 0.75, 0.5, 0.25]],
        ]
    )
    return {
        "rgb_hr": rgb,
        "bilinear_disparity_hr_px": torch.ones((1, 2, 4)) * 4.0,
        "t3_disparity_hr_px": torch.ones((1, 2, 4)) * 5.0,
        "t3_vggt_disparity_hr_px": torch.tensor(
            [[[6.0, 6.0, float("nan"), 6.0], [6.0, 6.0, 6.0, 6.0]]]
        ),
        "target_disparity_hr_px": torch.ones((1, 2, 4)) * 5.0,
        "target_trusted_mask": torch.tensor(
            [[[True, True, False, True], [True, True, True, True]]]
        ),
        "uncertainty_variance": torch.ones((1, 2, 4)) * 0.5,
    }


def test_temporal_flicker_panel_is_cpu_uint8_six_panel_fixed_scale() -> None:
    panel = eval_cli.build_temporal_flicker_panel(
        **_temporal_flicker_inputs(),
        disparity_range_hr_px=(0.0, 8.0),
        error_range_hr_px=(0.0, 4.0),
        uncertainty_range=(0.0, 1.0),
    )
    # 2 rows x (H + 24px label) and 3 columns x W, with even-size padding.
    assert panel.shape == (52, 12, 3)
    assert panel.dtype == np.uint8
    # The untrusted/non-finite error pixel is fail-closed to black in its tile.
    assert panel[50, 6].tolist() == [0, 0, 0]


def test_temporal_flicker_collector_streams_frames_and_publishes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames: list[np.ndarray] = []

    class FakeWriter:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.closed = False

        def append_data(self, frame: np.ndarray) -> None:
            assert frame.dtype == np.uint8
            frames.append(frame)
            self.path.write_bytes(b"fake-mp4")

        def close(self) -> None:
            self.closed = True

    def open_writer(
        path: Path,
        *,
        fps: int,
        frame_size_hw: tuple[int, int],
    ) -> tuple[FakeWriter, str]:
        assert fps == 7
        assert frame_size_hw == (52, 12)
        return FakeWriter(path), "fake"

    monkeypatch.setattr(eval_cli, "_open_temporal_flicker_video_writer", open_writer)
    collector = eval_cli.TemporalFlickerVideoCollector(
        tmp_path / "videos",
        enabled=True,
        fps=7,
        disparity_range_hr_px=(0.0, 8.0),
        error_range_hr_px=(0.0, 4.0),
        uncertainty_range=(0.0, 1.0),
    )
    inputs = _temporal_flicker_inputs()
    collector.append(sequence_id="seq/one", frame_id=4, timestamp=0.8, **inputs)
    collector.append(sequence_id="seq/one", frame_id=5, timestamp=1.0, **inputs)
    report = collector.finalize()
    assert len(frames) == 2
    assert report["status"] == "COMPLETE"
    assert report["metric_participation"] == "NONE"
    assert report["videos"] == [
        {
            "sequence_id": "seq/one",
            "path": str(tmp_path / "videos/seq_one.mp4"),
            "frame_count": 2,
            "backend": "fake",
        }
    ]
    assert (tmp_path / "videos/seq_one.mp4").read_bytes() == b"fake-mp4"
    assert not (tmp_path / "videos/.seq_one.incomplete.mp4").exists()


def test_temporal_flicker_collector_reports_missing_encoder_without_partial_mp4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(
        path: Path,
        *,
        fps: int,
        frame_size_hw: tuple[int, int],
    ) -> object:
        del path, fps, frame_size_hw
        raise eval_cli._TemporalFlickerVideoUnavailable("FFmpeg unavailable")

    monkeypatch.setattr(eval_cli, "_open_temporal_flicker_video_writer", unavailable)
    collector = eval_cli.TemporalFlickerVideoCollector(
        tmp_path / "videos",
        enabled=True,
        fps=5,
        disparity_range_hr_px=(0.0, 8.0),
        error_range_hr_px=(0.0, 4.0),
        uncertainty_range=(0.0, 1.0),
    )
    collector.append(sequence_id="seq", frame_id=4, timestamp=0.8, **_temporal_flicker_inputs())
    report = collector.finalize()
    assert report["status"] == "NOT_AVAILABLE"
    assert report["reason"] == "FFmpeg unavailable"
    assert report["videos"] == []
    assert list((tmp_path / "videos").glob("*.mp4")) == []


def test_temporal_flicker_abort_cleans_encoder_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWriter:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.closed = False

        def append_data(self, frame: np.ndarray) -> None:
            del frame
            self.path.write_bytes(b"partial")

        def close(self) -> None:
            self.closed = True

    writer: FakeWriter | None = None

    def open_writer(
        path: Path,
        *,
        fps: int,
        frame_size_hw: tuple[int, int],
    ) -> tuple[FakeWriter, str]:
        nonlocal writer
        del fps, frame_size_hw
        writer = FakeWriter(path)
        return writer, "fake"

    monkeypatch.setattr(eval_cli, "_open_temporal_flicker_video_writer", open_writer)
    collector = eval_cli.TemporalFlickerVideoCollector(
        tmp_path / "videos",
        enabled=True,
        fps=5,
        disparity_range_hr_px=(0.0, 8.0),
        error_range_hr_px=(0.0, 4.0),
        uncertainty_range=(0.0, 1.0),
    )
    collector.append(sequence_id="seq", frame_id=4, timestamp=0.8, **_temporal_flicker_inputs())
    collector.abort_for_interruption(KeyboardInterrupt("stop"))
    assert writer is not None and writer.closed
    assert list((tmp_path / "videos").glob("*.mp4")) == []


def test_temporal_flicker_prefers_imageio_then_falls_back_to_direct_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def imageio_missing(path: Path, *, fps: int) -> object:
        del path, fps
        raise ImportError("imageio absent")

    class FakeDirectWriter:
        pass

    direct_writer = FakeDirectWriter()

    def direct_writer_factory(
        path: Path,
        *,
        fps: int,
        frame_size_hw: tuple[int, int],
    ) -> FakeDirectWriter:
        assert path == tmp_path / "fallback.mp4"
        assert fps == 5
        assert frame_size_hw == (52, 12)
        return direct_writer

    monkeypatch.setattr(
        eval_cli, "_open_imageio_temporal_flicker_writer", imageio_missing
    )
    monkeypatch.setattr(
        eval_cli,
        "_open_direct_ffmpeg_temporal_flicker_writer",
        direct_writer_factory,
    )
    writer, backend = eval_cli._open_temporal_flicker_video_writer(
        tmp_path / "fallback.mp4",
        fps=5,
        frame_size_hw=(52, 12),
    )
    assert writer is direct_writer
    assert backend == "direct_ffmpeg_rgb24"


@pytest.mark.skipif(
    not eval_cli.DIRECT_FFMPEG_EXECUTABLE.is_file(),
    reason="system direct FFmpeg is unavailable",
)
def test_direct_ffmpeg_rgb24_writer_creates_decodable_small_mp4(tmp_path: Path) -> None:
    output = tmp_path / "tiny.mp4"
    writer = eval_cli._open_direct_ffmpeg_temporal_flicker_writer(
        output,
        fps=5,
        frame_size_hw=(2, 4),
    )
    writer.append_data(np.zeros((2, 4, 3), dtype=np.uint8))
    writer.append_data(np.full((2, 4, 3), 255, dtype=np.uint8))
    writer.close()
    assert output.is_file() and output.stat().st_size > 0
    decode = eval_cli.subprocess.run(
        [
            str(eval_cli.DIRECT_FFMPEG_EXECUTABLE),
            "-v",
            "error",
            "-i",
            str(output),
            "-f",
            "null",
            "-",
        ],
        check=False,
        stdout=eval_cli.subprocess.DEVNULL,
        stderr=eval_cli.subprocess.PIPE,
    )
    assert decode.returncode == 0, decode.stderr.decode("utf-8", errors="replace")


def test_temporal_flicker_config_is_opt_in_and_rejects_spatial_evaluation() -> None:
    temporal = eval_cli.resolve_evaluation_config(
        Path(__file__).parents[1] / "configs/temporal_x2.yaml"
    )
    assert temporal.eval.temporal_flicker_video is False
    assert temporal.eval.failure_samples_per_criterion == 0
    assert eval_cli.validate_evaluation_config(temporal) == "temporal"

    spatial = eval_cli.resolve_evaluation_config(
        Path(__file__).parents[1] / "configs/mvp_x2.yaml"
    )
    OmegaConf.update(spatial, "eval.temporal_flicker_video", True)
    with pytest.raises(ValueError, match="requires causal T=3"):
        eval_cli.validate_evaluation_config(spatial)

    OmegaConf.update(spatial, "eval.temporal_flicker_video", False)
    OmegaConf.update(spatial, "eval.failure_samples_per_criterion", 1)
    with pytest.raises(ValueError, match="failure sample bundles require causal T=3"):
        eval_cli.validate_evaluation_config(spatial)


def _formal_coverage_fixture(tmp_path: Path) -> SimpleNamespace:
    records = [
        ManifestRecord(
            sequence_id="seq",
            frame_id=index,
            timestamp=float(index),
            left_path=f"left-{index}.png",
            right_path=f"right-{index}.png",
            K=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            baseline_m=0.1,
        )
        for index in range(7)
    ]
    windows = build_causal_windows(
        records, student_sequence_length=3, vggt_context_pairs=5
    )
    derived_root = tmp_path / "derived"
    raw_root = tmp_path / "vggt"
    derived_root.mkdir()
    raw_root.mkdir()
    raw_manifest = raw_root / "cache_manifest.jsonl"
    raw_manifest.write_text("{}\n", encoding="utf-8")
    derived_manifest = derived_root / "cache_manifest.jsonl"
    derived_manifest.write_text("{}\n", encoding="utf-8")
    receipt = {
        "selection": {"start_window": 0, "limit": None, "selected_windows": 3},
        "counts": {"selected": 3},
        "inputs": {
            "vggt_available_windows": 3,
            "vggt_cache_manifest": str(raw_manifest),
            "vggt_cache_manifest_sha256": sha256_file(raw_manifest),
        },
    }
    (derived_root / "run_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    expected_endpoint_indices = {window.endpoint_index for window in windows}
    emitted = [
        window
        for window in windows
        if all(index in expected_endpoint_indices for index in window.student_indices)
    ]
    return SimpleNamespace(
        derived_cache_root=derived_root,
        records=records,
        derived_entries={index: object() for index in expected_endpoint_indices},
        windows=emitted,
    )


def test_formal_temporal_coverage_rejects_self_consistent_subset(tmp_path: Path) -> None:
    dataset = _formal_coverage_fixture(tmp_path)
    result = eval_cli._validate_formal_temporal_coverage(dataset)
    assert result["manifest_records"] == 7
    assert result["derived_endpoint_records"] == 3
    assert result["evaluable_t3_windows"] == 1

    dataset.derived_entries.pop(4)
    receipt_path = dataset.derived_cache_root / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["selection"]["selected_windows"] = 2
    receipt["counts"]["selected"] = 2
    receipt["inputs"]["vggt_available_windows"] = 2
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        eval_cli._validate_formal_temporal_coverage(dataset)


def test_checkpoint_completion_separates_intermediate_and_canonical_final() -> None:
    temporal_config = {
        "train": {
            "steps": 15_000,
            "steps_temporal": 15_000,
            "steps_spatial": 5_000,
        }
    }
    intermediate = eval_cli.checkpoint_training_completion(
        {"step": 7_500, "training_config": temporal_config},
        stage="temporal",
    )
    assert not intermediate["execution_complete"]
    assert intermediate["canonical_schedule"]
    assert not intermediate["final_training_checkpoint"]

    final = eval_cli.checkpoint_training_completion(
        {"step": 15_000, "training_config": temporal_config},
        stage="temporal",
    )
    assert final["execution_complete"]
    assert final["canonical_schedule"]
    assert final["final_training_checkpoint"]

    shortened_config = copy.deepcopy(temporal_config)
    shortened_config["train"]["steps"] = 7_500
    shortened = eval_cli.checkpoint_training_completion(
        {"step": 7_500, "training_config": shortened_config},
        stage="temporal",
    )
    assert shortened["execution_complete"]
    assert not shortened["canonical_schedule"]
    assert not shortened["final_training_checkpoint"]

    spatial = eval_cli.checkpoint_training_completion(
        {"step": 5_000, "training_config": temporal_config},
        stage="spatial",
    )
    assert spatial["final_training_checkpoint"]

    intermediate_eligibility = eval_cli.evaluation_eligibility_status(
        stage="temporal",
        full_selection=True,
        allow_non_holdout_smoke=False,
        formal_holdout=True,
        checkpoint_completion=intermediate,
        spatial_checkpoint_completion=spatial,
    )
    assert intermediate_eligibility == {
        "coverage_eligible": True,
        "final_training_checkpoint": False,
        "final_acceptance_eligible": False,
        "status": "INTERMEDIATE_CHECKPOINT_EVALUATION_COMPLETE",
    }

    final_eligibility = eval_cli.evaluation_eligibility_status(
        stage="temporal",
        full_selection=True,
        allow_non_holdout_smoke=False,
        formal_holdout=True,
        checkpoint_completion=final,
        spatial_checkpoint_completion=spatial,
    )
    assert final_eligibility["coverage_eligible"]
    assert final_eligibility["final_training_checkpoint"]
    assert final_eligibility["final_acceptance_eligible"]
    assert final_eligibility["status"] == "FINAL_CHECKPOINT_EVALUATION_COMPLETE"

    spatial_intermediate = eval_cli.checkpoint_training_completion(
        {"step": 2_500, "training_config": temporal_config},
        stage="spatial",
    )
    missing_final_stage_a = eval_cli.evaluation_eligibility_status(
        stage="temporal",
        full_selection=True,
        allow_non_holdout_smoke=False,
        formal_holdout=True,
        checkpoint_completion=final,
        spatial_checkpoint_completion=spatial_intermediate,
    )
    assert missing_final_stage_a["coverage_eligible"]
    assert not missing_final_stage_a["final_training_checkpoint"]
    assert not missing_final_stage_a["final_acceptance_eligible"]


def test_checkpoint_completion_rejects_missing_or_boolean_steps() -> None:
    with pytest.raises(ValueError, match="completion metadata"):
        eval_cli.checkpoint_training_completion(
            {"step": True, "training_config": {"train": {}}},
            stage="temporal",
        )
    with pytest.raises(ValueError, match="steps_temporal"):
        eval_cli.checkpoint_training_completion(
            {
                "step": 15_000,
                "training_config": {"train": {"steps": 15_000}},
            },
            stage="temporal",
        )


def _raw_vggt_receipt(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "vggt"
    root.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    identity = _cache_identity_dict("vggt-omega")
    config = {
        "causal": True,
        "context_pairs": 5,
        "current_left_view_index": 8,
        "view_order": [
            label
            for time_label in ("t-4", "t-3", "t-2", "t-1", "t")
            for label in (f"L[{time_label}]", f"R[{time_label}]")
        ],
    }
    (root / "run_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "config": config,
                "selected_windows": 3,
                "available_windows": 3,
                "written_records": 3,
                "reused_records": 0,
                "manifest_sha256": sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    return root, sha256_file(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("causal", False, "causal"),
        ("context_pairs", 4, "causal"),
        ("current_left_view_index", 6, "causal"),
    ),
)
def test_raw_vggt_receipt_mutations_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    root, manifest_sha256 = _raw_vggt_receipt(tmp_path)
    assert eval_cli._validated_raw_vggt_receipt(
        root, expected_manifest_sha256=manifest_sha256
    )["selected_windows"] == 3
    receipt_path = root / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["config"][field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        eval_cli._validated_raw_vggt_receipt(
            root, expected_manifest_sha256=manifest_sha256
        )


def _fake_transport(value: float) -> train.TemporalTransport:
    lr = torch.full((1, 1, 2, 2), value)
    hr = torch.full((1, 1, 4, 4), value)
    lr_bool = torch.ones_like(lr, dtype=torch.bool)
    hr_bool = torch.ones_like(hr, dtype=torch.bool)
    return train.TemporalTransport(
        disparity_history_hr_px=lr,
        confidence_history=torch.ones_like(lr),
        visibility_mask=lr_bool,
        valid_history=lr_bool,
        collision_mask=torch.zeros_like(lr_bool),
        photometric_residual=torch.zeros_like(lr),
        fractional_offset_px=torch.zeros((1, 2, 2, 2)),
        static_mask=lr_bool,
        geometry_consistent_mask=lr_bool,
        disparity_history_loss_hr_px=hr,
        confidence_history_hr=torch.ones_like(hr),
        visibility_mask_hr=hr_bool,
        valid_history_hr=hr_bool,
        collision_mask_hr=torch.zeros_like(hr_bool),
        photometric_residual_hr=torch.zeros_like(hr),
        static_mask_hr=hr_bool,
        geometry_consistent_mask_hr=hr_bool,
    )


def _fake_v2_transport(value: float) -> train.TemporalTransport:
    legacy = _fake_transport(value)
    topk = torch.full((1, 2, 2, 2), value)
    return train.TemporalTransport(
        disparity_history_hr_px=legacy.disparity_history_hr_px,
        confidence_history=legacy.confidence_history,
        visibility_mask=legacy.visibility_mask,
        valid_history=legacy.valid_history,
        collision_mask=legacy.collision_mask,
        photometric_residual=legacy.photometric_residual,
        fractional_offset_px=legacy.fractional_offset_px,
        static_mask=legacy.static_mask,
        geometry_consistent_mask=legacy.geometry_consistent_mask,
        disparity_history_loss_hr_px=legacy.disparity_history_loss_hr_px,
        confidence_history_hr=legacy.confidence_history_hr,
        visibility_mask_hr=legacy.visibility_mask_hr,
        valid_history_hr=legacy.valid_history_hr,
        collision_mask_hr=legacy.collision_mask_hr,
        photometric_residual_hr=legacy.photometric_residual_hr,
        static_mask_hr=legacy.static_mask_hr,
        geometry_consistent_mask_hr=legacy.geometry_consistent_mask_hr,
        topk_disparity_history_hr_px=topk,
        topk_confidence_history=torch.ones_like(topk),
        topk_fractional_offset_px=torch.zeros((1, 2, 2, 2, 2)),
        topk_temporal_age_frames=torch.ones_like(topk),
        topk_z_aware_weights=torch.full_like(topk, 0.5),
        topk_valid_mask=torch.ones_like(topk, dtype=torch.bool),
        warped_hidden_state=(
            torch.full((1, 1, 2, 2), value),
            torch.full((1, 1, 2, 2), value),
        ),
    )


def _v2_temporal_config() -> OmegaConf:
    payload = _temporal_checkpoint_config()
    payload["temporal_history_v2"] = {
        "enabled": True,
        "protocol_version": "topk_z_aware_hidden_warp_v2",
        "top_k": 2,
        "memory_frames": 2,
        "splat_footprint": "bilinear",
        "depth_temperature_m": 0.25,
        "age_temperature_frames": 3.0,
        "source_collision_penalty": 0.5,
        "collision_depth_gap_m": 0.05,
        "collision_relative_depth_gap": 0.05,
        "candidate_feature_channels": 32,
    }
    payload["temporal_residual_v2"] = {
        "enabled": True,
        "protocol_version": "teacher_gt_temporal_residual_v2",
        "reference": "teacher",
    }
    return OmegaConf.create(payload)


def _v2_unroll_batch() -> dict[str, torch.Tensor]:
    scalar_lr = torch.ones((1, 3, 1, 2, 2))
    scalar_hr = torch.ones((1, 3, 1, 4, 4))
    intrinsics = torch.tensor(
        [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
    return {
        "rgb_hr_sequence": torch.zeros((1, 3, 3, 4, 4)),
        "disparity_ffs_hr_px_sequence": scalar_lr.clone(),
        "confidence_ffs_sequence": scalar_lr.clone(),
        "valid_ffs_sequence": torch.ones_like(scalar_lr, dtype=torch.bool),
        "vggt_disparity_hr_px_sequence": scalar_lr.clone() * 7,
        "disparity_vggt_hr_px_sequence": scalar_lr.clone() * 7,
        "confidence_vggt_sequence": scalar_lr.clone(),
        "valid_vggt_sequence": torch.ones_like(scalar_lr, dtype=torch.bool),
        "static_prior_valid_sequence": torch.ones((1, 3), dtype=torch.bool),
        "temporal_pose_valid_sequence": torch.ones((1, 3), dtype=torch.bool),
        "teacher_disparity_hr_px_sequence": scalar_hr.clone() * 10,
        "teacher_confidence_sequence": scalar_hr.clone(),
        "teacher_valid_mask_sequence": torch.ones_like(scalar_hr, dtype=torch.bool),
        "teacher_trusted_mask_sequence": torch.ones_like(
            scalar_hr, dtype=torch.bool
        ),
        "K_hr_sequence": intrinsics,
        "baseline_m_sequence": torch.full((1, 3), 0.1),
    }


class _TemporalSpyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        rgb_hr: torch.Tensor,
        disparity_ffs_hr_px: torch.Tensor,
        confidence_ffs: torch.Tensor,
        **kwargs: object,
    ) -> ModelOutput:
        vggt = kwargs["disparity_vggt_hr_px"]
        valid_vggt = kwargs["valid_vggt"]
        history = kwargs.get("disparity_history_hr_px")
        topk_history = kwargs.get("history_topk_disparity_hr_px")
        hidden_state = kwargs.get("hidden_state")
        assert isinstance(vggt, torch.Tensor)
        assert isinstance(valid_vggt, torch.Tensor)
        self.calls.append(
            {
                "vggt_sum": float(vggt.sum()),
                "valid_vggt": bool(valid_vggt.any()),
                "history_ptr": (
                    None
                    if not isinstance(history, torch.Tensor)
                    else history.data_ptr()
                ),
                "topk_history": isinstance(topk_history, torch.Tensor),
                "hidden_mean": (
                    None
                    if not isinstance(hidden_state, tuple)
                    else float(hidden_state[0].mean())
                ),
            }
        )
        disparity = torch.nn.functional.interpolate(
            disparity_ffs_hr_px, scale_factor=2, mode="nearest"
        )
        hidden = (torch.ones((1, 1, 2, 2)), torch.ones((1, 1, 2, 2)))
        return ModelOutput(
            disparity_hr_px=disparity,
            disparity_raw_hr_px=disparity,
            source_weights=torch.zeros((1, 3, 2, 2)),
            log_variance=torch.zeros_like(disparity),
            uncertainty=torch.ones_like(disparity),
            hidden_state=hidden,
            anchor_gate=torch.ones_like(disparity),
            source_valid_mask=torch.ones((1, 3, 2, 2), dtype=torch.bool),
        )


def test_temporal_unroll_has_mask_only_and_true_no_vggt_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport_counter = iter((1.0, 11.0, 2.0, 12.0))
    monkeypatch.setattr(
        eval_cli,
        "_build_eval_transport",
        lambda **_: _fake_transport(next(transport_counter)),
    )
    batch = {
        "rgb_hr_sequence": torch.zeros((1, 3, 3, 4, 4)),
        "disparity_ffs_hr_px_sequence": torch.ones((1, 3, 1, 2, 2)),
        "confidence_ffs_sequence": torch.ones((1, 3, 1, 2, 2)),
        "valid_ffs_sequence": torch.ones((1, 3, 1, 2, 2), dtype=torch.bool),
        "vggt_disparity_hr_px_sequence": torch.ones((1, 3, 1, 2, 2)) * 7,
        "disparity_vggt_hr_px_sequence": torch.ones((1, 3, 1, 2, 2)) * 7,
        "confidence_vggt_sequence": torch.ones((1, 3, 1, 2, 2)),
        "valid_vggt_sequence": torch.ones((1, 3, 1, 2, 2), dtype=torch.bool),
        "static_prior_valid_sequence": torch.ones((1, 3), dtype=torch.bool),
        "temporal_pose_valid_sequence": torch.ones((1, 3), dtype=torch.bool),
    }
    model = _TemporalSpyModel()
    config = OmegaConf.create(_temporal_checkpoint_config())
    result = eval_cli._run_temporal_endpoint_ablation(model, batch, config=config)

    assert len(model.calls) == 9
    for time_index in range(3):
        on, mask_off, no_vggt = model.calls[3 * time_index : 3 * time_index + 3]
        assert on["vggt_sum"] == 28.0 and on["valid_vggt"] is True
        assert mask_off["vggt_sum"] == 28.0 and mask_off["valid_vggt"] is False
        assert no_vggt["vggt_sum"] == 0.0 and no_vggt["valid_vggt"] is False
        assert on["history_ptr"] == mask_off["history_ptr"]
        if time_index > 0:
            assert no_vggt["history_ptr"] != on["history_ptr"]
    assert result.shared_transport.disparity_history_hr_px[0, 0, 0, 0] == 2.0
    assert result.source_mask_off_transport is result.shared_transport
    assert result.no_vggt_transport.disparity_history_hr_px[0, 0, 0, 0] == 12.0
    assert result.reference_transport is None


def test_temporal_v2_unroll_uses_branch_memory_warped_hidden_and_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_lengths: list[int] = []
    values = iter((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def fake_topk(**kwargs: object) -> train.TemporalTransport:
        memory = kwargs["memory"]
        assert isinstance(memory, list)
        memory_lengths.append(len(memory))
        return _fake_v2_transport(next(values))

    reference_calls: list[int] = []

    def fake_reference(**kwargs: object) -> train.ReferenceTemporalWarp:
        time_index = kwargs["time_index"]
        assert isinstance(time_index, int)
        reference_calls.append(time_index)
        disparity = torch.full((1, 1, 4, 4), 10.0)
        prediction = kwargs["previous_prediction_disparity_hr_px"]
        assert isinstance(prediction, torch.Tensor)
        valid = torch.ones_like(disparity, dtype=torch.bool)
        return train.ReferenceTemporalWarp(
            disparity_hr_px=disparity,
            prediction_disparity_hr_px=prediction,
            valid_mask_hr=valid,
            visibility_mask_hr=valid,
            collision_mask_hr=torch.zeros_like(valid),
        )

    monkeypatch.setattr(eval_cli, "_build_eval_topk_transport", fake_topk)
    monkeypatch.setattr(
        eval_cli, "_build_eval_reference_transport", fake_reference
    )
    model = _TemporalSpyModel()
    result = eval_cli._run_temporal_endpoint_ablation(
        model,
        _v2_unroll_batch(),
        config=_v2_temporal_config(),
    )

    assert memory_lengths == [1, 1, 1, 2, 2, 2]
    assert reference_calls == [1, 1, 1, 2, 2, 2]
    assert len(model.calls) == 9
    assert all(not call["topk_history"] for call in model.calls[:3])
    assert [call["hidden_mean"] for call in model.calls[3:6]] == [1.0, 2.0, 3.0]
    assert [call["hidden_mean"] for call in model.calls[6:9]] == [4.0, 5.0, 6.0]
    assert all(call["topk_history"] for call in model.calls[3:])
    assert len({model.calls[index]["history_ptr"] for index in (6, 7, 8)}) == 3
    assert result.shared_transport.disparity_history_hr_px[0, 0, 0, 0] == 4.0
    assert result.source_mask_off_transport.disparity_history_hr_px[
        0, 0, 0, 0
    ] == 5.0
    assert result.no_vggt_transport.disparity_history_hr_px[0, 0, 0, 0] == 6.0
    assert result.reference_transport is not None


class _SpatialV2SpyModel(nn.Module):
    def forward(
        self,
        rgb_hr: torch.Tensor,
        disparity_ffs_hr_px: torch.Tensor,
        confidence_ffs: torch.Tensor,
        **_: object,
    ) -> ModelOutput:
        disparity = torch.nn.functional.interpolate(
            disparity_ffs_hr_px, scale_factor=2, mode="nearest"
        )
        return ModelOutput(
            disparity_hr_px=disparity,
            disparity_raw_hr_px=disparity,
            source_weights=torch.zeros((1, 3, 2, 2)),
            log_variance=torch.zeros_like(disparity),
            uncertainty=torch.ones_like(disparity),
            hidden_state=(
                torch.ones((1, 1, 2, 2)),
                torch.ones((1, 1, 2, 2)),
            ),
            anchor_gate=torch.ones_like(disparity),
            source_valid_mask=torch.ones((1, 3, 2, 2), dtype=torch.bool),
        )


def test_spatial_endpoint_v2_returns_adjacent_teacher_warp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_lengths: list[int] = []
    reference_times: list[int] = []

    def fake_topk(**kwargs: object) -> train.TemporalTransport:
        memory = kwargs["memory"]
        assert isinstance(memory, list)
        memory_lengths.append(len(memory))
        return _fake_v2_transport(float(len(memory)))

    def fake_reference(**kwargs: object) -> train.ReferenceTemporalWarp:
        time_index = kwargs["time_index"]
        assert isinstance(time_index, int)
        reference_times.append(time_index)
        disparity = torch.full((1, 1, 4, 4), float(time_index + 10))
        prediction = kwargs["previous_prediction_disparity_hr_px"]
        assert isinstance(prediction, torch.Tensor)
        valid = torch.ones_like(disparity, dtype=torch.bool)
        return train.ReferenceTemporalWarp(
            disparity_hr_px=disparity,
            prediction_disparity_hr_px=prediction,
            valid_mask_hr=valid,
            visibility_mask_hr=valid,
            collision_mask_hr=torch.zeros_like(valid),
        )

    monkeypatch.setattr(eval_cli, "_build_eval_topk_transport", fake_topk)
    monkeypatch.setattr(
        eval_cli, "_build_eval_reference_transport", fake_reference
    )
    result = eval_cli._run_spatial_endpoint(
        _SpatialV2SpyModel(),
        _v2_unroll_batch(),
        config=_v2_temporal_config(),
    )

    assert memory_lengths == [1, 2]
    assert reference_times == [1, 2]
    assert result.reference_transport is not None
    assert result.reference_transport.disparity_hr_px.unique().item() == 12.0


def _write_cache_receipt(
    root: Path, manifest: Path, component: str, identity: object
) -> None:
    (root / "run_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity.to_dict(),
                "manifest_sha256": sha256_file(manifest),
                "selected_records": 1,
                "written_records": 1,
                "reused_records": 0,
                "component": component,
            }
        ),
        encoding="utf-8",
    )


def test_stage_a_cli_writes_bilinear_and_t1_csv_rows(tmp_path: Path) -> None:
    manifest, observation_root, teacher_root, *_ = _make_cached_example(tmp_path)
    assert teacher_root is not None
    observation_identity = _identity("ffs-observation")
    teacher_identity = _identity("ffs-teacher")
    _write_cache_receipt(
        observation_root, manifest, "ffs-observation", observation_identity
    )
    _write_cache_receipt(teacher_root, manifest, "ffs-teacher", teacher_identity)

    config = train.resolve_config(
        Path(__file__).parents[1] / "configs" / "mvp_x2.yaml"
    )
    OmegaConf.update(
        config, "data.observation_cache_identity", observation_identity.to_dict()
    )
    OmegaConf.update(config, "data.teacher_cache_identity", teacher_identity.to_dict())
    OmegaConf.update(config, "data.manifest_path", str(manifest.resolve()))
    OmegaConf.update(
        config, "data.observation_cache_root", str(observation_root.resolve())
    )
    OmegaConf.update(config, "data.teacher_cache_root", str(teacher_root.resolve()))
    model = train.build_model(config)
    checkpoint = tmp_path / "spatial.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": model.state_dict(),
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "rng_states": {},
            "parameter_count": model.trainable_parameter_count,
            "step": 1,
            "config": OmegaConf.to_container(config, resolve=True),
            "git_hash": "a" * 40,
        },
        checkpoint,
    )
    output = tmp_path / "evaluation"
    args = eval_cli.build_parser().parse_args(
        [
            "--config",
            str(Path(__file__).parents[1] / "configs" / "mvp_x2.yaml"),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--observation-cache-root",
            str(observation_root),
            "--teacher-cache-root",
            str(teacher_root),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--visualization-samples",
            "0",
            "data.hr_crop=[8,12]",
        ]
    )
    assert eval_cli.run(args) == 0
    rows = (output / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5
    assert rows[1].startswith("bilinear,")
    assert rows[2].startswith("bilinear_clamp0,")
    assert rows[3].startswith("T1,")
    assert rows[4].startswith("T1_clamp0,")
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert report["status"] == "INTERMEDIATE_CHECKPOINT_EVALUATION_COMPLETE"
    assert "failure_sample_bundles" not in report
    assert not (output / "failures").exists()
    assert report["claims"]["coverage_eligible"] is True
    assert report["claims"]["final_training_checkpoint"] is False
    assert report["claims"]["final_acceptance_eligible"] is False
    assert report["claims"]["acceptance_eligible"] is False
    assert report["checkpoint_training_completion"]["actual_step"] == 1
    assert "temporal_flicker_video" not in report
    assert report["runtime_v3"]["contract_version"] == (
        "matched_candidate_forward_runtime_v1"
    )
    assert report["runtime_v3"]["timing_backend"] == "time.perf_counter"
    assert report["runtime_v3"]["model_forward_calls"] == 1
    assert report["runtime_v3"]["model_forward_latency_ms_mean"] > 0.0
    assert report["runtime_v3"]["cuda_peak_allocated_bytes"] is None
    per_record = (output / "per_record_metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(per_record) == 1
    assert report["per_record_metrics"]["records"] == 1
    assert report["per_record_metrics"]["sha256"] == sha256_file(
        output / "per_record_metrics.jsonl"
    )
