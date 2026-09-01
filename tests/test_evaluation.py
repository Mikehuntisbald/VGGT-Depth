from __future__ import annotations

import math
import copy
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert result.no_vggt_transport.disparity_history_hr_px[0, 0, 0, 0] == 12.0


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
    assert report["claims"]["coverage_eligible"] is True
    assert report["claims"]["final_training_checkpoint"] is False
    assert report["claims"]["final_acceptance_eligible"] is False
    assert report["claims"]["acceptance_eligible"] is False
    assert report["checkpoint_training_completion"]["actual_step"] == 1
