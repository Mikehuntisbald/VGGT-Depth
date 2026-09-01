from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from tools.decide_v3_conditioning import decide_manifest


SEEDS = (42, 43, 44)
ARMS = ("A0", "A1", "A2", "A3", "B0", "B1")
OUTPUT_METRICS = (
    "output_negative_rate",
    "output_invalid_rate",
    "output_nan_rate",
    "output_infinite_rate",
)
SWITCHES = {
    "A0": (False, False, False),
    "A1": (True, False, False),
    "A2": (False, True, False),
    "A3": (True, True, False),
    "B0": (True, True, False),
    "B1": (True, True, True),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float) -> dict[str, Any]:
    return {"valid": True, "count": 100, "numerator": value * 100, "value": value}


def _values(
    arm: str, *, unsafe_rays: bool = False, temporal_candidate: float = 0.45
) -> dict[str, float]:
    low = {"A0": 1.0, "A1": 0.98, "A2": 0.99, "A3": 0.96}.get(arm, 0.9)
    values = {
        "low_confidence_epe_px": low,
        "epe_px": 1.01 if unsafe_rays and arm == "A1" else 1.0,
        "boundary_epe_px": 1.0,
        "trusted_region_epe_px": 1.0,
        "invalid_region_completeness": 0.5,
        "output_negative_rate": 0.001,
        "output_invalid_rate": 0.001,
        "output_nan_rate": 0.0,
        "output_infinite_rate": 0.0,
    }
    if arm in ("B0", "B1"):
        values["temporal_residual_error_native_px"] = (
            0.5 if arm == "B0" else temporal_candidate
        )
    return values


def _write_arm(
    root: Path,
    *,
    seed: int,
    arm: str,
    write_records: bool,
    unsafe_rays: bool = False,
    temporal_eligible_records: int | None = None,
    temporal_candidate: float = 0.45,
    validation_manifest: Path,
    runtime_scale: float = 1.0,
) -> dict[str, str]:
    directory = root / f"seed_{seed}" / arm
    directory.mkdir(parents=True)
    stage_b = arm.startswith("B")
    method = "T3_VGGT" if stage_b else "T1"
    values = _values(
        arm,
        unsafe_rays=unsafe_rays,
        temporal_candidate=temporal_candidate,
    )
    method_metrics = {name: _metric(value) for name, value in values.items()}
    if stage_b:
        method_metrics["temporal_residual_error_paired_px"] = {
            "valid": False,
            "count": 0,
            "numerator": 0.0,
            "value": None,
        }
    use_rays, use_stereo, use_temporal = SWITCHES[arm]
    report: dict[str, Any] = {
        "stage": "T3_CAUSAL_STAGE_B" if stage_b else "T1_SPATIAL_ONLY",
        "records_evaluated": 238 if stage_b else 244,
        "crop_mode": "full",
        "manifest_path": str(validation_manifest.resolve()),
        "claims": {
            "final_acceptance_eligible": True,
            "full_validation_selection": True,
        },
        "runtime_v3": {
            "contract_version": "matched_candidate_forward_runtime_v1",
            "timing_backend": "torch.cuda.Event",
            "model_forward_calls": 244 if not stage_b else 714,
            "model_forward_latency_ms_mean": 10.0 * runtime_scale,
            "cuda_peak_allocated_bytes": 1_000.0 * runtime_scale,
            "cuda_peak_reserved_bytes": 2_000.0 * runtime_scale,
        },
        "cache_identities": {
            "observation": {"id": "validation-observation"},
            "teacher": {"id": "validation-teacher"},
        },
        "derived_cache_lineage": {"id": "validation-derived"} if stage_b else None,
        "device": "cuda:0",
        "resolved_config": {
            "seed": seed,
            "calibration_conditioning_v3": {
                "enabled": True,
                "protocol_version": "dense_rays_factorized_pose_v3",
                "use_rays": use_rays,
                "use_stereo_pose": use_stereo,
                "use_temporal_pose": use_temporal,
            },
            "data": {
                "sequence_length": 3 if stage_b else 1,
                "calibration_sidecar_lineage": {"id": "validation-calibration"},
            },
            "eval": {"crop_mode": "full"},
        },
        "methods": {method: method_metrics},
        "comparisons": {},
    }
    if stage_b:
        report["comparisons"] = {
            "T3_vs_T1_temporal": {
                "valid": True,
                "relative_change_percent": -12.0,
            }
        }
    metrics = directory / "metrics.json"
    metrics.write_text(json.dumps(report), encoding="utf-8")
    records = directory / "per_record_metrics.jsonl"
    if write_records:
        count = 238 if stage_b else 244
        with records.open("w", encoding="utf-8") as handle:
            for index in range(count):
                record_values: dict[str, float | None] = dict(values)
                if (
                    stage_b
                    and temporal_eligible_records is not None
                    and index >= temporal_eligible_records
                ):
                    record_values["temporal_residual_error_native_px"] = None
                handle.write(
                    json.dumps(
                        {
                            "record_id": f"record-{index:03d}",
                            "metrics": record_values,
                        }
                    )
                    + "\n"
                )
    return {
        "metrics_json": str(metrics),
        "metrics_sha256": _sha256(metrics),
        "per_record_jsonl": str(records),
        "per_record_jsonl_sha256": _sha256(records) if records.is_file() else None,
    }


def _manifest(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
    write_records: bool = True,
    unsafe_rays: bool = False,
    unique_calibrations: int = 1,
    temporal_pose_varies: bool = True,
    temporal_eligible_records: int | None = None,
    temporal_candidate: float = 0.45,
    runtime_regression_arm: str | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    validation_manifest = tmp_path / "validation.jsonl"
    validation_manifest.write_text("{}\n", encoding="utf-8")
    evidence: dict[str, Any] = {}
    for seed in seeds:
        evidence[str(seed)] = {
            arm: _write_arm(
                tmp_path,
                seed=seed,
                arm=arm,
                write_records=write_records,
                unsafe_rays=unsafe_rays,
                temporal_eligible_records=temporal_eligible_records,
                temporal_candidate=temporal_candidate,
                validation_manifest=validation_manifest,
                runtime_scale=(1.06 if arm == runtime_regression_arm else 1.0),
            )
            for arm in ARMS
        }
    payload = {
        "schema_version": 1,
        "component": "v3-experiment-decision-inputs",
        "expected_counts": {"stage_a_records": 244, "stage_b_windows": 238},
        "identifiability": {
            "unique_static_stereo_calibrations": unique_calibrations,
            "temporal_pose_varies": temporal_pose_varies,
        },
        "validation_manifest": {
            "path": str(validation_manifest.resolve()),
            "sha256": _sha256(validation_manifest),
        },
        "seeds": evidence,
    }
    path = tmp_path / "decision_inputs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_three_seed_paired_evidence_selects_rays_and_temporal_only(
    tmp_path: Path,
) -> None:
    report = decide_manifest(_manifest(tmp_path), bootstrap_replicates=200)

    assert report["decision"] == "GO"
    assert report["components"]["rays"]["status"] == "GO"
    assert report["components"]["static_stereo_pose"]["status"] == "NOT_IDENTIFIABLE"
    assert report["components"]["temporal_pose"]["status"] == "GO"
    assert report["recommended_switches"] == {
        "use_rays": True,
        "use_stereo_pose": True,
        "use_temporal_pose": True,
    }
    assert report["promotion_recipe"]["exact_evaluated_arm"] == "B1"
    assert "unidentifiable background" in report["promotion_recipe"][
        "claim_boundary"
    ]
    assert report["components"]["rays"]["bootstrap"]["ci95_upper"] < 0
    assert report["components"]["temporal_pose"]["bootstrap"]["ci95_upper"] < 0


def test_missing_per_record_bootstrap_fails_closed(tmp_path: Path) -> None:
    report = decide_manifest(
        _manifest(tmp_path, write_records=False), bootstrap_replicates=200
    )

    assert report["decision"] == "NO-GO"
    assert report["components"]["rays"]["status"] == "NO-GO"
    assert report["components"]["temporal_pose"]["status"] == "NO-GO"
    assert any(
        "per-record bootstrap data is unavailable" in reason
        for reason in report["components"]["rays"]["reasons"]
    )


def test_seed42_alone_is_screening_not_final_evidence(tmp_path: Path) -> None:
    report = decide_manifest(
        _manifest(tmp_path, seeds=(42,)), bootstrap_replicates=200
    )

    assert report["decision"] == "NO-GO"
    assert report["screening"]["continue_additional_seeds"] is True
    assert report["screening"]["final_evidence"] is False
    assert report["components"]["rays"]["status"] == "NO-GO"
    assert "exactly seeds 42,43,44" in report["components"]["rays"]["reasons"][0]


def test_safety_regression_overrides_primary_improvement(tmp_path: Path) -> None:
    report = decide_manifest(
        _manifest(tmp_path, unsafe_rays=True), bootstrap_replicates=200
    )

    rays = report["components"]["rays"]
    assert rays["status"] == "NO-GO"
    assert rays["aggregate_by_seed"]["42"]["checks"]["epe_px_degradation"] is False
    assert report["decision"] == "NO-GO"


def test_unasserted_temporal_variation_is_not_identifiable(tmp_path: Path) -> None:
    report = decide_manifest(
        _manifest(tmp_path, temporal_pose_varies=False), bootstrap_replicates=200
    )

    assert report["components"]["temporal_pose"]["status"] == "NOT_IDENTIFIABLE"
    assert report["components"]["rays"]["status"] == "GO"
    assert report["decision"] == "GO"
    assert report["promotion_recipe"]["exact_evaluated_arm"] == "A1"


def test_temporal_bootstrap_uses_valid_intersection_with_conservative_floor(
    tmp_path: Path,
) -> None:
    insufficient = decide_manifest(
        _manifest(tmp_path / "insufficient", temporal_eligible_records=29),
        bootstrap_replicates=200,
    )
    temporal = insufficient["components"]["temporal_pose"]
    assert temporal["status"] == "NO-GO"
    assert temporal["paired_coverage_by_seed"]["42"] == {
        "eligible_records": 29,
        "total_records": 238,
        "eligible_fraction": 29 / 238,
    }

    sufficient = decide_manifest(
        _manifest(tmp_path / "sufficient", temporal_eligible_records=30),
        bootstrap_replicates=200,
    )
    assert sufficient["components"]["temporal_pose"]["status"] == "GO"
    assert sufficient["components"]["temporal_pose"]["paired_coverage_by_seed"][
        "42"
    ]["eligible_records"] == 30


def test_temporal_pose_requires_five_percent_native_residual_gain(
    tmp_path: Path,
) -> None:
    report = decide_manifest(
        _manifest(tmp_path, temporal_candidate=0.48), bootstrap_replicates=200
    )

    temporal = report["components"]["temporal_pose"]
    assert temporal["status"] == "NO-GO"
    assert temporal["aggregate_by_seed"]["42"]["checks"][
        "primary_improvement"
    ] is False
    assert report["thresholds"]["min_temporal_pose_improvement_percent"] == 5.0


def test_runtime_regression_over_five_percent_blocks_candidate(tmp_path: Path) -> None:
    report = decide_manifest(
        _manifest(tmp_path, runtime_regression_arm="A1"), bootstrap_replicates=200
    )

    rays = report["components"]["rays"]
    assert rays["status"] == "NO-GO"
    assert rays["aggregate_by_seed"]["42"]["checks"][
        "runtime_model_forward_latency_ms_mean_degradation"
    ] is False
    assert report["thresholds"]["max_runtime_degradation_percent"] == 5.0


def test_negative_error_metric_is_rejected_even_with_matching_artifact_hash(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["seeds"]["42"]["A1"]
    metrics_path = Path(entry["metrics_json"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["methods"]["T1"]["low_confidence_epe_px"]["value"] = -1.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    entry["metrics_sha256"] = _sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = decide_manifest(manifest_path, bootstrap_replicates=200)

    assert report["decision"] == "NO-GO"
    assert any("must be non-negative" in error for error in report["input_errors"])
    assert report["recommended_switches"] == {
        "use_rays": False,
        "use_stereo_pose": False,
        "use_temporal_pose": False,
    }


def test_identifiable_static_go_without_temporal_go_selects_exact_a3(
    tmp_path: Path,
) -> None:
    report = decide_manifest(
        _manifest(
            tmp_path,
            unique_calibrations=2,
            temporal_candidate=0.48,
        ),
        bootstrap_replicates=200,
    )

    assert report["components"]["static_stereo_pose"]["status"] == "GO"
    assert report["components"]["temporal_pose"]["status"] == "NO-GO"
    assert report["promotion_recipe"]["exact_evaluated_arm"] == "A3"
    assert report["recommended_switches"] == {
        "use_rays": True,
        "use_stereo_pose": True,
        "use_temporal_pose": False,
    }
