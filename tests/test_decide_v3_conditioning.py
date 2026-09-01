from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

from tools.decide_v3_conditioning import _bootstrap_ci, decide_manifest


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


def _validation_records() -> list[dict[str, Any]]:
    return [
        {
            "sequence_id": "validation-sequence",
            "frame_id": index,
            "timestamp": float(index),
            "left_path": f"left/{index}.png",
            "right_path": f"right/{index}.png",
            "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 25.0], [0.0, 0.0, 1.0]],
            "baseline_m": 0.1,
            "gt_disparity_path": None,
        }
        for index in range(244)
    ]


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
    validation_records: list[dict[str, Any]],
    derived_lineage: Mapping[str, Any],
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
        "derived_cache_lineage": dict(derived_lineage) if stage_b else None,
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
    records = directory / "per_record_metrics.jsonl"
    if write_records:
        selected_records = validation_records[6:] if stage_b else validation_records
        with records.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(selected_records):
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
                            "record_id": (
                                f"{record['sequence_id']}/{record['frame_id']}"
                            ),
                            "sequence_id": record["sequence_id"],
                            "frame_id": record["frame_id"],
                            "timestamp": record["timestamp"],
                            "manifest_index": int(record["frame_id"]),
                            "method": method,
                            "metrics": record_values,
                        }
                    )
                    + "\n"
                )
    report["per_record_metrics"] = {
        "path": str(records.resolve()),
        "sha256": _sha256(records) if records.is_file() else None,
        "records": 238 if stage_b else 244,
        "paired_bootstrap_unit": "sequence_id/frame_id",
    }
    metrics = directory / "metrics.json"
    metrics.write_text(json.dumps(report), encoding="utf-8")
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
    validation_records = _validation_records()
    validation_manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in validation_records),
        encoding="utf-8",
    )
    derived_root = tmp_path / "validation_derived"
    derived_root.mkdir()
    derived_manifest = derived_root / "cache_manifest.jsonl"
    derived_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sequence_id": record["sequence_id"],
                    "frame_id": record["frame_id"],
                    "target_manifest_index": record["frame_id"],
                }
            )
            + "\n"
            for record in validation_records[4:]
        ),
        encoding="utf-8",
    )
    derived_receipt = derived_root / "run_receipt.json"
    derived_receipt.write_text(
        json.dumps(
            {
                "output": {"cache_manifest_sha256": _sha256(derived_manifest)}
            }
        ),
        encoding="utf-8",
    )
    derived_lineage = {
        "derived_cache_root": str(derived_root.resolve()),
        "run_receipt_sha256": _sha256(derived_receipt),
        "cache_manifest_sha256": _sha256(derived_manifest),
    }
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
                validation_records=validation_records,
                derived_lineage=derived_lineage,
                runtime_scale=(1.06 if arm == runtime_regression_arm else 1.0),
            )
            for arm in ARMS
        }
    identifiability: dict[str, Any] = {
        "unique_static_stereo_calibrations": unique_calibrations,
        "temporal_pose_varies": temporal_pose_varies,
    }
    if temporal_pose_varies:
        formal_ids = [
            f"{record['sequence_id']}/{record['frame_id']}"
            for record in validation_records[6:]
        ]
        audit_path = tmp_path / "temporal_pose_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "component": "v3-temporal-pose-variation-audit",
                    "status": "PASS",
                    "temporal_pose_varies": True,
                    "counts": {
                        "formal_temporal_endpoints": 238,
                        "formal_windows": 238,
                        "formal_pose_valid_windows": 238,
                    },
                    "ages": {"1": {"varies": True}, "2": {"varies": True}},
                    "formal_endpoint_binding": {
                        "available": True,
                        "record_ids": formal_ids,
                        "record_ids_sha256": hashlib.sha256(
                            json.dumps(
                                formal_ids,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest(),
                        "pose_valid_record_ids": formal_ids,
                    },
                    "inputs": {
                        "derived_root": str(derived_root.resolve()),
                        "run_receipt_sha256": _sha256(derived_receipt),
                        "cache_manifest_sha256": _sha256(derived_manifest),
                        "validation_manifest": {
                            "path": str(validation_manifest.resolve()),
                            "sha256": _sha256(validation_manifest),
                            "records": 244,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        identifiability["temporal_pose_variation_audit"] = {
            "path": str(audit_path.resolve()),
            "sha256": _sha256(audit_path),
        }
    payload = {
        "schema_version": 1,
        "component": "v3-experiment-decision-inputs",
        "expected_counts": {"stage_a_records": 244, "stage_b_windows": 238},
        "identifiability": identifiability,
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


def test_bare_temporal_boolean_never_promotes_b1(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["identifiability"].pop("temporal_pose_variation_audit")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = decide_manifest(manifest_path, bootstrap_replicates=200)

    assert report["components"]["temporal_pose"]["status"] == "NOT_IDENTIFIABLE"
    assert report["promotion_recipe"]["exact_evaluated_arm"] == "A1"
    assert report["recommended_switches"]["use_temporal_pose"] is False


def test_per_record_file_must_match_metrics_internal_binding(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["seeds"]["42"]["A1"]
    original = Path(entry["per_record_jsonl"])
    replacement = original.with_name("replacement.jsonl")
    replacement.write_bytes(original.read_bytes())
    entry["per_record_jsonl"] = str(replacement)
    entry["per_record_jsonl_sha256"] = _sha256(replacement)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = decide_manifest(manifest_path, bootstrap_replicates=200)

    assert report["decision"] == "NO-GO"
    assert any("metrics/per-record binding differs" in value for value in report["input_errors"])


def test_per_record_identity_must_belong_to_exact_manifest(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["seeds"]["42"]["A1"]
    records_path = Path(entry["per_record_jsonl"])
    rows = records_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["sequence_id"] = "foreign-sequence"
    first["record_id"] = f"foreign-sequence/{first['frame_id']}"
    rows[0] = json.dumps(first)
    records_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    new_sha = _sha256(records_path)
    entry["per_record_jsonl_sha256"] = new_sha
    metrics_path = Path(entry["metrics_json"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["per_record_metrics"]["sha256"] = new_sha
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    entry["metrics_sha256"] = _sha256(metrics_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = decide_manifest(manifest_path, bootstrap_replicates=200)

    assert report["decision"] == "NO-GO"
    assert "outside exact validation selection" in " ".join(
        report["components"]["rays"]["reasons"]
    )


def test_bootstrap_jointly_resamples_clusters_across_fixed_seed_strata() -> None:
    a = ("sequence", 10)
    b = ("sequence", 11)
    report = _bootstrap_ci(
        {
            42: {a: -10.0, b: 10.0},
            43: {a: 10.0, b: -10.0},
            44: {a: 0.0, b: 0.0},
        },
        replicates=200,
        random_seed=7,
    )

    assert report["cluster_unit"] == "sequence_id/source_frame_id"
    assert report["common_clusters"] == 2
    assert report["ci95_lower"] == 0.0
    assert report["ci95_upper"] == 0.0


def test_output_bad_rate_may_increase_but_must_remain_below_absolute_gate(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for seed in SEEDS:
        entry = payload["seeds"][str(seed)]["A1"]
        metrics_path = Path(entry["metrics_json"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric = metrics["methods"]["T1"]["output_invalid_rate"]
        metric.update({"value": 0.002, "numerator": 0.2})
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        entry["metrics_sha256"] = _sha256(metrics_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = decide_manifest(manifest_path, bootstrap_replicates=200)

    assert report["components"]["rays"]["status"] == "GO"


def test_temporal_audit_valid_subset_must_belong_to_formal_endpoints(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = payload["identifiability"]["temporal_pose_variation_audit"]
    audit_path = Path(identity["path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["formal_endpoint_binding"]["pose_valid_record_ids"][0] = "foreign/1"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    identity["sha256"] = _sha256(audit_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = decide_manifest(manifest_path, bootstrap_replicates=200)

    assert report["decision"] == "NO-GO"
    assert any("formal endpoint identities differ" in value for value in report["input_errors"])


def test_temporal_audit_rehashes_live_derived_manifest(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = payload["identifiability"]["temporal_pose_variation_audit"]
    audit = json.loads(Path(identity["path"]).read_text(encoding="utf-8"))
    derived_manifest = Path(audit["inputs"]["derived_root"]) / "cache_manifest.jsonl"
    derived_manifest.write_text("drift\n", encoding="utf-8")

    report = decide_manifest(manifest_path, bootstrap_replicates=200)

    assert report["decision"] == "NO-GO"
    assert any("derived inputs changed" in value for value in report["input_errors"])
