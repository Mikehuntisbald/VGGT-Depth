from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_spring_v3_1_ffs import (
    ARM_ORDER,
    EXPECTED_ENDPOINT_COUNT,
    EXPECTED_ENDPOINT_ID_SHA256,
    Paths,
    SpringV31Error,
    _cache_jobs,
    _eval_jobs,
    _geometry_jobs,
    _initializer_jobs,
    _matrix_receipt,
    _model_completion_evidence,
    _parse_arms,
    _parse_devices,
    _primary_metric_evidence,
    _runtime_evidence,
    _temporal_train_jobs,
    _validate_cache_inventory,
    _write_f7_blocked,
    build_parser,
    run,
)
from data.cache_dataset import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _args(tmp_path: Path):
    args = build_parser().parse_args(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--spring-root",
            str(tmp_path / "spring"),
            "--output-root",
            str(tmp_path / "run"),
            "--devices",
            "0,1,2,3",
        ]
    )
    args.devices = _parse_devices(args.devices)
    for name in (
        "project_root",
        "spring_root",
        "output_root",
        "ffs_checkpoint",
        "ffs_repo",
        "vggt_checkpoint",
        "vggt_repo",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args, Paths.from_args(args)


def _by_name(jobs):
    return {job.name: job for job in jobs}


def test_protocol_constants_freeze_seed42_common_domain() -> None:
    assert ARM_ORDER == ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
    assert EXPECTED_ENDPOINT_COUNT == 1302
    assert EXPECTED_ENDPOINT_ID_SHA256 == (
        "aa6ba30295b8d5ab0e1b4326a14fae61f9c8ec42641801cd8442097bc3ab5b57"
    )
    assert _parse_arms(["F6,F2", "F4"]) == ("F2", "F4", "F6")
    assert _parse_devices("3,1") == ("3", "1")
    with pytest.raises(SpringV31Error, match="duplicates"):
        _parse_devices("0,0")


def test_runner_rejects_every_seed_except_42(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--seed", "43", "--output-root", str(tmp_path / "run"), "--dry-run"]
    )
    with pytest.raises(SpringV31Error, match="only permits seed 42"):
        run(args)


def test_cache_plan_separates_full_and_half_ffs_domains(tmp_path: Path) -> None:
    args, paths = _args(tmp_path)
    jobs = _by_name(_cache_jobs(paths, args, ARM_ORDER))

    assert len(jobs) == 7
    full = jobs["cache_validation_full_ffs"]
    half = jobs["cache_validation_half_ffs"]
    assert (
        "--scale" in full.command
        and full.command[full.command.index("--scale") + 1] == "1"
    )
    assert (
        "--max-disp" in full.command
        and full.command[full.command.index("--max-disp") + 1] == "416"
    )
    assert full.expected_output.parent.name == "observation_full_resolution_maxdisp416"
    assert half.command[half.command.index("--scale") + 1] == "2"
    assert half.command[half.command.index("--max-disp") + 1] == "192"
    assert half.expected_output.parent.name == "observation"
    assert jobs["cache_train_spring_gt"].expected_output.parent.name == "teacher"
    gt_command = jobs["cache_train_spring_gt"].command
    assert gt_command[gt_command.index("--cache-dtype") + 1] == "float16"

    f1_jobs = _by_name(_cache_jobs(paths, args, ("F1",)))
    assert set(f1_jobs) == {"cache_validation_half_ffs"}


def test_geometry_and_training_dag_preserve_lineage_isolation(tmp_path: Path) -> None:
    args, paths = _args(tmp_path)
    geometry = _by_name(_geometry_jobs(paths, args, ARM_ORDER))
    initializers = _by_name(_initializer_jobs(paths, args, ARM_ORDER))
    temporal = _by_name(_temporal_train_jobs(paths, args, ARM_ORDER))

    assert len(geometry) == 4
    assert (
        "--rectified-calibration-sidecar"
        not in geometry["derive_train_legacy_v2_control"].command
    )
    assert (
        geometry["derive_train_legacy_v2_control"]
        .command[1]
        .endswith("tools/build_spring_gt_geometry.py")
    )
    assert "--sequence-warmup" in geometry["derive_train_legacy_v2_control"].command
    assert paths.train_legacy_derived.name == "derived_f3_v2_gt_pose_no_depth"
    assert paths.train_calibrated_derived.name == "derived_v31_calibrated_vggt"
    assert (
        "--rectified-calibration-sidecar"
        in geometry["derive_train_calibrated_v3_1"].command
    )

    assert set(initializers) == {"train_F2", "train_F3_stage_a_control"}
    assert "configs/spring_v3_1/F2.yaml" in " ".join(initializers["train_F2"].command)
    assert "F3_stage_a_control.yaml" in " ".join(
        initializers["train_F3_stage_a_control"].command
    )

    f2 = str(paths.arm_train("F2") / "final.pt")
    f3_control = str(paths.f3_initializer_train / "final.pt")
    assert (
        temporal["train_F3"].command[
            temporal["train_F3"].command.index("--init-from") + 1
        ]
        == f3_control
    )
    for arm in ("F4", "F5", "F6"):
        job = temporal[f"train_{arm}"]
        assert job.command[job.command.index("--init-from") + 1] == f2
        assert str(paths.train_calibrated_derived) in job.command


def test_eval_plan_uses_one_endpoint_crop_and_expected_sidecars(tmp_path: Path) -> None:
    args, paths = _args(tmp_path)
    jobs = _by_name(_eval_jobs(paths, args, ARM_ORDER))

    assert len(jobs) == 7
    for arm in ARM_ORDER[:-1]:
        command = jobs[f"eval_{arm}"].command
        assert "--spring-endpoint-index-list" in command
        assert command[command.index("--spring-endpoint-index-list") + 1] == str(
            paths.endpoint_index
        )
        assert command[command.index("--crop-mode") + 1] == "fixed"
        origin = command.index("--crop-origin")
        assert command[origin + 1 : origin + 3] == ("576", "348")
        if arm in ("F0", "F1"):
            # The baseline evaluator always emits its native Spring fields;
            # it intentionally has no opt-in CLI switch.
            assert "--spring-native-metrics" not in command
        else:
            assert "--spring-native-metrics" in command
        assert not any("spring_corrected" in value for value in command)
    for arm in ("F2", "F4", "F5", "F6"):
        assert "--calibration-sidecar" in jobs[f"eval_{arm}"].command
    assert "--calibration-sidecar" not in jobs["eval_F3"].command


def test_f7_is_only_materialized_as_optional_blocked(tmp_path: Path) -> None:
    args, paths = _args(tmp_path)
    payload = _write_f7_blocked(paths)

    assert payload["status"] == "OPTIONAL_BLOCKED"
    assert payload["optional"] is True
    assert not paths.arm_train("F7").exists()
    assert (paths.output_root / "arms/F7/status.json").is_file()


def test_cache_inventory_rejects_unlisted_pt_files(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for frame_id in range(1, 6):
        rows.append(
            {
                "sequence_id": "0005",
                "frame_id": frame_id,
                "timestamp": float(frame_id - 1),
                "left_path": f"left/{frame_id}.png",
                "right_path": f"right/{frame_id}.png",
                "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 25.0], [0.0, 0.0, 1.0]],
                "baseline_m": 0.065,
                "gt_disparity_path": f"disp/{frame_id}.dsp5",
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    root = tmp_path / "derived"
    canonical = root / "0005" / "5.pt"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical")
    (root / "cache_manifest.jsonl").write_text(
        json.dumps(
            {
                "target_manifest_index": 4,
                "sequence_id": "0005",
                "frame_id": 5,
                "cache_path": str(canonical),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = _validate_cache_inventory(root, manifest, sequence_warmup=4)
    assert evidence["canonical_pt_set"] is True

    stale = root / "0005" / "1.pt"
    stale.write_bytes(b"pre-warmup-stale")
    with pytest.raises(SpringV31Error, match="directory/inventory .pt set differs"):
        _validate_cache_inventory(root, manifest, sequence_warmup=4)


def test_completion_claims_runtime_and_metrics_fail_closed() -> None:
    claims = {
        "paper_accuracy": False,
        "paper_gt": True,
        "epipolar_refinement": False,
        "temporal_future_frames": False,
        "formal_holdout": True,
        "coverage_eligible": True,
        "final_training_checkpoint": True,
        "final_acceptance_eligible": True,
        "acceptance_eligible": True,
        "full_validation_selection": True,
    }
    runtime = {
        "contract_version": "matched_candidate_forward_runtime_v1",
        "timing_backend": "torch.cuda.Event",
        "model_forward_calls": 1302,
        "model_forward_latency_ms_mean": 2.0,
        "model_forward_latency_ms_min": 1.0,
        "model_forward_latency_ms_max": 3.0,
        "cuda_allocated_at_start_bytes": 0,
        "cuda_reserved_at_start_bytes": 0,
        "cuda_peak_allocated_bytes": 1,
        "cuda_peak_reserved_bytes": 1,
    }
    metric = {"value": 0.5, "numerator": 5.0, "count": 10, "valid": True}
    report = {
        "status": "FINAL_CHECKPOINT_EVALUATION_COMPLETE",
        "claims": claims,
        "device": "cuda",
        "elapsed_seconds": 4.0,
        "runtime_v3": runtime,
        "methods": {"T3_VGGT": {"epe_px": metric, "bad_1": metric}},
    }

    _model_completion_evidence(report, "F6")
    assert _runtime_evidence(report, "F6")["model_forward_calls"] == 1302
    assert _primary_metric_evidence(report, "F6")["epe_px"]["count"] == 10

    report["claims"]["final_acceptance_eligible"] = False
    with pytest.raises(SpringV31Error, match="final-acceptance"):
        _model_completion_evidence(report, "F6")
    report["claims"]["final_acceptance_eligible"] = True
    report["methods"]["T3_VGGT"]["epe_px"]["numerator"] = float("nan")
    with pytest.raises(SpringV31Error, match="invalid numerator/count"):
        _primary_metric_evidence(report, "F6")


def _config_protocol_for_matrix(paths: Paths) -> dict[str, object]:
    return {
        arm: {
            "path": str(
                (paths.project_root / f"configs/spring_v3_1/{arm}.yaml").resolve()
            ),
            "sha256": sha256_file(
                paths.project_root / f"configs/spring_v3_1/{arm}.yaml"
            ),
        }
        for arm in ARM_ORDER
    }


def test_matrix_completion_requires_exact_full_arm_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, paths = _args(tmp_path)
    for arm in ARM_ORDER[:-1]:
        metrics = paths.arm_eval(arm) / "metrics.json"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text("{}\n", encoding="utf-8")
    _write_f7_blocked(paths)

    def verified(_paths: Paths, arm: str) -> dict[str, object]:
        return {
            "status": (
                "SCREENING_ONLY"
                if arm in {"F0", "F1"}
                else "FINAL_CHECKPOINT_EVALUATION_COMPLETE"
            ),
            "primary_metrics": {
                "method": arm,
                "epe_px": {"value": 1.0, "numerator": 2.0, "count": 2},
                "bad_1": {"value": 0.5, "numerator": 1.0, "count": 2},
            },
        }

    monkeypatch.setattr("tools.run_spring_v3_1_ffs._verify_eval_result", verified)
    monkeypatch.setattr(
        "tools.run_spring_v3_1_ffs.repository_git_hash", lambda _path: "a" * 40
    )
    source = {"git_clean": True, "git_dirty_paths": [], "git_head": "a" * 40}
    common = {
        "source_snapshot": source,
        "config_protocol": _config_protocol_for_matrix(paths),
        "backbone_protocol": {},
        "job_results": {},
        "dry_run": False,
    }
    partial = _matrix_receipt(paths, arms=("F0",), **common)
    assert partial["status"] == "PHASE_COMPLETE_EVIDENCE_PENDING"
    assert partial["formal_completion"]["exact_arm_selection"] is False

    complete = _matrix_receipt(paths, arms=ARM_ORDER, **common)
    assert complete["status"] == "COMPLETE_WITH_OPTIONAL_F7_BLOCKED"
    assert complete["arms"]["F0"]["status"] == "SCREENING_ONLY"
    assert complete["arms"]["F2"]["status"] == ("FINAL_CHECKPOINT_EVALUATION_COMPLETE")
    assert complete["arms"]["F7"]["status"] == "OPTIONAL_BLOCKED"
    assert complete["results"][6]["epe_px"]["numerator"] == 2.0
