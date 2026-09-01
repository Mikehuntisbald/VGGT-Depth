from __future__ import annotations

import json
import hashlib
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tools.run_v3_experiments import (
    ARM_ORDER,
    CUDA_OOM_PATTERN,
    OrchestrationError,
    ProcessResult,
    _default_executor,
    build_parser,
    run_orchestration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _executor_worker(queue: Any) -> None:
    _default_executor(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        PROJECT_ROOT,
        lambda pid: queue.put(pid),
        lambda _text: None,
    )


def _touch(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _validation_manifest_text() -> str:
    return "".join(
        json.dumps(
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
        )
        + "\n"
        for index in range(244)
    )


def _arguments(tmp_path: Path, *extra: str) -> Any:
    train_manifest = _touch(tmp_path / "inputs" / "train.jsonl")
    validation_manifest = _touch(
        tmp_path / "inputs" / "validation.jsonl", _validation_manifest_text()
    )
    train_sidecar = _touch(tmp_path / "inputs" / "train_calibration.jsonl")
    train_sidecar.with_suffix(".receipt.json").write_text(
        json.dumps({"counts": {"unique_calibrations": 1}}), encoding="utf-8"
    )
    validation_sidecar = _touch(tmp_path / "inputs" / "validation_calibration.jsonl")
    receipt = validation_sidecar.with_suffix(".receipt.json")
    receipt.write_text(
        json.dumps({"counts": {"unique_calibrations": 1}}), encoding="utf-8"
    )
    roots = {}
    for name in (
        "train_observation",
        "train_teacher",
        "train_derived",
        "validation_observation",
        "validation_teacher",
        "validation_derived",
    ):
        roots[name] = tmp_path / "inputs" / name
        roots[name].mkdir(parents=True, exist_ok=True)
    argv = [
        "--project-root",
        str(PROJECT_ROOT),
        "--output-root",
        str(tmp_path / "run"),
        "--train-manifest",
        str(train_manifest),
        "--train-observation-cache-root",
        str(roots["train_observation"]),
        "--train-teacher-cache-root",
        str(roots["train_teacher"]),
        "--train-derived-cache-root",
        str(roots["train_derived"]),
        "--train-calibration-sidecar",
        str(train_sidecar),
        "--validation-manifest",
        str(validation_manifest),
        "--validation-observation-cache-root",
        str(roots["validation_observation"]),
        "--validation-teacher-cache-root",
        str(roots["validation_teacher"]),
        "--validation-derived-cache-root",
        str(roots["validation_derived"]),
        "--validation-calibration-sidecar",
        str(validation_sidecar),
        "--bootstrap-replicates",
        "200",
        *extra,
    ]
    return build_parser().parse_args(argv)


def _arm(command: list[str]) -> str:
    config = Path(command[command.index("--config") + 1]).name
    for arm, fragment in {
        "A0": "v3_a0_",
        "A1": "v3_a1_",
        "A2": "v3_a2_",
        "A3": "v3_a3_",
        "B0": "v3_b0_",
        "B1": "v3_b1_",
    }.items():
        if fragment in config:
            return arm
    raise AssertionError(config)


def _eval_report(arm: str, *, seed: int, manifest: Path) -> dict[str, Any]:
    stage_b = arm.startswith("B")
    method = "T3_VGGT" if stage_b else "T1"
    low = {"A0": 1.0, "A1": 0.98, "A2": 0.99, "A3": 0.96}.get(arm, 0.9)
    values = {
        "low_confidence_epe_px": low,
        "epe_px": 1.0,
        "boundary_epe_px": 1.0,
        "trusted_region_epe_px": 1.0,
        "invalid_region_completeness": 0.5,
        "output_negative_rate": 0.001,
        "output_invalid_rate": 0.001,
        "output_nan_rate": 0.0,
        "output_infinite_rate": 0.0,
    }
    if stage_b:
        values["temporal_residual_error_native_px"] = 0.5 if arm == "B0" else 0.45
    metric = lambda value: {  # noqa: E731 - compact schema fixture.
        "valid": True,
        "count": 100,
        "numerator": value * 100,
        "value": value,
    }
    method_metrics = {name: metric(value) for name, value in values.items()}
    if stage_b:
        method_metrics["temporal_residual_error_paired_px"] = {
            "valid": False,
            "count": 0,
            "numerator": 0.0,
            "value": None,
        }
    switches = {
        "A0": (False, False, False),
        "A1": (True, False, False),
        "A2": (False, True, False),
        "A3": (True, True, False),
        "B0": (True, True, False),
        "B1": (True, True, True),
    }[arm]
    report: dict[str, Any] = {
        "stage": "T3_CAUSAL_STAGE_B" if stage_b else "T1_SPATIAL_ONLY",
        "records_evaluated": 238 if stage_b else 244,
        "crop_mode": "full",
        "manifest_path": str(manifest.resolve()),
        "claims": {
            "final_acceptance_eligible": True,
            "full_validation_selection": True,
        },
        "runtime_v3": {
            "contract_version": "matched_candidate_forward_runtime_v1",
            "timing_backend": "torch.cuda.Event",
            "model_forward_calls": 244 if not stage_b else 714,
            "model_forward_latency_ms_mean": 10.0,
            "cuda_peak_allocated_bytes": 1_000.0,
            "cuda_peak_reserved_bytes": 2_000.0,
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
                "use_rays": switches[0],
                "use_stereo_pose": switches[1],
                "use_temporal_pose": switches[2],
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
        report["comparisons"]["T3_vs_T1_temporal"] = {
            "valid": True,
            "relative_change_percent": -12.0,
        }
    return report


class FakeExecutor:
    def __init__(self, *, oom_a0_high: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.oom_a0_high = oom_a0_high
        self.used_oom = False

    def __call__(self, command, cwd, on_started, emit) -> ProcessResult:
        command = list(command)
        self.commands.append(command)
        on_started(10_000 + len(self.commands))
        arm = _arm(command)
        if Path(command[1]).name == "train.py":
            output = Path(command[command.index("--output-dir") + 1])
            high = "train.micro_batch_size=4" in command
            if self.oom_a0_high and arm == "A0" and high and not self.used_oom:
                self.used_oom = True
                emit("torch.OutOfMemoryError: CUDA out of memory\n")
                return ProcessResult(exit_code=1)
            _touch(output / "final.pt", "checkpoint")
            _touch(output / "run_summary.json", "{}\n")
            emit("TRAINING_COMPLETE\n")
            return ProcessResult(exit_code=0)
        output = Path(command[command.index("--output-dir") + 1])
        seed = int(next(value.split("=", 1)[1] for value in command if value.startswith("seed=")))
        manifest = Path(command[command.index("--manifest") + 1])
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        report = _eval_report(arm, seed=seed, manifest=manifest)
        report["checkpoint"] = {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        }
        if "--spatial-checkpoint" in command:
            spatial_checkpoint = Path(
                command[command.index("--spatial-checkpoint") + 1]
            )
            report["spatial_checkpoint"] = {
                "checkpoint_sha256": hashlib.sha256(
                    spatial_checkpoint.read_bytes()
                ).hexdigest()
            }
        else:
            report["spatial_checkpoint"] = None
        _touch(
            output / "metrics.json",
            json.dumps(report),
        )
        emit("EVALUATION_COMPLETE\n")
        return ProcessResult(exit_code=0)


def test_dry_run_writes_ordered_plan_without_launching(tmp_path: Path) -> None:
    args = _arguments(tmp_path, "--dry-run")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must not launch a subprocess")

    assert run_orchestration(args, executor=forbidden) is None
    state = json.loads((tmp_path / "run" / "orchestration_state.json").read_text())
    assert state["status"] == "DRY_RUN"
    assert [entry["arm"] for entry in state["plan"][:6]] == list(ARM_ORDER)
    assert [entry["seed"] for entry in state["plan"][:6]] == [42] * 6
    assert "tools/audit_v3_temporal_pose.py" in state["source_snapshot"]["files"]
    assert not list((tmp_path / "run").rglob("process_attempt_*.log"))


def test_dry_run_binds_temporal_pose_variation_audit(tmp_path: Path) -> None:
    args = _arguments(tmp_path, "--dry-run")
    derived_root = Path(args.validation_derived_cache_root)
    manifest = _touch(derived_root / "cache_manifest.jsonl", "{}\n")
    receipt = _touch(
        derived_root / "run_receipt.json",
        json.dumps(
            {"output": {"cache_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
        ),
    )
    formal_ids = [f"validation-sequence/{index}" for index in range(6, 244)]
    audit_path = tmp_path / "inputs" / "temporal_pose_audit.json"
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
                    "formal_pose_valid_windows": 30,
                },
                "ages": {"1": {"varies": True}, "2": {"varies": True}},
                "formal_endpoint_binding": {
                    "available": True,
                    "record_ids": formal_ids,
                    "record_ids_sha256": hashlib.sha256(
                        json.dumps(
                            formal_ids, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                    "pose_valid_record_ids": formal_ids[:30],
                },
                "inputs": {
                    "derived_root": str(derived_root.resolve()),
                    "run_receipt_sha256": hashlib.sha256(
                        receipt.read_bytes()
                    ).hexdigest(),
                    "cache_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "validation_manifest": {
                        "path": str(Path(args.validation_manifest).resolve()),
                        "sha256": hashlib.sha256(
                            Path(args.validation_manifest).read_bytes()
                        ).hexdigest(),
                        "records": 244,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    args.temporal_pose_audit = audit_path

    assert run_orchestration(args) is None

    state = json.loads((tmp_path / "run" / "orchestration_state.json").read_text())
    identity = state["input_identities"]["temporal_pose_variation_audit"]
    assert identity["path"] == str(audit_path.resolve())
    assert state["execution_contract"]["temporal_pose_varies"] is True
    assert state["execution_contract"]["temporal_pose_identifiability_source"] == (
        "bound_audit_receipt"
    )


def test_temporal_pose_audit_rejects_derived_hash_drift(tmp_path: Path) -> None:
    args = _arguments(tmp_path, "--dry-run")
    derived_root = Path(args.validation_derived_cache_root)
    _touch(derived_root / "run_receipt.json", "changed\n")
    _touch(derived_root / "cache_manifest.jsonl", "{}\n")
    formal_ids = [f"validation-sequence/{index}" for index in range(6, 244)]
    audit_path = tmp_path / "inputs" / "temporal_pose_audit.json"
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
                    "formal_pose_valid_windows": 30,
                },
                "ages": {"1": {"varies": True}, "2": {"varies": True}},
                "formal_endpoint_binding": {
                    "available": True,
                    "record_ids": formal_ids,
                    "record_ids_sha256": hashlib.sha256(
                        json.dumps(
                            formal_ids, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                    "pose_valid_record_ids": formal_ids[:30],
                },
                "inputs": {
                    "derived_root": str(derived_root.resolve()),
                    "run_receipt_sha256": "0" * 64,
                    "cache_manifest_sha256": hashlib.sha256(
                        (derived_root / "cache_manifest.jsonl").read_bytes()
                    ).hexdigest(),
                    "validation_manifest": {
                        "path": str(Path(args.validation_manifest).resolve()),
                        "sha256": hashlib.sha256(
                            Path(args.validation_manifest).read_bytes()
                        ).hexdigest(),
                        "records": 244,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    args.temporal_pose_audit = audit_path

    with pytest.raises(OrchestrationError, match="receipt hash mismatch"):
        run_orchestration(args)


def test_cuda_oom_fallback_and_same_a3_stage_b_lineage(tmp_path: Path) -> None:
    args = _arguments(
        tmp_path,
        "--additional-seeds",
        "never",
    )
    executor = FakeExecutor(oom_a0_high=True)

    decision = run_orchestration(args, executor=executor)

    assert decision is not None and decision["decision"] == "NO-GO"
    high = tmp_path / "run" / "seed_42" / "A0" / "train_4x2"
    fallback = tmp_path / "run" / "seed_42" / "A0" / "train_2x4"
    high_receipt = json.loads(
        (high / "process_attempt_001.receipt.json").read_text()
    )
    assert high_receipt["cuda_oom_detected"] is True
    assert (fallback / "final.pt").is_file()
    assert not list((tmp_path / "run").rglob("*.log.partial"))
    train_commands = [cmd for cmd in executor.commands if Path(cmd[1]).name == "train.py"]
    assert [_arm(cmd) for cmd in train_commands] == [
        "A0",
        "A0",
        "A1",
        "A2",
        "A3",
        "B0",
        "B1",
    ]
    a3_final = (
        tmp_path / "run" / "seed_42" / "A3" / "train_4x2" / "final.pt"
    ).resolve()
    for arm in ("B0", "B1"):
        command = next(cmd for cmd in train_commands if _arm(cmd) == arm)
        assert Path(command[command.index("--init-from") + 1]) == a3_final
    assert json.loads((tmp_path / "run" / "run_receipt.json").read_text())[
        "status"
    ] == "COMPLETE"
    decision_inputs = json.loads(
        (tmp_path / "run" / "decision_inputs.json").read_text(
            encoding="utf-8"
        )
    )
    b1_entry = decision_inputs["seeds"]["42"]["B1"]
    b1_checkpoint = (
        tmp_path / "run" / "seed_42" / "B1" / "train_4x2" / "final.pt"
    )
    assert b1_entry["checkpoint_sha256"] == hashlib.sha256(
        b1_checkpoint.read_bytes()
    ).hexdigest()
    assert b1_entry["spatial_checkpoint_sha256"] == hashlib.sha256(
        a3_final.read_bytes()
    ).hexdigest()


def test_resume_recovers_completed_train_and_eval_without_relaunch(tmp_path: Path) -> None:
    first_args = _arguments(tmp_path, "--additional-seeds", "never")
    first = FakeExecutor()
    run_orchestration(first_args, executor=first)

    resumed_args = _arguments(
        tmp_path, "--additional-seeds", "never", "--resume"
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("completed artifacts must be recovered")

    decision = run_orchestration(resumed_args, executor=forbidden)

    assert decision is not None and decision["decision"] == "NO-GO"
    state = json.loads((tmp_path / "run" / "orchestration_state.json").read_text())
    assert state["jobs"]["seed_42.A3.train"]["status"] == "RECOVERED_COMPLETE"
    assert state["jobs"]["seed_42.B1.eval"]["status"] == "RECOVERED_COMPLETE"


def test_resume_rejects_metrics_relabelled_to_another_checkpoint(
    tmp_path: Path,
) -> None:
    first_args = _arguments(tmp_path, "--additional-seeds", "never")
    run_orchestration(first_args, executor=FakeExecutor())

    metrics_path = tmp_path / "run" / "seed_42" / "B1" / "eval" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["checkpoint"]["checkpoint_sha256"] = "0" * 64
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    state_path = tmp_path / "run" / "orchestration_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["jobs"]["seed_42.B1.eval"]["metrics_sha256"] = hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    resumed_args = _arguments(
        tmp_path, "--additional-seeds", "never", "--resume"
    )
    with pytest.raises(OrchestrationError, match="checkpoint SHA"):
        run_orchestration(resumed_args, executor=FakeExecutor())


def test_formal_runner_rejects_treatment_override_and_resume_dry_run(
    tmp_path: Path,
) -> None:
    treatment = _arguments(
        tmp_path / "treatment",
        "--train-override",
        "calibration_conditioning_v3.use_rays=false",
    )
    with pytest.raises(OrchestrationError, match="benign allowlist"):
        run_orchestration(treatment)

    conflicting = _arguments(tmp_path / "conflicting", "--resume", "--dry-run")
    with pytest.raises(OrchestrationError, match="mutually exclusive"):
        run_orchestration(conflicting)


def test_bare_non_cuda_out_of_memory_text_is_not_a_fallback_trigger() -> None:
    assert CUDA_OOM_PATTERN.search("torch.OutOfMemoryError: host allocation failed") is None
    assert CUDA_OOM_PATTERN.search("RuntimeError: CUDA out of memory") is not None


def test_bare_temporal_pose_assertion_is_rejected(tmp_path: Path) -> None:
    args = _arguments(tmp_path, "--dry-run", "--temporal-pose-varies")

    with pytest.raises(OrchestrationError, match="not formal evidence"):
        run_orchestration(args)


def test_resume_refuses_live_recorded_orphan_child(tmp_path: Path) -> None:
    first_args = _arguments(tmp_path, "--additional-seeds", "never")
    run_orchestration(first_args, executor=FakeExecutor())
    cmdline = [
        value.decode("utf-8", errors="surrogateescape")
        for value in Path(f"/proc/{os.getpid()}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    active = tmp_path / "run" / "active_process.json"
    active.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "child_pid": os.getpid(),
                "command": cmdline,
            }
        ),
        encoding="utf-8",
    )
    resumed = _arguments(tmp_path, "--additional-seeds", "never", "--resume")

    with pytest.raises(OrchestrationError, match="still live"):
        run_orchestration(resumed, executor=FakeExecutor())


def test_cache_manifest_drift_is_detected_between_subprocesses(tmp_path: Path) -> None:
    args = _arguments(tmp_path, "--additional-seeds", "never")
    derived_manifest = Path(args.validation_derived_cache_root) / "cache_manifest.jsonl"
    derived_manifest.write_text("initial\n", encoding="utf-8")
    executor = FakeExecutor()

    class MutatingExecutor:
        def __call__(self, command, cwd, on_started, emit):
            result = executor(command, cwd, on_started, emit)
            if Path(command[1]).name == "train.py" and _arm(list(command)) == "A0":
                derived_manifest.write_text("changed\n", encoding="utf-8")
            return result

    with pytest.raises(OrchestrationError, match="identity changed"):
        run_orchestration(args, executor=MutatingExecutor())


def test_sigterm_is_forwarded_to_child_process_group() -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(target=_executor_worker, args=(queue,))
    worker.start()
    child_pid = queue.get(timeout=10)
    os.kill(worker.pid, signal.SIGTERM)
    worker.join(timeout=15)

    assert not worker.is_alive()
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"forwarded child process {child_pid} is still live")
