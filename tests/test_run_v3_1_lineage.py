from __future__ import annotations

import json
from pathlib import Path

from tools.run_v3_experiments import (
    ARM_CONFIGS_BY_LINEAGE,
    ProcessResult,
    _invoke,
    build_parser,
    run_orchestration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arguments(tmp_path: Path, *, lineage: str | None = None):
    input_root = tmp_path / "inputs"
    input_root.mkdir(parents=True)
    files: dict[str, Path] = {}
    for name in ("train_manifest", "validation_manifest"):
        files[name] = input_root / f"{name}.jsonl"
        files[name].write_text("{}\n", encoding="utf-8")
    for name in ("train_sidecar", "validation_sidecar"):
        files[name] = input_root / f"{name}.jsonl"
        files[name].write_text("{}\n", encoding="utf-8")
        files[name].with_suffix(".receipt.json").write_text(
            json.dumps({"counts": {"unique_calibrations": 1}}),
            encoding="utf-8",
        )
    roots: dict[str, Path] = {}
    for name in (
        "train_observation",
        "train_teacher",
        "train_derived",
        "validation_observation",
        "validation_teacher",
        "validation_derived",
    ):
        roots[name] = input_root / name
        roots[name].mkdir()
    argv = [
        "--project-root",
        str(PROJECT_ROOT),
        "--output-root",
        str(tmp_path / "output"),
        "--train-manifest",
        str(files["train_manifest"]),
        "--train-observation-cache-root",
        str(roots["train_observation"]),
        "--train-teacher-cache-root",
        str(roots["train_teacher"]),
        "--train-derived-cache-root",
        str(roots["train_derived"]),
        "--train-calibration-sidecar",
        str(files["train_sidecar"]),
        "--validation-manifest",
        str(files["validation_manifest"]),
        "--validation-observation-cache-root",
        str(roots["validation_observation"]),
        "--validation-teacher-cache-root",
        str(roots["validation_teacher"]),
        "--validation-derived-cache-root",
        str(roots["validation_derived"]),
        "--validation-calibration-sidecar",
        str(files["validation_sidecar"]),
        "--dry-run",
    ]
    if lineage is not None:
        argv[0:0] = ["--lineage", lineage]
    return build_parser().parse_args(argv)


def test_lineage_default_is_immutable_v3(tmp_path: Path) -> None:
    args = _arguments(tmp_path)

    assert args.lineage == "v3"
    assert ARM_CONFIGS_BY_LINEAGE[args.lineage]["A0"].endswith(
        "v3_a0_control.yaml"
    )


def test_v3_1_dry_run_binds_configs_component_and_output(tmp_path: Path) -> None:
    args = _arguments(tmp_path, lineage="v3_1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must not invoke a subprocess")

    assert run_orchestration(args, executor=forbidden) is None
    output_root = tmp_path / "output"
    state = json.loads(
        (output_root / "orchestration_state.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (output_root / "run_receipt.json").read_text(encoding="utf-8")
    )

    assert state["lineage"] == "v3_1"
    assert state["component"] == "v3.1-experiment-orchestrator"
    assert state["output_identity"] == {
        "component": "v3.1-experiment-orchestrator",
        "lineage": "v3_1",
        "path": str(output_root.resolve()),
    }
    assert all(
        "configs/ablations/v3_1_" in relative
        for relative in state["arm_configs"].values()
    )
    assert all(entry["lineage"] == "v3_1" for entry in state["plan"])
    assert all("v3_1_" in Path(entry["config"]).name for entry in state["plan"])
    assert "configs/mvp_x2_v3_1.yaml" in state["source_snapshot"]["files"]
    assert receipt["component"] == "v3.1-experiment-orchestrator"
    assert receipt["lineage"] == "v3_1"
    assert receipt["output_identity"] == state["output_identity"]


def test_v3_1_process_receipt_has_distinct_component(tmp_path: Path) -> None:
    def executor(command, cwd, on_started, emit):
        del command, cwd
        on_started(12345)
        emit("complete\n")
        return ProcessResult(exit_code=0)

    receipt = _invoke(
        ["python", "train.py"],
        cwd=PROJECT_ROOT,
        job_directory=tmp_path / "job",
        active_process_path=tmp_path / "active.json",
        executor=executor,
        lineage="v3_1",
    )

    assert receipt["component"] == "v3.1-orchestrated-process"
    assert receipt["lineage"] == "v3_1"
    active = json.loads((tmp_path / "active.json").read_text(encoding="utf-8"))
    assert active["component"] == "v3.1-experiment-orchestrator"
    assert active["lineage"] == "v3_1"
