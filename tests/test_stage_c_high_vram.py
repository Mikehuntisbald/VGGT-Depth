from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import train_epipolar
import eval_epipolar


PROJECT_ROOT = Path(__file__).parents[1]
HIGH_VRAM_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "ablations"
    / "d025_stage_c_positivity_high_vram.yaml"
)
STANDARD_D025_CONFIG = (
    PROJECT_ROOT / "configs" / "ablations" / "d025_stage_c_positivity.yaml"
)


def _high_vram_config():
    return train_epipolar.resolve_epipolar_config(HIGH_VRAM_CONFIG)


def test_high_vram_config_is_exact_opt_in_4x2_profile() -> None:
    config = _high_vram_config()

    train_epipolar.validate_epipolar_config(config)
    profile = train_epipolar.stage_c_high_vram_from_config(config)

    assert profile.enabled is True
    assert profile.receipt_path is None
    assert profile.minimum_headroom_bytes == 2 * 1024**3
    assert int(config.train.micro_batch_size) == 4
    assert int(config.train.grad_accumulation) == 2
    assert int(config.train.effective_batch_size) == 8
    assert config.stage_c_high_vram.oom_fallback == {
        "micro_batch_size": 2,
        "grad_accumulation": 4,
        "effective_batch_size": 8,
    }
    assert config.ablation_protocol.canonical_stage_c_replacement is False


def test_standard_d025_and_canonical_stage_c_remain_disabled_and_2x4() -> None:
    for path in (STANDARD_D025_CONFIG, PROJECT_ROOT / "configs" / "epipolar_x2.yaml"):
        config = train_epipolar.resolve_epipolar_config(path)
        profile = train_epipolar.stage_c_high_vram_from_config(config)

        assert profile.enabled is False
        assert profile.receipt_path is None
        assert int(config.train.micro_batch_size) == 2
        assert int(config.train.grad_accumulation) == 4
        assert int(config.train.effective_batch_size) == 8
        assert config.get("stage_c_high_vram") is None


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (
            "stage_c_high_vram.protocol_version",
            "unknown",
            "protocol differs",
        ),
        (
            "stage_c_high_vram.minimum_headroom_bytes",
            0,
            "minimum_headroom_bytes must be positive",
        ),
        (
            "train.micro_batch_size",
            3,
            "schedule must be exactly 4x2=8",
        ),
        (
            "train.grad_accumulation",
            4,
            "schedule must be exactly 4x2=8",
        ),
        (
            "stage_c_high_vram.oom_fallback.micro_batch_size",
            1,
            "fallback must be the canonical 2x4 schedule",
        ),
        (
            "stage_c_positivity_ablation.enabled",
            False,
            "restricted to the D-025 opt-in arm",
        ),
    ],
)
def test_high_vram_parser_fails_closed_on_protocol_drift(
    path: str, value: object, match: str
) -> None:
    config = _high_vram_config()
    OmegaConf.update(config, path, value, merge=False)

    with pytest.raises(ValueError, match=match):
        train_epipolar.stage_c_high_vram_from_config(config)


def test_high_vram_runtime_bundle_is_a_separate_superset() -> None:
    canonical = train_epipolar.stage_c_runtime_relative_paths()
    controlled = train_epipolar.stage_c_runtime_relative_paths(
        controlled_ablation=True
    )
    high_vram = train_epipolar.stage_c_runtime_relative_paths(
        controlled_ablation=True,
        high_vram=True,
    )
    relative = "configs/ablations/d025_stage_c_positivity_high_vram.yaml"

    assert relative not in canonical
    assert relative not in controlled
    assert relative in high_vram
    assert set(canonical) < set(controlled) < set(high_vram)
    assert len(high_vram) == len(controlled) + 1

    with pytest.raises(ValueError, match="requires the controlled D-025 arm"):
        train_epipolar.stage_c_runtime_relative_paths(high_vram=True)
    with pytest.raises(ValueError, match="requires the controlled D-025 arm"):
        train_epipolar.stage_c_runtime_git_scopes(high_vram=True)


def test_high_vram_cli_is_explicit_and_does_not_change_default_config() -> None:
    args = train_epipolar.build_parser().parse_args(
        ["--config", str(STANDARD_D025_CONFIG)]
    )

    assert args.high_vram_preflight_receipt is None
    config = train_epipolar.resolve_epipolar_config(args.config)
    assert train_epipolar.stage_c_high_vram_from_config(config).enabled is False


def _receipt_fixture(tmp_path: Path) -> dict[str, object]:
    receipt_path = tmp_path / "high-vram-preflight.json"
    config = _high_vram_config()
    OmegaConf.update(
        config,
        "stage_c_high_vram.preflight_receipt_path",
        str(receipt_path.resolve()),
        merge=False,
    )
    base = {
        "path": str((tmp_path / "d025-final.pt").resolve()),
        "sha256": "b" * 64,
        "step": 15_000,
    }
    prerequisite = {
        "schema_version": 1,
        "component": "d025-stage-b-prerequisite-for-stage-c-positivity",
        "status": "PASS",
        "base_checkpoint": base,
        "formal_evaluation_audit": {
            "status": "D025_FINAL_CONTROLLED_COMPARISON_PASS"
        },
        "canonical_stage_c_replacement": False,
    }
    runtime = {
        "schema_version": 1,
        "git_head": "a" * 40,
        "relevant_paths_clean": True,
        "files": [],
        "bundle_sha256": "c" * 64,
    }
    training_runtime = {
        "device_type": "cuda",
        "precision": "bf16",
        "formal_cuda_bf16_eligible": True,
        "strict_determinism_eligible": True,
    }
    kwargs = {
        "config": config,
        "base_checkpoint": base,
        "d025_prerequisite": prerequisite,
        "runtime_source_bundle": runtime,
        "training_runtime": training_runtime,
        "peak_cuda_allocated_bytes": 18 * 1024**3,
        "peak_cuda_reserved_bytes": 20 * 1024**3,
        "cuda_free_before_bytes": 28 * 1024**3,
        "cuda_free_after_bytes": 11 * 1024**3,
        "cuda_total_bytes": 32 * 1024**3,
        "completed_micro_steps": 2,
        "gradient_norm": 0.5,
        "parameters_finite_after_step": True,
        "loss": {
            "total": 1.0,
            "disparity": 0.8,
            "correction_regularizer": 0.1,
            "positivity_penalty": 0.1,
            "valid_pixel_count": 4096,
        },
    }
    return {"path": receipt_path, "kwargs": kwargs}


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def test_high_vram_pass_receipt_binds_cuda_step_memory_and_lineage(
    tmp_path: Path,
) -> None:
    fixture = _receipt_fixture(tmp_path)
    kwargs = fixture["kwargs"]
    assert isinstance(kwargs, dict)
    receipt = train_epipolar.build_stage_c_high_vram_preflight_receipt(**kwargs)

    assert receipt["status"] == "PREFLIGHT_PASS"
    assert receipt["optimizer_step_executed"] is True
    assert receipt["completed_micro_steps"] == 2
    assert receipt["parameters_finite_after_step"] is True
    assert receipt["memory"]["headroom_bytes"] == 12 * 1024**3
    assert receipt["memory"]["minimum_headroom_bytes"] == 2 * 1024**3
    assert receipt["oom_fallback"] == {
        "micro_batch_size": 2,
        "grad_accumulation": 4,
        "effective_batch_size": 8,
    }

    receipt_path = fixture["path"]
    assert isinstance(receipt_path, Path)
    _write_receipt(receipt_path, receipt)
    validated = train_epipolar.validate_stage_c_high_vram_preflight_receipt(
        path=receipt_path,
        config=kwargs["config"],
        base_checkpoint=kwargs["base_checkpoint"],
        d025_prerequisite=kwargs["d025_prerequisite"],
        runtime_source_bundle=kwargs["runtime_source_bundle"],
        training_runtime=kwargs["training_runtime"],
    )

    assert validated["status"] == "PREFLIGHT_PASS"
    assert validated["formal_training_authorized"] is True
    assert validated["memory"]["headroom_bytes"] == 12 * 1024**3


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("completed_micro_steps", 1, "exactly two micro-batches"),
        ("gradient_norm", float("nan"), "diagnostics are invalid"),
        ("parameters_finite_after_step", False, "diagnostics are invalid"),
    ],
)
def test_high_vram_receipt_builder_rejects_incomplete_or_nonfinite_step(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    kwargs = _receipt_fixture(tmp_path)["kwargs"]
    assert isinstance(kwargs, dict)
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        train_epipolar.build_stage_c_high_vram_preflight_receipt(**kwargs)


def test_high_vram_insufficient_headroom_is_not_formal_pass(tmp_path: Path) -> None:
    fixture = _receipt_fixture(tmp_path)
    kwargs = fixture["kwargs"]
    assert isinstance(kwargs, dict)
    kwargs["peak_cuda_reserved_bytes"] = 31 * 1024**3
    receipt = train_epipolar.build_stage_c_high_vram_preflight_receipt(**kwargs)

    assert receipt["status"] == "PREFLIGHT_FAIL_INSUFFICIENT_HEADROOM"
    assert receipt["formal_training_authorized"] is False
    assert receipt["oom_fallback"] == train_epipolar.STAGE_C_HIGH_VRAM_FALLBACK
    receipt_path = fixture["path"]
    assert isinstance(receipt_path, Path)
    _write_receipt(receipt_path, receipt)

    with pytest.raises(ValueError, match="has not passed"):
        train_epipolar.validate_stage_c_high_vram_preflight_receipt(
            path=receipt_path,
            config=kwargs["config"],
            base_checkpoint=kwargs["base_checkpoint"],
            d025_prerequisite=kwargs["d025_prerequisite"],
            runtime_source_bundle=kwargs["runtime_source_bundle"],
            training_runtime=kwargs["training_runtime"],
        )


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("base", "lineage/runtime differs"),
        ("runtime", "lineage/runtime differs"),
        ("prerequisite", "lineage/runtime differs"),
        ("headroom", "headroom did not pass"),
        ("loss", "non-finite constant"),
    ],
)
def test_high_vram_validator_rejects_tampered_receipt(
    tmp_path: Path, tamper: str, match: str
) -> None:
    fixture = _receipt_fixture(tmp_path)
    kwargs = fixture["kwargs"]
    assert isinstance(kwargs, dict)
    receipt = train_epipolar.build_stage_c_high_vram_preflight_receipt(**kwargs)
    receipt = copy.deepcopy(receipt)
    if tamper == "base":
        receipt["base_checkpoint"]["sha256"] = "0" * 64
    elif tamper == "runtime":
        receipt["runtime_source_bundle"]["bundle_sha256"] = "0" * 64
    elif tamper == "prerequisite":
        receipt["d025_prerequisite_sha256"] = "0" * 64
    elif tamper == "headroom":
        receipt["memory"]["headroom_bytes"] += 1
    elif tamper == "loss":
        receipt["loss"]["total"] = float("nan")
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(tamper)
    receipt_path = fixture["path"]
    assert isinstance(receipt_path, Path)
    _write_receipt(receipt_path, receipt)

    with pytest.raises(ValueError, match=match):
        train_epipolar.validate_stage_c_high_vram_preflight_receipt(
            path=receipt_path,
            config=kwargs["config"],
            base_checkpoint=kwargs["base_checkpoint"],
            d025_prerequisite=kwargs["d025_prerequisite"],
            runtime_source_bundle=kwargs["runtime_source_bundle"],
            training_runtime=kwargs["training_runtime"],
        )


def test_high_vram_evaluator_revalidates_live_receipt_before_forward(
    tmp_path: Path,
) -> None:
    fixture = _receipt_fixture(tmp_path)
    kwargs = fixture["kwargs"]
    assert isinstance(kwargs, dict)
    receipt = train_epipolar.build_stage_c_high_vram_preflight_receipt(**kwargs)
    receipt_path = fixture["path"]
    assert isinstance(receipt_path, Path)
    _write_receipt(receipt_path, receipt)
    compact = train_epipolar.validate_stage_c_high_vram_preflight_receipt(
        path=receipt_path,
        config=kwargs["config"],
        base_checkpoint=kwargs["base_checkpoint"],
        d025_prerequisite=kwargs["d025_prerequisite"],
        runtime_source_bundle=kwargs["runtime_source_bundle"],
        training_runtime=kwargs["training_runtime"],
    )
    config = OmegaConf.to_container(kwargs["config"], resolve=True)
    assert isinstance(config, dict)
    base = kwargs["base_checkpoint"]
    assert isinstance(base, dict)
    metadata = {
        "config": config,
        "base_checkpoint": base,
        "d025_prerequisite": kwargs["d025_prerequisite"],
        "runtime_source_bundle": kwargs["runtime_source_bundle"],
        "training_runtime": kwargs["training_runtime"],
        "high_vram_preflight": compact,
    }
    base_metadata = {
        "path": base["path"],
        "checkpoint_sha256": base["sha256"],
        "step": base["step"],
    }

    assert eval_epipolar.validate_controlled_high_vram_preflight(
        metadata,
        base_metadata,
        kwargs["d025_prerequisite"],
    ) == compact

    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(
        eval_epipolar.CheckpointMismatchError,
        match="differs from recomputation",
    ):
        eval_epipolar.validate_controlled_high_vram_preflight(
            metadata,
            base_metadata,
            kwargs["d025_prerequisite"],
        )
