from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import train
import train_epipolar
import eval_epipolar
from models.epipolar_refiner import HREpipolarRefiner
from models.epipolar_stage import FrozenTemporalEpipolarStage
from tools import preflight_stage_c_d025_positivity


def _stage_c_config() -> object:
    config = train_epipolar.resolve_epipolar_config(
        Path(__file__).parents[1]
        / "configs"
        / "ablations"
        / "d025_stage_c_positivity.yaml"
    )
    OmegaConf.update(
        config,
        "stage_c_positivity_ablation.d025_training_audit_path",
        "/tmp/d025-training-audit.json",
    )
    OmegaConf.update(
        config,
        "stage_c_positivity_ablation.d025_evaluation_audit_path",
        "/tmp/d025-evaluation-audit.json",
    )
    return config


def _base_metadata(checkpoint_sha256: str = "b" * 64) -> dict[str, object]:
    config = train.resolve_config("configs/ablations/d025_positivity_t3.yaml")
    plain = OmegaConf.to_container(config, resolve=True)
    assert isinstance(plain, dict)
    return {
        "path": "/tmp/d025-final.pt",
        "checkpoint_sha256": checkpoint_sha256,
        "step": 15_000,
        "parameter_count": 1_619_882,
        "git_hash": "a" * 40,
        "training_config": plain,
    }


def _write_prerequisites(
    root: Path, checkpoint_sha256: str = "b" * 64
) -> tuple[Path, Path, dict]:
    audit = {
        "schema_version": 1,
        "component": "training-run-audit",
        "status": "PASS",
        "training_status": "TRAINING_COMPLETE",
        "warnings": [],
        "files": {
            "final_checkpoint": {
                "sha256": checkpoint_sha256,
                "step": 15_000,
                "configured_steps": 15_000,
            }
        },
        "validation": {
            "loss_schema": {
                "positivity_ablation_enabled": True,
                "terms": ["disparity", "positivity_penalty", "total"],
            }
        },
    }
    base_config = _base_metadata(checkpoint_sha256)["training_config"]
    metrics = {
        "schema_version": 1,
        "status": "FINAL_CHECKPOINT_EVALUATION_COMPLETE",
        "checkpoint": {
            "checkpoint_sha256": checkpoint_sha256,
            "git_hash": "a" * 40,
            "step": 15_000,
            "training_config": base_config,
        },
    }
    audit_path = root / "audit.json"
    d025_eval = root / "d025_eval"
    canonical_eval = root / "canonical_eval"
    d025_eval.mkdir()
    canonical_eval.mkdir()
    metrics_path = d025_eval / "metrics.json"
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    (d025_eval / "metrics.csv").write_text("method\nT3_VGGT\n", encoding="utf-8")
    canonical_metrics = canonical_eval / "metrics.json"
    canonical_metrics.write_text("{}\n", encoding="utf-8")
    (canonical_eval / "metrics.csv").write_text(
        "method\nT3_VGGT\n", encoding="utf-8"
    )
    frozen_preflight = root / "d025-preflight.json"
    frozen_preflight.write_text("{}\n", encoding="utf-8")
    canonical_report = root / "canonical-report.json"
    canonical_report.write_text("{}\n", encoding="utf-8")

    def identity(path: Path) -> dict[str, str]:
        return {
            "path": str(path.resolve()),
            "sha256": train_epipolar.sha256_file(path),
        }

    formal = {
        "schema_version": 1,
        "component": train_epipolar.D025_EVALUATION_AUDIT_COMPONENT,
        "status": "D025_FINAL_CONTROLLED_COMPARISON_PASS",
        "read_only": True,
        "final_gate": {
            "eligible": True,
            "result": "PASS",
            "limited_or_intermediate_cannot_pass": True,
        },
        "claims": {
            "raw_owner": "T3_VGGT",
            "clamp0_owner": False,
            "pseudo_gt_engineering_only": True,
            "paper_ground_truth": False,
            "paper_accuracy": False,
        },
        "gates": {"all_required_gates_pass": True, "gates": {}},
        "training": {
            "formal": True,
            "checkpoint_sha256": checkpoint_sha256,
            "git_hash": "a" * 40,
        },
        "artifacts": {
            "d025_training_audit": identity(audit_path),
            "d025_preflight": identity(frozen_preflight),
            "d025_metrics": identity(metrics_path),
            "d025_metrics_csv": identity(d025_eval / "metrics.csv"),
            "canonical_stage_b_report": identity(canonical_report),
            "canonical_metrics": identity(canonical_metrics),
            "canonical_metrics_csv": identity(canonical_eval / "metrics.csv"),
        },
    }
    formal_path = root / "d025-evaluation-audit.json"
    formal_path.write_text(json.dumps(formal) + "\n", encoding="utf-8")
    return audit_path, formal_path, formal


def test_stage_c_d025_config_is_opt_in_zero_only_and_canonical_is_unchanged() -> None:
    canonical = train_epipolar.resolve_epipolar_config("configs/epipolar_x2.yaml")
    assert not train_epipolar.stage_c_positivity_ablation_from_config(
        canonical
    ).enabled
    assert "stage_c_positivity_ablation" not in canonical

    config = _stage_c_config()
    train_epipolar.validate_epipolar_config(config)
    ablation = train_epipolar.stage_c_positivity_ablation_from_config(config)
    assert ablation.enabled
    assert ablation.correction_lower_bound_hr_px == 0.0
    assert ablation.pre_lower_bound_negative_penalty_weight == 0.10

    epsilon = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.update(
        epsilon,
        "stage_c_positivity_ablation.correction_lower_bound_hr_px",
        1e-6,
    )
    with pytest.raises(ValueError, match="exactly 0.0"):
        train_epipolar.validate_epipolar_config(epsilon)

    raw_metrics = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.update(
        raw_metrics,
        "stage_c_positivity_ablation.d025_evaluation_metrics_path",
        "/tmp/untrusted-raw-metrics.json",
    )
    with pytest.raises(ValueError, match="formal evaluation audit"):
        train_epipolar.validate_epipolar_config(raw_metrics)


def test_opt_in_refiner_is_state_dict_compatible_but_behavior_bound() -> None:
    torch.manual_seed(42)
    canonical = HREpipolarRefiner()
    torch.manual_seed(42)
    positivity = HREpipolarRefiner(positivity_floor_hr_px=0.0)

    assert canonical.state_dict().keys() == positivity.state_dict().keys()
    for name, value in canonical.state_dict().items():
        torch.testing.assert_close(value, positivity.state_dict()[name], rtol=0, atol=0)
    assert canonical.positivity_floor_hr_px is None
    assert positivity.positivity_floor_hr_px == 0.0


def test_d025_prerequisite_pass_binds_exact_base_audit_metrics_and_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_sha256 = "c" * 64
    audit, formal_path, formal = _write_prerequisites(tmp_path, checkpoint_sha256)
    monkeypatch.setattr(
        train_epipolar, "audit_d025_evaluation", lambda *args: formal
    )
    config = _stage_c_config()

    receipt = train_epipolar.validate_d025_stage_b_prerequisites(
        base_metadata=_base_metadata(checkpoint_sha256),
        stage_c_config=config,
        training_audit_path=audit,
        evaluation_audit_path=formal_path,
    )

    assert receipt["status"] == "PASS"
    assert receipt["base_checkpoint"]["sha256"] == checkpoint_sha256
    assert receipt["canonical_stage_c_replacement"] is False
    assert receipt["formal_evaluation_audit"]["status"] == (
        "D025_FINAL_CONTROLLED_COMPARISON_PASS"
    )
    assert receipt["formal_evaluation_audit"]["final_gate"]["result"] == "PASS"


def test_d025_prerequisite_rejects_receipt_that_differs_from_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_sha256 = "c" * 64
    audit, formal_path, formal = _write_prerequisites(tmp_path, checkpoint_sha256)
    recomputed = json.loads(json.dumps(formal))
    recomputed["gates"]["gates"] = {"unexpected": {"pass": True}}
    monkeypatch.setattr(
        train_epipolar, "audit_d025_evaluation", lambda *args: recomputed
    )

    with pytest.raises(ValueError, match="differs from recomputation"):
        train_epipolar.validate_d025_stage_b_prerequisites(
            base_metadata=_base_metadata(checkpoint_sha256),
            stage_c_config=_stage_c_config(),
            training_audit_path=audit,
            evaluation_audit_path=formal_path,
        )


@pytest.mark.parametrize("mutation", ["none", "missing", "tampered"])
def test_controlled_evaluator_recomputes_formal_prerequisite_before_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    checkpoint_sha256 = "f" * 64
    training_audit, formal_path, formal = _write_prerequisites(
        tmp_path, checkpoint_sha256
    )
    monkeypatch.setattr(
        train_epipolar, "audit_d025_evaluation", lambda *args: formal
    )
    config = _stage_c_config()
    OmegaConf.update(
        config,
        "stage_c_positivity_ablation.d025_training_audit_path",
        str(training_audit.resolve()),
    )
    OmegaConf.update(
        config,
        "stage_c_positivity_ablation.d025_evaluation_audit_path",
        str(formal_path.resolve()),
    )
    base = _base_metadata(checkpoint_sha256)
    embedded = train_epipolar.validate_d025_stage_b_prerequisites(
        base_metadata=base,
        stage_c_config=config,
        training_audit_path=training_audit,
        evaluation_audit_path=formal_path,
    )
    plain = OmegaConf.to_container(config, resolve=True)
    assert isinstance(plain, dict)
    metadata = {
        "config": plain,
        "base_checkpoint": embedded["base_checkpoint"],
        "d025_prerequisite": embedded,
    }
    if mutation == "missing":
        formal_path.unlink()
    elif mutation == "tampered":
        formal_path.write_text(
            formal_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )

    if mutation == "none":
        recomputed = eval_epipolar.validate_controlled_d025_prerequisite(
            metadata,
            base,
        )
        assert recomputed == embedded
    else:
        with pytest.raises(
            eval_epipolar.CheckpointMismatchError,
            match="no longer verifies|differs from recomputation",
        ):
            eval_epipolar.validate_controlled_d025_prerequisite(metadata, base)


@pytest.mark.parametrize("failure", ["incomplete", "formal_gate", "lineage"])
def test_d025_prerequisite_fails_closed_before_stage_c(
    tmp_path: Path, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_sha256 = "d" * 64
    audit_path, formal_path, formal = _write_prerequisites(
        tmp_path, checkpoint_sha256
    )
    monkeypatch.setattr(
        train_epipolar, "audit_d025_evaluation", lambda *args: formal
    )
    base = _base_metadata(checkpoint_sha256)
    if failure == "incomplete":
        base["step"] = 500
    elif failure == "formal_gate":
        changed = json.loads(formal_path.read_text(encoding="utf-8"))
        changed["status"] = "D025_FINAL_CONTROLLED_COMPARISON_FAIL"
        formal_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["files"]["final_checkpoint"]["sha256"] = "e" * 64
        audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        train_epipolar.validate_d025_stage_b_prerequisites(
            base_metadata=base,
            stage_c_config=_stage_c_config(),
            training_audit_path=audit_path,
            evaluation_audit_path=formal_path,
        )


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def test_controlled_checkpoint_never_claims_canonical_stage_c_completion() -> None:
    config = _stage_c_config()
    base = _Base()
    refiner = HREpipolarRefiner(
        feature_channels=8,
        correlation_groups=2,
        head_channels=12,
        positivity_floor_hr_px=0.0,
    )
    stage = FrozenTemporalEpipolarStage(
        base,
        refiner,
        lambda module, batch: torch.ones(1, 1, 2, 4),
    )
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    prerequisite = {
        "status": "PASS",
        "canonical_stage_c_replacement": False,
        "formal_evaluation_audit": {
            "path": "/tmp/d025-evaluation-audit.json",
            "sha256": "d" * 64,
            "component": train_epipolar.D025_EVALUATION_AUDIT_COMPONENT,
            "status": "D025_FINAL_CONTROLLED_COMPARISON_PASS",
            "final_gate": {
                "eligible": True,
                "result": "PASS",
                "limited_or_intermediate_cannot_pass": True,
            },
        },
    }

    payload = train_epipolar._stage_c_checkpoint_payload(
        stage=stage,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_steps=5_000,
        config=config,
        git_hash="a" * 40,
        runtime_source_bundle={"git_head": "a" * 40, "bundle_sha256": "b" * 64},
        training_runtime={
            "formal_cuda_bf16_eligible": True,
            "strict_determinism_eligible": True,
        },
        base_checkpoint={"path": "/tmp/base.pt", "sha256": "c" * 64, "step": 15_000},
        base_lineage={"valid": True},
        raw_lineage={"valid": True},
        base_completion={"complete": True},
        rectification_audit={"status": "PASS"},
        latest_loss={"total": 1.0, "positivity_penalty": 0.1},
        elapsed_seconds=1.0,
        batches_per_epoch=10,
        d025_prerequisite=prerequisite,
    )

    assert payload["completion"]["formal_training_complete"] is False
    assert payload["completion"]["controlled_ablation_training_complete"] is True
    assert payload["completion"]["canonical_stage_c_replacement"] is False
    assert payload["experiment_role"] == "CONTROLLED_D025_STAGE_C_ABLATION"

    plain_config = OmegaConf.to_container(config, resolve=True)
    assert isinstance(plain_config, dict)
    completion = eval_epipolar.checkpoint_completion_status(
        {"config": plain_config, "step": 5_000},
        _base_metadata(),
    )
    assert completion["stage_c"]["controlled_ablation"] is True
    assert completion["stage_c"]["controlled_ablation_execution_complete"] is True
    assert completion["stage_c"]["complete"] is False
    assert completion["all_complete"] is False
    assert completion["controlled_ablation_all_complete"] is True


def test_controlled_checkpoint_loader_preserves_prerequisite_metadata(
    tmp_path: Path,
) -> None:
    config = _stage_c_config()
    rectification = {"path": "/tmp/rectification.json", "sha256": "a" * 64}
    OmegaConf.update(
        config,
        "data.epipolar_rectification_audit_path",
        rectification["path"],
        merge=False,
    )
    OmegaConf.update(
        config,
        "data.epipolar_rectification_audit",
        rectification,
        merge=False,
    )
    OmegaConf.update(
        config,
        "train.initialization_checkpoint",
        "/tmp/d025-final.pt",
        merge=False,
    )
    refiner = HREpipolarRefiner(positivity_floor_hr_px=0.0)
    stage = FrozenTemporalEpipolarStage(
        _Base(),
        refiner,
        lambda module, batch: torch.ones(1, 1, 1, 1),
    )
    optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=2e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update: train_epipolar.learning_rate_multiplier(
            update,
            total_steps=5_000,
            warmup_steps=500,
        ),
    )
    sum(parameter.sum() for parameter in refiner.parameters()).backward()
    optimizer.step()
    scheduler.step()
    base_checkpoint = {
        "path": "/tmp/d025-final.pt",
        "sha256": "b" * 64,
        "step": 15_000,
    }
    prerequisite = {
        "schema_version": 1,
        "component": "d025-stage-b-prerequisite-for-stage-c-positivity",
        "status": "PASS",
        "protocol_version": train_epipolar.STAGE_C_D025_POSITIVITY_PROTOCOL,
        "base_checkpoint": base_checkpoint,
        "training_audit": {
            "path": "/tmp/d025-training.json",
            "sha256": "c" * 64,
            "status": "PASS",
        },
        "formal_evaluation_audit": {
            "path": "/tmp/d025-evaluation.json",
            "sha256": "d" * 64,
            "component": train_epipolar.D025_EVALUATION_AUDIT_COMPONENT,
            "status": "D025_FINAL_CONTROLLED_COMPARISON_PASS",
            "final_gate": {
                "eligible": True,
                "result": "PASS",
                "limited_or_intermediate_cannot_pass": True,
            },
        },
        "evaluated_metrics": {
            "path": "/tmp/d025-metrics.json",
            "sha256": "e" * 64,
            "raw_owner": "T3_VGGT",
        },
        "canonical_stage_c_replacement": False,
    }
    runtime = {
        "device": "cuda:0",
        "device_type": "cuda",
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_capability": [12, 0],
        "torch_version": str(torch.__version__),
        "cuda_version": "12.8",
        "cuda_available": True,
        "bf16_supported": True,
        "autocast_enabled": True,
        "autocast_dtype": "torch.bfloat16",
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "strict_determinism_eligible": True,
        "formal_cuda_bf16_eligible": True,
    }
    payload = train_epipolar._stage_c_checkpoint_payload(
        stage=stage,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_steps=1,
        config=config,
        git_hash="f" * 40,
        runtime_source_bundle={"fixture": True},
        training_runtime=runtime,
        base_checkpoint=base_checkpoint,
        base_lineage={},
        raw_lineage={},
        base_completion={"complete": True},
        rectification_audit=rectification,
        latest_loss={"total": 1.0, "positivity_penalty": 0.0},
        elapsed_seconds=1.0,
        batches_per_epoch=4,
        d025_prerequisite=prerequisite,
    )
    checkpoint = tmp_path / "controlled-stage-c.pt"
    torch.save(payload, checkpoint)
    plain = OmegaConf.to_container(config, resolve=True)
    assert isinstance(plain, dict)

    _, metadata = eval_epipolar.load_stage_c_checkpoint(
        checkpoint,
        evaluation_config=plain,
    )

    assert metadata["experiment_role"] == "CONTROLLED_D025_STAGE_C_ABLATION"
    assert metadata["d025_prerequisite"] == prerequisite


def test_canonical_completion_status_preserves_legacy_schema() -> None:
    canonical = train_epipolar.resolve_epipolar_config("configs/epipolar_x2.yaml")
    plain = OmegaConf.to_container(canonical, resolve=True)
    assert isinstance(plain, dict)

    completion = eval_epipolar.checkpoint_completion_status(
        {"config": plain, "step": 5_000},
        _base_metadata(),
    )

    assert set(completion) == {"stage_c", "stage_b_base", "all_complete"}
    assert set(completion["stage_c"]) == {
        "actual_step",
        "configured_steps",
        "declared_epipolar_schedule_steps",
        "execution_complete",
        "canonical_schedule",
        "complete",
    }
    assert completion["all_complete"] is True


def test_cpu_preflight_emits_pass_only_for_bound_completed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = train.resolve_config("configs/ablations/d025_positivity_t3.yaml")
    plain_base = OmegaConf.to_container(base_config, resolve=True)
    assert isinstance(plain_base, dict)
    checkpoint = tmp_path / "d025-final.pt"
    torch.save(
        {
            "schema_version": 1,
            "step": 15_000,
            "config": plain_base,
            "git_hash": "a" * 40,
            "parameter_count": 1,
            "model": {"weight": torch.ones(1)},
        },
        checkpoint,
    )
    checkpoint_sha256 = train_epipolar.sha256_file(checkpoint)
    audit, formal_path, formal = _write_prerequisites(tmp_path, checkpoint_sha256)
    monkeypatch.setattr(
        train_epipolar, "audit_d025_evaluation", lambda *args: formal
    )

    receipt = preflight_stage_c_d025_positivity.run_preflight(
        SimpleNamespace(
            config=Path("configs/ablations/d025_stage_c_positivity.yaml"),
            d025_base_checkpoint=checkpoint,
            d025_training_audit=audit,
            d025_evaluation_audit=formal_path,
            receipt=None,
        )
    )

    assert receipt["status"] == "PREFLIGHT_PASS"
    assert receipt["read_only"] is True
    assert receipt["gpu_used"] is False
    assert receipt["canonical_stage_c_replacement"] is False
