from __future__ import annotations

from pathlib import Path
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import eval as eval_cli
import train
from utils.checkpoint import CHECKPOINT_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "spring_v3_1"
LEARNED_ARMS = ("F2", "F3_stage_a_control", "F3", "F4", "F5", "F6")


def _resolve_train(name: str) -> DictConfig:
    config = train.resolve_config(CONFIG_ROOT / f"{name}.yaml")
    if bool(config.calibration_conditioning_v3.enabled):
        OmegaConf.update(
            config,
            "data.calibration_sidecar_path",
            "/tmp/spring-v31-config-contract-calibration.jsonl",
            merge=False,
        )
    train.validate_training_config(config)
    return config


def _resolve_eval(name: str) -> DictConfig:
    config = eval_cli.resolve_evaluation_config(CONFIG_ROOT / f"{name}.yaml")
    if bool(config.calibration_conditioning_v3.enabled):
        OmegaConf.update(
            config,
            "data.calibration_sidecar_path",
            "/tmp/spring-v31-config-contract-calibration.jsonl",
            merge=False,
        )
    eval_cli.validate_evaluation_config(config)
    return config


def _model_signature(config: DictConfig) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(value.shape)
        for name, value in train.build_model(config).state_dict().items()
    }


@pytest.mark.parametrize(
    (
        "name",
        "stage",
        "sequence_length",
        "derived_contract",
        "pose_source",
        "use_vggt_pose",
        "use_vggt_depth",
        "top_k",
        "use_temporal_pose",
    ),
    (
        ("F2", "spatial", 1, "calibrated_stereo_v2", "gt", False, False, 4, False),
        (
            "F3_stage_a_control",
            "spatial",
            1,
            "legacy_v1",
            "gt",
            False,
            False,
            2,
            False,
        ),
        ("F3", "temporal", 3, "legacy_v1", "gt", False, False, 2, False),
        (
            "F4",
            "temporal",
            3,
            "calibrated_stereo_v2",
            "gt",
            False,
            False,
            4,
            True,
        ),
        (
            "F5",
            "temporal",
            3,
            "calibrated_stereo_v2",
            "gt",
            False,
            True,
            4,
            True,
        ),
        (
            "F6",
            "temporal",
            3,
            "calibrated_stereo_v2",
            "vggt",
            True,
            True,
            4,
            True,
        ),
    ),
)
def test_learned_arm_resolved_contracts(
    name: str,
    stage: str,
    sequence_length: int,
    derived_contract: str,
    pose_source: str,
    use_vggt_pose: bool,
    use_vggt_depth: bool,
    top_k: int,
    use_temporal_pose: bool,
) -> None:
    training = _resolve_train(name)
    evaluation = _resolve_eval(name)

    assert train.validate_training_config(training) == stage
    assert eval_cli.validate_evaluation_config(evaluation) == stage
    assert int(training.seed) == 42
    assert int(training.data.sequence_length) == sequence_length
    assert str(training.data.derived_contract) == derived_contract
    assert bool(training.data.require_cache_inventory_lineage) is True
    assert train.temporal_pose_source_from_config(training) == pose_source
    assert bool(training.model.use_vggt_pose) is use_vggt_pose
    assert bool(training.model.use_vggt_depth) is use_vggt_depth
    assert int(training.temporal_history_v2.top_k) == top_k
    assert (
        bool(training.calibration_conditioning_v3.use_temporal_pose)
        is use_temporal_pose
    )
    assert int(training.train.steps_spatial) == 5_000
    assert int(training.train.steps) == 15_000

    supervision = train.supervision_target_from_config(training)
    assert supervision.cache_component == "spring-ground-truth"
    assert supervision.target_type == "spring_v2_disp1_ground_truth"
    assert supervision.paper_ground_truth

    assert OmegaConf.to_container(
        training.supervision, resolve=True
    ) == OmegaConf.to_container(evaluation.supervision, resolve=True)
    assert str(evaluation.data.derived_contract) == derived_contract
    assert train.temporal_pose_source_from_config(evaluation) == pose_source


def test_stage_a_initializers_are_strictly_topology_compatible() -> None:
    f2 = _resolve_train("F2")
    f2_signature = _model_signature(f2)
    for arm in ("F4", "F5", "F6"):
        temporal = _resolve_train(arm)
        assert _model_signature(temporal) == f2_signature
        assert dict(f2.measurement_ownership_v3_1) == dict(
            temporal.measurement_ownership_v3_1
        )
        assert dict(f2.temporal_candidate_fusion_v3_1) == dict(
            temporal.temporal_candidate_fusion_v3_1
        )

    f2_calibration = dict(f2.calibration_conditioning_v3)
    assert f2_calibration["use_rays"] is True
    assert f2_calibration["use_stereo_pose"] is True
    assert f2_calibration["use_temporal_pose"] is False

    f3_stage_a = _resolve_train("F3_stage_a_control")
    f3 = _resolve_train("F3")
    assert _model_signature(f3_stage_a) == _model_signature(f3)
    assert dict(f3_stage_a.temporal_history_v2) == dict(f3.temporal_history_v2)
    assert str(f3_stage_a.data.derived_contract) == "legacy_v1"
    assert str(f3.data.derived_contract) == "legacy_v1"


@pytest.mark.parametrize("arm", ("F4", "F5", "F6"))
def test_f2_checkpoint_passes_strict_v31_stage_b_initialization(
    tmp_path: Path, arm: str
) -> None:
    f2 = _resolve_train("F2")
    f2_model = train.build_model(f2)
    checkpoint = tmp_path / "f2-final.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "parameter_count": f2_model.trainable_parameter_count,
            "config": OmegaConf.to_container(f2, resolve=True),
            "model": f2_model.state_dict(),
            "step": 5_000,
            "git_hash": "test",
        },
        checkpoint,
    )

    temporal = _resolve_train(arm)
    temporal_model = train.build_model(temporal)
    lineage = train.load_model_initialization_checkpoint(
        checkpoint,
        model=temporal_model,
        expected_parameter_count=temporal_model.trainable_parameter_count,
        required_sequence_length=1,
        required_seed=42,
        required_calibration_conditioning_v3=dict(f2.calibration_conditioning_v3),
        required_config_sections={
            "measurement_ownership_v3_1": dict(temporal.measurement_ownership_v3_1),
            "temporal_candidate_fusion_v3_1": dict(
                temporal.temporal_candidate_fusion_v3_1
            ),
            "supervision": dict(temporal.supervision),
        },
    )
    assert lineage["completed_step"] == 5_000
    assert lineage["source_seed"] == 42


def test_f3_control_checkpoint_passes_strict_stage_b_initialization(
    tmp_path: Path,
) -> None:
    stage_a = _resolve_train("F3_stage_a_control")
    stage_a_model = train.build_model(stage_a)
    checkpoint = tmp_path / "f3-control-final.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "parameter_count": stage_a_model.trainable_parameter_count,
            "config": OmegaConf.to_container(stage_a, resolve=True),
            "model": stage_a_model.state_dict(),
            "step": 5_000,
            "git_hash": "test",
        },
        checkpoint,
    )

    temporal = _resolve_train("F3")
    temporal_model = train.build_model(temporal)
    lineage = train.load_model_initialization_checkpoint(
        checkpoint,
        model=temporal_model,
        expected_parameter_count=temporal_model.trainable_parameter_count,
        required_sequence_length=1,
        required_config_sections={"supervision": dict(temporal.supervision)},
    )
    assert lineage["completed_step"] == 5_000
    assert lineage["calibration_conditioning_v3"] is None


def test_baseline_and_optional_configs_are_seed42_common_domain() -> None:
    for arm, scale, max_disp, prediction in (
        ("F0", 1, 384, "direct_full_resolution_ffs"),
        ("F1", 2, 192, "align_corners_false_bilinear"),
    ):
        config = OmegaConf.load(CONFIG_ROOT / f"{arm}.yaml")
        assert config.seed == 42
        assert config.arm == arm
        assert config.stage == "baseline"
        assert int(config.ffs.spatial_scale) == scale
        assert int(config.ffs.max_disp) == max_disp
        assert str(config.evaluation.prediction) == prediction
        assert list(config.evaluation.crop_hr) == [384, 768]
        assert list(config.evaluation.crop_origin_hr_xy) == [576, 348]
        assert str(config.evaluation.endpoint_protocol) == (
            "spring_seed42_common_fixed384_v1"
        )

    f7 = OmegaConf.load(CONFIG_ROOT / "F7.yaml")
    assert f7.seed == 42
    assert f7.arm == "F7"
    assert f7.stage == "optional_blocked"
    assert f7.depends_on == "F6"
    assert str(f7.required_contract.lineage) == "v3.1"
    assert "legacy v2" in str(f7.blocked_reason)
