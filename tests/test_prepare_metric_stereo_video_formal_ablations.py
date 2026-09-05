from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.prepare_metric_stereo_video_formal_ablations import (
    FORMAL_ARMS,
    resolve_arm_configs,
)
from tools.train_metric_stereo_video import _read_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[dict, dict]:
    matrix = _read_config(
        PROJECT_ROOT / "configs/metric_stereo_video/formal_ablation_matrix.yaml"
    )
    base = _read_config(PROJECT_ROOT / matrix["base_config"])
    return matrix, base


def test_formal_matrix_resolves_seven_independent_eight_gpu_arms() -> None:
    matrix, base = _inputs()
    resolved = resolve_arm_configs(matrix, base)

    assert tuple(resolved) == FORMAL_ARMS
    assert len({config["train"]["output_dir"] for config in resolved.values()}) == 7
    for arm_id, config in resolved.items():
        assert config["formal_ablation"]["arm_id"] == arm_id
        assert config["formal_ablation"]["independently_trained"] is True
        assert config["formal_ablation"]["world_size"] == 8
        assert config["seed"] == base["seed"]
        assert config["data"] == base["data"]
        assert config["train"]["steps"] == base["train"]["steps"]

    assert resolved["A1"]["fusion"]["enable_vggt_gauge"] is True
    assert resolved["A1"]["fusion"]["enable_vggt_dense_features"] is False
    assert resolved["A2"]["fusion"]["enable_vggt_gauge"] is False
    assert resolved["A2"]["fusion"]["enable_vggt_dense_features"] is True
    assert resolved["A6"]["fusion"]["visibility_aware_gating"] is False


def test_formal_matrix_rejects_shared_control_drift() -> None:
    matrix, base = _inputs()
    changed = copy.deepcopy(matrix)
    changed["shared_controls"]["optimizer_steps"] += 1
    with pytest.raises(ValueError, match="optimizer_steps"):
        resolve_arm_configs(changed, base)
