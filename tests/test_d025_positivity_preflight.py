from __future__ import annotations

import pytest
from omegaconf import OmegaConf

import train
from tools.preflight_d025_positivity import (
    PROTOCOL_NAME,
    REQUIRED_UPDATES,
    validate_full_rerun_protocol,
)


def test_d025_config_declares_a_full_stage_b_rerun_from_stage_a() -> None:
    config = train.resolve_config("configs/ablations/d025_positivity_t3.yaml")
    train.validate_training_config(config)
    protocol = validate_full_rerun_protocol(config)

    assert protocol == {
        "name": PROTOCOL_NAME,
        "required_updates": REQUIRED_UPDATES,
        "stage_b_warm_start": "forbidden",
        "trainer_stage": "temporal",
    }


def test_d025_preflight_rejects_shortened_or_stage_b_warm_start_protocol() -> None:
    shortened = train.resolve_config("configs/ablations/d025_positivity_t3.yaml")
    OmegaConf.update(shortened, "train.steps", 10, merge=False)
    with pytest.raises(ValueError, match="train.steps"):
        validate_full_rerun_protocol(shortened)

    warm_start = train.resolve_config("configs/ablations/d025_positivity_t3.yaml")
    OmegaConf.update(
        warm_start, "ablation_protocol.stage_b_warm_start", "allowed", merge=False
    )
    with pytest.raises(ValueError, match="warm start"):
        validate_full_rerun_protocol(warm_start)
