from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from utils.checkpoint import (
    CheckpointMismatchError,
    load_model_initialization_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _objects() -> tuple[
    nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / float(step + 1)
    )
    return model, optimizer, scheduler


def test_checkpoint_round_trip_restores_complete_training_state(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model, optimizer, scheduler = _objects()
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    config = {"seed": 42, "model": {"width": 3}, "train": {"steps": 4}}
    checkpoint = tmp_path / "nested" / "latest.pt"

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=1,
        config=config,
        git_hash="a" * 40,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    completed_step = load_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_config=config,
        expected_parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
    )

    assert completed_step == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name])
    assert random.random() == expected_random
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert set(
        (
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "step",
            "config",
            "git_hash",
            "rng_states",
            "parameter_count",
        )
    ).issubset(payload)
    assert payload["scaler"] == {}


def test_checkpoint_rejects_config_and_model_identity_mismatch(tmp_path: Path) -> None:
    model, optimizer, scheduler = _objects()
    config = {"seed": 42, "model": {"width": 3}}
    checkpoint = tmp_path / "checkpoint.pt"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=0,
        config=config,
        git_hash="unknown",
        parameter_count=parameter_count,
    )

    with pytest.raises(CheckpointMismatchError, match="config"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config={"seed": 43, "model": {"width": 3}},
            expected_parameter_count=parameter_count,
            restore_rng=False,
        )
    with pytest.raises(CheckpointMismatchError, match="parameter count"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config=config,
            expected_parameter_count=parameter_count + 1,
            restore_rng=False,
        )


def test_historical_checkpoint_resume_accepts_only_exact_disabled_v3_defaults(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _objects()
    historical_config = {
        "seed": 42,
        "data": {"sequence_length": 1, "scale": 2},
        "model": {"width": 3},
        "train": {"stage": "spatial", "steps": 4},
    }
    current_legacy_config = copy.deepcopy(historical_config)
    current_legacy_config["data"].update(
        {
            "calibration_sidecar_path": None,
            "derived_contract": "legacy_v1",
            "calibration_sidecar_lineage": None,
        }
    )
    current_legacy_config["calibration_conditioning_v3"] = {
        "enabled": False,
        "protocol_version": "disabled",
        "use_rays": False,
        "use_stereo_pose": False,
        "use_temporal_pose": False,
    }
    checkpoint = tmp_path / "historical_v2.pt"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=1,
        config=historical_config,
        git_hash="a" * 40,
        parameter_count=parameter_count,
    )

    assert (
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config=current_legacy_config,
            expected_parameter_count=parameter_count,
            restore_rng=False,
        )
        == 1
    )

    for field, value in (
        ("derived_contract", "calibrated_stereo_v2"),
        ("calibration_sidecar_path", "/tmp/calibration.jsonl"),
    ):
        incompatible = copy.deepcopy(current_legacy_config)
        incompatible["data"][field] = value
        with pytest.raises(CheckpointMismatchError, match="config"):
            load_training_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_config=incompatible,
                expected_parameter_count=parameter_count,
                restore_rng=False,
            )
    enabled = copy.deepcopy(current_legacy_config)
    enabled["calibration_conditioning_v3"].update(
        {
            "enabled": True,
            "protocol_version": "dense_rays_factorized_pose_v3",
            "use_rays": True,
        }
    )
    with pytest.raises(CheckpointMismatchError, match="config"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config=enabled,
            expected_parameter_count=parameter_count,
            restore_rng=False,
        )


def test_temporal_initialization_loads_model_only_from_spatial_checkpoint(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _objects()
    config = {
        "data": {"sequence_length": 1},
        "train": {"stage": "spatial"},
    }
    checkpoint = tmp_path / "spatial.pt"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=7,
        config=config,
        git_hash="a" * 40,
        parameter_count=parameter_count,
    )
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    temporal_model, temporal_optimizer, _ = _objects()
    optimizer_before = temporal_optimizer.state_dict()

    lineage = load_model_initialization_checkpoint(
        checkpoint,
        model=temporal_model,
        expected_parameter_count=parameter_count,
        required_sequence_length=1,
    )

    for name, value in temporal_model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    assert temporal_optimizer.state_dict() == optimizer_before
    assert lineage["completed_step"] == 7
    assert lineage["checkpoint_sha256"]


@pytest.mark.parametrize(
    ("switches", "source_seed", "accepted"),
    (
        ((True, True, False), 42, True),
        ((False, False, False), 42, False),
        ((True, False, False), 42, False),
        ((False, True, False), 42, False),
        ((True, True, True), 42, False),
        ((True, True, False), 43, False),
    ),
)
def test_v3_temporal_initialization_requires_same_seed_a3_treatment(
    tmp_path: Path,
    switches: tuple[bool, bool, bool],
    source_seed: int,
    accepted: bool,
) -> None:
    model, optimizer, scheduler = _objects()
    treatment = {
        "enabled": True,
        "protocol_version": "dense_rays_factorized_pose_v3",
        "use_rays": switches[0],
        "use_stereo_pose": switches[1],
        "use_temporal_pose": switches[2],
    }
    required_a3 = {
        "enabled": True,
        "protocol_version": "dense_rays_factorized_pose_v3",
        "use_rays": True,
        "use_stereo_pose": True,
        "use_temporal_pose": False,
    }
    config = {
        "seed": source_seed,
        "data": {
            "sequence_length": 1,
            "derived_contract": "calibrated_stereo_v2",
        },
        "train": {"stage": "spatial"},
        "calibration_conditioning_v3": treatment,
    }
    checkpoint = tmp_path / f"candidate_{source_seed}_{switches}.pt"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=5_000,
        config=config,
        git_hash="a" * 40,
        parameter_count=parameter_count,
    )
    temporal_model, _, _ = _objects()

    if not accepted:
        with pytest.raises(CheckpointMismatchError, match="treatment|seed"):
            load_model_initialization_checkpoint(
                checkpoint,
                model=temporal_model,
                expected_parameter_count=parameter_count,
                required_sequence_length=1,
                required_seed=42,
                required_calibration_conditioning_v3=required_a3,
            )
        return

    lineage = load_model_initialization_checkpoint(
        checkpoint,
        model=temporal_model,
        expected_parameter_count=parameter_count,
        required_sequence_length=1,
        required_seed=42,
        required_calibration_conditioning_v3=required_a3,
    )
    assert lineage["source_seed"] == 42
    assert lineage["calibration_conditioning_v3"] == required_a3
