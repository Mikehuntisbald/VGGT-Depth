from __future__ import annotations

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
