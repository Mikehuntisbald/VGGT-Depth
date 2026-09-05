from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.distributed.fsdp as torch_fsdp
from torch import nn

from losses.metric_stereo_video import MetricStereoVideoLoss
from tests.test_metric_stereo_video_system import _batch, _system
import tools.train_metric_stereo_video as metric_trainer
from tools.train_metric_stereo_video import (
    CheckpointResumeError,
    DistributedContext,
    _all_ranks_finite,
    _load_checkpoint,
    _lr_multiplier,
    _save_checkpoint,
    build_training_loss,
)


def _supervised_batch(frames: int = 3) -> dict[str, torch.Tensor]:
    batch = _batch(frames=frames)
    batch_size, _, _, _, height, width = batch["rgb"].shape
    disparity = torch.ones(batch_size, frames, 1, height, width)
    valid = torch.ones_like(disparity, dtype=torch.bool)
    batch.update(
        {
            "disparity_gt_left_px": disparity,
            "disparity_gt_right_px": disparity.clone(),
            "valid_gt_left": valid,
            "valid_gt_right": valid.clone(),
            "previous_disparity_gt_left_px": disparity[:, -2].clone(),
            "previous_valid_gt_left": valid[:, -2].clone(),
            "previous_disparity_gt_available": torch.ones(
                batch_size, dtype=torch.bool
            ),
            "dynamic_mask_current": torch.zeros(
                batch_size, 1, height, width, dtype=torch.bool
            ),
            "dynamic_mask_available": torch.ones(batch_size, dtype=torch.bool),
            "T_left_camera_from_world": torch.eye(4)
            .reshape(1, 1, 4, 4)
            .repeat(batch_size, frames, 1, 1),
        }
    )
    return batch


def test_complete_training_loss_mapping_is_finite_and_differentiable() -> None:
    system, stereo, vggt, _geometry = _system()
    system.train()
    batch = _supervised_batch()
    output = system(batch)
    result = build_training_loss(
        output,
        batch,
        MetricStereoVideoLoss(),
        aligned_vggt_weight=0.1,
    )

    assert torch.isfinite(result.total)
    assert set(result.breakdown.active_terms) == {
        "disparity",
        "depth",
        "temporal",
        "stereo_reprojection",
        "temporal_reprojection",
        "left_right_consistency",
        "right_disparity_supervision",
        "scale",
        "uncertainty",
        "validity",
    }
    result.total.backward()
    assert stereo.model.disparity_lr_px.grad is not None
    assert vggt.model.aggregator.projection.weight.grad is not None
    assert vggt.model.dense_head.depth_scale.grad is not None


def test_cosine_schedule_has_warmup_and_nonzero_floor() -> None:
    assert _lr_multiplier(0, 100, 10, 0.05) == 0.1
    assert _lr_multiplier(9, 100, 10, 0.05) == 1.0
    assert _lr_multiplier(100, 100, 10, 0.05) == 0.05


def test_single_rank_finite_consensus() -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    assert _all_ranks_finite(torch.tensor(1.0), context)
    assert not _all_ranks_finite(torch.tensor(float("nan")), context)


def test_fsdp_mixed_precision_does_not_create_batch_norm_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system, *_ = _system()
    captured: dict[str, object] = {}

    def fake_fsdp(model: nn.Module, **kwargs: object) -> nn.Module:
        captured.update(kwargs)
        return model

    monkeypatch.setattr(torch_fsdp, "FullyShardedDataParallel", fake_fsdp)
    monkeypatch.setattr(
        metric_trainer,
        "_apply_vggt_activation_checkpointing",
        lambda model: 1,
    )
    wrapped = metric_trainer._wrap_distributed(
        system,
        {"train": {"activation_checkpointing": True}},
        DistributedContext(0, 0, 2, torch.device("cpu")),
        no_fsdp=False,
    )

    assert wrapped is system
    assert captured["auto_wrap_policy"] is not None
    mixed_precision = captured["mixed_precision"]
    assert isinstance(mixed_precision, torch_fsdp.MixedPrecision)
    assert mixed_precision.cast_root_forward_inputs is False
    ignored_classes = tuple(mixed_precision._module_classes_to_ignore)
    assert ignored_classes == (type(system.vggt_backbone.model.dense_head),)
    assert not isinstance(nn.SyncBatchNorm(4), ignored_classes)


def _checkpoint_config(tmp_path: Path) -> dict[str, object]:
    train_manifest = tmp_path / "train.jsonl"
    validation_manifest = tmp_path / "validation.jsonl"
    train_manifest.write_text("{\"sample\":1}\n", encoding="utf-8")
    validation_manifest.write_text("{\"sample\":2}\n", encoding="utf-8")
    return {
        "data": {
            "train_manifest": str(train_manifest),
            "validation_manifest": str(validation_manifest),
        },
        "train": {
            "steps": 10,
            "output_dir": str(tmp_path / "output"),
            "gradient_accumulation": 2,
            "log_interval": 1,
            "validation_interval": 2,
            "validation_batches": 1,
            "checkpoint_interval": 2,
            "keep_last_checkpoints": 2,
        },
        "model": {"width": 3},
    }


def test_single_rank_checkpoint_round_trip_and_cursor(tmp_path: Path) -> None:
    config = _checkpoint_config(tmp_path)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    checkpoint = _save_checkpoint(
        model,
        optimizer,
        step=4,
        output_dir=tmp_path / "output",
        config=config,
        context=context,
        accumulation=2,
        batches_per_epoch=7,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    state = _load_checkpoint(
        model,
        optimizer,
        resume=checkpoint,
        config=config,
        context=context,
        accumulation=2,
        batches_per_epoch=7,
    )

    assert (state.step, state.epoch, state.batch_in_epoch) == (4, 1, 1)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert set(manifest["rank_file_sha256"]) == {"rank_0000.pt"}
    assert set(manifest["input_sha256"]) == {"train_manifest", "validation_manifest"}
    assert "tools/train_metric_stereo_video.py" in manifest["runtime_source_sha256"]


def test_checkpoint_resume_rejects_incomplete_or_changed_data(tmp_path: Path) -> None:
    config = _checkpoint_config(tmp_path)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = _save_checkpoint(
        model,
        optimizer,
        step=1,
        output_dir=tmp_path / "output",
        config=config,
        context=context,
        accumulation=2,
        batches_per_epoch=7,
    )
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CheckpointResumeError, match="not complete"):
        _load_checkpoint(
            model,
            optimizer,
            resume=checkpoint,
            config=config,
            context=context,
            accumulation=2,
            batches_per_epoch=7,
        )

    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    Path(config["data"]["train_manifest"]).write_text(  # type: ignore[index]
        "{\"sample\":999}\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointResumeError, match="manifest contents"):
        _load_checkpoint(
            model,
            optimizer,
            resume=checkpoint,
            config=config,
            context=context,
            accumulation=2,
            batches_per_epoch=7,
        )


def test_resume_binds_schedule_horizon_but_allows_runtime_cadence(
    tmp_path: Path,
) -> None:
    config = _checkpoint_config(tmp_path)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = _save_checkpoint(
        model,
        optimizer,
        step=1,
        output_dir=tmp_path / "output",
        config=config,
        context=context,
        accumulation=2,
        batches_per_epoch=7,
    )

    changed_horizon = copy.deepcopy(config)
    changed_horizon["train"]["steps"] = 20  # type: ignore[index]
    with pytest.raises(CheckpointResumeError, match="training config differs"):
        _load_checkpoint(
            model,
            optimizer,
            resume=checkpoint,
            config=changed_horizon,
            context=context,
            accumulation=2,
            batches_per_epoch=7,
        )

    runtime_only = copy.deepcopy(config)
    runtime_train = runtime_only["train"]
    assert isinstance(runtime_train, dict)
    runtime_train.update(
        {
            "output_dir": str(tmp_path / "resumed_elsewhere"),
            "log_interval": 3,
            "validation_interval": 5,
            "validation_batches": 4,
            "checkpoint_interval": 6,
            "keep_last_checkpoints": 1,
        }
    )
    state = _load_checkpoint(
        model,
        optimizer,
        resume=checkpoint,
        config=runtime_only,
        context=context,
        accumulation=2,
        batches_per_epoch=7,
    )
    assert (state.step, state.epoch, state.batch_in_epoch) == (1, 0, 2)
