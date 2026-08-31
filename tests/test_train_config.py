from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import train
from data.cache_dataset import CacheIdentity, canonical_json_sha256
from models.ffs_omega_tsr import ModelOutput


def _write_minimal_config(path: Path) -> None:
    path.write_text(
        "experiment: unit_test\n"
        "data:\n"
        "  sequence_length: 1\n"
        "train:\n"
        "  steps_spatial: 4\n"
        "  warmup_steps: 2\n",
        encoding="utf-8",
    )


def test_config_defaults_dotlist_and_contract_validation(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_minimal_config(path)
    config = train.resolve_config(
        path, ["train.steps_spatial=2", "train.warmup_steps=1"]
    )
    train.validate_stage_a_config(config)
    assert config.seed == 42
    assert config.train.steps_spatial == 2
    assert config.train.learning_rate == pytest.approx(2.0e-4)

    bad_sequence = train.resolve_config(path, ["data.sequence_length=3"])
    with pytest.raises(ValueError, match="T=1"):
        train.validate_stage_a_config(bad_sequence)
    bad_compile = train.resolve_config(path, ["train.compile_model=true"])
    with pytest.raises(ValueError, match="compile"):
        train.validate_stage_a_config(bad_compile)
    with pytest.raises(Exception):
        train.resolve_config(path, ["train.misspelled_key=1"])


def test_temporal_config_resolves_to_strict_causal_stage_b() -> None:
    config = train.resolve_config("configs/temporal_x2.yaml")
    assert train.validate_training_config(config) == "temporal"
    assert config.data.sequence_length == 3
    assert config.train.steps == 15000
    noncausal = train.resolve_config(
        "configs/temporal_x2.yaml", ["vggt.causal=false"]
    )
    with pytest.raises(ValueError, match="non-causal"):
        train.validate_stage_b_config(noncausal)


def test_accumulation_boundary_and_learning_rate_schedule() -> None:
    assert [train.should_optimizer_step(step, 4) for step in range(1, 9)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    values = [
        train.learning_rate_multiplier(index, total_steps=6, warmup_steps=2)
        for index in range(6)
    ]
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.0)
    # A tiny debug run may intentionally end inside the configured warmup.
    assert train.learning_rate_multiplier(0, total_steps=1, warmup_steps=500) == pytest.approx(
        1.0 / 500.0
    )


def test_completed_run_summary_is_atomic_and_cpu_peaks_are_null(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    summary = train.build_run_summary(
        stage="temporal",
        completed_steps=15_000,
        run_steps=15_000,
        elapsed_seconds=300.0,
        device=torch.device("cpu"),
        git_hash="a" * 40,
        resolved_config={"seed": 42, "data": {"sequence_length": 3}},
        final_checkpoint_path=checkpoint,
    )
    output = tmp_path / "receipts" / "run_summary.json"
    train.write_run_summary_atomic(output, summary)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["status"] == "TRAINING_COMPLETE"
    assert loaded["steps_per_second"] == pytest.approx(50.0)
    assert loaded["device_name"] is None
    assert loaded["peak_cuda_allocated_bytes"] is None
    assert loaded["peak_cuda_reserved_bytes"] is None
    assert len(loaded["config_fingerprint"]) == 64
    assert len(loaded["final_checkpoint"]["sha256"]) == 64
    assert not list(output.parent.glob("*.tmp"))


class _EpochDataset(Dataset[int]):
    def __init__(self, size: int) -> None:
        self.size = size
        self.epoch = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def _integer_loader(dataset: _EpochDataset, sampler: train.DeterministicEpochSampler):
    return DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        drop_last=True,
        collate_fn=lambda values: tuple(values),
    )


def test_epoch_sampler_and_micro_step_resume_reproduce_exact_next_batches() -> None:
    dataset = _EpochDataset(10)
    sampler = train.DeterministicEpochSampler(len(dataset), seed=42)
    iterator = train._infinite_batches(
        _integer_loader(dataset, sampler), dataset, sampler
    )
    consumed = [next(iterator) for _ in range(7)]
    expected_next = [next(iterator) for _ in range(4)]

    resumed_dataset = _EpochDataset(10)
    resumed_sampler = train.DeterministicEpochSampler(len(resumed_dataset), seed=42)
    resumed = train._infinite_batches(
        _integer_loader(resumed_dataset, resumed_sampler),
        resumed_dataset,
        resumed_sampler,
        start_micro_step=len(consumed),
    )
    assert [next(resumed) for _ in range(4)] == expected_next


def test_receipt_identity_is_bound_to_manifest_sha_and_component(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence_id": "s",
                "frame_id": 1,
                "timestamp": 0.0,
                "left_path": "/left.png",
                "right_path": "/right.png",
                "K": [[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
                "baseline_m": 0.1,
                "gt_disparity_path": None,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    identity = CacheIdentity(
        component="ffs-observation",
        upstream_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        torch_version="2.10.0",
        cuda_version="12.8",
        config_sha256=canonical_json_sha256({"scale": 2}),
    )
    root = tmp_path / "observation"
    root.mkdir()
    receipt = {
        "schema_version": 1,
        "identity": identity.to_dict(),
        "manifest_sha256": train.sha256_file(manifest),
        "selected_records": 1,
        "written_records": 1,
        "reused_records": 0,
    }
    (root / "run_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    assert train.load_receipt_identity(
        root, expected_component="ffs-observation", manifest_path=manifest
    ) == identity

    receipt["manifest_sha256"] = "0" * 64
    (root / "run_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest SHA-256"):
        train.load_receipt_identity(
            root, expected_component="ffs-observation", manifest_path=manifest
        )


def test_stage_a_loss_empty_masks_is_finite_and_differentiable() -> None:
    prediction = torch.full((1, 1, 4, 6), 4.0, requires_grad=True)
    log_variance = torch.zeros_like(prediction, requires_grad=True)
    source_weights = torch.softmax(
        torch.zeros(1, 3, 2, 3, requires_grad=True), dim=1
    )
    output = ModelOutput(
        disparity_hr_px=prediction,
        disparity_raw_hr_px=prediction,
        source_weights=source_weights,
        log_variance=log_variance,
        uncertainty=torch.exp(log_variance),
        hidden_state=(),
        anchor_gate=torch.ones_like(prediction),
        source_valid_mask=torch.zeros(1, 3, 2, 3, dtype=torch.bool),
    )
    batch = {
        "teacher_disparity_hr_px": torch.full_like(prediction, float("nan")),
        "teacher_confidence": torch.ones_like(prediction),
        "teacher_trusted_mask": torch.zeros_like(prediction, dtype=torch.bool),
        "observation_disparity_lr_px": torch.full((1, 1, 2, 3), float("nan")),
        "observation_confidence": torch.ones(1, 1, 2, 3),
        "observation_trusted_mask": torch.zeros(1, 1, 2, 3, dtype=torch.bool),
    }
    loss = train.compute_stage_a_loss(output, batch)
    assert torch.isfinite(loss.total)
    assert loss.total.item() == 0.0
    assert loss.temporal.requires_grad
    assert loss.epipolar.requires_grad
    loss.total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
