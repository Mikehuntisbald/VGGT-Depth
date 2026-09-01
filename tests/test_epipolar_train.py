from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

import train_epipolar
from data.collate import collate_temporal_training_samples
from data.temporal_training_dataset import CachedTemporalTrainingDataset
from models.epipolar_refiner import HREpipolarRefiner
from models.epipolar_stage import FrozenTemporalEpipolarStage
from test_temporal_training_dataset import _make_temporal_cache
from train import build_model
from eval import _run_temporal_endpoint_ablation
from utils.seed import STRICT_CUBLAS_WORKSPACE_CONFIG, seed_everything


def _config() -> object:
    config = train_epipolar.resolve_epipolar_config(
        Path(__file__).parents[1] / "configs" / "epipolar_x2.yaml"
    )
    OmegaConf.update(config, "train.gradient_clip", 1.0)
    OmegaConf.update(config, "train.correction_regularizer_weight", 0.01)
    return config


def test_epipolar_config_enforces_config_driven_plus_minus_two_search() -> None:
    config = _config()
    train_epipolar.validate_epipolar_config(config)
    assert list(config.model.epipolar_offsets_hr_px) == [-2, -1, 0, 1, 2]

    invalid = copy.deepcopy(config)
    OmegaConf.update(invalid, "model.epipolar_offsets_hr_px", [-1, 0, 1])
    with pytest.raises(ValueError, match=r"\[-2,\+2\]"):
        train_epipolar.validate_epipolar_config(invalid)

    fp32 = copy.deepcopy(config)
    OmegaConf.update(fp32, "train.precision", "fp32")
    with pytest.raises(ValueError, match="precision=bf16"):
        train_epipolar.validate_epipolar_config(fp32)


def test_exact_frozen_stage_b_endpoint_predictor_cpu_shape(tmp_path: Path) -> None:
    manifest, observation, teacher, derived, _ = _make_temporal_cache(tmp_path)
    temporal = CachedTemporalTrainingDataset(
        manifest,
        observation,
        teacher,
        derived,
        crop_size_hr_hw=(4, 8),
        crop_mode="fixed",
        fixed_crop_origin_hr_xy=(2, 2),
    )
    batch = collate_temporal_training_samples([temporal[0]])
    config = _config()
    base = build_model(config).eval()

    with torch.no_grad():
        disparity = train_epipolar.predict_frozen_stage_b_endpoint(
            base, batch, config=config
        )
        evaluation_disparity = _run_temporal_endpoint_ablation(
            base, batch, config=config
        ).vggt_on.disparity_hr_px

    assert disparity.shape == (1, 1, 4, 8)
    assert torch.isfinite(disparity).all()
    assert torch.equal(disparity, evaluation_disparity)


class _TinyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 1, 1)


def _tiny_stage(seed: int) -> FrozenTemporalEpipolarStage:
    torch.manual_seed(seed)
    base = _TinyBase()

    def predictor(module: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(module, _TinyBase)
        return module.projection(batch["rgb_hr_sequence"][:, -1]).abs() + 1.0

    return FrozenTemporalEpipolarStage(
        base,
        HREpipolarRefiner(
            feature_channels=8,
            correlation_groups=2,
            head_channels=12,
        ),
        predictor,
    )


def _tiny_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(999)
    left = torch.rand((1, 3, 3, 6, 10), generator=generator)
    right = torch.rand((1, 3, 6, 10), generator=generator)
    target = torch.full((1, 3, 1, 6, 10), 2.0)
    intrinsics = torch.tensor(
        [[100.0, 0.0, 5.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]
    )
    return {
        "rgb_hr_sequence": left,
        "rgb_right_hr": right,
        "teacher_disparity_hr_px_sequence": target,
        "teacher_confidence_sequence": torch.ones_like(target),
        "teacher_trusted_mask_sequence": torch.ones_like(target, dtype=torch.bool),
        "K_hr_sequence": intrinsics.reshape(1, 1, 3, 3).repeat(1, 3, 1, 1),
        "K_right_hr": intrinsics.reshape(1, 3, 3),
        "epipolar_right_row_scale": torch.ones(1),
        "epipolar_right_row_offset_hr_px": torch.zeros(1),
        "epipolar_right_row_mapping_source": [
            "audited_same_row_rectified_pixels_v1"
        ],
    }


def test_deterministic_cpu_dry_forward_and_one_optimizer_step() -> None:
    config = _config()
    batch = _tiny_batch()
    first = _tiny_stage(42)
    second = _tiny_stage(42)
    first_base_before = {
        name: value.detach().clone() for name, value in first.base_model.state_dict().items()
    }

    dry_state = {
        name: value.detach().clone() for name, value in first.refiner.state_dict().items()
    }
    dry_loss = train_epipolar._stage_loss(first, batch, config)
    assert torch.isfinite(dry_loss.total)
    for name, value in first.refiner.state_dict().items():
        torch.testing.assert_close(value, dry_state[name])

    first_optimizer = torch.optim.AdamW(first.refiner.parameters(), lr=2e-4)
    second_optimizer = torch.optim.AdamW(second.refiner.parameters(), lr=2e-4)
    first_loss = train_epipolar.run_one_epipolar_optimizer_step(
        first, batch, config, first_optimizer
    )
    second_loss = train_epipolar.run_one_epipolar_optimizer_step(
        second, batch, config, second_optimizer
    )

    assert first_loss.detached_scalars() == pytest.approx(
        second_loss.detached_scalars()
    )
    for (first_name, first_value), (second_name, second_value) in zip(
        first.refiner.state_dict().items(), second.refiner.state_dict().items(), strict=True
    ):
        assert first_name == second_name
        torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
    for name, value in first.base_model.state_dict().items():
        torch.testing.assert_close(value, first_base_before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in first.base_model.parameters())


def test_optimizer_cannot_include_frozen_base_parameters() -> None:
    stage = _tiny_stage(42)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=2e-4)
    with pytest.raises(ValueError, match="exactly"):
        train_epipolar.run_one_epipolar_optimizer_step(
            stage, _tiny_batch(), _config(), optimizer
        )


def test_formal_rectification_audit_is_bound_and_fail_closed(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    receipt_path = repository / "reports" / "m6" / "epipolar_rectification_audit.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_train_sha256 = receipt["manifests"]["train"]["sha256"]
    audit = train_epipolar._validated_rectification_audit(
        receipt_path,
        expected_train_manifest_sha256=expected_train_sha256,
    )

    assert audit["status"] == "PASS"
    assert audit["contract_version"] == "audited_same_row_rectified_pixels_v1"
    assert audit["counts"]["sampled_frames"] == 96
    assert audit["pixel_evidence"]["p95_abs_right_y_minus_left_y_px"] < 3.0

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["status"] = "FAIL"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not publish"):
        train_epipolar._validated_rectification_audit(
            tampered_path,
            expected_train_manifest_sha256=expected_train_sha256,
        )


def test_runtime_source_bundle_rejects_scoped_dirty_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_epipolar.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M train_epipolar.py\n"),
    )

    with pytest.raises(RuntimeError, match="committed and clean"):
        train_epipolar._runtime_source_bundle()


def test_runtime_source_bundle_hashes_stage_c_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_epipolar.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=""),
    )
    monkeypatch.setattr(train_epipolar, "repository_git_hash", lambda path: "a" * 40)

    bundle = train_epipolar._runtime_source_bundle()
    paths = [record["path"] for record in bundle["files"]]

    assert "eval_epipolar.py" in paths
    assert paths == list(train_epipolar.stage_c_runtime_relative_paths())
    assert "configs/ablations/d025_positivity_t3.yaml" not in paths
    assert "configs/ablations/d025_stage_c_positivity.yaml" not in paths
    assert "tools/audit_d025_evaluation.py" not in paths
    assert len(paths) == 52

    controlled_bundle = train_epipolar._runtime_source_bundle(
        controlled_ablation=True
    )
    controlled_paths = [record["path"] for record in controlled_bundle["files"]]
    assert controlled_paths == list(
        train_epipolar.stage_c_runtime_relative_paths(controlled_ablation=True)
    )
    assert "configs/ablations/d025_positivity_t3.yaml" in controlled_paths
    assert "configs/ablations/d025_stage_c_positivity.yaml" in controlled_paths
    assert "tools/audit_d025_evaluation.py" in controlled_paths
    assert len(controlled_paths) == 55


def test_cpu_runtime_receipt_is_never_formal_bf16_eligible() -> None:
    seed_everything(42, deterministic=True, warn_only=False)
    runtime = train_epipolar._training_runtime(
        torch.device("cpu"), use_bf16=False
    )

    assert runtime["device_type"] == "cpu"
    assert runtime["autocast_enabled"] is False
    assert runtime["autocast_dtype"] is None
    assert runtime["deterministic_algorithms_enabled"] is True
    assert runtime["deterministic_algorithms_warn_only"] is False
    assert runtime["cublas_workspace_config"] == STRICT_CUBLAS_WORKSPACE_CONFIG
    assert runtime["cudnn_deterministic"] is True
    assert runtime["cudnn_benchmark"] is False
    assert runtime["strict_determinism_eligible"] is True
    assert runtime["formal_cuda_bf16_eligible"] is False


def test_seed_everything_enforces_fail_closed_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    seed_everything(42, deterministic=True, warn_only=False)
    runtime = train_epipolar._training_runtime(
        torch.device("cpu"), use_bf16=False
    )

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert runtime["deterministic_algorithms_enabled"] is True
    assert runtime["deterministic_algorithms_warn_only"] is False
    assert runtime["cudnn_deterministic"] is True
    assert runtime["cudnn_benchmark"] is False
    assert runtime["strict_determinism_eligible"] is True


def test_seed_everything_default_preserves_warning_only_policy() -> None:
    seed_everything(42, deterministic=True)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()

    # Leave the shared pytest process on the formal Stage-C policy.
    seed_everything(42, deterministic=True, warn_only=False)


@pytest.mark.parametrize(
    ("dry_run", "run_steps"),
    [(True, None), (True, 10), (False, 1)],
)
def test_cpu_smoke_policy_allows_only_explicit_bounded_execution(
    dry_run: bool, run_steps: int | None
) -> None:
    seed_everything(42, deterministic=True, warn_only=False)
    runtime = train_epipolar._training_runtime(
        torch.device("cpu"), use_bf16=False
    )
    train_epipolar._validate_execution_mode(
        torch.device("cpu"),
        allow_cpu_smoke=True,
        dry_run=dry_run,
        run_steps=run_steps,
        training_runtime=runtime,
    )


@pytest.mark.parametrize(
    ("allow_cpu_smoke", "dry_run", "run_steps", "message"),
    [
        (False, True, None, "requires --allow-cpu-smoke"),
        (True, False, None, "limited to dry-run or one step"),
        (True, False, 2, "limited to dry-run or one step"),
    ],
)
def test_cpu_smoke_policy_rejects_unapproved_or_unbounded_execution(
    allow_cpu_smoke: bool,
    dry_run: bool,
    run_steps: int | None,
    message: str,
) -> None:
    seed_everything(42, deterministic=True, warn_only=False)
    runtime = train_epipolar._training_runtime(
        torch.device("cpu"), use_bf16=False
    )
    with pytest.raises(RuntimeError, match=message):
        train_epipolar._validate_execution_mode(
            torch.device("cpu"),
            allow_cpu_smoke=allow_cpu_smoke,
            dry_run=dry_run,
            run_steps=run_steps,
            training_runtime=runtime,
        )


def test_cuda_policy_fails_closed_when_runtime_is_not_bf16_eligible() -> None:
    seed_everything(42, deterministic=True, warn_only=False)
    runtime = train_epipolar._training_runtime(
        torch.device("cpu"), use_bf16=False
    )
    runtime["device_name"] = "test-device"
    runtime["formal_cuda_bf16_eligible"] = False
    with pytest.raises(RuntimeError, match="native CUDA bf16"):
        train_epipolar._validate_execution_mode(
            torch.device("cuda:0"),
            allow_cpu_smoke=False,
            dry_run=False,
            run_steps=None,
            training_runtime=runtime,
        )


def test_execution_policy_rejects_warn_only_determinism() -> None:
    seed_everything(42, deterministic=True, warn_only=False)
    runtime = train_epipolar._training_runtime(
        torch.device("cpu"), use_bf16=False
    )
    runtime["deterministic_algorithms_warn_only"] = True
    runtime["strict_determinism_eligible"] = False
    with pytest.raises(RuntimeError, match="warn_only=false"):
        train_epipolar._validate_execution_mode(
            torch.device("cpu"),
            allow_cpu_smoke=True,
            dry_run=True,
            run_steps=None,
            training_runtime=runtime,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_runtime_receipt_binds_actual_device_and_native_bf16() -> None:
    seed_everything(42, deterministic=True, warn_only=False)
    current_device = torch.cuda.current_device()
    with torch.cuda.device(current_device):
        native_bf16_supported = torch.cuda.is_bf16_supported(
            including_emulation=False
        )
    runtime = train_epipolar._training_runtime(
        torch.device("cuda"), use_bf16=True
    )

    assert runtime["device"] == f"cuda:{current_device}"
    assert runtime["device_type"] == "cuda"
    assert runtime["device_name"] == torch.cuda.get_device_name()
    assert runtime["device_capability"] == list(torch.cuda.get_device_capability())
    assert runtime["bf16_supported"] is native_bf16_supported
    assert runtime["autocast_enabled"] is True
    assert runtime["autocast_dtype"] == "torch.bfloat16"
    assert runtime["strict_determinism_eligible"] is True
    assert runtime["formal_cuda_bf16_eligible"] is native_bf16_supported


class _EpochIndexDataset(Dataset[tuple[int, int]]):
    def __init__(self, size: int) -> None:
        self.size = size
        self.epoch = 0

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.epoch, index


def test_epipolar_data_cursor_reproduces_exact_next_batches() -> None:
    dataset = _EpochIndexDataset(10)
    sampler = train_epipolar.DeterministicEpochSampler(len(dataset), seed=42)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, drop_last=True)
    uninterrupted = train_epipolar._infinite_batches(loader, dataset, sampler)
    consumed = [next(uninterrupted) for _ in range(7)]
    del consumed
    expected = [next(uninterrupted) for _ in range(4)]

    resumed_dataset = _EpochIndexDataset(10)
    resumed_sampler = train_epipolar.DeterministicEpochSampler(
        len(resumed_dataset), seed=42
    )
    resumed_loader = DataLoader(
        resumed_dataset,
        batch_size=2,
        sampler=resumed_sampler,
        drop_last=True,
    )
    resumed = train_epipolar._infinite_batches(
        resumed_loader,
        resumed_dataset,  # type: ignore[arg-type]
        resumed_sampler,
        start_micro_step=7,
    )
    actual = [next(resumed) for _ in range(4)]

    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        torch.testing.assert_close(expected_batch[0], actual_batch[0], rtol=0, atol=0)
        torch.testing.assert_close(expected_batch[1], actual_batch[1], rtol=0, atol=0)


def test_resume_checkpoint_restores_model_optimizer_scheduler_rng_and_cursor(
    tmp_path: Path,
) -> None:
    seed_everything(42, deterministic=True, warn_only=False)
    config = _config()
    first = _tiny_stage(42)
    first_optimizer = torch.optim.AdamW(
        first.refiner.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )
    first_scheduler = torch.optim.lr_scheduler.LambdaLR(
        first_optimizer,
        lr_lambda=lambda update_index: train_epipolar.learning_rate_multiplier(
            update_index,
            total_steps=int(config.train.steps_epipolar),
            warmup_steps=int(config.train.warmup_steps),
        ),
    )
    train_epipolar.run_one_epipolar_optimizer_step(
        first, _tiny_batch(), config, first_optimizer
    )
    first_scheduler.step()

    runtime = train_epipolar._training_runtime(
        torch.device("cpu"), use_bf16=False
    )
    source_bundle = {
        "git_head": "a" * 40,
        "bundle_sha256": "b" * 64,
    }
    base_checkpoint = {
        "path": "/tmp/base.pt",
        "sha256": "c" * 64,
        "step": 15_000,
    }
    payload = train_epipolar._stage_c_checkpoint_payload(
        stage=first,
        optimizer=first_optimizer,
        scheduler=first_scheduler,
        completed_steps=1,
        config=config,
        git_hash="a" * 40,
        runtime_source_bundle=source_bundle,
        training_runtime=runtime,
        base_checkpoint=base_checkpoint,
        base_lineage={"valid": True},
        raw_lineage={"valid": True},
        base_completion={"complete": True},
        rectification_audit={"status": "PASS"},
        latest_loss={"total": 1.0},
        elapsed_seconds=2.0,
        batches_per_epoch=3,
    )
    checkpoint = tmp_path / "latest.pt"
    train_epipolar.atomic_torch_save(payload, checkpoint)
    expected_next_rng = torch.rand(8)

    # The uninterrupted arm advances one more update from the saved boundary.
    train_epipolar.run_one_epipolar_optimizer_step(
        first, _tiny_batch(), config, first_optimizer
    )
    first_scheduler.step()

    # The frozen Stage-B base is external to the Stage-C payload and is bound
    # by SHA-256, so the resumed arm must reconstruct that identical base.
    second = _tiny_stage(42)
    second_optimizer = torch.optim.AdamW(
        second.refiner.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )
    second_scheduler = torch.optim.lr_scheduler.LambdaLR(
        second_optimizer,
        lr_lambda=lambda update_index: train_epipolar.learning_rate_multiplier(
            update_index,
            total_steps=int(config.train.steps_epipolar),
            warmup_steps=int(config.train.warmup_steps),
        ),
    )
    step, elapsed, loss = train_epipolar._load_stage_c_training_checkpoint(
        checkpoint,
        stage=second,
        optimizer=second_optimizer,
        scheduler=second_scheduler,
        expected_config=train_epipolar._resolved_dict(config),
        expected_git_hash="a" * 40,
        expected_runtime_source_bundle=source_bundle,
        expected_training_runtime=runtime,
        expected_base_checkpoint=base_checkpoint,
        batches_per_epoch=3,
    )
    assert step == 1
    assert elapsed == 2.0
    assert loss == {"total": 1.0}
    torch.testing.assert_close(torch.rand(8), expected_next_rng, rtol=0, atol=0)
    train_epipolar.run_one_epipolar_optimizer_step(
        second, _tiny_batch(), config, second_optimizer
    )
    second_scheduler.step()

    for (first_name, first_value), (second_name, second_value) in zip(
        first.refiner.state_dict().items(),
        second.refiner.state_dict().items(),
        strict=True,
    ):
        assert first_name == second_name
        torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
    assert first_scheduler.state_dict() == second_scheduler.state_dict()
    for first_state, second_state in zip(
        first_optimizer.state.values(), second_optimizer.state.values(), strict=True
    ):
        assert first_state.keys() == second_state.keys()
        for name in first_state:
            first_value = first_state[name]
            second_value = second_state[name]
            if isinstance(first_value, torch.Tensor):
                torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
            else:
                assert first_value == second_value


def test_resume_log_reconciliation_discards_only_post_checkpoint_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"step": step, "loss": {"total": float(step)}} for step in range(1, 6)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    retained = train_epipolar._reconcile_training_log(
        path, completed_step=3, log_interval=1
    )

    assert retained == 3
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [
        1,
        2,
        3,
    ]

    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:3]) + '{"step":',
        encoding="utf-8",
    )
    assert (
        train_epipolar._reconcile_training_log(
            path, completed_step=3, log_interval=1
        )
        == 3
    )
    assert len(path.read_text().splitlines()) == 3

    # A syntactically valid checkpoint row without its newline delimiter must
    # be normalized before append, or the next JSON object would be glued to it.
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:2])
        + json.dumps(rows[2]),
        encoding="utf-8",
    )
    assert (
        train_epipolar._reconcile_training_log(
            path, completed_step=3, log_interval=1
        )
        == 3
    )
    assert path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rows[3]) + "\n")
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [
        1,
        2,
        3,
        4,
    ]

    path.write_text(
        json.dumps({"step": 1}) + "\n" + json.dumps({"step": 3}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(train_epipolar.CheckpointMismatchError, match="strict"):
        train_epipolar._reconcile_training_log(
            path, completed_step=1, log_interval=1
        )


def test_formal_base_completion_gate_is_fail_closed() -> None:
    complete = {
        "step": 15_000,
        "training_config": {
            "train": {"steps": 15_000, "steps_temporal": 15_000}
        },
    }
    receipt = train_epipolar._validate_base_completion(
        complete, expected_steps=15_000, required=True
    )
    assert receipt["complete"]

    incomplete = copy.deepcopy(complete)
    incomplete["step"] = 5_000
    with pytest.raises(ValueError, match="completed canonical Stage-B"):
        train_epipolar._validate_base_completion(
            incomplete, expected_steps=15_000, required=True
        )
    smoke = train_epipolar._validate_base_completion(
        incomplete, expected_steps=15_000, required=False
    )
    assert not smoke["complete"]


def test_epipolar_config_rejects_optimizer_provenance_drift() -> None:
    config = _config()
    OmegaConf.update(config, "train.optimizer", "sgd")
    with pytest.raises(ValueError, match="AdamW"):
        train_epipolar.validate_epipolar_config(config)


def test_checkpoint_boundary_rejects_nonfinite_refiner_state() -> None:
    stage = _tiny_stage(42)
    optimizer = torch.optim.AdamW(stage.refiner.parameters(), lr=2e-4)
    with torch.no_grad():
        next(stage.refiner.parameters()).reshape(-1)[0] = torch.nan
    with pytest.raises(FloatingPointError, match="non-finite Stage-C refiner"):
        train_epipolar._validate_finite_training_state(stage.refiner, optimizer)
