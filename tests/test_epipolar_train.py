from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

import train_epipolar
from data.collate import collate_temporal_training_samples
from data.temporal_training_dataset import CachedTemporalTrainingDataset
from models.epipolar_refiner import HREpipolarRefiner
from models.epipolar_stage import FrozenTemporalEpipolarStage
from test_temporal_training_dataset import _make_temporal_cache
from train import build_model
from eval import _run_temporal_endpoint_ablation


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
