#!/usr/bin/env python3
"""Materialize independently trainable A0-A6 configs from the frozen matrix."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.train_metric_stereo_video import _read_config  # noqa: E402


FORMAL_ARMS = tuple(f"A{index}" for index in range(7))


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(
            f"formal shared control {name}={expected!r} disagrees with "
            f"base config value {actual!r}"
        )


def resolve_arm_configs(
    matrix: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Resolve A0-A6 while enforcing the shared-training contract."""

    if matrix.get("independent_training_required") is not True:
        raise ValueError("formal matrix must require independent training")
    controls = matrix.get("shared_controls")
    arms = matrix.get("trainable_arms")
    if not isinstance(controls, Mapping) or not isinstance(arms, Mapping):
        raise ValueError("formal matrix requires shared_controls and trainable_arms")
    _require_equal(base.get("seed"), controls.get("seed"), "seed")
    data = base.get("data")
    train = base.get("train")
    if not isinstance(data, Mapping) or not isinstance(train, Mapping):
        raise ValueError("base config requires data and train mappings")
    _require_equal(data.get("clip_length"), controls.get("clip_length"), "clip_length")
    _require_equal(data.get("crop_size_hw"), controls.get("crop_size_hw"), "crop_size_hw")
    _require_equal(train.get("steps"), controls.get("optimizer_steps"), "optimizer_steps")
    _require_equal(
        train.get("micro_batch_size_per_rank"),
        controls.get("micro_batch_size_per_rank"),
        "micro_batch_size_per_rank",
    )
    _require_equal(
        train.get("gradient_accumulation"),
        controls.get("gradient_accumulation"),
        "gradient_accumulation",
    )
    _require_equal(
        train.get("validation_interval"),
        controls.get("validation_interval"),
        "validation_interval",
    )
    if controls.get("world_size") != 8:
        raise ValueError("formal ablations require all eight GPUs")

    resolved: dict[str, dict[str, Any]] = {}
    for arm_id in FORMAL_ARMS:
        arm = arms.get(arm_id)
        if not isinstance(arm, Mapping):
            raise ValueError(f"formal matrix is missing {arm_id}")
        fusion = arm.get("fusion")
        disabled = arm.get("disabled_loss_weights", {})
        if not isinstance(fusion, Mapping) or not isinstance(disabled, Mapping):
            raise ValueError(f"{arm_id} fusion/loss overrides must be mappings")
        config = copy.deepcopy(dict(base))
        config["experiment"] = f"metric_stereo_video_formal_{arm_id.lower()}"
        config.setdefault("fusion", {}).update(dict(fusion))
        for loss_name, value in disabled.items():
            if loss_name not in config.get("loss", {}):
                raise ValueError(f"{arm_id} disables unknown loss weight {loss_name!r}")
            if float(value) != 0.0:
                raise ValueError(f"{arm_id} disabled loss {loss_name!r} must be zero")
            config["loss"][loss_name] = 0.0
        config["train"]["output_dir"] = (
            f"runs/metric_stereo_video/formal_{arm_id.lower()}_seed{controls['seed']}"
        )
        config["formal_ablation"] = {
            "arm_id": arm_id,
            "name": arm.get("name"),
            "independently_trained": True,
            "world_size": 8,
            "checkpoint_selection": controls.get("checkpoint_selection"),
            "evaluation_contract": matrix.get("evaluation_contract"),
        }
        resolved[arm_id] = config
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/metric_stereo_video/formal_ablation_matrix.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/metric_stereo_video/formal_configs"),
    )
    args = parser.parse_args()
    matrix = _read_config(args.matrix)
    base_config = matrix.get("base_config")
    if not isinstance(base_config, str) or not base_config:
        raise ValueError("formal matrix requires base_config")
    base = _read_config(Path(base_config))
    resolved = resolve_arm_configs(matrix, base)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm_id, config in resolved.items():
        output_path = output_dir / f"{arm_id.lower()}.yaml"
        output_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        print(f"{arm_id}\t{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
