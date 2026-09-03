from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.cache_spring_ffs import (
    _cache_variant,
    _git_head,
    _resolve_manifest_path,
    build_parser,
)
from tools.cache_spring_gt import _record_gt_path
from data.cache_dataset import CacheIdentity, sha256_file
from train import load_cache_inventory_lineage


def test_spring_cache_resolves_relative_image_paths_from_manifest(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "manifests"
    image = manifest_dir / "frames" / "left.png"
    image.parent.mkdir(parents=True)
    image.touch()

    assert _resolve_manifest_path("frames/left.png", manifest_dir) == image.resolve()
    absolute = tmp_path / "absolute.png"
    absolute.touch()
    assert _resolve_manifest_path(str(absolute), manifest_dir) == absolute.resolve()


def test_spring_gt_cache_resolves_relative_disparity_paths_from_manifest(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "manifests"
    disparity = manifest_dir / "disp" / "frame.dsp5"
    disparity.parent.mkdir(parents=True)
    disparity.touch()

    class Record:
        gt_disparity_path = "disp/frame.dsp5"

    assert _record_gt_path(Record(), manifest_dir) == disparity.resolve()


def test_spring_cache_requires_a_git_checkout(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="readable Git checkout"):
        _git_head(tmp_path)


def test_spring_cache_git_revision_is_exact() -> None:
    repo = Path(__file__).resolve().parents[1] / "third_party" / "Fast-FoundationStereo"
    revision = _git_head(repo)
    assert len(revision) == 40
    assert revision == revision.lower()
    assert all(char in "0123456789abcdef" for char in revision)


def test_spring_observation_variants_have_separate_identity_and_output_roots() -> None:
    legacy_default = _cache_variant("observation")
    half = _cache_variant("observation", 2)
    full = _cache_variant("observation", 1)

    assert legacy_default == half
    assert half.component == "ffs-observation"
    assert half.output_directory == "observation"
    assert half.default_iterations == 4
    assert half.default_max_disp == 192
    assert half.default_hr_equivalent_max_disp == 384

    assert full.component == "ffs-observation-full-resolution"
    assert full.output_directory == "observation_full_resolution"
    assert full.default_iterations == 4
    assert full.default_max_disp == 384
    assert full.default_hr_equivalent_max_disp == 384
    assert full.component != half.component
    assert full.output_directory != half.output_directory


def test_spring_teacher_remains_full_resolution_and_rejects_half_scale() -> None:
    teacher = _cache_variant("teacher")
    assert teacher.scale == 1
    assert teacher.default_iterations == 8
    assert teacher.default_max_disp == 416
    with pytest.raises(ValueError, match="teacher scale must be 1"):
        _cache_variant("teacher", 2)


def test_spring_cache_parser_keeps_half_observation_as_cli_default() -> None:
    args = build_parser().parse_args(
        ["--manifest", "manifest.jsonl", "--output", "cache", "--role", "observation"]
    )
    assert args.scale is None
    assert _cache_variant(args.role, args.scale).resolution_mode == "half"
    assert args.repo.name == "Fast-FoundationStereo"


def test_training_cache_lineage_binds_receipt_and_inventory(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"frame_id":1}\n', encoding="utf-8")
    root = tmp_path / "observation"
    root.mkdir()
    inventory = root / "cache_manifest.jsonl"
    inventory.write_text('{"cache_sha256":"abc"}\n', encoding="utf-8")
    identity = CacheIdentity(
        component="ffs-observation",
        upstream_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        torch_version="test",
        cuda_version=None,
        config_sha256="c" * 64,
    )
    receipt = {
        "schema_version": 1,
        "identity": identity.to_dict(),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "cache_manifest": str(inventory.resolve()),
        "cache_manifest_sha256": sha256_file(inventory),
    }
    (root / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    lineage = load_cache_inventory_lineage(
        root, expected_identity=identity, manifest_path=manifest
    )
    assert lineage["cache_manifest_sha256"] == sha256_file(inventory)

    inventory.write_text('{"cache_sha256":"mutated"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="inventory lineage differs"):
        load_cache_inventory_lineage(
            root, expected_identity=identity, manifest_path=manifest
        )
