from __future__ import annotations

from pathlib import Path

import pytest

from tools.cache_spring_ffs import _git_head, _resolve_manifest_path
from tools.cache_spring_gt import _record_gt_path


def test_spring_cache_resolves_relative_image_paths_from_manifest(tmp_path: Path) -> None:
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
    repo = Path(__file__).resolve().parents[1] / "third_party" / "FoundationStereo"
    revision = _git_head(repo)
    assert len(revision) == 40
    assert revision == revision.lower()
    assert all(char in "0123456789abcdef" for char in revision)
