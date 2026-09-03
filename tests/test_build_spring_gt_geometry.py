from __future__ import annotations

from dataclasses import dataclass

import pytest

from tools.build_spring_gt_geometry import _selected_records, build_parser


@dataclass(frozen=True)
class _Record:
    sequence_id: str
    frame_id: int


def test_sequence_warmup_preserves_original_indices_and_common_t3_floor() -> None:
    records = [
        *(_Record("a", frame_id) for frame_id in range(1, 8)),
        *(_Record("b", frame_id) for frame_id in range(1, 8)),
    ]
    selected = _selected_records(records, sequence_warmup=4)

    assert [(index, record.sequence_id, record.frame_id) for index, record in selected] == [
        (4, "a", 5),
        (5, "a", 6),
        (6, "a", 7),
        (11, "b", 5),
        (12, "b", 6),
        (13, "b", 7),
    ]
    # Three retained derived frames are first available at frame 7.
    assert selected[2][1].frame_id == 7
    assert selected[5][1].frame_id == 7


def test_sequence_warmup_validation_and_cli_default() -> None:
    records = [_Record("a", 1)]
    assert build_parser().parse_args(
        ["--manifest", "m", "--observation-root", "o", "--output", "x"]
    ).sequence_warmup == 0
    with pytest.raises(ValueError, match="non-negative"):
        _selected_records(records, sequence_warmup=-1)
    with pytest.raises(ValueError, match="removed every"):
        _selected_records(records, sequence_warmup=1)
