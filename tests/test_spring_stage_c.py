from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.manifest import ManifestRecord, write_manifest
from data.training_dataset import build_causal_windows
from tools.spring_stage_c import (
    SPRING_STAGE_C_PROTOCOL,
    SpringStageCError,
    require_spring_stage_c_coverage,
    audit_spring_screening_report,
    spring_temporal_coverage,
    strict_spring_holdout_lineage,
    validate_spring_checkpoint_marker,
    validate_spring_manifest,
)


def _record(sequence: str, frame: int) -> ManifestRecord:
    return ManifestRecord(
        sequence_id=sequence,
        frame_id=frame,
        timestamp=float(frame),
        left_path=f"/{sequence}/left/{frame}.png",
        right_path=f"/{sequence}/right/{frame}.png",
        K=((100.0, 0.0, 32.0), (0.0, 100.0, 24.0), (0.0, 0.0, 1.0)),
        baseline_m=0.065,
        gt_disparity_path=f"/{sequence}/disp/{frame}.dsp5",
        extras={"dataset": "spring", "split": "validation"},
    )


def _dataset(tmp_path: Path) -> tuple[SimpleNamespace, Path]:
    records = [_record("0007", index) for index in range(7)]
    manifest = tmp_path / "validation.jsonl"
    write_manifest(manifest, records)
    raw_manifest = tmp_path / "raw.jsonl"
    raw_manifest.write_text("{}\n", encoding="utf-8")
    derived_root = tmp_path / "derived"
    derived_root.mkdir()
    (derived_root / "cache_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    candidates = build_causal_windows(
        records, student_sequence_length=3, vggt_context_pairs=5
    )
    endpoints = {int(window.endpoint_index) for window in candidates}
    (derived_root / "run_receipt.json").write_text(
        json.dumps(
            {
                "selection": {"start_window": 0, "limit": None, "selected_windows": len(endpoints)},
                "counts": {"selected": len(endpoints)},
                "inputs": {
                    "vggt_cache_manifest": str(raw_manifest),
                    "vggt_cache_manifest_sha256": __import__(
                        "hashlib"
                    ).sha256(raw_manifest.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = SimpleNamespace(
        records=records,
        derived_entries={index: object() for index in endpoints},
        windows=[
            window
            for window in candidates
            if all(index in endpoints for index in window.student_indices)
        ],
        derived_cache_root=derived_root,
        manifest_path=manifest,
    )
    return dataset, manifest


def test_spring_coverage_is_manifest_bound_and_noncanonical(tmp_path: Path) -> None:
    dataset, manifest = _dataset(tmp_path)
    coverage = spring_temporal_coverage(dataset)
    assert coverage["protocol"] == SPRING_STAGE_C_PROTOCOL
    assert coverage["canonical"] is False
    assert coverage["manifest_records"] == 7
    assert coverage["derived_endpoint_records"] == 3
    # Raw five-pair VGGT geometry starts at endpoint 4; T=3 therefore has one
    # fully-derived causal window in a seven-frame split (endpoints 4,5,6).
    assert coverage["evaluable_t3_windows"] == 1
    assert require_spring_stage_c_coverage(
        coverage, manifest_sha256=coverage["manifest_sha256"]
    ) == coverage
    with pytest.raises(SpringStageCError, match="manifest SHA"):
        require_spring_stage_c_coverage(coverage, manifest_sha256="0" * 64)


def test_spring_coverage_rejects_subset_derived_cache(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path)
    dataset.derived_entries = {max(dataset.derived_entries): object()}
    with pytest.raises(SpringStageCError, match="incomplete"):
        spring_temporal_coverage(dataset)


def test_spring_manifest_rejects_non_spring_and_holdout_overlap(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.jsonl"
    records = [_record("0007", index) for index in range(2)]
    records[0] = ManifestRecord(
        sequence_id=records[0].sequence_id,
        frame_id=records[0].frame_id,
        timestamp=records[0].timestamp,
        left_path=records[0].left_path,
        right_path=records[0].right_path,
        K=records[0].K,
        baseline_m=records[0].baseline_m,
        gt_disparity_path=records[0].gt_disparity_path,
        extras={"dataset": "not_spring", "split": "validation"},
    )
    write_manifest(manifest, records)
    with pytest.raises(SpringStageCError, match="not exclusively Spring"):
        validate_spring_manifest(manifest, expected_split="validation")

    def _original(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {"same_manifest": False, "sequence_overlap": ["0007"]}

    with pytest.raises(SpringStageCError, match="overlap"):
        strict_spring_holdout_lineage(_original)


def test_spring_checkpoint_marker_binds_training_adapter_bytes(tmp_path: Path) -> None:
    adapter = tmp_path / "train_adapter.py"
    adapter.write_text("print('adapter')\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    marker = hashlib.sha256(adapter.read_bytes()).hexdigest()
    contract = tmp_path / "contract.py"
    contract.write_text("contract\n", encoding="utf-8")
    contract_marker = hashlib.sha256(contract.read_bytes()).hexdigest()
    import torch

    torch.save(
        {
            "config": {
                "spring_stage_c_protocol": SPRING_STAGE_C_PROTOCOL,
                "spring_stage_c_train_adapter_sha256": marker,
                "spring_stage_c_contract_sha256": contract_marker,
            }
        },
        checkpoint,
    )
    result = validate_spring_checkpoint_marker(
        checkpoint, train_adapter_path=adapter, contract_path=contract
    )
    assert result["train_adapter_sha256"] == marker
    adapter.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SpringStageCError, match="adapter SHA"):
        validate_spring_checkpoint_marker(
            checkpoint, train_adapter_path=adapter, contract_path=contract
        )


def test_spring_report_audit_is_explicitly_noncanonical(tmp_path: Path) -> None:
    evaluator = tmp_path / "eval.py"
    evaluator.write_text("screening evaluator\n", encoding="utf-8")
    manifest = tmp_path / "validation.jsonl"
    write_manifest(manifest, [_record("0007", index) for index in range(2)])
    manifest_sha = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    report = {
        "stage": "SPRING_STAGE_C_EPIPOLAR_SCREENING",
        "status": "SPRING_STAGE_C_SCREENING",
        "claims": {
            "spring_screening": True,
            "acceptance_eligible": False,
            "formal_holdout": False,
        },
        "spring_screening": {
            "protocol": SPRING_STAGE_C_PROTOCOL,
            "canonical": False,
            "limit": 1,
            "manifest": str(manifest.resolve()),
        },
        "formal_coverage": {
            "protocol": SPRING_STAGE_C_PROTOCOL,
            "canonical": False,
            "manifest_sha256": manifest_sha,
            "manifest_records": 2,
            "derived_endpoint_records": 1,
            "evaluable_t3_windows": 1,
        },
        "canonical_coverage": {"canonical": False},
        "lineage": {
            "spring_protocol": {"protocol": SPRING_STAGE_C_PROTOCOL},
            "held_out_validation": {"same_manifest": False, "sequence_overlap": []},
        },
        "source_hashes": {
            "evaluator_path": str(evaluator),
            "evaluator_sha256": __import__("hashlib").sha256(
                evaluator.read_bytes()
            ).hexdigest(),
        },
    }
    result = audit_spring_screening_report(
        report, validation_manifest=manifest
    )
    assert result["status"] == "PASS"
    assert result["acceptance_eligible"] is False
