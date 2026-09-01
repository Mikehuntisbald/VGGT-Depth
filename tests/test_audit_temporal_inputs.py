from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from tools.audit_temporal_inputs import (
    TemporalInputAuditError,
    audit_temporal_inputs,
    main,
)


VIEW_ORDER = [
    "L[t-4]",
    "R[t-4]",
    "L[t-3]",
    "R[t-3]",
    "L[t-2]",
    "R[t-2]",
    "L[t-1]",
    "R[t-1]",
    "L[t]",
    "R[t]",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _identity(component: str, marker: str) -> dict[str, Any]:
    return {
        "component": component,
        "upstream_commit": marker * 40,
        "checkpoint_sha256": marker * 64,
        "torch_version": "2.test",
        "cuda_version": "12.8",
        "config_sha256": ("f" if marker != "f" else "e") * 64,
    }


def _save_cache(
    path: Path,
    *,
    identity: dict[str, Any],
    metadata: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "identity": identity,
            "metadata": metadata,
            "tensors": tensors,
        },
        path,
    )


def _make_manifest(root: Path, sequences: list[tuple[str, int]], name: str) -> tuple[Path, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    token = 0
    for sequence_id, count in sequences:
        for position in range(count):
            image_dir = root / "images" / name / f"{token:04d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            left = image_dir / "left.png"
            right = image_dir / "right.png"
            left.write_bytes(f"left-{name}-{token}".encode())
            right.write_bytes(f"right-{name}-{token}".encode())
            frame_id = 100 + position * 2
            timestamp = float(position) * 0.2
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "left_path": str(left.resolve()),
                    "right_path": str(right.resolve()),
                    "K": [[100.0, 0.0, 2.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
                    "baseline_m": 0.1,
                    "gt_disparity_path": None,
                    "rectified": True,
                    "source_frame_index": frame_id,
                    "source_time_sec": timestamp,
                    "token": f"{name}-{token}",
                }
            )
            token += 1
    path = root / f"{name}.jsonl"
    _jsonl(path, rows)
    return path, rows


def _causal_endpoints(rows: list[dict[str, Any]]) -> list[tuple[int, list[int]]]:
    by_sequence: dict[str, list[int]] = {}
    order: list[str] = []
    for index, row in enumerate(rows):
        sequence = row["sequence_id"]
        if sequence not in by_sequence:
            by_sequence[sequence] = []
            order.append(sequence)
        by_sequence[sequence].append(index)
    endpoints: list[tuple[int, list[int]]] = []
    for sequence in order:
        indices = by_sequence[sequence]
        for position in range(4, len(indices)):
            endpoints.append((indices[position], indices[position - 4 : position + 1]))
    return sorted(endpoints)


def _build_fixture(root: Path) -> dict[str, Path]:
    train_manifest, rows = _make_manifest(root, [("seq-a", 7), ("seq-b", 6)], "train")
    validation_manifest, _ = _make_manifest(root, [("seq-v", 5)], "validation")
    manifest_sha = _sha(train_manifest)
    observation_root = root / "observation"
    teacher_root = root / "teacher"
    raw_root = root / "raw-vggt"
    derived_root = root / "derived"
    observation_identity = _identity("ffs-observation", "a")
    teacher_identity = _identity("ffs-teacher", "b")
    raw_identity = _identity("vggt-omega", "c")
    observation_config = {
        "role": "observation",
        "scale": 2,
        "iterations": 4,
        "right_left_check": True,
        "provisional_checkpoint_role": False,
    }
    teacher_config = {
        "role": "teacher",
        "scale": 1,
        "iterations": 8,
        "right_left_check": True,
        "provisional_checkpoint_role": False,
    }
    observation_sha: dict[int, str] = {}
    for index, row in enumerate(rows):
        source = {
            "manifest_path": str(train_manifest.resolve()),
            "manifest_record": row,
            "left_sha256": _sha(Path(row["left_path"])),
            "right_sha256": _sha(Path(row["right_path"])),
        }
        valid = torch.tensor([[[[True, True, False], [True, False, False]]]])
        lr_error = torch.tensor([[[[0.1, 0.2, float("inf")], [0.3, float("inf"), float("inf")]]]])
        observation_path = observation_root / row["sequence_id"] / f"{row['frame_id']}.pt"
        _save_cache(
            observation_path,
            identity=observation_identity,
            metadata={"source": source, "config": observation_config},
            tensors={
                "observation_disparity_hr_px": torch.ones(1, 1, 2, 3),
                "observation_valid_mask": valid,
                "observation_left_right_error_lr_px": lr_error,
            },
        )
        observation_sha[index] = _sha(observation_path)
        _save_cache(
            teacher_root / row["sequence_id"] / f"{row['frame_id']}.pt",
            identity=teacher_identity,
            metadata={"source": source, "config": teacher_config},
            tensors={
                "teacher_disparity_hr_px": torch.ones(1, 1, 4, 6),
                "teacher_valid_mask": valid,
                "teacher_left_right_error_hr_px": lr_error,
            },
        )
    for cache_root, identity, config in (
        (observation_root, observation_identity, observation_config),
        (teacher_root, teacher_identity, teacher_config),
    ):
        (cache_root / "run_receipt.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest": str(train_manifest.resolve()),
                    "manifest_sha256": manifest_sha,
                    "selected_records": len(rows),
                    "written_records": len(rows),
                    "reused_records": 0,
                    "identity": identity,
                    "config": config,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    raw_config = {
        "schema_version": 1,
        "causal": True,
        "context_pairs": 5,
        "view_order": VIEW_ORDER,
        "current_left_view_index": 8,
    }
    endpoints = _causal_endpoints(rows)
    raw_rows: list[dict[str, Any]] = []
    raw_sha: dict[int, str] = {}
    for selection_index, (endpoint_index, context_indices) in enumerate(endpoints):
        record = rows[endpoint_index]
        ordered_images = []
        for view_index, (context_index, side) in enumerate(
            (item for context_index in context_indices for item in ((context_index, "left"), (context_index, "right")))
        ):
            path = Path(rows[context_index][f"{side}_path"])
            ordered_images.append(
                {
                    "view_index": view_index,
                    "view_label": VIEW_ORDER[view_index],
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
        source = {
            "manifest_path": str(train_manifest.resolve()),
            "manifest_sha256": manifest_sha,
            "manifest_indices": context_indices,
            "manifest_records": [rows[index] for index in context_indices],
            "target_manifest_index": endpoint_index,
            "target_sequence_id": record["sequence_id"],
            "target_frame_id": record["frame_id"],
            "target_timestamp": record["timestamp"],
            "ordered_images": ordered_images,
            "view_order": VIEW_ORDER,
            "causal": True,
        }
        raw_path = raw_root / record["sequence_id"] / f"{record['frame_id']}.pt"
        _save_cache(
            raw_path,
            identity=raw_identity,
            metadata={"source": source, "config": raw_config},
            tensors={
                "vggt_depth_current_left_arbitrary": torch.ones(1, 2, 3),
                "vggt_extrinsics_camera_from_world": torch.zeros(10, 3, 4),
            },
        )
        raw_sha[endpoint_index] = _sha(raw_path)
        raw_rows.append(
            {
                "selection_index": selection_index,
                "target_manifest_index": endpoint_index,
                "sequence_id": record["sequence_id"],
                "frame_id": record["frame_id"],
                "timestamp": record["timestamp"],
                "cache_path": str(raw_path.resolve()),
                "status": "written",
            }
        )
    _jsonl(raw_root / "cache_manifest.jsonl", raw_rows)
    raw_receipt = {
        "schema_version": 1,
        "manifest": str(train_manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "available_windows": len(endpoints),
        "selected_windows": len(endpoints),
        "written_records": len(endpoints),
        "reused_records": 0,
        "identity": raw_identity,
        "config": raw_config,
    }
    (raw_root / "run_receipt.json").write_text(
        json.dumps(raw_receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    derived_config = {
        "schema_version": 1,
        "algorithm": "baseline_metric_scale+scale_only_alignment+strict_pose_quality",
        "extrinsics_convention": "camera-from-world",
        "previous_left_view_index": 6,
        "current_left_view_index": 8,
        "invalid_temporal_pose_policy": "zero-filled with false validity tensor",
    }
    derived_rows: list[dict[str, Any]] = []
    by_sequence: dict[str, dict[str, int]] = {}
    pose_valid_count = 0
    static_valid_count = 0
    for selection_index, (endpoint_index, context_indices) in enumerate(endpoints):
        record = rows[endpoint_index]
        previous = rows[context_indices[-2]]
        pose_valid = selection_index % 2 == 0
        static_valid = selection_index % 3 == 0
        pose_valid_count += int(pose_valid)
        static_valid_count += int(static_valid)
        sequence_counts = by_sequence.setdefault(
            record["sequence_id"],
            {
                "selected": 0,
                "pose_valid": 0,
                "pose_rejected": 0,
                "static_prior_valid": 0,
                "static_prior_rejected": 0,
            },
        )
        sequence_counts["selected"] += 1
        sequence_counts["pose_valid" if pose_valid else "pose_rejected"] += 1
        sequence_counts["static_prior_valid" if static_valid else "static_prior_rejected"] += 1
        observation_path = observation_root / record["sequence_id"] / f"{record['frame_id']}.pt"
        raw_path = raw_root / record["sequence_id"] / f"{record['frame_id']}.pt"
        linkage = {
            "target_manifest_record": record,
            "target_sequence_id": record["sequence_id"],
            "target_frame_id": record["frame_id"],
            "target_timestamp": record["timestamp"],
            "vggt_raw_identity": raw_identity,
            "ffs_raw_identity": observation_identity,
            "previous_left": {
                "view_index": 6,
                "view_label": "L[t-1]",
                "path": previous["left_path"],
            },
            "current_left": {
                "view_index": 8,
                "view_label": "L[t]",
                "path": record["left_path"],
            },
        }
        disparity = torch.ones(1, 2, 3) if static_valid else torch.zeros(1, 2, 3)
        confidence = torch.full((1, 2, 3), 0.5) if static_valid else torch.zeros(1, 2, 3)
        valid_mask = torch.ones(1, 2, 3, dtype=torch.bool) if static_valid else torch.zeros(1, 2, 3, dtype=torch.bool)
        extrinsics = torch.ones(10, 3, 4) if pose_valid else torch.zeros(10, 3, 4)
        derived_path = derived_root / record["sequence_id"] / f"{record['frame_id']}.pt"
        _save_cache(
            derived_path,
            identity=_identity("vggt-ffs-derived-geometry", "d"),
            metadata={
                "config": derived_config,
                "target": {
                    "sequence_id": record["sequence_id"],
                    "frame_id": record["frame_id"],
                    "timestamp": record["timestamp"],
                },
                "source": {
                    "ffs_cache_path": str(observation_path.resolve()),
                    "ffs_cache_sha256": observation_sha[endpoint_index],
                    "vggt_cache_path": str(raw_path.resolve()),
                    "vggt_cache_sha256": raw_sha[endpoint_index],
                    "linkage": linkage,
                },
                "pose_quality": {
                    "pose_valid": pose_valid,
                    "alignment": {"static_prior_valid": static_valid},
                },
            },
            tensors={
                "vggt_extrinsics_camera_from_world_metric_temporal": extrinsics,
                "vggt_disparity_current_left_aligned_hr_px": disparity,
                "vggt_aligned_confidence": confidence,
                "vggt_aligned_valid_mask": valid_mask,
                "temporal_pose_valid": torch.tensor(pose_valid),
                "static_prior_valid": torch.tensor(static_valid),
            },
        )
        derived_rows.append(
            {
                "selection_index": selection_index,
                "target_manifest_index": endpoint_index,
                "sequence_id": record["sequence_id"],
                "frame_id": record["frame_id"],
                "timestamp": record["timestamp"],
                "cache_path": str(derived_path.resolve()),
                "cache_sha256": _sha(derived_path),
                "ffs_cache_path": str(observation_path.resolve()),
                "ffs_cache_sha256": observation_sha[endpoint_index],
                "vggt_cache_path": str(raw_path.resolve()),
                "vggt_cache_sha256": raw_sha[endpoint_index],
                "pose_valid": pose_valid,
                "static_prior_valid": static_valid,
                "status": "written",
            }
        )
    _jsonl(derived_root / "cache_manifest.jsonl", derived_rows)
    run_manifest = derived_root / "run_manifests" / "formal.jsonl"
    _jsonl(run_manifest, derived_rows)
    derived_manifest_sha = _sha(derived_root / "cache_manifest.jsonl")
    counts = {
        "selected": len(endpoints),
        "written": len(endpoints),
        "reused": 0,
        "pose_valid": pose_valid_count,
        "pose_rejected": len(endpoints) - pose_valid_count,
        "static_prior_valid": static_valid_count,
        "static_prior_rejected": len(endpoints) - static_valid_count,
    }
    derived_receipt = {
        "schema_version": 1,
        "component": "vggt-ffs-derived-geometry-batch",
        "selection": {"start_window": 0, "limit": None, "selected_windows": len(endpoints)},
        "counts": counts,
        "config": derived_config,
        "inputs": {
            "ffs_root": str(observation_root.resolve()),
            "vggt_root": str(raw_root.resolve()),
            "vggt_available_windows": len(endpoints),
            "vggt_cache_manifest": str((raw_root / "cache_manifest.jsonl").resolve()),
            "vggt_cache_manifest_sha256": _sha(raw_root / "cache_manifest.jsonl"),
        },
        "output": {
            "root": str(derived_root.resolve()),
            "cache_manifest": str((derived_root / "cache_manifest.jsonl").resolve()),
            "cache_manifest_sha256": derived_manifest_sha,
            "run_cache_manifest": str(run_manifest.resolve()),
            "run_cache_manifest_sha256": derived_manifest_sha,
        },
        "raw_input_audit": {
            "passed": True,
            "canonical_receipt_complete_manifest_coverage": True,
            "canonical_receipt_sha256": _sha(raw_root / "run_receipt.json"),
            "ffs_identity": observation_identity,
            "vggt_identity": raw_identity,
            "all_float_tensors_finite_records": len(endpoints),
            "causal_target_valid_records": len(endpoints),
            "ffs_identity_match_records": len(endpoints),
            "vggt_identity_match_records": len(endpoints),
            "weights_only_safe_load_records": len(endpoints),
        },
        "safe_zero_audit": {
            "passed": True,
            "all_float_tensors_finite_records": len(endpoints),
            "manifest_metadata_tensor_validity_consistent_records": len(endpoints),
            "records_audited": len(endpoints),
            "weights_only_safe_load_records": len(endpoints),
            "pose_rejected_zero_temporal_extrinsics": counts["pose_rejected"],
            "static_rejected_zero_prior_tensors": counts["static_prior_rejected"],
        },
        "canonical_update": {
            "current_selected_windows": len(endpoints),
            "existing_selected_windows": len(endpoints),
        },
        "by_sequence": {
            sequence_id: {"counts": sequence_counts}
            for sequence_id, sequence_counts in by_sequence.items()
        },
    }
    (derived_root / "run_receipt.json").write_text(
        json.dumps(derived_receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "observation_root": observation_root,
        "teacher_root": teacher_root,
        "raw_vggt_root": raw_root,
        "derived_root": derived_root,
    }


def _audit(paths: dict[str, Path]) -> dict[str, Any]:
    return audit_temporal_inputs(**paths)


def test_complete_temporal_lineage_passes_and_is_read_only(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    roots = [paths[key] for key in ("observation_root", "teacher_root", "raw_vggt_root", "derived_root")]
    mtimes = {
        str(path): path.stat().st_mtime_ns
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }

    report = _audit(paths)

    assert report["status"] == "PASS"
    assert report["manifests"]["train"]["record_count"] == 13
    assert report["causal_contract"]["expected_raw_and_derived_endpoints"] == 5
    assert report["causal_contract"]["evaluable_t3_endpoints"] == 1
    assert report["lineage_closure"]["per_record_weights_only_safe_load"] is True
    assert {
        str(path): path.stat().st_mtime_ns
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    } == mtimes


def test_raw_future_or_wrong_context_index_fails_closed(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    first_raw = next(paths["raw_vggt_root"].glob("*/*.pt"))
    payload = torch.load(first_raw, map_location="cpu", weights_only=True)
    payload["metadata"]["source"]["manifest_indices"][-1] += 1
    torch.save(payload, first_raw)

    with pytest.raises(TemporalInputAuditError, match="context indices mismatch"):
        _audit(paths)


def test_missing_safe_zero_proof_fails_closed(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    receipt_path = paths["derived_root"] / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["safe_zero_audit"]["passed"] = False
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(TemporalInputAuditError, match="safe_zero_audit did not pass"):
        _audit(paths)


def test_train_validation_sequence_overlap_is_rejected(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    validation_path = paths["validation_manifest"]
    rows = [json.loads(line) for line in validation_path.read_text().splitlines()]
    for row in rows:
        row["sequence_id"] = "seq-a"
    _jsonl(validation_path, rows)

    with pytest.raises(TemporalInputAuditError, match="sequence leakage"):
        _audit(paths)


def test_cli_refuses_report_inside_cache_root(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    output = paths["derived_root"] / "audit.json"
    exit_code = main(
        [
            "--train-manifest",
            str(paths["train_manifest"]),
            "--validation-manifest",
            str(paths["validation_manifest"]),
            "--observation-root",
            str(paths["observation_root"]),
            "--teacher-root",
            str(paths["teacher_root"]),
            "--raw-vggt-root",
            str(paths["raw_vggt_root"]),
            "--derived-root",
            str(paths["derived_root"]),
            "--json-out",
            str(output),
        ]
    )
    assert exit_code == 2
    assert not output.exists()
