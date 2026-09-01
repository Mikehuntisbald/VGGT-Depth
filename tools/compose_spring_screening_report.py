#!/usr/bin/env python3
"""Compose an auditable Spring seed-42 screening report.

The training/evaluation entry points intentionally expose a richer legacy
metric schema than the Spring arm contract.  This utility maps only exact
semantic fields and leaves unsupported Spring-specific fields as JSON null;
it never infers detail/match/rigid-region values from a differently defined
metric.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


REQUIRED = (
    "overall_epe",
    "overall_1px",
    "high_detail_epe",
    "high_detail_1px",
    "low_detail_epe",
    "matched_epe",
    "unmatched_completion_1px",
    "unmatched_completion_2px",
    "rigid_temporal_residual_error",
    "non_rigid_temporal_residual_error",
    "boundary_epe",
    "ffs_trusted_measurement_error",
    "negative_rate",
    "zero_rate",
    "invalid_rate",
)
DIAGNOSTICS = (
    "age_2_survival_rate",
    "unique_age_fraction",
    "phase_variance",
    "candidate_depth_spread",
    "attention_entropy",
    "gain_by_fractional_phase_bucket",
    "gain_by_camera_motion_bucket",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def manifest_summary(path: Path) -> dict[str, Any]:
    """Return the bounded split facts used in the report protocol section."""

    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return {
        "records": len(rows),
        "sequences": sorted({str(row.get("sequence_id")) for row in rows}),
    }


def metric_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = value.get("value")
    return value


def method_metrics(report: Mapping[str, Any], method: str) -> dict[str, Any]:
    methods = report.get("methods")
    if not isinstance(methods, Mapping) or not isinstance(methods.get(method), Mapping):
        return {}
    return dict(methods[method])


def get_metric(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            value = metric_value(metrics[name])
            if value is not None:
                return value
    return None


def empty_contract() -> dict[str, Any]:
    return {name: None for name in (*REQUIRED, *DIAGNOSTICS)}


def map_arm(name: str, report: Mapping[str, Any]) -> dict[str, Any]:
    result = empty_contract()
    direct = report.get("metrics")
    if isinstance(direct, Mapping):
        for key in (*REQUIRED, *DIAGNOSTICS):
            if key in direct:
                result[key] = metric_value(direct[key])

    if name == "S0":
        # S0 is produced by the Spring-specific evaluator and already uses
        # the exact contract names.
        return result

    # ``eval.py --spring-native-metrics`` emits a separate, explicit Spring_GT
    # side channel.  Prefer it whenever present; never mix one native field
    # with legacy pseudo-GT fields from the same arm.
    native_report = report.get("spring_native_metrics")
    if (
        isinstance(native_report, Mapping)
        and native_report.get("status") == "AVAILABLE"
    ):
        native_methods = native_report.get("methods")
        if isinstance(native_methods, Mapping):
            native_method = (
                "T3_VGGT_epipolar"
                if name == "S6"
                else "T3_VGGT"
                if name in {"S4", "S5"}
                else "T1"
                if name == "S1"
                else "T3"
            )
            values = native_methods.get(native_method)
            if isinstance(values, Mapping):
                for key in REQUIRED:
                    if key in values:
                        result[key] = metric_value(values[key])
                native_topk = native_report.get("topk_diagnostics")
                if isinstance(native_topk, Mapping):
                    for key in DIAGNOSTICS:
                        if key in native_topk:
                            value = native_topk[key]
                            # Bucket gains are intentionally structured
                            # mappings, unlike scalar MetricResult receipts.
                            result[key] = (
                                value
                                if isinstance(value, Mapping)
                                and "value" not in value
                                else metric_value(value)
                            )
                return result

    if name == "S6":
        method = "T3_VGGT_epipolar"
    elif name in {"S4", "S5"}:
        method = "T3_VGGT"
    else:
        method = "T1" if name == "S1" else "T3"
    values = method_metrics(report, method)
    aliases = {
        "overall_epe": ("epe_px",),
        "overall_1px": ("bad_1",),
        "boundary_epe": ("boundary_epe_px",),
        "negative_rate": ("output_negative_rate",),
        "zero_rate": ("output_zero_rate",),
        "invalid_rate": ("output_invalid_rate",),
    }
    for target, names in aliases.items():
        if result[target] is None:
            result[target] = get_metric(values, *names)

    topk = report.get("diagnostics", {}).get("topk_candidate_complementarity_v3_1")
    if isinstance(topk, Mapping):
        topk_aliases = {
            "age_2_survival_rate": ("age2_survival_rate", "age_2_survival_rate"),
            "unique_age_fraction": ("unique_age_fraction",),
            "phase_variance": ("fractional_phase_variance", "phase_variance"),
            "candidate_depth_spread": ("candidate_depth_spread_m", "candidate_depth_spread"),
            "attention_entropy": (
                "metric_attention_weight_entropy",
                "topk_weight_entropy",
                "attention_entropy",
            ),
        }
        for target, names in topk_aliases.items():
            result[target] = get_metric(topk, *names)
    return result


def gpu_snapshot() -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used_mib": int(parts[1]),
                "memory_free_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_percent": int(parts[4]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--s0", type=Path, required=True)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--blocked", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    s0 = read_json(args.s0)
    blocked = read_json(args.blocked)
    validation_info = manifest_summary(args.manifest)
    train_manifest = args.arm_root.parent / "manifests" / "train.jsonl"
    train_info = (
        manifest_summary(train_manifest)
        if train_manifest.is_file()
        else {"records": None, "sequences": []}
    )
    arm_reports: dict[str, dict[str, Any]] = {"S0": s0}
    for name in ("S1", "S2", "S3", "S4", "S5", "S6"):
        path = args.arm_root / name / "eval" / "metrics.json"
        if path.is_file():
            arm_reports[name] = read_json(path)

    arms: list[dict[str, Any]] = []
    specs = {
        "S0": ("LR FFS bilinear", "none", False, "COMPLETE"),
        "S1": ("T1 spatial TSR", "gt", False, "COMPLETE"),
        "S2": ("T3 history, no VGGT, GT pose", "gt", False, "COMPLETE"),
        "S3": ("T3 top-K attention, GT pose", "gt", False, "COMPLETE"),
        "S4": ("S3 + VGGT depth, GT pose", "gt", True, "BLOCKED"),
        "S5": ("S4 + VGGT pose", "vggt", True, "BLOCKED"),
        "S6": ("S5 + HR epipolar refiner", "vggt", True, "BLOCKED"),
    }
    for name in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
        report = arm_reports.get(name)
        if report is None:
            continue
        primary = map_arm(name, report)
        method_name = (
            "T3_VGGT_epipolar"
            if name == "S6"
            else "T3_VGGT"
            if name in {"S4", "S5"}
            else "T1"
            if name == "S1"
            else "T3"
        )
        native_report = report.get("spring_native_metrics")
        native_methods = (
            native_report.get("methods")
            if isinstance(native_report, Mapping)
            and isinstance(native_report.get("methods"), Mapping)
            else {}
        )
        raw_method = (
            {}
            if name == "S0"
            else native_methods.get(method_name, method_metrics(report, method_name))
        )
        if not isinstance(raw_method, Mapping):
            raw_method = {}
        legacy_raw_method = method_metrics(report, method_name)
        native_topk = (
            native_report.get("topk_diagnostics")
            if isinstance(native_report, Mapping)
            else None
        )
        auxiliary = {
            "temporal_residual_error_native_px": get_metric(
                legacy_raw_method,
                "temporal_residual_error_native_px",
            ),
            "legacy_temporal_error_native_px": get_metric(
                legacy_raw_method, "temporal_disparity_error_native_px"
            ),
            "temporal_residual_error_paired_px": get_metric(
                legacy_raw_method, "temporal_residual_error_paired_px"
            ),
            "topk_diagnostics_status": (
                "AVAILABLE"
                if isinstance(native_topk, Mapping)
                or isinstance(
                    report.get("diagnostics", {}).get(
                        "topk_candidate_complementarity_v3_1"
                    ),
                    Mapping,
                )
                else "UNAVAILABLE"
            ),
        }
        arms.append(
            {
                "arm": name,
                "purpose": specs[name][0],
                "pose_source": specs[name][1],
                "use_vggt_depth": specs[name][2],
                # Preserve the evaluator's screening status verbatim.  A
                # one-step/non-holdout artifact must not be relabeled simply
                # as a formal "COMPLETE" result in the consolidated table.
                "status": str(report.get("status", specs[name][3])),
                "records_evaluated": report.get("records_evaluated", report.get("metrics", {}).get("frames")),
                "windows_evaluated": report.get("windows_evaluated"),
                "evaluation_status": report.get("status"),
                "evaluation_crop_mode": report.get(
                    "crop_mode",
                    report.get("crop_contract", {}).get("evaluation_crop_mode")
                    if isinstance(report.get("crop_contract"), Mapping)
                    else None,
                ),
                "evaluation_hr_crop": report.get(
                    "hr_crop",
                    report.get("fixed_hr_crop")
                    if isinstance(report.get("fixed_hr_crop"), list)
                    else None,
                ),
                "metrics": primary,
                "auxiliary": auxiliary,
                "raw_metrics_path": str((args.s0 if name == "S0" else args.arm_root / name / "eval" / "metrics.json").resolve()),
            }
        )

    blocked_arms: list[dict[str, Any]] = []
    for name in ("S4", "S5", "S6"):
        if name in arm_reports:
            continue
        row = next((item for item in blocked.get("arms", []) if item.get("arm") == name), None)
        blocked_arms.append(
            {
                "arm": name,
                "purpose": specs[name][0],
                "pose_source": specs[name][1],
                "use_vggt_depth": specs[name][2],
                "status": "BLOCKED",
                "blockers": [] if row is None else row.get("blockers", []),
                "resource_blocker": (
                    "raw VGGT-Omega cache requires CUDA; both GPUs are occupied by vLLM and have insufficient free memory"
                    if name in {"S4", "S5"}
                    else "raw VGGT unavailable; Spring Stage-C adapter/cache prerequisites are incomplete"
                ),
                "metrics": empty_contract(),
            }
        )
    arms.extend(blocked_arms)

    vggt_arm_reports = [arm_reports.get(name) for name in ("S4", "S5", "S6")]
    any_vggt_result = any(isinstance(report, Mapping) for report in vggt_arm_reports)
    vggt_devices = {
        str(report.get("device")).lower()
        for report in vggt_arm_reports
        if isinstance(report, Mapping) and report.get("device") is not None
    }
    if not any_vggt_result:
        device_note = "CPU for S0-S3; CUDA pending for VGGT-dependent S4-S6"
    elif vggt_devices == {"cpu"}:
        device_note = "CPU for S0-S6 (VGGT CPU fallback; resident vLLM left untouched)"
    elif vggt_devices == {"cuda"} or vggt_devices == {"cuda:0"}:
        device_note = "CPU for S0-S3; CUDA for VGGT-dependent S4-S6"
    else:
        device_note = (
            "CPU for S0-S3; VGGT-dependent arms used devices "
            + ", ".join(sorted(vggt_devices))
        )

    payload = {
        "schema_version": 1,
        "report": "spring_seed42_screening",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "seed": 42,
            "bounded_records_per_split": (
                validation_info["records"]
                if train_info["records"] == validation_info["records"]
                else None
            ),
            "train_records": train_info["records"],
            "validation_records": validation_info["records"],
            "train_sequences": train_info["sequences"],
            "validation_sequences": validation_info["sequences"],
            "expected_raw_vggt_endpoints_per_split": {
                "train": max(0, int(train_info["records"] or 0) - 4),
                "validation": max(0, int(validation_info["records"] or 0) - 4),
            },
            "expected_evaluable_t3_windows_with_vggt": {
                "train": max(0, int(train_info["records"] or 0) - 6),
                "validation": max(0, int(validation_info["records"] or 0) - 6),
            },
            "formal_coverage": False,
            "screening_only": True,
            "training_steps": 1,
            "device": device_note,
        },
        "lineage": {
            "spring_gt_teacher": "dataset GT (disp1_left), independent from FFS observation",
            "ffs_observation": "FoundationStereo 11-33-40 model_best_bp2.pth",
            "gt_pose": "Spring cam_data/extrinsics.txt, world_to_camera_opencv",
            "vggt_pose": (
                "VGGT-Omega derived pose cache for S5/S6; independent from Spring GT pose"
                if any_vggt_result
                else "not materialized; independent switch remains explicit"
            ),
            "train_manifest": str(train_manifest.expanduser().resolve()),
            "validation_manifest": str(args.manifest.expanduser().resolve()),
            "sequence_disjoint": True,
        },
        "required_metrics": list(REQUIRED),
        "topk_diagnostics": list(DIAGNOSTICS),
        "arms": arms,
        "resource_snapshot": {"gpus": gpu_snapshot()},
        "blocked_receipt": str(args.blocked.resolve()),
        "monitor": {
            "script": str((args.arm_root.parent.parent.parent / "tools" / "monitor_spring_vggt.py").resolve()),
            "log": str(
                (
                    args.arm_root.parent.parent.parent
                    / "runs"
                    / f"spring_seed42_vggt_live_{args.arm_root.parent.name.removeprefix('spring_seed42_')}"
                    / "monitor_session.jsonl"
                ).resolve()
            ),
            "policy": "launch requested arms only when all visible GPUs have >=12000 MiB free; never stop vLLM",
        },
        "notes": [
            "S0 uses the Spring-specific dense-GT evaluator; S1-S3 use the existing evaluator unless an explicit --spring-native-metrics side channel is present.",
            "null means unavailable/not evaluated; no Spring detail/match/rigid split or top-K diagnostic is imputed from a pseudo-domain.",
            "S1-S3 are one-step CPU screening smoke results and must not be interpreted as trained-model quality conclusions.",
            (
                "S4/S5 are blocked by missing raw/derived VGGT caches; current GPU memory is insufficient while vLLM occupies both GPUs."
                if not any_vggt_result
                else "S4-S6 VGGT-dependent artifacts were evaluated through the bounded screening path; all used the recorded device and no resident vLLM process was modified."
            ),
            "S6 uses tools/train_spring_epipolar.py and tools/eval_spring_epipolar.py; any produced result remains bounded screening-only and cannot replace canonical Stage-C evidence.",
            "S6 native Spring fields are an explicitly marked exact-zero-correction reuse of the fixed-crop S5 base side-channel; the S6 pseudo-GT evaluator itself remains the direct Stage-C receipt.",
            "Crop comparability: S0-S5 native rows use full-resolution evaluation, while the Stage-C S6 receipt and its exact-zero native reuse use the required fixed 384x768 center crop; absolute cross-arm ranking across these crop domains is exploratory only.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        "# Spring seed=42 seven-arm screening",
        "",
        f"This is a bounded {validation_info['records']}-frame-per-split screening smoke (train sequences {', '.join(train_info['sequences']) or 'unknown'}, validation sequences {', '.join(validation_info['sequences']) or 'unknown'}; one optimizer step). It is not formal full-corpus coverage.",
        "Crop note: S0-S5 native metrics are full-resolution; S6 Stage-C and its exact-zero native reuse are the required fixed 384x768 center crop, so cross-arm absolute values are exploratory rather than a single-domain ranking.",
        "",
        "| Arm | Pose | VGGT depth | Status | Overall EPE | Overall 1px | High-detail EPE | Low-detail EPE | Matched EPE | Unmatched @1/@2 | Boundary EPE | FFS trusted err | Neg/Zero/Invalid |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for arm in arms:
        m = arm["metrics"]
        def fmt(key: str) -> str:
            value = m.get(key)
            return "UNAVAILABLE" if value is None else f"{float(value):.6g}" if isinstance(value, (int, float)) else str(value)
        lines.append(
            f"| {arm['arm']} | {arm['pose_source']} | {'yes' if arm['use_vggt_depth'] else 'no'} | {arm['status']} | {fmt('overall_epe')} | {fmt('overall_1px')} | {fmt('high_detail_epe')} | {fmt('low_detail_epe')} | {fmt('matched_epe')} | {fmt('unmatched_completion_1px')}/{fmt('unmatched_completion_2px')} | {fmt('boundary_epe')} | {fmt('ffs_trusted_measurement_error')} | {fmt('negative_rate')}/{fmt('zero_rate')}/{fmt('invalid_rate')} |"
        )
    lines.extend(
        [
            "",
            "The table above uses the explicit native Spring side-channel where available; legacy-only fields remain `UNAVAILABLE`. See the JSON for the complete contract and blockers.",
            "",
            f"Machine-readable report: `{args.output_json.resolve()}`",
        ]
    )
    native_rows = [
        arm
        for arm in arms
        if isinstance(arm.get("metrics"), Mapping)
        and arm.get("arm") in {"S1", "S2", "S3", "S4", "S5", "S6"}
        and arm["metrics"].get("high_detail_1px") is not None
    ]
    if native_rows:
        lines.extend(
            [
                "",
                "## Native Spring temporal/top-K diagnostics",
                "",
                "| Arm | High-detail 1px | Rigid residual | Non-rigid residual | Age-2 survival | Unique-age | Phase variance | Candidate depth spread (m) | Attention entropy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in native_rows:
            metrics = arm["metrics"]
            def native_fmt(value: Any) -> str:
                if value is None:
                    return "UNAVAILABLE"
                if isinstance(value, (int, float)):
                    return f"{float(value):.6g}"
                return str(value)
            lines.append(
                "| {arm} | {hd} | {rigid} | {nonrigid} | {age2} | {unique} | {phase} | {depth} | {entropy} |".format(
                    arm=arm["arm"],
                    hd=native_fmt(metrics.get("high_detail_1px")),
                    rigid=native_fmt(metrics.get("rigid_temporal_residual_error")),
                    nonrigid=native_fmt(metrics.get("non_rigid_temporal_residual_error")),
                    age2=native_fmt(metrics.get("age_2_survival_rate")),
                    unique=native_fmt(metrics.get("unique_age_fraction")),
                    phase=native_fmt(metrics.get("phase_variance")),
                    depth=native_fmt(metrics.get("candidate_depth_spread")),
                    entropy=native_fmt(metrics.get("attention_entropy")),
                )
            )
        lines.extend(
            [
                "",
                "Fractional-phase and camera-motion gain buckets are structured mappings in the JSON; camera-motion bins are exploratory GT-pose tertiles when available.",
            ]
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(args.output_json.resolve()), "markdown": str(args.output_md.resolve()), "arms": [a["status"] for a in arms]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
