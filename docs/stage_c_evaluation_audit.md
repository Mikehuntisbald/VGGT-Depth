# Stage-C evaluation final-gate audit

`tools/audit_epipolar_evaluation.py` is a CPU-only, read-only receipt auditor.
It does not import a checkpoint or run inference. It validates an existing
Stage-C `metrics.json` against:

- the matching Stage-C training audit;
- the Stage-C `run_summary.json` when training is complete;
- the canonical Stage-B final evaluation report and its checkpoint/metrics;
- the exact held-out coverage, source hashes, and training/evaluation lineage.

## Formal evaluation source

The running formal Stage-C checkpoint is bound to the clean source bundle at:

```text
/tmp/ffs_omega_tsr_stagec_formal_source_4e6b7eb
git 4e6b7eb488201227e46b30e2ac90d34991466f2c
bundle 6a2cb37fb56dd79661c75d16c2128bc5355b857ea371d2b2c5de488e63acc4e5
eval_epipolar.py d88f69bc49ff8410c3628f5d6db3c9595a4ac791e07d0e72870ee3914468204e
```

Run the formal evaluator from that worktree. The current main-worktree
evaluator has additional visualization code and is not byte-identical to the
training-bound 52-file bundle, even though its metrics schema is unchanged.

## Final audit command

The audit output must be outside the evaluated directory:

```bash
python tools/audit_epipolar_evaluation.py \
  --evaluation-dir outputs/ffs_omega_tsr_x2/stage_c_eval \
  --stage-c-training-audit reports/m6/stage_c_training_audit_final.json \
  --stage-c-training-summary outputs/ffs_omega_tsr_x2/stage_c/run_summary.json \
  --stage-b-final-report reports/m4/stage_b_eval_final.json \
  --json-out reports/m6/stage_c_evaluation_final_audit.json
```

For an intermediate or limited diagnostic, omit
`--stage-c-training-summary` and supply the audit matching that exact Stage-C
checkpoint. Such a report is always `INELIGIBLE_FOR_FINAL_GATE`.

## Exit codes

- `0`: valid artifacts and `STAGE_C_M5_GATE_PASS`.
- `1`: valid artifacts, but metric gate failure or final-gate ineligibility.
- `2`: malformed, inconsistent, stale, or SHA-mismatched artifacts.

## Gate ownership

Only raw `T3_VGGT_epipolar` versus raw `T3_VGGT_base` owns gates. The auditor
requires improvement in boundary EPE and Bad-1. EPE and low-confidence EPE are
also judged and reported, but remain diagnostics under the original M5 gate.
The trusted-region degradation limit is 2%. Raw invalid, negative, NaN, and
the combined sign rate must stay below 0.5%; non-worsening versus the raw base
is reported separately as a diagnostic rather than inventing a new M5 gate.
The resulting status is deliberately named `STAGE_C_M5_GATE_PASS`, not a
whole-project all-gates pass: completeness and refined temporal consistency
are not re-established by this evaluator.

The full-corpus raw base and clamp0-base rows must exactly reproduce the
canonical Stage-B `T3_VGGT` and `T3_VGGT_clamp0` rows. Clamp0 remains a physical
postprocess diagnostic and never owns acceptance.

For archival integrity the auditor also:

- checks correction/confidence/candidate-coverage domains and paired
  better/worse/unchanged/finite count conservation;
- checks every horizontal correspondence method/domain count partition and
  its derived rates without letting that diagnostic alter accuracy masks;
- reopens the recorded train/validation raw-VGGT receipts/manifests and
  train/validation derived receipts/manifests, then verifies their bytes;
- parses `metrics.csv`, requires all four method rows to match `metrics.json`,
  and recomputes raw/clamp comparisons from the CSV rows;
- records both `metrics.json` and `metrics.csv` SHA-256 values in the audit.

The saturation threshold cannot be reconstructed from aggregate correction
statistics alone. The report labels that check `NOT_AUDITABLE` instead of
guessing a per-pixel result.

The evaluator's legal degenerate schemas are preserved: an empty candidate
domain is reported as `NOT_AUDITABLE`, and paired finite/nonfinite coverage is
still audited when any nonfinite pair makes the strict outcome and mean
improvement aggregates invalid. For an all-finite paired domain, outcome counts
must partition the domain and mean improvement must equal raw base EPE minus raw
refined EPE on that exact domain. Finite correction/confidence aggregate counts
are also preserved as subsets of candidate coverage; nonfinite values are
reported rather than turning a legal evaluator record into a schema error.

All accuracy targets are trusted HR FFS pseudo-GT. Therefore every report keeps
`paper_ground_truth=false`, `paper_accuracy=false`, and
`paper_claim_eligible=false`. The evaluator does not publish a refined Stage-C
temporal metric, so this audit cannot claim temporal improvement or
non-regression after epipolar refinement.
