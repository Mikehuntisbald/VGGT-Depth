# Spring v3.1 + Fast-FoundationStereo completion monitor

Last poll: 2026-09-03 (Asia/Shanghai)

- F0/F1/F2: canonical fixed `384×768`, explicit origin `(576,348)`, all 1,302 common endpoints evaluated.
- F3: canonical Stage-B resume is active on GPU0 from checkpoint step 2,000; it is queued to finish at 15,000 steps, then receive a 1,302-endpoint fixed-crop evaluation.
- F4/F5: 15,000-step Stage-B runs are queued serially on GPU0 after F3; each will be evaluated into a new `eval_common_fixed384_final` directory.
- F6: canonical 15,000-step Stage-B run is active on GPU1 from the F2 Stage-A final checkpoint; a GPU1 monitor will run its 1,302-endpoint evaluation when `final.pt` appears.
- F7: optional Stage-C adapter is not implemented in the v3.1 common-domain queue and remains explicitly `OPTIONAL_NOT_RUN`.
- Current resource policy: one temporal training process per GPU (F3/F4/F5 on GPU0, F6 on GPU1); no vLLM process is resident. Fixed-crop evaluation is used to stay within 24-GiB GPU memory.
- Both active training logs have finite numeric values; no NaN/Inf/OOM/traceback has been observed.

Durable monitor logs:

- `runs/spring_v31_ffs/monitor_gpu0.log`
- `runs/spring_v31_ffs/monitor_gpu1.log`

The consolidated evidence report is regenerated with:

```bash
python tools/generate_spring_v31_ffs_common_domain_report.py
```
