# Checkpoint placement

Large checkpoints are ignored by Git. M0 expects:

- `checkpoints/ffs/20-30-48/model_best_bp2_serialize.pth` for the fast observation model;
- `checkpoints/ffs/23-36-37/model_best_bp2_serialize.pth` for the HR teacher;
- `checkpoints/vggt/vggt_omega_1b_512.pt` for VGGT-Omega.

The two role-bound FFS artifacts used by the formal M1 caches are:

| Role | Tier | Bytes | SHA-256 |
|---|---:|---:|---|
| observation | `20-30-48` | 62,078,956 | `98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692` |
| HR pseudo-teacher | `23-36-37` | 71,098,210 | `af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990` |

The formal cache identity additionally binds the pinned upstream commit and
the complete inference configuration; a matching filename or tier label alone
is not sufficient for cache reuse.

On this machine the VGGT path is a symlink to the existing ModelScope cache:

```text
/home/haoyi/.cache/modelscope/hub/models/facebook/VGGT-Omega/vggt_omega_1b_512.pt
```

The cached file is 4,576,706,117 bytes with SHA-256
`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.

M0 additionally keeps the signed NVIDIA NGC v1.2 FFS artifact under
`checkpoints/ffs/ngc-v1.2/`. NGC does not publish its mapping to the GitHub
speed/accuracy tier names, so it is an official interface-smoke artifact, not
an observation/teacher identity substitute.

Every used artifact must have its byte size, SHA-256, source, and license status
recorded in `REPORT.md` and in its smoke receipt. A filename match is not proof
of official byte identity when upstream publishes no checksum.
