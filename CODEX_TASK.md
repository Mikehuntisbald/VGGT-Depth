# FFS-Ω-TSR 主任务书

你是一名资深 PyTorch、双目视觉、三维几何和 CUDA 工程师。请在当前仓库实现项目
FFS-Ω-TSR：使用冻结的 Fast-FoundationStereo 产生低分辨率 metric disparity，使用
冻结的 VGGT-Omega 产生几何 depth/confidence/camera pose，训练一个轻量 causal
temporal super-resolution head，将低分辨率 disparity 重建为高分辨率 disparity。

执行状态（2026-09-01）：

- M0：`PASS_WITH_FALLBACK`；fallback 仅属于历史 NGC interface probe。
- M1：`PASS`；正式 observation/teacher 共 3,031 帧已完成并审计。
- M2：`PASS`；train 2,779 / validation 240 个 causal VGGT 窗口均已缓存、
  metric 化、完整质量门控并逐条 safe-zero 审计。
- M3：实现与数值测试完成；temporal training 使用 HR forward splat +
  z-buffer，invalid pose 严格禁用。
- M4/M5：T=1 与 T=3 训练/评测闭环已通过真实缓存 BF16 smoke；正式训练和
  go/no-go 精度结论尚未完成。
- M6：独立 HR epipolar refiner 已实现并通过 CUDA smoke，尚未接入正式训练。

用户已要求持续按本文件顺序推进。不得把未运行的训练、精度或验收项标记为通过。

## 一、不可变目标

1. 第一版空间比例为 x2：
   - HR 图像：默认 1280x720，训练随机裁剪 768x384。
   - FFS 输入：HR 图像下采样 2 倍。
   - 输出：HR disparity，单位必须是 HR pixels。
2. 时序为 causal：
   - student sequence length T=3。
   - VGGT context 使用当前和过去 4 个时刻。
   - VGGT 输入顺序：`L[t-4], R[t-4], ..., L[t], R[t]`。
   - 不允许使用未来帧。
3. FFS 和 VGGT-Omega 全程冻结：
   - 使用 `model.eval()`。
   - 使用 `torch.inference_mode()`。
   - 不创建它们的 optimizer state。
   - 默认训练时只读取离线 cache。
4. 首版 trainable model 控制在 12M 参数以内。
5. 当前 FFS disparity 是 metric owner。VGGT 和 history 主要处理低置信、invalid、遮挡和
   弱纹理区域。禁止网络在 FFS 高置信区域进行不受限的大幅修改。
6. 首版只实现 PyTorch，不实现 TensorRT、ONNX、ROS。PyTorch 基线正确后再考虑部署。
7. 所有几何代码都必须有单元测试。不允许通过“视觉上差不多”判断坐标、尺度和 pose
   convention。

## 二、环境要求

创建或使用三个独立环境：

- `env-ffs`：运行 Fast-FoundationStereo。
- `env-vggt`：运行 VGGT-Omega。
- `env-tsr`：训练重建模型。

它们通过磁盘 cache 交换数据。RTX 5090 必须使用已经验证能够支持 Blackwell 的 CUDA
12.8+ PyTorch build。不要因为上游 requirements 自动降级到 cu124。

先运行并记录：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name())
    print("capability:", torch.cuda.get_device_capability())
PY
```

将结果写入 `REPORT.md`。将上游仓库放入：

```text
third_party/Fast-FoundationStereo
third_party/vggt-omega
```

不要直接修改 `third_party`。必须通过 adapter、hook 或局部 wrapper 扩展行为。如果 FFS
的 Triton volume backend 在 5090 上失败，第一版自动回退到
`optimize_build_volume="pytorch1"`，并在 `REPORT.md` 记录，不要因此阻塞整个项目。

## 三、数据接口

实现 JSONL manifest。每行至少包含：

```json
{
  "sequence_id": "seq001",
  "frame_id": 12,
  "timestamp": 0.4,
  "left_path": "...",
  "right_path": "...",
  "K": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "baseline_m": 0.12,
  "gt_disparity_path": null
}
```

输入必须是已经 rectified 的双目图。随机 crop 后必须更新：

```text
cx_crop = cx - crop_x
cy_crop = cy - crop_y
```

下采样 s 倍后：

```text
fx_lr = fx_hr / s
fy_lr = fy_hr / s
cx_lr = cx_hr / s
cy_lr = cy_hr / s
```

crop origin 必须是 s 的整数倍，保证 HR/LR 对齐。实现并测试：

- crop intrinsics
- resize intrinsics
- disparity unit conversion
- depth/disparity conversion

核心公式：

```text
d_hr_unit = s * d_lr
Z = fx_hr * baseline / d_hr_unit
Z = fx_lr * baseline / d_lr
```

这两个深度结果必须数值一致。

## 四、FFS Adapter

实现 `src/backbones/ffs_adapter.py`。

1. 输入 RGB，范围按官方实现要求处理；输入尺寸自动 pad 到 32 的倍数，输出后正确
   unpad。
2. 输出 dataclass：

   ```text
   FFSOutput(
       disparity_lr_px,
       disparity_hr_px,
       confidence,
       entropy,
       last_update_magnitude,
       left_right_error,
       valid_mask,
       metadata,
   )
   ```

3. 不修改 upstream FFS，使用 forward hook 捕获 classifier logits 和最后一轮 update
   block 输出。
4. cost confidence：

   ```text
   p = softmax(logits, dim=disparity_dim)
   entropy = -sum(p * log(p + eps)) / log(D)
   cost_conf = 1 - entropy
   ```

5. update confidence：`update_conf = exp(-abs(last_delta) / tau)`。
6. cache 阶段可选运行 right-to-left FFS：

   ```text
   lr_error = abs(d_left(x) - d_right(x - d_left(x)))
   lr_conf = exp(-lr_error / sigma)
   ```

7. 最终 confidence：`conf = cost_conf * update_conf * lr_conf`。
8. FFS 输出 disparity 的单位必须明确记录。低分辨率输出转换成 HR pixel disparity 时必须
   乘 scale。
9. teacher cache：
   - 高精度 checkpoint。
   - 8 iterations。
   - HR 输入。
   - 用 confidence、LR consistency 和 valid mask 筛选 pseudo-GT。
10. observation cache：
    - 较快 checkpoint。
    - 4 iterations。
    - LR 输入。

实现 `tools/smoke_ffs.py` 和 `tools/cache_ffs.py`。

## 五、VGGT-Omega Adapter

实现 `src/backbones/vggt_omega_adapter.py`。

1. 输入过去 5 个双目时刻，共 10 张图：`L0,R0,L1,R1,...,L4,R4`。
2. 使用官方预处理逻辑，但必须返回每张图的：
   - `original_size`
   - `resized_size`
   - crop rectangle
   - scale
   - original-to-model transform
   - model-to-original transform
3. 缓存：
   - depth
   - depth confidence
   - extrinsics
   - predicted intrinsics，仅用于诊断
   - camera/register tokens
   - preprocessing metadata
4. 项目几何计算使用真实标定 K，不使用 VGGT 预测 intrinsics 替换真实标定。
5. 假设 extrinsics 是 camera-from-world，写单元测试验证 relative transform 方向。
6. 使用真实 stereo baseline 恢复 pose metric scale：

   ```text
   C = -R.T @ t
   b_i = norm(C_right_i - C_left_i)
   alpha = baseline_real / median(b_i)
   ```

   将所有 VGGT translation 和 depth 同时乘 `alpha`。
7. 输出 pose quality：
   - baseline coefficient of variation
   - relative stereo rotation error
   - reprojection residual
   - valid flag

   pose quality 不合格时，后续模型必须禁用 VGGT temporal pose。
8. VGGT depth prior 还要在可靠 FFS 区域进行 robust scale-only alignment：

   ```text
   a = argmin_a sum w * huber(d_ffs - a / z_vggt)
   d_vggt_aligned = a / z_vggt
   ```

   实现 IRLS 或加权中位数近似。默认不使用 additive shift。不足够可靠像素时返回 invalid
   prior。

实现 `tools/smoke_vggt.py` 和 `tools/cache_vggt.py`。

## 六、时序 z-buffer reprojection

实现 `src/geometry/zbuffer_reproject.py`。

输入：

- previous disparity/depth
- previous confidence
- K
- `E_previous` camera-from-world
- `E_current` camera-from-world

相对变换：

```text
T_current_previous = E_current @ inverse(E_previous)
```

必须实现 forward projection 和 z-buffer。输出：

```text
WarpResult(
    disparity,
    depth,
    confidence,
    valid_mask,
    visibility_mask,
    collision_mask,
    projected_uv,
    fractional_offset,
)
```

多个历史点投影到同一像素时保留最近深度。至少实现以下 pytest：

1. identity pose：warp 后 disparity 与输入一致。
2. 纯相机 x translation：投影方向和像素偏移正确。
3. 两个深度点投到同一像素：保留更近的点。
4. 出视野点：`valid=false`。
5. world-to-camera / camera-to-world convention：使用明确的数值例子测试。

不要求该 forward splat 对 pose 可导；history 输入默认 detach。

## 七、可训练模型

实现 `src/models/ffs_omega_tsr.py`。模型运行在 LR grid。

输入通道：

- current FFS disparity，HR pixel unit
- current FFS confidence
- VGGT aligned disparity
- VGGT confidence
- history warped disparity
- history confidence
- history visibility
- photometric residual
- fractional offset `du,dv`
- current HR left RGB

结构：

1. lightweight RGB pyramid encoder：channels = `[32,64,96]`。
2. geometry encoder：output channels = 64。
3. two-layer ConvGRU：hidden channels = 96。
4. heads：
   - three-source gate logits
   - bounded LR residual disparity
   - convex upsampling mask
   - shallow HR residual
   - log variance

源融合：

```text
weights = masked_softmax([logit_ffs, logit_vggt, logit_history])
d_mix = w_ffs * d_ffs + w_vggt * d_vggt + w_history * d_history
```

无效 source 的 logit 设置成非常小的值再 softmax。

LR residual：

```text
d_refined_lr = d_mix + residual_limit * tanh(delta_lr)
```

经过 convex x2 upsampling 和 HR residual 得到 `d_raw_hr`。

FFS anchor：

```text
gate = clamp(1 - upsample(conf_ffs) + 0.1, 0.1, 1.0)
d_final = d_ffs_bilinear_hr + gate * (d_raw_hr - d_ffs_bilinear_hr)
```

输出：

```text
ModelOutput(
    disparity_hr,
    disparity_raw_hr,
    source_weights,
    uncertainty,
    hidden_state,
)
```

总参数量必须打印并小于 12M。

## 八、损失

实现：

```text
L =
  1.00 * L_disparity +
  0.50 * L_measurement +
  0.20 * L_gradient +
  0.10 * L_temporal +
  0.05 * L_epipolar +
  0.01 * L_uncertainty +
  0.02 * L_gate_regularizer
```

`L_measurement`：在 LR 像素中心采样 HR prediction，除以 scale 后与原始 FFS LR
disparity 比较，只在 FFS trusted mask 中计算。

`L_temporal` 只在以下区域计算：

- visible
- non-collision
- low photometric residual
- geometry consistent

`L_gate_regularizer`：在高 FFS confidence 区域鼓励 `w_ffs` 较大；在 FFS invalid 区域
禁止强制 `w_ffs`。所有 loss 都必须处理 empty mask，不能产生 NaN。

## 九、训练阶段

### Stage A：空间 baseline

- sequence length = 1
- 不使用 history
- 不使用 VGGT pose
- 先比较 bilinear FFS vs RGB-guided spatial model
- 训练 5000 steps

### Stage B：时序模型

- sequence length = 3
- 使用 VGGT pose 和 z-buffer history
- 从 Stage A 初始化
- 训练 15000 steps

### Stage C：local HR epipolar refinement

实现左右 HR feature。围绕 predicted disparity 搜索 `[-2,-1,0,1,2]` HR pixels，构造
groupwise correlation，预测 bounded correction。从 Stage B 初始化继续训练。

### Stage D：x4

只有 x2 通过验收后才增加配置。从 x2 权重初始化，不从头训练。

默认训练配置：

```text
precision: bf16
micro batch: 2
gradient accumulation: 4
effective batch: 8
optimizer: AdamW
lr: 2e-4
weight decay: 1e-4
warmup: 500
gradient clip: 1.0
crop: 384x768
workers: 8
pin_memory: true
persistent_workers: true
torch.compile: false
```

OOM 时首先将 micro batch 改为 1，不要先缩小 crop 或改变模型。

## 十、评测

实现：

- EPE
- Bad-1
- Bad-2
- boundary EPE
- low-confidence-region EPE
- invalid-region completeness
- temporal disparity error
- trusted-region degradation
- point-cloud point-to-plane error
- invalid/negative/NaN rate

必须分别报告：

1. bilinear LR FFS
2. T=1 spatial model
3. T=3 temporal model
4. T=3 + VGGT prior
5. T=3 + VGGT prior + HR epipolar refinement

内部验收条件：

- T1 相对 bilinear：low-confidence EPE 至少下降 10%。
- T3 相对 T1：temporal error 至少下降 10%。
- invalid/hole completeness：至少提升 15%。
- FFS trusted region：误差恶化不超过 2%。
- NaN/negative output：小于 0.5%。

生成可视化：

- RGB
- LR FFS bilinear
- VGGT aligned disparity
- warped history
- final disparity
- pseudo-GT/GT
- absolute error
- source weights
- uncertainty
- point cloud
- temporal flicker video

## 十一、工程规范

1. 使用 dataclass 和 type hints。
2. 所有 public function 写清输入 shape、单位和坐标系。
3. 所有 disparity tensor 名称必须包含单位，例如 `disparity_lr_px`、
   `disparity_hr_px`。
4. 不允许使用含糊变量名 `depth_or_disp`。
5. cache 中写入：
   - upstream git commit
   - checkpoint hash
   - torch version
   - CUDA version
   - config hash
   - source image hash
6. cache 版本不一致时明确报错，不能静默复用。
7. 随机种子固定为 42。
8. 训练 checkpoint 必须包含 model、optimizer、scheduler、scaler、step、config、git
   hash。
9. 每完成一个 milestone，实际运行对应 smoke test/pytest。
10. 不允许只写代码不执行。
11. 非关键歧义采用最小可行实现，并记录在 `DECISIONS.md`。
12. 缺少 checkpoint、模型访问权限或数据路径时，输出精确缺失项和预期路径，不要使用
    假数据伪造实验结果。

## 十二、执行顺序

### M0

- 创建目录结构
- 检查 5090/PyTorch/CUDA
- 拉取或检查 third_party
- smoke test FFS/VGGT
- 建立 REPORT.md

### M1

- manifest
- crop/intrinsics/disparity 单元测试
- FFS adapter 和 cache
- 可视化 cache

### M2

- VGGT adapter
- baseline scale
- depth alignment
- pose quality
- 可视化 cache

### M3

- z-buffer reprojection
- 全部几何单元测试

### M4

- T=1 model
- losses
- train/eval
- baseline 报告

### M5

- T=3 ConvGRU
- temporal training/eval
- source-weight 和 uncertainty 可视化

### M6

- HR epipolar refinement
- 完整消融

最后生成：

- `README.md`
- `RUNBOOK.md`
- `DECISIONS.md`
- `REPORT.md`
- 可复现实验命令
- 指标 CSV
- 可视化目录
- 失败样例目录

现在先执行 M0。
