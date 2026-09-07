# 论文指导下的 temporal 改进：实训验收

6 组实验均完成 12,500 步训练及全部 1,294 个验证端点评估。五项同时通过的版本：无。

参考方法： [CODD, WACV 2023](https://arxiv.org/html/2111.09337v2) 的 reset/fusion 分开监督；[TC-Stereo, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04579.pdf) 的学习特征匹配及排除相邻峰后的歧义判断。两个官方仓库的代码版本和逐项适配差异见 docs/temporal_repair_literature_protocol.md。

| 版本 | 全 GT EPE ↓ | temporal ↓ | 好像素受损 % ↓ | 相对 A5 新增 >5px 退化 % ↓ | 固定机会恢复 % ↑ |
|---|---:|---:|---:|---:|---:|
| v1 | 0.330162 | 0.258275 | 1.548202 | 0.118310 | 35.673887 |
| codd_component_capacity_control | 0.330574 | 0.258818 | 1.589809 | 0.112042 | 35.694728 |
| codd_component_codd | 0.329217 | 0.262984 | 0.843670 | 0.139273 | 26.037213 |
| codd_component_codd_regret | 0.329435 | 0.263695 | 0.802135 | 0.121470 | 25.487554 |
| learned_codd_capacity_control | 0.330377 | 0.258191 | 1.632115 | 0.112395 | 35.701692 |
| learned_codd_codd | 0.328236 | 0.259630 | 0.958170 | 0.141593 | 28.060019 |
| learned_codd_codd_regret | 0.328598 | 0.260488 | 0.952968 | 0.121973 | 27.507997 |

capacity_control 使用 v1 原损失；codd 使用官方误差阈值 5/1 px、独立归一化的接受/拒绝监督及 0.2 的近似等优融合正则；codd_regret 额外惩罚相对当前预测的大退化。supervision 组只有原 31 维输入；learned 组加入 24 维实际 A5 FFS 双目特征代价和代价峰差。每组内部结构、数据、初始化与训练预算相同，没有验证集校准。

验收条件和分母完全沿用 v1：整体与 temporal 不退步、好像素受损率下降、相对 A5 的 >5 px 退化率下降、固定机会区域恢复率提高。另报相对 v1 新增/消除大退化、1 px 退化、严重度、覆盖率与原生边界/动态/细节分区。完整结果见 results.json 与 evaluation_*/metrics.json。

固定失败点对照见 fixed_failure_points.json；其中序列 0005 帧 10 的 (253,147) 是前一轮增强监督错误拒绝可靠历史的案例，帧 32 的 (581,200) 是 v1 过度采纳坏历史的案例。failure_cases/ 使用一致色标展示 GT、原候选、预测和误差。

结论边界：这仍是冻结 A5 与历史候选的组件实验，没有复现 CODD 的 RAFT3D 非刚体运动、空间卷积融合和递归记忆，也没有复现 TC-Stereo 的对比代价学习和视差/梯度迭代细化。失败不能被表述为论文方法无效。全量评估集已反复用于开发；按序列 bootstrap 区间见 results.json，不宣称独立确认。

Stereo Any Video 的式 (12) 使用 t+1 帧，不能直接放入现有严格因果验收；相关机制若采用，需要单独实现因果版本：[ICCV 2025 论文](https://arxiv.org/html/2503.05549v2)。
