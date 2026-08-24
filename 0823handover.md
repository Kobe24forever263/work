# 0823 Handover：T-RO 论文实验进展、Stage 19 结果与下一阶段

更新日期：2026-08-23  
目标期刊：IEEE Transactions on Robotics（T-RO）  
当前主线：异构 Carter–Go2 双层仓库中的事件驱动 SMDP 调度、动态运输模式选择与连续负载适应

## 1. 当前结论

Stage 19 已完成并通过正式数据质量审查。新增实验在同一个不重置的 episode 中依次施加
`NORMAL → DENSE → BURST → RECOVERY` 四段负载，每段 20 个任务。机器人位置、楼层、
货物、队列和资源占用在阶段边界全部持续保留。30 个已冻结完整策略（MEDIUM、DENSE、
BURST 训练负载各 10 个独立训练种子）在 100 个全新共享测试种子上完成正式评估，共形成：

- PPO：3,000 episodes、240,000 个任务；
- 规则基线与单狗空白对照：200 episodes、16,000 个任务；
- 任务流 fingerprint 不一致 0 次；
- 重复 PPO 任务主键 0 条；
- 非法动作 0 次；
- 资源泄漏 0 次；
- PPO episode 终止时未记账任务 0 个。

因此 Stage 19 的数据可以进入正式论文分析。核心发现不是“策略固定选择某一种运输方式”，
而是同一策略在持续状态下根据队列、机器人位置、候选 ETA、资源占用和交接风险改变
`SINGLE_CAR`、`SINGLE_DOG`、`CAR_DOG_CAR` 的选择比例。

## 2. 本阶段解决了什么问题

旧的连续负载实验只有 `NORMAL → DENSE → BURST` 上升段，不能回答突发任务结束后系统能否
清空积压。旧结果也缺少完整逐 episode、逐任务行，无法独立重算交叉种子置信区间。

本阶段补齐了两项 T-RO 关键缺口：

1. 增加 `RECOVERY` 段，检查负载回落后等待时间和运输模式是否恢复；
2. 保存每个模型、测试种子、任务和阶段的完整明细，支持训练种子 × 测试种子的两维统计。

这对应 `TRO_实验缺口与图表路线图_副本.html` 中原先未完成的“连续负载适应与恢复”和
“结果明细与统计口径”两项。前者现已具备逻辑仿真正式数据，后者已完成修复。

## 3. Stage 19 实现与协议

### 3.1 场景

新增 `MIXED_LOAD_RECOVERY`：

| 阶段 | 任务数 | 到达负载 | 状态重置 |
|---|---:|---|---|
| NORMAL | 20 | normal interval | 否 |
| DENSE | 20 | dense interval | 否 |
| BURST | 20 | burst interval | 否 |
| RECOVERY | 20 | 恢复为 normal interval | 否 |

任务类别在各阶段保持同一套 balanced 20-task 组成，阶段间只改变到达强度。这样可把运输模式
变化主要归因于系统负载和持续机器人状态，而不是任务类别比例突然改变。

### 3.2 策略与对照

- `MEDIUM` PPO：10 个在 MEDIUM 到达负载下训练的 `FULL_CONTEXT_V2` 权重；
- `DENSE` PPO：10 个在 DENSE 到达负载下训练的 `FULL_CONTEXT_V2` 权重；
- `BURST` PPO：10 个在 BURST 到达负载下训练的 `FULL_CONTEXT_V2` 权重；
- 时间贪心规则基线：相同环境、动作掩码和任务流；
- 单狗空白对照：只开放 `SINGLE_DOG`，用于量化异构协作系统相对四 Go2 独立运输的收益。

这里的 MEDIUM/DENSE/BURST 是训练时的任务到达负载，不是三种运输模式。每个 PPO 权重在
每一个决策点仍同时看到合法的单车、单狗和车—狗—车候选。

### 3.3 随机性与冻结

- 训练权重：测试前冻结 30/30，通过训练摘要、参数更新、有限数值、动作合法性和资源安全门禁；
- 正式测试种子：`56000000–56000099`；
- bootstrap 种子：`57000000`；
- 交接采样：`TASK_KEYED_COMMON_RANDOM_NUMBERS`；
- 配对键：测试种子、任务 ID 与交接类型；
- bootstrap：10,000 次，交叉重采样训练种子与共享测试种子；
- profile-to-profile 比较为探索性；PPO-to-baseline 为本阶段主要锁定比较。

### 3.4 失败记账修复

单狗空白对照在长连续任务中可能进入 `NO_FEASIBLE_CHAIN`。旧统计只登记一次全局失败，
会遗漏终止时仍未服务的任务，从而虚高成功率。本阶段将所有终止时未完成任务显式记为
`TERMINAL_UNRESOLVED`，保留原始阶段、到达时间和任务 ID，并全部纳入成功率分母。

同时，80 个任务使旧 3,600 s episode 上限可能截断 Recovery 尾部。本阶段只对新四阶段
场景把上限调整为 5,400 s；旧场景仍保持 3,600 s，不改变既有实验口径。

## 4. 正式结果

### 4.1 绝对性能

下表是训练种子和 100 个测试种子上的均值。回报为原始 episode return；连续 80 任务会
持续累计 outstanding-time 等成本，因此出现负值是正常的，比较时应看相同协议下的相对高低，
不能把“是否为正”当作性能门禁。

| 方法 | 成功率 | 原始回报 | 成功吞吐（task/h） | 平均等待（s） | P95 等待（s） | 平均流转（s） | 距离/任务 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MEDIUM PPO | 99.159% | -431.94 | 71.62 | 166.02 | 659.34 | 210.43 | 50.87 |
| DENSE PPO | 97.990% | -313.01 | 70.80 | 124.96 | 519.09 | 168.32 | 60.91 |
| BURST PPO | 98.066% | -308.35 | 70.85 | 123.03 | 503.34 | 166.30 | 60.07 |
| 时间规则 | 97.087% | -457.12 | 70.16 | 173.38 | 727.11 | 216.04 | 71.09 |
| 单狗空白对照 | 96.938% | -935.09 | 69.82 | 330.87 | 1286.68 | 387.41 | 45.23 |

### 4.2 相对时间规则基线

差值均为 `PPO − 规则基线`；等待时间为负表示 PPO 更好。

| 训练负载 | 成功率差 | 回报差 | 成功吞吐差（task/h） | 平均等待差（s） | P95 等待差（s） |
|---|---:|---:|---:|---:|---:|
| MEDIUM | +2.071 pp [1.749, 2.403] | +25.18 [-10.18, 61.87] | +1.458 [1.236, 1.695] | -7.37 [-19.76, 4.34] | -67.77 [-118.51, -18.61] |
| DENSE | +0.903 pp [0.544, 1.309] | +144.11 [129.25, 159.11] | +0.636 [0.381, 0.923] | -48.42 [-53.72, -43.24] | -208.02 [-233.75, -182.40] |
| BURST | +0.979 pp [0.554, 1.440] | +148.77 [133.60, 164.00] | +0.693 [0.398, 1.021] | -50.36 [-55.61, -45.10] | -223.77 [-249.28, -198.21] |

可确认的结论：

- 三组 PPO 的成功率和成功吞吐均高于时间规则，95% CI 不跨 0；
- DENSE/BURST PPO 的回报、平均等待和 P95 等待均稳定优于规则；
- MEDIUM PPO 的平均等待和回报 CI 跨 0，不应声称这两项显著优于规则；
- MEDIUM PPO 的 P95 等待仍有明确改善，说明其主要降低尾部等待而非稳定降低均值。

### 4.3 相对单狗空白对照

| 训练负载 | 成功率差 | 回报差 | 成功吞吐差（task/h） | 平均等待差（s） | P95 等待差（s） |
|---|---:|---:|---:|---:|---:|
| MEDIUM | +2.221 pp [1.450, 3.054] | +503.16 [453.14, 555.37] | +1.799 [1.237, 2.412] | -164.85 [-183.40, -147.44] | -627.34 [-701.69, -551.92] |
| DENSE | +1.053 pp [0.178, 1.959] | +622.08 [582.55, 663.71] | +0.978 [0.378, 1.621] | -205.90 [-220.69, -191.33] | -767.59 [-830.56, -706.31] |
| BURST | +1.129 pp [0.241, 2.064] | +626.74 [586.87, 667.80] | +1.034 [0.403, 1.703] | -207.84 [-222.98, -193.72] | -783.34 [-845.85, -722.39] |

三组 PPO 相对单狗空白对照的所有主要效率与成功率 CI 均不跨 0。这组结果可以支撑“异构
Carter–Go2 调度相对只使用 Go2 的系统级收益”，但不能单独证明收益全部来自强化学习；
因此论文中必须同时保留时间规则对照。

### 4.4 同一策略是否会随负载改变运输方式

车—狗—车（`CAR_DOG_CAR`）占比：

| 策略 | NORMAL | DENSE | BURST | RECOVERY | BURST−NORMAL（95% CI） |
|---|---:|---:|---:|---:|---:|
| MEDIUM PPO | 20.71% | 6.08% | 5.17% | 6.14% | -15.54 pp [-18.01, -12.98] |
| DENSE PPO | 32.22% | 21.36% | 19.88% | 14.59% | -12.35 pp [-15.18, -9.44] |
| BURST PPO | 31.62% | 21.04% | 18.87% | 15.27% | -12.75 pp [-15.84, -9.81] |
| 时间规则 | 35.00% | 34.85% | 33.15% | 29.55% | -1.85 pp（描述统计） |

三个 PPO cohort 在 BURST 阶段均显著减少资源占用较重的车—狗—车链。这不是“以后无脑
选择单狗”，而是在同时存在三种候选时，根据队列和资源状态减少三机器人协作链占用。
规则基线的模式比例变化明显较小。

Recovery 后半段相对前半段的平均等待变化：

| 策略 | late − early（s） | 95% CI |
|---|---:|---:|
| MEDIUM PPO | -174.90 | [-201.89, -147.45] |
| DENSE PPO | -93.84 | [-101.35, -86.93] |
| BURST PPO | -104.07 | [-114.24, -94.02] |

三组策略的 CI 均完全小于 0，说明负载回落后 Recovery 后半段等待低于前半段，积压正在
清空。当前定义是任务级恢复趋势，不等同于严格控制理论中的 settling time；论文图中应画
完整 queue/waiting event curve 后再定义 recovery time。

### 4.5 三组 PPO 之间的安全—效率权衡

MEDIUM PPO 相比 DENSE/BURST：

- 成功率约高 1.1–1.2 pp，成功吞吐约高 0.76–0.82 task/h；
- 平均等待约高 41–43 s，P95 等待约高 140–156 s；
- 原始回报约低 119–124；
- 平均每任务距离约低 9–10。

这些探索性差异的两维 95% CI 均不跨 0。MEDIUM 更偏向少交接、低里程和更高成功率，
DENSE/BURST 更偏向低等待和高回报。DENSE 与 BURST 的成功率、回报、成功吞吐和平均等待
差异 CI 均跨 0；只有 DENSE 的 P95 等待比 BURST 高约 15.75 s，CI [1.13, 29.70]。

因此不能宣称 BURST 权重全面优于 DENSE，也不应只凭单个指标选择部署策略。这一结果反而
支持下一阶段训练一个面向混合负载的单一策略，并在论文中报告安全—效率 Pareto 权衡。

## 5. Stage 18 与 Stage 19 的论文证据关系

Stage 18 已完成 MEDIUM/DENSE/BURST 多训练种子完整策略与语义消融，并完成 fresh locked
test。最稳定的消融结论是 ETA/cost 上下文对性能有实质贡献；其余 position、history、
explicit queue/resource 与 explicit handover risk 的独立贡献在宽 multiplicity family 下
不够稳定，不应写成全部显著。Stage 18 的正式审查见：

`results/stage18_formal_quality_review_v4/stage18_formal_quality_review.html`

Stage 19 不替代 Stage 18，而是回答另一个问题：冻结策略能否在同一持久 episode 中面对
负载上升与回落，并实时改变运输模式。两阶段可以在论文中分别承担：

- Stage 18：主性能、上下文消融与多种子可靠性；
- Stage 19：连续非平稳负载下的模式适应与恢复。

## 6. 数据质量结论与论文声明边界

### 6.1 可用于论文的内容

- 30 个独立 PPO 训练权重和 100 个 fresh test seeds；
- 与规则、单狗对照共享任务流和 TASK_KEYED 交接随机数；
- 成功率、回报、成功吞吐、平均/P95 等待、流转时间、距离；
- 分阶段运输模式比例；
- Recovery 前后半段等待变化；
- 训练种子 × 测试种子 crossed bootstrap 95% CI；
- 非法动作、资源泄漏、失败类型和任务完整性。

### 6.2 不能宣称的内容

- 不能称为真实机器人或 Gazebo 动力学结果；当前是逻辑 SMDP 环境；
- 交接 P95 仍来自 provisional logic simulation，不是真实抓取/接触实验；
- 货物所有权切换是逻辑事务，未验证真实吸附或抓取动力学；
- SCAN-Planner 属于低层路线执行，不是 PPO 直接输出控制量；
- Stage 19 的三个 cohort 分别在三档固定负载训练，尚不是一个新训练的 mixed-curriculum 权重；
- 四阶段任务类别比例固定，只改变到达间隔，尚未覆盖任务组成和故障强度同时漂移；
- profile-to-profile 结论属于探索性，未做独立 confirmatory family 的多重比较校正。

### 6.3 当前数据异常与处理

单狗对照共有 245/8,000 个 `TERMINAL_UNRESOLVED`，原因是持久状态下出现
`NO_FEASIBLE_CHAIN`。这不是数据缺失，而是空白对照的系统失败；所有未完成任务都已补记为
失败并进入分母。PPO 不存在 terminal unresolved。PPO 与规则中的失败均来自已定义的
`HANDOVER_TIMEOUT`。

## 7. 论文图表建议

Stage 19 现可直接支持以下图表：

1. **连续负载模式适应图**：横轴四阶段，纵轴 `CAR_DOG_CAR` 占比；画 MEDIUM/DENSE/BURST
   PPO 和规则基线，显示训练种子点、均值及两维 95% CI。
2. **恢复曲线**：由逐任务 arrival/start/finish 时间重建 waiting queue 和 active task 曲线，
   在阶段切换处画竖线，并定义 queue recovery time 或 queue AUC。
3. **安全—效率 Pareto 图**：横轴平均/P95 等待，纵轴成功率；每个训练种子一个点，突出
   MEDIUM 与 DENSE/BURST 的权衡。
4. **主比较表**：PPO、规则、单狗的成功率、成功吞吐、等待、回报和距离，正文保留主要指标，
   完整数值放 supplementary material。
5. **失败组成图**：`HANDOVER_TIMEOUT` 与 `TERMINAL_UNRESOLVED`，明确单狗空白对照的失败机制。

不建议只画“回报柱状图”。原始回报受 episode 长度和多项成本共同影响，必须与成功率、等待、
吞吐和距离共同解释。

## 8. 下一阶段执行顺序

### P0：先完成 Stage 19 论文图与统计复核

1. 从逐任务明细重建 queue length、active task 和 phase timeline；
2. 明确定义 queue AUC、peak queue、recovery time；
3. 输出上述 3 张主图和 1 张正式表；
4. 对主指标预注册 Holm family，探索性指标使用 BH 或只报告 CI；
5. 更新 `TRO_实验缺口与图表路线图_副本.html`，把“恢复段缺失”和“逐任务数据缺失”标记为完成。

验收目标：图表中的每个点都能追溯到 `(profile, train_seed, test_seed, task_id)`；图表、表格与
summary JSON 的点估计一致；不使用 smoke 数据形成正式结论。

### P1：训练一个真正的 mixed-curriculum 单一策略

此处仍只训练一个策略，不按运输模式拆权重。建议训练 episode 随机组合 NORMAL、DENSE、BURST
和 RECOVERY 的顺序、持续长度与切换时刻，避免策略记住固定的“第 1 段一定 NORMAL”时间模板。
机器人位置和资源状态继续跨阶段保留。动作空间仍同时开放三种运输方案，由策略逐任务选择。

先执行 2–3 个种子的短门禁，确认：

- 在相同 episode 中模式比例随观测变化；
- 80 个任务完整记账；
- 无非法动作、货物或资源泄漏；
- 回报、成功率、等待不发生明显崩溃；
- checkpoint、断点恢复和训练进度日志正常。

短门禁通过后，再由用户手动启动 10 个独立种子的正式长训练。长训练结束后冻结权重，使用全新
测试种子与本阶段三个 fixed-profile cohort、规则基线和单狗对照比较。不得使用 Stage 19 的
`56000000–56000099` 继续调参。

### P2：补 T-RO 仍缺的外部有效性

1. 决策延迟与规模扩展：候选数量、机器人数量、队列长度对 p50/p95/p99 推理时间；
2. 鲁棒性：导航失败、楼梯占用、交接超时分布漂移、机器人临时不可用；
3. RViz 闭环案例：同层、单狗跨层、车—狗—车、一次失败恢复；
4. 若硬件可用，采集真实 Carter→Go2、Go2→Carter 交接时间戳替换 provisional P95；
5. 仿真与 RViz/硬件结果必须分开报告。

## 9. 复现入口

项目根目录：`/Users/lab4099/Desktop/Mujoco/work`

核心回归测试：

```bash
cd /Users/lab4099/Desktop/Mujoco/work
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python \
-m pytest -q src/warehouse_core/test/test_core.py
```

重新检查冻结权重：

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python \
scripts/run_stage19_freeze_gate.py
```

正式评估脚本支持断点续跑；当前结果已完成，重跑会跳过验证通过的模型文件：

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python \
scripts/run_stage19_mixed_recovery_evaluation.py --execute
```

重算数据质量与 crossed bootstrap：

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python \
scripts/summarize_stage19_mixed_recovery.py
```

## 10. 关键文件

- Stage 19 随机种子协议：
  `src/warehouse_bringup/config/experiment_seeds_stage19_mixed_recovery.yaml`
- 30 权重冻结清单：
  `results/stage19_mixed_recovery/stage19_full_policy_freeze_manifest.json`
- 正式 campaign 状态：
  `results/stage19_mixed_recovery/locked_test_v1/campaign.status.json`
- 两组基线：
  `results/stage19_mixed_recovery/locked_test_v1/baselines.json`
- 正式统计摘要：
  `results/stage19_mixed_recovery/stage19_mixed_recovery_summary.json`
- 可复核分析笔记本：
  `results/stage19_mixed_recovery/stage19_quality_audit.ipynb`
- 评估脚本：
  `scripts/run_stage19_mixed_recovery_evaluation.py`
- 统计脚本：
  `scripts/summarize_stage19_mixed_recovery.py`
- Stage 18 正式质量审查：
  `results/stage18_formal_quality_review_v4/stage18_formal_quality_review.html`

## 11. 当前阶段验收状态

Stage 19：**通过**。

通过依据：四阶段连续 episode 已实现；30 个冻结模型和两组基线完成 100 个共享测试种子；
任务级数据完整；随机任务流一致；非法动作和资源泄漏均为 0；crossed bootstrap 可重算；
三个 PPO cohort 均显示负载相关的运输模式变化和 Recovery 积压下降。

原计划先画 Stage 19 论文图；2026-08-23 用户调整优先级为“先完成实验，绘图后置”。因此当前
先执行 Stage 20 mixed-curriculum，Stage 19 的 queue AUC/recovery time 图表代码保留但暂停。

## 12. Stage 20 mixed-curriculum 实施与短门禁（2026-08-23）

Stage 20 不再把 MEDIUM、DENSE、BURST 分开训练成三个策略，而是在每个持续 episode 内随机组合
`NORMAL`、`DENSE`、`BURST`、`RECOVERY` 四个阶段，训练同一个策略根据实时任务与资源状态选择
`SINGLE_CAR`、`SINGLE_DOG` 或 `CAR_DOG_CAR`。阶段标签不输入策略；机器人位置、占用资源和队列
在阶段切换时不重置。

冻结协议：

- 每个 episode 固定 80 个任务；四段长度分别为 12–28 且总和为 80；
- 阶段顺序随机，但约束 BURST 必须先于 RECOVERY；切换空隙也随机；
- 同层/跨层任务比例按阶段分别平衡，跨层任务上行/下行平衡；
- 10 个训练种子为 `61000000–61090000`，验证种子从 `62000000` 起；
- 正式训练每种子 2000 updates、每 update 4 episodes，共 8000 episodes；
- 新鲜锁定测试种子从 `64000000` 起，不与 Stage 19 正式测试种子重叠。

三种子短门禁已通过。种子 1/2/3 各评估 320 个任务，总计 960 个任务，940 成功、20 次失败均按
既定交接超时语义记账；平均成功率 97.92%，平均 resolved throughput 65.24 tasks/h。三个种子均为
零非法动作、零货物/资源泄漏，均观察到并发执行、三种运输模式和多种随机阶段顺序。种子 1 已从
update 2 精确续跑至 update 3，检查点中的奖励归一化器及 NumPy、PyTorch、PPO RNG 状态完整。
短门禁只证明接口、随机日程、持久状态、安全性和断点恢复，不作为收敛或优于基线的证据。

短门禁汇总：

`results/stage20_mixed_curriculum/short_gate/stage20_short_gate.summary.json`

正式长训练必须由用户手动启动，入口为：

```bash
cd /Users/lab4099/Desktop/Mujoco/work
./scripts/run_stage20_mixed_curriculum_long.command
```

该入口顺序运行 10 个种子，已有合格结果自动跳过，中断后重新执行会从每个种子的最新检查点
续跑；进度状态写入：

`results/stage20_mixed_curriculum/stage20_long_batch_01_10.status.json`

若用户决定使用两个终端并行执行 `1–5` 与 `6–10`，状态文件会分别命名为
`stage20_long_batch_01_05.status.json` 与 `stage20_long_batch_06_10.status.json`，不会互相覆盖。

2026-08-24 修正：完整训练但未通过上下文模式切换门禁的种子必须作为正式失败样本保留，不能
选择性重训或删除。批处理器会将其标记为 `COMPLETED_NOT_ACCEPTED` 后继续下一个种子；只有正式
summary 缺失或不完整时才停止。最终 campaign 可为 `COMPLETED_WITH_GATE_FAILURES`，该状态属于
真实实验结果，不等同于程序故障。

每个种子的最终权重位于：

`results/stage20_mixed_curriculum/long/seed_XX/stage20_ppo.pt`

全部 10 个权重完成后必须先冻结清单，再用 `64000000–64000099` 运行一次正式锁定测试；对比对象
包括 Stage 19 三个 fixed-profile cohort、规则基线与单狗空白对照。只有该锁定测试及 crossed
bootstrap 完成后，才能判断 mixed-curriculum 是否优于固定负载训练，随后再恢复论文绘图。

Stage 20 关键文件：

- 随机种子与冻结协议：
  `src/warehouse_bringup/config/experiment_seeds_stage20_mixed_curriculum.yaml`
- episode 生成器：`src/warehouse_core/warehouse_core/stage14_training.py`
- 单种子入口：`scripts/run_stage20_mixed_curriculum.py`
- 10 种子手动长训练入口：`scripts/run_stage20_mixed_curriculum_long.command`
- 批处理与续跑器：`scripts/run_stage20_mixed_curriculum_long_batch.py`
- 短门禁汇总器：`scripts/summarize_stage20_short_gate.py`

## 13. Stage 20 正式冻结与锁定测试（2026-08-24）

10 个预注册 Stage 20 权重已全部冻结，未选择性删除种子 4、7、9。冻结清单对 summary、history、
最终/最佳 checkpoint 和 warm-start 权重保存 SHA-256，并核查模型元数据、奖励归一化器、NumPy/
PyTorch/PPO RNG 状态及训练/验证/测试/bootstrap 种子互斥。冻结门禁通过：7 个训练门禁通过权重
与 3 个门禁失败权重全部保留。

冻结清单：

`results/stage20_mixed_curriculum/stage20_policy_freeze_manifest.json`

正式锁定测试只运行一次，使用全新 `64000000–64000099`：

- Stage 20 mixed-curriculum：10 个权重；
- Stage 19 MEDIUM/DENSE/BURST：各 10 个权重，共 30 个；
- 基线：time-greedy rule 与 single-dog only；
- 每个权重 100 个 episode、每 episode 80 个任务；
- 40 个策略共 320,000 个任务，所有方法使用同一个 task fingerprint；
- 所有策略和基线非法动作、资源泄漏均为 0；
- 10 个 Stage 20 种子全部纳入统计，未按锁定测试结果筛选。

正式结果目录：

`results/stage20_mixed_curriculum/locked_test_v1/`

质量审计与 10,000 次 crossed bootstrap 摘要：

`results/stage20_mixed_curriculum/stage20_locked_test_summary.json`

统计口径为训练种子 × 共享测试种子的 crossed bootstrap。episode-level P95 的平均与 pooled-task
P95 分开命名和报告；episode throughput 平均与总成功数/总时间也分开，禁止混用 estimand。

主要结果（Stage 20 mixed-curriculum 为 A）：

- 相对 time-greedy rule：成功率差 `+0.039` 个百分点，95% CI
  `[-0.239, +0.320]`，没有明确差异；回报提高 `+89.46 [69.98, 108.63]`；平均等待减少
  `30.14 s [24.28, 36.09]`；mean-episode throughput 差异不明确。
- 相对 single-dog only：成功率差 `-0.374` 个百分点，95% CI
  `[-1.206, +0.589]`，没有明确差异；回报提高 `+548.23 [497.91, 601.45]`；平均等待减少
  `178.88 s [160.88, 198.10]`，但每任务距离增加 `25.72 [23.76, 27.78]`。
- 相对 Stage 19 MEDIUM：回报提高 `+61.01 [28.43, 92.87]`、平均等待减少
  `22.94 s [12.60, 33.05]`，但成功率降低 `2.494` 个百分点、throughput 降低、距离增加。
- 相对 Stage 19 DENSE/BURST：Stage 20 在成功率、回报、平均等待、throughput 和距离上均显著较差。

因此 Stage 20 支持“相对规则和单狗对照显著降低等待并改善综合回报”，但不支持“随机混合负载训练
全面优于固定高负载专用策略”。这一负面结果必须保留。模式结果给出一个可能机制：Stage 20 总体
选择 `SINGLE_CAR 48.52% / SINGLE_DOG 18.71% / CAR_DOG_CAR 32.78%`，且四阶段之间变化较小；
Stage 19 DENSE/BURST 的车–狗–车比例更低且 BURST→RECOVERY 的切换幅度更明显。

`64000000–64000099` 已经解锁并使用，禁止再用于调参、重新选择权重或验证 Stage 21。若继续改进
mixed-curriculum，必须建立全新的训练/验证/锁定测试种子族，并把本次结果作为探索性设计依据。
下一实验优先补外部有效性（规模扩展、推理延迟、导航失败/楼梯占用/机器人不可用鲁棒性）；若另开
Stage 21，应显式提高策略对负载历史和资源拥塞变化的辨识能力，而不是直接硬编码阶段标签。
