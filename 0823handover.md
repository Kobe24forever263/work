# 0823 Handover：T-RO 论文实验进展、Stage 19 结果与下一阶段

更新日期：2026-08-23  
目标期刊：IEEE Transactions on Robotics（T-RO）  
当前主线：异构 Carter–Go2 双层仓库中的事件驱动 SMDP 调度、动态运输模式选择与连续负载适应

## 1. 当前结论

Stage 18 固定负载、Stage 19 连续负载压力测试与 Stage 20 mixed-curriculum 均已完成正式质量审查。
当前论文的主结论应表述为：
**本文提出的事件驱动 masked SMDP-PPO 调度框架在固定负载与随机混合负载下均表现良好**，而不是
要求某一个 mixed-curriculum 权重在所有指标上全面超过每个负载专用权重。

固定负载的直接证据来自 Stage 18 matched-load locked test：MEDIUM、DENSE、BURST 三档完整 PPO
策略的成功率分别为 `98.08%`、`96.30%`、`96.44%`，总体范围为 `96.30–98.08%`；三档回报与
有效吞吐的点估计均高于对应 time-greedy rule。这里不把未经相应置信区间支持的点估计写成
“统计显著优于”。

连续负载适应证据来自 Stage 19。该实验在同一个不重置的 episode 中依次施加
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

因此 Stage 19 的数据可以进入正式论文分析。三个 fixed-profile cohort 相对 time-greedy rule 的成功率
提高 `0.90–2.07` 个百分点、有效吞吐提高 `0.64–1.46 tasks/h`。核心发现不是“策略固定选择某一种运输方式”，
而是同一策略在持续状态下根据队列、机器人位置、候选 ETA、资源占用和交接风险改变
`SINGLE_CAR`、`SINGLE_DOG`、`CAR_DOG_CAR` 的选择比例。

混合负载证据来自 Stage 20：mixed-curriculum cohort 在随机持续混合负载下保持 `96.51%` 成功率，
与规则基线的成功率无明确差异，同时相对规则基线回报提高 `89.46`、平均等待减少 `30.14 s`；
相对单狗空白对照平均等待减少 `178.88 s`。这说明同一方法框架能够覆盖固定负载与混合负载两类
运行条件；负载专用策略在匹配条件下仍可能具有更高效率，应作为适用边界而非否定主结论。

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

综合 Stage 18、Stage 19 与 Stage 20，正式论文可以主张：**本文提出的策略框架在固定负载和随机混合负载下
均取得稳定且有竞争力的表现**。其中，固定负载 matched-load test 中三档 PPO 均保持超过 `96%` 的
成功率，且回报和有效吞吐点估计均高于规则基线；连续负载测试进一步显示成功率与有效吞吐的显著
改善；mixed-curriculum
cohort 保持 `96.51%` 成功率，并相对规则和单狗对照显著降低等待、改善综合回报。该结论不等价于
“一个混合权重全面优于所有固定高负载专用策略”：Stage 20 相对 Stage 19 DENSE/BURST 在部分效率
指标上仍较弱，这一边界必须保留。模式结果给出一个可能机制：Stage 20 总体
选择 `SINGLE_CAR 48.52% / SINGLE_DOG 18.71% / CAR_DOG_CAR 32.78%`，且四阶段之间变化较小；
Stage 19 DENSE/BURST 的车–狗–车比例更低且 BURST→RECOVERY 的切换幅度更明显。

`64000000–64000099` 已经解锁并使用，禁止再用于调参、重新选择权重或验证 Stage 21。若继续改进
mixed-curriculum，必须建立全新的训练/验证/锁定测试种子族，并把本次结果作为探索性设计依据。
下一实验优先补外部有效性（规模扩展、推理延迟、导航失败/楼梯占用/机器人不可用鲁棒性）；若另开
Stage 21，应显式提高策略对负载历史和资源拥塞变化的辨识能力，而不是直接硬编码阶段标签。

## 14. Stage 21 外部有效性：计算延迟与规模扩展（2026-08-25）

Stage 21 首先补不需要重新训练的计算开销证据。基准使用 Stage 20 已冻结的 seed 1 权重及其
SHA-256，在 CPU 单线程上分别计时：候选生成、合法性掩码、完整固定形状编码、策略前向和端到端
调度决策。环境创建、任务执行、ROS 通信、导航和机器人控制均不进入计时区间。

规模矩阵为：

- 机器人数量：`4 / 7 / 10`；
- 可用楼梯：`1 / 2 / 4`；
- 等待任务：`1 / 2 / 4 / 8 / 16 / 32`；
- 共 `54` 个组合；
- 模型边界：最多 `10` 个机器人、`4` 部楼梯、top-8 可见任务和 `1536` 个候选槽。

短门禁命令：

```bash
cd /Users/lab4099/Desktop/Mujoco/work
./scripts/run_stage21_latency_scaling.command --smoke
```

短门禁已通过。54 个组合全部存在合法动作，最大候选数 `576/1536`，全部分位数有限且有序；
端到端调度决策最坏 p99 为 `4.812 ms`，低于预注册的 `100 ms` 工程预算。等待任务超过 top-8 后
候选数量按设计保持不变，说明计算量受策略可见窗口限制；这不是超过 8 个任务的无损全局扩展能力。

短门禁结果：

`results/stage21_external_validity/latency_scaling_smoke.json`

正式计时命令：

```bash
cd /Users/lab4099/Desktop/Mujoco/work
./scripts/run_stage21_latency_scaling.command --warmup 20 --repeats 500
```

正式输出：

`results/stage21_external_validity/latency_scaling_formal.json`

正式验收标准：54 个组合完整；冻结权重哈希一致；每个组件每场景保存 500 个原始样本；p50、p95、
p99 有序且有限；候选数不超过 1536；所有场景至少一个合法动作；端到端 p99 不超过 100 ms。
正式结果只能支持“训练形状边界内的 CPU 调度计算开销”，不能外推到更大团队、ROS 端到端时延或
机器人控制周期。

2026-08-25 正式计时已完成并通过。共保存 `54 × 5 × 500 = 135,000` 个原始延迟样本。最坏组件
p99 分别为：候选生成 `0.980 ms`、合法性掩码 `1.091 ms`、完整编码 `3.679 ms`、策略前向
`1.359 ms`、端到端调度决策 `4.815 ms`。最坏端到端场景为 `R10_S4_Q16`，共有 10 个机器人、
4 部楼梯、16 个等待任务、8 个可见任务和 576 个合法候选，其 p50/p95/p99 分别为
`4.555 / 4.741 / 4.815 ms`。

在完整 10 机器人、4 楼梯条件下，等待任务从 1 增至 8 时，候选数从 72 增至 576，端到端 p99
从 `1.745 ms` 增至 `4.781 ms`；等待任务继续增至 16/32 后，候选数保持 576，p99 分别为
`4.815/4.771 ms`。因此当前主要计算瓶颈是候选特征构造与编码，而不是神经网络前向。该结果来自
一台 Mac M4 的单次 CPU 协议，论文中应报告硬件和线程设置，不能解释为跨硬件普遍时延。

完成正式计时后，Stage 21 第二子实验为故障鲁棒性：在共享任务流下分别注入导航失败、楼梯临时
占用和机器人不可用，配对比较 mixed-curriculum PPO、time-greedy rule 与 single-dog control 的
成功率、等待、恢复时间、失败类型和资源泄漏。该实验必须使用新的 Stage 21 测试种子族。

## 15. Stage 21 外部有效性：故障鲁棒性（2026-08-25）

已新增逻辑故障注入层，并保持 Stage 14/20 的默认环境行为不变。既有 Stage 14 十 episode 接口回归
门禁重新运行并通过。三类故障语义如下：

- `NAVIGATION_FAILURE`：在预注册绝对时间窗口内，使用 task-keyed 共同随机数按运输链路暴露故障；
  较长的运输链具有更多独立暴露段，失败任务增加 15 s 逻辑恢复时间；
- `STAIR_OUTAGE`：两部确定性抽样楼梯在共享绝对时间窗口内暂时不可用于新任务；已开始任务不抢占；
- `ROBOT_OUTAGE`：一辆车和一只 Go2 分别在共享窗口内暂时不可用于新任务；已开始任务执行完毕后，
  若窗口仍有效则继续保持不可用；
- `CONTROL`：相同任务流但不注入故障，用于计算每种方法自身的性能退化。

短门禁命令：

```bash
cd /Users/lab4099/Desktop/Mujoco/work
./scripts/run_stage21_fault_robustness.command --smoke
```

短门禁使用 `66000000–66000001` 两个测试种子，每个场景、每种方法共 160 个任务。PPO、
time-greedy rule 和 single-dog only 在每个场景中共享任务 fingerprint、故障计划 fingerprint 和
task-keyed 导航故障随机数。24 项完整性断言全部通过：所有任务有终态、非法动作 0、资源泄漏 0，
导航失败真实触发，楼梯与机器人故障窗口均被环境观察。

短门禁点估计只用于检查故障是否激活，不能形成优越性结论。CONTROL 下 PPO/规则/单狗成功率分别
为 `97.50% / 97.50% / 94.38%`；NAVIGATION_FAILURE 下分别为
`92.50% / 93.13% / 88.75%`。楼梯和机器人故障在两个 episode 中主要改变等待时间，没有改变三种
方法各自的成功率；机器人故障下 PPO 等待偶然下降，属于小样本波动，禁止解释为故障带来收益。

短门禁结果：

- 汇总：`results/stage21_external_validity/fault_robustness_smoke.json`
- episode/task/fault 原始行：
  `results/stage21_external_validity/fault_robustness_smoke.raw.json`

关键实现：

- 故障环境：`src/warehouse_core/warehouse_core/stage21_faults.py`
- 三方评估：`scripts/run_stage21_fault_robustness.py`
- 一键入口：`scripts/run_stage21_fault_robustness.command`

当前非 smoke 入口只评估一个 Stage 20 权重，属于协议试跑，不足以形成论文结果。正式实验必须另建
all-weight campaign：冻结 Stage 20 的 10 个权重，使用预留的 `66100000–66100099` 作为 100 个
共享测试种子；规则和单狗基线各运行一次；每个故障场景对 10 个训练种子 × 100 个测试种子做
crossed bootstrap。正式验收需满足：任务与故障 fingerprint 全部一致；所有权重和基线任务完整；
非法动作、资源泄漏均为 0；每类故障真实激活；报告相对 CONTROL 的成功率、回报、成功吞吐、
平均/P95 等待、平均/P95 flow time 和恢复时间退化，并单独报告 PPO 相对两组基线的差值与 95% CI。

当前结果仍是逻辑 SMDP 故障抽象，不是 SCAN-Planner、ROS 或实体机器人故障实验。论文可用它说明
调度层在可观察资源失效和任务级导航失败下的鲁棒性，但必须把闭环导航故障验证列为外部有效性边界。

### 15.1 正式冻结协议与运行入口

正式协议已冻结并通过门禁。冻结对象包括 Stage 20 的 10 个权重、故障配置、锁定测试种子清单、
bootstrap 方案和实验脚本哈希。测试使用 `66100000–66100099` 共 100 个新种子，bootstrap 固定为
`66200000`，每项统计执行 10,000 次重采样。PPO 使用 10 个训练种子与 100 个测试种子的 crossed
bootstrap；time-greedy rule 和 single-dog only 使用配对测试种子 bootstrap。

正式一键入口：

```bash
cd /Users/lab4099/Desktop/Mujoco/work
./scripts/run_stage21_fault_locked.command
```

关键输出：

- 冻结清单：`results/stage21_external_validity/stage21_fault_protocol_freeze_manifest.json`
- 正式原始结果：`results/stage21_external_validity/fault_robustness_campaign/locked_v1/`
- 正式汇总：`results/stage21_external_validity/fault_robustness_locked_summary.json`

正式 campaign 已于 2026-08-25 完成并通过。共生成 48 个方法/场景结果文件，覆盖 40 个 PPO 作业
和 8 个基线作业，合计 384,000 个任务；其中 363,719 个完成、20,281 个失败。非法动作与资源泄漏
均为 0。所有方法和场景共享同一个任务流 fingerprint；每个场景内部共享同一个故障 fingerprint。
40 个 PPO、8 个基线、导航失败激活、楼梯/机器人故障窗口可见性等 7 项正式断言全部通过。

### 15.2 正式结果与可支持结论

PPO 的绝对表现为：

| 场景 | 成功率 | 成功吞吐（task/h） | 平均等待（s） | P95 等待（s） | 平均 flow time（s） | 恢复时间（s） |
|---|---:|---:|---:|---:|---:|---:|
| CONTROL | 96.74% | 68.60 | 105.28 | 446.39 | 149.51 | — |
| NAVIGATION_FAILURE | 88.70% | 62.88 | 111.65 | 467.53 | 157.04 | 35.93 |
| STAIR_OUTAGE | 96.72% | 68.58 | 107.69 | 455.09 | 152.28 | 32.69 |
| ROBOT_OUTAGE | 96.79% | 68.62 | 108.97 | 452.84 | 153.37 | 33.46 |

相对 time-greedy rule，PPO 在四个场景中的成功率与成功吞吐差异 95% CI 均跨 0，不能声称有显著
提升；但平均等待分别降低 `32.42 / 34.43 / 32.16 / 29.76 s`，P95 等待分别降低
`134.41 / 138.51 / 129.76 / 122.65 s`，平均与 P95 flow time 也全部显著降低，所有这些时间指标的
95% CI 均不跨 0。PPO 回报相对规则基线提高约 `87.76–101.02`，95% CI 均不跨 0。两者恢复时间
差异的 95% CI 均跨 0，因此不能声称 PPO 比规则基线恢复更快。

相对 single-dog only，PPO 在四个场景中的平均等待降低 `190.57–213.95 s`，P95 等待降低
`695.78–794.77 s`，平均/P95 flow time 和回报也均显著更优。成功率差异 95% CI 仍均跨 0；
STAIR_OUTAGE 与 ROBOT_OUTAGE 的成功吞吐显著提高，CONTROL 与 NAVIGATION_FAILURE 的吞吐差异
尚不显著。PPO 的每任务行驶距离更长约 `25.67–26.46 m`，这是多智能体协作换取低等待和低 flow
time 的代价，不能描述为距离效率提高。

相对 CONTROL，NAVIGATION_FAILURE 使 PPO 成功率显著下降 `8.04` 个百分点
（95% CI `[-8.78, -7.35]` pp），成功吞吐下降 `5.72 task/h`，平均等待增加 `6.36 s`；说明任务级
导航失败是当前最强的外部扰动。STAIR_OUTAGE 和 ROBOT_OUTAGE 的成功率差异 CI 均跨 0，成功吞吐
也没有显著变化，但平均等待分别显著增加 `2.41 s` 和 `3.69 s`。因此正式证据支持：在可观察的
楼梯或机器人临时不可用时，调度器能保持成功率与吞吐，并以小幅等待增加完成绕行/换机器人；面对
任务级导航失败时性能会可量化下降，但仍保持约 88.70% 成功率。

论文表述必须保持边界：这是逻辑 SMDP 层的任务级导航失败与资源可用性窗口，不包含 ROS 消息丢失、
SCAN-Planner 路径失败、动力学失稳、感知误差或实体硬件故障。正式数据可支撑“调度层故障鲁棒性”
和“相对规则基线显著降低等待/flow time”，不能支撑“端到端系统或硬件鲁棒性”，也不能声称成功率
或吞吐显著优于 time-greedy rule。
