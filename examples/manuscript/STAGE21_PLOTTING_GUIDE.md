# Stage 21 论文绘图说明

更新时间：2026-08-25

本说明用于从 GitHub 中保存的冻结数据生成 IEEE Transactions on Robotics 论文图表。此阶段只允许
读取和排版，禁止重新训练、替换权重、修改测试种子、重新定义指标或覆盖正式结果。

## 1. 数据入口

数据目录：`examples/manuscript/data/stage21/`

| 文件 | 用途 |
|---|---|
| `latency_scaling_formal.json` | 54 个规模组合、5 个计时组件及每组件 500 个原始延迟样本 |
| `latency_summary.csv` | 适合 Origin/MATLAB 的 54 行延迟汇总 |
| `fault_robustness_locked_summary.json` | 故障正式实验的绝对指标、差值、95% CI 和 CONTROL 退化 |
| `fault_absolute.csv` | 4 场景 × 3 方法的绝对均值、标准差和样本数 |
| `fault_comparisons.csv` | PPO 相对两个基线的差值和 95% CI |
| `fault_degradation_from_control.csv` | 三类故障相对 CONTROL 的方法内退化和 95% CI |
| `stage21_fault_protocol_freeze_manifest.json` | 权重、配置、脚本和种子冻结证明 |
| `campaign_*.status.json` | 40 个 PPO 与 8 个基线作业的完成状态 |

CSV 是从冻结 JSON 直接展开的格式副本，没有重新估计统计量。论文引用时以 JSON 为权威来源，CSV
仅用于绘图软件导入。

## 2. 固定顺序和符号

场景顺序固定为：

1. `CONTROL`
2. `NAVIGATION_FAILURE`
3. `STAIR_OUTAGE`
4. `ROBOT_OUTAGE`

方法顺序固定为：

1. `PPO`
2. `TIME_GREEDY_RULE`
3. `SINGLE_DOG_ONLY`

图中建议显示名称：`Ours (SMDP-PPO)`、`Time-greedy`、`Single-dog`。正文第一次出现时给出全称，
后续可以使用简写。

符号方向：

- `PPO_MINUS_*`：正值表示 PPO 数值更大，负值表示 PPO 数值更小；
- 对成功率、回报、成功吞吐，正值通常更好；
- 对等待时间、flow time、恢复时间和距离，负值通常更好；
- `difference_from_control = fault - control`，成功率为负表示故障退化，等待为正表示故障退化。

## 3. 推荐图表

### Figure A：调度计算延迟与规模扩展

分析问题：在训练形状边界内，候选数量增加是否使 CPU 调度时间超过工程预算？

数据：`latency_summary.csv`。

建议采用双面板：

- Panel (a)：筛选 `stair_count = 4`，横轴为 `waiting_task_count`，纵轴为
  `end_to_end_p99_ms`，按 `team_size = 4/7/10` 绘制三条折线；横轴显示
  `1, 2, 4, 8, 16, 32`，可使用 log2 等距坐标但必须保留真实刻度标签；
- Panel (b)：五个计时组件在全部 54 个场景中的最大 p99，使用水平条形图。最大值应为：候选生成
  `0.980 ms`、合法性掩码 `1.091 ms`、完整编码 `3.679 ms`、策略前向 `1.359 ms`、端到端决策
  `4.815 ms`。

不要把 100 ms 预算和 0–5 ms 数据放在同一线性纵轴，否则主体会被压扁。应在副标题或角标写
`Worst p99 = 4.815 ms < 100 ms budget`。Panel (a) 在等待任务超过 8 后趋于饱和，是因为 top-8
可见任务边界，不代表超过 8 个任务仍具有无损全局感知能力。

推荐标题：`CPU scheduling latency within the trained shape boundary`。

### Figure B：故障场景中的三方法绝对表现

分析问题：在相同任务流与故障计划下，三种方法的绝对成功率和等待时间如何？

数据：`fault_absolute.csv`。

建议采用两个分面，每个场景严格画三根柱子：

- Panel (a)：筛选 `metric = success_rate`，纵轴转换为百分数；
- Panel (b)：筛选 `metric = mean_waiting_time`，单位为秒；
- 横轴为四个场景，组内顺序为 PPO、time-greedy、single-dog；
- 在柱顶保留 1 位小数的百分数或整数秒标签。

这张图是描述性绝对均值图。`std` 的统计粒度在 PPO 与基线之间不同，禁止直接把 `std` 当作 95%
CI 画在柱子上。显著性判断必须来自 Figure C 的正式 bootstrap 区间。

推荐标题：`Absolute task outcomes under logical fault injection`。

### Figure C：PPO 相对基线的正式效应量

分析问题：PPO 的等待时间改善是否跨训练种子和锁定测试种子保持稳定？

数据：`fault_comparisons.csv`。

这是主论文优先级最高的故障图。建议使用 dot-and-whisker/forest plot，而不是继续堆叠柱子：

- 选择 `metric = mean_waiting_time` 和 `p95_waiting_time`，可做上下两个面板；
- 横轴为 `difference`，误差线端点为 `ci95_low`、`ci95_high`；
- 纵轴为四个场景；
- `PPO_MINUS_TIME_GREEDY_RULE` 与 `PPO_MINUS_SINGLE_DOG_ONLY` 使用两种点形和线型；
- 在 `x = 0` 处画深灰虚线；区间完全位于 0 左侧表示 PPO 显著降低等待时间；
- 不使用星号替代置信区间。

若使用 Origin，需要新增两列非对称误差：

- `XErr- = difference - ci95_low`
- `XErr+ = ci95_high - difference`

推荐标题：`Paired waiting-time effects of SMDP-PPO versus baselines`。

### Figure D：故障相对正常场景的性能退化

分析问题：每类故障给各方法带来多大退化，哪些扰动最难？

数据：`fault_degradation_from_control.csv`。

建议采用两个 forest-plot 面板：

- Panel (a)：`metric = success_rate`，将差值乘 100，单位为 percentage points；
- Panel (b)：`metric = mean_waiting_time`，单位为秒；
- 场景只包含三类故障，不包含 CONTROL；
- 三种方法可以在同一场景行上轻微纵向错开，仍需用点形/线型区分，不能只靠颜色；
- 0 参考线必须保留。

该图支持“导航失败是当前最强扰动”。它不支持“硬件故障鲁棒性”，因为实验只在逻辑 SMDP 层注入
任务级导航失败和可观察的资源不可用窗口。

推荐标题：`Performance degradation relative to the no-fault control`。

## 4. 推荐表格

主文可放一张紧凑表，行是四种场景，列为 PPO 的成功率、成功吞吐、平均等待和 P95 等待；规则基线
差值与 95% CI 可放在后四列或补充材料。精确数字以
`fault_robustness_locked_summary.json.absolute` 和 `.comparisons` 为准。

必须在表注写明：10 个训练种子、100 个共享锁定测试种子、PPO 使用 crossed bootstrap、基线使用
paired bootstrap、10,000 次重采样。

## 5. Origin 操作建议

1. 使用 `Data > Connect to File > CSV` 导入对应 CSV，不要手工复制终端输出。
2. 将 `scenario` 与 `method/comparison` 设为 categorical，并按本说明固定顺序排序。
3. Figure B 选择 `Grouped Columns`；每个场景必须保留三根柱子。
4. Figure C/D 选择 `Scatter with X Error Bars`，用上文公式生成左右非对称误差列。
5. 百分数统一在导入后乘 100；原始 JSON 中成功率范围为 0–1。
6. 保存 Origin 工程时记录数据文件的相对路径，不要链接本机 `results/` 的绝对路径。
7. 输出矢量 PDF，同时输出 600 dpi PNG 仅供预览；最终 LaTeX 优先使用 PDF。

## 6. 视觉规范

- IEEE 双栏宽图：约 7.16 in；单栏宽图：约 3.5 in；
- 最终排版后的坐标、图例和注释建议 8–9 pt，不低于 7.5 pt；
- 背景为白色，网格线使用浅灰，轴线和文字使用深灰；
- 建议配色：PPO `#4472C4`，time-greedy `#ED7D31`，single-dog `#A5A5A5`；
- 同时使用实心/空心、圆/方/三角或不同 hatch，确保灰度打印仍可辨认；
- 禁止红绿二元配色、渐变背景、3D 柱子和截断的绝对柱状图纵轴；
- 同类面板保持相同刻度与场景顺序；
- 导出 PDF 后检查字体嵌入、负号、希腊字母、误差线端帽和图例是否被裁切。

## 7. 可写入论文的结论边界

正式数据支持：

- PPO 与 time-greedy 的成功率和成功吞吐相当；
- PPO 在四个场景中显著降低平均/P95 等待及 flow time；
- 楼梯和机器人临时不可用时，成功率与吞吐没有显著下降，但等待小幅增加；
- 任务级导航失败使 PPO 成功率下降约 8.04 percentage points；
- single-dog 距离较短，但等待和 flow time 明显更差。

正式数据不支持：

- PPO 的成功率或吞吐显著优于 time-greedy；
- PPO 比 time-greedy 显著更快恢复；
- 结果可直接外推到 ROS、SCAN-Planner、动力学、感知或实体硬件故障；
- top-8 之外的等待任务被策略无损感知。

## 8. 出图前验收清单

- 图中数值可逐项回查到冻结 JSON；
- 场景、方法和指标方向没有颠倒；
- 置信区间来自 `ci95_low/ci95_high`，没有用标准差替代；
- 三方法绝对图每个场景确实有三根柱子；
- forest plot 保留 0 参考线并使用非对称误差；
- 图题为描述性标题，论文正文再解释结论；
- caption 包含样本数、bootstrap 方式、指标单位和逻辑故障边界；
- PDF 字体嵌入、线条和文字在最终单双栏尺寸下清晰。
