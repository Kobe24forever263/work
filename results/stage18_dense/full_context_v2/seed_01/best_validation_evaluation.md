# Stage 14 SMDP-PPO留出集评估

- 权重：`/Users/lab4099/Desktop/Mujoco/work/results/stage18_dense/full_context_v2/seed_01/stage18_ppo.best.pt`
- 执行模式：`CONCURRENT`
- 到达负载：`DENSE`
- 留出episode：100

| 指标 | 学习策略 | 规则基线 |
|---|---:|---:|
| 成功率 | 96.80% | 96.45% |
| 原始episode平均回报 | -13.5527 | -33.2558 |
| 吞吐(任务/小时) | 124.428 | 123.554 |
| 平均仿真时间(s) | 578.650 | 582.740 |
| 非法动作 | 0 | 0 |
| 资源泄漏 | 0 | 0 |

## 按环境选择方案

以下数据按当前任务与车队条件下的最低代价参考模式分组，检查同一策略是否随环境切换；它不是模式配额。

- 条件选择矩阵：`{"CAR_DOG_CAR": {"CAR_DOG_CAR": 641, "SINGLE_DOG": 69}, "SINGLE_CAR": {"SINGLE_CAR": 1000}, "SINGLE_DOG": {"SINGLE_DOG": 290}}`
- 各参考模式对角一致率：`{"CAR_DOG_CAR": 0.9028169014084507, "SINGLE_CAR": 1.0, "SINGLE_DOG": 1.0}`

该报告比较的是逻辑仿真时间，不等同于RViz墙钟时间。
