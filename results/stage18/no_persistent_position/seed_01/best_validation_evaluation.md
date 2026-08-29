# Stage 14 SMDP-PPO留出集评估

- 权重：`/Users/lab4099/Desktop/Mujoco/work/results/stage18/no_persistent_position/seed_01/stage18_ppo.best.pt`
- 执行模式：`CONCURRENT`
- 到达负载：`MEDIUM`
- 留出episode：100

| 指标 | 学习策略 | 规则基线 |
|---|---:|---:|
| 成功率 | 95.15% | 96.45% |
| 原始episode平均回报 | 9.2567 | 17.9740 |
| 吞吐(任务/小时) | 55.707 | 56.166 |
| 平均仿真时间(s) | 1292.467 | 1281.913 |
| 非法动作 | 0 | 0 |
| 资源泄漏 | 0 | 0 |

## 按环境选择方案

以下数据按当前任务与车队条件下的最低代价参考模式分组，检查同一策略是否随环境切换；它不是模式配额。

- 条件选择矩阵：`{"CAR_DOG_CAR": {"CAR_DOG_CAR": 710}, "SINGLE_CAR": {"SINGLE_CAR": 1000}, "SINGLE_DOG": {"CAR_DOG_CAR": 290}}`
- 各参考模式对角一致率：`{"CAR_DOG_CAR": 1.0, "SINGLE_CAR": 1.0, "SINGLE_DOG": 0.0}`

该报告比较的是逻辑仿真时间，不等同于RViz墙钟时间。
