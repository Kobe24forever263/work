# Stage 14 SMDP-PPO留出集评估

- 权重：`/Users/lab4099/Desktop/Mujoco/work/results/stage14/ppo/long/realistic_concurrent_medium_reward_v2_context_v2/stage14_realistic_concurrent_medium_reward_v2_context_v2.best.pt`
- 执行模式：`CONCURRENT`
- 到达负载：`MEDIUM`
- 留出episode：30

| 指标 | 学习策略 | 规则基线 |
|---|---:|---:|
| 成功率 | 98.33% | 97.33% |
| 原始episode平均回报 | 18.9767 | 18.7125 |
| 吞吐(任务/小时) | 55.812 | 55.913 |
| 平均仿真时间(s) | 1290.039 | 1287.709 |
| 非法动作 | 0 | 0 |
| 资源泄漏 | 0 | 0 |

## 按环境选择方案

以下数据按当前任务与车队条件下的最低代价参考模式分组，检查同一策略是否随环境切换；它不是模式配额。

- 条件选择矩阵：`{"CAR_DOG_CAR": {"CAR_DOG_CAR": 136, "SINGLE_DOG": 74}, "SINGLE_CAR": {"SINGLE_CAR": 300}, "SINGLE_DOG": {"SINGLE_DOG": 90}}`
- 各参考模式对角一致率：`{"CAR_DOG_CAR": 0.6476190476190476, "SINGLE_CAR": 1.0, "SINGLE_DOG": 1.0}`

该报告比较的是逻辑仿真时间，不等同于RViz墙钟时间。
