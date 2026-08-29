# Stage 14 SMDP-PPO留出集评估

- 权重：`/Users/lab4099/Desktop/Mujoco/work/results/stage14/ppo/smoke/stage14_smdp_ppo_mixed_medium.best.pt`
- 执行模式：`MIXED`
- 到达负载：`MEDIUM`
- 留出episode：4

| 指标 | 学习策略 | 规则基线 |
|---|---:|---:|
| 成功率 | 97.50% | 98.75% |
| 平均回报 | 24.2626 | 37.0667 |
| 吞吐(任务/小时) | 55.431 | 56.195 |
| 平均仿真时间(s) | 1298.906 | 1281.261 |
| 非法动作 | 0 | 0 |
| 资源泄漏 | 0 | 0 |

该报告比较的是逻辑仿真时间，不等同于RViz墙钟时间。
