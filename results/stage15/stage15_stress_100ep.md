# Stage 15预注册配对压力评估

- 权重：`/Users/lab4099/Desktop/Mujoco/work/results/stage14/ppo/long/realistic_concurrent_medium_reward_v2_context_v2/stage14_realistic_concurrent_medium_reward_v2_context_v2.best.pt`
- 每场景episode：100
- 固定种子：20266001～20266100
- 所有差值均为PPO减规则基线；场景和种子在运行前固定。

| 场景 | PPO成功率 | 规则成功率 | 回报差 | 吞吐差 | 平均等待差(s) |
|---|---:|---:|---:|---:|---:|
| MEDIUM | 97.95% | 96.70% | 0.726 | -0.128 | 0.441 |
| DENSE | 97.90% | 96.70% | 14.934 | -0.888 | -17.456 |
| CROSS_HEAVY | 96.60% | 94.20% | 0.549 | -0.319 | 1.942 |
| BURST | 97.90% | 96.70% | 27.867 | -2.123 | -32.764 |

PASS仅代表任务流配对、安全和结果完整，不自动宣称PPO优于规则。
是否存在显著优势必须结合JSON中的配对bootstrap 95%置信区间判断。
