# Stage 14 串行/并发任务模式规则基线对比

SERIAL保留单主任务；CONCURRENT允许最多4个无冲突任务在途。当前是规则基线，不是RL结果。

## MEDIUM

| 指标 | SERIAL | CONCURRENT |
|---|---:|---:|
| 成功率 | 96.90% | 96.90% |
| 吞吐(任务/小时) | 55.731 | 56.003 |
| 平均等待(s) | 5.261 | 1.951 |
| P95等待(s) | 25.717 | 14.758 |
| 平均流转(s) | 44.795 | 41.618 |
| P95流转(s) | 86.329 | 82.431 |
| 峰值在途任务 | 0 | 2 |

并发吞吐提升0.49%，平均等待降低62.92%。

## DENSE

| 指标 | SERIAL | CONCURRENT |
|---|---:|---:|
| 成功率 | 96.90% | 96.90% |
| 吞吐(任务/小时) | 84.424 | 122.133 |
| 平均等待(s) | 308.698 | 76.145 |
| P95等待(s) | 669.818 | 321.060 |
| 平均流转(s) | 348.138 | 116.164 |
| P95流转(s) | 683.629 | 341.482 |
| 峰值在途任务 | 0 | 4 |

并发吞吐提升44.67%，平均等待降低75.33%。

## 强化学习计划

分别训练SERIAL-only、CONCURRENT-only，并用MIXED按episode交替两种模式。观测显式包含模式位；三组使用相同训练/验证/测试种子，分别报告成功率、吞吐、等待、流转时间、里程、交接超时及资源冲突。

```json
{
  "400_episodes_resolve_8000_tasks": true,
  "paired_task_streams_equal": true,
  "fixed_seed_replay_equal": true,
  "only_controlled_handover_failures": true,
  "concurrent_reaches_overlap": true,
  "concurrent_throughput_not_worse": true,
  "concurrent_waiting_not_worse": true,
  "success_delta_within_2pct": true,
  "no_state_or_resource_leaks": true
}
```
