# 阶段 11 三种任务发布负载对比报告

日期：2026-08-05  
随机种子：`20260805`

## 结论

`normal`、`medium` 和 `overload` 三种负载均完成20个持续状态任务，结果均为
`20/20 PASS`。三档使用相同任务内容，只改变任务到达时间；每档均完成同种子
重放检查。

| 负载 | 到达间隔 | 平均任务时长 | 平均等待 | P95等待 | 最大等待 | 队列峰值 |
|---|---:|---:|---:|---:|---:|---:|
| normal | 50–70 s | 33.59 s | 1.44 s | 3.35 s | 16.18 s | 1 |
| medium | 30–50 s | 33.59 s | 8.34 s | 23.35 s | 36.18 s | 1 |
| overload | 2–6 s | 37.72 s | 387.45 s | 643.46 s | 645.90 s | 17 |

## 共同验收结果

- 每档任务完成率均为100%；
- 每档均覆盖一楼同层、二楼同层、上楼和下楼各5次；
- 每档均使用16种动态机器人/Go2/楼梯组合；
- 每档40个非法动作探针全部被动作掩码拒绝；
- 任务结束后队列、货物和机器人任务字段均清空；
- SMDP转移数据完整；
- 同种子重放结果一致。

## 判断

`normal` 模式的平均到达间隔大于平均服务时间，队列基本不积压，适合作为RViz
连续任务演示和阶段11主线稳定性测试。

`medium` 模式接近服务能力，存在可观察但受控的等待，适合比较规则策略与强化
学习的任务排序和组合分配能力。

`overload` 模式的任务发布速度约为服务速度的9–10倍，队列快速增长。该结果是
有效压力基线，但不适合作为日常训练主负载。后续不应通过丢弃任务掩盖过载，
应继续记录吞吐量、队列长度和截止时间违约。

## 证据

- `results/stage11/stage11_persistent_normal_20.jsonl`
- `results/stage11/stage11_persistent_normal_20.summary.json`
- `results/stage11/stage11_persistent_medium_20.jsonl`
- `results/stage11/stage11_persistent_medium_20.summary.json`
- `results/stage11/stage11_persistent_overload_20.jsonl`
- `results/stage11/stage11_persistent_overload_20.summary.json`
- 执行器：`scripts/run_stage11_persistent_20.py`

后续默认使用 `normal` 模式进行RViz 3–5任务人工验收；自动稳定性评估同时保留
`normal` 和 `medium`。`overload` 在阶段15压力与泛化测试中复用。

