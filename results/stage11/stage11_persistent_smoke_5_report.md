# 阶段 11 持久状态连续任务 5 任务烟雾报告

日期：2026-08-05

## 结论

阶段 11 第一个纯逻辑里程碑通过：`5/5 PASS`。

- 异步到达任务全部进入队列且无丢失；
- 同一时刻只执行一个主任务；
- 机器人位置与累计距离跨任务保留；
- 使用了多种 Carter/Go2 分配组合；
- `dog_2` 完成上楼后，其二楼状态被后续下楼任务正确继承；
- 下楼完成后 `dog_2` 回到一楼，不是被任务 reset；
- 所有任务结束后 `cargo_id` 和 `task_id` 均清空；
- 每个决策保存了完整 SMDP 转移数据。

本次总逻辑时间为 `242.14 s`，核心回归测试为 `13/13 PASS`。

## 证据

- 逐决策数据：`results/stage11/stage11_persistent_smoke_5.jsonl`
- 汇总：`results/stage11/stage11_persistent_smoke_5.summary.json`
- 环境实现：`src/warehouse_core/warehouse_core/persistent_dispatch.py`
- 执行入口：`scripts/run_stage11_persistent_smoke.py`

下一步扩展到固定种子的 20 个连续任务，补齐任务优先级、截止时间、等待时长、
完整动作候选与掩码统计，并检查不同机器人组合和四类任务覆盖。

