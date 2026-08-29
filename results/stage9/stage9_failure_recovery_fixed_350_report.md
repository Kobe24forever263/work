# 阶段 9 七类故障恢复 350 次验收报告

日期：2026-08-04

## 结论

阶段 9 固定故障注入门禁通过：`350/350 PASS`，失败 0 次。七类故障各注入
50 次，全部在恢复时限内完成资源清理和所有权恢复。

| 故障 | 次数 | 恢复结果 |
|---|---:|---|
| 规划失败 | 50 | RETRY，50/50 |
| 对接超时 | 50 | RETRY，50/50 |
| 机器人失效 | 50 | REASSIGN，50/50 |
| 楼梯占用超时 | 50 | RETRY，50/50 |
| 区域锁超时 | 50 | RETRY，50/50 |
| 传感器失效 | 50 | RETRY，50/50 |
| 任务取消 | 50 | CANCELLED，50/50 |

## 每次故障的验收项目

- 任务锁、货物锁、区域锁和交接区锁全部释放；
- 当前楼梯占用和任务对应的楼梯排队项全部清理；
- 中断的部分交接恢复给安全发送方；
- 货物保持唯一所有者，接收方无残留 `cargo_id`；
- 机器人失效时标记故障并进入重新分配；
- 可恢复故障进入 RETRY，取消任务进入 CANCELLED；
- 后续任务可以立即重新申请相同区域、交接区和楼梯；
- 所有恢复都在 5 秒逻辑时限内完成。

## 实现与证据

- 恢复协调器：`src/warehouse_core/warehouse_core/recovery.py`
- 楼梯任务清理：`src/warehouse_core/warehouse_core/spatial.py`
- 执行入口：`scripts/run_stage9_failure_recovery_350.command`
- 逐次结果：`results/stage9/stage9_failure_recovery_fixed_350.jsonl`
- 汇总：`results/stage9/stage9_failure_recovery_fixed_350.summary.json`
- 核心测试：`11/11 PASS`

本阶段验证确定性的状态恢复和资源回收，不包含随机扰动，也不替代阶段 16 的
Gazebo 物理故障实测。下一阶段为阶段 10 规则调度基线：覆盖全部任务类型，
运行 200 个固定任务，验证调度生成合法的车—狗—车运输链、资源顺序、任务完成
状态和统计指标。

