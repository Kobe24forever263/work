# 阶段 7 货物所有权固定任务 200 次验收报告

日期：2026-08-04

## 结论

阶段 7 固定任务门禁通过：`200/200 PASS`，失败 `0` 次。

四条跨楼层协作链东北、西北、西南、东南各执行 50 次。每次任务都完成以下
可审计状态流转：

1. 一楼 Carter 在装货点取得货物；
2. 一楼 Carter 向对应 Go2 提交货物；
3. Go2 向二楼 Carter 提交货物；
4. 二楼 Carter 在卸货点完成交付并释放货物。

## 所有权验收

- 运输期间始终只有一个机器人持有货物；
- `Cargo.owner_id` 始终与机器人 `cargo_id` 一致；
- 两次交接均经过 TRANSFER、VERIFY、COMMIT；
- 没有货物复制；
- 没有货物丢失；
- 完成交付后所有机器人均无货物残留；
- 完成交付后货物为 `DELIVERED`，所有者和约束均已清空。

## 路线结果

| 路线 | 协作链 | 结果 |
|---|---|---:|
| 东北 | `car_f1_1 → dog_1 → car_f2_2` | 50/50 |
| 西北 | `car_f1_2 → dog_2 → car_f2_1` | 50/50 |
| 西南 | `car_f1_3 → dog_3 → car_f2_1` | 50/50 |
| 东南 | `car_f1_4 → dog_4 → car_f2_2` | 50/50 |

## 证据

- 逐次结果：`results/stage7/stage7_cargo_ownership_fixed_200.jsonl`
- 汇总：`results/stage7/stage7_cargo_ownership_fixed_200.summary.json`
- 执行入口：`scripts/run_stage7_cargo_200.command`
- 状态实现：`src/warehouse_core/warehouse_core/cargo_task.py`

本轮是固定路线上的货物事务与所有权门禁，不包含随机扰动，也不替代阶段 16
的 Gazebo 动力学验收。下一阶段进入阶段 8 正式交接：对车到狗、狗到车以及
同层交接三类事务进行成功、拒绝和回滚测试，每类 100 次，成功率目标不低于
95%，且失败后不得残留错误所有权或锁。

