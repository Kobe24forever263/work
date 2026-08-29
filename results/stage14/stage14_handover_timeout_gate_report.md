# Stage 14 P95 交接超时逻辑门

## 结论

当前逻辑门通过。统一暂定 P95 为 `2.697871 s`；严格大于该值的单次交接记为 `HANDOVER_TIMEOUT`，当前任务失败且不重试，资源、货物占用和机器人任务状态必须全部释放。

## 数据可信度

现有 Stage 8 的 300 条记录没有开始与结束时间，不能用于 P95。这里的 1000 条样本（车→狗 500、狗→车 500）属于带固定随机种子的逻辑仿真标定，只允许用于正式强化学习前的逻辑训练，不得宣称为 RViz 或实机交接性能。后续每类至少收集 500 条带时间戳的测量记录后，按同一 nearest-rank 方法替换阈值。

## 统计结果

- 标定样本超过 P95：`50/1000`（`5.00%`）。
- 500 个持久状态回合：`10000` 个任务，其中成功 `9643`，交接超时失败 `357`。
- 交接尝试：`6866` 次，超时 `357` 次（`5.20%`）。
- 协作任务：`3500` 个，任务失败率 `10.20%`；因为每个车—狗—车任务有两次交接，理论暴露失败率约为 `9.75%`。
- 三种运输模式：`{'SINGLE_CAR': 5000, 'CAR_DOG_CAR': 3500, 'SINGLE_DOG': 1500}`。

## 失败语义

`duration_s > P95` 才超时；`duration_s == P95` 仍成功。超时事务先回滚到唯一安全所有者，然后任务层将货物标为失败、清空参与机器人携货与任务字段、释放楼梯/交接点/区域资源，并让机器人从当前位置进入待命位。任务失败不立刻终止整个回合，只有回合已处理任务数达到上限时才正常结束。

## 验收断言

```json
{
  "calibration_has_1000_current_mode_samples": true,
  "nearest_rank_p95_has_exactly_five_percent_above": true,
  "strict_boundary_accepts_equal_and_rejects_above": true,
  "five_hundred_episodes_resolve_ten_thousand_tasks": true,
  "logical_attempt_timeout_rate_is_close_to_five_percent": true,
  "cooperative_task_failure_rate_matches_two_attempt_exposure": true,
  "every_failed_task_is_handover_timeout": true,
  "no_resource_robot_or_stair_leaks": true,
  "fixed_seed_replay_equal": true
}
```
