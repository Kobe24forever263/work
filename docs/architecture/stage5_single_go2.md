# 阶段 5：单 Go2 双向楼梯可靠性

日期：2026-07-30  
状态：工具链完成、双向烟雾通过；正式批量门禁尚未执行

## 已实现

- 4 条上楼路线和 4 条下楼路线，共 8 个 13 航点 YAML。
- `single_go2_stair_validation.launch.py` 可选择 `dog_1..dog_4` 和
  `up/down`，支持初始位置随机扰动。
- `run_single_go2_stair.command` 用于带 Gazebo 和 RViz 的单次可视化检查。
- `run_stage5_batch.py` 提供固定随机种子、逐次超时、失败分类、JSONL 原始证据
  和汇总 JSON。
- 正式计划按 4 条楼梯 × 2 个方向 × 每组合 20 次生成 160 次试验，满足总数
  不少于 100 次且每条楼梯每个方向不少于 20 次的双重门禁。

## 当前证据

种子 `20260730` 的烟雾测试：

- `dog_1 up`：PASS
- `dog_1 down`：PASS

结果保存在 `results/stage5/stage5_smoke.jsonl` 与对应汇总文件。

## 尚未完成

- 160 次正式批量运行尚未执行。
- 当前 SCAN `open_loop` 验证证明点云路径和运动学执行可达，但不等于真实
  接触动力学关节控制。因此不能用烟雾结果把阶段 5 标记为完成。
- 正式报告需要成功率、耗时分布、失败原因分布、各楼梯/方向分组统计、随机
  种子和软件版本。

正式计划命令：

```bash
python3 /Users/lab4099/Desktop/Mujoco/work/scripts/run_stage5_batch.py \
  --seed 20260730 --repeats 20 \
  --output /Users/lab4099/Desktop/Mujoco/work/results/stage5/stage5_full.jsonl
```
