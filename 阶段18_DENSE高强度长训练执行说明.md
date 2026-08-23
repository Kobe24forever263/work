# 阶段18：DENSE高强度长训练执行说明

## 实验目的

DENSE实验采用与MEDIUM相同的模型、奖励、PPO参数、10个独立训练种子和配对消融流程，仅将并发任务到达间隔从MEDIUM的55至75秒缩短为DENSE的15至25秒，以检验资源竞争和排队状态在高负载下的价值。

所有DENSE结果独立保存到 `results/stage18_dense/`，不会覆盖MEDIUM权重。训练、validation和后续test均须标明 `arrival_profile=DENSE`。

## 实验顺序

1. 完整模型 `full_context_v2`：10个种子。
2. 十种子继续门通过后，再运行：
   - `no_queue_resource`：10个种子；
   - `no_handover_cues`：10个种子；
   - `no_persistent_position`：10个种子。
3. 锁定各种子的best checkpoint后，才运行锁定test seeds。

## 单次训练规格

- `execution_mode=CONCURRENT`
- `arrival_profile=DENSE`
- 2000 updates
- 每update 4 episodes
- 每种子8000 episodes
- 每50 updates使用固定validation seeds选择best checkpoint
- 每20 updates输出一次进度

## 手动启动完整模型10种子

```bash
cd /Users/lab4099/Desktop/Mujoco/work

/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python \
scripts/run_stage18_dense_full_ten_seed_batch.py
```

预计约7至8小时。脚本支持重启：已完成且通过校验的种子会跳过。

## 完整模型通过后，分别启动三项消融

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python \
scripts/run_stage18_context_ablation_batch.py no_queue_resource --arrival-profile DENSE

/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python \
scripts/run_stage18_context_ablation_batch.py no_handover_cues --arrival-profile DENSE

/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python \
scripts/run_stage18_context_ablation_batch.py no_persistent_position --arrival-profile DENSE
```

每个消融批次约7至8小时。不要并行运行这些CPU训练。

## 验收标准

- 10/10种子完成且 `passed=true`；
- 每个摘要均为 `arrival_profile=DENSE`；
- 非法动作数为0；
- 资源泄漏数为0；
- 三种运输模式均被暴露并实际选择；
- 完整模型继续门通过后才开始消融；
- 结果路径、权重和曲线与MEDIUM完全隔离。
