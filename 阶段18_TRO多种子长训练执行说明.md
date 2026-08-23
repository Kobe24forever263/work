# 阶段18：T-RO多种子长训练执行说明

## 目的

本阶段主实验对完整模型和3个上下文消融条件分别进行10个端到端独立训练种子，共40次正式训练。每次训练包含独立网络初始化、独立规则蒸馏warm-start、独立PPO任务流和独立minibatch顺序；验证集在所有条件间共享，最终测试集保持锁定。

## 条件

1. `full_context_v2`
2. `no_queue_resource`
3. `no_handover_cues`
4. `no_persistent_position`
后续算法消融另行执行：

- `fixed_gamma`
- `flat_masked_ppo`

RTAW尚未实现，不包含在本批40次主实验中。

## 单次正式训练规格

- 2000 updates
- 每update 4 episodes
- 8000 PPO episodes / 160000 tasks
- 30 epochs独立warm-start，共400个规则示范任务
- 每50 updates在固定100个validation seeds上选best checkpoint
- 每50 updates保存断点
- 每20 updates输出一次进度
- 不使用test seeds选模型

## 启动方法

进入工作区：

```bash
cd /Users/lab4099/Desktop/Mujoco/work
```

先只查看某次训练计划，不执行：

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python scripts/run_stage18_train_one.py full_context_v2 1
```

确认后执行该次完整训练：

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python scripts/run_stage18_train_one.py full_context_v2 1 --phase all --updates 2000 --execute
```

第二个参数为训练种子编号，只允许`1`至`10`。第一个参数替换为其他条件即可。先完成`full_context_v2`的10个种子并进行稳定性评估，再执行三种上下文消融。

完整模型第6至10个种子的串行批次（仅在当前seed 05结束后手动执行）：

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python scripts/run_stage18_full_seed_06_10_batch.py
```

完整模型10种子通过后，三种上下文消融分别启动。每个批次包含该消融的10个种子，约7至8小时：

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python scripts/run_stage18_context_ablation_batch.py no_queue_resource
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python scripts/run_stage18_context_ablation_batch.py no_handover_cues
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python scripts/run_stage18_context_ablation_batch.py no_persistent_position
```

## 分开执行warm-start和PPO

只执行warm-start：

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python scripts/run_stage18_train_one.py full_context_v2 1 --phase warm-start --execute
```

warm-start通过后只执行PPO：

```bash
/Users/lab4099/Desktop/Mujoco/task2_handover/.pixi/envs/default/bin/python scripts/run_stage18_train_one.py full_context_v2 1 --phase ppo --updates 2000 --execute
```

## 输出位置

每次运行独占一个目录：

```text
results/stage18/<condition>/seed_01/
```

主要文件：

- `warm_start.pt`：该种子的独立预热权重
- `warm_start.summary.json`：预热验收结果
- `stage18_ppo.pt`：最终权重
- `stage18_ppo.best.pt`：固定验证集选出的最佳权重
- `stage18_ppo.history.json`：完整训练曲线
- `stage18_ppo.summary.json`：训练摘要
- `checkpoints/`：每50 updates的断点
- `run_plan.json`：本次训练使用的参数和种子范围

## 完成标准

单次训练结束不等于论文结论成立。首先检查：

- `passed=true`
- 8000 episodes完成
- action width为30
- 非法动作数为0
- 资源泄漏数为0
- 训练种子、验证种子和condition与计划一致
- `best.pt`、history和summary均存在

10个Full种子全部完成后，自动生成十种子继续门报告。只有完整模型跨种子结果稳定，才启动30次上下文消融，避免在配置异常时浪费整批计算。三种消融均使用与完整模型成对一致的10个训练种子。
