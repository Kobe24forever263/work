# 阶段14 SMDP-PPO训练与测试说明

日期：2026-08-09

## 1. 当前边界

当前已完成Context V2上下文策略预热和10个episode的SMDP-PPO短时门禁，但没有替用户
启动长时间训练。短门禁验证任务上下文绑定、状态—候选交互、变时折扣、动作掩码、
PPO更新、权重保存和独立评估链路。它不能证明策略已经收敛，也不能证明学习策略优于
规则调度器。

已通过的三组短门禁：

| 训练模式 | 训练episode | 留出任务 | 非法动作 | 资源泄漏 | 状态 |
|---|---:|---:|---:|---:|---|
| SERIAL | 4 | 80 | 0 | 0 | PASS |
| CONCURRENT | 4 | 80 | 0 | 0 | PASS |
| MIXED | 4 | 80 | 0 | 0 | PASS |
| Context V2 CONCURRENT | 10 | 100 | 0 | 0 | PASS |

旧短门禁权重位于`results/stage14/ppo/smoke/`，只保留为历史接口证据。Context V2预热
权重位于`results/stage14/context_warm_start/stage14_context_policy_warm_start.pt`；新PPO
短门禁位于`results/stage14/ppo/smoke_context_v2/`，两者均不应用作最终部署权重。

## 2. 正式部署长训练

真实系统正式部署仍启用`CONCURRENT`，允许最多4个无冲突任务在途。每次任务决策由同一套
策略在合法候选中选择`SINGLE_CAR`、`SINGLE_DOG`或`CAR_DOG_CAR`，并继续选择具体
机器人、楼梯和待命点。SERIAL在阶段15单独训练为降级策略和并发消融基线；MIXED继续
只保留为环境门禁。两者都不替代CONCURRENT正式部署权重。

策略目标不是学习固定模式分布。每个28维候选动作会直接携带当前任务的优先级、等待、
剩余时限、起终点三维坐标、相对最低合法候选的代价差/比值和参与机器人数量；网络显式
融合状态、候选及两者逐元素交互。由此，同一运输模式在不同位置、占用、楼梯和期限下可
得到不同分数。正式验收按条件选择矩阵判断，而不是要求三种模式次数平均。

阶段15现新增一套`SERIAL + MEDIUM`长训练作为系统降级策略和并发能力消融基线；它不
改变CONCURRENT作为正式部署方向的结论。SERIAL训练方法、独立输出目录和三方验收见
`阶段15_SERIAL_PPO训练与验收说明.md`。

长训练必须由用户在终端手动启动，唯一正式命令为：

```bash
cd /Users/lab4099/Desktop/Mujoco/work/scripts
./run_stage14_realistic_ppo_long.command
```

该命令固定使用`CONCURRENT + MEDIUM + Reward V2`，默认执行2000个update，每个
update采集4个完整episode，即共8000个训练episode、160000个任务，按首轮实测约为
47万次高层SMDP决策。终端显示实时进度条，每20个update才打印一次指标、执行留出评估
并保存检查点。旧项目曾运行约500万timesteps，但旧环境与当前事件驱动SMDP的动作
粒度不同，不能把timesteps、update和episode直接视作同一“轮”。
运行中可以使用`Control-C`停止，已经保存的检查点不会消失。

推荐用于测试的最佳权重只有一个：

`results/stage14/ppo/long/realistic_concurrent_medium_reward_v2_context_v2/stage14_realistic_concurrent_medium_reward_v2_context_v2.best.pt`

同目录`stage14_realistic_concurrent_medium_reward_v2_context_v2.pt`是最后一个update的
权重。优先评估best，同时保留final用于判断后期训练是否退化。`checkpoints/`每20个
update保存一次。

Reward V2将任务时间、失败和机器人占用重新校准，但没有为任何运输模式增加专属奖励。
训练使用可保存的SMDP运行回报标准差进行归一化；归一化统计随best、final和检查点权重
共同保存，恢复训练时继续使用同一统计。旧16维动作/Context V1权重与新28维Context V2
网络不兼容，程序会明确拒绝续训。首次执行正式脚本会自动载入已通过的上下文预热权重，
然后从新的PPO优化器、critic和回报归一化统计开始。

Reward V2门禁证据位于：

- `results/stage14/reward_v2/stage14_reward_v2_gate.summary.json`
- `results/stage14/reward_v2/stage14_reward_v2_gate_report.md`

训练进度中的`raw_episode_reward`是20任务episode原始总回报，用于和规则基线比较；
`normalized_step_reward`是PPO实际使用的归一化单步回报。归一化不会强制回报落入0～1。

## 3. 断点继续

正式长训练脚本的第一个参数是检查点路径，第二个参数是“本次额外增加的update数量”。
例如从第1000个update继续追加1000个update：

```bash
./run_stage14_realistic_ppo_long.command \
  /Users/lab4099/Desktop/Mujoco/work/results/stage14/ppo/long/realistic_concurrent_medium_reward_v2_context_v2/checkpoints/stage14_realistic_concurrent_medium_reward_v2_context_v2_update_01000.pt \
  1000
```

恢复时执行模式和负载必须与检查点一致。机器人状态不会在单个episode内部逐任务复位；
新的episode仍按训练环境定义重新初始化，这是当前重置边界。

## 4. 测试权重效果

训练结束后，使用独立留出种子将学习策略与规则调度器放在同一任务流上比较：

```bash
cd /Users/lab4099/Desktop/Mujoco/work/scripts
./evaluate_stage14_realistic_ppo.command
```

该命令默认读取正式best权重并测试30个留出episode，即600个任务。也可以将自定义
权重作为第一个参数、episode数量作为第二个参数。评估会在权重旁生成：

- `*.evaluation.json`：完整机器可读指标；
- `*.evaluation.md`：学习策略与规则基线对照表。

至少检查以下指标：任务成功率、平均回报、任务吞吐、平均逻辑仿真时间、非法动作、资源
泄漏和条件选择矩阵。并发策略还必须出现2个以上同时在途任务。条件矩阵按“当前环境中
最低代价合法模式→策略所选模式”分组；三种参考模式各自的对角选择率必须至少20%，以
排除全局模式塌缩，但不把规则答案或固定模式比例强加为100%目标。逻辑仿真时间不等于
RViz墙钟时间，进入RViz前还要使用真实规划、移动、对准和交接时间戳进行校准。

## 5. 验收原则

强化学习不要求达到100%成功。P95交接超时会保留自然失败率，但以下安全门禁不能放宽：

- 非法动作必须为0；
- 货物、机器人、楼梯、区域和泊位资源泄漏必须为0；
- 固定留出种子的评估必须可复现；
- 同一正式策略必须在课程中暴露单车、单狗和车—狗—车三种运输候选，不能拆成三套权重；
- 同一策略必须随任务几何、期限、机器人位置/占用和楼梯状态切换方案；不能因为历史上
  某种模式平均回报较高，就对后续任务固定选择该模式；
- 学习策略是否优于规则策略，应由成功率、回报、吞吐、等待、流转时间和里程共同判断，
  不能只挑一个有利指标。
