# 阶段15 SERIAL PPO训练与验收说明

日期：2026-08-09

## 1. 目标与边界

SERIAL表示同一时间最多一个配送任务在途，不表示一个episode只有一个任务。每个episode
仍包含20个异步到达任务；当前任务执行期间新任务继续入队，任务结束后机器人位置、楼层、
累计里程、任务数和待命泊位全部保留，策略再从等待队列中选择下一任务及运输方案。只有
episode累计解决20个任务后才执行整体重置。

SERIAL策略继续使用同一个Context V2网络，在合法候选中共同选择`SINGLE_CAR`、
`SINGLE_DOG`和`CAR_DOG_CAR`，不拆成三套权重。Carter速度2.0m/s、Go2速度0.8m/s，
Reward V2、284维状态、`1536 × 28`候选动作、SMDP变时折扣和回报标准差归一化均与
CONCURRENT正式策略相同。实验中唯一主动改变的变量是最大在途任务数由4变为1。

## 2. 已完成短门禁

10个训练episode的SERIAL接口门禁已通过：5 updates、每update 2 episode、共324个
事件决策。PPO参数L2变化为1.218345，损失、熵、折扣和归一化统计均为有限值。5个留出
episode共100任务，完成97、交接超时失败3，成功率97%；单车50、车—狗—车35、单狗15，
三种模式均被当前上下文选出。非法动作0、资源泄漏0，最大在途任务正确记录为1。

短门禁权重位于：

`results/stage14/ppo/smoke_serial_context_v2/stage14_serial_context_v2_smoke.best.pt`

该权重只证明训练接口可用，禁止用于正式性能结论或最终演示。

## 3. 正式长训练（必须由用户手动启动）

终端执行：

```bash
cd /Users/lab4099/Desktop/Mujoco/work/scripts
./run_stage14_realistic_serial_ppo_long.command
```

默认执行2000 updates，每update 4个完整episode，共8000 episode、160000任务。训练从与
并发策略相同的Context V2监督预热权重开始，不从CONCURRENT PPO权重继续，从而避免把
并发critic和资源占用偏好带入SERIAL对照。优化器和奖励归一化统计独立建立。

正式输出目录：

`results/stage14/ppo/long/realistic_serial_medium_reward_v2_context_v2/`

- `stage14_realistic_serial_medium_reward_v2_context_v2.best.pt`：首选评估权重；
- `stage14_realistic_serial_medium_reward_v2_context_v2.pt`：最终update权重；
- `checkpoints/`：每20 updates保存一个断点；
- `*.history.json`：逐update训练历史；
- `*.summary.json`：训练门禁和留出摘要。

中断后恢复示例：

```bash
./run_stage14_realistic_serial_ppo_long.command \
  /Users/lab4099/Desktop/Mujoco/work/results/stage14/ppo/long/realistic_serial_medium_reward_v2_context_v2/checkpoints/stage14_realistic_serial_medium_reward_v2_context_v2_update_01000.pt \
  1000
```

第二个参数表示本次追加的update数量。

## 4. 正式三方评估

长训练结束后执行：

```bash
cd /Users/lab4099/Desktop/Mujoco/work/scripts
./evaluate_stage14_realistic_serial_ppo.command
```

默认读取SERIAL best权重，使用30个独立episode、600任务，与SERIAL时间规则和SERIAL风险
规则进行任务级配对比较。三者使用相同任务、初态和按任务ID绑定的交接耗时样本。可将
episode数作为第二个参数，例如正式扩大到100个episode：

```bash
./evaluate_stage14_realistic_serial_ppo.command \
  /Users/lab4099/Desktop/Mujoco/work/results/stage14/ppo/long/realistic_serial_medium_reward_v2_context_v2/stage14_realistic_serial_medium_reward_v2_context_v2.best.pt \
  100
```

结果保存到`results/stage15/stage15_serial_medium_three_way_*ep.json`及同名Markdown。

## 5. 验收标准

- 全部任务进入明确的`COMPLETED`或受控`FAILED`；
- 最大在途任务必须等于1；
- 非法动作、货物残留和机器人/楼梯/区域/泊位资源泄漏必须为0；
- 三种合法运输模式均被暴露，策略选择随任务与持久位置变化，不能塌缩为固定模式；
- 相对SERIAL风险规则同时报告成功率、原始回报、平均/P95等待、平均/P95流转、总里程、
  已解决吞吐和成功吞吐，不以单一指标判定；
- 再使用同一任务种子比较SERIAL PPO与CONCURRENT PPO，量化并发能力带来的系统收益。

SERIAL权重定位为系统降级策略和并发能力消融基线。即使SERIAL在低负载下表现良好，也不
自动替代当前CONCURRENT正式部署方向。

## 6. 正式训练与独立验收结果（2026-08-09）

用户已完成2000 updates正式训练，共8000 episode、160000任务；训练摘要门禁通过，参数
L2变化24.1985，奖励归一化统计有效，非法动作和资源泄漏均为0。最佳权重出现在update
940，后续评估统一使用`stage14_realistic_serial_medium_reward_v2_context_v2.best.pt`，
不以最后update权重替代best。

随后完成100个独立MEDIUM episode、每个策略2000任务的三方配对评估。SERIAL PPO完成
1943任务，成功率97.15%；时间规则完成1925任务，成功率96.25%；风险规则完成1938任务，
成功率96.90%。PPO相对时间规则成功率提高0.90个百分点、成功吞吐提高0.459任务/小时、
每任务总里程减少7.176米；相对风险规则成功率提高0.25个百分点、成功吞吐提高0.153
任务/小时、每任务总里程减少2.310米。上述成功率、成功吞吐和里程方向的配对95%置信
区间均不跨0。

当前SERIAL PPO并非所有指标更优：相对风险规则平均流转时间增加0.166秒，95%置信区间
[0.003,0.325]；平均等待差0.055秒且置信区间跨0。正式结论是“可靠性和里程略优，时延
近似持平”，不能写成全面优于规则。

正式证据：

- `results/stage14/ppo/long/realistic_serial_medium_reward_v2_context_v2/*.summary.json`；
- `results/stage15/stage15_serial_medium_three_way_100ep.json`；
- `results/stage15/stage15_serial_medium_three_way_100ep.md`。

## 7. SERIAL与CONCURRENT配对消融结果（2026-08-09）

新增`scripts/run_stage15_serial_concurrent_comparison.command`。两套独立训练的best权重在
完全相同的任务、初态和按任务ID绑定的交接样本上比较，差值统一为CONCURRENT减SERIAL。
MEDIUM、DENSE、BURST各100 episode、每套策略每场景2000任务，全部任务有明确结果，
非法动作0、资源泄漏0；SERIAL最大在途1，CONCURRENT随负载达到2、3、4。

| 场景 | 成功率变化 | 成功吞吐变化（任务/h） | 平均等待变化（s） | P95等待变化（s） | 回报变化 |
|---|---:|---:|---:|---:|---:|
| MEDIUM | +0.45个百分点 | +0.589 | -3.482 | -12.034 | +2.526 |
| DENSE | +0.40个百分点 | +38.135 | -84.114 | -241.701 | +65.632 |
| BURST | +0.40个百分点 | +61.859 | -171.034 | -332.511 | +132.268 |

除BURST成功率差的置信区间下界约等于0外，表中吞吐、等待和回报优势的配对95%置信区间
均不跨0。并发收益随负载显著扩大，证明CONCURRENT仍应作为主策略；SERIAL保留为降级
模式和消融基线。

60任务连续`NORMAL→DENSE→BURST`在SERIAL下会触及当前episode决策步上限，导致少量任务
未解析。因此本次公平消融改用三个独立20任务场景，并保留“所有任务必须有结果”的硬门禁，
没有把未解析任务伪装成普通失败。若后续必须比较60任务连续轨迹，应先单独提高并验证
SERIAL环境决策步上限。

正式证据：`results/stage15/stage15_serial_vs_concurrent_100ep.json`及同名Markdown。
