# 0831 Stage 24 强优化基线 Pilot 与信息可见性决策

更新日期：2026-08-31  
目标期刊：IEEE Transactions on Robotics（T-RO）  
状态：`PILOT_ONLY`；正式 locked test 尚未授权。

## 1. 本阶段完成内容

Stage 24 已实现一个不修改 Stage 23 冻结源码的 causal rolling-horizon
optimization baseline。该基线：

- 复用现有候选生成器、合法动作 mask、资源状态和三种运输模式；
- 对完整合法候选集先做 reward-aligned 因果筛选，再对覆盖当前任务的最多
  24 个候选做三事件 nominal rollout；
- rollout tail 使用冻结的 time-greedy rule；
- 规划克隆把真实 task-keyed 交接采样器替换为公开 nominal duration，并用
  固定每次交接 5% timeout probability 计算期望风险修正；
- 固定 1,000 ms wall-clock budget，预留 25 ms 用于收尾；
- 超时后可采用已完成 rollout 的 incumbent；只有没有合法 incumbent 或内部
  错误时才回退 time-greedy；
- 候选被裁剪或搜索超时时将 optimality gap 记为未知，不伪报为 0。

新增文件：

- `src/warehouse_core/warehouse_core/stage24_rolling_optimizer.py`
- `src/warehouse_bringup/config/experiment_stage24_rolling_optimizer.yaml`
- `scripts/run_stage24_optimizer_interface_gate.py`
- `scripts/run_stage24_optimizer_smoke.py`
- `scripts/summarize_stage24_optimizer_pilot.py`

## 2. 必须先解决的信息可见性问题

审计确认，`MARKOV_CONTEXT_V3` 在 episode 初始时会编码全部 unresolved task，
其中包括尚未 release 的任务。一次直接检查得到：当前时刻 `0.0 s`，waiting
任务 1 个、pending 任务 79 个，而网络输入中的 pending task row 也是 79 个；
每行包含 source、target、priority、deadline 和 release-time offset。

因此 Stage 23 结果的可解释边界必须二选一：

1. `ANNOUNCED_PENDING`：79 个 pending order 被定义为 WMS 已提前公告的订单。
   在此前提下，Stage 23 策略没有读取“私有未来”，当前权重可以继续用于公平比较；
2. `RELEASED_ONLY`：任务只有被 publisher 正式下发后才可见。这个口径更符合当前
   项目对异步任务发布器的描述，但 Stage 23 策略输入含未来任务，不能直接作为同信息
   条件下的正式比较对象；必须新增 causal observation 并重新训练 PPO。

当前 Stage 24 协议同时实现两个可见性接口，但正式测试状态固定为
`blocked_pending_visibility_decision`，没有分配任何正式 seed。

## 3. 接口门结果

接口门 10/10 通过：

- 同 seed 决策确定；
- 所有选择均在当前合法 mask 内；
- selector 不修改真实环境；
- `RELEASED_ONLY` 对所有未发布任务属性的变动保持不变；
- 把真实 handover provider 替换为 fail-fast sentinel 后仍能完成选择；
- 极小预算会显式记录 timeout，并执行合法 time-greedy fallback；
- pilot、microcase 与 Stage 23 locked seed 分区互斥；
- 正式 Stage 24 测试仍处于阻塞状态。

结果：
`results/stage24_rolling_optimizer/interface_gate/stage24_optimizer_interface_gate.json`

## 4. 计算预算 Pilot

### 4.1 100 ms 档：未通过

调试流 `72110000` 显示：

- 预算合规率 86.34%；
- p95/p99 latency：108.64/121.74 ms；
- 57/227 次搜索触及预算；
- 0 非法动作、0 资源泄漏、0 fallback。

该档保留为计算敏感性失败结果，不作为主预算。

### 4.2 1,000 ms 档：通过

用全新开发流 `72111000–72111019` 运行 20 个 persistent mixed-curriculum
episodes，每条 80 tasks：

| 计算指标 | 结果 |
|---|---:|
| 决策数 | 4,561 |
| p50 latency | 0.056 ms |
| p95 latency | 765.229 ms |
| p99 latency | 956.418 ms |
| 最大 latency | 1,011.208 ms |
| 1,000 ms 内完成 | 99.759% |
| 搜索 timeout | 0.811% |
| fallback | 0 |
| 全根精确枚举 | 79.105% |
| shortlist/timeout 导致 gap 未知 | 20.895% |

p50 很低主要来自只有 `WAIT` 的事件，不能作为实际 assignment 求解延迟的单独代表。
验收依据是全决策 p95/p99、预算合规率和 timeout/fallback 联合报告。

## 5. 20 流配对 Pilot 结果

统计口径：每个 episode 内先计算 80 个任务的 P95 waiting，再对 20 个共享流做
10,000 次 paired episode bootstrap。该统计只用于检查方法方向和冻结 1 s
计算预算，不是正式确认性证据。

| 指标 | Rolling optimizer | Time-greedy | Optimizer - baseline | 95% paired CI |
|---|---:|---:|---:|---:|
| Mean episode P95 waiting (s) | 505.135 | 624.026 | -118.891 | [-173.131, -72.049] |
| Success rate | 99.438% | 97.188% | +2.250 pp | [+1.625, +2.875] pp |
| Successful throughput (tasks/h) | 68.743 | 67.370 | +1.372 | [+0.963, +1.803] |
| Mean waiting (s) | 113.291 | 141.471 | -28.180 | [-40.314, -18.074] |
| Mean episode P95 flow (s) | 551.517 | 655.942 | -104.425 | [-154.338, -62.991] |
| Distance/task | 53.544 | 72.446 | -18.902 | [-20.097, -17.713] |
| Episode reward | -270.130 | -356.840 | +86.711 | [+57.947, +121.290] |

主指标在 18/20 个共享流上方向更优。安全检查为 0 illegal action、0 resource
leak，task fingerprint 和 handover-potential fingerprint 完全一致。

模式使用存在明显差异：optimizer 使用 `SINGLE_CAR=800`、`SINGLE_DOG=638`、
`CAR_DOG_CAR=162`；time-greedy 为 `799/276/525`。这说明 optimizer 降低了
高风险 relay 使用，但 pilot 不能把相关性直接解释为因果机制。

原始与汇总结果：

- `results/stage24_rolling_optimizer/pilot/stage24_optimizer_smoke_20ep_seed72111000.json`
- `results/stage24_rolling_optimizer/pilot/stage24_optimizer_paired_pilot_summary.json`

## 6. 当前结论边界

可以说：Stage 24 强优化基线的接口、安全、因果 handover 边界和 1 s 计算预算
在 20 个非正式流上通过；其相对 time-greedy 的 pilot 方向值得进入下一门。

不能说：optimizer 已正式优于 PPO 或 time-greedy；不能把 20 个开发流作为论文
主结果；不能称所有决策都得到全局最优解；不能在信息可见性未冻结时分配或运行
Stage 24 正式 seed。

## 7. 下一步唯一前置决策

推荐选择 `RELEASED_ONLY`，因为当前项目将任务描述为异步下发，而不是提前公开完整
订单簿。选择后应：

1. 新增不包含 pending task 明细的 causal observation contract；
2. 做 state/observation sufficiency 审计，并把问题准确表述为 MDP 或 POMDP；
3. 用新的训练/验证 seed 重训 PPO，长训练仍由用户手动启动；
4. 冻结 causal PPO 与当前 optimizer；
5. 预注册 noninferiority margins、全新正式 streams 和 crossed/paired 统计方案；
6. 一次性比较 causal PPO、rolling optimizer、time-greedy、single-dog。

若选择 `ANNOUNCED_PENDING`，则必须把“全部订单及 release time 在调度前由 WMS
公告”写入系统假设，并让所有比较方法获得完全相同公告；当前 Stage 23 权重才可直接
进入 Stage 24 正式比较。
