# Draft-to-workspace evidence map

This file is for internal verification and is not a source of experimental claims.

| Draft statement | Current source of truth |
|---|---|
| Two-floor warehouse geometry and four stairs | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_bringup/config/scene.yaml` |
| Six Carter and four Go2 robots | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_bringup/config/robots.yaml` |
| Carter 2.0 m/s, Go2 0.8 m/s | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_core/warehouse_core/persistent_dispatch.py` |
| Task mix and arrival profiles | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_core/warehouse_core/stage14_training.py` |
| Three legal transport modes and compound assignment | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_core/warehouse_core/persistent_dispatch.py` |
| Transactional cargo ownership | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_core/warehouse_core/cargo_task.py` |
| Concurrent event model and resource claims | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_core/warehouse_core/concurrent_dispatch.py` |
| 284-D state, 1536 slots, current 30-D candidate | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_core/warehouse_core/stage12_encoding.py` |
| Context–candidate network and 96-unit layers | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_core/warehouse_core/stage14_policy.py` |
| PPO, variable-discount GAE, reward normalization | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_core/warehouse_core/stage14_ppo.py` |
| Reward V2 override | `/Users/lab4099/Desktop/Mujoco/work/src/warehouse_core/warehouse_core/stage14_training.py` |
| Current 10-seed protocol | `/Users/lab4099/Desktop/Mujoco/work/阶段18_TRO多种子长训练执行说明.md` |
| Stage 18 fixed-load validation trajectories (Medium) | `/Users/lab4099/Desktop/Mujoco/work/results/stage18/full_context_v2/seed_*/stage18_ppo.history.json` |
| Stage 18 fixed-load validation trajectories (Dense) | `/Users/lab4099/Desktop/Mujoco/work/results/stage18_dense/full_context_v2/seed_*/stage18_ppo.history.json` |
| Stage 18 fixed-load validation trajectories (Burst) | `/Users/lab4099/Desktop/Mujoco/work/results/stage18_burst/full_context_v2/seed_*/stage18_ppo.history.json` |
| Training-trajectory figure generator, long-form data, and validation audit | `scripts/make_training_convergence_figure.py`; `figures/data/training_convergence.csv`; `figures/data/training_convergence_validation.json` |
| Stage 19 protocol, claim boundary, and handover | `/Users/lab4099/Desktop/Mujoco/work/0823handover.md` |
| Stage 19 formal summary and crossed-bootstrap intervals | `/Users/lab4099/Desktop/Mujoco/work/results/stage19_mixed_recovery/stage19_mixed_recovery_summary.json` |
| Stage 19 per-training-seed and per-task locked-test records | `/Users/lab4099/Desktop/Mujoco/work/results/stage19_mixed_recovery/locked_test_v1/` |
| Stage 19 formal data-quality audit | `/Users/lab4099/Desktop/Mujoco/work/results/stage19_mixed_recovery/stage19_quality_audit.ipynb` |
| Stage 19 figure generator and raw-to-summary checks | `scripts/make_stage19_figures.py` |
| Stage 21 CPU latency and scaling evidence | `examples/manuscript/data/stage21/latency_scaling_formal.json` |
| Stage 21 logical fault-robustness evidence | `examples/manuscript/data/stage21/fault_robustness_locked_summary.json` |
| Stage 21 frozen protocol and model hashes | `examples/manuscript/data/stage21/stage21_fault_protocol_freeze_manifest.json` |
| Stage 21 chart contracts, Origin instructions, and claim boundaries | `examples/manuscript/STAGE21_PLOTTING_GUIDE.md` |
| Go2 simulation-model inset in the architecture figure | User-generated screenshot `/Users/lab4099/Desktop/workasset/截屏2026-08-24 10.44.30.png`; manuscript asset `go2_simulation_model.png`; SHA-256 `979a7c678a8042ecc13150ec5e4791ba4b31e880c50820d9779c1fc401d19701` |
| Carter simulation-model inset in the architecture figure | User-generated screenshot `/Users/lab4099/Desktop/workasset/截屏2026-08-24 10.24.06.png`; manuscript asset `carter_simulation_model.png`; SHA-256 `3d2aaef6704fccc906c7de625de6743e6a4ff5229c7c7fbf4b97344796159710` |
| Two-floor warehouse overview figure | User-generated screenshot `/Users/lab4099/Desktop/截屏2026-08-24 17.17.58.png`; manuscript asset `warehouse_simulation_overview.png`; SHA-256 `ab03755bef8876664740d169aee0fbd1ce982d1363a0256df7947689c245e2a7` |

## Important version cautions

- Current Stage 18 code uses 30 candidate features. The older Stage 14/15 checkpoints and principal Stage 15 JSON files use 28.
- The learned action does **not** independently select a handover point or standby policy. Handover points follow from the chosen stair; standby is deterministic nearest-compatible placement.
- `WAIT` is exposed only when no legal candidate exists.
- “Up to four concurrent tasks” is a capacity, not a claim that four tasks are typically active.
- High-level distances are Euclidean estimates, not SCAN-Planner trajectory lengths or energy measurements.
- Stage 19 `MEDIUM`, `DENSE`, and `BURST` are fixed training-load cohorts, not transport modes and not one mixed-curriculum policy.
- Figure 3 combines an optimization diagnostic with validation evidence; it is not a formal locked test. Panel (a) plots the logged composite objective `actor_loss + 0.5 * value_loss - 0.01 * entropy`, using per-seed medians in non-overlapping 50-update blocks and then ten-seed medians/IQRs. Panel (b) reports ten-seed validation-success medians/IQRs at the matching 50-update cadence.
- PPO loss is not a direct performance metric: the on-policy sample distribution and value targets change during training, and loss magnitude must not rank the three workload profiles. The Stage 18 curves support stable feasible behavior after warm start, not monotonic convergence under every load. Formal tests use preregistered terminal checkpoints rather than retrospectively selected low-loss points.
- The four Stage 19 phases hold task composition fixed and vary only arrival intensity.
- Recovery late-minus-early waiting is a task-level clearance trend, not a control-theoretic settling time.
- The time-greedy adaptation trajectory is descriptive because it has no training-seed dimension.
- No method/result statement should call the implementation MAPPO, hardware validated, or dynamics validated.
- Stage 21 latency excludes environment construction, task execution, ROS communication, navigation, and robot control.
- Stage 21 fault results are logical SMDP injections, not ROS, SCAN-Planner, dynamics, perception, or hardware faults.
- Stage 21 does not show a significant PPO success-rate or throughput advantage over the time-greedy rule; its
  statistically supported advantage is lower waiting and flow time.
