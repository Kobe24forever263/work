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
| Stage 19 protocol, claim boundary, and handover | `/Users/lab4099/Desktop/Mujoco/work/0823handover.md` |
| Stage 19 formal summary and crossed-bootstrap intervals | `/Users/lab4099/Desktop/Mujoco/work/results/stage19_mixed_recovery/stage19_mixed_recovery_summary.json` |
| Stage 19 per-training-seed and per-task locked-test records | `/Users/lab4099/Desktop/Mujoco/work/results/stage19_mixed_recovery/locked_test_v1/` |
| Stage 19 formal data-quality audit | `/Users/lab4099/Desktop/Mujoco/work/results/stage19_mixed_recovery/stage19_quality_audit.ipynb` |
| Stage 19 figure generator and raw-to-summary checks | `scripts/make_stage19_figures.py` |

## Important version cautions

- Current Stage 18 code uses 30 candidate features. The older Stage 14/15 checkpoints and principal Stage 15 JSON files use 28.
- The learned action does **not** independently select a handover point or standby policy. Handover points follow from the chosen stair; standby is deterministic nearest-compatible placement.
- `WAIT` is exposed only when no legal candidate exists.
- “Up to four concurrent tasks” is a capacity, not a claim that four tasks are typically active.
- High-level distances are Euclidean estimates, not SCAN-Planner trajectory lengths or energy measurements.
- Stage 19 `MEDIUM`, `DENSE`, and `BURST` are fixed training-load cohorts, not transport modes and not one mixed-curriculum policy.
- The four Stage 19 phases hold task composition fixed and vary only arrival intensity.
- Recovery late-minus-early waiting is a task-level clearance trend, not a control-theoretic settling time.
- The time-greedy adaptation trajectory is descriptive because it has no training-seed dimension.
- No method/result statement should call the implementation MAPPO, hardware validated, or dynamics validated.
