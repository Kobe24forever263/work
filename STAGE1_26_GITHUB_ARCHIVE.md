# Stage 1–26 GitHub archive

Archive date: 2026-09-04

This repository snapshot contains the source code, ROS 2 launch/configuration
files, experiment protocols, handover notes, aggregated tables/figures, raw
JSON/CSV histories, locked-test outputs, and the final long-run policy weight
for each completed Stage 23–26 training seed. Earlier Stage 1–22 evidence that
was already committed remains unchanged.

To keep the Git repository usable, the snapshot intentionally excludes:

- periodic intermediate files under `checkpoints/`;
- temporary RViz/SCAN runtime logs;
- six bulk flattened export tables (`training_history_every20.csv`,
  `normalized_metric_blocks.csv`, `json_scalar_metrics.csv`,
  `transport_mode_counts.csv`, `file_manifest.csv`, and
  `existing_csv_long.csv`) that are reproducibly generated from the committed
  source results;
- build, install, Python cache, and operating-system metadata;
- Stage 27 source code, protocols, results, and notes, which are still under
  active experimental review.

The excluded periodic checkpoints are redundant with the committed final
weights and histories. They remain in the local workspace and can be archived
separately if binary storage or Git LFS is introduced later.

Key Stage 23–26 locations:

- `results/stage23_markov_continuous/`
- `results/stage24_rolling_optimizer/`
- `results/stage25_causal_online/`
- `results/stage26_recurrent_causal/`
- `exports/2026-08-31_stage1-25_experiment_package/`
- `src/warehouse_bringup/config/experiment_stage23_locked_test.yaml`
- `src/warehouse_bringup/config/experiment_stage24_rolling_optimizer.yaml`
- `src/warehouse_bringup/config/experiment_stage25_causal_online.yaml`
- `src/warehouse_bringup/config/experiment_stage26_recurrent_causal.yaml`
