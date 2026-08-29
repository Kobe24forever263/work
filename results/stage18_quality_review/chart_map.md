# Chart map

| Section | Question | Family/type | Fields | Supported claim |
|---|---|---|---|---|
| Full vs rule | Does reward improve by load? | Comparison/bar | profile, mean_delta | Observed reward advantage grows with load |
| Throughput definition | Does completed-only productivity improve? | Comparison/bar | profile, mean_delta | Successful throughput differs from resolved-task rate |
| Ablation | Which ablation consistently degrades? | Grouped comparison/bar | profile, condition, mean_delta | Only the persistent-context bundle degrades consistently |
| Cross-load | How does a trained profile transfer? | Comparison/bar | route, reward_delta, training_profile | Cross-load files exist but need formal statistics |

Palette policy: single-root for single-series charts; relaxed multi-category for the ablation and cross-load grouped charts. All bars use zero-reference context and exact values remain available in tables/tooltips.
