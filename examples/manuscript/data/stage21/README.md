# Stage 21 frozen plotting data

This directory contains the GitHub-sized, paper-facing data products for Stage 21. The authoritative sources are
the frozen JSON files; CSV files are direct format-only projections for Origin, MATLAB, R, or Python.

Included evidence:

- 54 latency/scaling scenarios with 500 raw timing samples per component;
- formal fault-robustness summary from 10 PPO training seeds and 100 shared locked-test seeds;
- protocol freeze manifest and campaign completion records;
- four chart-ready CSV tables.

The 494 MB per-episode/per-task campaign directory remains under the repository-ignored `results/` tree and is not
duplicated here. It can be regenerated with `scripts/run_stage21_fault_locked.command` after verifying the frozen
manifest. Excluding the raw campaign does not remove the point estimates, confidence intervals, seed counts,
fingerprints, assertions, or plotting inputs used by the paper.

See `examples/manuscript/STAGE21_PLOTTING_GUIDE.md` for field definitions, chart contracts, Origin instructions,
visual style, and claim boundaries.
