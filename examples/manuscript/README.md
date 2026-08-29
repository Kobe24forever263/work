# T-RO Overleaf draft

This is an anonymous, editable draft for an IEEE Transactions on Robotics regular-paper submission.

## Overleaf

Upload `TRO_Overleaf_Draft.zip` with **New Project → Upload Project**. The main document is `main.tex`. The project uses the current official `IEEEtran.cls` supplied by the IEEE Transactions LaTeX package.

## Current writing status

Written now:

- abstract and four index terms;
- introduction and contribution statement;
- related work;
- system and task definition;
- event-driven SMDP formulation;
- 284-D state and dynamic `1536 × 30` candidate encoding;
- deterministic feasibility mask;
- context–candidate actor–critic;
- Reward V2 and duration-aware PPO;
- explicit 284-D state and 30-D candidate feature-breakdown table;
- frozen Stage 18 fresh locked feature-ablation table with multiplicity boundary;
- locked Stage 19 mixed-load protocol and data-integrity statement;
- absolute performance table;
- paired-difference and load-adaptation/recovery figures;
- frozen Stage 20 mixed-curriculum generalization table and crossed-bootstrap figure;
- frozen Stage 21 fixed-shape CPU-latency benchmark;
- frozen Stage 21 logical-fault comparison with two rule controls;
- crossed-bootstrap forest plot for mean and P95 waiting-time effects;
- limitations and scope-bounded conclusion.

Intentionally not written as findings:

- uncorrected confirmatory comparisons among the three fixed-load PPO cohorts;
- unstable position/history/queue-resource/handover-risk feature effects or unfinished algorithmic ablations;
- learning/optimization baseline comparisons beyond the two rule controls;
- closed-loop navigation, physical handover timing, or hardware evidence.

The Results section uses the frozen Stage 19--21 locked tests. Training-profile comparisons remain exploratory. Stage 20 tests ten mixed-curriculum policies on randomized persistent load; Stage 21 evaluates the same frozen cohort against time-greedy and single-dog controls on 100 shared test seeds in four logical scenarios.

## Version boundary

The draft describes the current 284-state-feature, 30-feature-per-candidate implementation. Stage 19 uses 30 frozen fixed-load FULL_CONTEXT_V2 policies; Stage 21 uses ten frozen Stage 20 mixed-curriculum policies. Earlier Stage 14/15 results used a 28-D candidate representation and must not be reported as results of this 30-D model.

The paper-package copy of the authoritative Stage 20 locked-test summary and policy-freeze manifest is under `figures/data/stage20/`; the GitHub manuscript tree keeps the same files under `data/stage20/`. Their SHA-256 digests are `646b34ae...689eb12` and `81b47e9e...9b913f6`, respectively; both formal gates pass.

## Reproducing the Stage 19 figures

Run `scripts/make_stage19_figures.py`. It reads the frozen summary and the per-training-seed locked-test JSON files from the workspace, reconstructs all plotted point estimates, checks them against the summary, writes the plotting data to `figures/data/`, and regenerates both vector PDFs. `figures/data/stage19_figure_validation.json` records the source paths, bootstrap design, and validation status.

draw.io Desktop is installed separately for editable system diagrams and flowcharts. Statistical result figures remain script-generated so numerical updates are reproducible.

## Reproducing the Stage 20 figure

Run `scripts/make_stage20_figure.py` with the bundled Python environment. It reads the frozen `data/stage20/stage20_locked_test_summary.json`, validates the crossed-bootstrap design and test dimensions, and regenerates `figures/fig_stage20_generalization.pdf`. The plotted values are also written to `figures/data/stage20_generalization_differences.csv`; `data/stage20/stage20_figure_validation.json` records the source SHA-256 and plotted-point count. The figure uses a common direction in all four panels: positive values favor mixed-curriculum PPO. The single-dog control remains in the exact-value manuscript table because its much larger waiting-time differences would compress the specialist comparisons.

## Reproducing the Stage 21 figures

Run `scripts/make_stage21_figures.py` with the bundled Python environment. It verifies the nine-file SHA-256 manifest, the 40 PPO and eight baseline job records, campaign and freeze gates, every CSV value against the authoritative JSON, all confidence-interval directions, and the complete 54-scenario latency grid. It then regenerates `figures/fig_stage21_latency_scaling.pdf` and `figures/fig_stage21_fault_waiting_effects.pdf`. The paper-package audit is `figures/data/stage21/stage21_figure_validation.json`; the GitHub manuscript copy is `data/stage21/stage21_figure_validation.json`.

Stage 21 faults are injected only in the logical SMDP. They are not evidence of ROS, SCAN-Planner, perception, dynamics, or hardware fault tolerance. The latency plateau beyond eight waiting tasks is a consequence of the top-eight visibility boundary, not lossless scaling beyond the trained shape.

## Anonymity

The initial-review source contains no author, affiliation, funding, acknowledgment, biography, institutional link, or identifying PDF metadata. Add those only after acceptance. Before submission, also inspect figures, videos, repository links, and generated PDF metadata.

## Bibliography and Zotero

`references.bib` is the source-verified bibliography snapshot used by Overleaf. Zotero remains useful for reading PDFs and correcting metadata. If entries are edited in Zotero, export the updated records as BibTeX and merge them into `references.bib`; avoid importing the whole library blindly because that can create duplicate keys.

## Journal rules

See `TRO_FORMAT_REQUIREMENTS.md`. The initial regular-paper PDF may be at most 18 pages, including references. A practical no-overlength-charge target for the final accepted paper is 12 pages.
