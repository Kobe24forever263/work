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
- locked Stage 19 mixed-load protocol and data-integrity statement;
- absolute performance table;
- paired-difference and load-adaptation/recovery figures;
- limitations and scope-bounded conclusion.

Intentionally not written as findings:

- uncorrected confirmatory comparisons among the three fixed-load PPO cohorts;
- unfinished or unstable context and algorithmic ablations;
- learning/optimization baseline comparisons;
- closed-loop navigation, physical handover timing, or hardware evidence.

The Results section uses only the fresh Stage 19 locked test. Training-profile comparisons remain exploratory; the main comparisons are PPO versus the time-greedy rule on shared test streams.

## Version boundary

The draft describes the current 284-state-feature, 30-feature-per-candidate implementation. The inserted results come from 30 frozen FULL_CONTEXT_V2 policies evaluated in Stage 19. Earlier Stage 14/15 results used a 28-D candidate representation and must not be reported as results of this 30-D model.

## Reproducing the Stage 19 figures

Run `scripts/make_stage19_figures.py`. It reads the frozen summary and the per-training-seed locked-test JSON files from the workspace, reconstructs all plotted point estimates, checks them against the summary, writes the plotting data to `figures/data/`, and regenerates both vector PDFs. `figures/data/stage19_figure_validation.json` records the source paths, bootstrap design, and validation status.

draw.io Desktop is installed separately for editable system diagrams and flowcharts. Statistical result figures remain script-generated so numerical updates are reproducible.

## Anonymity

The initial-review source contains no author, affiliation, funding, acknowledgment, biography, institutional link, or identifying PDF metadata. Add those only after acceptance. Before submission, also inspect figures, videos, repository links, and generated PDF metadata.

## Bibliography and Zotero

`references.bib` is the source-verified bibliography snapshot used by Overleaf. Zotero remains useful for reading PDFs and correcting metadata. If entries are edited in Zotero, export the updated records as BibTeX and merge them into `references.bib`; avoid importing the whole library blindly because that can create duplicate keys.

## Journal rules

See `TRO_FORMAT_REQUIREMENTS.md`. The initial regular-paper PDF may be at most 18 pages, including references. A practical no-overlength-charge target for the final accepted paper is 12 pages.
