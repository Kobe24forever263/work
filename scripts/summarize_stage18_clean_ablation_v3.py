#!/usr/bin/env python3
"""Summarize frozen MEDIUM clean-ablation tests with crossed pairing."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage18_clean_ablation_locked_test_v3"
CONFIG = ROOT / "src/warehouse_bringup/config/experiment_seeds_stage18_clean_v3.yaml"
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_stage18_locked_test_v2 import (  # noqa: E402
    METRICS, add_multiplicity, paired_stats)

CONDITIONS = {
    "full_context_v2": "FULL_CONTEXT_V2",
    "no_position_only": "NO_POSITION_ONLY",
    "no_eta_cost_only": "NO_ETA_COST_ONLY",
    "no_history_only": "NO_HISTORY_ONLY",
    "no_explicit_queue_resource": "NO_EXPLICIT_QUEUE_RESOURCE",
    "no_explicit_handover_risk": "NO_EXPLICIT_HANDOVER_RISK",
}


def load(condition: str) -> list[dict]:
    rows = []
    for seed in range(1, 11):
        path = OUT / condition / f"seed_{seed:02d}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != "warehouse_stage18_clean_locked_test_v3":
            raise ValueError(f"unexpected result schema: {path}")
        if row.get("evaluation_protocol", {}).get("handover_sampling") != \
                "TASK_KEYED_COMMON_RANDOM_NUMBERS":
            raise ValueError(f"unpaired handover timing: {path}")
        rows.append(row)
    return rows


def family(comparison: str, metric: str) -> str:
    if metric == "resolved_throughput_tasks_per_hour":
        return "diagnostic"
    if comparison == "full_context_v2_minus_rule_baseline":
        return "primary_full_vs_rule"
    return "primary_clean_feature_contribution"


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed_start"]))
    samples = int(cfg["statistics"]["bootstrap_samples"])
    full = load("full_context_v2")
    rows = []
    for metric in METRICS:
        rows.append({
            "profile": "MEDIUM", "comparison": "full_context_v2_minus_rule_baseline",
            "metric": metric,
            "hypothesis_family": family("full_context_v2_minus_rule_baseline", metric),
            **paired_stats(full, "policy", full, "rule_baseline", metric, rng, samples),
        })
    for condition in tuple(CONDITIONS)[1:]:
        ablation = load(condition)
        comparison = f"full_context_v2_minus_{condition}"
        for metric in METRICS:
            rows.append({
                "profile": "MEDIUM", "comparison": comparison, "metric": metric,
                "hypothesis_family": family(comparison, metric),
                **paired_stats(full, "policy", ablation, "policy", metric, rng, samples),
            })
    add_multiplicity(rows)
    report = {
        "schema_version": "warehouse_stage18_clean_statistics_v3",
        "claim_boundary": (
            "Fresh MEDIUM-only locked test after the frozen clean-ablation "
            "training cohort. No model selection may use these seeds."),
        "test_seed_start": int(cfg["test"]["seed_start"]),
        "test_seed_count": int(cfg["test"]["seed_count"]),
        "bootstrap_seed": int(cfg["statistics"]["bootstrap_seed_start"]),
        "bootstrap_samples": samples,
        "method": {
            "randomness_pairing": "TASK_KEYED common random numbers",
            "uncertainty": "crossed bootstrap over 10 training seeds and 100 shared test seeds",
            "throughput": {"successful": "completed/total simulated time*3600",
                           "resolved": "(completed+failed)/total simulated time*3600"},
            "multiplicity": "Holm within predeclared families and globally; BH-FDR reported exploratorily",
        },
        "comparisons": rows,
    }
    output = OUT / "stage18_clean_ablation_locked_test_v3_statistics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    fields = ["comparison", "metric", "hypothesis_family", "left_estimate",
              "right_estimate", "mean_delta", "ci_low", "ci_high",
              "exact_two_sided_sign_flip_p", "family_holm_adjusted_p",
              "global_holm_adjusted_p", "positive_training_seed_count"]
    with (OUT / "stage18_clean_ablation_locked_test_v3_statistics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            low, high = row["crossed_bootstrap_95ci"]
            writer.writerow({**{key: row.get(key) for key in fields},
                             "ci_low": low, "ci_high": high})
    print(json.dumps({"output": str(output), "comparison_count": len(rows)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
