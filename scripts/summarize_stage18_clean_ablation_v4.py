#!/usr/bin/env python3
"""Summarize DENSE/BURST clean-ablation v4 locked tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage18_clean_ablation_locked_test_v4"
CONFIG = ROOT / "src/warehouse_bringup/config/experiment_seeds_stage18_clean_v4.yaml"
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_stage18_locked_test_v2 import (  # noqa: E402
    METRICS, add_multiplicity, paired_stats)

PROFILES = ("DENSE", "BURST")
ABLATIONS = (
    "no_position_only", "no_eta_cost_only", "no_history_only",
    "no_explicit_queue_resource", "no_explicit_handover_risk",
)


def load(profile: str, condition: str) -> list[dict]:
    rows = []
    for seed in range(1, 11):
        path = OUT / profile.lower() / condition / f"seed_{seed:02d}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != "warehouse_stage18_clean_locked_test_v4":
            raise ValueError(f"unexpected schema: {path}")
        if row.get("evaluation_protocol", {}).get("handover_sampling") != \
                "TASK_KEYED_COMMON_RANDOM_NUMBERS":
            raise ValueError(f"unpaired handover timing: {path}")
        rows.append(row)
    return rows


def load_single_dog(profile: str) -> list[dict]:
    path = OUT / profile.lower() / "baselines/single_dog_only.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("schema_version") != "warehouse_stage18_single_dog_baseline_v4":
        raise ValueError(f"unexpected single-dog schema: {path}")
    result = row["single_dog_only"]
    if (result.get("allowed_transport_modes") != ["SINGLE_DOG"] or
            result.get("illegal_action_count") != 0 or
            result.get("resource_leak_count") != 0):
        raise ValueError(f"invalid single-dog baseline: {path}")
    return [{"single_dog_only": result} for _ in range(10)]


def family(comparison: str, metric: str) -> str:
    if metric == "resolved_throughput_tasks_per_hour":
        return "diagnostic"
    if comparison == "full_context_v2_minus_rule_baseline":
        return "primary_full_vs_rule"
    if comparison == "full_context_v2_minus_single_dog_only":
        return "primary_multiagent_vs_single_dog"
    return "primary_clean_feature_contribution"


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = []
    for profile in PROFILES:
        rng = np.random.default_rng(
            int(cfg["profiles"][profile]["bootstrap_seed_start"]))
        samples = int(cfg["statistics"]["bootstrap_samples"])
        full = load(profile, "full_context_v2")
        dog = load_single_dog(profile)
        for comparison, right, right_side in (
                ("full_context_v2_minus_rule_baseline", full, "rule_baseline"),
                ("full_context_v2_minus_single_dog_only", dog, "single_dog_only")):
            for metric in METRICS:
                rows.append({"profile": profile, "comparison": comparison,
                             "metric": metric, "hypothesis_family": family(comparison, metric),
                             **paired_stats(full, "policy", right, right_side,
                                            metric, rng, samples)})
        for condition in ABLATIONS:
            ablation = load(profile, condition)
            comparison = f"full_context_v2_minus_{condition}"
            for metric in METRICS:
                rows.append({"profile": profile, "comparison": comparison,
                             "metric": metric, "hypothesis_family": family(comparison, metric),
                             **paired_stats(full, "policy", ablation, "policy",
                                            metric, rng, samples)})
    add_multiplicity(rows)
    report = {
        "schema_version": "warehouse_stage18_clean_statistics_v4",
        "claim_boundary": "Fresh DENSE/BURST locked tests after frozen cohorts; no model selection may use these seeds.",
        "profiles": {profile: {
            "test_seed_start": int(cfg["profiles"][profile]["test_seed_start"]),
            "test_seed_count": int(cfg["test"]["seed_count"]),
            "bootstrap_seed": int(cfg["profiles"][profile]["bootstrap_seed_start"]),
        } for profile in PROFILES},
        "bootstrap_samples": int(cfg["statistics"]["bootstrap_samples"]),
        "method": {
            "randomness_pairing": "TASK_KEYED common random numbers",
            "uncertainty": "crossed bootstrap over 10 training seeds and 100 shared test seeds",
            "successful_throughput": "completed/total simulated time*3600",
            "multiplicity": "Holm within predeclared families and globally; BH-FDR exploratory",
        },
        "comparisons": rows,
    }
    output = OUT / "stage18_clean_ablation_locked_test_v4_statistics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    fields = ["profile", "comparison", "metric", "hypothesis_family",
              "left_estimate", "right_estimate", "mean_delta", "ci_low", "ci_high",
              "exact_two_sided_sign_flip_p", "family_holm_adjusted_p",
              "global_holm_adjusted_p", "positive_training_seed_count"]
    with (OUT / "stage18_clean_ablation_locked_test_v4_statistics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            low, high = row["crossed_bootstrap_95ci"]
            writer.writerow({key: (low if key == "ci_low" else high if key == "ci_high"
                                    else row.get(key)) for key in fields})
    print(json.dumps({"output": str(output), "comparison_count": len(rows)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
