#!/usr/bin/env python3
"""Evaluate ten paired Stage 18 best checkpoints on shared validation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np  # noqa: F401
import torch


WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK_ROOT / "src" / "warehouse_core"))

from warehouse_core.stage14_ppo import evaluate, load_checkpoint


METRICS = ("mean_reward", "success_rate", "throughput_tasks_per_hour",
           "mean_simulated_time", "cost_reference_mode_agreement")


def interval(values: list[float], seed: int, count: int = 20000) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    means = sorted(statistics.mean(values[rng.randrange(size)]
                                   for _ in range(size))
                   for _ in range(count))
    return [means[int(.025 * count)], means[int(.975 * count)]]


def run(condition: str, seed_index: int, profile_root: str,
        validation_seed: int, episodes: int) -> dict:
    weight = (WORK_ROOT / "results" / profile_root / condition /
              f"seed_{seed_index:02d}" / "stage18_ppo.best.pt")
    model, checkpoint = load_checkpoint(weight, torch.device("cpu"))
    seeds = list(range(validation_seed, validation_seed + episodes))
    variant = checkpoint.get("observation_variant", "FULL_CONTEXT_V2")
    result = evaluate(
        model, seeds, checkpoint["execution_mode"],
        checkpoint["arrival_profile"], torch.device("cpu"),
        observation_variant=variant)
    return {
        "seed_index": seed_index,
        "weight": str(weight),
        "checkpoint_update": checkpoint["update"],
        "observation_variant": variant,
        "illegal_action_count": result["illegal_action_count"],
        "resource_leak_count": result["resource_leak_count"],
        "transport_modes": result["transport_modes"],
        **{metric: result[metric] for metric in METRICS},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("comparison")
    parser.add_argument("--arrival-profile", choices=("MEDIUM", "DENSE"),
                        default="MEDIUM")
    parser.add_argument("--validation-seed", type=int, default=42000000)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    root = "stage18" if args.arrival_profile == "MEDIUM" else "stage18_dense"
    reference = [run(args.reference, seed, root, args.validation_seed,
                     args.episodes) for seed in range(1, 11)]
    comparison = [run(args.comparison, seed, root, args.validation_seed,
                      args.episodes) for seed in range(1, 11)]
    paired = {}
    for offset, metric in enumerate(METRICS):
        values = [reference[i][metric] - comparison[i][metric]
                  for i in range(10)]
        bounds = interval(values, 44000200 + offset)
        paired[metric] = {
            "direction": "reference_minus_comparison",
            "per_seed": values,
            "mean": statistics.mean(values),
            "sample_stdev": statistics.stdev(values),
            "bootstrap_95_interval": bounds,
            "interval_excludes_zero": bounds[0] > 0 or bounds[1] < 0,
        }
    report = {
        "schema_version": "warehouse_stage18_best_checkpoint_paired_v1",
        "claim_boundary": (
            "Best-checkpoint confirmation on checkpoint-selection validation "
            "seeds; not locked test-set evidence."),
        "arrival_profile": args.arrival_profile,
        "reference_condition": args.reference,
        "comparison_condition": args.comparison,
        "validation_seed_start": args.validation_seed,
        "validation_episode_count": args.episodes,
        "reference": reference, "comparison": comparison, "paired": paired,
        "quality": {
            "illegal_action_count": sum(
                row["illegal_action_count"] for row in reference + comparison),
            "resource_leak_count": sum(
                row["resource_leak_count"] for row in reference + comparison),
            "all_variants_match_condition": all(
                row["observation_variant"] == "FULL_CONTEXT_V2"
                for row in reference) and all(
                    row["observation_variant"] == "NO_HANDOVER_CUES"
                    for row in comparison),
        },
    }
    output = (WORK_ROOT / "results" / root /
              f"paired_best_{args.reference}_vs_{args.comparison}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "quality": report["quality"],
                      "paired": paired}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
