#!/usr/bin/env python3
"""Freeze and verify final DENSE/BURST full and clean-ablation cohorts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np  # Stable Apple Silicon OpenMP load order.
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/warehouse_bringup/config/experiment_seeds_stage18_clean_v4.yaml"
CONDITIONS = {
    "full_context_v2": "FULL_CONTEXT_V2",
    "no_position_only": "NO_POSITION_ONLY",
    "no_eta_cost_only": "NO_ETA_COST_ONLY",
    "no_history_only": "NO_HISTORY_ONLY",
    "no_explicit_queue_resource": "NO_EXPLICIT_QUEUE_RESOURCE",
    "no_explicit_handover_risk": "NO_EXPLICIT_HANDOVER_RISK",
}
REQUIRED = (
    "ppo_parameters_updated", "training_metrics_are_finite",
    "discount_contract_is_valid", "evaluation_resolved_all_tasks",
    "evaluation_actions_all_legal", "evaluation_has_no_resource_leak",
    "one_policy_exposes_all_three_transport_modes",
    "context_conditioned_metrics_cover_every_assignment",
    "rule_baseline_has_no_resource_leak", "concurrent_mode_overlaps_when_requested",
)
SOURCES = (
    ROOT / "src/warehouse_core/warehouse_core/stage12_encoding.py",
    ROOT / "src/warehouse_core/warehouse_core/stage14_ppo.py",
    ROOT / "src/warehouse_core/warehouse_core/stage14_training.py",
    ROOT / "scripts/run_stage14_smdp_ppo.py",
    ROOT / "scripts/run_stage18_train_one.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def freeze(profile: str, cfg: dict) -> tuple[Path, bool]:
    profile_cfg = cfg["profiles"][profile]
    results = ROOT / profile_cfg["training_root"]
    output = ROOT / profile_cfg["freeze_manifest"]
    rows, failures = [], []
    for condition, variant in CONDITIONS.items():
        for seed in range(1, 11):
            run = results / condition / f"seed_{seed:02d}"
            paths = {
                "summary": run / "stage18_ppo.summary.json",
                "history": run / "stage18_ppo.history.json",
                "final_weight": run / "stage18_ppo.pt",
                "best_weight": run / "stage18_ppo.best.pt",
            }
            missing = [str(path) for path in paths.values() if not path.exists()]
            if missing:
                failures.append({"condition": condition, "seed": seed,
                                 "failure": "missing", "paths": missing})
                continue
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            history = json.loads(paths["history"].read_text(encoding="utf-8"))
            checks = {
                "history_is_2000_contiguous_updates": len(history) == 2000 and
                    [row.get("update") for row in history] == list(range(1, 2001)),
                "summary_updates_and_episodes": summary.get("updates") == 2000 and
                    summary.get("episodes_trained_total") == 8000,
                "metadata_matches_condition": summary.get("arrival_profile") == profile and
                    summary.get("observation_variant") == variant,
                "summary_and_history_are_finite": finite(summary) and finite(history),
                "hard_safety_and_training_assertions": all(
                    summary.get("assertions", {}).get(key) is True for key in REQUIRED),
                "checkpoints_load_and_match": True,
            }
            try:
                payloads = [torch.load(paths[key], map_location="cpu", weights_only=False)
                            for key in ("final_weight", "best_weight")]
                checks["checkpoints_load_and_match"] = all(
                    item.get("arrival_profile") == profile and
                    item.get("observation_variant") == variant for item in payloads)
            except Exception as exc:
                checks["checkpoints_load_and_match"] = False
                failures.append({"condition": condition, "seed": seed,
                                 "failure": "checkpoint_load", "detail": repr(exc)})
            if not all(checks.values()):
                failures.append({"condition": condition, "seed": seed,
                                 "failure": "checks", "checks": checks})
            rows.append({
                "condition": condition, "observation_variant": variant,
                "seed_index": seed, "checks": checks,
                **{f"{name}_path": str(path) for name, path in paths.items()},
                **{f"{name}_sha256": sha256(path) for name, path in paths.items()},
            })
    report = {
        "schema_version": "warehouse_stage18_clean_ablation_freeze_v4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "Training-output and provenance freeze only; no test performance is certified.",
        "cohort": {"arrival_profile": profile, "condition_count": 6,
                   "training_seed_count_per_condition": 10,
                   "expected_runs": 60, "actual_runs": len(rows)},
        "hard_gate": {"passed": not failures, "failures": failures,
                      "required_assertions": list(REQUIRED)},
        "source_hash_snapshot_after_training": {
            str(path.relative_to(ROOT)): sha256(path) for path in SOURCES},
        "runs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return output, not failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", choices=("DENSE", "BURST"),
                        default=("DENSE", "BURST"))
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = []
    for profile in args.profiles:
        path, passed = freeze(profile, cfg)
        rows.append({"profile": profile, "manifest": str(path), "passed": passed})
    passed = all(row["passed"] for row in rows)
    print(json.dumps({"stage": 18, "gate": "CLEAN_ABLATION_FREEZE_V4",
                      "profiles": rows, "passed": passed}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
