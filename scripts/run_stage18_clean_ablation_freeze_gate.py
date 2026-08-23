#!/usr/bin/env python3
"""Freeze and verify the 50-run MEDIUM clean-ablation training cohort."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np  # Ensures the macOS OpenMP runtime load order is stable.
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "stage18"
OUT = ROOT / "results" / "stage18_clean_ablation" / "medium_freeze_manifest.json"
CONDITIONS = {
    "no_position_only": "NO_POSITION_ONLY",
    "no_eta_cost_only": "NO_ETA_COST_ONLY",
    "no_history_only": "NO_HISTORY_ONLY",
    "no_explicit_queue_resource": "NO_EXPLICIT_QUEUE_RESOURCE",
    "no_explicit_handover_risk": "NO_EXPLICIT_HANDOVER_RISK",
}
REQUIRED_ASSERTIONS = (
    "ppo_parameters_updated", "training_metrics_are_finite",
    "discount_contract_is_valid", "evaluation_resolved_all_tasks",
    "evaluation_actions_all_legal", "evaluation_has_no_resource_leak",
    "one_policy_exposes_all_three_transport_modes",
    "context_conditioned_metrics_cover_every_assignment",
    "rule_baseline_has_no_resource_leak", "concurrent_mode_overlaps_when_requested",
)
SOURCE_FILES = (
    ROOT / "src/warehouse_core/warehouse_core/stage12_encoding.py",
    ROOT / "src/warehouse_core/warehouse_core/stage14_ppo.py",
    ROOT / "src/warehouse_core/warehouse_core/stage14_training.py",
    ROOT / "scripts/run_stage14_smdp_ppo.py",
    ROOT / "scripts/run_stage14_policy_smoke.py",
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


def main() -> int:
    rows, failures = [], []
    for condition, variant in CONDITIONS.items():
        for seed in range(1, 11):
            run = RESULTS / condition / f"seed_{seed:02d}"
            summary_path = run / "stage18_ppo.summary.json"
            history_path = run / "stage18_ppo.history.json"
            final_path = run / "stage18_ppo.pt"
            best_path = run / "stage18_ppo.best.pt"
            missing = [str(path) for path in (
                summary_path, history_path, final_path, best_path)
                if not path.exists()]
            if missing:
                failures.append({"condition": condition, "seed": seed,
                                 "failure": "missing", "paths": missing})
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            history = json.loads(history_path.read_text(encoding="utf-8"))
            checks = {
                "history_is_2000_contiguous_updates": (
                    len(history) == 2000 and
                    [row.get("update") for row in history] == list(range(1, 2001))),
                "summary_updates_and_episodes": (
                    summary.get("updates") == 2000 and
                    summary.get("episodes_trained_total") == 8000),
                "metadata_matches_condition": (
                    summary.get("arrival_profile") == "MEDIUM" and
                    summary.get("observation_variant") == variant),
                "summary_and_history_are_finite": finite(summary) and finite(history),
                "hard_safety_and_training_assertions": all(
                    summary.get("assertions", {}).get(key) is True
                    for key in REQUIRED_ASSERTIONS),
                "final_and_best_checkpoint_load": True,
            }
            try:
                final = torch.load(final_path, map_location="cpu", weights_only=False)
                best = torch.load(best_path, map_location="cpu", weights_only=False)
                checks["checkpoint_metadata_matches"] = (
                    final.get("arrival_profile") == "MEDIUM" and
                    best.get("arrival_profile") == "MEDIUM" and
                    final.get("observation_variant") == variant and
                    best.get("observation_variant") == variant)
            except Exception as exc:  # Preserve the evidence in the manifest.
                checks["final_and_best_checkpoint_load"] = False
                checks["checkpoint_metadata_matches"] = False
                failures.append({"condition": condition, "seed": seed,
                                 "failure": "checkpoint_load", "detail": repr(exc)})
            if not all(checks.values()):
                failures.append({"condition": condition, "seed": seed,
                                 "failure": "checks", "checks": checks})
            rows.append({
                "condition": condition, "observation_variant": variant,
                "seed_index": seed, "summary": str(summary_path),
                "history": str(history_path), "final_weight": str(final_path),
                "best_weight": str(best_path), "summary_sha256": sha256(summary_path),
                "history_sha256": sha256(history_path),
                "final_weight_sha256": sha256(final_path),
                "best_weight_sha256": sha256(best_path), "checks": checks,
                "summary_passed": summary.get("passed"),
                "context_mode_reference_gate": summary.get(
                    "contextual_mode_acceptance", {}).get("formal_gate_passed"),
            })
    report = {
        "schema_version": "warehouse_stage18_clean_ablation_medium_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Training-output freeze only. It does not use or certify final "
            "test performance."),
        "cohort": {"arrival_profile": "MEDIUM", "condition_count": 5,
                   "training_seed_count_per_condition": 10,
                   "expected_runs": 50, "actual_runs": len(rows)},
        "hard_gate": {"required_assertions": list(REQUIRED_ASSERTIONS),
                      "passed": not failures, "failures": failures},
        "semantic_note": (
            "context_mode_reference_gate is retained as an outcome, not a hard "
            "training-integrity gate; it is expected to fail for NO_ETA_COST_ONLY."),
        "source_hash_snapshot_after_training": {
            str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_FILES},
        "runs": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"output": str(OUT), "run_count": len(rows),
                      "failure_count": len(failures), "passed": not failures},
                     ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
