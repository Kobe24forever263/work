#!/usr/bin/env python3
"""Freeze the 30 already-trained full-context policies before Stage 19."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage19_mixed_recovery/stage19_full_policy_freeze_manifest.json"
ROOTS = {
    "MEDIUM": ROOT / "results/stage18",
    "DENSE": ROOT / "results/stage18_dense",
    "BURST": ROOT / "results/stage18_burst",
}
REQUIRED_ASSERTIONS = (
    "ppo_parameters_updated",
    "training_metrics_are_finite",
    "discount_contract_is_valid",
    "evaluation_resolved_all_tasks",
    "evaluation_actions_all_legal",
    "evaluation_has_no_resource_leak",
    "one_policy_exposes_all_three_transport_modes",
    "context_conditioned_metrics_cover_every_assignment",
    "rule_baseline_has_no_resource_leak",
    "concurrent_mode_overlaps_when_requested",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    rows = []
    failures = []
    for profile, training_root in ROOTS.items():
        for seed in range(1, 11):
            directory = training_root / "full_context_v2" / f"seed_{seed:02d}"
            paths = {
                "summary": directory / "stage18_ppo.summary.json",
                "history": directory / "stage18_ppo.history.json",
                "final_weight": directory / "stage18_ppo.pt",
                "best_weight": directory / "stage18_ppo.best.pt",
            }
            missing = [name for name, path in paths.items() if not path.exists()]
            if missing:
                failures.append(f"{profile}/seed_{seed:02d}: missing {missing}")
                continue
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            checks = {
                "training_passed": summary.get("passed") is True,
                "profile_matches": summary.get("arrival_profile") == profile,
                "full_context_v2": summary.get("observation_variant") == "FULL_CONTEXT_V2",
                "updates_complete": int(summary.get("updates", 0)) == 2000,
                "episodes_complete": int(summary.get("episodes_trained_total", 0)) >= 8000,
                "required_assertions": all(
                    summary.get("assertions", {}).get(name) is True
                    for name in REQUIRED_ASSERTIONS),
            }
            if not all(checks.values()):
                failures.append(f"{profile}/seed_{seed:02d}: {checks}")
            rows.append({
                "training_profile": profile,
                "training_seed_index": seed,
                "checks": checks,
                **{f"{name}_path": str(path) for name, path in paths.items()},
                **{f"{name}_sha256": sha256(path) for name, path in paths.items()},
            })
    payload = {
        "schema_version": "warehouse_stage19_full_policy_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Training-output and provenance freeze only; no mixed-load test "
            "performance is certified."),
        "cohort": {
            "profiles": list(ROOTS),
            "training_seed_count_per_profile": 10,
            "expected_runs": 30,
            "actual_runs": len(rows),
        },
        "hard_gate": {"passed": not failures and len(rows) == 30,
                      "failures": failures},
        "runs": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"output": str(OUT), "runs": len(rows),
                      "passed": payload["hard_gate"]["passed"],
                      "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if payload["hard_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
