#!/usr/bin/env python3
"""Plan or run the one-time fresh locked test for clean MEDIUM ablations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np  # Must load before torch on this Apple Silicon Pixi runtime.
import torch
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/warehouse_core"))
from warehouse_core.stage14_ppo import evaluate, load_checkpoint  # noqa: E402

CONFIG = ROOT / "src/warehouse_bringup/config/experiment_seeds_stage18_clean_v3.yaml"
FREEZE = ROOT / "results/stage18_clean_ablation/medium_freeze_manifest.json"
OUT = ROOT / "results/stage18_clean_ablation_locked_test_v3"
STATUS = OUT / "campaign.status.json"
SUMMARY = ROOT / "scripts/summarize_stage18_clean_ablation_v3.py"
CONDITIONS = {
    "full_context_v2": "FULL_CONTEXT_V2",
    "no_position_only": "NO_POSITION_ONLY",
    "no_eta_cost_only": "NO_ETA_COST_ONLY",
    "no_history_only": "NO_HISTORY_ONLY",
    "no_explicit_queue_resource": "NO_EXPLICIT_QUEUE_RESOURCE",
    "no_explicit_handover_risk": "NO_EXPLICIT_HANDOVER_RISK",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight(condition: str, seed: int) -> Path:
    return ROOT / "results/stage18" / condition / f"seed_{seed:02d}" / "stage18_ppo.best.pt"


def output(condition: str, seed: int) -> Path:
    return OUT / condition / f"seed_{seed:02d}.json"


def valid(path: Path, condition: str, seed: int, test_start: int,
          test_count: int, weight_hash: str) -> bool:
    if not path.exists(): return False
    try: row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return False
    return (row.get("schema_version") == "warehouse_stage18_clean_locked_test_v3" and
            row.get("condition") == condition and row.get("training_seed_index") == seed and
            row.get("test_seed_start") == test_start and row.get("test_episode_count") == test_count and
            row.get("weight_sha256") == weight_hash and
            row.get("evaluation_protocol", {}).get("handover_sampling") == "TASK_KEYED_COMMON_RANDOM_NUMBERS" and
            len(row.get("policy", {}).get("episodes", [])) == test_count and
            len(row.get("rule_baseline", {}).get("episodes", [])) == test_count and
            row["policy"].get("illegal_action_count") == 0 and
            row["policy"].get("resource_leak_count") == 0 and
            row["rule_baseline"].get("resource_leak_count") == 0)


def save_status(rows: list[dict], state: str, cfg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "campaign": "CLEAN_ABLATION_MEDIUM_LOCKED_TEST_V3", "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_seed_start": cfg["test"]["seed_start"],
        "test_episode_count": cfg["test"]["seed_count"],
        "handover_sampling": "TASK_KEYED_COMMON_RANDOM_NUMBERS", "jobs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Run the fresh locked test; default is plan only.")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if not frozen.get("hard_gate", {}).get("passed"):
        raise RuntimeError("training freeze gate did not pass")
    test_start, test_count = int(cfg["test"]["seed_start"]), int(cfg["test"]["seed_count"])
    jobs = [(condition, seed) for condition in CONDITIONS for seed in range(1, 11)]
    print(json.dumps({"stage": 18, "campaign": "CLEAN_ABLATION_MEDIUM_LOCKED_TEST_V3",
        "state": "PLAN", "job_count": len(jobs), "policy_episodes": len(jobs)*test_count,
        "rule_episodes": len(jobs)*test_count, "test_seed_range": [test_start, test_start+test_count-1],
        "conditions": list(CONDITIONS), "execute": args.execute}, ensure_ascii=False, indent=2))
    if not args.execute: return 0
    seeds = list(range(test_start, test_start + test_count)); rows=[]
    device=torch.device("cpu"); torch.set_num_threads(4); save_status(rows,"RUNNING",cfg)
    progress=tqdm(jobs,desc="Stage18 clean v3 TASK_KEYED",unit="job",dynamic_ncols=True)
    for condition, seed in progress:
        checkpoint=weight(condition,seed); hash_value=digest(checkpoint); target=output(condition,seed)
        if valid(target,condition,seed,test_start,test_count,hash_value):
            rows.append({"condition":condition,"training_seed_index":seed,"state":"SKIPPED_ALREADY_COMPLETE"}); save_status(rows,"RUNNING",cfg); continue
        model,payload=load_checkpoint(checkpoint,device)
        if payload.get("arrival_profile") != "MEDIUM" or payload.get("observation_variant") != CONDITIONS[condition]:
            raise ValueError(f"checkpoint metadata mismatch: {checkpoint}")
        policy=evaluate(model,seeds,"CONCURRENT","MEDIUM",device,observation_variant=CONDITIONS[condition],handover_sampling="TASK_KEYED")
        rule=evaluate(model,seeds,"CONCURRENT","MEDIUM",device,use_rule=True,observation_variant=CONDITIONS[condition],handover_sampling="TASK_KEYED")
        row={"schema_version":"warehouse_stage18_clean_locked_test_v3",
             "claim_boundary":"Fresh v3 test after frozen MEDIUM clean-ablation cohort.",
             "training_profile":"MEDIUM","evaluation_profile":"MEDIUM","condition":condition,"training_seed_index":seed,
             "weight":str(checkpoint),"weight_sha256":hash_value,"checkpoint_update":payload.get("update"),
             "test_seed_start":test_start,"test_episode_count":test_count,
             "evaluation_protocol":{"handover_sampling":"TASK_KEYED_COMMON_RANDOM_NUMBERS","pairing_key":"(test_seed, task_id, handover_kind)","randomness_pairing":"shared_across_policy_and_baselines","result_version":"clean_v3_non_overwriting"},
             "policy":policy,"rule_baseline":rule,
             "safety":{"policy_actions_all_legal":policy["illegal_action_count"]==0,"policy_has_no_resource_leak":policy["resource_leak_count"]==0,"rule_has_no_resource_leak":rule["resource_leak_count"]==0}}
        target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        if not valid(target,condition,seed,test_start,test_count,hash_value): raise RuntimeError(f"result validation failed: {target}")
        rows.append({"condition":condition,"training_seed_index":seed,"state":"COMPLETED","output":str(target)}); save_status(rows,"RUNNING",cfg)
    save_status(rows,"COMPLETED",cfg); subprocess.run([sys.executable,str(SUMMARY)],cwd=ROOT,check=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
