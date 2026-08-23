#!/usr/bin/env python3
"""Freeze the T-RO seed protocol and create a reproducibility SHA256 manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
import yaml


WORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (
    WORK_ROOT / "src" / "warehouse_bringup" / "config" /
    "experiment_seeds.yaml")


def integer_range(start: int, count: int) -> set[int]:
    return set(range(int(start), int(start) + int(count)))


def validate_seed_protocol(config: dict) -> tuple[dict, dict[str, set[int]]]:
    train = config["training"]
    episode_count = int(train["episodes_per_run"])
    warm_count = int(train["warm_start_seed_count"])
    warm_offset = int(train["warm_start_seed_offset"])
    ppo_offset = int(train["ppo_seed_offset"])
    train_ranges = {}
    within_run_disjoint = True
    for index, seed in enumerate(train["independent_run_base_seeds"]):
        warm = integer_range(seed + warm_offset, warm_count)
        ppo = integer_range(seed + ppo_offset, episode_count)
        within_run_disjoint &= not bool(warm & ppo)
        train_ranges[f"training_run_{index + 1}"] = warm | ppo
    namespaces = {
        **train_ranges,
        "validation": integer_range(
            config["validation"]["seed_start"],
            config["validation"]["seed_count"]),
        "test": integer_range(
            config["test"]["seed_start"], config["test"]["seed_count"]),
        "smoke": integer_range(
            config["smoke"]["seed_start"], config["smoke"]["seed_count"]),
        "bootstrap": integer_range(
            config["statistics"]["bootstrap_seed_start"],
            config["statistics"]["seed_count"]),
    }
    overlaps = {}
    names = list(namespaces)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = namespaces[left] & namespaces[right]
            if shared:
                overlaps[f"{left}::{right}"] = {
                    "count": len(shared), "first": min(shared)}
    checks = {
        "protocol_version_present": bool(config["protocol"]["version"]),
        "five_independent_training_seeds": (
            len(train["independent_run_base_seeds"]) == 5),
        "training_seed_bases_unique": (
            len(set(train["independent_run_base_seeds"])) == 5),
        "warm_start_and_ppo_seeds_disjoint_within_run": within_run_disjoint,
        "all_seed_namespaces_disjoint": not overlaps,
        "test_is_paired_across_methods": bool(
            config["test"]["paired_across_methods"]),
        "test_is_locked_until_checkpoint_selection": bool(
            config["test"]["run_once_after_checkpoint_lock"]),
    }
    return {"assertions": checks, "overlaps": overlaps}, namespaces


def files_to_freeze() -> list[Path]:
    patterns = (
        "scripts/*.py",
        "src/warehouse_core/warehouse_core/*.py",
        "src/warehouse_bringup/config/*.yaml",
        "docs/architecture/*.md",
    )
    files = set()
    for pattern in patterns:
        files.update(path for path in WORK_ROOT.glob(pattern) if path.is_file())
    for filename in ("README.md", "0809交接文档.md", "文件检索的指引.md"):
        path = WORK_ROOT / filename
        if path.exists():
            files.add(path)
    return sorted(files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-config", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=(
        WORK_ROOT / "results" / "stage16" /
        "stage16_reproducibility_snapshot.json"))
    args = parser.parse_args()
    seed_config = yaml.safe_load(args.seed_config.read_text(encoding="utf-8"))
    seed_gate, namespaces = validate_seed_protocol(seed_config)
    file_rows = [{
        "path": str(path.relative_to(WORK_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in files_to_freeze()]
    canonical = json.dumps(file_rows, sort_keys=True, separators=(",", ":"))
    snapshot_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    report = {
        "stage": 16,
        "gate": "FROZEN_SEEDS_AND_REPRODUCIBILITY_SNAPSHOT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_protocol": str(args.seed_config.resolve()),
        "seed_protocol_version": seed_config["protocol"]["version"],
        "seed_namespaces": {
            name: {"first": min(values), "last": max(values),
                   "count": len(values)}
            for name, values in namespaces.items()},
        "seed_assertions": seed_gate["assertions"],
        "seed_overlaps": seed_gate["overlaps"],
        "snapshot_sha256": snapshot_sha256,
        "file_count": len(file_rows),
        "files": file_rows,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "passed": all(seed_gate["assertions"].values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "snapshot_sha256": snapshot_sha256,
        "file_count": len(file_rows),
        "passed": report["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
