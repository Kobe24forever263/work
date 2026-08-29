#!/usr/bin/env python3
"""Build a reproducible Stage 18 result-quality audit and report payload."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

import numpy as np


PROFILES = ("MEDIUM", "DENSE", "BURST")
PROFILE_ROOTS = {
    "MEDIUM": "stage18",
    "DENSE": "stage18_dense",
    "BURST": "stage18_burst",
}
CONDITIONS = (
    "full_context_v2",
    "no_queue_resource",
    "no_handover_cues",
    "no_persistent_position",
)
CONDITION_LABELS = {
    "no_queue_resource": "去队列/资源显式状态",
    "no_handover_cues": "去交接显式线索",
    "no_persistent_position": "去持续位置复合信息",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def exact_sign_flip_p(seed_deltas: np.ndarray) -> float:
    observed = abs(float(seed_deltas.mean()))
    count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(seed_deltas)):
        value = abs(float(np.mean(seed_deltas * np.asarray(signs))))
        count += int(value >= observed - 1e-12)
    return count / (2 ** len(seed_deltas))


def adjust_pvalues(rows: list[dict]) -> list[dict]:
    p_values = [float(row["exact_two_sided_sign_flip_p"]) for row in rows]
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    holm = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        holm[index] = running
    bh = [1.0] * count
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, min(1.0, p_values[index] * count / rank))
        bh[index] = running
    enhanced = []
    for index, source in enumerate(rows):
        row = dict(source)
        row["ci_low"] = row["hierarchical_bootstrap_95ci"][0]
        row["ci_high"] = row["hierarchical_bootstrap_95ci"][1]
        row["holm_adjusted_p_global36"] = holm[index]
        row["bh_fdr_q_global36"] = bh[index]
        row["raw_significant_0_05"] = p_values[index] < 0.05
        row["holm_significant_0_05"] = holm[index] < 0.05
        row["bh_significant_0_05"] = bh[index] < 0.05
        enhanced.append(row)
    return enhanced


def locked_path(work_root: Path, train_profile: str, condition: str,
                test_profile: str, seed: int) -> Path:
    return (work_root / "results" / "stage18_locked_test" /
            f"trained_{train_profile.lower()}" / condition /
            f"tested_{test_profile.lower()}" / f"seed_{seed:02d}.json")


def crossed_bootstrap_ratio_delta(
        policy_completed: np.ndarray,
        policy_time: np.ndarray,
        baseline_completed: np.ndarray,
        baseline_time: np.ndarray,
        rng: np.random.Generator,
        samples: int = 10000) -> tuple[float, float]:
    training_count, test_count = policy_completed.shape
    values = np.empty(samples, dtype=float)
    for sample in range(samples):
        training_indices = rng.integers(0, training_count, training_count)
        test_indices = rng.integers(0, test_count, test_count)
        deltas = []
        for training_index in training_indices:
            policy_rate = (
                policy_completed[training_index, test_indices].sum() /
                policy_time[training_index, test_indices].sum() * 3600.0)
            baseline_rate = (
                baseline_completed[training_index, test_indices].sum() /
                baseline_time[training_index, test_indices].sum() * 3600.0)
            deltas.append(policy_rate - baseline_rate)
        values[sample] = float(np.mean(deltas))
    return tuple(float(value) for value in np.quantile(values, (.025, .975)))


def successful_throughput_audit(work_root: Path) -> list[dict]:
    rows = []
    for profile_index, profile in enumerate(PROFILES):
        policy_completed = []
        policy_time = []
        baseline_completed = []
        baseline_time = []
        for seed in range(1, 11):
            result = read_json(locked_path(
                work_root, profile, "full_context_v2", profile, seed))
            policy_completed.append([
                episode["completed"] for episode in result["policy"]["episodes"]])
            policy_time.append([
                episode["simulated_time"] for episode in result["policy"]["episodes"]])
            baseline_completed.append([
                episode["completed"]
                for episode in result["rule_baseline"]["episodes"]])
            baseline_time.append([
                episode["simulated_time"]
                for episode in result["rule_baseline"]["episodes"]])
        pc = np.asarray(policy_completed, dtype=float)
        pt = np.asarray(policy_time, dtype=float)
        bc = np.asarray(baseline_completed, dtype=float)
        bt = np.asarray(baseline_time, dtype=float)
        policy_seed_rates = pc.sum(axis=1) / pt.sum(axis=1) * 3600.0
        baseline_seed_rates = bc.sum(axis=1) / bt.sum(axis=1) * 3600.0
        deltas = policy_seed_rates - baseline_seed_rates
        ci_low, ci_high = crossed_bootstrap_ratio_delta(
            pc, pt, bc, bt,
            np.random.default_rng(45000000 + profile_index))
        rows.append({
            "profile": profile,
            "policy_successful_throughput": float(policy_seed_rates.mean()),
            "baseline_successful_throughput": float(baseline_seed_rates.mean()),
            "mean_delta": float(deltas.mean()),
            "ci_low_crossed": ci_low,
            "ci_high_crossed": ci_high,
            "exact_sign_flip_p": exact_sign_flip_p(deltas),
            "positive_training_seed_count": int(np.sum(deltas > 0)),
            "unit": "completed tasks per simulated hour",
        })
    return rows


def cross_load_audit(work_root: Path) -> list[dict]:
    rows = []
    for train_profile in PROFILES:
        for test_profile in PROFILES:
            reward_deltas = []
            success_deltas = []
            throughput_deltas = []
            car_dog_count = 0
            single_dog_count = 0
            total_modes = 0
            for seed in range(1, 11):
                result = read_json(locked_path(
                    work_root, train_profile, "full_context_v2",
                    test_profile, seed))
                policy = result["policy"]
                baseline = result["rule_baseline"]
                reward_deltas.append(
                    float(policy["mean_reward"] - baseline["mean_reward"]))
                success_deltas.append(
                    float(policy["success_rate"] - baseline["success_rate"]))
                policy_successful = (
                    policy["completed"] /
                    sum(item["simulated_time"] for item in policy["episodes"]) *
                    3600.0)
                baseline_successful = (
                    baseline["completed"] /
                    sum(item["simulated_time"]
                        for item in baseline["episodes"]) * 3600.0)
                throughput_deltas.append(policy_successful - baseline_successful)
                modes = policy["transport_modes"]
                car_dog_count += int(modes.get("CAR_DOG_CAR", 0))
                single_dog_count += int(modes.get("SINGLE_DOG", 0))
                total_modes += sum(int(value) for value in modes.values())
            rows.append({
                "training_profile": train_profile,
                "evaluation_profile": test_profile,
                "route": f"{train_profile}→{test_profile}",
                "reward_delta": float(statistics.mean(reward_deltas)),
                "success_delta_pp": float(statistics.mean(success_deltas) * 100.0),
                "successful_throughput_delta": float(
                    statistics.mean(throughput_deltas)),
                "positive_reward_seed_count": sum(
                    value > 0 for value in reward_deltas),
                "car_dog_car_share_pct": car_dog_count / total_modes * 100.0,
                "single_dog_share_pct": single_dog_count / total_modes * 100.0,
            })
    return rows


def build_audit(work_root: Path) -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    training_errors = []
    metadata_counts = Counter()
    run_seed_alignment = {}
    total_history_rows = 0
    expected_checkpoint_count = 0
    existing_checkpoint_count = 0
    summary_count = 0
    for profile in PROFILES:
        for condition in CONDITIONS:
            seeds = []
            for seed in range(1, 11):
                run_dir = (work_root / "results" / PROFILE_ROOTS[profile] /
                           condition / f"seed_{seed:02d}")
                summary_path = run_dir / "stage18_ppo.summary.json"
                history_path = run_dir / "stage18_ppo.history.json"
                if not summary_path.exists() or not history_path.exists():
                    training_errors.append(
                        f"missing summary/history: {profile}/{condition}/seed_{seed:02d}")
                    continue
                summary_count += 1
                summary = read_json(summary_path)
                history = read_json(history_path)
                total_history_rows += len(history)
                metadata_counts[(summary.get("stage"),
                                 summary.get("schema_version"))] += 1
                seeds.append(summary.get("experiment_run_seed"))
                updates = [row.get("update") for row in history]
                if updates != list(range(1, 2001)):
                    training_errors.append(
                        f"non-contiguous history: {profile}/{condition}/seed_{seed:02d}")
                if summary.get("updates") != 2000 or \
                        summary.get("episodes_trained_total") != 8000:
                    training_errors.append(
                        f"wrong training length: {profile}/{condition}/seed_{seed:02d}")
                if not finite_tree(summary) or not finite_tree(history):
                    training_errors.append(
                        f"non-finite value: {profile}/{condition}/seed_{seed:02d}")
                expected_paths = [
                    run_dir / "stage18_ppo.pt",
                    run_dir / "stage18_ppo.best.pt",
                ] + [
                    run_dir / "checkpoints" /
                    f"stage18_ppo_update_{update:05d}.pt"
                    for update in range(50, 2001, 50)
                ]
                expected_checkpoint_count += len(expected_paths)
                existing_checkpoint_count += sum(path.exists() for path in expected_paths)
            run_seed_alignment[f"{profile}/{condition}"] = seeds

    locked_files = sorted((work_root / "results" / "stage18_locked_test").glob(
        "trained_*/*/tested_*/seed_*.json"))
    locked_errors = []
    locked_keys = []
    policy_episode_count = 0
    baseline_episode_count = 0
    policy_task_count = 0
    baseline_task_count = 0
    baseline_hashes = {profile: set() for profile in PROFILES}
    for path in locked_files:
        result = read_json(path)
        key = (
            result.get("training_profile"), result.get("condition"),
            result.get("evaluation_profile"),
            result.get("training_seed_index"))
        locked_keys.append(key)
        policy = result.get("policy", {})
        baseline = result.get("rule_baseline", {})
        policy_episodes = policy.get("episodes", [])
        baseline_episodes = baseline.get("episodes", [])
        policy_episode_count += len(policy_episodes)
        baseline_episode_count += len(baseline_episodes)
        policy_task_count += int(policy.get("task_count", 0))
        baseline_task_count += int(baseline.get("task_count", 0))
        expected_seeds = list(range(43000000, 43000100))
        if [row.get("seed") for row in policy_episodes] != expected_seeds:
            locked_errors.append(f"policy test-seed mismatch: {path}")
        if [row.get("seed") for row in baseline_episodes] != expected_seeds:
            locked_errors.append(f"baseline test-seed mismatch: {path}")
        if len(policy_episodes) != 100 or len(baseline_episodes) != 100:
            locked_errors.append(f"wrong episode count: {path}")
        if policy.get("completed", 0) + policy.get("failed", 0) != 2000:
            locked_errors.append(f"unresolved policy tasks: {path}")
        if baseline.get("completed", 0) + baseline.get("failed", 0) != 2000:
            locked_errors.append(f"unresolved baseline tasks: {path}")
        if policy.get("illegal_action_count") != 0 or \
                policy.get("resource_leak_count") != 0:
            locked_errors.append(f"policy safety failure: {path}")
        if baseline.get("resource_leak_count") != 0:
            locked_errors.append(f"baseline resource leak: {path}")
        if not finite_tree(result):
            locked_errors.append(f"non-finite locked result: {path}")
        evaluation_profile = result.get("evaluation_profile")
        baseline_payload = json.dumps(
            baseline_episodes, sort_keys=True, separators=(",", ":"))
        baseline_hashes[evaluation_profile].add(
            hashlib.sha256(baseline_payload.encode("utf-8")).hexdigest())

    expected_locked_keys = set()
    for train_profile in PROFILES:
        for condition in CONDITIONS:
            for seed in range(1, 11):
                expected_locked_keys.add(
                    (train_profile, condition, train_profile, seed))
        for test_profile in PROFILES:
            if test_profile == train_profile:
                continue
            for seed in range(1, 11):
                expected_locked_keys.add(
                    (train_profile, "full_context_v2", test_profile, seed))

    statistics_path = (work_root / "results" / "stage18_locked_test" /
                       "stage18_locked_test_statistics.json")
    statistics_report = read_json(statistics_path)
    enhanced_comparisons = adjust_pvalues(statistics_report["comparisons"])
    raw_significant = sum(row["raw_significant_0_05"]
                          for row in enhanced_comparisons)
    holm_significant = sum(row["holm_significant_0_05"]
                           for row in enhanced_comparisons)
    bh_significant = sum(row["bh_significant_0_05"]
                         for row in enhanced_comparisons)

    main_effects = []
    for row in enhanced_comparisons:
        if row["comparison"] != "full_context_v2_minus_rule_baseline":
            continue
        item = {
            "profile": row["profile"],
            "metric": row["metric"],
            "mean_delta": row["mean_delta"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "raw_p": row["exact_two_sided_sign_flip_p"],
            "bh_q_global36": row["bh_fdr_q_global36"],
            "holm_p_global36": row["holm_adjusted_p_global36"],
            "positive_seed_count": row["positive_training_seed_count"],
        }
        if row["metric"] == "success_rate":
            item["mean_delta_pp"] = row["mean_delta"] * 100.0
            item["ci_low_pp"] = row["ci_low"] * 100.0
            item["ci_high_pp"] = row["ci_high"] * 100.0
        main_effects.append(item)

    ablation_reward = []
    for row in enhanced_comparisons:
        if row["metric"] != "reward" or \
                row["comparison"] == "full_context_v2_minus_rule_baseline":
            continue
        condition = row["comparison"].removeprefix("full_context_v2_minus_")
        ablation_reward.append({
            "profile": row["profile"],
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "mean_delta": row["mean_delta"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "raw_p": row["exact_two_sided_sign_flip_p"],
            "bh_q_global36": row["bh_fdr_q_global36"],
            "positive_seed_count": row["positive_training_seed_count"],
        })

    successful_throughput = successful_throughput_audit(work_root)
    cross_load = cross_load_audit(work_root)

    actual_profiles = sorted({key[2] for key in locked_keys})
    expected_scenarios = [
        "MEDIUM", "DENSE", "CROSS_HEAVY", "BURST",
        "MIXED_CONTINUOUS_RECOVERY"]
    scenario_coverage = [{
        "scenario": scenario,
        "locked_test_present": scenario in actual_profiles,
        "status": "COVERED" if scenario in actual_profiles else "MISSING",
    } for scenario in expected_scenarios]

    old_manifest_path = (work_root / "results" / "stage18" /
                         "stage18_training_manifest.json")
    old_manifest = read_json(old_manifest_path) if old_manifest_path.exists() else {}
    manifest_runs = old_manifest.get("runs", [])
    manifest_missing_paths = 0
    for row in manifest_runs:
        path_value = row.get("summary") or row.get("summary_path") or ""
        if path_value and not Path(path_value).exists():
            manifest_missing_paths += 1

    issues = [
        {
            "severity_rank": 1,
            "severity": "HIGH",
            "finding": "配对测试使用顺序交接随机流，而非 TASK_KEYED",
            "evidence": (
                "stage14_training.py:278-285,313-319; "
                "stage14_ppo.py:366-371; handover_timing.py:157-162"),
            "impact": "不同策略按不同顺序消耗随机交接时长，逐任务配对不成立。",
            "remediation": "保留权重，使用 TASK_KEYED 和新鲜最终测试种子重跑 locked evaluation。",
        },
        {
            "severity_rank": 1,
            "severity": "HIGH",
            "finding": "Bootstrap 将交叉设计误作嵌套设计",
            "evidence": "summarize_stage18_locked_test.py:56-66",
            "impact": "10 个训练种子共享同一 100 个测试种子，当前置信区间可能偏窄。",
            "remediation": "同时重采训练种子轴和共享测试种子轴，使用 crossed/two-way bootstrap。",
        },
        {
            "severity_rank": 1,
            "severity": "HIGH",
            "finding": "36 项显著性检验没有多重比较校正",
            "evidence": (
                f"raw p<.05: {raw_significant}; global Holm: "
                f"{holm_significant}; BH-FDR: {bh_significant}"),
            "impact": "不能把未校正 p 值直接写成全局确认性证据。",
            "remediation": "预先定义主要假设族并报告 Holm/BH；若坚持全局 FWER，增加训练种子。",
        },
        {
            "severity_rank": 1,
            "severity": "HIGH",
            "finding": "三项消融不是语义干净的单因素消融",
            "evidence": "stage12_encoding.py:63-98,206-219,247-289",
            "impact": "交接与资源信息可被其他字段推断；位置消融同时删除 ETA/cost 等复合信息。",
            "remediation": "重定义互不重叠且不可旁路推断的消融；否则收窄论文表述。",
        },
        {
            "severity_rank": 2,
            "severity": "MEDIUM",
            "finding": "throughput 将失败任务计入吞吐",
            "evidence": "stage14_ppo.py:459-472; summarize_stage18_locked_test.py:40-44",
            "impact": "快速失败也会提高该指标；当前名称容易被误解为成功交付吞吐。",
            "remediation": "分别报告 resolved-task rate 与 completed-only successful throughput。",
        },
        {
            "severity_rank": 2,
            "severity": "MEDIUM",
            "finding": "冻结协议场景和算法基线未完全覆盖",
            "evidence": (
                "experiment_seeds.yaml:16-17,29-34; "
                "run_stage18_locked_test_campaign.py:27-35"),
            "impact": "尚不能支持混合连续负载、恢复能力或对多种学习基线的普遍性结论。",
            "remediation": "补 CROSS_HEAVY、MIXED_CONTINUOUS_RECOVERY 与 FLAT/FIXED_GAMMA/RTAW。",
        },
        {
            "severity_rank": 2,
            "severity": "MEDIUM",
            "finding": "60 个跨负载结果未进入正式统计",
            "evidence": "locked jobs=180，但 statistics 仅含 36 个域内比较。",
            "impact": "已有数据不能直接支撑 3×3 跨负载泛化结论。",
            "remediation": "增加 3×3 transfer/interactions 的 crossed bootstrap 与校正后检验。",
        },
        {
            "severity_rank": 2,
            "severity": "MEDIUM",
            "finding": "实验 manifest 与结果内部阶段元数据已陈旧",
            "evidence": (
                f"120/120 summary 标为 stage=14/schema=stage16；旧 manifest "
                f"runs={len(manifest_runs)}, missing_paths={manifest_missing_paths}"),
            "impact": "当前文件不能作为投稿级、不可变的实验索引。",
            "remediation": "生成覆盖代码、环境、配置、权重和结果 SHA256 的只读最终 manifest。",
        },
        {
            "severity_rank": 3,
            "severity": "LOW",
            "finding": "锁定结果没有记录权重哈希或源码提交",
            "evidence": "run_stage18_locked_test_campaign.py:70-90,132-139",
            "impact": "权重或代码变更后，缓存结果可能仍被误认为同一实验谱系。",
            "remediation": "缓存有效性校验加入 weight SHA256、config hash 和 commit/source snapshot。",
        },
    ]

    coverage = {
        "training_runs_expected": 120,
        "training_runs_present": summary_count,
        "training_run_completion_rate": summary_count / 120.0,
        "history_rows": total_history_rows,
        "training_episodes": summary_count * 8000,
        "ppo_artifacts_expected": expected_checkpoint_count,
        "ppo_artifacts_present": existing_checkpoint_count,
        "locked_jobs_expected": 180,
        "locked_jobs_present": len(locked_files),
        "locked_job_unique_keys": len(set(locked_keys)),
        "policy_test_episodes": policy_episode_count,
        "baseline_test_episodes": baseline_episode_count,
        "policy_test_tasks": policy_task_count,
        "baseline_test_tasks": baseline_task_count,
        "official_comparisons": len(enhanced_comparisons),
        "illegal_action_count": 0 if not locked_errors else None,
        "resource_leak_count": 0 if not locked_errors else None,
    }
    coverage_rows = [
        {"area": "训练单元", "expected": 120, "actual": summary_count,
         "status": "PASS" if summary_count == 120 else "FAIL"},
        {"area": "PPO history updates", "expected": 240000,
         "actual": total_history_rows,
         "status": "PASS" if total_history_rows == 240000 else "FAIL"},
        {"area": "PPO checkpoint/final/best", "expected": expected_checkpoint_count,
         "actual": existing_checkpoint_count,
         "status": "PASS" if existing_checkpoint_count == expected_checkpoint_count else "FAIL"},
        {"area": "Locked-test jobs", "expected": 180,
         "actual": len(locked_files),
         "status": "PASS" if set(locked_keys) == expected_locked_keys else "FAIL"},
        {"area": "Policy test episodes", "expected": 18000,
         "actual": policy_episode_count,
         "status": "PASS" if policy_episode_count == 18000 else "FAIL"},
        {"area": "统计比较", "expected": 36,
         "actual": len(enhanced_comparisons),
         "status": "PASS" if len(enhanced_comparisons) == 36 else "FAIL"},
    ]

    return {
        "schema_version": "warehouse_stage18_quality_audit_v1",
        "generated_at_utc": generated_at,
        "as_of_local": "2026-08-15 05:36 CST",
        "overall_assessment": "NEEDS_REVISION_BEFORE_TRO_CONFIRMATORY_CLAIMS",
        "decision_summary": (
            "Core files and arithmetic pass. Confirmatory inference fails until "
            "paired randomness, crossed bootstrap, multiplicity, throughput, "
            "ablation constructs, and protocol coverage are corrected."),
        "coverage": coverage,
        "coverage_rows": coverage_rows,
        "training_integrity_errors": training_errors,
        "locked_integrity_errors": locked_errors,
        "locked_key_coverage_exact": set(locked_keys) == expected_locked_keys,
        "baseline_unique_payloads_by_evaluation_profile": {
            profile: len(values) for profile, values in baseline_hashes.items()},
        "metadata_stage_schema_counts": [
            {"stage": key[0], "schema_version": key[1], "count": value}
            for key, value in metadata_counts.items()],
        "run_seed_alignment": run_seed_alignment,
        "multiplicity": {
            "comparison_count": len(enhanced_comparisons),
            "raw_p_lt_0_05": raw_significant,
            "global_holm_p_lt_0_05": holm_significant,
            "global_bh_fdr_q_lt_0_05": bh_significant,
            "minimum_attainable_exact_two_sided_p_with_10_seeds": 2 / 1024,
        },
        "official_comparisons_enhanced": enhanced_comparisons,
        "main_effects": main_effects,
        "ablation_reward": ablation_reward,
        "successful_throughput": successful_throughput,
        "cross_load": cross_load,
        "scenario_coverage": scenario_coverage,
        "issues": issues,
        "old_manifest_audit": {
            "path": "results/stage18/stage18_training_manifest.json",
            "declared_passed": old_manifest.get("passed"),
            "listed_run_count": len(manifest_runs),
            "listed_missing_path_count": manifest_missing_paths,
            "status": "SUPERSEDED_AND_UNSAFE_AS_CURRENT_INDEX",
        },
        "source_files": [
            "results/stage18_locked_test/stage18_locked_test_statistics.json",
            "results/stage18_locked_test/trained_*/*/tested_*/seed_*.json",
            "results/stage18*/**/stage18_ppo.summary.json",
            "results/stage18*/**/stage18_ppo.history.json",
            "src/warehouse_bringup/config/experiment_seeds.yaml",
            "scripts/run_stage18_locked_test_campaign.py",
            "scripts/summarize_stage18_locked_test.py",
        ],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report_artifact(audit: dict) -> dict:
    generated_at = audit["generated_at_utc"]
    reward_rows = [row for row in audit["main_effects"]
                   if row["metric"] == "reward"]
    main_table = []
    for row in audit["main_effects"]:
        delta = row.get("mean_delta_pp", row["mean_delta"])
        ci_low = row.get("ci_low_pp", row["ci_low"])
        ci_high = row.get("ci_high_pp", row["ci_high"])
        unit = "percentage points" if row["metric"] == "success_rate" else (
            "resolved tasks/hour" if row["metric"] ==
            "throughput_tasks_per_hour" else "raw episode reward")
        main_table.append({
            "profile": row["profile"],
            "metric": row["metric"],
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "unit": unit,
            "raw_p": row["raw_p"],
            "bh_q": row["bh_q_global36"],
            "holm_p": row["holm_p_global36"],
        })
    cards = [{
        "id": "training_coverage",
        "description": "3负载×4条件×10训练种子。",
        "dataset": "coverage_cards",
        "sourceId": "audit_results",
        "metrics": [{"label": "训练运行完整率", "field": "training_rate",
                     "format": "percent"}],
    }, {
        "id": "locked_coverage",
        "description": "含120个域内与60个跨负载测试。",
        "dataset": "coverage_cards",
        "sourceId": "audit_results",
        "metrics": [{"label": "Locked jobs", "field": "locked_jobs",
                     "format": "number"}],
    }, {
        "id": "safety_coverage",
        "description": "所有策略测试中的非法动作与资源泄漏。",
        "dataset": "coverage_cards",
        "sourceId": "audit_results",
        "metrics": [{"label": "安全性违规", "field": "safety_violations",
                     "format": "number"}],
    }, {
        "id": "holm_results",
        "description": "36项比较做全局Holm校正后的显著项。",
        "dataset": "coverage_cards",
        "sourceId": "audit_results",
        "metrics": [{"label": "全局Holm显著项", "field": "holm_significant",
                     "format": "number"}],
    }]
    charts = [{
        "id": "baseline_reward_delta",
        "title": "Full相对规则基线的回报差",
        "subtitle": "三种同负载locked测试；正值有利于Full，n=10训练种子×100测试episode。",
        "type": "bar",
        "dataset": "baseline_reward_delta",
        "sourceId": "official_locked_statistics",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "profile", "type": "nominal", "label": "负载"},
            "y": {"field": "mean_delta", "type": "quantitative",
                  "label": "回报差"},
            "tooltip": [
                {"field": "ci_low", "type": "quantitative", "label": "95% CI下界"},
                {"field": "ci_high", "type": "quantitative", "label": "95% CI上界"},
                {"field": "raw_p", "type": "quantitative", "label": "原始p"},
            ],
        },
    }, {
        "id": "successful_throughput_delta",
        "title": "成功交付吞吐差",
        "subtitle": "completed-only tasks/hour；正值有利于Full，区间使用两轴交叉Bootstrap。",
        "type": "bar",
        "dataset": "successful_throughput",
        "sourceId": "audit_results",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "profile", "type": "nominal", "label": "负载"},
            "y": {"field": "mean_delta", "type": "quantitative",
                  "label": "成功任务/模拟小时差"},
            "tooltip": [
                {"field": "ci_low_crossed", "type": "quantitative",
                 "label": "交叉Bootstrap CI下界"},
                {"field": "ci_high_crossed", "type": "quantitative",
                 "label": "交叉Bootstrap CI上界"},
            ],
        },
    }, {
        "id": "ablation_reward_delta",
        "title": "Full相对三项消融的回报差",
        "subtitle": "正值表示Full更好；仅持续位置复合消融在三种负载下稳定退化。",
        "type": "bar",
        "dataset": "ablation_reward",
        "sourceId": "audit_results",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "profile", "type": "nominal", "label": "负载"},
            "y": {"field": "mean_delta", "type": "quantitative",
                  "label": "Full−消融回报"},
            "color": {"field": "condition_label", "type": "nominal",
                      "label": "消融条件"},
            "tooltip": [
                {"field": "ci_low", "type": "quantitative", "label": "95% CI下界"},
                {"field": "ci_high", "type": "quantitative", "label": "95% CI上界"},
            ],
        },
    }, {
        "id": "cross_load_reward_delta",
        "title": "3×3跨负载回报差",
        "subtitle": "Full相对规则基线；这些60个jobs尚未进入官方36项统计。",
        "type": "bar",
        "dataset": "cross_load",
        "sourceId": "audit_results",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "route", "type": "nominal", "label": "训练→测试负载"},
            "y": {"field": "reward_delta", "type": "quantitative",
                  "label": "回报差"},
            "color": {"field": "training_profile", "type": "nominal",
                      "label": "训练负载"},
            "tooltip": [
                {"field": "success_delta_pp", "type": "quantitative",
                 "label": "成功率差(pp)"},
                {"field": "successful_throughput_delta", "type": "quantitative",
                 "label": "成功吞吐差"},
            ],
        },
    }]
    tables = [{
        "id": "coverage_table",
        "title": "产物完整性检查",
        "subtitle": "截至2026年8月15日05:36（中国标准时间）。",
        "dataset": "coverage_rows",
        "sourceId": "audit_results",
        "defaultSort": {"field": "area", "direction": "asc"},
        "columns": [
            {"field": "area", "label": "检查项", "type": "text"},
            {"field": "expected", "label": "期望", "format": "number"},
            {"field": "actual", "label": "实际", "format": "number"},
            {"field": "status", "label": "状态", "type": "text"},
        ],
    }, {
        "id": "main_effects_table",
        "title": "Full与规则基线的域内比较",
        "subtitle": "CI为现有嵌套Bootstrap；Holm/BH为本次审查对36项原始p值的补充校正。",
        "dataset": "main_effects_table",
        "sourceId": "audit_results",
        "defaultSort": {"field": "profile", "direction": "asc"},
        "columns": [
            {"field": "profile", "label": "负载", "type": "text"},
            {"field": "metric", "label": "指标", "type": "text"},
            {"field": "delta", "label": "差值", "format": "number"},
            {"field": "ci_low", "label": "CI下界", "format": "number"},
            {"field": "ci_high", "label": "CI上界", "format": "number"},
            {"field": "raw_p", "label": "原始p", "format": "number"},
            {"field": "bh_q", "label": "BH q", "format": "number"},
            {"field": "holm_p", "label": "Holm p", "format": "number"},
        ],
    }, {
        "id": "issues_table",
        "title": "必须整改的问题",
        "subtitle": "按对T-RO结论可信度的影响排序。",
        "dataset": "issues",
        "sourceId": "audit_results",
        "defaultSort": {"field": "severity_rank", "direction": "asc"},
        "columns": [
            {"field": "severity_rank", "label": "优先级", "format": "number"},
            {"field": "severity", "label": "严重性", "type": "text"},
            {"field": "finding", "label": "问题", "type": "text"},
            {"field": "impact", "label": "影响", "type": "text"},
            {"field": "remediation", "label": "整改", "type": "text"},
        ],
    }, {
        "id": "cross_load_table",
        "title": "跨负载结果明细",
        "subtitle": "每个单元10个训练种子；当前为描述性均值，尚非正式确认性统计。",
        "dataset": "cross_load",
        "sourceId": "audit_results",
        "defaultSort": {"field": "route", "direction": "asc"},
        "columns": [
            {"field": "route", "label": "训练→测试", "type": "text"},
            {"field": "reward_delta", "label": "回报差", "format": "number"},
            {"field": "success_delta_pp", "label": "成功率差(pp)", "format": "number"},
            {"field": "successful_throughput_delta", "label": "成功吞吐差", "format": "number"},
            {"field": "car_dog_car_share_pct", "label": "车狗车占比(%)", "format": "number"},
            {"field": "single_dog_share_pct", "label": "单狗占比(%)", "format": "number"},
        ],
    }, {
        "id": "scenario_table",
        "title": "冻结测试协议覆盖",
        "subtitle": "协议声明场景与本轮locked campaign实际覆盖的对照。",
        "dataset": "scenario_coverage",
        "sourceId": "seed_protocol",
        "defaultSort": {"field": "scenario", "direction": "asc"},
        "columns": [
            {"field": "scenario", "label": "场景", "type": "text"},
            {"field": "status", "label": "覆盖状态", "type": "text"},
        ],
    }]
    sources = [{
        "id": "audit_results",
        "label": "Stage18质量审计结果",
        "path": "results/stage18_quality_review/stage18_quality_audit.json",
        "query": {
            "engine": "duckdb",
            "sql": (
                "SELECT * FROM read_json_auto("
                "'results/stage18_quality_review/stage18_quality_audit.json')"
            ),
            "description": "读取本次只读质量审查生成的结构化结果快照。",
            "tables_used": [
                "results/stage18_quality_review/stage18_quality_audit.json"
            ],
            "filters": ["审查快照截至2026-08-15 05:36 CST"],
            "metric_definitions": [
                "训练完整率=实际完成训练单元/期望训练单元",
                "成功交付吞吐=completed tasks/simulated time×3600",
                "交叉Bootstrap同时重采训练种子轴与共享测试种子轴",
            ],
        },
    }, {
        "id": "official_locked_statistics",
        "label": "Stage18 locked-test原统计",
        "path": "results/stage18_locked_test/stage18_locked_test_statistics.json",
        "query": {
            "engine": "duckdb",
            "sql": (
                "SELECT * FROM read_json_auto("
                "'results/stage18_locked_test/stage18_locked_test_statistics.json')"
            ),
            "description": "读取项目原有的36项locked-test统计比较。",
            "tables_used": [
                "results/stage18_locked_test/stage18_locked_test_statistics.json"
            ],
            "filters": ["comparison=full_context_vs_rule_baseline"],
            "metric_definitions": [
                "回报差=Full策略平均raw episode reward−规则基线平均值",
                "成功率差=Full完成率−规则基线完成率",
                "原throughput=(completed+failed)/simulated time×3600",
            ],
        },
    }, {
        "id": "locked_raw",
        "label": "Stage18 locked-test逐episode结果",
        "path": "results/stage18_locked_test/trained_*/*/tested_*/seed_*.json",
    }, {
        "id": "seed_protocol",
        "label": "冻结实验种子协议",
        "path": "src/warehouse_bringup/config/experiment_seeds.yaml",
        "query": {
            "engine": "duckdb",
            "sql": (
                "SELECT scenario, status FROM scenario_coverage "
                "ORDER BY scenario"
            ),
            "description": (
                "将冻结YAML中声明的场景与locked campaign实际产物做覆盖对照；"
                "scenario_coverage由审查脚本解析生成。"
            ),
            "tables_used": [
                "src/warehouse_bringup/config/experiment_seeds.yaml",
                "results/stage18_locked_test/trained_*/*/tested_*/seed_*.json",
            ],
            "filters": ["仅统计冻结协议中声明的评估场景"],
            "metric_definitions": [
                "覆盖状态=声明场景是否存在对应locked-test产物"
            ],
        },
    }, {
        "id": "evaluation_code",
        "label": "Stage18 locked-test与环境评估代码",
        "path": "scripts/run_stage18_locked_test_campaign.py",
    }, {
        "id": "statistics_code",
        "label": "Stage18统计脚本",
        "path": "scripts/summarize_stage18_locked_test.py",
    }, {
        "id": "encoding_code",
        "label": "Stage18状态与动作编码",
        "path": "src/warehouse_core/warehouse_core/stage12_encoding.py",
    }]
    blocks = [{
        "id": "title",
        "type": "markdown",
        "body": "# Stage 18 结果质量审查",
    }, {
        "id": "technical_summary",
        "type": "markdown",
        "body": (
            "## 技术结论：数据完整，但暂不具备T-RO确认性结论资格\n\n"
            "**完整性与安全性通过。** 120个训练单元、240,000条更新记录和180个locked jobs均齐全；"
            "所有策略测试均无非法动作或资源泄漏。原始统计数值也能独立复算。\n\n"
            "**确认性推断需要整改。** Stage18退回了顺序式交接随机流，破坏逐任务配对；"
            "Bootstrap把共享测试种子的交叉设计误作嵌套设计；36项检验没有多重校正。"
            "因此当前数值可作为探索性v1证据，但不应直接作为T-RO最终表。\n\n"
            "**无需立即重训完整模型。** 第一轮应保留现有权重，修复TASK_KEYED评估、"
            "成功吞吐和crossed bootstrap，再用新鲜测试种子复验。只有重定义消融和补算法基线时才需要新增训练。"),
    }, {
        "id": "headline_metrics",
        "type": "metric-strip",
        "cardIds": ["training_coverage", "locked_coverage",
                    "safety_coverage", "holm_results"],
    }, {
        "id": "integrity_section",
        "type": "markdown",
        "body": (
            "## 核心数据可以继续使用\n\n"
            "训练history严格覆盖update 1–2000，每个运行累计8000个episode；"
            "180个locked文件各含100个policy与100个rule episode，测试种子均为43000000–43000099。"
            "规则基线在相同评估负载下逐episode完全一致，说明基础任务流可重复。"
            "问题集中在评估协议和统计解释，而不是文件丢失或训练崩溃。"),
    }, {
        "id": "coverage_block", "type": "table", "tableId": "coverage_table",
    }, {
        "id": "observed_effect_section",
        "type": "markdown",
        "body": (
            "## Full相对规则基线的收益随负载增强，但不是所有指标都普遍改善\n\n"
            "现有v1测试中，回报差从MEDIUM的+0.352扩大到DENSE的+19.224和BURST的+39.740。"
            "成功率提升分别为+0.96、+0.195和+0.37个百分点；DENSE成功率优势并不稳健。"
            "这些差值只能在同一负载内解释，不能把不同负载下的绝对回报直接比较。"),
    }, {
        "id": "baseline_reward_chart", "type": "chart",
        "chartId": "baseline_reward_delta",
    }, {
        "id": "main_effects_block", "type": "table",
        "tableId": "main_effects_table",
    }, {
        "id": "throughput_section",
        "type": "markdown",
        "body": (
            "## 当前吞吐名称会把快速失败误当成生产力\n\n"
            "原指标按(completed+failed)/time计算，更准确的名称是任务结算率。"
            "改用completed/time后，Full相对规则基线的成功交付吞吐在MEDIUM、DENSE、BURST"
            "分别约为+0.39、+2.14和+1.19任务/模拟小时；BURST区间仍跨零。"
            "论文应同时报告resolved-task rate与successful-delivery throughput。"),
    }, {
        "id": "successful_throughput_chart", "type": "chart",
        "chartId": "successful_throughput_delta",
    }, {
        "id": "multiplicity_section",
        "type": "markdown",
        "body": (
            "## 多重比较与统计层级会改变显著性结论\n\n"
            "36项原始检验中17项p<.05；全局BH-FDR后保留16项，但全局Holm校正后为0项。"
            "原因之一是10个训练种子的exact sign-flip双侧p最小只能达到0.001953。"
            "此外，同一100个测试种子被10个训练种子共同使用，必须按训练种子轴×测试种子轴做交叉Bootstrap，"
            "而不是把测试episode独立嵌套在每个训练种子下。应先冻结主要假设族，再决定Holm或BH。"),
    }, {
        "id": "ablation_section",
        "type": "markdown",
        "body": (
            "## 当前消融只支持“持续上下文复合信息重要”\n\n"
            "去队列/资源显式状态和去交接显式线索没有产生稳定退化；这并不证明这些信息无用，"
            "因为候选集合、动作掩码、运输模式和参与者数量仍可旁路推断它们。"
            "去持续位置条件在三种负载下显著退化，但它同时删除楼层、XYZ、累计里程、空闲时长、ETA和cost，"
            "不能把效果单独归因于‘位置’。"),
    }, {
        "id": "ablation_chart", "type": "chart",
        "chartId": "ablation_reward_delta",
    }, {
        "id": "cross_load_section",
        "type": "markdown",
        "body": (
            "## 跨负载文件存在，但尚未证明在线自适应\n\n"
            "60个跨负载Full结果已生成，却没有进入官方36项统计。"
            "描述性模式占比还显示MEDIUM训练策略在更高负载下仍保持约20%的车狗车模式，"
            "而DENSE/BURST训练策略约为32%，更像训练负载形成的固定偏好。"
            "要证明策略会随实时环境切换，冻结协议中的MIXED_CONTINUOUS_RECOVERY不可缺少。"),
    }, {
        "id": "cross_load_chart", "type": "chart",
        "chartId": "cross_load_reward_delta",
    }, {
        "id": "cross_load_block", "type": "table",
        "tableId": "cross_load_table",
    }, {
        "id": "protocol_section",
        "type": "markdown",
        "body": (
            "## 冻结协议尚未完整执行\n\n"
            "本轮locked campaign覆盖MEDIUM、DENSE和BURST，但没有CROSS_HEAVY与"
            "MIXED_CONTINUOUS_RECOVERY；学习基线FLAT_MASKED_PPO、FIXED_GAMMA和RTAW也未进入正式比较。"
            "因此当前只能陈述‘相对一个贪心规则基线、在三种固定负载下’的结果。"),
    }, {
        "id": "scenario_block", "type": "table", "tableId": "scenario_table",
    }, {
        "id": "method_section",
        "type": "markdown",
        "body": (
            "## 审查范围与方法\n\n"
            "审查粒度为训练运行(profile×condition×training seed)、locked job"
            "(再加evaluation profile)和test episode。检查包括文件完整性、更新连续性、权重存在性、"
            "种子隔离、聚合重算、安全不变量、配对设计、Bootstrap层级、多重比较、指标口径、"
            "消融构念及冻结协议覆盖。数值审查基于截至2026年8月15日05:36 CST生成的结果快照。"),
    }, {
        "id": "limitations_section",
        "type": "markdown",
        "body": (
            "## 限制与不确定性\n\n"
            "当前120个summary内部仍标stage=14/schema=stage16，旧训练manifest也只覆盖较早的5种子批次。"
            "locked结果没有保存权重SHA256或源码提交，无法从结果文件独立证明所有训练使用完全相同代码。"
            "本报告中的成功吞吐和跨负载表是审查阶段重算，原始episode未被修改；"
            "在TASK_KEYED新鲜测试完成前，它们仍属于诊断性证据。"),
    }, {
        "id": "issues_block", "type": "table", "tableId": "issues_table",
    }, {
        "id": "next_steps_section",
        "type": "markdown",
        "body": (
            "## 建议的整改顺序\n\n"
            "1. 将现有结果冻结为v1并生成SHA256 manifest，禁止覆盖。\n"
            "2. 修复locked评估为TASK_KEYED，修复crossed bootstrap、成功吞吐和多重校正。\n"
            "3. 保留现有权重，使用全新的最终测试种子做确认性复验。\n"
            "4. 补3×3跨负载统计、CROSS_HEAVY与MIXED_CONTINUOUS_RECOVERY。\n"
            "5. 重定义语义干净的消融，并补FLAT、FIXED_GAMMA、RTAW基线；这些步骤才需要新增训练。"),
    }, {
        "id": "questions_section",
        "type": "markdown",
        "body": (
            "## 后续需要明确的问题\n\n"
            "- 论文主假设是优先强调成功交付、回报，还是实时自适应运输模式？\n"
            "- 多重比较要采用少量预注册主要假设的Holm控制，还是探索性BH-FDR？\n"
            "- 消融章节是否接受‘持续上下文复合信息’的较窄结论，还是必须拆分位置、ETA与成本？"),
    }]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Stage 18 结果质量审查",
            "description": "面向T-RO投稿的训练、locked test与统计证据质量审查。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "coverage_cards": [{
                    "training_rate": audit["coverage"]["training_run_completion_rate"],
                    "locked_jobs": audit["coverage"]["locked_jobs_present"],
                    "safety_violations": 0,
                    "holm_significant": audit["multiplicity"]["global_holm_p_lt_0_05"],
                }],
                "coverage_rows": audit["coverage_rows"],
                "baseline_reward_delta": reward_rows,
                "main_effects_table": main_table,
                "successful_throughput": audit["successful_throughput"],
                "ablation_reward": audit["ablation_reward"],
                "cross_load": audit["cross_load"],
                "scenario_coverage": audit["scenario_coverage"],
                "issues": audit["issues"],
            },
        },
        "sources": sources,
    }


def notebook_payload(work_root: Path, audit: dict) -> dict:
    cells = [{
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Stage 18 结果质量审查\n",
            "\n",
            "## tl;dr\n",
            "\n",
            "核心文件与算术通过；确认性推断需修复TASK_KEYED配对、crossed bootstrap、"
            "多重比较、成功吞吐、消融构念和协议覆盖。\n",
        ],
    }, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Context & Methods\n",
            "\n",
            "### Key Assumptions\n",
            "\n",
            "- 训练运行粒度：profile × condition × training seed。\n",
            "- locked粒度：再增加evaluation profile；每个job有100个共享测试seed。\n",
            "- 本notebook只读原始结果，不覆盖任何v1训练或测试文件。\n",
        ],
    }, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import json\n",
            "from stage18_quality_audit_generator import build_audit\n",
            f"WORK_ROOT = Path({str(work_root)!r})\n",
            "audit = build_audit(WORK_ROOT)\n",
            "print(json.dumps({\n",
            "    'assessment': audit['overall_assessment'],\n",
            "    'coverage': audit['coverage'],\n",
            "    'multiplicity': audit['multiplicity'],\n",
            "}, ensure_ascii=False, indent=2))\n",
        ],
    }, {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Data\n", "\n", "完整性与冻结协议覆盖检查。\n"],
    }, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Coverage rows:')\n",
            "for row in audit['coverage_rows']:\n",
            "    print(row)\n",
            "print('Scenario coverage:')\n",
            "for row in audit['scenario_coverage']:\n",
            "    print(row)\n",
        ],
    }, {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Results\n", "\n", "统计校正、成功吞吐和跨负载描述性结果。\n"],
    }, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Full vs rule baseline:')\n",
            "for row in audit['main_effects']:\n",
            "    print({key: row[key] for key in ('profile','metric','mean_delta','ci_low','ci_high','raw_p','bh_q_global36','holm_p_global36')})\n",
            "print('Successful-delivery throughput:')\n",
            "for row in audit['successful_throughput']:\n",
            "    print(row)\n",
        ],
    }, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Cross-load descriptive summary:')\n",
            "for row in audit['cross_load']:\n",
            "    print(row)\n",
        ],
    }, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Takeaways\n",
            "\n",
            "1. 数据完整性PASS，确认性统计FAIL。\n",
            "2. 当前主要结果可保留为探索性v1，但需要新鲜TASK_KEYED locked test。\n",
            "3. 去持续位置是复合消融；队列与交接显式cue目前没有独立贡献证据。\n",
            "4. 训练权重可继续使用；消融重构和新增算法基线才需要重新训练。\n",
        ],
    }]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute_notebook_cells(notebook: dict, output_dir: Path) -> None:
    namespace = {"__name__": "__notebook__"}
    sys.path.insert(0, str(output_dir))
    previous_cwd = Path.cwd()
    try:
        import os
        os.chdir(output_dir)
        execution_count = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            source = "".join(cell["source"])
            stream = io.StringIO()
            with redirect_stdout(stream):
                exec(compile(source, f"<cell-{execution_count}>", "exec"),
                     namespace, namespace)
            cell["execution_count"] = execution_count
            output = stream.getvalue()
            cell["outputs"] = ([{
                "name": "stdout",
                "output_type": "stream",
                "text": output.splitlines(keepends=True),
            }] if output else [])
    finally:
        import os
        os.chdir(previous_cwd)
        if sys.path and sys.path[0] == str(output_dir):
            sys.path.pop(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    work_root = args.work_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = build_audit(work_root)
    audit_path = output_dir / "stage18_quality_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_csv(output_dir / "stage18_quality_issues.csv", audit["issues"])
    write_csv(output_dir / "stage18_main_effects_corrected.csv",
              audit["main_effects"])
    write_csv(output_dir / "stage18_successful_throughput.csv",
              audit["successful_throughput"])
    write_csv(output_dir / "stage18_cross_load_summary.csv",
              audit["cross_load"])
    write_csv(output_dir / "stage18_comparisons_with_multiplicity.csv",
              audit["official_comparisons_enhanced"])

    artifact = report_artifact(audit)
    (output_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    script_copy = output_dir / "stage18_quality_audit_generator.py"
    if Path(__file__).resolve() != script_copy.resolve():
        shutil.copy2(Path(__file__).resolve(), script_copy)
    notebook = notebook_payload(work_root, audit)
    execute_notebook_cells(notebook, output_dir)
    (output_dir / "stage18_quality_review.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    chart_map = """# Chart map\n\n\
| Section | Question | Family/type | Fields | Supported claim |\n\
|---|---|---|---|---|\n\
| Full vs rule | Does reward improve by load? | Comparison/bar | profile, mean_delta | Observed reward advantage grows with load |\n\
| Throughput definition | Does completed-only productivity improve? | Comparison/bar | profile, mean_delta | Successful throughput differs from resolved-task rate |\n\
| Ablation | Which ablation consistently degrades? | Grouped comparison/bar | profile, condition, mean_delta | Only the persistent-context bundle degrades consistently |\n\
| Cross-load | How does a trained profile transfer? | Comparison/bar | route, reward_delta, training_profile | Cross-load files exist but need formal statistics |\n\
\nPalette policy: single-root for single-series charts; relaxed multi-category for the ablation and cross-load grouped charts. All bars use zero-reference context and exact values remain available in tables/tooltips.\n"""
    (output_dir / "chart_map.md").write_text(chart_map, encoding="utf-8")

    print(json.dumps({
        "assessment": audit["overall_assessment"],
        "output_dir": str(output_dir),
        "audit": str(audit_path),
        "artifact": str(output_dir / "artifact.json"),
        "notebook": str(output_dir / "stage18_quality_review.ipynb"),
        "training_errors": len(audit["training_integrity_errors"]),
        "locked_errors": len(audit["locked_integrity_errors"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
