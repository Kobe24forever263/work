#!/usr/bin/env python3
"""Build curated Stage 1--25 tables and publication-grade teacher figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "exports" / "2026-08-31_stage1-25_experiment_package"
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
QA = PACKAGE / "figure_qa"
FIGURES.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#777777"


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "PingFang SC", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_rows(name: str, rows: list[dict[str, Any]]) -> Path:
    path = TABLES / name
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def lookup(rows: list[dict[str, Any]], match: dict[str, Any], value_key: str) -> float:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in match.items()):
            return float(row[value_key])
    raise KeyError(f"No row for {match}")


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str, source_note: str) -> None:
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.19, top=0.78)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    (FIGURES / f"{stem}.source.txt").write_text(source_note + "\n", encoding="utf-8")
    plt.close(fig)


def grouped_bars(
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    ylabel: str,
    title: str,
    subtitle: str,
    stem: str,
    source_note: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(labels))
    width = min(0.7 / max(len(series), 1), 0.28)
    start = -(len(series) - 1) * width / 2
    for index, (name, values, color) in enumerate(series):
        bars = ax.bar(x + start + index * width, values, width, label=name, color=color, edgecolor="none", linewidth=0)
        ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", y=1.12)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    if ylim:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend(frameon=False, ncol=min(3, len(series)), loc="upper center", bbox_to_anchor=(0.5, -0.17))
    style_axes(ax)
    # Bar labels are the primary quantitative guides; removing horizontal grid
    # strokes avoids collisions with labels near rounded tick levels.
    ax.grid(False)
    save_figure(fig, stem, source_note)


def mean_value(block: dict[str, Any], metric: str) -> float:
    value = block[metric]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def main() -> int:
    coverage = read_rows(TABLES / "stage_coverage.csv")

    # Curated source tables -------------------------------------------------
    stage18_path = ROOT / "results/stage18_locked_test_v2/stage18_locked_test_v2_statistics.csv"
    stage18 = read_rows(stage18_path)
    stage18_primary = [
        row for row in stage18
        if row["comparison"] == "full_context_v2_minus_rule_baseline"
        and row["metric"] in {"reward", "success_rate", "successful_throughput_tasks_per_hour"}
    ]
    write_rows("curated_stage18_primary_comparison.csv", stage18_primary)

    s19 = load_json("results/stage19_mixed_recovery/stage19_mixed_recovery_summary.json")
    s19_rows: list[dict[str, Any]] = []
    for profile, block in s19["profiles"].items():
        for metric, record in block["metrics"].items():
            s19_rows.append({
                "stage": 19, "method": f"PPO_{profile}", "profile": profile,
                "metric": metric, "value": record.get("mean"),
                "training_seed_sd": record.get("training_seed_sd"),
                "source_path": "results/stage19_mixed_recovery/stage19_mixed_recovery_summary.json",
            })
    for method, block in s19["baselines"].items():
        for metric, record in block["metrics"].items():
            value = record.get("mean") if isinstance(record, dict) else record
            seed_sd = record.get("training_seed_sd") if isinstance(record, dict) else ""
            s19_rows.append({
                "stage": 19, "method": method, "profile": "MIXED_STREAM",
                "metric": metric, "value": value,
                "training_seed_sd": seed_sd,
                "source_path": "results/stage19_mixed_recovery/stage19_mixed_recovery_summary.json",
            })
    write_rows("curated_stage19_mixed_recovery_metrics.csv", s19_rows)

    s20 = load_json("results/stage20_mixed_curriculum/stage20_locked_test_summary.json")
    s20_rows: list[dict[str, Any]] = []
    for method, block in {**s20["cohort_point_estimates"], **s20["baseline_point_estimates"]}.items():
        for metric, record in block.items():
            s20_rows.append({
                "stage": 20, "method": method, "metric": metric,
                "value": record.get("mean"),
                "training_seed_sd": record.get("training_seed_sd_of_test_seed_means"),
                "source_path": "results/stage20_mixed_curriculum/stage20_locked_test_summary.json",
            })
    write_rows("curated_stage20_mixed_curriculum_metrics.csv", s20_rows)

    s21 = load_json("results/stage21_external_validity/fault_robustness_locked_summary.json")
    s21_rows: list[dict[str, Any]] = []
    for scenario, methods in s21["absolute"].items():
        for method, metrics in methods.items():
            for metric, record in metrics.items():
                value = record.get("mean") if isinstance(record, dict) else record
                s21_rows.append({
                    "stage": 21, "scenario": scenario, "method": method,
                    "metric": metric, "value": value,
                    "source_path": "results/stage21_external_validity/fault_robustness_locked_summary.json",
                })
    write_rows("curated_stage21_fault_robustness_metrics.csv", s21_rows)

    s23 = load_json("results/stage23_markov_continuous/stage23_locked_test_summary.json")
    s23_rows: list[dict[str, Any]] = []
    methods23 = {"STAGE23_POLICY": s23["policy_point_estimates"], **s23["baseline_point_estimates"]}
    for method, metrics in methods23.items():
        for metric, record in metrics.items():
            s23_rows.append({
                "stage": 23, "method": method, "metric": metric,
                "value": record.get("mean"),
                "training_seed_sd": record.get("training_seed_sd_of_test_seed_means"),
                "source_path": "results/stage23_markov_continuous/stage23_locked_test_summary.json",
            })
    write_rows("curated_stage23_locked_test_metrics.csv", s23_rows)

    s24 = load_json("results/stage24_rolling_optimizer/pilot/stage24_optimizer_paired_pilot_summary.json")
    s24_rows: list[dict[str, Any]] = []
    for metric, record in s24["comparisons"].items():
        s24_rows.append({
            "stage": 24, "metric": metric, "optimizer": record["optimizer"],
            "time_greedy": record["time_greedy"],
            "optimizer_minus_time_greedy": record["optimizer_minus_time_greedy"],
            "ci_low": record["paired_percentile_bootstrap_95ci"][0],
            "ci_high": record["paired_percentile_bootstrap_95ci"][1],
            "direction": record["direction"], "stream_count": record["stream_count"],
            "evidence_level": "PILOT_ONLY",
            "source_path": "results/stage24_rolling_optimizer/pilot/stage24_optimizer_paired_pilot_summary.json",
        })
    write_rows("curated_stage24_optimizer_pilot_metrics.csv", s24_rows)

    stage25_long_relative = "results/stage25_causal_online/long/seed_01/sentinel_fresh_validation.json"
    stage25_smoke_relative = "results/stage25_causal_online/validation/stage25_causal_smoke_v2_task_keyed_fresh_validation.json"
    stage25_relative = stage25_long_relative if (ROOT / stage25_long_relative).exists() else stage25_smoke_relative
    s25 = load_json(stage25_relative)
    stage25_evidence_level = "LONG_SEED01_SENTINEL" if stage25_relative == stage25_long_relative else "SMOKE_FRESH_SEED_SAFETY_ONLY"
    s25_rows: list[dict[str, Any]] = []
    for method in ("policy", "rule_baseline"):
        for metric, value in s25[method].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                s25_rows.append({
                    "stage": 25, "method": method, "metric": metric,
                    "value": value, "evidence_level": stage25_evidence_level,
                    "source_path": stage25_relative,
                })
    write_rows("curated_stage25_causal_smoke_metrics.csv", s25_rows)

    # Figure 01: result file counts by stage.
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    stage_values = [int(row["stage"]) for row in coverage]
    result_counts = [int(row["total_result_files"]) for row in coverage]
    present_values = [row["results_present"].lower() == "true" for row in coverage]
    bars = ax.bar(stage_values, result_counts, color=[BLUE if value else "#D0D0D0" for value in present_values])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(range(1, 26))
    ax.set_xlabel("Stage")
    ax.set_ylabel("Result files (symlog scale)")
    ax.set_title("Current experimental evidence is concentrated in later stages", loc="left", fontweight="bold", y=1.12)
    ax.text(0, 1.025, "All files presently found under results/; zero means no result artifact found", transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    style_axes(ax)
    save_figure(fig, "fig01_stage_result_file_coverage", "Source: tables/stage_coverage.csv; workspace inventory generated 2026-08-31.")

    # Figure 02: data volume by stage.
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    values = np.array([float(row["data_volume_mb"]) for row in coverage], dtype=float)
    positive_values = np.where(values > 0, values, 0.001)
    if np.any(positive_values <= 0):
        raise ValueError("Log-scale data must be strictly positive")
    ax.bar(stage_values, positive_values, color=SKY)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 26))
    ax.set_xlabel("Stage")
    ax.set_ylabel("Data volume (MB, log scale)")
    ax.set_title("Checkpoint-heavy stages dominate storage", loc="left", fontweight="bold", y=1.12)
    ax.text(0, 1.025, "Binary checkpoints are inventoried but not duplicated in the teacher package", transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    style_axes(ax)
    save_figure(fig, "fig02_stage_data_volume", "Source: tables/stage_coverage.csv; file sizes from local filesystem.")

    # Stage 18 locked-test figures.
    profiles18 = ["MEDIUM", "DENSE", "BURST"]
    s18_success_left = [lookup(stage18_primary, {"profile": profile, "metric": "success_rate"}, "left_estimate") for profile in profiles18]
    s18_success_right = [lookup(stage18_primary, {"profile": profile, "metric": "success_rate"}, "right_estimate") for profile in profiles18]
    grouped_bars(
        ["Medium", "Dense", "Burst"],
        [("SMDP-PPO", s18_success_left, BLUE), ("Rule baseline", s18_success_right, ORANGE)],
        "Success rate", "Stage 18 locked-test success rate", "Ten training seeds; 100 shared test episodes per seed", "fig03_stage18_success_rate",
        "Source: results/stage18_locked_test_v2/stage18_locked_test_v2_statistics.csv.", (0.90, 1.005),
    )
    s18_thr_left = [lookup(stage18_primary, {"profile": profile, "metric": "successful_throughput_tasks_per_hour"}, "left_estimate") for profile in profiles18]
    s18_thr_right = [lookup(stage18_primary, {"profile": profile, "metric": "successful_throughput_tasks_per_hour"}, "right_estimate") for profile in profiles18]
    grouped_bars(
        ["Medium", "Dense", "Burst"],
        [("SMDP-PPO", s18_thr_left, BLUE), ("Rule baseline", s18_thr_right, ORANGE)],
        "Completed tasks per simulated hour", "Stage 18 successful throughput", "Completed-only throughput; same locked-test protocol as success rate", "fig04_stage18_successful_throughput",
        "Source: results/stage18_locked_test_v2/stage18_locked_test_v2_statistics.csv.", None,
    )

    # Stage 19 profile policy metrics.
    order19 = ["PPO_MEDIUM", "PPO_DENSE", "PPO_BURST", "time_greedy_rule", "single_dog_only"]
    grouped_bars(
        ["PPO-M", "PPO-D", "PPO-B", "Rule", "Dog only"],
        [("Mean waiting time", [lookup(s19_rows, {"method": item, "metric": "mean_waiting_time"}, "value") for item in order19], BLUE)],
        "Seconds (lower is better)", "Stage 19 mixed-load recovery: waiting time", "Profiles are fixed-trained PPO policies evaluated on the locked mixed stream", "fig05_stage19_mixed_waiting_time",
        "Source: results/stage19_mixed_recovery/stage19_mixed_recovery_summary.json.", None,
    )

    # Stage 20 mixed curriculum versus fixed policies and baselines.
    order20 = ["STAGE20_MIXED_CURRICULUM", "STAGE19_MEDIUM", "STAGE19_DENSE", "STAGE19_BURST", "TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"]
    grouped_bars(
        ["Mixed PPO", "Fixed M", "Fixed D", "Fixed B", "Rule", "Dog only"],
        [("Mean episode P95", [lookup(s20_rows, {"method": item, "metric": "mean_episode_p95_waiting_time"}, "value") for item in order20], GREEN)],
        "Seconds (lower is better)", "Stage 20 mixed-curriculum tail waiting time", "One continuous mixed-load episode; 100 shared test streams", "fig06_stage20_mixed_curriculum_p95_wait",
        "Source: results/stage20_mixed_curriculum/stage20_locked_test_summary.json.", None,
    )

    # Stage 21 external validity.
    scenarios = ["CONTROL", "NAVIGATION_FAILURE", "STAIR_OUTAGE", "ROBOT_OUTAGE"]
    series21 = []
    for method, label, color in [("PPO", "SMDP-PPO", BLUE), ("TIME_GREEDY_RULE", "Rule", ORANGE), ("SINGLE_DOG_ONLY", "Dog only", GRAY)]:
        series21.append((label, [lookup(s21_rows, {"method": method, "scenario": s, "metric": "success_rate"}, "value") for s in scenarios], color))
    grouped_bars(
        ["Control", "Nav failure", "Stair outage", "Robot outage"], series21,
        "Success rate", "Stage 21 logical fault robustness", "Logical SMDP faults only; not ROS, dynamics, or hardware fault evidence", "fig07_stage21_fault_success_rate",
        "Source: results/stage21_external_validity/fault_robustness_locked_summary.json.", (0.70, 1.01),
    )

    # Stage 23 formal comparison.
    order23 = ["STAGE23_POLICY", "TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"]
    labels23 = ["Stage 23 PPO", "Rule", "Dog only"]
    for metric, ylabel, title, stem, ylim in [
        ("success_rate", "Success rate", "Stage 23 locked-test success rate", "fig08_stage23_success_rate", (0.94, 1.0)),
        ("mean_episode_p95_waiting_time", "Seconds (lower is better)", "Stage 23 locked-test tail waiting time", "fig09_stage23_p95_waiting_time", None),
    ]:
        grouped_bars(
            labels23, [(metric, [lookup(s23_rows, {"method": item, "metric": metric}, "value") for item in order23], PURPLE)],
            ylabel, title, "Ten frozen policies; 100 shared held-out streams", stem,
            "Source: results/stage23_markov_continuous/stage23_locked_test_summary.json.", ylim,
        )

    # Stage 24 optimizer pilot.
    grouped_bars(
        ["Rolling optimizer", "Time greedy"],
        [("Mean episode P95", [s24["comparisons"]["episode_p95_waiting_time"]["optimizer"], s24["comparisons"]["episode_p95_waiting_time"]["time_greedy"]], RED)],
        "Seconds (lower is better)", "Stage 24 rolling optimizer pilot", "Pilot only: 20 paired streams; used for compute-budget selection", "fig10_stage24_optimizer_pilot_p95_wait",
        "Source: results/stage24_rolling_optimizer/pilot/stage24_optimizer_paired_pilot_summary.json.", None,
    )

    # Stage 25 smoke-only validation.
    grouped_bars(
        ["Causal PPO seed 01", "Rule"],
        [("Success rate", [s25["policy"]["success_rate"], s25["rule_baseline"]["success_rate"]], BLUE)],
        "Success rate", "Stage 25 sentinel fresh-seed validation", "Seed 01 long run; 40 streams; expansion gate failed", "fig11_stage25_smoke_success_rate",
        f"Source: {stage25_relative}.", (0.94, 1.005),
    )

    # Stage 23 training curve across ten long-run seeds.
    curve_rows: list[dict[str, float]] = []
    for path in sorted((ROOT / "results/stage23_markov_continuous/long").glob("seed_*/stage23_ppo.history.json")):
        seed = int(path.parent.name.split("_")[-1])
        history = json.loads(path.read_text(encoding="utf-8"))
        for record in history:
            update = record.get("update")
            reward = record.get("raw_episode_reward_mean", record.get("rollout_mean_reward"))
            if isinstance(update, int) and isinstance(reward, (int, float)):
                curve_rows.append({"seed": seed, "update": update, "reward": float(reward)})
    if curve_rows:
        grouped: dict[int, list[float]] = {}
        for row in curve_rows:
            grouped.setdefault(int(row["update"]), []).append(float(row["reward"]))
        updates = np.array(sorted(grouped), dtype=int)
        means = np.array([np.mean(grouped[int(update)]) for update in updates], dtype=float)
        stds = np.array([np.std(grouped[int(update)], ddof=1) if len(grouped[int(update)]) > 1 else 0.0 for update in updates], dtype=float)
        counts = np.array([len(grouped[int(update)]) for update in updates], dtype=int)
        kernel = np.ones(50, dtype=float)
        smooth = np.convolve(means, kernel, mode="full")[:len(means)] / np.minimum(np.arange(1, len(means) + 1), 50)
        smooth_sd = np.convolve(stds, kernel, mode="full")[:len(stds)] / np.minimum(np.arange(1, len(stds) + 1), 50)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(updates, smooth, color=BLUE, label="Mean across seeds (50-update smooth)")
        ax.fill_between(updates, smooth - smooth_sd, smooth + smooth_sd, color=SKY, alpha=0.25, linewidth=0)
        ax.set_xlabel("PPO update")
        ax.set_ylabel("Raw episode reward")
        ax.set_title("Stage 23 training trajectory across ten seeds", loc="left", fontweight="bold", y=1.12)
        ax.text(0, 1.025, "Band: across-seed standard deviation; reward is unnormalized environment return", transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
        style_axes(ax)
        ax.grid(False)
        save_figure(fig, "fig12_stage23_training_reward", "Source: results/stage23_markov_continuous/long/seed_*/stage23_ppo.history.json; ten seeds.")
        write_rows("curated_stage23_training_curve.csv", [
            {"update": int(update), "mean": mean, "std": std, "count": int(count), "smooth": sm, "smooth_sd": ssd}
            for update, mean, std, count, sm, ssd in zip(updates, means, stds, counts, smooth, smooth_sd)
        ])

    limitations = [
        {"scope": "Stage 1-4", "status": "No result file found", "interpretation": "Only support/code evidence may exist; do not report numerical performance."},
        {"scope": "Stage 5-13", "status": "Early gates/logs present", "interpretation": "Useful for implementation traceability, not necessarily formal statistical claims."},
        {"scope": "Stage 18", "status": "Multiple versions coexist", "interpretation": "Use locked-test v2 for primary performance; retain earlier reviews as protocol history."},
        {"scope": "Stage 19-21", "status": "Formal summaries present", "interpretation": "Use source-specific claim boundaries, especially for logical fault injection."},
        {"scope": "Stage 22", "status": "Training artifacts and short gates present", "interpretation": "No single final locked-test summary located in the current workspace."},
        {"scope": "Stage 23", "status": "Formal locked summary present", "interpretation": "Strongest current held-out evidence; guardrails remain descriptive where margins were not preregistered."},
        {"scope": "Stage 24", "status": "Pilot only", "interpretation": "Direction/compute-budget evidence only, not a confirmatory performance claim."},
        {"scope": "Stage 25", "status": "Seed 01 long run complete; sentinel expansion gate failed", "interpretation": "Do not expand the same policy to seeds 02-10; test a recurrent policy before further long training."},
    ]
    write_rows("data_quality_and_claim_boundaries.csv", limitations)

    figure_manifest = []
    for png in sorted(FIGURES.glob("*.png")):
        stem = png.stem
        figure_manifest.append({
            "figure": stem,
            "png": png.relative_to(PACKAGE).as_posix(),
            "pdf": f"figures/{stem}.pdf",
            "svg": f"figures/{stem}.svg",
            "tiff": f"figures/{stem}.tiff",
            "source_note": f"figures/{stem}.source.txt",
        })
    write_rows("figure_manifest.csv", figure_manifest)
    print(json.dumps({"figure_count": len(figure_manifest), "figures": [row["figure"] for row in figure_manifest]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
