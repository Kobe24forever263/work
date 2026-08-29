#!/usr/bin/env python3
"""Build the Stage 19 manuscript figures from the locked-test evidence.

The script independently reconstructs point estimates from the per-training-seed
JSON files, checks them against the frozen Stage 19 summary, writes plotting data
to CSV, and creates vector PDF figures suitable for IEEEtran/Overleaf.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, HexColor


SOURCE_ROOT = Path("/Users/lab4099/Desktop/Mujoco/work/results/stage19_mixed_recovery")
LOCKED = SOURCE_ROOT / "locked_test_v1"
SUMMARY_PATH = SOURCE_ROOT / "stage19_mixed_recovery_summary.json"
OUT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = OUT_ROOT / "figures"
DATA_DIR = FIG_DIR / "data"
TABLE_DIR = OUT_ROOT / "tables"

PROFILES = ("MEDIUM", "DENSE", "BURST")
PHASES = ("NORMAL", "DENSE", "BURST", "RECOVERY")
PROFILE_LABELS = {"MEDIUM": "Medium", "DENSE": "Dense", "BURST": "Burst"}
CURVE_POINTS_PER_PHASE = 41
COLORS = {
    "MEDIUM": HexColor("#0072B2"),
    "DENSE": HexColor("#D55E00"),
    "BURST": HexColor("#009E73"),
    "RULE": HexColor("#555555"),
    "SINGLE_DOG": HexColor("#CC7A00"),
    "GRID": HexColor("#D9D9D9"),
    "TEXT": HexColor("#1F1F1F"),
}
LIGHT_COLORS = {
    "MEDIUM": HexColor("#A7D5EE"),
    "DENSE": HexColor("#F2BEA4"),
    "BURST": HexColor("#A7DDCC"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_adaptation(episode: dict) -> tuple[dict[str, float], float]:
    shares = {
        phase: episode["transport_modes_by_phase"].get(phase, {}).get(
            "CAR_DOG_CAR", 0
        ) / 20.0
        for phase in PHASES
    }
    recovery = sorted(
        (task for task in episode["tasks"] if task["phase"] == "RECOVERY"),
        key=lambda task: task["arrival_time"],
    )
    assert len(recovery) == 20
    early = np.mean([task["waiting_time"] for task in recovery[:10]])
    late = np.mean([task["waiting_time"] for task in recovery[10:]])
    return shares, float(late - early)


def phase_bounds(episode: dict) -> dict[str, tuple[float, float]]:
    """Return half-open phase windows using the first arrival in each phase."""
    tasks = episode["tasks"]
    starts = {
        phase: min(float(task["arrival_time"]) for task in tasks if task["phase"] == phase)
        for phase in PHASES
    }
    ordered = [starts[phase] for phase in PHASES] + [float(episode["simulated_time"])]
    return {phase: (ordered[idx], ordered[idx + 1]) for idx, phase in enumerate(PHASES)}


def queue_length(tasks: list[dict], time_s: float, episode_end: float) -> int:
    return sum(
        float(task["arrival_time"]) <= time_s
        < (float(task["started_at"]) if task["started_at"] is not None else episode_end)
        for task in tasks
    )


def active_count(tasks: list[dict], time_s: float) -> int:
    return sum(
        task["started_at"] is not None
        and task["finished_at"] is not None
        and float(task["started_at"]) <= time_s < float(task["finished_at"])
        for task in tasks
    )


def interval_overlap(start: float, end: float, left: float, right: float) -> float:
    return max(0.0, min(end, right) - max(start, left))


def episode_queue_metrics(episode: dict) -> dict:
    """Reconstruct queue/workload dynamics from task-level timestamps.

    Q(t) counts arrived but not-yet-started tasks. Queue AUC is therefore exactly
    the accumulated waiting time in task-seconds. Queue recovery time is defined
    as the delay from the first Recovery arrival until every pre-Recovery task has
    started. Outstanding recovery time uses completion instead of start.
    """
    tasks = episode["tasks"]
    end = float(episode["simulated_time"])
    bounds = phase_bounds(episode)
    phase_metrics: dict[str, dict[str, float]] = {}
    curves: dict[str, dict[str, list[float]]] = {}
    for phase in PHASES:
        left, right = bounds[phase]
        auc = 0.0
        for task in tasks:
            wait_end = float(task["started_at"]) if task["started_at"] is not None else end
            auc += interval_overlap(float(task["arrival_time"]), wait_end, left, right)
        event_times = {left, right}
        for task in tasks:
            for field in ("arrival_time", "started_at", "finished_at"):
                value = task[field]
                if value is not None and left <= float(value) <= right:
                    event_times.add(float(value))
        phase_metrics[phase] = {
            "queue_auc_task_s": float(auc),
            "peak_queue": int(max(queue_length(tasks, t, end) for t in event_times)),
            "peak_active": int(max(active_count(tasks, t) for t in event_times)),
            "duration_s": float(right - left),
        }
        samples = np.linspace(left, right, CURVE_POINTS_PER_PHASE)
        curves[phase] = {
            "queue": [float(queue_length(tasks, float(t), end)) for t in samples],
            "active": [float(active_count(tasks, float(t))) for t in samples],
        }

    recovery_start = bounds["RECOVERY"][0]
    carried = [task for task in tasks if float(task["arrival_time"]) < recovery_start]
    queue_clear = max(
        [float(task["started_at"]) for task in carried if task["started_at"] is not None]
        + [recovery_start]
    ) - recovery_start
    outstanding_clear = max(
        [float(task["finished_at"]) for task in carried if task["finished_at"] is not None]
        + [recovery_start]
    ) - recovery_start
    return {
        "phase_metrics": phase_metrics,
        "queue_auc_total_task_s": float(sum(row["queue_auc_task_s"] for row in phase_metrics.values())),
        "peak_queue_episode": int(max(row["peak_queue"] for row in phase_metrics.values())),
        "peak_active_episode": int(max(row["peak_active"] for row in phase_metrics.values())),
        "queue_recovery_time_s": float(queue_clear),
        "outstanding_recovery_time_s": float(outstanding_clear),
        "curves": curves,
    }


def load_and_validate() -> tuple[dict, dict, dict]:
    summary = read_json(SUMMARY_PATH)
    quality = summary["data_quality"]
    assert quality["safe_for_formal_analysis"] is True
    assert quality["policy_tasks"] == 240000
    assert quality["baseline_tasks"] == 16000
    assert quality["task_stream_fingerprint_mismatches"] == 0
    assert quality["duplicate_policy_task_keys"] == 0
    assert quality["illegal_action_count"] == 0
    assert quality["resource_leak_count"] == 0

    raw: dict[str, dict] = {}
    metric_fields = {
        "success_rate": "success_rate",
        "successful_throughput_tasks_per_hour": "successful_throughput_tasks_per_hour",
        "mean_waiting_time": "mean_waiting_time",
        "p95_waiting_time": "p95_waiting_time",
    }
    for profile in PROFILES:
        paths = sorted((LOCKED / profile.lower()).glob("seed_*.json"))
        assert len(paths) == 10
        metric_arrays = {name: [] for name in metric_fields}
        phase_seed_means = {phase: [] for phase in PHASES}
        recovery_seed_means = []
        episodes_by_seed: list[list[dict]] = []
        source_paths: list[str] = []
        for expected_seed, path in enumerate(paths, start=1):
            payload = read_json(path)
            assert payload["training_seed_index"] == expected_seed
            episodes = payload["result"]["episodes"]
            assert len(episodes) == 100
            episodes_by_seed.append(episodes)
            source_paths.append(str(path))
            for episode in episodes:
                assert episode["task_count"] == 80
                assert len(episode["tasks"]) == 80
            for metric, field in metric_fields.items():
                metric_arrays[metric].append(
                    np.asarray([episode[field] for episode in episodes], dtype=float)
                )
            per_episode = [episode_adaptation(episode) for episode in episodes]
            for phase in PHASES:
                phase_seed_means[phase].append(
                    float(np.mean([row[0][phase] for row in per_episode]))
                )
            recovery_seed_means.append(float(np.mean([row[1] for row in per_episode])))

        metric_arrays = {
            metric: np.stack(values) for metric, values in metric_arrays.items()
        }
        raw[profile] = {
            "metrics": metric_arrays,
            "phase_seed_means": phase_seed_means,
            "recovery_seed_means": recovery_seed_means,
            "episodes_by_seed": episodes_by_seed,
            "source_paths": source_paths,
        }

        for metric, values in metric_arrays.items():
            frozen = summary["profiles"][profile]["metrics"][metric]["mean"]
            np.testing.assert_allclose(values.mean(), frozen, atol=1e-12, rtol=0)
        for phase in PHASES:
            frozen = summary["profiles"][profile]["adaptation"][
                f"cdc_share_{phase.lower()}"
            ]["estimate"]
            np.testing.assert_allclose(
                np.mean(phase_seed_means[phase]), frozen, atol=1e-12, rtol=0
            )
        frozen_recovery = summary["profiles"][profile]["adaptation"][
            "recovery_wait_late_minus_early"
        ]["estimate"]
        np.testing.assert_allclose(
            np.mean(recovery_seed_means), frozen_recovery, atol=1e-10, rtol=0
        )

    baseline_payload = read_json(LOCKED / "baselines.json")
    baseline_episodes = baseline_payload["time_greedy_rule"]["result"]["episodes"]
    single_dog_episodes = baseline_payload["single_dog_only"]["result"]["episodes"]
    assert len(baseline_episodes) == 100
    assert len(single_dog_episodes) == 100
    baseline_metrics = {
        metric: np.asarray([episode[field] for episode in baseline_episodes], dtype=float)
        for metric, field in metric_fields.items()
    }
    baseline_adaptation = [episode_adaptation(episode) for episode in baseline_episodes]
    baseline = {
        "metrics": baseline_metrics,
        "episodes": baseline_episodes,
        "source_path": str(LOCKED / "baselines.json"),
        "phase_means": {
            phase: float(np.mean([row[0][phase] for row in baseline_adaptation]))
            for phase in PHASES
        },
        "recovery_mean": float(np.mean([row[1] for row in baseline_adaptation])),
        "single_dog": {
            "episodes": single_dog_episodes,
            "metrics": {
                metric: np.asarray(
                    [episode[field] for episode in single_dog_episodes], dtype=float
                )
                for metric, field in metric_fields.items()
            },
        },
    }

    for profile in PROFILES:
        for metric in metric_fields:
            estimate = raw[profile]["metrics"][metric].mean() - baseline_metrics[metric].mean()
            frozen = summary["comparisons"][profile]["time_greedy_rule"][metric][
                "estimate"
            ]
            np.testing.assert_allclose(estimate, frozen, atol=1e-10, rtol=0)

    return summary, raw, baseline


def write_csvs(summary: dict, raw: dict, baseline: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "stage19_rule_differences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["profile", "metric", "estimate", "ci95_low", "ci95_high"])
        for profile in PROFILES:
            for metric in (
                "success_rate",
                "successful_throughput_tasks_per_hour",
                "mean_waiting_time",
                "p95_waiting_time",
            ):
                item = summary["comparisons"][profile]["time_greedy_rule"][metric]
                scale = 100.0 if metric == "success_rate" else 1.0
                writer.writerow(
                    [profile, metric, item["estimate"] * scale,
                     item["ci95"][0] * scale, item["ci95"][1] * scale]
                )

    with (DATA_DIR / "stage19_mode_share.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["profile", "training_seed", "phase", "cdc_share_percent",
             "aggregate_ci95_low_percent", "aggregate_ci95_high_percent"]
        )
        for profile in PROFILES:
            for phase in PHASES:
                item = summary["profiles"][profile]["adaptation"][
                    f"cdc_share_{phase.lower()}"
                ]
                for seed, value in enumerate(raw[profile]["phase_seed_means"][phase], 1):
                    writer.writerow(
                        [profile, seed, phase, value * 100.0,
                         item["ci95"][0] * 100.0, item["ci95"][1] * 100.0]
                    )
        for phase in PHASES:
            writer.writerow(["TIME_GREEDY_RULE", "", phase,
                             baseline["phase_means"][phase] * 100.0, "", ""])

    with (DATA_DIR / "stage19_recovery_wait.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["profile", "training_seed", "late_minus_early_wait_s",
             "aggregate_ci95_low_s", "aggregate_ci95_high_s"]
        )
        for profile in PROFILES:
            item = summary["profiles"][profile]["adaptation"][
                "recovery_wait_late_minus_early"
            ]
            for seed, value in enumerate(raw[profile]["recovery_seed_means"], 1):
                writer.writerow([profile, seed, value, item["ci95"][0], item["ci95"][1]])
        writer.writerow(["TIME_GREEDY_RULE", "", baseline["recovery_mean"], "", ""])


def mean_seed_curve(episodes: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    episode_rows = [episode_queue_metrics(episode)["curves"] for episode in episodes]
    return {
        phase: {
            field: np.mean(
                np.asarray([row[phase][field] for row in episode_rows], dtype=float),
                axis=0,
            )
            for field in ("queue", "active")
        }
        for phase in PHASES
    }


def build_p0_data(raw: dict, baseline: dict) -> dict:
    payload: dict = {"profiles": {}, "baselines": {}}
    for profile in PROFILES:
        seed_metrics = []
        seed_curves = []
        for seed_idx, episodes in enumerate(raw[profile]["episodes_by_seed"], start=1):
            episode_metrics = [episode_queue_metrics(episode) for episode in episodes]
            seed_metrics.append(episode_metrics)
            seed_curves.append(mean_seed_curve(episodes))
        payload["profiles"][profile] = {
            "seed_metrics": seed_metrics,
            "seed_curves": seed_curves,
        }

    for key, episodes in (
        ("TIME_GREEDY_RULE", baseline["episodes"]),
        ("SINGLE_DOG_ONLY", baseline["single_dog"]["episodes"]),
    ):
        payload["baselines"][key] = {
            "episode_metrics": [episode_queue_metrics(episode) for episode in episodes],
            "curve": mean_seed_curve(episodes),
        }
    return payload


def write_p0_csvs(summary: dict, raw: dict, baseline: dict, p0: dict) -> None:
    queue_path = DATA_DIR / "stage19_queue_episode_metrics.csv"
    phase_fields = [
        f"{phase.lower()}_{suffix}"
        for phase in PHASES
        for suffix in ("queue_auc_task_s", "peak_queue", "peak_active", "duration_s")
    ]
    with queue_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "method", "training_profile", "training_seed", "test_seed",
            "task_stream_fingerprint", "source_json", "task_count",
            "queue_auc_total_task_s", "peak_queue_episode", "peak_active_episode",
            "queue_recovery_time_s", "outstanding_recovery_time_s", *phase_fields,
        ])

        def write_episode_row(method: str, profile: str, train_seed: str | int,
                              episode: dict, metrics: dict, source_path: str) -> None:
            phase_values = []
            for phase in PHASES:
                row = metrics["phase_metrics"][phase]
                phase_values.extend([
                    row["queue_auc_task_s"], row["peak_queue"],
                    row["peak_active"], row["duration_s"],
                ])
            writer.writerow([
                method, profile, train_seed, episode["seed"],
                episode["task_stream_fingerprint"], source_path, episode["task_count"],
                metrics["queue_auc_total_task_s"], metrics["peak_queue_episode"],
                metrics["peak_active_episode"], metrics["queue_recovery_time_s"],
                metrics["outstanding_recovery_time_s"], *phase_values,
            ])

        for profile in PROFILES:
            for seed_idx, (episodes, metrics_rows, source_path) in enumerate(zip(
                raw[profile]["episodes_by_seed"],
                p0["profiles"][profile]["seed_metrics"],
                raw[profile]["source_paths"],
            ), start=1):
                for episode, metrics in zip(episodes, metrics_rows):
                    write_episode_row("PPO", profile, seed_idx, episode, metrics, source_path)
        for key, episodes in (
            ("TIME_GREEDY_RULE", baseline["episodes"]),
            ("SINGLE_DOG_ONLY", baseline["single_dog"]["episodes"]),
        ):
            for episode, metrics in zip(episodes, p0["baselines"][key]["episode_metrics"]):
                write_episode_row(key, key, "", episode, metrics, baseline["source_path"])

    with (DATA_DIR / "stage19_queue_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "method", "training_profile", "training_seed", "phase",
            "phase_fraction", "mean_queue_length", "mean_active_tasks",
            "test_episode_count", "source_scope",
        ])
        fractions = np.linspace(0.0, 1.0, CURVE_POINTS_PER_PHASE)
        for profile in PROFILES:
            for seed_idx, curve in enumerate(p0["profiles"][profile]["seed_curves"], start=1):
                for phase in PHASES:
                    for fraction, queue, active in zip(
                        fractions, curve[phase]["queue"], curve[phase]["active"]
                    ):
                        writer.writerow([
                            "PPO", profile, seed_idx, phase, fraction, queue, active,
                            100, raw[profile]["source_paths"][seed_idx - 1],
                        ])
        for key in ("TIME_GREEDY_RULE", "SINGLE_DOG_ONLY"):
            curve = p0["baselines"][key]["curve"]
            for phase in PHASES:
                for fraction, queue, active in zip(
                    fractions, curve[phase]["queue"], curve[phase]["active"]
                ):
                    writer.writerow([
                        key, key, "", phase, fraction, queue, active, 100,
                        baseline["source_path"],
                    ])

    with (DATA_DIR / "stage19_safety_efficiency_pareto.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "method", "training_profile", "training_seed", "success_rate_percent",
            "p95_waiting_time_s", "mean_waiting_time_s",
            "successful_throughput_tasks_per_hour", "distance_per_task",
            "test_episode_count",
        ])
        for profile in PROFILES:
            metrics = raw[profile]["metrics"]
            for seed_idx in range(10):
                writer.writerow([
                    "PPO", profile, seed_idx + 1,
                    metrics["success_rate"][seed_idx].mean() * 100.0,
                    metrics["p95_waiting_time"][seed_idx].mean(),
                    metrics["mean_waiting_time"][seed_idx].mean(),
                    metrics["successful_throughput_tasks_per_hour"][seed_idx].mean(),
                    np.mean([
                        episode["distance_per_task"]
                        for episode in raw[profile]["episodes_by_seed"][seed_idx]
                    ]),
                    100,
                ])
        baseline_summary = summary["baselines"]
        for key in ("time_greedy_rule", "single_dog_only"):
            item = baseline_summary[key]["metrics"]
            writer.writerow([
                key.upper(), key.upper(), "", item["success_rate"]["mean"] * 100.0,
                item["p95_waiting_time"]["mean"], item["mean_waiting_time"]["mean"],
                item["successful_throughput_tasks_per_hour"]["mean"],
                item["distance_per_task"]["mean"], 100,
            ])


def write_formal_table(summary: dict) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / "table_stage19_absolute.tex"
    rows = [
        ("PPO--Medium", summary["profiles"]["MEDIUM"]["metrics"]),
        ("PPO--Dense", summary["profiles"]["DENSE"]["metrics"]),
        ("PPO--Burst", summary["profiles"]["BURST"]["metrics"]),
        ("Time-greedy rule", summary["baselines"]["time_greedy_rule"]["metrics"]),
        ("Single-dog only", summary["baselines"]["single_dog_only"]["metrics"]),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Method & Success (\%) & Return & Throughput & Mean wait & P95 wait & Mean flow & Distance/task \\",
        r"\midrule",
    ]
    for label, metrics in rows:
        lines.append(
            f"{label} & {metrics['success_rate']['mean'] * 100.0:.3f} & "
            f"{metrics['reward']['mean']:.2f} & "
            f"{metrics['successful_throughput_tasks_per_hour']['mean']:.2f} & "
            f"{metrics['mean_waiting_time']['mean']:.2f} & "
            f"{metrics['p95_waiting_time']['mean']:.2f} & "
            f"{metrics['mean_flow_time']['mean']:.2f} & "
            f"{metrics['distance_per_task']['mean']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def set_pdf_metadata(c: canvas.Canvas, title: str) -> None:
    c.setTitle(title)
    c.setAuthor("Anonymous")
    c.setSubject("Stage 19 locked-test figure; generated from frozen JSON evidence")
    c.setCreator("make_stage19_figures.py (ReportLab)")


def draw_marker(c: canvas.Canvas, x: float, y: float, color: Color,
                shape: str = "circle", size: float = 3.0, fill: bool = True) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color if fill else Color(1, 1, 1))
    c.setLineWidth(0.7)
    if shape == "square":
        c.rect(x - size, y - size, 2 * size, 2 * size, fill=int(fill), stroke=1)
    elif shape == "triangle":
        path = c.beginPath()
        path.moveTo(x, y + size * 1.15)
        path.lineTo(x - size, y - size)
        path.lineTo(x + size, y - size)
        path.close()
        c.drawPath(path, fill=int(fill), stroke=1)
    elif shape == "diamond":
        path = c.beginPath()
        path.moveTo(x, y + size)
        path.lineTo(x - size, y)
        path.lineTo(x, y - size)
        path.lineTo(x + size, y)
        path.close()
        c.drawPath(path, fill=int(fill), stroke=1)
    else:
        c.circle(x, y, size, fill=int(fill), stroke=1)


def draw_axes_box(c: canvas.Canvas, left: float, bottom: float, width: float,
                  height: float) -> None:
    c.setStrokeColor(COLORS["TEXT"])
    c.setLineWidth(0.55)
    c.line(left, bottom, left + width, bottom)
    c.line(left, bottom, left, bottom + height)


def draw_h_error(c: canvas.Canvas, x0: float, x1: float, y: float,
                 color: Color, cap: float = 2.5, width: float = 1.15) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x0, y, x1, y)
    c.line(x0, y - cap, x0, y + cap)
    c.line(x1, y - cap, x1, y + cap)


def draw_v_error(c: canvas.Canvas, x: float, y0: float, y1: float,
                 color: Color, cap: float = 2.5, width: float = 1.0) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x, y0, x, y1)
    c.line(x - cap, y0, x + cap, y0)
    c.line(x - cap, y1, x + cap, y1)


def draw_paired_panel(c: canvas.Canvas, left: float, bottom: float,
                      width: float, height: float, title: str,
                      xlabel: str, xlim: tuple[float, float], ticks: Iterable[float],
                      values: dict[str, tuple[float, float, float]]) -> None:
    label_w, title_h, axis_h = 43.0, 13.0, 19.0
    plot_left = left + label_w
    plot_bottom = bottom + axis_h
    plot_w = width - label_w - 7.0
    plot_h = height - title_h - axis_h
    xmin, xmax = xlim

    def sx(value: float) -> float:
        return plot_left + (value - xmin) / (xmax - xmin) * plot_w

    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica-Bold", 8.6)
    c.drawString(left, bottom + height - 8.0, title)

    for tick in ticks:
        x = sx(tick)
        c.setStrokeColor(COLORS["GRID"])
        c.setLineWidth(0.4)
        c.line(x, plot_bottom, x, plot_bottom + plot_h)
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 7.0)
        label = f"{tick:g}"
        c.drawCentredString(x, plot_bottom - 8.2, label)
    if xmin < 0 < xmax:
        c.setStrokeColor(HexColor("#777777"))
        c.setDash(2, 2)
        c.setLineWidth(0.75)
        c.line(sx(0), plot_bottom, sx(0), plot_bottom + plot_h)
        c.setDash()
    draw_axes_box(c, plot_left, plot_bottom, plot_w, plot_h)

    shapes = {"MEDIUM": "circle", "DENSE": "square", "BURST": "triangle"}
    ys = np.linspace(plot_bottom + plot_h * 0.78, plot_bottom + plot_h * 0.22, 3)
    for profile, y in zip(PROFILES, ys):
        estimate, lo, hi = values[profile]
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 7.2)
        c.drawRightString(plot_left - 4, y - 2, PROFILE_LABELS[profile])
        draw_h_error(c, sx(lo), sx(hi), y, COLORS[profile])
        draw_marker(c, sx(estimate), y, COLORS[profile], shapes[profile], 2.8)

    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica", 7.2)
    c.drawCentredString(plot_left + plot_w / 2, bottom + 1.5, xlabel)


def make_paired_figure(summary: dict) -> Path:
    path = FIG_DIR / "fig_stage19_paired_differences.pdf"
    width, height = 7.16 * 72, 3.28 * 72
    c = canvas.Canvas(str(path), pagesize=(width, height), pageCompression=1)
    set_pdf_metadata(c, "Stage 19 paired differences against the time-greedy rule")

    specs = [
        ("(a) Task success", "Difference (percentage points)", (-0.3, 2.7), (0, 1, 2),
         "success_rate", 100.0),
        ("(b) Successful throughput", "Difference (tasks/h)", (-0.2, 1.9), (0, 0.5, 1, 1.5),
         "successful_throughput_tasks_per_hour", 1.0),
        ("(c) Mean waiting time", "Difference (s)", (-60, 10), (-50, -25, 0),
         "mean_waiting_time", 1.0),
        ("(d) 95th-percentile waiting time", "Difference (s)", (-260, 20), (-200, -100, 0),
         "p95_waiting_time", 1.0),
    ]
    margin_x, margin_y, gap_x, gap_y = 5.0, 7.0, 7.0, 5.0
    panel_w = (width - 2 * margin_x - gap_x) / 2
    panel_h = (height - 2 * margin_y - gap_y) / 2
    for idx, spec in enumerate(specs):
        row, col = divmod(idx, 2)
        left = margin_x + col * (panel_w + gap_x)
        bottom = height - margin_y - (row + 1) * panel_h - row * gap_y
        title, xlabel, xlim, ticks, metric, scale = spec
        values = {}
        for profile in PROFILES:
            item = summary["comparisons"][profile]["time_greedy_rule"][metric]
            values[profile] = (
                item["estimate"] * scale,
                item["ci95"][0] * scale,
                item["ci95"][1] * scale,
            )
        draw_paired_panel(c, left, bottom, panel_w, panel_h, title, xlabel,
                          xlim, ticks, values)

    c.showPage()
    c.save()
    return path


def make_adaptation_figure(summary: dict, raw: dict, baseline: dict) -> Path:
    path = FIG_DIR / "fig_stage19_adaptation_recovery.pdf"
    width, height = 7.16 * 72, 3.08 * 72
    c = canvas.Canvas(str(path), pagesize=(width, height), pageCompression=1)
    set_pdf_metadata(c, "Stage 19 load adaptation and recovery")

    # Panel (a): relay share by phase.
    left, bottom, panel_w, panel_h = 8.0, 9.0, 322.0, height - 16.0
    plot_left, plot_bottom = left + 39.0, bottom + 24.0
    plot_w, plot_h = panel_w - 47.0, panel_h - 48.0
    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica-Bold", 8.6)
    c.drawString(left, bottom + panel_h - 8, "(a) Relay-mode share across a continuous load sequence")

    def sy_share(value: float) -> float:
        return plot_bottom + value / 40.0 * plot_h

    x_positions = np.linspace(plot_left + 5, plot_left + plot_w - 5, 4)
    for tick in (0, 10, 20, 30, 40):
        y = sy_share(tick)
        c.setStrokeColor(COLORS["GRID"])
        c.setLineWidth(0.4)
        c.line(plot_left, y, plot_left + plot_w, y)
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 7.0)
        c.drawRightString(plot_left - 4, y - 2, str(tick))
    draw_axes_box(c, plot_left, plot_bottom, plot_w, plot_h)
    c.setFont("Helvetica", 7.2)
    for x, phase in zip(x_positions, PHASES):
        c.drawCentredString(x, plot_bottom - 10, phase.title())
    c.saveState()
    c.translate(left + 8, plot_bottom + plot_h / 2)
    c.rotate(90)
    c.setFont("Helvetica", 7.2)
    c.drawCentredString(0, 0, "Car-dog-car task share (%)")
    c.restoreState()

    shapes = {"MEDIUM": "circle", "DENSE": "square", "BURST": "triangle"}
    dash = {"MEDIUM": (), "DENSE": (4, 2), "BURST": (1.2, 1.6)}
    for pidx, profile in enumerate(PROFILES):
        for phase_idx, phase in enumerate(PHASES):
            for seed_idx, value in enumerate(raw[profile]["phase_seed_means"][phase]):
                jitter = (seed_idx - 4.5) * 0.65 + (pidx - 1) * 0.22
                draw_marker(c, x_positions[phase_idx] + jitter, sy_share(value * 100),
                            LIGHT_COLORS[profile], shapes[profile], 1.25)
        aggregate = []
        for phase_idx, phase in enumerate(PHASES):
            item = summary["profiles"][profile]["adaptation"][
                f"cdc_share_{phase.lower()}"
            ]
            x = x_positions[phase_idx]
            y = sy_share(item["estimate"] * 100)
            draw_v_error(c, x, sy_share(item["ci95"][0] * 100),
                         sy_share(item["ci95"][1] * 100), COLORS[profile])
            aggregate.append((x, y))
        c.setStrokeColor(COLORS[profile])
        c.setLineWidth(1.3)
        c.setDash(*dash[profile]) if dash[profile] else c.setDash()
        for first, second in zip(aggregate[:-1], aggregate[1:]):
            c.line(first[0], first[1], second[0], second[1])
        c.setDash()
        for x, y in aggregate:
            draw_marker(c, x, y, COLORS[profile], shapes[profile], 2.7)

    rule_points = [(x, sy_share(baseline["phase_means"][phase] * 100))
                   for x, phase in zip(x_positions, PHASES)]
    c.setStrokeColor(COLORS["RULE"])
    c.setLineWidth(1.0)
    c.setDash(2, 2)
    for first, second in zip(rule_points[:-1], rule_points[1:]):
        c.line(first[0], first[1], second[0], second[1])
    c.setDash()
    for x, y in rule_points:
        draw_marker(c, x, y, COLORS["RULE"], "diamond", 2.5, fill=False)

    legend_y = bottom + panel_h - 21.0
    legend_items = [
        ("MEDIUM", "Medium", "circle"), ("DENSE", "Dense", "square"),
        ("BURST", "Burst", "triangle"), ("RULE", "Time-greedy", "diamond")
    ]
    legend_xs = (left + 58, left + 122, left + 179, left + 239)
    for x, (key, label, shape) in zip(legend_xs, legend_items):
        c.setStrokeColor(COLORS[key])
        c.setLineWidth(1.0)
        if key == "RULE":
            c.setDash(2, 2)
        c.line(x, legend_y, x + 12, legend_y)
        c.setDash()
        draw_marker(c, x + 6, legend_y, COLORS[key], shape, 2.1, fill=key != "RULE")
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 6.8)
        c.drawString(x + 15, legend_y - 2, label)

    # Panel (b): recovery waiting-time change.
    left2, bottom2, panel_w2, panel_h2 = 337.0, 9.0, width - 345.0, height - 16.0
    plot_left2, plot_bottom2 = left2 + 62.0, bottom2 + 24.0
    plot_w2, plot_h2 = panel_w2 - 70.0, panel_h2 - 48.0
    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica-Bold", 8.6)
    c.drawString(left2, bottom2 + panel_h2 - 8, "(b) Recovery queue clearance")
    xmin, xmax = -230.0, 20.0

    def sx_wait(value: float) -> float:
        return plot_left2 + (value - xmin) / (xmax - xmin) * plot_w2

    for tick in (-200, -150, -100, -50, 0):
        x = sx_wait(tick)
        c.setStrokeColor(COLORS["GRID"])
        c.setLineWidth(0.4)
        c.line(x, plot_bottom2, x, plot_bottom2 + plot_h2)
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 7.0)
        c.drawCentredString(x, plot_bottom2 - 9, str(tick))
    c.setStrokeColor(HexColor("#777777"))
    c.setDash(2, 2)
    c.line(sx_wait(0), plot_bottom2, sx_wait(0), plot_bottom2 + plot_h2)
    c.setDash()
    draw_axes_box(c, plot_left2, plot_bottom2, plot_w2, plot_h2)

    rows = ("MEDIUM", "DENSE", "BURST", "RULE")
    ys = np.linspace(plot_bottom2 + plot_h2 * 0.82, plot_bottom2 + plot_h2 * 0.18, 4)
    for row_idx, (key, y) in enumerate(zip(rows, ys)):
        label = PROFILE_LABELS.get(key, "Time-greedy")
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 7.2)
        c.drawRightString(plot_left2 - 4, y - 2, label)
        if key == "RULE":
            draw_marker(c, sx_wait(baseline["recovery_mean"]), y,
                        COLORS["RULE"], "diamond", 2.7, fill=False)
            continue
        for seed_idx, value in enumerate(raw[key]["recovery_seed_means"]):
            jitter = (seed_idx - 4.5) * 0.38
            draw_marker(c, sx_wait(value), y + jitter, LIGHT_COLORS[key],
                        shapes[key], 1.25)
        item = summary["profiles"][key]["adaptation"][
            "recovery_wait_late_minus_early"
        ]
        draw_h_error(c, sx_wait(item["ci95"][0]), sx_wait(item["ci95"][1]),
                     y, COLORS[key])
        draw_marker(c, sx_wait(item["estimate"]), y, COLORS[key], shapes[key], 2.8)

    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica", 7.1)
    c.drawCentredString(plot_left2 + plot_w2 / 2, bottom2 + 1.5,
                        "Late - early recovery waiting time (s)")
    c.setFont("Helvetica-Oblique", 6.7)
    c.drawString(plot_left2 + 2, bottom2 + panel_h2 - 22, "Negative = queue clearance")

    c.showPage()
    c.save()
    return path


def draw_phase_axis(c: canvas.Canvas, left: float, bottom: float, width: float,
                    height: float, ymax: float, yticks: Iterable[float],
                    ylabel: str) -> tuple[callable, callable]:
    def sx(value: float) -> float:
        return left + value / 4.0 * width

    def sy(value: float) -> float:
        return bottom + value / ymax * height

    for phase_idx in range(4):
        if phase_idx % 2:
            c.setFillColor(HexColor("#F5F5F5"))
            c.rect(sx(phase_idx), bottom, sx(phase_idx + 1) - sx(phase_idx),
                   height, fill=1, stroke=0)
    for tick in yticks:
        y = sy(float(tick))
        c.setStrokeColor(COLORS["GRID"])
        c.setLineWidth(0.4)
        c.line(left, y, left + width, y)
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 5.8)
        c.drawRightString(left - 4, y - 2, f"{tick:g}")
    for boundary in (1, 2, 3):
        c.setStrokeColor(HexColor("#777777"))
        c.setDash(2, 2)
        c.setLineWidth(0.6)
        c.line(sx(boundary), bottom, sx(boundary), bottom + height)
    c.setDash()
    draw_axes_box(c, left, bottom, width, height)
    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica", 6.2)
    for idx, phase in enumerate(PHASES):
        c.drawCentredString(sx(idx + 0.5), bottom - 10, phase.title())
    c.saveState()
    c.translate(left - 27, bottom + height / 2)
    c.rotate(90)
    c.setFont("Helvetica", 6.2)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()
    return sx, sy


def flattened_curve(curve: dict, field: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for phase_idx, phase in enumerate(PHASES):
        fractions = np.linspace(0.0, 1.0, CURVE_POINTS_PER_PHASE)
        start = 0 if phase_idx == 0 else 1
        xs.extend((phase_idx + fractions[start:]).tolist())
        ys.extend(np.asarray(curve[phase][field], dtype=float)[start:].tolist())
    return np.asarray(xs), np.asarray(ys)


def make_queue_timeline_figure(p0: dict) -> Path:
    path = FIG_DIR / "fig_stage19_queue_timeline.pdf"
    width, height = 7.16 * 72, 3.18 * 72
    c = canvas.Canvas(str(path), pagesize=(width, height), pageCompression=1)
    set_pdf_metadata(c, "Stage 19 reconstructed queue and active-task timeline")

    profile_curves = {
        profile: {
            phase: {
                field: np.mean(np.asarray([
                    seed_curve[phase][field]
                    for seed_curve in p0["profiles"][profile]["seed_curves"]
                ], dtype=float), axis=0)
                for field in ("queue", "active")
            }
            for phase in PHASES
        }
        for profile in PROFILES
    }
    profile_curves["RULE"] = p0["baselines"]["TIME_GREEDY_RULE"]["curve"]
    profile_curves["SINGLE_DOG"] = p0["baselines"]["SINGLE_DOG_ONLY"]["curve"]
    shapes = {
        "MEDIUM": "circle", "DENSE": "square", "BURST": "triangle",
        "RULE": "diamond", "SINGLE_DOG": "circle",
    }
    dashes = {
        "MEDIUM": (), "DENSE": (4, 2), "BURST": (1.2, 1.6),
        "RULE": (2, 2), "SINGLE_DOG": (6, 2),
    }
    labels = {
        "MEDIUM": "PPO--Medium", "DENSE": "PPO--Dense", "BURST": "PPO--Burst",
        "RULE": "Time-greedy", "SINGLE_DOG": "Single-dog",
    }

    panels = [
        ("(a) Waiting queue", "queue", 30.0, (0, 10, 20, 30), "Waiting tasks"),
        ("(b) Concurrent execution", "active", 4.0, (0, 1, 2, 3, 4), "Active tasks"),
    ]
    margin_x, gap, panel_bottom = 39.0, 34.0, 31.0
    panel_w = (width - margin_x - 16.0 - gap) / 2
    panel_h = height - 63.0
    for panel_idx, (title, field, ymax, yticks, ylabel) in enumerate(panels):
        left = margin_x + panel_idx * (panel_w + gap)
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(left - 30, height - 12, title)
        sx, sy = draw_phase_axis(c, left, panel_bottom, panel_w, panel_h,
                                 ymax, yticks, ylabel)
        for key in ("MEDIUM", "DENSE", "BURST", "RULE", "SINGLE_DOG"):
            xs, ys = flattened_curve(profile_curves[key], field)
            c.setStrokeColor(COLORS[key])
            c.setLineWidth(1.2 if key in PROFILES else 0.95)
            if dashes[key]:
                c.setDash(*dashes[key])
            points = list(zip([sx(float(x)) for x in xs], [sy(float(y)) for y in ys]))
            for first, second in zip(points[:-1], points[1:]):
                c.line(first[0], first[1], second[0], second[1])
            c.setDash()
            for phase_mid in (0.5, 1.5, 2.5, 3.5):
                idx = int(round(phase_mid * (CURVE_POINTS_PER_PHASE - 1)))
                idx = min(idx, len(points) - 1)
                x, y = points[idx]
                draw_marker(c, x, y, COLORS[key], shapes[key], 1.9,
                            fill=key != "RULE")

    legend_y = 11.0
    legend_x = 18.0
    for idx, key in enumerate(("MEDIUM", "DENSE", "BURST", "RULE", "SINGLE_DOG")):
        x = legend_x + idx * 99.0
        c.setStrokeColor(COLORS[key])
        c.setLineWidth(1.0)
        if dashes[key]:
            c.setDash(*dashes[key])
        c.line(x, legend_y, x + 14, legend_y)
        c.setDash()
        draw_marker(c, x + 7, legend_y, COLORS[key], shapes[key], 2.0,
                    fill=key != "RULE")
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 5.8)
        c.drawString(x + 18, legend_y - 2, labels[key])
    c.setFont("Helvetica-Oblique", 5.6)
    c.drawRightString(width - 8, 2.5, "Time is normalized independently within each phase")
    c.showPage()
    c.save()
    return path


def make_pareto_figure(summary: dict, raw: dict) -> Path:
    path = FIG_DIR / "fig_stage19_safety_efficiency_pareto.pdf"
    width, height = 7.16 * 72, 3.05 * 72
    c = canvas.Canvas(str(path), pagesize=(width, height), pageCompression=1)
    set_pdf_metadata(c, "Stage 19 safety-efficiency Pareto view")
    left, bottom, plot_w, plot_h = 55.0, 34.0, width - 82.0, height - 62.0
    xmin, xmax = 450.0, 1350.0
    ymin, ymax = 96.0, 100.0

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def sy(value: float) -> float:
        return bottom + (value - ymin) / (ymax - ymin) * plot_h

    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica-Bold", 8.0)
    c.drawString(8, height - 11, "Safety--efficiency trade-off across independent training seeds")
    for tick in (500, 750, 1000, 1250):
        x = sx(tick)
        c.setStrokeColor(COLORS["GRID"])
        c.setLineWidth(0.4)
        c.line(x, bottom, x, bottom + plot_h)
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 5.8)
        c.drawCentredString(x, bottom - 9, str(tick))
    for tick in (96, 97, 98, 99, 100):
        y = sy(tick)
        c.setStrokeColor(COLORS["GRID"])
        c.line(left, y, left + plot_w, y)
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 5.8)
        c.drawRightString(left - 4, y - 2, str(tick))
    draw_axes_box(c, left, bottom, plot_w, plot_h)
    c.setFont("Helvetica", 6.2)
    c.drawCentredString(left + plot_w / 2, bottom - 20, "95th-percentile waiting time (s)")
    c.saveState()
    c.translate(12, bottom + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Task success (%)")
    c.restoreState()

    shapes = {"MEDIUM": "circle", "DENSE": "square", "BURST": "triangle"}
    for profile in PROFILES:
        success = raw[profile]["metrics"]["success_rate"].mean(axis=1) * 100.0
        p95 = raw[profile]["metrics"]["p95_waiting_time"].mean(axis=1)
        for xval, yval in zip(p95, success):
            draw_marker(c, sx(float(xval)), sy(float(yval)), COLORS[profile],
                        shapes[profile], 2.5, fill=False)
        draw_marker(c, sx(float(p95.mean())), sy(float(success.mean())), COLORS[profile],
                    shapes[profile], 4.0, fill=True)
        c.setFillColor(COLORS[profile])
        c.setFont("Helvetica-Bold", 6.2)
        c.drawString(sx(float(p95.mean())) + 6, sy(float(success.mean())) + 2,
                     PROFILE_LABELS[profile])

    for key, color_key, label, dx, dy in (
        ("time_greedy_rule", "RULE", "Time-greedy", 7, -8),
        ("single_dog_only", "SINGLE_DOG", "Single-dog", -60, 8),
    ):
        metrics = summary["baselines"][key]["metrics"]
        xval = metrics["p95_waiting_time"]["mean"]
        yval = metrics["success_rate"]["mean"] * 100.0
        draw_marker(c, sx(xval), sy(yval), COLORS[color_key], "diamond", 3.8,
                    fill=color_key != "RULE")
        c.setFillColor(COLORS[color_key])
        c.setFont("Helvetica-Bold", 6.2)
        c.drawString(sx(xval) + dx, sy(yval) + dy, label)
    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica-Oblique", 5.8)
    c.drawRightString(width - 8, 5, "Focused axes; upper-left indicates better safety and tail latency")
    c.showPage()
    c.save()
    return path


def write_validation(summary: dict, outputs: list[Path], table_path: Path) -> None:
    payload = {
        "source_summary": str(SUMMARY_PATH),
        "source_schema": summary["schema_version"],
        "source_created_at_utc": summary["created_at_utc"],
        "quality_severity": summary["data_quality"]["severity"],
        "safe_for_formal_analysis": summary["data_quality"]["safe_for_formal_analysis"],
        "point_estimates_reconstructed_from_raw_json": True,
        "raw_to_summary_tolerance": "absolute <= 1e-10",
        "bootstrap_intervals": summary["protocol"]["bootstrap_design"],
        "bootstrap_samples": summary["protocol"]["bootstrap_samples"],
        "queue_definition": "arrived but not started; half-open [arrival_time, started_at)",
        "queue_auc_definition": "integral of queue length in task-seconds; equals accumulated waiting time",
        "peak_queue_definition": "maximum reconstructed waiting-queue length over task events",
        "queue_recovery_time_definition": "seconds from first Recovery arrival until all pre-Recovery tasks have started",
        "outstanding_recovery_time_definition": "seconds from first Recovery arrival until all pre-Recovery tasks have finished",
        "timeline_alignment": "each phase independently normalized to [0,1]; no state reset at phase boundaries",
        "traceability_keys": ["training_profile", "training_seed", "test_seed", "task_stream_fingerprint", "task_id"],
        "formal_table": str(table_path),
        "plot_data": [
            str(DATA_DIR / "stage19_rule_differences.csv"),
            str(DATA_DIR / "stage19_mode_share.csv"),
            str(DATA_DIR / "stage19_recovery_wait.csv"),
            str(DATA_DIR / "stage19_queue_episode_metrics.csv"),
            str(DATA_DIR / "stage19_queue_curve.csv"),
            str(DATA_DIR / "stage19_safety_efficiency_pareto.csv"),
        ],
        "outputs": [str(path) for path in outputs],
    }
    (DATA_DIR / "stage19_figure_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary, raw, baseline = load_and_validate()
    write_csvs(summary, raw, baseline)
    p0 = build_p0_data(raw, baseline)
    write_p0_csvs(summary, raw, baseline, p0)
    table_path = write_formal_table(summary)
    outputs = [
        make_paired_figure(summary),
        make_adaptation_figure(summary, raw, baseline),
        make_queue_timeline_figure(p0),
        make_pareto_figure(summary, raw),
    ]
    write_validation(summary, outputs, table_path)
    print(json.dumps({
        "outputs": [str(path) for path in outputs],
        "formal_table": str(table_path),
        "status": "PASS",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
