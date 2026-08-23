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

PROFILES = ("MEDIUM", "DENSE", "BURST")
PHASES = ("NORMAL", "DENSE", "BURST", "RECOVERY")
PROFILE_LABELS = {"MEDIUM": "Medium", "DENSE": "Dense", "BURST": "Burst"}
COLORS = {
    "MEDIUM": HexColor("#0072B2"),
    "DENSE": HexColor("#D55E00"),
    "BURST": HexColor("#009E73"),
    "RULE": HexColor("#555555"),
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
        for expected_seed, path in enumerate(paths, start=1):
            payload = read_json(path)
            assert payload["training_seed_index"] == expected_seed
            episodes = payload["result"]["episodes"]
            assert len(episodes) == 100
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
    assert len(baseline_episodes) == 100
    baseline_metrics = {
        metric: np.asarray([episode[field] for episode in baseline_episodes], dtype=float)
        for metric, field in metric_fields.items()
    }
    baseline_adaptation = [episode_adaptation(episode) for episode in baseline_episodes]
    baseline = {
        "metrics": baseline_metrics,
        "phase_means": {
            phase: float(np.mean([row[0][phase] for row in baseline_adaptation]))
            for phase in PHASES
        },
        "recovery_mean": float(np.mean([row[1] for row in baseline_adaptation])),
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
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, bottom + height - 8.0, title)

    for tick in ticks:
        x = sx(tick)
        c.setStrokeColor(COLORS["GRID"])
        c.setLineWidth(0.4)
        c.line(x, plot_bottom, x, plot_bottom + plot_h)
        c.setFillColor(COLORS["TEXT"])
        c.setFont("Helvetica", 5.8)
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
        c.setFont("Helvetica", 6.2)
        c.drawRightString(plot_left - 4, y - 2, PROFILE_LABELS[profile])
        draw_h_error(c, sx(lo), sx(hi), y, COLORS[profile])
        draw_marker(c, sx(estimate), y, COLORS[profile], shapes[profile], 2.8)

    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica", 6.1)
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
    c.setFont("Helvetica-Bold", 7.5)
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
        c.setFont("Helvetica", 5.8)
        c.drawRightString(plot_left - 4, y - 2, str(tick))
    draw_axes_box(c, plot_left, plot_bottom, plot_w, plot_h)
    c.setFont("Helvetica", 6.2)
    for x, phase in zip(x_positions, PHASES):
        c.drawCentredString(x, plot_bottom - 10, phase.title())
    c.saveState()
    c.translate(left + 8, plot_bottom + plot_h / 2)
    c.rotate(90)
    c.setFont("Helvetica", 6.2)
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
        c.setFont("Helvetica", 5.7)
        c.drawString(x + 15, legend_y - 2, label)

    # Panel (b): recovery waiting-time change.
    left2, bottom2, panel_w2, panel_h2 = 337.0, 9.0, width - 345.0, height - 16.0
    plot_left2, plot_bottom2 = left2 + 62.0, bottom2 + 24.0
    plot_w2, plot_h2 = panel_w2 - 70.0, panel_h2 - 48.0
    c.setFillColor(COLORS["TEXT"])
    c.setFont("Helvetica-Bold", 7.5)
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
        c.setFont("Helvetica", 5.8)
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
        c.setFont("Helvetica", 6.1)
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
    c.setFont("Helvetica", 6.0)
    c.drawCentredString(plot_left2 + plot_w2 / 2, bottom2 + 1.5,
                        "Late - early recovery waiting time (s)")
    c.setFont("Helvetica-Oblique", 5.7)
    c.drawString(plot_left2 + 2, bottom2 + panel_h2 - 22, "Negative values indicate queue clearance")

    c.showPage()
    c.save()
    return path


def write_validation(summary: dict, outputs: list[Path]) -> None:
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
        "outputs": [str(path) for path in outputs],
    }
    (DATA_DIR / "stage19_figure_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary, raw, baseline = load_and_validate()
    write_csvs(summary, raw, baseline)
    outputs = [make_paired_figure(summary), make_adaptation_figure(summary, raw, baseline)]
    write_validation(summary, outputs)
    print(json.dumps({"outputs": [str(path) for path in outputs], "status": "PASS"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
