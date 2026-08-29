#!/usr/bin/env python3
"""Build the Stage 20 mixed-curriculum manuscript figure.

Chart contract
--------------
Question: Does one mixed-curriculum policy remain competitive with the rule
baseline while trading performance against fixed-load specialists?
Takeaway: Success and throughput are statistically comparable to time-greedy,
whereas mean and P95 waiting improve; fixed-load specialists retain advantages
on several matched metrics.
Form: Four faceted dot-and-interval panels with a common interpretation:
positive values favor mixed-curriculum PPO. The figure uses four comparison
rows (time-greedy plus three specialists); the single-dog control remains in the
exact-value table because its much larger waiting differences would compress the
specialist comparisons.
Surface: Static vector PDF for IEEEtran/Overleaf, 515.5 pt wide.
Palette: Blue for the rule baseline, orange open squares for specialists, gray
zero reference; marker shape and fill preserve meaning in grayscale.
Source: data/stage20/stage20_locked_test_summary.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "stage20" / "stage20_locked_test_summary.json"
FIGURE = ROOT / "figures" / "fig_stage20_generalization.pdf"
CSV_OUT = ROOT / "figures" / "data" / "stage20_generalization_differences.csv"
VALIDATION = ROOT / "data" / "stage20" / "stage20_figure_validation.json"

WIDTH = 515.5
HEIGHT = 252.0
TEXT = HexColor("#202124")
MUTED = HexColor("#5F6368")
GRID = HexColor("#DADCE0")
BLUE = HexColor("#1769AA")
ORANGE = HexColor("#D55E00")

COMPARATORS = (
    ("TIME_GREEDY_RULE", "Time-greedy", "rule"),
    ("STAGE19_MEDIUM", "Medium specialist", "specialist"),
    ("STAGE19_DENSE", "Dense specialist", "specialist"),
    ("STAGE19_BURST", "Burst specialist", "specialist"),
)

PANELS = (
    ("success_rate", "(a) Task success", "Advantage (percentage points)", 1.0),
    (
        "mean_episode_successful_throughput_tasks_per_hour",
        "(b) Successful throughput",
        "Advantage (tasks/h)",
        1.0,
    ),
    ("mean_waiting_time", "(c) Mean waiting time", "Improvement (s)", -1.0),
    (
        "mean_episode_p95_waiting_time",
        "(d) 95th-percentile waiting time",
        "Improvement (s)",
        -1.0,
    ),
)


def read_source() -> dict:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["bootstrap"]["design"] == "crossed_training_seed_by_shared_test_seed"
    assert payload["bootstrap"]["samples"] == 10_000
    assert payload["test_episode_count"] == 100
    assert payload["tasks_per_episode"] == 80
    return payload


def comparison_key(comparator: str) -> str:
    return f"STAGE20_MIXED_CURRICULUM_vs_{comparator}"


def rows_from(payload: dict) -> list[dict]:
    rows = []
    for metric, _, _, direction_scale in PANELS:
        for comparator, label, group in COMPARATORS:
            item = payload["primary_comparisons"][comparison_key(comparator)][metric]
            raw = float(item["a_minus_b"])
            ci_low, ci_high = map(float, item["crossed_bootstrap_95ci"])
            if metric == "success_rate":
                raw *= 100.0
                ci_low *= 100.0
                ci_high *= 100.0
            plotted = raw * direction_scale
            plotted_low = min(ci_low * direction_scale, ci_high * direction_scale)
            plotted_high = max(ci_low * direction_scale, ci_high * direction_scale)
            assert plotted_low <= plotted <= plotted_high
            rows.append(
                {
                    "metric": metric,
                    "comparator": comparator,
                    "comparator_label": label,
                    "group": group,
                    "mixed_minus_comparator": raw,
                    "raw_ci95_low": ci_low,
                    "raw_ci95_high": ci_high,
                    "plotted_advantage": plotted,
                    "plotted_ci95_low": plotted_low,
                    "plotted_ci95_high": plotted_high,
                }
            )
    assert len(rows) == 16
    return rows


def tick_values(low: float, high: float) -> list[float]:
    span = high - low
    candidates = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
    target = span / 4.0
    step = min(candidates, key=lambda value: abs(value - target))
    start = int(low // step) * step
    values = []
    value = start
    while value <= high + 1e-9:
        if value >= low - 1e-9:
            values.append(value)
        value += step
    return values


def fmt_tick(value: float) -> str:
    if abs(value) >= 10 or abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.1f}"


def draw_marker(c: canvas.Canvas, x: float, y: float, group: str) -> None:
    c.setLineWidth(0.9)
    if group == "rule":
        c.setStrokeColor(BLUE)
        c.setFillColor(BLUE)
        c.circle(x, y, 3.1, fill=1, stroke=1)
    else:
        c.setStrokeColor(ORANGE)
        c.setFillColor(white)
        c.rect(x - 3.0, y - 3.0, 6.0, 6.0, fill=1, stroke=1)


def draw_panel(
    c: canvas.Canvas,
    panel_rows: list[dict],
    left: float,
    bottom: float,
    width: float,
    height: float,
    title: str,
    xlabel: str,
) -> None:
    label_width = 72.0
    plot_left = left + label_width
    plot_width = width - label_width - 8.0
    values = [float(row[key]) for row in panel_rows for key in ("plotted_ci95_low", "plotted_ci95_high")]
    extent = max(max(abs(min(values)), abs(max(values))) * 1.13, 0.2)
    x_min, x_max = -extent, extent

    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 9.0)
    c.drawString(left, bottom + height + 13.0, title)

    row_gap = height / 4.6
    y_positions = [bottom + height - 0.7 * row_gap - idx * row_gap for idx in range(4)]
    axis_y = bottom + 4.0

    def x_coord(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * plot_width

    for tick in tick_values(x_min, x_max):
        x = x_coord(tick)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(x, axis_y, x, bottom + height)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.2)
        c.drawCentredString(x, axis_y - 8.0, fmt_tick(tick))

    zero_x = x_coord(0.0)
    c.setStrokeColor(MUTED)
    c.setLineWidth(0.8)
    c.line(zero_x, axis_y, zero_x, bottom + height)
    c.setStrokeColor(TEXT)
    c.setLineWidth(0.55)
    c.line(plot_left, axis_y, plot_left + plot_width, axis_y)

    for row, y in zip(panel_rows, y_positions):
        c.setFillColor(TEXT)
        c.setFont("Helvetica", 7.3)
        c.drawRightString(plot_left - 5.0, y - 2.2, row["comparator_label"])
        x_low = x_coord(float(row["plotted_ci95_low"]))
        x_high = x_coord(float(row["plotted_ci95_high"]))
        x_mid = x_coord(float(row["plotted_advantage"]))
        color = BLUE if row["group"] == "rule" else ORANGE
        c.setStrokeColor(color)
        c.setLineWidth(1.0)
        c.line(x_low, y, x_high, y)
        c.line(x_low, y - 2.5, x_low, y + 2.5)
        c.line(x_high, y - 2.5, x_high, y + 2.5)
        draw_marker(c, x_mid, y, row["group"])

    c.setFillColor(TEXT)
    c.setFont("Helvetica", 7.5)
    c.drawCentredString(plot_left + plot_width / 2.0, bottom - 15.0, xlabel)


def write_outputs(payload: dict, rows: list[dict]) -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(FIGURE), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle("Stage 20 mixed-curriculum generalization")
    c.setAuthor("Anonymous")
    c.setSubject("Frozen Stage 20 crossed-bootstrap comparisons")
    c.setCreator("make_stage20_figure.py (ReportLab)")
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)

    positions = (
        (10.0, 139.0),
        (266.0, 139.0),
        (10.0, 27.0),
        (266.0, 27.0),
    )
    for (metric, title, xlabel, _), (left, bottom) in zip(PANELS, positions):
        panel_rows = [row for row in rows if row["metric"] == metric]
        draw_panel(c, panel_rows, left, bottom, 240.0, 76.0, title, xlabel)

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7.0)
    c.drawRightString(WIDTH - 10.0, 4.5, "Positive values favor mixed-curriculum PPO")
    c.showPage()
    c.save()

    validation = {
        "passed": True,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "bootstrap_design": payload["bootstrap"]["design"],
        "bootstrap_samples": payload["bootstrap"]["samples"],
        "training_seed_count": 10,
        "shared_test_seed_count": payload["test_episode_count"],
        "plotted_point_count": len(rows),
        "figure": str(FIGURE.relative_to(ROOT)),
        "csv": str(CSV_OUT.relative_to(ROOT)),
        "single_dog_omitted_from_figure_reason": (
            "Its much larger waiting differences would compress the specialist comparisons; "
            "it remains in the manuscript table."
        ),
    }
    VALIDATION.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    payload = read_source()
    rows = rows_from(payload)
    write_outputs(payload, rows)
    print(f"wrote {FIGURE}")
    print(f"wrote {CSV_OUT}")
    print(f"wrote {VALIDATION}")


if __name__ == "__main__":
    main()
