#!/usr/bin/env python3
"""Validate frozen Stage 21 evidence and build IEEE-ready vector figures.

Inputs are the manuscript copy of the Stage 21 release data.  The script checks
the supplied SHA-256 manifest, campaign completeness, CSV/JSON agreement, the
crossed-bootstrap confidence intervals, and the latency benchmark grid before
creating two vector PDF figures with ReportLab.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = next(
    (
        path
        for path in (FIG_DIR / "data" / "stage21", ROOT / "data" / "stage21")
        if (path / "SHA256SUMS").exists()
    ),
    FIG_DIR / "data" / "stage21",
)

WIDTH = 515.5
TEXT = HexColor("#202124")
MUTED = HexColor("#5F6368")
GRID = HexColor("#DADCE0")
BLUE = HexColor("#1769AA")
BLUE_MID = HexColor("#4C91C6")
BLUE_LIGHT = HexColor("#8AB9D8")
ORANGE = HexColor("#D55E00")
PALE_BLUE = HexColor("#EAF3F8")
PALE_GRAY = HexColor("#F5F6F7")

SCENARIOS = (
    "CONTROL",
    "NAVIGATION_FAILURE",
    "STAIR_OUTAGE",
    "ROBOT_OUTAGE",
)
SCENARIO_LABELS = {
    "CONTROL": "Control",
    "NAVIGATION_FAILURE": "Navigation failure",
    "STAIR_OUTAGE": "Stair outage",
    "ROBOT_OUTAGE": "Robot outage",
}
COMPARISONS = (
    "PPO_MINUS_TIME_GREEDY_RULE",
    "PPO_MINUS_SINGLE_DOG_ONLY",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(a: float, b: float, atol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def validate_sha256() -> list[str]:
    checked = []
    for line in (DATA_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(None, 1)
        filename = filename.strip()
        actual = hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest()
        assert actual == digest, f"SHA-256 mismatch: {filename}"
        checked.append(filename)
    assert len(checked) == 9
    return checked


def validate_campaigns() -> dict:
    ppo = read_json(DATA_DIR / "campaign_ppo_01_10.status.json")
    baseline = read_json(DATA_DIR / "campaign_baselines_01_01.status.json")
    assert ppo["state"] == "COMPLETED" and baseline["state"] == "COMPLETED"
    assert ppo["job_count"] == len(ppo["jobs"]) == 40
    assert baseline["job_count"] == len(baseline["jobs"]) == 8
    assert all(job["state"] == "COMPLETED" for job in ppo["jobs"] + baseline["jobs"])
    assert ppo["task_count"] == 320_000
    assert baseline["task_count"] == 64_000
    freeze = read_json(DATA_DIR / "stage21_fault_protocol_freeze_manifest.json")
    assert freeze["hard_gate"]["passed"] is True
    assert all(freeze["hard_gate"]["assertions"].values())
    return {
        "ppo_jobs": ppo["job_count"],
        "baseline_jobs": baseline["job_count"],
        "ppo_tasks": ppo["task_count"],
        "baseline_tasks": baseline["task_count"],
        "freeze_gate_passed": True,
    }


def validate_fault_data() -> tuple[list[dict[str, str]], dict]:
    summary = read_json(DATA_DIR / "fault_robustness_locked_summary.json")
    assert summary["passed"] is True
    assert all(summary["assertions"].values())
    protocol = summary["protocol"]
    assert protocol["training_seed_count"] == 10
    assert protocol["test_seed_count"] == 100
    assert protocol["bootstrap_samples"] == 10_000

    absolute = read_csv(DATA_DIR / "fault_absolute.csv")
    comparisons = read_csv(DATA_DIR / "fault_comparisons.csv")
    degradation = read_csv(DATA_DIR / "fault_degradation_from_control.csv")
    assert len(absolute) == 108
    assert len(comparisons) == 70
    assert len(degradation) == 72

    for row in absolute:
        frozen = summary["absolute"][row["scenario"]][row["method"]][row["metric"]]
        if row["mean"]:
            assert close(row["mean"], frozen["mean"])
        else:
            assert frozen["mean"] is None
        if row["sample_count"]:
            assert int(row["sample_count"]) == frozen["sample_count"]

    for row in comparisons:
        frozen = summary["comparisons"][row["scenario"]][row["comparison"]][row["metric"]]
        ci = frozen.get("crossed_bootstrap_95ci", frozen.get("paired_bootstrap_95ci"))
        assert ci is not None
        assert close(row["difference"], frozen["difference"])
        assert close(row["ci95_low"], ci[0])
        assert close(row["ci95_high"], ci[1])
        assert float(row["ci95_low"]) <= float(row["difference"]) <= float(row["ci95_high"])

    for row in degradation:
        frozen = summary["degradation_from_control"][row["scenario"]][row["method"]][row["metric"]]
        ci = frozen.get("crossed_bootstrap_95ci", frozen.get("paired_bootstrap_95ci"))
        assert ci is not None
        assert close(row["difference_from_control"], frozen["difference"])
        assert close(row["ci95_low"], ci[0])
        assert close(row["ci95_high"], ci[1])
        assert float(row["ci95_low"]) <= float(row["difference_from_control"]) <= float(row["ci95_high"])

    selected = [
        row
        for row in comparisons
        if row["scenario"] in SCENARIOS
        and row["comparison"] in COMPARISONS
        and row["metric"] in {"mean_waiting_time", "p95_waiting_time"}
    ]
    assert len(selected) == 16
    return selected, {
        "absolute_rows": len(absolute),
        "comparison_rows": len(comparisons),
        "degradation_rows": len(degradation),
        "bootstrap_replicates": protocol["bootstrap_samples"],
        "selected_forest_points": len(selected),
    }


def validate_latency_data() -> tuple[list[dict[str, str]], dict]:
    formal = read_json(DATA_DIR / "latency_scaling_formal.json")
    rows = read_csv(DATA_DIR / "latency_summary.csv")
    assert formal["passed"] is True and all(formal["assertions"].values())
    assert len(rows) == len(formal["rows"]) == 54
    assert {int(row["team_size"]) for row in rows} == {4, 7, 10}
    assert {int(row["stair_count"]) for row in rows} == {1, 2, 4}
    assert {int(row["waiting_task_count"]) for row in rows} == {1, 2, 4, 8, 16, 32}

    components = {
        "candidate_generation_p99_ms": ("candidate_generation", "p99_ms"),
        "legality_mask_p99_ms": ("legality_mask", "p99_ms"),
        "encoder_total_p99_ms": ("encoder_total", "p99_ms"),
        "policy_forward_p99_ms": ("policy_forward", "p99_ms"),
        "end_to_end_p50_ms": ("end_to_end_decision", "p50_ms"),
        "end_to_end_p95_ms": ("end_to_end_decision", "p95_ms"),
        "end_to_end_p99_ms": ("end_to_end_decision", "p99_ms"),
    }
    indexed = {row["scenario_id"]: row for row in formal["rows"]}
    for row in rows:
        frozen = indexed[row["scenario_id"]]
        for field in ("team_size", "stair_count", "waiting_task_count", "visible_task_count", "candidate_count", "legal_candidate_count"):
            assert int(row[field]) == int(frozen[field])
        for field, (component, statistic) in components.items():
            assert close(row[field], frozen["components"][component][statistic])
            assert float(row[field]) >= 0.0

    max_by_component = {
        "Candidate generation": max(float(row["candidate_generation_p99_ms"]) for row in rows),
        "Legality mask": max(float(row["legality_mask_p99_ms"]) for row in rows),
        "Complete encoder": max(float(row["encoder_total_p99_ms"]) for row in rows),
        "Policy forward": max(float(row["policy_forward_p99_ms"]) for row in rows),
        "End-to-end": max(float(row["end_to_end_p99_ms"]) for row in rows),
    }
    worst = max(rows, key=lambda row: float(row["end_to_end_p99_ms"]))
    assert worst["scenario_id"] == "R10_S4_Q16"
    assert max_by_component["End-to-end"] < formal["protocol"]["engineering_p99_budget_ms"]
    return rows, {
        "scenario_count": len(rows),
        "samples_per_component_per_scenario": formal["protocol"]["repeats_per_component"],
        "maximum_visible_tasks": formal["protocol"]["maximum_visible_tasks"],
        "maximum_candidate_slots": formal["protocol"]["maximum_candidate_slots"],
        "engineering_p99_budget_ms": formal["protocol"]["engineering_p99_budget_ms"],
        "worst_scenario": worst["scenario_id"],
        "worst_end_to_end_p99_ms": float(worst["end_to_end_p99_ms"]),
        "max_p99_ms": max_by_component,
    }


def metadata(c: canvas.Canvas, title: str) -> None:
    c.setTitle(title)
    c.setAuthor("Anonymous")
    c.setSubject("Stage 21 frozen-data manuscript figure")
    c.setCreator("make_stage21_figures.py (ReportLab)")


def marker(c: canvas.Canvas, x: float, y: float, color: Color, shape: str, size: float = 3.0, filled: bool = True) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color if filled else white)
    c.setLineWidth(0.8)
    if shape == "square":
        c.rect(x - size, y - size, 2 * size, 2 * size, fill=int(filled), stroke=1)
    elif shape == "triangle":
        p = c.beginPath()
        p.moveTo(x, y + size * 1.15)
        p.lineTo(x - size, y - size)
        p.lineTo(x + size, y - size)
        p.close()
        c.drawPath(p, fill=int(filled), stroke=1)
    elif shape == "diamond":
        p = c.beginPath()
        p.moveTo(x, y + size)
        p.lineTo(x - size, y)
        p.lineTo(x, y - size)
        p.lineTo(x + size, y)
        p.close()
        c.drawPath(p, fill=int(filled), stroke=1)
    else:
        c.circle(x, y, size, fill=int(filled), stroke=1)


def axes(c: canvas.Canvas, left: float, bottom: float, width: float, height: float) -> None:
    c.setStrokeColor(TEXT)
    c.setLineWidth(0.6)
    c.line(left, bottom, left + width, bottom)
    c.line(left, bottom, left, bottom + height)


def draw_latency_figure(rows: list[dict[str, str]], summary: dict) -> Path:
    path = FIG_DIR / "fig_stage21_latency_scaling.pdf"
    height = 238.0
    c = canvas.Canvas(str(path), pagesize=(WIDTH, height), pageCompression=1)
    metadata(c, "Stage 21 scheduling latency and fixed-shape scaling")
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, height, fill=1, stroke=0)

    # Panel (a): queue scaling at the maximum tested stair count.
    left, bottom, pw, ph = 36.0, 39.0, 225.0, 158.0
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 9.2)
    c.drawString(left, 220, "(a) End-to-end p99 at four stairs")
    axes(c, left, bottom, pw, ph)
    y_max = 5.2
    for tick in range(0, 6):
        y = bottom + ph * tick / y_max
        c.setStrokeColor(GRID)
        c.setLineWidth(0.4)
        c.line(left, y, left + pw, y)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.4)
        c.drawRightString(left - 4, y - 2.4, str(tick))
    queues = [1, 2, 4, 8, 16, 32]
    x_step = pw / (len(queues) - 1)
    for idx, queue in enumerate(queues):
        x = left + idx * x_step
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.4)
        c.drawCentredString(x, bottom - 12, str(queue))
    # The encoder observes at most eight waiting tasks; the plot intentionally
    # shows the measured plateau rather than extrapolating outside this shape.
    cap_x = left + 3.5 * x_step
    c.setStrokeColor(MUTED)
    c.setDash(2, 2)
    c.setLineWidth(0.55)
    c.line(cap_x, bottom, cap_x, bottom + ph)
    c.setDash()
    c.setFillColor(PALE_GRAY)
    c.rect(cap_x, bottom, left + pw - cap_x, ph, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7.2)
    c.drawCentredString((cap_x + left + pw) / 2, bottom + ph - 9, "top-8 visibility cap")

    styles = {
        4: (BLUE_LIGHT, "circle"),
        7: (BLUE_MID, "square"),
        10: (BLUE, "triangle"),
    }
    for team in (4, 7, 10):
        series = sorted(
            (row for row in rows if int(row["team_size"]) == team and int(row["stair_count"]) == 4),
            key=lambda row: int(row["waiting_task_count"]),
        )
        assert len(series) == 6
        color, shape = styles[team]
        points = []
        for idx, row in enumerate(series):
            x = left + idx * x_step
            value = float(row["end_to_end_p99_ms"])
            y = bottom + ph * value / y_max
            points.append((x, y))
        c.setStrokeColor(color)
        c.setLineWidth(1.2)
        for first, second in zip(points, points[1:]):
            c.line(first[0], first[1], second[0], second[1])
        for x, y in points:
            marker(c, x, y, color, shape, 2.7, filled=(team != 4))

    c.setFillColor(TEXT)
    c.setFont("Helvetica", 7.6)
    c.drawCentredString(left + pw / 2, 11, "Waiting tasks in queue")
    c.saveState()
    c.translate(10, bottom + ph / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "End-to-end p99 latency (ms)")
    c.restoreState()
    legend_y = 207.0
    legend_x = left + 78
    for offset, team in enumerate((4, 7, 10)):
        color, shape = styles[team]
        x = legend_x + offset * 48
        marker(c, x, legend_y, color, shape, 2.5, filled=(team != 4))
        c.setFillColor(TEXT)
        c.setFont("Helvetica", 7.3)
        c.drawString(x + 5, legend_y - 2.3, f"{team} robots")

    # Panel (b): maxima over all 54 scenarios.
    left2, bottom2, pw2, ph2 = 306.0, 39.0, 194.0, 158.0
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 9.2)
    c.drawString(left2 - 6, 220, "(b) Maximum component p99 over 54 scenarios")
    labels = ["Candidate generation", "Legality mask", "Complete encoder", "Policy forward", "End-to-end"]
    values = [summary["max_p99_ms"][label] for label in labels]
    bar_left = left2 + 71
    bar_width = pw2 - 71
    max_x = 5.2
    axes(c, bar_left, bottom2, bar_width, ph2)
    for tick in range(0, 6):
        x = bar_left + bar_width * tick / max_x
        c.setStrokeColor(GRID)
        c.setLineWidth(0.4)
        c.line(x, bottom2, x, bottom2 + ph2)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.4)
        c.drawCentredString(x, bottom2 - 12, str(tick))
    y_step = ph2 / 5
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = bottom2 + ph2 - (idx + 0.68) * y_step
        c.setFillColor(TEXT)
        c.setFont("Helvetica", 7.2)
        c.drawRightString(bar_left - 5, y + 1.1, label)
        c.setFillColor(BLUE if label == "End-to-end" else BLUE_MID)
        c.rect(bar_left, y - 4, bar_width * value / max_x, 8, fill=1, stroke=0)
        c.setFillColor(TEXT)
        c.setFont("Helvetica", 7.2)
        c.drawString(bar_left + bar_width * value / max_x + 3, y - 2.3, f"{value:.3f}")
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 7.6)
    c.drawCentredString(bar_left + bar_width / 2, 11, "Maximum p99 latency (ms)")
    c.setFillColor(PALE_BLUE)
    c.roundRect(left2 + 3, 201, pw2 - 3, 13, 3, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 7.2)
    c.drawCentredString(left2 + pw2 / 2, 205, f"Worst p99 = {summary['worst_end_to_end_p99_ms']:.3f} ms; engineering budget = 100 ms")

    c.showPage()
    c.save()
    return path


def draw_forest_panel(c: canvas.Canvas, rows: list[dict[str, str]], metric: str, left: float, title: str, x_min: float, x_max: float, ticks: list[float]) -> None:
    bottom, pw, ph = 39.0, 204.0, 157.0
    label_width = 72.0
    plot_left = left + label_width
    plot_width = pw - label_width
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 9.2)
    c.drawString(left, 220, title)
    axes(c, plot_left, bottom, plot_width, ph)
    def sx(value: float) -> float:
        return plot_left + plot_width * (value - x_min) / (x_max - x_min)
    for tick in ticks:
        x = sx(tick)
        c.setStrokeColor(GRID if tick != 0 else MUTED)
        if tick == 0:
            c.setDash(2, 2)
        else:
            c.setDash()
        c.setLineWidth(0.55 if tick == 0 else 0.4)
        c.line(x, bottom, x, bottom + ph)
        c.setDash()
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.3)
        c.drawCentredString(x, bottom - 12, f"{tick:g}")
    group_step = ph / 4
    by_key = {(row["scenario"], row["comparison"]): row for row in rows if row["metric"] == metric}
    for idx, scenario in enumerate(SCENARIOS):
        center = bottom + ph - (idx + 0.5) * group_step
        c.setFillColor(TEXT)
        c.setFont("Helvetica", 7.4)
        c.drawRightString(plot_left - 6, center - 2.3, SCENARIO_LABELS[scenario])
        if idx:
            c.setStrokeColor(GRID)
            c.setLineWidth(0.35)
            c.line(plot_left, center + group_step / 2, plot_left + plot_width, center + group_step / 2)
        for comparison, offset, color, shape, filled in (
            ("PPO_MINUS_TIME_GREEDY_RULE", 4.0, BLUE, "circle", True),
            ("PPO_MINUS_SINGLE_DOG_ONLY", -4.0, ORANGE, "square", False),
        ):
            row = by_key[(scenario, comparison)]
            estimate = float(row["difference"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            y = center + offset
            c.setStrokeColor(color)
            c.setLineWidth(1.1)
            c.line(sx(low), y, sx(high), y)
            c.line(sx(low), y - 2.4, sx(low), y + 2.4)
            c.line(sx(high), y - 2.4, sx(high), y + 2.4)
            marker(c, sx(estimate), y, color, shape, 2.7, filled)
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 7.4)
    c.drawCentredString(plot_left + plot_width / 2, 11, "PPO - baseline difference (s); negative favors PPO")


def draw_fault_figure(rows: list[dict[str, str]]) -> Path:
    path = FIG_DIR / "fig_stage21_fault_waiting_effects.pdf"
    height = 238.0
    c = canvas.Canvas(str(path), pagesize=(WIDTH, height), pageCompression=1)
    metadata(c, "Stage 21 logical-fault waiting-time effects")
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, height, fill=1, stroke=0)
    draw_forest_panel(c, rows, "mean_waiting_time", 12.0, "(a) Mean waiting time", -250, 20, [-200, -150, -100, -50, 0])
    draw_forest_panel(c, rows, "p95_waiting_time", 276.0, "(b) 95th-percentile waiting time", -950, 50, [-800, -600, -400, -200, 0])

    # Shared legend uses both color and marker fill/shape so it survives grayscale.
    legend_y = 207.0
    marker(c, 155, legend_y, BLUE, "circle", 2.7, True)
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 7.4)
    c.drawString(161, legend_y - 2.3, "PPO - time-greedy")
    marker(c, 254, legend_y, ORANGE, "square", 2.7, False)
    c.setFillColor(TEXT)
    c.drawString(260, legend_y - 2.3, "PPO - single-dog")
    c.setFont("Helvetica-Oblique", 7.2)
    c.setFillColor(MUTED)
    c.drawRightString(WIDTH - 6, 226, "Markers: estimates; bars: crossed-bootstrap 95% CIs")
    c.showPage()
    c.save()
    return path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sha_files = validate_sha256()
    campaigns = validate_campaigns()
    fault_rows, fault_validation = validate_fault_data()
    latency_rows, latency_validation = validate_latency_data()
    latency_path = draw_latency_figure(latency_rows, latency_validation)
    fault_path = draw_fault_figure(fault_rows)
    validation = {
        "stage": 21,
        "status": "PASS",
        "source": "Frozen Stage 21 manuscript data copied from GitHub main commit 14110377c0607fe7deec0367513720f1874ffb7c",
        "sha256_files_checked": sha_files,
        "campaigns": campaigns,
        "fault_robustness": fault_validation,
        "latency_scaling": latency_validation,
        "outputs": [latency_path.name, fault_path.name],
    }
    (DATA_DIR / "stage21_figure_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
