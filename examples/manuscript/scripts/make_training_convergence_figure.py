#!/usr/bin/env python3
"""Build the fixed-load PPO optimization and validation-trajectory figure.

Chart contract
--------------
Question: Does the logged PPO optimization objective stabilize across the three
fixed-load cohorts, while validation task success remains feasible?
Takeaway: Medium's objective loss decreases strongly, whereas Dense and Burst
settle into profile-specific plateaus; validation success remains high.  PPO
loss is an optimization diagnostic rather than a direct performance measure.
Form: Two highlighted multi-series line panels.  Panel (a) shows the logged PPO
objective loss, summarized within non-overlapping 50-update blocks per seed;
panel (b) shows validation task success at the matching cadence.  Lines are
seed medians and bands are interquartile ranges.
Data sufficiency: 3 profiles x 10 independent seeds x 2000 PPO updates for loss
and 40 validation checkpoints (every 50 updates); every checkpoint contains
100 episodes and 2000 tasks.
Surface: Static vector PDF for an IEEEtran single-column float, 252 pt wide.
Palette: Explicit blue/orange/green roots.  Solid/dashed/dash-dot lines and
circle/square/triangle markers retain identity in grayscale.
Source: completed Stage 18 full-context PPO histories under results/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
FIGURE = ROOT / "figures" / "fig_training_convergence.pdf"
CSV_OUT = ROOT / "figures" / "data" / "training_convergence.csv"
VALIDATION = ROOT / "figures" / "data" / "training_convergence_validation.json"

WIDTH = 252.0
HEIGHT = 326.0
TEXT = HexColor("#202124")
MUTED = HexColor("#5F6368")
GRID = HexColor("#DADCE0")
ZERO = HexColor("#777777")

PROFILES = (
    {
        "name": "Medium",
        "glob": "results/stage18/full_context_v2/seed_*/stage18_ppo.history.json",
        "color": HexColor("#1769AA"),
        "fill": HexColor("#DDECF5"),
        "dash": None,
        "marker": "circle",
    },
    {
        "name": "Dense",
        "glob": "results/stage18_dense/full_context_v2/seed_*/stage18_ppo.history.json",
        "color": HexColor("#D55E00"),
        "fill": HexColor("#F7E2D5"),
        "dash": [5.0, 2.4],
        "marker": "square",
    },
    {
        "name": "Burst",
        "glob": "results/stage18_burst/full_context_v2/seed_*/stage18_ppo.history.json",
        "color": HexColor("#009E73"),
        "fill": HexColor("#DDF1EA"),
        "dash": [5.0, 2.0, 1.2, 2.0],
        "marker": "triangle",
    },
)

EXPECTED_UPDATES = list(range(50, 2001, 50))
LOSS_BLOCK_SIZE = 50
LOSS_FIELDS = (
    "loss",
    "actor_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
)
EXPECTED_PPO_CONFIG = {
    "clip_ratio": 0.2,
    "gae_lambda": 0.95,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.01,
    "max_gradient_norm": 1.0,
    "learning_rate": 3e-4,
    "ppo_epochs": 4,
    "minibatch_size": 32,
}


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPO)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_rows() -> tuple[list[dict], list[Path]]:
    rows: list[dict] = []
    source_paths: list[Path] = []
    for profile in PROFILES:
        paths = sorted(REPO.glob(profile["glob"]))
        assert len(paths) == 10, (profile["name"], len(paths))
        for path in paths:
            history = json.loads(path.read_text(encoding="utf-8"))
            assert len(history) == 2000
            assert [int(item["update"]) for item in history] == list(range(1, 2001))
            assert all(
                field in item and math.isfinite(float(item[field]))
                for item in history
                for field in LOSS_FIELDS
            )
            assert all(
                math.isclose(
                    float(item["loss"]),
                    float(item["actor_loss"])
                    + EXPECTED_PPO_CONFIG["value_coefficient"] * float(item["value_loss"])
                    - EXPECTED_PPO_CONFIG["entropy_coefficient"] * float(item["entropy"]),
                    rel_tol=0.0,
                    abs_tol=2e-8,
                )
                for item in history
            )
            evaluations = [item for item in history if "evaluation" in item]
            assert [int(item["update"]) for item in evaluations] == EXPECTED_UPDATES
            baseline_return = float(evaluations[0]["evaluation"]["mean_reward"])
            for item in evaluations:
                evaluation = item["evaluation"]
                update = int(item["update"])
                loss_block = history[update - LOSS_BLOCK_SIZE:update]
                assert [int(row["update"]) for row in loss_block] == list(
                    range(update - LOSS_BLOCK_SIZE + 1, update + 1)
                )
                assert int(evaluation["episode_count"]) == 100
                assert int(evaluation["task_count"]) == 2000
                assert evaluation["reward_units"] == "raw_environment_episode_sum"
                assert int(evaluation["illegal_action_count"]) == 0
                assert int(evaluation["resource_leak_count"]) == 0
                success_rate = float(evaluation["success_rate"])
                assert math.isclose(
                    success_rate,
                    float(evaluation["completed"]) / float(evaluation["task_count"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                mean_return = float(evaluation["mean_reward"])
                rows.append(
                    {
                        "profile": profile["name"],
                        "seed": path.parent.name,
                        "source_history": str(path.relative_to(REPO)),
                        "ppo_update": update,
                        "loss_block_start_update": update - LOSS_BLOCK_SIZE + 1,
                        "loss_block_end_update": update,
                        "ppo_objective_loss_block_median": quantile(
                            [float(row["loss"]) for row in loss_block], 0.50
                        ),
                        "ppo_actor_loss_block_median": quantile(
                            [float(row["actor_loss"]) for row in loss_block], 0.50
                        ),
                        "ppo_value_loss_block_median": quantile(
                            [float(row["value_loss"]) for row in loss_block], 0.50
                        ),
                        "ppo_entropy_block_median": quantile(
                            [float(row["entropy"]) for row in loss_block], 0.50
                        ),
                        "ppo_approximate_kl_block_median": quantile(
                            [float(row["approximate_kl"]) for row in loss_block], 0.50
                        ),
                        "ppo_clip_fraction_block_median": quantile(
                            [float(row["clip_fraction"]) for row in loss_block], 0.50
                        ),
                        "evaluation_episode_count": int(evaluation["episode_count"]),
                        "evaluation_task_count": int(evaluation["task_count"]),
                        "raw_environment_mean_return": mean_return,
                        "return_change_from_update_50": mean_return - baseline_return,
                        "task_success_rate_percent": 100.0 * success_rate,
                        "illegal_action_count": int(evaluation["illegal_action_count"]),
                        "resource_leak_count": int(evaluation["resource_leak_count"]),
                    }
                )
            source_paths.append(path)
            summary_path = path.with_name("stage18_ppo.summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert summary["ppo"] == EXPECTED_PPO_CONFIG
            assert int(summary["validation_seed_start"]) == 42_000_000
            assert int(summary["validation_seed_end"]) == 42_000_099
            source_paths.append(summary_path)
    assert len(rows) == 3 * 10 * 40
    return rows, source_paths


def aggregate(rows: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for profile in PROFILES:
        name = profile["name"]
        profile_rows = [row for row in rows if row["profile"] == name]
        points = []
        for update in EXPECTED_UPDATES:
            checkpoint = [row for row in profile_rows if row["ppo_update"] == update]
            assert len(checkpoint) == 10
            loss = [float(row["ppo_objective_loss_block_median"]) for row in checkpoint]
            success = [float(row["task_success_rate_percent"]) for row in checkpoint]
            points.append(
                {
                    "update": update,
                    "loss_q1": quantile(loss, 0.25),
                    "loss_median": quantile(loss, 0.50),
                    "loss_q3": quantile(loss, 0.75),
                    "success_q1": quantile(success, 0.25),
                    "success_median": quantile(success, 0.50),
                    "success_q3": quantile(success, 0.75),
                }
            )
        output[name] = points
    return output


def marker(c: canvas.Canvas, x: float, y: float, profile: dict, size: float = 2.7) -> None:
    color = profile["color"]
    shape = profile["marker"]
    c.setStrokeColor(color)
    c.setFillColor(color if shape == "circle" else white)
    c.setLineWidth(0.85)
    if shape == "square":
        c.rect(x - size, y - size, 2.0 * size, 2.0 * size, fill=1, stroke=1)
    elif shape == "triangle":
        path = c.beginPath()
        path.moveTo(x, y + size * 1.12)
        path.lineTo(x - size, y - size)
        path.lineTo(x + size, y - size)
        path.close()
        c.drawPath(path, fill=1, stroke=1)
    else:
        c.circle(x, y, size, fill=1, stroke=1)


def draw_legend(c: canvas.Canvas) -> None:
    starts = (16.0, 91.0, 162.0)
    c.setFont("Helvetica", 7.5)
    for profile, x in zip(PROFILES, starts):
        c.setStrokeColor(profile["color"])
        c.setLineWidth(1.35)
        c.setDash(profile["dash"] or [])
        c.line(x, 313.0, x + 20.0, 313.0)
        marker(c, x + 10.0, 313.0, profile, size=2.4)
        c.setFillColor(TEXT)
        c.drawString(x + 25.0, 310.3, profile["name"])
    c.setDash([])


def draw_panel(
    c: canvas.Canvas,
    aggregates: dict[str, list[dict]],
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
    title: str,
    metric: str,
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    y_label: str,
) -> None:
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(left - 35.0, bottom + height + 14.0, title)

    def x_coord(value: float) -> float:
        return left + (value - 50.0) / 1950.0 * width

    def y_coord(value: float) -> float:
        return bottom + (value - y_min) / (y_max - y_min) * height

    for value in y_ticks:
        y = y_coord(value)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.setDash([])
        c.line(left, y, left + width, y)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.1)
        c.drawRightString(left - 5.0, y - 2.4, f"{value:g}")

    for value in (50, 500, 1000, 1500, 2000):
        x = x_coord(value)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(x, bottom, x, bottom + height)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.0)
        c.drawCentredString(x, bottom - 10.0, f"{value}")

    if y_min < 0.0 < y_max:
        c.setStrokeColor(ZERO)
        c.setLineWidth(0.75)
        c.setDash([2.2, 2.2])
        c.line(left, y_coord(0.0), left + width, y_coord(0.0))
        c.setDash([])

    c.setStrokeColor(TEXT)
    c.setLineWidth(0.6)
    c.line(left, bottom, left + width, bottom)
    c.line(left, bottom, left, bottom + height)

    for profile in PROFILES:
        points = aggregates[profile["name"]]
        low_key = f"{metric}_q1"
        mid_key = f"{metric}_median"
        high_key = f"{metric}_q3"
        band = c.beginPath()
        first = points[0]
        band.moveTo(x_coord(first["update"]), y_coord(first[high_key]))
        for point in points[1:]:
            band.lineTo(x_coord(point["update"]), y_coord(point[high_key]))
        for point in reversed(points):
            band.lineTo(x_coord(point["update"]), y_coord(point[low_key]))
        band.close()
        c.setFillColor(profile["fill"])
        c.drawPath(band, fill=1, stroke=0)

    for profile in PROFILES:
        points = aggregates[profile["name"]]
        mid_key = f"{metric}_median"
        c.setStrokeColor(profile["color"])
        c.setLineWidth(1.35)
        c.setDash(profile["dash"] or [])
        path = c.beginPath()
        path.moveTo(x_coord(points[0]["update"]), y_coord(points[0][mid_key]))
        for point in points[1:]:
            path.lineTo(x_coord(point["update"]), y_coord(point[mid_key]))
        c.drawPath(path, fill=0, stroke=1)
        for index, point in enumerate(points):
            if index % 5 == 0 or index == len(points) - 1:
                marker(c, x_coord(point["update"]), y_coord(point[mid_key]), profile)
        c.setDash([])

    c.setFillColor(TEXT)
    c.setFont("Helvetica", 7.5)
    c.drawCentredString(left + width / 2.0, bottom - 24.0, "Post-warm-start PPO update")
    c.saveState()
    c.translate(left - 32.0, bottom + height / 2.0)
    c.rotate(90)
    c.drawCentredString(0.0, 0.0, y_label)
    c.restoreState()


def write_outputs(rows: list[dict], aggregates: dict[str, list[dict]], source_paths: list[Path]) -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    c = canvas.Canvas(str(FIGURE), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle("Fixed-load PPO optimization and validation trajectories")
    c.setAuthor("Anonymous")
    c.setSubject("Ten-seed Stage 18 PPO loss and validation trajectories")
    c.setCreator("make_training_convergence_figure.py (ReportLab)")
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    draw_legend(c)
    draw_panel(
        c,
        aggregates,
        left=51.0,
        bottom=176.0,
        width=188.0,
        height=94.0,
        title="(a) PPO objective loss (50-update blocks)",
        metric="loss",
        y_min=0.0,
        y_max=0.17,
        y_ticks=[0.0, 0.05, 0.10, 0.15],
        y_label="PPO objective loss",
    )
    draw_panel(
        c,
        aggregates,
        left=51.0,
        bottom=35.0,
        width=188.0,
        height=94.0,
        title="(b) Validation task success",
        metric="success",
        y_min=95.8,
        y_max=99.0,
        y_ticks=[96, 97, 98, 99],
        y_label="Task success (%)",
    )
    c.showPage()
    c.save()

    final_summary = {}
    for profile in PROFILES:
        points = aggregates[profile["name"]]
        minimum = min(points, key=lambda point: point["loss_median"])
        first = points[0]
        final = points[-1]
        final_summary[profile["name"]] = {
            "first_block_median_ppo_objective_loss": first["loss_median"],
            "minimum_median_ppo_objective_loss": minimum["loss_median"],
            "minimum_median_ppo_objective_loss_update": minimum["update"],
            "final_block_median_ppo_objective_loss": final["loss_median"],
            "final_median_success_percent": final["success_median"],
        }
    payload = {
        "passed": True,
        "question": "PPO optimization stability of the three fixed-load cohorts",
        "source_history_count": 30,
        "source_summary_count": 30,
        "combined_source_sha256": source_digest(source_paths),
        "profiles": [profile["name"] for profile in PROFILES],
        "training_seed_count_per_profile": 10,
        "training_update_count_per_seed": 2000,
        "loss_fields_verified": list(LOSS_FIELDS),
        "ppo_config_verified": EXPECTED_PPO_CONFIG,
        "ppo_objective_definition": "actor_loss + 0.5 * value_loss - 0.01 * entropy",
        "loss_block_width_updates": LOSS_BLOCK_SIZE,
        "validation_checkpoint_count_per_seed": 40,
        "checkpoint_updates": EXPECTED_UPDATES,
        "evaluation_episodes_per_checkpoint": 100,
        "evaluation_tasks_per_checkpoint": 2000,
        "plotted_statistic": "per-seed median within each non-overlapping 50-update loss block, then median and interquartile range across independent training seeds; validation success uses seed median and interquartile range",
        "temporal_smoothing": "none; non-overlapping 50-update block aggregation only",
        "validation_return_retained_in_csv_but_not_plotted": True,
        "validation_seed_range": [42_000_000, 42_000_099],
        "locked_test_reused": False,
        "final_summary": final_summary,
        "figure": str(FIGURE.relative_to(ROOT)),
        "csv": str(CSV_OUT.relative_to(ROOT)),
    }
    VALIDATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rows, source_paths = load_rows()
    aggregates = aggregate(rows)
    write_outputs(rows, aggregates, source_paths)
    print(FIGURE)
    print(CSV_OUT)
    print(VALIDATION)


if __name__ == "__main__":
    main()
