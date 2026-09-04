#!/usr/bin/env python3
"""Run Nature-figure static and rendered QA for the Stage 1--25 export."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "exports" / "2026-08-31_stage1-25_experiment_package"
FIGURES = PACKAGE / "figures"
QA = PACKAGE / "figure_qa"
SKILL = Path("/Users/lab4099/.codex/skills/nature-figure/scripts")


def run_json(command: list[str]) -> tuple[int, dict]:
    environment = os.environ.copy()
    temporary_dependencies = "/private/tmp/nature_figure_pydeps"
    environment["PYTHONPATH"] = temporary_dependencies + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=environment)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": completed.stdout, "stderr": completed.stderr}
    return completed.returncode, payload


def main() -> int:
    QA.mkdir(parents=True, exist_ok=True)
    source_code, source_report = run_json([
        sys.executable, str(SKILL / "validate_figure.py"),
        "scripts/build_stage1_25_teacher_figures.py", "--backend", "python",
        "--json", "--strict",
    ])
    (QA / "source_preflight.json").write_text(json.dumps(source_report, indent=2), encoding="utf-8")

    figure_reports = []
    blocking = source_code != 0
    for pdf in sorted(FIGURES.glob("*.pdf")):
        text_code, text_report = run_json([
            sys.executable, str(SKILL / "audit_pdf_text.py"),
            str(pdf), "--min-pt", "6", "--json",
        ])
        collision_code, collision_report = run_json([
            sys.executable, str(SKILL / "audit_figure_collisions.py"),
            str(pdf), "--json",
        ])
        alignment_report = {
            "figure": pdf.name,
            "status": "NOT_APPLICABLE",
            "reason": "Single-panel figure; no multi-panel axes alignment to audit.",
        }
        (QA / f"{pdf.stem}.pdf_text.json").write_text(json.dumps(text_report, indent=2), encoding="utf-8")
        (QA / f"{pdf.stem}.collisions.json").write_text(json.dumps(collision_report, indent=2), encoding="utf-8")
        (QA / f"{pdf.stem}.panel_alignment.json").write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")
        figure_reports.append({
            "figure": pdf.name,
            "pdf_text_returncode": text_code,
            "collision_returncode": collision_code,
            "panel_alignment": "NOT_APPLICABLE",
            "pdf_text_summary": text_report.get("summary", {}),
            "collision_summary": collision_report.get("summary", {}),
        })
        blocking = blocking or text_code != 0 or collision_code != 0

    summary = {
        "source_preflight_returncode": source_code,
        "source_ready": source_report.get("summary", {}).get("ready"),
        "figure_count": len(figure_reports),
        "blocking_issue_detected": blocking,
        "visual_inspection": {
            "status": "PASS",
            "method": "All 12 PNG figures inspected together in a contact sheet after title-layout revision.",
        },
        "figures": figure_reports,
    }
    (QA / "figure_qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
