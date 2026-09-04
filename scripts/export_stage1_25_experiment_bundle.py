#!/usr/bin/env python3
"""Export all currently available Stage 1--25 evidence into compact tables.

The exporter is deliberately evidence-preserving: missing stages remain explicit,
binary checkpoints are inventoried rather than duplicated, and every normalized
row retains its source path.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "exports" / "2026-08-31_stage1-25_experiment_package"
STAGE_RE = re.compile(r"stage[_-]?(\d+)", re.IGNORECASE)
SEED_RE = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)
PROFILE_NAMES = ("MEDIUM", "DENSE", "BURST", "MIXED_CURRICULUM")

METRIC_KEYS = {
    "episode_count", "episodes", "episodes_trained_this_run",
    "episodes_trained_total", "task_count", "scheduled_task_count",
    "completed", "failed", "unresolved", "resolution_rate", "success_rate",
    "mean_reward", "reward", "terminal_adjusted_reward",
    "mean_terminal_adjusted_reward", "throughput_tasks_per_hour",
    "resolved_throughput_tasks_per_hour",
    "successful_throughput_tasks_per_hour", "mean_waiting_time",
    "mean_waiting_time_s", "p95_waiting_time", "p95_waiting_time_sensitivity",
    "mean_flow_time", "mean_flow_time_s", "distance_per_task",
    "distance_per_task_m", "illegal_action_count", "resource_leak_count",
    "maximum_active_tasks", "handover_timeout_count", "mean_simulated_time",
    "optimizer_mean_ms", "optimizer_p95_ms", "optimizer_max_ms",
    "objective", "objective_value", "regret", "mean_regret",
}

HISTORY_KEYS = (
    "update", "episodes_trained", "rollout_steps", "rollout_mean_reward",
    "raw_episode_reward_mean", "loss", "actor_loss", "value_loss", "entropy",
    "approximate_kl", "clip_fraction", "gradient_norm", "raw_reward_mean",
    "normalized_reward_mean", "normalized_reward_std",
    "return_normalizer_variance", "rollout_completed", "rollout_failed",
    "rollout_peak_active_tasks",
)


def stage_from_path(path: Path) -> int | None:
    match = STAGE_RE.search(path.as_posix())
    if match:
        value = int(match.group(1))
        if 1 <= value <= 25:
            return value
    return None


def seed_from_path(path: Path) -> int | None:
    matches = SEED_RE.findall(path.as_posix())
    return int(matches[-1]) if matches else None


def profile_from_text(text: str) -> str:
    upper = text.upper()
    for profile in PROFILE_NAMES:
        if profile in upper:
            return profile
    return ""


def classify(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return "model_checkpoint"
    if name.endswith(".history.json"):
        return "training_history"
    if suffix == ".jsonl":
        return "episode_or_event_stream"
    if suffix == ".csv":
        return "tabular_result"
    if "manifest" in name or "freeze" in name:
        return "manifest"
    if "summary" in name or "statistics" in name:
        return "summary"
    if "gate" in name:
        return "gate"
    if "status" in name or "progress" in name:
        return "status"
    if suffix in {".md", ".html", ".ipynb", ".png", ".svg", ".pdf"}:
        return "report_or_figure"
    if suffix == ".json":
        return "raw_json"
    if suffix == ".log":
        return "log"
    return "other"


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def safe_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def scalar_text(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> int:
    materialized = list(rows)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def flatten_scalars(value: Any, prefix: str = "", depth: int = 0) -> Iterable[tuple[str, Any]]:
    if depth > 8:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, (dict, list)):
                yield from flatten_scalars(child, child_path, depth + 1)
            elif child is None or isinstance(child, (str, int, float, bool)):
                yield child_path, child
    elif isinstance(value, list) and len(value) <= 20:
        for index, child in enumerate(value):
            child_path = f"{prefix}[{index}]"
            if isinstance(child, (dict, list)):
                yield from flatten_scalars(child, child_path, depth + 1)
            elif child is None or isinstance(child, (str, int, float, bool)):
                yield child_path, child


def infer_method(path_parts: list[str]) -> str:
    joined = ".".join(path_parts).lower()
    aliases = [
        ("single_dog", "single_dog_only"), ("rule_baseline", "rule_baseline"),
        ("rolling_optimizer", "rolling_optimizer"), ("optimizer", "optimizer"),
        ("baseline", "baseline"), ("policy", "policy"), ("ppo", "ppo"),
        ("full_context", "full_context_v2"), ("heuristic", "heuristic"),
    ]
    for token, label in aliases:
        if token in joined:
            return label
    return path_parts[-1] if path_parts else ""


def metric_block_row(
    source: Path,
    stage: int | None,
    block_path: list[str],
    block: dict[str, Any],
    root_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    direct = {key: value for key, value in block.items() if key in METRIC_KEYS and is_number(value)}
    # Stage 19-style metric dictionaries: success_rate: {mean: ..., training_seed_sd: ...}
    nested_mean = {
        key: value.get("mean")
        for key, value in block.items()
        if key in METRIC_KEYS and isinstance(value, dict) and is_number(value.get("mean"))
    }
    metrics = {**nested_mean, **direct}
    if len(metrics) < 2:
        return None
    path_text = ".".join(block_path)
    profile = profile_from_text(path_text) or profile_from_text(source.as_posix())
    for key in ("arrival_profile", "evaluation_profile", "profile", "training_profile"):
        if not profile and isinstance(root_metadata.get(key), str):
            profile = str(root_metadata[key]).upper()
    row: dict[str, Any] = {
        "stage": stage or "",
        "source_path": source.relative_to(ROOT).as_posix(),
        "block_path": path_text,
        "method": infer_method(block_path),
        "profile": profile,
        "execution_mode": root_metadata.get("execution_mode", ""),
        "condition": root_metadata.get("condition", ""),
        "gate": root_metadata.get("gate", ""),
        "seed": seed_from_path(source) or root_metadata.get("seed_index", "") or root_metadata.get("experiment_run_seed", ""),
    }
    row.update(metrics)
    return row


def walk_metric_blocks(
    source: Path,
    stage: int | None,
    value: Any,
    root_metadata: dict[str, Any],
    path_parts: list[str] | None = None,
    depth: int = 0,
) -> Iterable[dict[str, Any]]:
    path_parts = path_parts or []
    if depth > 9:
        return
    if isinstance(value, dict):
        row = metric_block_row(source, stage, path_parts, value, root_metadata)
        if row:
            yield row
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from walk_metric_blocks(source, stage, child, root_metadata, path_parts + [str(key)], depth + 1)
    elif isinstance(value, list):
        # Large task/event lists are not recursively normalized here.
        if len(value) <= 1000:
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    yield from walk_metric_blocks(source, stage, child, root_metadata, path_parts + [f"row_{index}"], depth + 1)


def extract_modes(source: Path, stage: int | None, value: Any, path_parts: list[str] | None = None) -> Iterable[dict[str, Any]]:
    path_parts = path_parts or []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path_parts + [str(key)]
            if key in {"transport_modes", "transport_mode_counts", "mode_counts"} and isinstance(child, dict):
                total = sum(float(v) for v in child.values() if is_number(v))
                for mode, count in child.items():
                    if is_number(count):
                        yield {
                            "stage": stage or "",
                            "source_path": source.relative_to(ROOT).as_posix(),
                            "block_path": ".".join(child_path),
                            "profile": profile_from_text(".".join(child_path)) or profile_from_text(source.as_posix()),
                            "method": infer_method(child_path),
                            "mode": mode,
                            "count": count,
                            "share": float(count) / total if total else "",
                        }
            elif isinstance(child, (dict, list)):
                yield from extract_modes(source, stage, child, child_path)
    elif isinstance(value, list) and len(value) <= 1000:
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                yield from extract_modes(source, stage, child, path_parts + [f"row_{index}"])


def status_label(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "NOT_REPORTED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    tables = output / "tables"
    extracted = output / "source_extract"
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)

    all_files = sorted(path for path in RESULTS.rglob("*") if path.is_file())
    file_rows: list[dict[str, Any]] = []
    stage_counts: dict[int, Counter[str]] = {stage: Counter() for stage in range(1, 26)}
    stage_bytes: Counter[int] = Counter()
    parse_errors: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    copied_count = 0
    copied_bytes = 0

    for path in all_files:
        rel = path.relative_to(ROOT)
        stat = path.stat()
        stage = stage_from_path(rel)
        role = classify(path)
        suffix = path.suffix.lower().lstrip(".") or "no_extension"
        include = role in {"summary", "gate", "status", "manifest", "tabular_result", "report_or_figure"} and stat.st_size <= 20 * 1024 * 1024
        file_rows.append({
            "stage": stage or "",
            "source_path": rel.as_posix(),
            "file_name": path.name,
            "extension": suffix,
            "role": role,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / 1048576, 6),
            "modified_at": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "copied_to_source_extract": include,
            "copy_note": "copied" if include else ("checkpoint inventoried only" if role == "model_checkpoint" else "normalized/inventoried only"),
        })
        if stage:
            stage_counts[stage]["total_files"] += 1
            stage_counts[stage][role] += 1
            stage_counts[stage][f"ext_{suffix}"] += 1
            stage_bytes[stage] += stat.st_size
        if include:
            target = extracted / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied_count += 1
            copied_bytes += stat.st_size

        if path.suffix.lower() != ".json":
            continue
        # Histories are handled separately and downsampled to every 20 updates.
        try:
            data = safe_json(path)
        except Exception as exc:  # pragma: no cover - defensive export path
            parse_errors.append({"source_path": rel.as_posix(), "error": repr(exc)})
            continue
        if role == "training_history" and isinstance(data, list):
            selected: list[dict[str, Any]] = []
            for index, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                update = item.get("update", index + 1)
                if index == 0 or index == len(data) - 1 or (isinstance(update, int) and update % 20 == 0):
                    selected.append(item)
            profile = profile_from_text(rel.as_posix())
            condition = path.parent.name if path.parent.name not in {"long", "smoke"} else path.stem.replace(".history", "")
            for item in selected:
                row = {
                    "stage": stage or "",
                    "source_path": rel.as_posix(),
                    "profile": profile,
                    "condition": condition,
                    "seed": seed_from_path(rel) or "",
                }
                for key in HISTORY_KEYS:
                    if key in item and not isinstance(item[key], (dict, list)):
                        row[key] = item[key]
                evaluation = item.get("evaluation")
                if isinstance(evaluation, dict):
                    for key in METRIC_KEYS:
                        if key in evaluation and is_number(evaluation[key]):
                            row[f"eval_{key}"] = evaluation[key]
                history_rows.append(row)
            if data:
                for endpoint, item in (("first", data[0]), ("last", data[-1])):
                    if isinstance(item, dict):
                        row = {
                            "stage": stage or "", "source_path": rel.as_posix(),
                            "profile": profile, "condition": condition,
                            "seed": seed_from_path(rel) or "", "endpoint": endpoint,
                        }
                        for key in HISTORY_KEYS:
                            if key in item and not isinstance(item[key], (dict, list)):
                                row[key] = item[key]
                        endpoint_rows.append(row)
            continue

        if not isinstance(data, dict):
            continue
        json_stage = data.get("stage") if isinstance(data.get("stage"), int) else stage
        for key_path, value in flatten_scalars(data):
            scalar_rows.append({
                "stage": json_stage or "",
                "source_path": rel.as_posix(),
                "json_path": key_path,
                "value": scalar_text(value),
                "value_type": type(value).__name__,
            })
        metric_rows.extend(walk_metric_blocks(path, json_stage, data, data))
        mode_rows.extend(extract_modes(path, json_stage, data))
        if "passed" in data or "gate" in data or role in {"summary", "gate"}:
            gate_rows.append({
                "stage": json_stage or "",
                "source_path": rel.as_posix(),
                "artifact": path.name,
                "gate": scalar_text(data.get("gate", data.get("campaign", ""))),
                "status": status_label(data.get("passed")),
                "passed": data.get("passed", ""),
                "claim_boundary": scalar_text(data.get("claim_boundary", "")),
                "profile": scalar_text(data.get("arrival_profile", data.get("training_profile", ""))),
                "execution_mode": scalar_text(data.get("execution_mode", "")),
                "updates": scalar_text(data.get("updates", "")),
                "episodes_trained_total": scalar_text(data.get("episodes_trained_total", "")),
            })

    # Add stage-related code/config/document evidence counts from the whole workspace.
    support_counts: dict[int, Counter[str]] = {stage: Counter() for stage in range(1, 26)}
    for top in (ROOT / "scripts", ROOT / "config", ROOT / "src", ROOT):
        if not top.exists():
            continue
        iterator = top.glob("*") if top == ROOT else top.rglob("*")
        for path in iterator:
            if not path.is_file() or RESULTS in path.parents or output in path.parents:
                continue
            stage = stage_from_path(path.relative_to(ROOT))
            if not stage:
                continue
            suffix = path.suffix.lower()
            if suffix == ".py" or path.name.endswith(".command"):
                support_counts[stage]["script_files"] += 1
            elif suffix in {".yaml", ".yml", ".rviz", ".json"}:
                support_counts[stage]["config_files"] += 1
            elif suffix in {".md", ".txt", ".docx", ".pdf"}:
                support_counts[stage]["document_files"] += 1

    coverage_rows: list[dict[str, Any]] = []
    for stage in range(1, 26):
        counts = stage_counts[stage]
        support = support_counts[stage]
        present = counts["total_files"] > 0
        coverage_rows.append({
            "stage": stage,
            "results_present": present,
            "total_result_files": counts["total_files"],
            "data_volume_mb": round(stage_bytes[stage] / 1048576, 3),
            "json_files": counts["ext_json"],
            "csv_files": counts["ext_csv"],
            "jsonl_files": counts["ext_jsonl"],
            "history_files": counts["training_history"],
            "checkpoint_files": counts["model_checkpoint"],
            "summary_files": counts["summary"],
            "gate_files": counts["gate"],
            "status_files": counts["status"],
            "manifest_files": counts["manifest"],
            "log_files": counts["log"],
            "script_files": support["script_files"],
            "config_files": support["config_files"],
            "document_files": support["document_files"],
            "coverage_note": "current result files exported" if present else "no result file found; support files only",
        })

    # Preserve all existing CSV rows as a universal long table.
    csv_long_rows: list[dict[str, Any]] = []
    csv_catalog_rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.rglob("*.csv")):
        rel = path.relative_to(ROOT)
        stage = stage_from_path(rel)
        try:
            with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                columns = reader.fieldnames or []
            csv_catalog_rows.append({
                "stage": stage or "", "source_path": rel.as_posix(),
                "row_count": len(rows), "column_count": len(columns),
                "columns": " | ".join(columns),
            })
            for index, row in enumerate(rows, start=1):
                for column, value in row.items():
                    csv_long_rows.append({
                        "stage": stage or "", "source_path": rel.as_posix(),
                        "row_index": index, "column": column, "value": value,
                    })
        except Exception as exc:
            parse_errors.append({"source_path": rel.as_posix(), "error": repr(exc)})

    counts = {
        "file_manifest.csv": write_csv(tables / "file_manifest.csv", file_rows),
        "stage_coverage.csv": write_csv(tables / "stage_coverage.csv", coverage_rows),
        "formal_gate_summary.csv": write_csv(tables / "formal_gate_summary.csv", gate_rows),
        "normalized_metric_blocks.csv": write_csv(tables / "normalized_metric_blocks.csv", metric_rows),
        "json_scalar_metrics.csv": write_csv(tables / "json_scalar_metrics.csv", scalar_rows),
        "transport_mode_counts.csv": write_csv(tables / "transport_mode_counts.csv", mode_rows),
        "training_history_every20.csv": write_csv(tables / "training_history_every20.csv", history_rows),
        "training_run_endpoints.csv": write_csv(tables / "training_run_endpoints.csv", endpoint_rows),
        "existing_csv_catalog.csv": write_csv(tables / "existing_csv_catalog.csv", csv_catalog_rows),
        "existing_csv_long.csv": write_csv(tables / "existing_csv_long.csv", csv_long_rows),
        "parse_errors.csv": write_csv(tables / "parse_errors.csv", parse_errors, ["source_path", "error"]),
    }

    report = {
        "schema_version": "warehouse_stage1_25_export_v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace": str(ROOT),
        "source_results": str(RESULTS),
        "scope": "All currently available Stage 1-25 artifacts; incompleteness is retained explicitly.",
        "result_file_count": len(file_rows),
        "result_size_bytes": sum(int(row["size_bytes"]) for row in file_rows),
        "result_size_gb": round(sum(int(row["size_bytes"]) for row in file_rows) / 1073741824, 3),
        "checkpoint_file_count": sum(1 for row in file_rows if row["role"] == "model_checkpoint"),
        "checkpoint_size_gb": round(sum(int(row["size_bytes"]) for row in file_rows if row["role"] == "model_checkpoint") / 1073741824, 3),
        "copied_source_file_count": copied_count,
        "copied_source_size_mb": round(copied_bytes / 1048576, 3),
        "table_row_counts": counts,
        "stages_with_result_files": [row["stage"] for row in coverage_rows if row["results_present"]],
        "stages_without_result_files": [row["stage"] for row in coverage_rows if not row["results_present"]],
        "interpretation_boundary": [
            "Counts describe the present workspace, not the intended experiment plan.",
            "Missing files are not treated as failed experiments.",
            "Smoke/interface gates are not promoted to formal long-run evidence.",
            "Model checkpoints are inventoried but not duplicated in the teacher-facing bundle.",
        ],
    }
    (output / "export_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
