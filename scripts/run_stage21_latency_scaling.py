#!/usr/bin/env python3
"""Stage 21 CPU inference-latency and bounded-scale benchmark.

This experiment measures scheduling computation only.  Environment creation,
navigation, and task execution are deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import sys
from time import perf_counter_ns

# Preserve the Apple Silicon import order used by the training pipeline.
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))

from warehouse_core.stage12_encoding import Stage12Encoder  # noqa: E402
from warehouse_core.stage14_policy import MaskedCandidateActorCritic  # noqa: E402
from warehouse_core.stage14_ppo import load_checkpoint  # noqa: E402
from warehouse_core.stage14_training import build_stage14_environment  # noqa: E402


MANIFEST = (
    ROOT / "results" / "stage20_mixed_curriculum" /
    "stage20_policy_freeze_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "stage21_external_validity" /
    "latency_scaling_smoke.json"
)
FORMAL_OUTPUT = (
    ROOT / "results" / "stage21_external_validity" /
    "latency_scaling_formal.json"
)
TEAM_IDS = {
    4: ("dog_1", "dog_3", "car_f1_1", "car_f2_1"),
    7: (
        "dog_1", "dog_2", "dog_3",
        "car_f1_1", "car_f1_2", "car_f2_1", "car_f2_2",
    ),
    10: (
        "dog_1", "dog_2", "dog_3", "dog_4",
        "car_f1_1", "car_f1_2", "car_f1_3", "car_f1_4",
        "car_f2_1", "car_f2_2",
    ),
}
STAIR_IDS = {
    1: ("STAIR_NE",),
    2: ("STAIR_NE", "STAIR_SW"),
    4: ("STAIR_NE", "STAIR_NW", "STAIR_SW", "STAIR_SE"),
}
QUEUE_SIZES = (1, 2, 4, 8, 16, 32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_weight(seed_index: int) -> tuple[Path, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in manifest["runs"]:
        if int(row["training_seed_index"]) == seed_index:
            path = Path(row["best_weight_path"])
            expected = row["best_weight_sha256"]
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"frozen weight hash mismatch: {path}")
            return path, actual
    raise ValueError(f"Stage 20 seed {seed_index} is not in the freeze manifest")


def configure_dog_floors(dispatch) -> None:
    placements = {
        "dog_1": (1, "F1_NE", (20.8, 17.8, .45)),
        "dog_2": (1, "F1_NW", (-20.8, 17.8, .45)),
        "dog_3": (2, "F2_WEST", (-8.8, -6.0, 4.7)),
        "dog_4": (2, "F2_EAST", (8.8, -6.0, 4.7)),
    }
    for robot_id, (floor, region, xyz) in placements.items():
        runtime = dispatch.robots[robot_id]
        runtime.robot.current_floor = floor
        runtime.robot.current_region = region
        runtime.xyz = xyz


def build_state(seed: int, team_size: int, stair_count: int,
                queue_size: int):
    dispatch, _ = build_stage14_environment(
        seed, execution_mode="CONCURRENT",
        arrival_schedule="MIXED_CURRICULUM",
        handover_sampling="TASK_KEYED")
    configure_dog_floors(dispatch)

    last_arrival = max(
        item.arrival_time for item in dispatch.queue.pending_arrivals)
    dispatch.now = last_arrival + 1.0
    dispatch.queue.advance(dispatch.now)
    ranked = dispatch.queue.policy_view(dispatch.now, limit=80)
    if len(ranked) < queue_size:
        raise RuntimeError("not enough generated tasks for scaling state")
    dispatch.queue.waiting = ranked[:queue_size]
    dispatch.queue.pending_arrivals = []

    keep_robots = set(TEAM_IDS[team_size])
    dispatch.robots = {
        robot_id: runtime for robot_id, runtime in dispatch.robots.items()
        if robot_id in keep_robots
    }
    keep_stairs = set(STAIR_IDS[stair_count])
    dispatch.stairs = {
        stair_id: stair for stair_id, stair in dispatch.stairs.items()
        if stair_id in keep_stairs
    }
    return dispatch


def timed_samples(function, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        function()
    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = perf_counter_ns()
            function()
            samples.append((perf_counter_ns() - start) / 1_000_000.0)
    finally:
        if was_enabled:
            gc.enable()
    return samples


def summary(samples: list[float]) -> dict[str, float | list[float]]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "sample_count": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "samples_ms": [float(item) for item in values],
    }


def benchmark_scenario(model: MaskedCandidateActorCritic, *, seed: int,
                       team_size: int, stair_count: int, queue_size: int,
                       warmup: int, repeats: int) -> dict:
    dispatch = build_state(seed, team_size, stair_count, queue_size)
    encoder = Stage12Encoder("FULL_CONTEXT_V2")
    candidates = dispatch.enumerate_candidate_actions()
    legal = dispatch.build_action_mask(candidates)
    if not candidates or not any(legal):
        raise RuntimeError(
            f"scenario has no legal assignments: "
            f"robots={team_size}, stairs={stair_count}, queue={queue_size}")
    if len(candidates) > encoder.MAX_ACTIONS:
        raise RuntimeError(
            f"candidate count {len(candidates)} exceeds {encoder.MAX_ACTIONS}")
    decision = encoder.encode(dispatch)
    observation = {
        "state": decision.observation,
        "action_features": decision.action_features,
    }

    def candidate_build():
        return dispatch.enumerate_candidate_actions()

    def mask_build():
        return dispatch.build_action_mask(candidates)

    def encoder_total():
        return encoder.encode(dispatch)

    def policy_forward():
        return model.select_action(
            observation, decision.action_mask, deterministic=True)

    def decision_total():
        encoded = encoder.encode(dispatch)
        obs = {
            "state": encoded.observation,
            "action_features": encoded.action_features,
        }
        return model.select_action(obs, encoded.action_mask, deterministic=True)

    components = {}
    for name, function in (
            ("candidate_generation", candidate_build),
            ("legality_mask", mask_build),
            ("encoder_total", encoder_total),
            ("policy_forward", policy_forward),
            ("end_to_end_decision", decision_total)):
        components[name] = summary(timed_samples(function, warmup, repeats))

    return {
        "scenario_id": f"R{team_size}_S{stair_count}_Q{queue_size}",
        "team_size": team_size,
        "stair_count": stair_count,
        "waiting_task_count": queue_size,
        "visible_task_count": min(queue_size, encoder.MAX_TASKS),
        "candidate_count": len(candidates),
        "legal_candidate_count": int(sum(legal)),
        "components": components,
    }


def validate(rows: list[dict], budget_ms: float) -> dict[str, bool]:
    finite = True
    ordered = True
    for row in rows:
        for metric in row["components"].values():
            values = [metric[key] for key in (
                "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                "min_ms", "max_ms")]
            finite &= all(math.isfinite(value) and value >= 0 for value in values)
            ordered &= (
                metric["min_ms"] <= metric["p50_ms"] <=
                metric["p95_ms"] <= metric["p99_ms"] <= metric["max_ms"])

    lookup = {
        (row["team_size"], row["stair_count"], row["waiting_task_count"]):
        row["candidate_count"] for row in rows
    }
    saturation = all(
        lookup[(team, stair, 8)] == lookup[(team, stair, 16)] ==
        lookup[(team, stair, 32)]
        for team in TEAM_IDS for stair in STAIR_IDS)
    monotone_queue = all(
        lookup[(team, stair, low)] <= lookup[(team, stair, high)]
        for team in TEAM_IDS for stair in STAIR_IDS
        for low, high in zip((1, 2, 4), (2, 4, 8)))
    checks = {
        "all_expected_scenarios_present": len(rows) == (
            len(TEAM_IDS) * len(STAIR_IDS) * len(QUEUE_SIZES)),
        "all_metrics_finite_and_nonnegative": finite,
        "percentiles_ordered": ordered,
        "candidate_capacity_not_exceeded": all(
            row["candidate_count"] <= Stage12Encoder.MAX_ACTIONS
            for row in rows),
        "all_scenarios_have_legal_action": all(
            row["legal_candidate_count"] > 0 for row in rows),
        "queue_latency_boundary_exercised": saturation,
        "candidate_count_monotone_through_visible_queue": monotone_queue,
        "end_to_end_p99_within_engineering_budget": all(
            row["components"]["end_to_end_decision"]["p99_ms"] <= budget_ms
            for row in rows),
    }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage20-seed", type=int, default=1)
    parser.add_argument("--benchmark-seed", type=int, default=65000000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--budget-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.warmup = min(args.warmup, 3)
        args.repeats = min(args.repeats, 20)
    output = args.output or (DEFAULT_OUTPUT if args.smoke else FORMAL_OUTPUT)
    if args.warmup < 0 or args.repeats < 5 or args.budget_ms <= 0:
        parser.error("warmup>=0, repeats>=5, and budget-ms>0 are required")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    weight, weight_hash = frozen_weight(args.stage20_seed)
    model, checkpoint = load_checkpoint(weight, torch.device("cpu"))
    model.eval()

    scenarios = [
        (team, stair, queue)
        for team in TEAM_IDS for stair in STAIR_IDS for queue in QUEUE_SIZES
    ]
    random.Random(args.benchmark_seed + 21).shuffle(scenarios)
    rows = []
    with torch.inference_mode():
        for index, (team, stair, queue) in enumerate(scenarios, start=1):
            print(
                f"Stage21 latency {index:02d}/{len(scenarios)} "
                f"robots={team} stairs={stair} queue={queue}",
                flush=True)
            rows.append(benchmark_scenario(
                model, seed=args.benchmark_seed,
                team_size=team, stair_count=stair, queue_size=queue,
                warmup=args.warmup, repeats=args.repeats))
    rows.sort(key=lambda row: (
        row["team_size"], row["stair_count"], row["waiting_task_count"]))
    checks = validate(rows, args.budget_ms)
    result = {
        "schema_version": "warehouse_stage21_latency_scaling_v1",
        "stage": 21,
        "gate": "LATENCY_SCALING_SMOKE" if args.smoke else "LATENCY_SCALING",
        "claim_boundary": (
            "CPU scheduling-computation benchmark within the trained fixed "
            "shape only; excludes navigation, ROS transport, and actuation."),
        "weight": {
            "training_stage": 20,
            "training_seed_index": args.stage20_seed,
            "path": str(weight),
            "sha256": weight_hash,
            "model_metadata": checkpoint["model_metadata"],
        },
        "protocol": {
            "benchmark_seed": args.benchmark_seed,
            "warmup_per_component": args.warmup,
            "repeats_per_component": args.repeats,
            "team_sizes": list(TEAM_IDS),
            "stair_counts": list(STAIR_IDS),
            "waiting_task_counts": list(QUEUE_SIZES),
            "maximum_visible_tasks": Stage12Encoder.MAX_TASKS,
            "maximum_robots": Stage12Encoder.MAX_ROBOTS,
            "maximum_stairs": Stage12Encoder.MAX_STAIRS,
            "maximum_candidate_slots": Stage12Encoder.MAX_ACTIONS,
            "engineering_p99_budget_ms": args.budget_ms,
            "timed_components_overlap": {
                "encoder_total": "includes candidate generation and mask",
                "end_to_end_decision": "includes encoder_total and policy_forward",
            },
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "rows": rows,
        "assertions": checks,
        "passed": all(checks.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "scenario_count": len(rows),
        "max_candidate_count": max(row["candidate_count"] for row in rows),
        "max_end_to_end_p99_ms": max(
            row["components"]["end_to_end_decision"]["p99_ms"]
            for row in rows),
        "assertions": checks,
        "passed": result["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
