#!/usr/bin/env python3
"""Run the existing PPO implementation on the Stage 25 causal interface.

The frozen Stage 14/23 trainer is reused without editing it.  This launcher
injects the versioned released-only encoder/environment and expands only the
local CLI observation choice.  Checkpoints therefore remain compatible with
the established actor-critic implementation while carrying an explicit
``CAUSAL_RELEASED_V4`` provenance contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))
sys.path.insert(0, str(ROOT / "scripts"))

import warehouse_core.stage14_ppo as ppo_module
from warehouse_core.stage25_causal_observation import (
    CAUSAL_RELEASED_V4, CausalReleasedEncoder,
    CausalReleasedWarehouseDispatchGymEnv)
import run_stage14_smdp_ppo as frozen_trainer


def _argument_value(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _rewrite_summary(output: Path) -> None:
    summary_path = output.with_suffix(".summary.json")
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    smoke = summary["updates"] * summary["episodes_trained_this_run"] <= 80
    summary.update({
        "schema_version": "warehouse_stage25_causal_training_v1",
        "algorithm": "centralized_masked_causal_observation_smdp_ppo",
        "stage": 25,
        "gate": (
            "SHORT_CAUSAL_RELEASED_V4_PPO_SMOKE" if smoke else
            "CAUSAL_RELEASED_V4_PPO_TRAINING_RUN"),
        "claim_boundary": (
            "Causal interface smoke only; no convergence or superiority claim."
            if smoke else
            "Training only; independent locked-test acceptance is required."),
        "observation_variant": CAUSAL_RELEASED_V4,
        "visibility_contract": "RELEASED_ONLY",
        "decision_process_claim": "CAUSAL_PARTIALLY_OBSERVED_SMDP",
        "stage23_frozen_sources_modified": False,
        "handover_sampling_contract": "TASK_KEYED",
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main() -> int:
    output_text = _argument_value("--output")
    if output_text is None:
        output = (ROOT / "results" / "stage25_causal_online" / "smoke" /
                  "stage25_causal_ppo.pt")
        sys.argv.extend(["--output", str(output)])
    else:
        output = Path(output_text).expanduser().resolve()

    if "--observation-variant" not in sys.argv:
        sys.argv.extend(["--observation-variant", CAUSAL_RELEASED_V4])

    # Runtime-only dependency injection.  No Stage 14/23 source file changes.
    ppo_module.WarehouseDispatchGymEnv = \
        CausalReleasedWarehouseDispatchGymEnv
    original_evaluate = ppo_module.evaluate

    def causal_evaluate(*args, **kwargs):
        kwargs["handover_sampling"] = "TASK_KEYED"
        return original_evaluate(*args, **kwargs)

    # The frozen trainer imported evaluate by value, so patch both references.
    ppo_module.evaluate = causal_evaluate
    frozen_trainer.evaluate = causal_evaluate
    frozen_trainer.Stage12Encoder = CausalReleasedEncoder
    frozen_trainer.RESULT_SCHEMA_VERSION = \
        "warehouse_stage25_causal_training_v1"
    frozen_trainer.ALGORITHM_NAME = \
        "centralized_masked_causal_observation_smdp_ppo"

    original_add_argument = argparse.ArgumentParser.add_argument

    def causal_add_argument(parser, *names, **kwargs):
        if "--observation-variant" in names:
            kwargs["choices"] = (CAUSAL_RELEASED_V4,)
            kwargs["default"] = CAUSAL_RELEASED_V4
        return original_add_argument(parser, *names, **kwargs)

    argparse.ArgumentParser.add_argument = causal_add_argument
    try:
        status = int(frozen_trainer.main())
    finally:
        argparse.ArgumentParser.add_argument = original_add_argument
    _rewrite_summary(output)
    rewritten = json.loads(
        output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    print("STAGE25_FINAL_SUMMARY")
    print(json.dumps({
        key: rewritten.get(key) for key in (
            "stage", "gate", "observation_variant",
            "visibility_contract", "handover_sampling_contract",
            "updates", "episodes_trained_total", "passed")
    }, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
