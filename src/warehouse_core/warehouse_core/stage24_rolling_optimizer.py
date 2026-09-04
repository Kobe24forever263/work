"""Causal root-rollout optimizer for the Stage 24 strong baseline.

This module is deliberately versioned separately from the frozen Stage 23
evaluation stack.  It scores the full legal root catalog with a transparent
reward-aligned surrogate, rolls out a deterministic bounded shortlist on
planning clones, follows the existing time-greedy rule for the rollout tail,
and executes only the selected root action in the real environment.

Planning clones never query the task-keyed handover provider.  They replace it
with the public nominal handover duration before taking a simulated step.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from time import perf_counter_ns
from typing import Any

from .persistent_dispatch import Assignment
from .stage14_training import WarehouseDispatchGymEnv


RELEASED_ONLY = "RELEASED_ONLY"
ANNOUNCED_PENDING = "ANNOUNCED_PENDING"
VISIBILITY_MODES = (RELEASED_ONLY, ANNOUNCED_PENDING)
TIME_GREEDY = "TIME_GREEDY"


@dataclass(frozen=True)
class Stage24OptimizerConfig:
    """Fixed planning contract for one Stage 24 optimizer instance.

    ``horizon_decisions`` counts the evaluated root decision.  A value of
    three therefore evaluates the root assignment followed by at most two
    time-greedy tail decisions.  ``max_root_rollouts`` bounds planning clones
    after every legal candidate has been screened.  ``budget_reserve_ms`` is
    held back for deterministic selection and diagnostics so the returned
    action remains inside the declared wall-clock budget.
    """

    visibility_mode: str = RELEASED_ONLY
    horizon_decisions: int = 3
    wall_clock_budget_ms: float = 1000.0
    fallback_policy: str = TIME_GREEDY
    tie_tolerance: float = 1e-12
    handover_timeout_probability_per_attempt: float = 0.05
    max_root_rollouts: int = 24
    budget_reserve_ms: float = 25.0

    def __post_init__(self) -> None:
        visibility = self.visibility_mode.upper()
        fallback = self.fallback_policy.upper()
        if visibility not in VISIBILITY_MODES:
            raise ValueError(
                f"unsupported visibility_mode: {self.visibility_mode}")
        if fallback != TIME_GREEDY:
            raise ValueError(
                "Stage 24 currently supports only TIME_GREEDY fallback")
        if self.horizon_decisions <= 0:
            raise ValueError("horizon_decisions must be positive")
        if (not math.isfinite(self.wall_clock_budget_ms) or
                self.wall_clock_budget_ms <= 0.0):
            raise ValueError("wall_clock_budget_ms must be finite and positive")
        if (not math.isfinite(self.tie_tolerance) or
                self.tie_tolerance < 0.0):
            raise ValueError("tie_tolerance must be finite and non-negative")
        if (not math.isfinite(
                self.handover_timeout_probability_per_attempt) or
                not 0.0 <=
                self.handover_timeout_probability_per_attempt < 1.0):
            raise ValueError(
                "handover_timeout_probability_per_attempt must be in [0, 1)")
        if self.max_root_rollouts <= 0:
            raise ValueError("max_root_rollouts must be positive")
        if (not math.isfinite(self.budget_reserve_ms) or
                self.budget_reserve_ms < 0.0 or
                self.budget_reserve_ms >= self.wall_clock_budget_ms):
            raise ValueError(
                "budget_reserve_ms must be non-negative and below budget")
        object.__setattr__(self, "visibility_mode", visibility)
        object.__setattr__(self, "fallback_policy", fallback)


class _NominalHandoverProvider:
    """Planning-only provider that cannot expose a realised keyed sample."""

    def __init__(self, duration_s: float):
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("nominal handover duration must be positive")
        self.mean_duration_s = float(duration_s)
        self.call_count = 0

    def __call__(self, handover_kind: str,
                 task_id: str | None = None) -> float:
        del handover_kind, task_id
        self.call_count += 1
        return self.mean_duration_s


def _stable_action_key(action: Assignment | str | None,
                       action_index: int) -> tuple:
    """Return a deterministic key independent of enumeration side effects."""

    if isinstance(action, Assignment):
        return (
            0, float(action.estimated_cost),
            action.task_id, action.transport_mode,
            action.pickup_carter, action.dog_id, action.stair_id,
            action.receiving_carter, action.pickup_handover_point,
            action.receive_handover_point, action.post_task_policy,
            action_index,
        )
    return (1, str(action), action_index)


class RollingHorizonOptimizer:
    """Bounded root rollout with a deterministic time-greedy tail.

    The optimizer is causal under its declared visibility contract:

    * ``RELEASED_ONLY`` removes all unreleased arrivals from each planning
      clone.  The rollout can use waiting tasks and already-active assignments
      only.
    * ``ANNOUNCED_PENDING`` retains the pre-announced order book and its
      release times.  It still replaces realised handover outcomes with the
      nominal planning model.

    Selection is exact only when the legal root catalog fits inside the
    shortlist and every shortlisted root finishes within budget.  Otherwise
    ``exact_gap`` is deliberately ``None``; a score margin is reported only as
    a search diagnostic and never mislabeled as an optimality certificate.
    """

    def __init__(self, config: Stage24OptimizerConfig | None = None):
        self.config = config or Stage24OptimizerConfig()
        self.last_diagnostics: dict[str, Any] = {}
        self.diagnostics: list[dict[str, Any]] = []

    def clear_diagnostics(self) -> None:
        self.last_diagnostics = {}
        self.diagnostics.clear()

    @staticmethod
    def _time_greedy_action(env: WarehouseDispatchGymEnv) -> int:
        """Call the frozen base rule explicitly, avoiding wrapper recursion."""

        return int(WarehouseDispatchGymEnv.rule_action(env))

    def _planning_clone(
            self, env: WarehouseDispatchGymEnv,
    ) -> tuple[WarehouseDispatchGymEnv, _NominalHandoverProvider]:
        """Clone public state, then install a non-oracle handover model."""

        clone = deepcopy(env)
        if clone.dispatch is None:
            raise RuntimeError("environment must be reset before planning")
        nominal = float(clone.dispatch.handover_nominal_s)
        provider = _NominalHandoverProvider(nominal)
        clone.dispatch.handover_duration_provider = provider
        clone.dispatch.handover_timing_source = (
            "stage24_planning_nominal_non_oracle")

        if self.config.visibility_mode == RELEASED_ONLY:
            # Do not inspect the unreleased records; remove them wholesale.
            clone.dispatch.queue.pending_arrivals.clear()
            active_count = len(getattr(clone.dispatch, "active_tasks", {}))
            clone.dispatch.task_limit = (
                clone.dispatch.completed + clone.dispatch.failed +
                len(clone.dispatch.queue.waiting) + active_count)
            # Re-encode after the visibility projection so the tail rule sees
            # exactly the projected planning state.
            clone.decision = clone._encode_decision()
        return clone, provider

    @staticmethod
    def _legal_root_actions(
            env: WarehouseDispatchGymEnv,
    ) -> list[tuple[int, Assignment | str]]:
        if env.decision is None:
            raise RuntimeError("environment must be reset before planning")
        roots = [
            (index, action)
            for index, (action, legal) in enumerate(zip(
                env.decision.action_ids, env.decision.action_mask))
            if bool(legal) and action is not None
        ]
        return sorted(
            roots, key=lambda row: _stable_action_key(row[1], row[0]))

    def _screening_score(
            self, env: WarehouseDispatchGymEnv,
            action: Assignment | str) -> float:
        """Fast non-oracle score for deterministic root shortlisting.

        All terms are current-state quantities.  ETA is converted into the
        same units as the outstanding-time reward; deadline and completion
        terms use the public continuous-time reward configuration.  Relay
        risk uses only the frozen task-independent timeout probability.
        """

        if not isinstance(action, Assignment):
            return 0.0
        dispatch = env.dispatch
        queued = dispatch._task(action.task_id)
        task = queued.task
        config = dispatch.reward_config
        age = max(0.0, dispatch.now - queued.arrival_time)
        eta = max(0.0, float(action.estimated_cost))
        predicted_finish = dispatch.now + eta
        deadline_value = (
            config.on_time_bonus if predicted_finish <= task.deadline else
            -config.timeout_penalty)
        relay_risk = 0.0
        if action.transport_mode == dispatch.CAR_DOG_CAR:
            probability = 1.0 - (
                1.0 - self.config.
                handover_timeout_probability_per_attempt) ** 2
            relay_risk = probability * (
                config.completion_bonus + config.on_time_bonus +
                config.timeout_penalty)
        return (
            config.completion_bonus + deadline_value - relay_risk -
            eta / config.outstanding_time_scale +
            age / config.outstanding_time_scale +
            0.05 * float(task.priority))

    def _shortlist(
            self, env: WarehouseDispatchGymEnv,
            roots: list[tuple[int, Assignment | str]],
    ) -> list[tuple[int, Assignment | str]]:
        """Screen every root, preserving task coverage before global fill."""

        limit = min(self.config.max_root_rollouts, len(roots))
        if len(roots) <= limit:
            return roots
        assignments = [row for row in roots if isinstance(row[1], Assignment)]
        if not assignments:
            return roots[:limit]

        def rank(row):
            index, action = row
            return (
                -self._screening_score(env, action),
                _stable_action_key(action, index))

        by_task: dict[str, list[tuple[int, Assignment | str]]] = {}
        for row in assignments:
            by_task.setdefault(row[1].task_id, []).append(row)
        for rows in by_task.values():
            rows.sort(key=rank)

        selected: list[tuple[int, Assignment | str]] = []
        selected_indices: set[int] = set()
        # First preserve at least one feasible assignment for every visible
        # task whenever the shortlist capacity permits.
        for task_id in sorted(by_task):
            row = by_task[task_id][0]
            selected.append(row)
            selected_indices.add(row[0])
            if len(selected) >= limit:
                return sorted(selected, key=rank)
        for row in sorted(assignments, key=rank):
            if row[0] in selected_indices:
                continue
            selected.append(row)
            selected_indices.add(row[0])
            if len(selected) >= limit:
                break
        return sorted(selected, key=rank)

    def _rollout(
            self, env: WarehouseDispatchGymEnv, root_index: int,
            deadline_ns: int,
    ) -> tuple[float, int, int, bool, int, float]:
        """Return score, work counts, budget status and relay-risk cost."""

        clone, nominal_provider = self._planning_clone(env)
        if clone.decision is None:
            raise RuntimeError("planning clone has no encoded decision")
        if (root_index >= len(clone.decision.action_mask) or
                not bool(clone.decision.action_mask[root_index])):
            raise RuntimeError("root action changed under planning clone")

        score = 0.0
        discount_prefix = 1.0
        steps = 0
        relay_decisions = 0
        expected_risk_correction = 0.0
        action_index = root_index
        terminated = truncated = False
        while steps < self.config.horizon_decisions:
            if perf_counter_ns() >= deadline_ns:
                return (score, steps, nominal_provider.call_count, True,
                        relay_decisions, expected_risk_correction)
            decoded = clone.decision.action_ids[action_index]
            relay = bool(
                isinstance(decoded, Assignment) and
                decoded.transport_mode == clone.dispatch.CAR_DOG_CAR)
            _, reward, terminated, truncated, info = clone.step(action_index)
            score += discount_prefix * float(reward)
            transition_discount = float(info.get("discount", 1.0))
            if relay:
                # The nominal transition is deliberately non-oracle and hence
                # always succeeds.  Correct its score by the expected loss of
                # at least one timeout in a two-handover relay.  This mirrors
                # the frozen Stage 15 risk rule without consulting a task-keyed
                # realization.
                probability = (
                    1.0 - (1.0 - self.config.
                           handover_timeout_probability_per_attempt) ** 2)
                reward_config = clone.dispatch.reward_config
                loss_if_timeout = (
                    reward_config.completion_bonus +
                    reward_config.on_time_bonus +
                    reward_config.timeout_penalty)
                correction = probability * loss_if_timeout
                score -= discount_prefix * transition_discount * correction
                relay_decisions += 1
                expected_risk_correction += (
                    discount_prefix * transition_discount * correction)
            discount_prefix *= transition_discount
            steps += 1
            if perf_counter_ns() >= deadline_ns:
                return (score, steps, nominal_provider.call_count, True,
                        relay_decisions, expected_risk_correction)
            if terminated or truncated:
                break
            # A planning clone preserves the dynamic class of ``env``.  Call
            # the frozen base implementation explicitly so a
            # Stage24OptimizerGymEnv clone cannot recursively invoke this
            # optimizer.
            action_index = self._time_greedy_action(clone)
        return (score, steps, nominal_provider.call_count, False,
                relay_decisions, expected_risk_correction)

    def _publish(self, diagnostics: dict[str, Any]) -> None:
        self.last_diagnostics = diagnostics
        self.diagnostics.append(dict(diagnostics))

    def select_action(self, env: WarehouseDispatchGymEnv) -> int:
        """Select one real action and store auditable decision diagnostics."""

        if env.dispatch is None or env.decision is None:
            raise RuntimeError("environment must be reset before planning")
        started_ns = perf_counter_ns()
        deadline_ns = started_ns + int(
            self.config.wall_clock_budget_ms * 1_000_000.0)
        search_deadline_ns = deadline_ns - int(
            self.config.budget_reserve_ms * 1_000_000.0)
        roots = self._legal_root_actions(env)
        if not roots:
            raise RuntimeError("encoded decision contains no legal root action")

        base = {
            "schema_version": "warehouse_stage24_optimizer_diagnostics_v1",
            "decision_time_s": float(env.dispatch.now),
            "visibility_mode": self.config.visibility_mode,
            "horizon_decisions": self.config.horizon_decisions,
            "wall_clock_budget_ms": self.config.wall_clock_budget_ms,
            "budget_ms": self.config.wall_clock_budget_ms,
            "candidate_count": len(roots),
            "root_candidate_count": len(roots),
            "screened_candidate_count": len(roots),
            "shortlist_count": 0,
            "pruned_candidate_count": 0,
            "shortlist_coverage": 0.0,
            "rollout_count": 0,
            "evaluated_root_count": 0,
            "rollout_step_count": 0,
            "nominal_handover_call_count": 0,
            "planning_handover_model": "NOMINAL_NON_ORACLE",
            "oracle_handover_samples_read": 0,
            "handover_timeout_probability_per_attempt": (
                self.config.handover_timeout_probability_per_attempt),
            "relay_timeout_probability": (
                1.0 - (1.0 - self.config.
                       handover_timeout_probability_per_attempt) ** 2),
            "relay_expected_timeout_risk_formula": (
                "discount_at_completion * (1-(1-p_attempt)^2) * "
                "(completion_bonus+on_time_bonus+timeout_penalty)"),
            "relay_expected_timeout_risk_correction_active": True,
            "rollout_relay_decision_count": 0,
            "rollout_expected_timeout_risk_correction": 0.0,
            "selected_expected_timeout_risk_correction": None,
            "timeout": False,
            "timed_out": False,
            "fallback": False,
            "fallback_used": False,
            "fallback_policy": self.config.fallback_policy,
            "fallback_reason": "",
            "evaluated_all_candidates": False,
            "exact_root_enumeration": False,
            "exact_gap": None,
            "certified_gap_status": "UNKNOWN",
            "incumbent_score_margin": None,
            "search_truncated": False,
            "budget_compliant": False,
            "selected_action_index": None,
            "selected_rollout_score": None,
            "best_rollout_score": None,
            "rollout_error_type": "",
            "rollout_error_message": "",
            "config": asdict(self.config),
        }

        # A sole WAIT action has no competing root assignment.  Executing it
        # directly is exact and avoids fabricating unreleased tasks in the
        # RELEASED_ONLY planning projection.
        if len(roots) == 1 and not isinstance(roots[0][1], Assignment):
            selected = roots[0][0]
            elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
            base.update({
                "latency_ms": elapsed_ms,
                "budget_compliant": (
                    elapsed_ms <= self.config.wall_clock_budget_ms),
                "shortlist_count": 1,
                "shortlist_coverage": 1.0,
                "evaluated_all_candidates": True,
                "exact_root_enumeration": True,
                "exact_gap": 0.0,
                "certified_gap_status": "EXACT_ZERO",
                "selected_action_index": selected,
            })
            self._publish(base)
            return selected

        shortlist = self._shortlist(env, roots)
        base.update({
            "shortlist_count": len(shortlist),
            "pruned_candidate_count": len(roots) - len(shortlist),
            "shortlist_coverage": len(shortlist) / max(len(roots), 1),
        })
        best_index: int | None = None
        best_score: float | None = None
        second_best_score: float | None = None
        rollout_count = 0
        rollout_steps = 0
        nominal_calls = 0
        relay_decisions = 0
        risk_correction_total = 0.0
        timed_out = False
        rollout_error: Exception | None = None
        best_key: tuple | None = None
        best_risk_correction: float | None = None
        for root_index, root_action in shortlist:
            if perf_counter_ns() >= search_deadline_ns:
                timed_out = True
                break
            try:
                (score, steps, calls, over_budget, relay_count,
                 risk_correction) = self._rollout(
                    env, root_index, search_deadline_ns)
            except Exception as exc:  # Fallback is part of the baseline contract.
                rollout_error = exc
                break
            rollout_steps += steps
            nominal_calls += calls
            relay_decisions += relay_count
            risk_correction_total += risk_correction
            if over_budget:
                timed_out = True
                break
            rollout_count += 1
            action_key = _stable_action_key(root_action, root_index)
            if (best_score is None or
                    score > best_score + self.config.tie_tolerance or
                    (abs(score - best_score) <= self.config.tie_tolerance and
                     (best_key is None or action_key < best_key))):
                if best_score is not None:
                    second_best_score = best_score
                best_index = root_index
                best_score = score
                best_key = action_key
                best_risk_correction = risk_correction
            elif (second_best_score is None or
                  score > second_best_score):
                second_best_score = score

        evaluated_shortlist = (
            rollout_error is None and not timed_out and
            rollout_count == len(shortlist))
        evaluated_all = evaluated_shortlist and len(shortlist) == len(roots)
        fallback_used = rollout_error is not None or best_index is None
        if fallback_used:
            selected = self._time_greedy_action(env)
            reason = (
                "ROLLOUT_ERROR" if rollout_error is not None else
                "WALL_CLOCK_BUDGET" if timed_out else
                "NO_COMPLETE_ROLLOUT")
        else:
            selected = int(best_index)
            reason = ""

        elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
        score_margin = (
            best_score - second_best_score
            if best_score is not None and second_best_score is not None else
            None)
        base.update({
            "latency_ms": elapsed_ms,
            "budget_compliant": elapsed_ms <= self.config.wall_clock_budget_ms,
            "rollout_count": rollout_count,
            "evaluated_root_count": rollout_count,
            "rollout_step_count": rollout_steps,
            "nominal_handover_call_count": nominal_calls,
            "rollout_relay_decision_count": relay_decisions,
            "rollout_expected_timeout_risk_correction": (
                risk_correction_total),
            "selected_expected_timeout_risk_correction": (
                best_risk_correction if not fallback_used else None),
            "timeout": timed_out,
            "timed_out": timed_out,
            "fallback": fallback_used,
            "fallback_used": fallback_used,
            "fallback_reason": reason,
            "evaluated_all_candidates": evaluated_all,
            "exact_root_enumeration": evaluated_all,
            "exact_gap": 0.0 if evaluated_all else None,
            "certified_gap_status": (
                "EXACT_ZERO" if evaluated_all else
                "UNKNOWN_TIMEOUT" if timed_out else
                "UNKNOWN_PRUNED"),
            "incumbent_score_margin": score_margin,
            "search_truncated": not evaluated_all,
            "selected_action_index": selected,
            "selected_rollout_score": (
                best_score if not fallback_used else None),
            "best_rollout_score": best_score,
            "rollout_error_type": (
                type(rollout_error).__name__ if rollout_error else ""),
            "rollout_error_message": (
                str(rollout_error) if rollout_error else ""),
        })
        self._publish(base)
        return selected

    def select_action_with_diagnostics(
            self, env: WarehouseDispatchGymEnv,
    ) -> tuple[int, dict[str, Any]]:
        action = self.select_action(env)
        return action, dict(self.last_diagnostics)


class Stage24OptimizerGymEnv(WarehouseDispatchGymEnv):
    """Drop-in env whose ``rule_action`` delegates to the Stage 24 optimizer.

    This lets the frozen evaluator's TIME_GREEDY_RULE call path exercise the
    new baseline through ``env_factory`` without editing that evaluator.
    """

    def __init__(self, *args,
                 optimizer_config: Stage24OptimizerConfig | None = None,
                 optimizer: RollingHorizonOptimizer | None = None,
                 **kwargs):
        if optimizer is not None and optimizer_config is not None:
            raise ValueError(
                "pass optimizer or optimizer_config, not both")
        super().__init__(*args, **kwargs)
        self.stage24_optimizer = optimizer or RollingHorizonOptimizer(
            optimizer_config)

    @property
    def optimizer_diagnostics(self) -> list[dict[str, Any]]:
        return self.stage24_optimizer.diagnostics

    @property
    def last_optimizer_diagnostics(self) -> dict[str, Any]:
        return self.stage24_optimizer.last_diagnostics

    def reset(self, *, seed=None, options=None):
        self.stage24_optimizer.clear_diagnostics()
        return super().reset(seed=seed, options=options)

    def rule_action(self) -> int:
        return self.stage24_optimizer.select_action(self)


__all__ = [
    "ANNOUNCED_PENDING",
    "RELEASED_ONLY",
    "RollingHorizonOptimizer",
    "Stage24OptimizerConfig",
    "Stage24OptimizerGymEnv",
    "TIME_GREEDY",
    "VISIBILITY_MODES",
]
