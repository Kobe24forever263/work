"""Concurrent event-driven extension of the persistent dispatch environment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable

from .domain import Task
from .persistent_dispatch import (
    Assignment, AssignmentExecution, DispatchTransition,
    PersistentDispatchEnvironment, RobotRuntime)


@dataclass
class ActiveAssignment:
    assignment: Assignment
    task: Task
    started_at: float
    finish_at: float
    participant_ids: tuple[str, ...]
    resources: tuple[str, ...]
    execution: AssignmentExecution
    final_runtimes: dict[str, RobotRuntime]
    handover_events: tuple[dict, ...] = ()


@dataclass
class AdvanceMetrics:
    waiting_cost: float = 0.0
    active_robot_seconds: float = 0.0
    finished: list[ActiveAssignment] = field(default_factory=list)


class ConcurrentPersistentDispatchEnvironment(PersistentDispatchEnvironment):
    """Allow independent joint assignments to remain active concurrently.

    High-level actions still dispatch complete legal transport chains.  A
    dispatch reserves its robots, regions, handover points, stair and planned
    standby slots until a future completion event.  Other non-conflicting
    chains may be dispatched during that interval.  No robot position is
    reset between tasks.
    """

    def __init__(self, *args, max_concurrent_tasks: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        if max_concurrent_tasks <= 1:
            raise ValueError("concurrent mode requires max_concurrent_tasks > 1")
        self.execution_mode = "CONCURRENT"
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks: dict[str, ActiveAssignment] = {}
        self.active_standby_claims: dict[str, str] = {}
        self.completion_events: list[dict] = []

    def enumerate_candidate_actions(self) -> list[Assignment]:
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            return []
        return super().enumerate_candidate_actions()

    @staticmethod
    def _participant_ids(assignment: Assignment) -> tuple[str, ...]:
        return tuple(dict.fromkeys(robot_id for robot_id in (
            assignment.pickup_carter, assignment.dog_id,
            assignment.receiving_carter) if robot_id))

    def _schedule_assignment(self, assignment: Assignment) -> ActiveAssignment:
        queued = self._task(assignment.task_id)
        task = queued.task
        participant_ids = self._participant_ids(assignment)
        resources = tuple(self._claim_resources(assignment, task))
        before = {robot_id: deepcopy(self.robots[robot_id])
                  for robot_id in participant_ids}
        handover_start = len(self.handover_events)
        new_standby_claims: list[str] = []
        try:
            execution = self._execute_claimed_assignment(assignment, task)
            final_runtimes = {
                robot_id: deepcopy(self.robots[robot_id])
                for robot_id in participant_ids}
            future_handover_events = tuple(
                deepcopy(event)
                for event in self.handover_events[handover_start:])
            del self.handover_events[handover_start:]
            for runtime in final_runtimes.values():
                slot_id = runtime.standby_slot_id
                if not slot_id:
                    continue
                if slot_id in self.active_standby_claims:
                    raise RuntimeError(
                        f"planned standby slot already claimed: {slot_id}")
                self.active_standby_claims[slot_id] = task.task_id
                new_standby_claims.append(slot_id)
        except Exception:
            del self.handover_events[handover_start:]
            for slot_id in new_standby_claims:
                if self.active_standby_claims.get(slot_id) == task.task_id:
                    del self.active_standby_claims[slot_id]
            for robot_id, runtime in before.items():
                self.robots[robot_id] = runtime
            self._release_resources(resources, task.task_id)
            raise

        # Restore the dispatch-time state while keeping participants busy.  The
        # captured final state is applied only at the completion event.
        for robot_id, runtime in before.items():
            self.robots[robot_id] = runtime
            self.robots[robot_id].robot.task_id = task.task_id
            self.robots[robot_id].robot.cargo_id = ""
            self.robots[robot_id].standby_slot_id = ""
        active = ActiveAssignment(
            assignment=assignment,
            task=task,
            started_at=self.now,
            finish_at=self.now + execution.duration,
            participant_ids=participant_ids,
            resources=resources,
            execution=execution,
            final_runtimes=final_runtimes,
            handover_events=future_handover_events,
        )
        self.active_tasks[task.task_id] = active
        self.queue.remove(task.task_id)
        return active

    def _finalize_active(self, active: ActiveAssignment) -> None:
        for robot_id, runtime in active.final_runtimes.items():
            self.robots[robot_id] = deepcopy(runtime)
        self.handover_events.extend(deepcopy(active.handover_events))
        self._release_resources(active.resources, active.task.task_id)
        for runtime in active.final_runtimes.values():
            slot_id = runtime.standby_slot_id
            if slot_id and self.active_standby_claims.get(slot_id) == \
                    active.task.task_id:
                del self.active_standby_claims[slot_id]
        del self.active_tasks[active.task.task_id]
        if active.execution.success:
            self.completed += 1
            result = "COMPLETED"
        else:
            self.failed += 1
            result = "FAILED"
        self.completion_events.append({
            "task_id": active.task.task_id,
            "started_at": active.started_at,
            "finished_at": active.finish_at,
            "duration": active.execution.duration,
            "task_result": result,
            "failure_reason": active.execution.failure_reason,
            "transport_mode": active.assignment.transport_mode,
            "participant_ids": active.participant_ids,
            "handover_durations_s": list(
                active.execution.handover_durations),
        })

    def _advance_to(self, target_time: float) -> AdvanceMetrics:
        if target_time < self.now:
            raise ValueError("concurrent event target precedes current time")
        metrics = AdvanceMetrics()
        while self.now < target_time:
            event_times = [target_time]
            if self.queue.pending_arrivals:
                event_times.append(self.queue.pending_arrivals[0].arrival_time)
            if self.active_tasks:
                event_times.append(min(
                    item.finish_at for item in self.active_tasks.values()))
            event_time = min(value for value in event_times
                             if value >= self.now)
            elapsed = event_time - self.now
            metrics.waiting_cost += (
                len(self.queue.waiting) + len(self.active_tasks)) * elapsed
            metrics.active_robot_seconds += sum(
                len(item.participant_ids) for item in self.active_tasks.values()
            ) * elapsed
            self.now = event_time

            finished = sorted(
                (item for item in self.active_tasks.values()
                 if item.finish_at <= self.now + 1e-12),
                key=lambda item: (item.finish_at, item.task.task_id))
            for item in finished:
                self._finalize_active(item)
                metrics.finished.append(item)
            self.queue.advance(self.now)
        return metrics

    def _components(self, metrics: AdvanceMetrics) -> dict[str, float]:
        config = self.reward_config
        successful = [item for item in metrics.finished
                      if item.execution.success]
        failed = [item for item in metrics.finished
                  if not item.execution.success]
        deadline = sum(
            config.on_time_bonus if item.finish_at <= item.task.deadline
            else -config.timeout_penalty for item in successful)
        return {
            "outstanding_time": -metrics.waiting_cost /
                                config.outstanding_time_scale,
            "completion": len(successful) * config.completion_bonus,
            "deadline": deadline,
            "handover": -sum(
                len(item.execution.handover_durations)
                for item in metrics.finished) * config.handover_penalty,
            "active_robot_time": -(metrics.active_robot_seconds *
                                   config.active_robot_time_penalty),
            "distance": -sum(
                item.execution.travelled for item in metrics.finished) *
                        config.distance_penalty,
            "failure": -len(failed) * config.timeout_penalty,
        }

    @staticmethod
    def _resolved_rows(metrics: AdvanceMetrics) -> list[dict]:
        return [{
            "task_id": item.task.task_id,
            "task_result": ("COMPLETED" if item.execution.success
                            else "FAILED"),
            "failure_reason": item.execution.failure_reason,
            "transport_mode": item.assignment.transport_mode,
            "finished_at": item.finish_at,
        } for item in metrics.finished]

    def _episode_flags(self) -> tuple[bool, bool, str]:
        terminated = (
            self.completed + self.failed >= self.task_limit and
            not self.active_tasks)
        truncated = self.now >= self.time_limit and not terminated
        reason = "TASK_LIMIT" if terminated else "TIME_LIMIT" if truncated else ""
        return terminated, truncated, reason

    def _cleanup_truncated_active(self) -> None:
        for task_id, active in list(self.active_tasks.items()):
            for robot_id in active.participant_ids:
                runtime = self.robots[robot_id]
                runtime.robot.task_id = ""
                runtime.robot.cargo_id = ""
                runtime.previous_result = "TIME_LIMIT"
            self._release_resources(active.resources, task_id)
            for slot_id, owner in list(self.active_standby_claims.items()):
                if owner == task_id:
                    del self.active_standby_claims[slot_id]
            del self.active_tasks[task_id]

    def step(self, action_index: int) -> DispatchTransition:
        state = self.get_mdp_state()
        candidates = self.enumerate_candidate_actions()
        mask = self.build_action_mask(candidates)
        assignment = self.decode_action(action_index, candidates)
        if not mask[action_index]:
            raise ValueError("selected assignment is masked")
        start = self.now
        active = self._schedule_assignment(assignment)
        target = min(start + self.decision_gap, self.time_limit)
        metrics = self._advance_to(target)
        components = self._components(metrics)
        elapsed = self.now - start
        terminated, truncated, reason = self._episode_flags()
        action = dict(assignment.__dict__)
        action.update({
            "task_result": "DISPATCHED",
            "failure_reason": "",
            "scheduled_finish_at": active.finish_at,
            "active_task_count": len(self.active_tasks),
            "resolved_tasks": self._resolved_rows(metrics),
        })
        if any(row["task_id"] == assignment.task_id
               for row in action["resolved_tasks"]):
            row = next(row for row in action["resolved_tasks"]
                       if row["task_id"] == assignment.task_id)
            action["task_result"] = row["task_result"]
            action["failure_reason"] = row["failure_reason"]
        if truncated:
            self._cleanup_truncated_active()
        config = self.reward_config
        transition = DispatchTransition(
            state, action, mask, sum(components.values()), elapsed,
            self.get_mdp_state(), terminated, truncated, components,
            config.gamma0 ** (elapsed / config.discount_tau_s), reason)
        self.transitions.append(transition)
        return transition

    def wait_for_next_event(self) -> DispatchTransition:
        if not self.active_tasks and not self.queue.pending_arrivals:
            raise RuntimeError("WAIT has no concurrent event")
        state = self.get_mdp_state()
        start = self.now
        event_times = [self.time_limit]
        if self.active_tasks:
            event_times.append(min(
                item.finish_at for item in self.active_tasks.values()))
        if self.queue.pending_arrivals:
            event_times.append(self.queue.pending_arrivals[0].arrival_time)
        metrics = self._advance_to(min(event_times))
        components = self._components(metrics)
        elapsed = self.now - start
        terminated, truncated, reason = self._episode_flags()
        action = {
            "type": "WAIT",
            "active_task_count": len(self.active_tasks),
            "resolved_tasks": self._resolved_rows(metrics),
        }
        if truncated:
            self._cleanup_truncated_active()
        config = self.reward_config
        transition = DispatchTransition(
            state, action, [True], sum(components.values()), elapsed,
            self.get_mdp_state(), terminated, truncated, components,
            config.gamma0 ** (elapsed / config.discount_tau_s), reason)
        self.transitions.append(transition)
        return transition
