"""Deterministic failure recovery for task, resource and cargo state."""

from dataclasses import dataclass
from enum import Enum

from .domain import Cargo, CargoPhase, Robot
from .reservations import ReservationBook
from .spatial import StairManager


class FaultType(str, Enum):
    PLANNING_FAILURE = "PLANNING_FAILURE"
    HANDOVER_TIMEOUT = "HANDOVER_TIMEOUT"
    ROBOT_FAILURE = "ROBOT_FAILURE"
    STAIR_TIMEOUT = "STAIR_TIMEOUT"
    REGION_LOCK_TIMEOUT = "REGION_LOCK_TIMEOUT"
    SENSOR_FAILURE = "SENSOR_FAILURE"
    TASK_CANCELLED = "TASK_CANCELLED"


class RecoveryDisposition(str, Enum):
    RETRY = "RETRY"
    REASSIGN = "REASSIGN"
    FAIL_TASK = "FAIL_TASK"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class RecoveryResult:
    task_id: str
    fault: FaultType
    disposition: RecoveryDisposition
    released_general_locks: int
    released_stair_claims: int
    ownership_valid: bool
    completed_within_limit: bool


class FailureRecoveryCoordinator:
    RETRYABLE = {
        FaultType.PLANNING_FAILURE,
        FaultType.STAIR_TIMEOUT,
        FaultType.REGION_LOCK_TIMEOUT,
        FaultType.SENSOR_FAILURE,
    }

    def __init__(self, reservations: ReservationBook,
                 stairs: StairManager, clock, recovery_limit_s: float = 5.0):
        self.reservations = reservations
        self.stairs = stairs
        self.clock = clock
        self.recovery_limit_s = recovery_limit_s

    @staticmethod
    def _ownership_valid(cargo: Cargo, robots: tuple[Robot, ...]) -> bool:
        holders = [robot.robot_id for robot in robots
                   if robot.cargo_id == cargo.cargo_id]
        return len(holders) == 1 and holders[0] == cargo.owner_id

    def recover(self, task_id: str, fault: FaultType, cargo: Cargo,
                robots: tuple[Robot, ...], safe_owner: Robot) -> RecoveryResult:
        started = self.clock()
        # Any partial transfer is deterministically restored to the safe owner.
        for robot in robots:
            robot.cargo_id = ""
        safe_owner.cargo_id = cargo.cargo_id
        cargo.owner_id = safe_owner.robot_id
        cargo.constraint_id = f"constraint:{safe_owner.robot_id}"
        cargo.phase = CargoPhase.RECOVERY

        general = self.reservations.release_task(task_id)
        stair = self.stairs.release_task(task_id)

        if fault == FaultType.TASK_CANCELLED:
            disposition = RecoveryDisposition.CANCELLED
        elif fault == FaultType.HANDOVER_TIMEOUT:
            disposition = RecoveryDisposition.FAIL_TASK
        elif fault == FaultType.ROBOT_FAILURE:
            safe_owner.available = False
            safe_owner.failure_code = fault.value
            disposition = RecoveryDisposition.REASSIGN
        elif fault in self.RETRYABLE:
            disposition = RecoveryDisposition.RETRY
        else:
            raise ValueError(f"unsupported fault: {fault}")

        ownership = self._ownership_valid(cargo, robots)
        elapsed = self.clock() - started
        return RecoveryResult(
            task_id, fault, disposition, general, stair, ownership,
            elapsed <= self.recovery_limit_s)
