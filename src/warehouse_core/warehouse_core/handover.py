from dataclasses import dataclass

from .domain import Cargo, CargoPhase, HandoverPhase, Robot


class HandoverTimeoutError(RuntimeError):
    """Raised after a timed-out transfer has been rolled back safely."""

    failure_code = "HANDOVER_TIMEOUT"

    def __init__(self, elapsed_s: float, timeout_s: float):
        self.elapsed_s = elapsed_s
        self.timeout_s = timeout_s
        super().__init__(
            f"handover took {elapsed_s:.6f}s, exceeding "
            f"P95 timeout {timeout_s:.6f}s")


@dataclass(frozen=True)
class Alignment:
    correct_zone: bool
    positions_match: bool
    distance: float
    distance_min: float
    distance_max: float
    yaw_error: float
    max_yaw_error: float
    linear_speed: float
    angular_speed: float
    max_linear_speed: float
    max_angular_speed: float
    stable_duration: float
    required_stable_duration: float
    collision_ok: bool = True
    zone_exclusive: bool = True

    @property
    def valid(self) -> bool:
        return all((
            self.correct_zone, self.positions_match, self.collision_ok,
            self.zone_exclusive, self.distance_min <= self.distance <= self.distance_max,
            abs(self.yaw_error) <= self.max_yaw_error,
            abs(self.linear_speed) <= self.max_linear_speed,
            abs(self.angular_speed) <= self.max_angular_speed,
            self.stable_duration >= self.required_stable_duration,
        ))


class HandoverTransaction:
    def __init__(self, cargo: Cargo, sender: Robot, receiver: Robot):
        if cargo.owner_id != sender.robot_id or sender.cargo_id != cargo.cargo_id:
            raise ValueError("sender does not uniquely own cargo")
        if receiver.cargo_id or not receiver.available:
            raise ValueError("receiver is not eligible")
        self.cargo, self.sender, self.receiver = cargo, sender, receiver
        self.phase = HandoverPhase.PREPARE
        self._old_constraint = cargo.constraint_id
        self.elapsed_s = 0.0
        self.failure_code = ""

    def transfer(self, alignment: Alignment, new_constraint_id: str) -> None:
        if not alignment.valid:
            self.phase = HandoverPhase.ROLLBACK
            raise ValueError("handover alignment conditions are not satisfied")
        self.phase = HandoverPhase.TRANSFER
        self.cargo.phase = CargoPhase.HANDOVER_TRANSFER
        self.sender.cargo_id = ""
        self.cargo.owner_id = self.receiver.robot_id
        self.cargo.constraint_id = new_constraint_id
        self.receiver.cargo_id = self.cargo.cargo_id
        self.phase = HandoverPhase.VERIFY
        self.cargo.phase = CargoPhase.HANDOVER_VERIFY

    def verify_and_commit(self, owner_confirmed: bool, pose_stable: bool,
                          sender_clear: bool, receiver_confirmed: bool,
                          *, elapsed_s: float = 0.0,
                          timeout_s: float | None = None) -> bool:
        if elapsed_s < 0:
            raise ValueError("handover elapsed time must be non-negative")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("handover timeout must be positive")
        self.elapsed_s = elapsed_s
        if timeout_s is not None and elapsed_s > timeout_s:
            self.failure_code = HandoverTimeoutError.failure_code
            self.rollback()
            return False
        valid = all((owner_confirmed, pose_stable, sender_clear, receiver_confirmed,
                     self.cargo.owner_id == self.receiver.robot_id,
                     self.sender.cargo_id == "", self.receiver.cargo_id == self.cargo.cargo_id))
        if valid:
            self.phase = HandoverPhase.COMMIT
            self.cargo.phase = (CargoPhase.CARRIED_BY_DOG
                                if self.receiver.robot_type == "dog"
                                else CargoPhase.CARRIED_BY_CAR)
            return True
        self.rollback()
        return False

    def rollback(self) -> None:
        self.receiver.cargo_id = ""
        self.sender.cargo_id = self.cargo.cargo_id
        self.cargo.owner_id = self.sender.robot_id
        self.cargo.constraint_id = self._old_constraint
        self.cargo.phase = CargoPhase.RECOVERY
        self.phase = HandoverPhase.ROLLBACK
