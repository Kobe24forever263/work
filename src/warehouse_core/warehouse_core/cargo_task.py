"""Auditable cargo ownership lifecycle for a cross-floor transport task."""

from dataclasses import dataclass, field

from .domain import Cargo, CargoPhase, Robot
from .handover import Alignment, HandoverTimeoutError, HandoverTransaction


@dataclass(frozen=True)
class OwnershipEvent:
    sequence: int
    event: str
    phase: str
    owner_id: str | None
    holders: tuple[str, ...]


@dataclass
class CargoTaskLifecycle:
    cargo: Cargo
    robots: tuple[Robot, ...]
    events: list[OwnershipEvent] = field(default_factory=list)

    def _holders(self) -> tuple[str, ...]:
        return tuple(sorted(r.robot_id for r in self.robots
                            if r.cargo_id == self.cargo.cargo_id))

    def audit(self, event: str, *, delivered: bool = False) -> None:
        holders = self._holders()
        if delivered:
            valid = (self.cargo.phase == CargoPhase.DELIVERED and
                     self.cargo.owner_id is None and not holders and
                     self.cargo.constraint_id is None)
        else:
            valid = (len(holders) == 1 and
                     self.cargo.owner_id == holders[0] and
                     self.cargo.phase != CargoPhase.DELIVERED)
        if not valid:
            raise AssertionError(
                f"cargo ownership invariant failed after {event}: "
                f"phase={self.cargo.phase.value}, owner={self.cargo.owner_id}, "
                f"holders={holders}, constraint={self.cargo.constraint_id}")
        self.events.append(OwnershipEvent(
            len(self.events) + 1, event, self.cargo.phase.value,
            self.cargo.owner_id, holders))

    def pickup(self, carrier: Robot, constraint_id: str) -> None:
        if self.cargo.phase != CargoPhase.WAITING or self.cargo.owner_id is not None:
            raise ValueError("cargo is not available for pickup")
        if carrier.cargo_id or not carrier.available:
            raise ValueError("pickup carrier is not eligible")
        self.cargo.phase = CargoPhase.PICKUP_PROCESSING
        self.cargo.owner_id = carrier.robot_id
        self.cargo.constraint_id = constraint_id
        carrier.cargo_id = self.cargo.cargo_id
        self.cargo.phase = (CargoPhase.CARRIED_BY_DOG
                            if carrier.robot_type == "dog"
                            else CargoPhase.CARRIED_BY_CAR)
        self.audit("pickup_commit")

    def handover(self, sender: Robot, receiver: Robot, alignment: Alignment,
                 constraint_id: str, *, elapsed_s: float = 0.0,
                 timeout_s: float | None = None) -> None:
        tx = HandoverTransaction(self.cargo, sender, receiver)
        tx.transfer(alignment, constraint_id)
        if not tx.verify_and_commit(
                True, True, True, True,
                elapsed_s=elapsed_s, timeout_s=timeout_s):
            if tx.failure_code == HandoverTimeoutError.failure_code:
                self.audit(
                    f"handover_timeout:{sender.robot_id}->{receiver.robot_id}")
                raise HandoverTimeoutError(elapsed_s, timeout_s)
            raise AssertionError("verified handover unexpectedly rolled back")
        self.audit(f"handover_commit:{sender.robot_id}->{receiver.robot_id}")

    def deliver(self, carrier: Robot) -> None:
        if (self.cargo.owner_id != carrier.robot_id or
                carrier.cargo_id != self.cargo.cargo_id):
            raise ValueError("delivery carrier does not own cargo")
        self.cargo.phase = CargoPhase.DELIVERING
        carrier.cargo_id = ""
        self.cargo.owner_id = None
        self.cargo.constraint_id = None
        self.cargo.phase = CargoPhase.DELIVERED
        self.audit("delivery_commit", delivered=True)
