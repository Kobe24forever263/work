from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CargoPhase(str, Enum):
    WAITING = "WAITING"
    RESERVED = "RESERVED"
    PICKUP_PROCESSING = "PICKUP_PROCESSING"
    CARRIED_BY_CAR = "CARRIED_BY_CAR"
    CARRIED_BY_DOG = "CARRIED_BY_DOG"
    HANDOVER_PREPARE = "HANDOVER_PREPARE"
    HANDOVER_ALIGN = "HANDOVER_ALIGN"
    HANDOVER_TRANSFER = "HANDOVER_TRANSFER"
    HANDOVER_VERIFY = "HANDOVER_VERIFY"
    DELIVERING = "DELIVERING"
    DELIVERED = "DELIVERED"
    RECOVERY = "RECOVERY"
    FAILED = "FAILED"


class HandoverPhase(str, Enum):
    PREPARE = "PREPARE"
    RESERVE = "RESERVE"
    APPROACH = "APPROACH"
    ALIGN = "ALIGN"
    TRANSFER = "TRANSFER"
    VERIFY = "VERIFY"
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"


@dataclass(frozen=True)
class Point:
    point_id: str
    floor: int
    region: str
    xyz: tuple[float, float, float]


@dataclass
class Robot:
    robot_id: str
    robot_type: str
    home_floor: int
    home_region: str
    current_floor: int
    current_region: str
    available: bool = True
    task_id: str = ""
    cargo_id: str = ""
    failure_code: str = ""


@dataclass(frozen=True)
class Task:
    task_id: str
    cargo_id: str
    source: Point
    target: Point
    priority: int
    deadline: float


@dataclass
class Cargo:
    cargo_id: str
    task_id: str
    phase: CargoPhase = CargoPhase.WAITING
    owner_id: Optional[str] = None
    constraint_id: Optional[str] = None


@dataclass(frozen=True)
class Leg:
    robot_id: str
    skill: str
    source_id: str
    target_id: str
    resource_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TransportChain:
    task_id: str
    stair_id: Optional[str]
    legs: tuple[Leg, ...]
    estimated_cost: float
