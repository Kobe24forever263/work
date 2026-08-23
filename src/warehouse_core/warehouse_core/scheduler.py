from math import dist
from typing import Optional

from .domain import Leg, Robot, Task, TransportChain


class RuleScheduler:
    """Fixed-region baseline with costed alternate stairs and no low-level control."""

    def __init__(self, robots: list[Robot], stairs: dict, weights: dict):
        self.robots = {robot.robot_id: robot for robot in robots}
        self.stairs = stairs
        self.w = weights

    def _available(self, robot_type: str, floor: Optional[int] = None,
                   region: Optional[str] = None) -> list[Robot]:
        return [r for r in self.robots.values()
                if r.robot_type == robot_type and r.available and not r.failure_code
                and (floor is None or r.current_floor == floor)
                and (region is None or r.home_region == region)]

    def _choose_car(self, floor: int, region: str) -> Robot:
        preferred = self._available("car", floor, region)
        candidates = preferred or self._available("car", floor)
        if not candidates:
            raise RuntimeError(f"no available car on floor {floor}")
        return sorted(candidates, key=lambda r: (r.home_region != region, r.robot_id))[0]

    def plan(self, task: Task) -> TransportChain:
        source_car = self._choose_car(task.source.floor, task.source.region)
        if task.source.floor == task.target.floor:
            target_car = self._choose_car(task.target.floor, task.target.region)
            if source_car.robot_id == target_car.robot_id:
                legs = (Leg(source_car.robot_id, "NAVIGATE_CARRY",
                            task.source.point_id, task.target.point_id),)
            else:
                hub = f"F{task.source.floor}_CENTER"
                legs = (
                    Leg(source_car.robot_id, "NAVIGATE_CARRY", task.source.point_id, hub,
                        (f"handover:{hub}",)),
                    Leg(target_car.robot_id, "RECEIVE_AND_DELIVER", hub, task.target.point_id,
                        (f"handover:{hub}",)),
                )
            return TransportChain(task.task_id, None, legs,
                                  self.w["navigation"] * len(legs) + self.w["handover"] * (len(legs)-1))

        target_car = self._choose_car(task.target.floor, task.target.region)
        candidates = []
        for stair_id, stair in self.stairs.items():
            dog = self.robots[stair["dog_id"]]
            if not dog.available or dog.failure_code or stair["state"] in {"BLOCKED", "FAULT"}:
                continue
            source_key = "floor1_xy" if task.source.floor == 1 else "floor2_xy"
            target_key = "floor2_xy" if task.target.floor == 2 else "floor1_xy"
            travel = dist(task.source.xyz[:2], tuple(stair[source_key]))
            travel += dist(task.target.xyz[:2], tuple(stair[target_key]))
            cost = (self.w["navigation"] * travel + self.w["waiting"] * stair["queue_length"]
                    + self.w["handover"] * 2 + self.w["handover_risk"] * stair["risk"]
                    + self.w["robot_occupancy"] * (not dog.available))
            candidates.append((cost, stair_id, dog))
        if not candidates:
            raise RuntimeError("no feasible cross-floor chain")
        cost, stair_id, dog = min(candidates, key=lambda item: (item[0], item[1]))
        lower = f"{stair_id}_F1_SEND"
        upper = f"{stair_id}_F2_RECEIVE"
        if task.source.floor == 1:
            legs = (
                Leg(source_car.robot_id, "NAVIGATE_CARRY", task.source.point_id, lower,
                    (f"handover:{lower}",)),
                Leg(dog.robot_id, "STAIR_CARRY", lower, upper,
                    (f"stair:{stair_id}", f"handover:{lower}", f"handover:{upper}")),
                Leg(target_car.robot_id, "RECEIVE_AND_DELIVER", upper, task.target.point_id,
                    (f"handover:{upper}",)),
            )
        else:
            legs = (
                Leg(source_car.robot_id, "NAVIGATE_CARRY", task.source.point_id, upper,
                    (f"handover:{upper}",)),
                Leg(dog.robot_id, "STAIR_CARRY", upper, lower,
                    (f"stair:{stair_id}", f"handover:{upper}", f"handover:{lower}")),
                Leg(target_car.robot_id, "RECEIVE_AND_DELIVER", lower, task.target.point_id,
                    (f"handover:{lower}",)),
            )
        return TransportChain(task.task_id, stair_id, legs, cost)


def action_mask(robot: Robot, has_candidate: bool, stair_available: bool,
                handover_role: str = "") -> dict[str, bool]:
    return {
        "ACCEPT_TASK": robot.available and not robot.task_id and has_candidate,
        "REJECT_TASK": has_candidate,
        "NAVIGATE": robot.available,
        "SELECT_STAIR": robot.robot_type == "dog" and stair_available,
        "ASCEND_STAIRS": robot.robot_type == "dog" and robot.current_floor == 1 and stair_available,
        "DESCEND_STAIRS": robot.robot_type == "dog" and robot.current_floor == 2 and stair_available,
        "SEND_HANDOVER": bool(robot.cargo_id) and handover_role == "sender",
        "ACCEPT_HANDOVER": not robot.cargo_id and handover_role == "receiver",
        "WAIT": True, "REPLAN": bool(robot.task_id), "TRANSFER_TASK": bool(robot.task_id),
        "RETURN_HOME": not robot.cargo_id, "HOLD": True,
    }
