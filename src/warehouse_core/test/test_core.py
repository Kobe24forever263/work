import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from warehouse_core.domain import Cargo, CargoPhase, Point, Robot, Task
from warehouse_core.handover import (
    Alignment, HandoverTimeoutError, HandoverTransaction)
from warehouse_core.handover_timing import (
    CAR_TO_DOG, DOG_TO_CAR, HandoverTimeoutPolicy,
    build_provisional_calibration,
    build_provisional_policy_and_keyed_provider)
from warehouse_core.cargo_task import CargoTaskLifecycle
from warehouse_core.reservations import ReservationBook
from warehouse_core.recovery import (
    FailureRecoveryCoordinator, FaultType, RecoveryDisposition)
from warehouse_core.scheduler import RuleScheduler, action_mask
from warehouse_core.spatial import (
    FloorTransitionTracker, RegionManager, StairManager)
from warehouse_core.persistent_dispatch import (
    Assignment, AsyncTaskQueue, PersistentDispatchEnvironment, QueuedTask,
    RewardConfig, RobotRuntime, StandbySlot)
from warehouse_core.concurrent_dispatch import (
    ConcurrentPersistentDispatchEnvironment)
from warehouse_core.stage12_encoding import Stage12Encoder
from warehouse_core.stage14_training import (
    STAGE14_REWARD_CONFIG, Stage14EpisodeSpec, WarehouseDispatchGymEnv,
    build_balanced_arrivals, build_mixed_load_arrivals,
    build_mixed_recovery_arrivals, build_mixed_curriculum_arrivals)
from warehouse_core.stage14_policy import MaskedCandidateActorCritic
from warehouse_core.stage14_ppo import (
    RolloutStep, RunningReturnNormalizer, compute_smdp_gae, evaluate)


def robot(robot_id, kind, floor, region):
    return Robot(robot_id, kind, floor, region, floor, region)


def test_task_keyed_handover_samples_do_not_depend_on_call_order():
    _, first = build_provisional_policy_and_keyed_provider(20260809)
    _, second = build_provisional_policy_and_keyed_provider(20260809)
    expected = {
        ("T1", CAR_TO_DOG): first(CAR_TO_DOG, "T1"),
        ("T1", DOG_TO_CAR): first(DOG_TO_CAR, "T1"),
        ("T2", CAR_TO_DOG): first(CAR_TO_DOG, "T2"),
    }
    actual = {
        ("T2", CAR_TO_DOG): second(CAR_TO_DOG, "T2"),
        ("T1", DOG_TO_CAR): second(DOG_TO_CAR, "T1"),
        ("T1", CAR_TO_DOG): second(CAR_TO_DOG, "T1"),
    }
    assert actual == expected


def test_reservation_is_exclusive_and_owner_checked():
    now = [0.0]
    book = ReservationBook(clock=lambda: now[0])
    first = book.reserve("stair:S1", "dog_1", "T1", 10)
    assert first
    assert book.reserve("stair:S1", "dog_2", "T2", 10) is None
    assert not book.release(first.reservation_id, "dog_2")
    assert book.release(first.reservation_id, "dog_1")


def test_failed_verify_rolls_unique_owner_back():
    sender, receiver = robot("car_f1_1", "car", 1, "F1_NE"), robot("dog_1", "dog", 1, "F1_NE")
    sender.cargo_id = "C1"
    cargo = Cargo("C1", "T1", CargoPhase.CARRIED_BY_CAR, sender.robot_id, "old")
    tx = HandoverTransaction(cargo, sender, receiver)
    tx.transfer(Alignment(True, True, .5, .35, .75, .1, .2, .01, .01, .03, .05, 2, 1.5), "new")
    assert not tx.verify_and_commit(True, False, True, True)
    assert cargo.owner_id == sender.robot_id and sender.cargo_id == "C1" and not receiver.cargo_id


def test_invalid_alignment_rejects_without_changing_owner():
    sender = robot("car_f1_1", "car", 1, "F1_NE")
    receiver = robot("dog_1", "dog", 1, "F1_NE")
    sender.cargo_id = "C_BAD_ALIGN"
    cargo = Cargo("C_BAD_ALIGN", "T_BAD_ALIGN", CargoPhase.CARRIED_BY_CAR,
                  sender.robot_id, "old_constraint")
    tx = HandoverTransaction(cargo, sender, receiver)
    invalid = Alignment(True, True, 1.2, .35, .75, 0, .2, 0, 0, .03, .05, 2, 1.5)
    try:
        tx.transfer(invalid, "new_constraint")
        assert False, "invalid alignment should be rejected"
    except ValueError:
        pass
    assert tx.phase.value == "ROLLBACK"
    assert cargo.owner_id == sender.robot_id
    assert sender.cargo_id == cargo.cargo_id and not receiver.cargo_id
    assert cargo.constraint_id == "old_constraint"


def test_cross_floor_cargo_has_one_owner_then_no_residual_holder():
    car1 = robot("car_f1_1", "car", 1, "F1_NE")
    dog = robot("dog_1", "dog", 1, "F1_NE")
    car2 = robot("car_f2_2", "car", 2, "F2_EAST")
    cargo = Cargo("C2", "T2")
    lifecycle = CargoTaskLifecycle(cargo, (car1, dog, car2))
    aligned = Alignment(True, True, .5, .35, .75, 0, .2, 0, 0, .03, .05, 2, 1.5)
    lifecycle.pickup(car1, "car1_constraint")
    lifecycle.handover(car1, dog, aligned, "dog_constraint")
    lifecycle.handover(dog, car2, aligned, "car2_constraint")
    lifecycle.deliver(car2)
    assert cargo.phase == CargoPhase.DELIVERED
    assert cargo.owner_id is None
    assert not any(r.cargo_id for r in (car1, dog, car2))
    assert len(lifecycle.events) == 4


def test_action_mask_prevents_car_stairs_and_empty_sender():
    car = robot("car_f1_1", "car", 1, "F1_NE")
    mask = action_mask(car, True, True, "sender")
    assert not mask["ASCEND_STAIRS"] and not mask["SELECT_STAIR"] and not mask["SEND_HANDOVER"]


def test_cross_floor_chain_is_car_dog_car_and_costs_queue():
    robots = [
        robot("dog_1", "dog", 1, "F1_NE"), robot("dog_2", "dog", 1, "F1_NW"),
        robot("car_f1_1", "car", 1, "F1_NE"), robot("car_f2_1", "car", 2, "F2_WEST"),
    ]
    stairs = {
        "S1": {"dog_id": "dog_1", "floor1_xy": [10, 10], "floor2_xy": [10, 5], "state": "FREE", "queue_length": 9, "risk": .1},
        "S2": {"dog_id": "dog_2", "floor1_xy": [-10, 10], "floor2_xy": [-10, 5], "state": "FREE", "queue_length": 0, "risk": .1},
    }
    weights = {"navigation": 1, "waiting": 10, "handover": 5, "handover_risk": 1, "robot_occupancy": 1}
    task = Task("T", "C", Point("A", 1, "F1_NE", (9, 9, 0)), Point("B", 2, "F2_WEST", (-9, 5, 4)), 1, 300)
    chain = RuleScheduler(robots, stairs, weights).plan(task)
    assert chain.stair_id == "S2"
    assert [leg.robot_id for leg in chain.legs] == ["car_f1_1", "dog_2", "car_f2_1"]


def test_downstairs_cost_uses_floor2_entry_and_floor1_exit():
    robots = [
        robot("dog_ne", "dog", 2, "F2_EAST"),
        robot("dog_nw", "dog", 2, "F2_WEST"),
        robot("car_f2", "car", 2, "F2_EAST"),
        robot("car_f1", "car", 1, "F1_NW"),
    ]
    stairs = {
        "NE": {"dog_id": "dog_ne", "floor1_xy": [100, 100],
               "floor2_xy": [10, 0], "state": "FREE",
               "queue_length": 0, "risk": 0},
        "NW": {"dog_id": "dog_nw", "floor1_xy": [-10, 0],
               "floor2_xy": [-100, 100], "state": "FREE",
               "queue_length": 0, "risk": 0},
    }
    weights = {"navigation": 1, "waiting": 1, "handover": 1,
               "handover_risk": 1, "robot_occupancy": 1}
    task = Task("TD", "CD",
                Point("F2", 2, "F2_EAST", (9, 0, 4.5)),
                Point("F1", 1, "F1_NW", (-9, 0, .45)), 1, 300)
    assert RuleScheduler(robots, stairs, weights).plan(task).stair_id == "NE"


def test_stair_rejects_car_queues_dog_and_checks_owner():
    now = [0.0]
    manager = StairManager(
        ["S1"], {"dog_1": "dog", "dog_2": "dog", "car_1": "car"},
        clock=lambda: now[0])
    assert manager.reserve("S1", "car_1", "T0", 10)[1] == \
        "ROBOT_NOT_STAIR_CAPABLE"
    first, reason = manager.reserve("S1", "dog_1", "T1", 10)
    assert first and reason == "ACCEPTED"
    assert manager.reserve("S1", "dog_2", "T2", 10)[1] == "QUEUED"
    assert not manager.release(first.reservation_id, "dog_2")
    assert manager.enter("S1", "dog_1")
    assert manager.release(first.reservation_id, "dog_1")
    assert manager.resources["S1"].lease.robot_id == "dog_2"


def test_stair_timeout_promotes_queue():
    now = [0.0]
    manager = StairManager(
        ["S1"], {"dog_1": "dog", "dog_2": "dog"},
        clock=lambda: now[0])
    manager.reserve("S1", "dog_1", "T1", 2)
    manager.reserve("S1", "dog_2", "T2", 10)
    now[0] = 3.0
    manager.purge()
    assert manager.resources["S1"].lease.robot_id == "dog_2"


def test_failure_recovery_releases_all_locks_and_restores_owner():
    now = [0.0]
    book = ReservationBook(clock=lambda: now[0])
    stairs = StairManager(["S1"], {"dog_1": "dog", "dog_2": "dog"},
                          clock=lambda: now[0])
    sender = robot("car_f1_1", "car", 1, "F1_NE")
    receiver = robot("dog_1", "dog", 1, "F1_NE")
    cargo = Cargo("C_REC", "T_REC", CargoPhase.HANDOVER_VERIFY,
                  receiver.robot_id, "partial")
    receiver.cargo_id = cargo.cargo_id
    book.reserve("region:F1_NE", sender.robot_id, "T_REC", 20)
    book.reserve("handover:NE", sender.robot_id, "T_REC", 20)
    stairs.reserve("S1", receiver.robot_id, "T_REC", 20)
    result = FailureRecoveryCoordinator(
        book, stairs, lambda: now[0]).recover(
            "T_REC", FaultType.HANDOVER_TIMEOUT, cargo,
            (sender, receiver), sender)
    assert result.disposition == RecoveryDisposition.FAIL_TASK
    assert result.released_general_locks == 2
    assert result.released_stair_claims == 1
    assert result.ownership_valid and result.completed_within_limit
    assert book.reserve("region:F1_NE", "next_robot", "T_NEXT", 20)
    lease, reason = stairs.reserve("S1", "dog_2", "T_NEXT", 20)
    assert lease and reason == "ACCEPTED"


def test_handover_p95_uses_strict_greater_than_and_current_modes_only():
    samples = build_provisional_calibration()
    policy = HandoverTimeoutPolicy.from_samples(samples)
    assert len(samples) == 1000
    assert {sample.handover_kind for sample in samples} == {
        "CAR_TO_DOG", "DOG_TO_CAR"}
    assert sum(sample.duration_s > policy.timeout_s for sample in samples) == 50
    assert not policy.timed_out(policy.timeout_s)
    assert policy.timed_out(policy.timeout_s + 1e-9)


def test_handover_timeout_rolls_back_unique_owner():
    sender = robot("car", "car", 1, "F1")
    receiver = robot("dog", "dog", 1, "F1")
    cargo = Cargo("C_TIMEOUT", "T_TIMEOUT", CargoPhase.CARRIED_BY_CAR,
                  sender.robot_id, "old_constraint")
    sender.cargo_id = cargo.cargo_id
    tx = HandoverTransaction(cargo, sender, receiver)
    aligned = Alignment(
        True, True, .5, .35, .75, 0, .2, 0, 0, .03, .05, 2, 1.5)
    tx.transfer(aligned, "dog_constraint")
    assert not tx.verify_and_commit(
        True, True, True, True, elapsed_s=2.500001, timeout_s=2.5)
    assert tx.failure_code == HandoverTimeoutError.failure_code
    assert tx.phase.value == "ROLLBACK"
    assert cargo.owner_id == sender.robot_id
    assert sender.cargo_id == cargo.cargo_id and not receiver.cargo_id


def test_floor_switch_needs_order_stability_and_localization():
    now = [0.0]
    tracker = FloorTransitionTracker(1.5, clock=lambda: now[0])
    # Height alone must never switch floors.
    assert tracker.update(
        z=4.56, entry_distance=99, exit_distance=0, speed=0,
        reserved=False, occupied=False, localization_ok=True) == 1
    tracker.update(
        z=.45, entry_distance=.2, exit_distance=9, speed=.1,
        reserved=True, occupied=False, localization_ok=True)
    tracker.update(
        z=1.0, entry_distance=.5, exit_distance=7, speed=.2,
        reserved=True, occupied=True, localization_ok=True)
    tracker.update(
        z=4.56, entry_distance=7, exit_distance=.3, speed=0,
        reserved=True, occupied=True, localization_ok=True)
    now[0] = 2.0
    assert tracker.update(
        z=4.56, entry_distance=7, exit_distance=.3, speed=0,
        reserved=True, occupied=True, localization_ok=False) == 1
    now[0] = 3.0
    tracker.update(
        z=4.56, entry_distance=7, exit_distance=.3, speed=0,
        reserved=True, occupied=True, localization_ok=True)
    now[0] = 4.6
    assert tracker.update(
        z=4.56, entry_distance=7, exit_distance=.3, speed=0,
        reserved=True, occupied=True, localization_ok=True) == 2


def test_region_capacity_inputs_are_configuration_driven():
    manager = RegionManager({
        "F1_NE": {"floor": 1, "bounds_xy": [0, 10, 0, 10], "capacity": 1},
        "F2_E": {"floor": 2, "bounds_xy": [0, 10, 0, 10], "capacity": 2},
    })
    assert manager.locate(3, 4, 1) == "F1_NE"
    assert manager.locate(3, 4, 2) == "F2_E"
    assert manager.occupancy({"dog_1": "F1_NE"})["F1_NE"] == ["dog_1"]


def test_async_queue_keeps_all_tasks_but_limits_policy_view():
    source = Point("A", 1, "F1_NE", (0, 0, 0))
    target = Point("B", 1, "F1_NE", (1, 0, 0))
    arrivals = [QueuedTask(Task(f"T{i}", f"C{i}", source, target,
                                i % 3, 100 + i), i)
                for i in range(10)]
    queue = AsyncTaskQueue(arrivals)
    assert queue.advance(20) == 10
    assert len(queue.waiting) == 10
    view = queue.policy_view(20, limit=8)
    assert len(view) == 8 and len(queue.waiting) == 10


def _small_persistent_env(extra_arrivals=(), deadline=100):
    source = Point("A", 1, "F1_A", (0, 0, .45))
    target = Point("B", 2, "F2_B", (2, 0, 4.7))
    task = Task("T1", "C1", source, target, 1, deadline)
    runtimes = [
        RobotRuntime(robot("car1", "car", 1, "F1_A"), (0, 0, .45)),
        RobotRuntime(robot("dog1", "dog", 1, "F1_A"), (1, 0, .45)),
        RobotRuntime(robot("car2", "car", 2, "F2_B"), (2, 0, 4.7)),
    ]
    stairs = {"S1": {"floor1_xyz": (1, 0, .45),
                       "floor2_xyz": (1, 0, 4.7), "state": "FREE"}}
    queue = AsyncTaskQueue([QueuedTask(task, 0), *extra_arrivals])
    slots = [
        StandbySlot("C1", "car", 1, (-2, 0, .45)),
        StandbySlot("C2", "car", 2, (3, 0, 4.7)),
        StandbySlot("D2", "dog", 2, (0, 0, 4.7)),
    ]
    return PersistentDispatchEnvironment(runtimes, stairs, queue,
                                         task_limit=1, decision_gap=3,
                                         standby_slots=slots)


def test_persistent_mask_rejects_busy_faulted_or_loaded_participant():
    env = _small_persistent_env()
    valid = next(action for action in env.enumerate_candidate_actions()
                 if action.transport_mode == env.CAR_DOG_CAR)
    for robot_id, field, value in (
            ("car1", "task_id", "BUSY"),
            ("car2", "cargo_id", "OTHER_CARGO"),
            ("dog1", "failure_code", "FAULT")):
        setattr(env.robots[robot_id].robot, field, value)
        assert not env.is_assignment_legal(valid)
        setattr(env.robots[robot_id].robot, field, "")


def test_persistent_resources_are_claimed_and_released():
    env = _small_persistent_env()
    env.step(0)
    assert not env.resource_claims
    assert env.stairs["S1"]["state"] == "FREE"
    claims = [event for event in env.resource_events if event["event"] == "CLAIM"]
    releases = [event for event in env.resource_events if event["event"] == "RELEASE"]
    assert claims and len(claims) == len(releases)
    assert any(event["resource"] == "stair:S1" for event in claims)


def test_persistent_handover_timeout_fails_task_without_ending_early():
    durations = iter((2.0, 2.6))
    env = _small_persistent_env()
    env.handover_timeout_s = 2.5
    env.handover_duration_provider = lambda _: next(durations)
    candidates = env.enumerate_candidate_actions()
    action_index = next(
        index for index, action in enumerate(candidates)
        if action.transport_mode == env.CAR_DOG_CAR)
    transition = env.step(action_index)
    assert transition.action["task_result"] == "FAILED"
    assert transition.action["failure_reason"] == "HANDOVER_TIMEOUT"
    assert transition.reward_components["failure"] < 0
    assert env.failed == 1 and env.completed == 0
    assert transition.terminated and transition.termination_reason == "TASK_LIMIT"
    assert len(env.handover_events) == 2
    assert not env.handover_events[0]["timed_out"]
    assert env.handover_events[1]["timed_out"]
    assert not env.resource_claims
    assert env.stairs["S1"]["state"] == "FREE"
    assert all(not runtime.robot.task_id and not runtime.robot.cargo_id
               for runtime in env.robots.values())


def test_first_handover_timeout_keeps_persistent_dog_floor_balance():
    env = _small_persistent_env()
    env.handover_timeout_s = 2.5
    env.handover_duration_provider = lambda _: 2.6
    candidates = env.enumerate_candidate_actions()
    action_index = next(
        index for index, action in enumerate(candidates)
        if action.transport_mode == env.CAR_DOG_CAR)
    transition = env.step(action_index)
    assert transition.action["failure_reason"] == "HANDOVER_TIMEOUT"
    assert len(env.handover_events) == 1
    assert env.robots["dog1"].robot.current_floor == 2
    assert env.robots["dog1"].standby_slot_id == "D2"


def test_smdp_waiting_cost_integrates_mid_interval_arrival():
    source = Point("Q1", 1, "F1_A", (0, 0, .45))
    target = Point("Q2", 1, "F1_A", (1, 0, .45))
    later = QueuedTask(Task("T2", "C2", source, target, 1, 100), 1.5)
    env = _small_persistent_env([later])
    transition = env.step(0)
    # T1 is outstanding throughout. T2 contributes only after t=1.5.
    execution_duration = transition.delta_time - env.decision_gap
    expected = -(execution_duration + transition.delta_time - 1.5) / 100
    assert abs(transition.reward_components["outstanding_time"] - expected) < 1e-9
    assert abs(transition.reward - sum(transition.reward_components.values())) < 1e-9
    assert 0 < transition.discount < 1


def test_stage12_encoding_is_fixed_shape_and_exposes_stair_tradeoff():
    source = Point("F1_NE", 1, "F1_NE", (12, 8, .45))
    target = Point("F2_SW", 2, "F2_WEST", (-7, -6.5, 4.7))
    task = Task("T_BALANCE", "C_BALANCE", source, target, 2, 300)
    runtimes = [
        RobotRuntime(robot("car_f1", "car", 1, "F1_NE"), (10, 8, .45)),
        RobotRuntime(robot("dog", "dog", 1, "F1_NE"), (20, 18, .45)),
        RobotRuntime(robot("car_f2", "car", 2, "F2_EAST"), (5.5, 0, 4.7)),
    ]
    stairs = {
        "NE": {"floor1_xyz": (20, 18, .45),
               "floor2_xyz": (13, 10, 4.7), "state": "FREE"},
        "SW": {"floor1_xyz": (-20, -18, .45),
               "floor2_xyz": (-10, -7, 4.7), "state": "FREE"},
    }
    env = PersistentDispatchEnvironment(
        runtimes, stairs, AsyncTaskQueue([QueuedTask(task, 0)]), task_limit=1)
    encoded = Stage12Encoder().encode(env)
    assert encoded.observation.shape == (Stage12Encoder.OBSERVATION_SIZE,)
    assert encoded.action_features.shape == (
        Stage12Encoder.MAX_ACTIONS, Stage12Encoder.ACTION_WIDTH)
    assert encoded.action_mask.shape == (Stage12Encoder.MAX_ACTIONS,)
    actions = [(index, action) for index, action in enumerate(encoded.action_ids)
               if action is not None]
    ne_index = next(index for index, action in actions
                    if action.stair_id == "NE" and
                    action.transport_mode == env.CAR_DOG_CAR)
    sw_index = next(index for index, action in actions
                    if action.stair_id == "SW" and
                    action.transport_mode == env.CAR_DOG_CAR)
    # NE is cheaper on the source side; SW is cheaper on the target side.
    assert encoded.action_features[ne_index, 12] < encoded.action_features[sw_index, 12]
    assert encoded.action_features[ne_index, 14] > encoded.action_features[sw_index, 14]
    assert encoded.action_mask[ne_index] and encoded.action_mask[sw_index]
    assert not encoded.action_mask[len(actions):].any()
    assert Stage12Encoder().decode(encoded, ne_index) == actions[ne_index][1]


def test_stage12_encoding_is_deterministic_and_includes_task_target():
    env = _small_persistent_env()
    encoder = Stage12Encoder()
    first = encoder.encode(env)
    second = encoder.encode(env)
    assert np.array_equal(first.observation, second.observation)
    assert np.array_equal(first.action_features, second.action_features)
    assert np.array_equal(first.action_mask, second.action_mask)
    task_offset = encoder.GLOBAL_WIDTH
    target_xyz = first.observation[
        task_offset + 10:task_offset + encoder.TASK_WIDTH]
    assert np.allclose(target_xyz, np.asarray((2 / 25, 0, 4.7 / 5),
                                              dtype=np.float32))


def test_stage12_candidate_directly_binds_task_context_and_relative_cost():
    env = _small_persistent_env()
    encoded = Stage12Encoder().encode(env)
    legal_indices = np.flatnonzero(encoded.action_mask)
    assignments = [
        (index, encoded.action_ids[index]) for index in legal_indices
        if isinstance(encoded.action_ids[index], Assignment)]
    assert assignments
    for index, assignment in assignments:
        task = env._task(assignment.task_id).task
        feature = encoded.action_features[index]
        assert np.allclose(feature[19:22], Stage12Encoder._xyz(
            task.source.xyz))
        assert np.allclose(feature[22:25], Stage12Encoder._xyz(
            task.target.xyz))
        assert feature[25] >= 0
        assert feature[26] >= 1 / 3 - 1e-6
    assert min(encoded.action_features[index, 25]
               for index, _ in assignments) == 0


def test_stage12_wait_is_the_only_legal_action_without_ready_candidate():
    env = _small_persistent_env()
    env.robots["car1"].robot.failure_code = "FAULT"
    env.robots["dog1"].robot.failure_code = "FAULT"
    encoder = Stage12Encoder()
    encoded = encoder.encode(env)
    assert encoded.action_mask.sum() == 1
    assert encoder.decode(encoded, 0) == encoder.WAIT_ACTION
    try:
        encoder.decode(encoded, 1)
        assert False, "padding must not be decodable"
    except ValueError:
        pass


def test_transport_modes_exclude_car_to_car_and_use_configured_speeds():
    source = Point("A", 1, "F1_A", (0, 0, .45))
    target = Point("B", 1, "F1_B", (8, 0, .45))
    task = Task("TM", "CM", source, target, 1, 100)
    runtimes = [
        RobotRuntime(robot("car_a", "car", 1, "F1_A"), (0, 0, .45)),
        RobotRuntime(robot("car_b", "car", 1, "F1_B"), (8, 0, .45)),
        RobotRuntime(robot("dog_a", "dog", 1, "F1_A"), (0, 0, .45)),
    ]
    env = PersistentDispatchEnvironment(
        runtimes, {}, AsyncTaskQueue([QueuedTask(task, 0)]), task_limit=1,
        robot_speeds={"car": 2.0, "dog": .8})
    actions = env.enumerate_candidate_actions()
    assert {action.transport_mode for action in actions} == {
        env.SINGLE_CAR, env.SINGLE_DOG}
    assert not any(action.pickup_carter and action.receiving_carter and
                   action.pickup_carter != action.receiving_carter
                   for action in actions)
    fastest_car = min(action.estimated_cost for action in actions
                      if action.transport_mode == env.SINGLE_CAR)
    fastest_dog = min(action.estimated_cost for action in actions
                      if action.transport_mode == env.SINGLE_DOG)
    assert fastest_car < fastest_dog
    state = env.get_mdp_state()
    speeds = {item["type"]: item["nominal_speed_mps"]
              for item in state["robots"]}
    assert speeds == {"car": 2.0, "dog": .8}


def test_stage13_time_discount_and_episode_boundary_are_explicit():
    env = _small_persistent_env()
    env.reward_config = RewardConfig(gamma0=.99, discount_tau_s=10)
    transition = env.step(0)
    assert transition.terminated and not transition.truncated
    assert transition.termination_reason == "TASK_LIMIT"
    assert abs(transition.discount -
               .99 ** (transition.delta_time / 10)) < 1e-12
    assert transition.reward_components["completion"] > 0
    assert transition.reward_components["handover"] <= 0


def test_stage13_timeout_is_truncation_not_success_termination():
    env = _small_persistent_env()
    env.task_limit = 10
    env.time_limit = .5
    transition = env.step(0)
    assert transition.truncated and not transition.terminated
    assert transition.termination_reason == "TIME_LIMIT"


def test_stage13_deadline_miss_is_penalized_without_severe_termination():
    env = _small_persistent_env(deadline=0)
    env.task_limit = 10
    transition = env.step(0)
    assert transition.reward_components["deadline"] == \
        -env.reward_config.timeout_penalty
    assert transition.reward_components["failure"] == 0
    assert not transition.terminated and not transition.truncated


def test_stage13_severe_failure_cleans_cargo_tasks_and_resources():
    for reason in sorted(PersistentDispatchEnvironment.SEVERE_FAILURE_REASONS):
        env = _small_persistent_env()
        action = next(item for item in env.enumerate_candidate_actions()
                      if item.transport_mode == env.CAR_DOG_CAR)
        task = env._task(action.task_id).task
        env._claim_resources(action, task)
        participants = (action.pickup_carter, action.dog_id,
                        action.receiving_carter)
        for robot_id in participants:
            env.robots[robot_id].robot.task_id = task.task_id
        env.robots[action.pickup_carter].robot.cargo_id = task.cargo_id

        transition = env.abort_episode(reason, task.task_id, elapsed=2.5)

        assert transition.terminated and not transition.truncated
        assert transition.termination_reason == reason
        assert transition.reward_components["failure"] == \
            -env.reward_config.severe_failure_penalty
        assert transition.reward == sum(transition.reward_components.values())
        assert not env.resource_claims
        assert all(stair["state"] == "FREE" for stair in env.stairs.values())
        assert all(not runtime.robot.task_id and not runtime.robot.cargo_id
                   for runtime in env.robots.values())
        assert env.failed == 1 and not env.queue.waiting
        assert (sum(event["event"] == "CLAIM" for event in env.resource_events)
                == sum(event["event"] == "RELEASE"
                       for event in env.resource_events))


def test_stage13_no_feasible_chain_has_explicit_terminal_transition():
    env = _small_persistent_env()
    for runtime in env.robots.values():
        runtime.robot.failure_code = "FAULT"
    assert not env.enumerate_candidate_actions()
    transition = env.abort_episode("NO_FEASIBLE_CHAIN", elapsed=1)
    assert transition.action["type"] == "ABORT"
    assert transition.termination_reason == "NO_FEASIBLE_CHAIN"
    assert transition.next_state["failed_count"] == 1


def test_stage14_balanced_curriculum_is_exactly_half_cross_floor():
    arrivals = build_balanced_arrivals(20260835)
    types = Counter(task_type for _, task_type in arrivals)
    assert types == {
        "f1_same": 3, "f1_cross_region": 2,
        "f2_same": 3, "f2_cross_region": 2,
        "cross_up": 5, "cross_down": 5}
    assert sum(task_type.startswith("cross_")
               for _, task_type in arrivals) == 10


def test_stage15_cross_heavy_curriculum_is_balanced_and_fixed_shape():
    spec = Stage14EpisodeSpec(
        f1_same=1, f1_cross_region=1,
        f2_same=1, f2_cross_region=1,
        cross_up=8, cross_down=8)
    arrivals = build_balanced_arrivals(
        20260840, spec=spec, arrival_interval=(1.0, 3.0))
    types = Counter(task_type for _, task_type in arrivals)
    assert len(arrivals) == 20
    assert types["cross_up"] == types["cross_down"] == 8
    env = WarehouseDispatchGymEnv(
        seed=20260840, execution_mode="CONCURRENT",
        arrival_interval=(1.0, 3.0), episode_spec=spec)
    observation, info = env.reset(seed=20260840)
    assert observation["state"].shape == (284,)
    assert observation["action_features"].shape == (
        1536, Stage12Encoder.ACTION_WIDTH)
    assert info["action_mask"].any()


def test_stage15_mixed_load_schedule_is_one_continuous_unique_episode():
    arrivals = build_mixed_load_arrivals(20266001)
    assert len(arrivals) == 60
    assert len({item.task.task_id for item, _ in arrivals}) == 60
    assert len({item.task.cargo_id for item, _ in arrivals}) == 60
    assert [task_type.split(":", 1)[0] for _, task_type in arrivals] == (
        ["NORMAL"] * 20 + ["DENSE"] * 20 + ["BURST"] * 20)
    times = [item.arrival_time for item, _ in arrivals]
    assert times == sorted(times)
    assert times[20] > times[19]
    assert times[40] > times[39]


def test_stage15_mixed_load_environment_preserves_fixed_tensor_contract():
    env = WarehouseDispatchGymEnv(
        seed=20266001, execution_mode="CONCURRENT",
        handover_sampling="TASK_KEYED", arrival_schedule="MIXED_LOAD")
    observation, info = env.reset(seed=20266001)
    assert env.dispatch.task_limit == 60
    assert observation["state"].shape == (284,)
    assert observation["action_features"].shape == (
        1536, Stage12Encoder.ACTION_WIDTH)
    assert info["pending_arrivals"] == 59
    assert observation["state"][2] == .4


def test_stage19_mixed_recovery_schedule_is_one_continuous_unique_episode():
    arrivals = build_mixed_recovery_arrivals(20268001)
    assert len(arrivals) == 80
    assert len({item.task.task_id for item, _ in arrivals}) == 80
    assert len({item.task.cargo_id for item, _ in arrivals}) == 80
    assert [task_type.split(":", 1)[0] for _, task_type in arrivals] == (
        ["NORMAL"] * 20 + ["DENSE"] * 20 + ["BURST"] * 20 +
        ["RECOVERY"] * 20)
    times = [item.arrival_time for item, _ in arrivals]
    assert times == sorted(times)
    assert all(times[index] > times[index - 1] for index in (20, 40, 60))


def test_stage19_mixed_recovery_environment_preserves_tensor_contract():
    env = WarehouseDispatchGymEnv(
        seed=20268001, execution_mode="CONCURRENT",
        handover_sampling="TASK_KEYED",
        arrival_schedule="MIXED_LOAD_RECOVERY")
    observation, info = env.reset(seed=20268001)
    assert env.dispatch.task_limit == 80
    assert env.dispatch.time_limit == 5400
    assert observation["state"].shape == (284,)
    assert observation["action_features"].shape == (
        1536, Stage12Encoder.ACTION_WIDTH)
    assert info["pending_arrivals"] == 79
    assert observation["state"][2] == .4


def test_stage20_mixed_curriculum_randomizes_order_length_and_gaps():
    first, metadata = build_mixed_curriculum_arrivals(
        20269001, return_metadata=True)
    repeated, repeated_metadata = build_mixed_curriculum_arrivals(
        20269001, return_metadata=True)
    second, second_metadata = build_mixed_curriculum_arrivals(
        20269002, return_metadata=True)
    assert metadata == repeated_metadata
    assert [(item.arrival_time, kind) for item, kind in first] == [
        (item.arrival_time, kind) for item, kind in repeated]
    assert len(first) == len(second) == 80
    assert sum(metadata["phase_lengths"]) == 80
    assert all(12 <= value <= 28 and value % 4 == 0
               for value in metadata["phase_lengths"])
    assert sorted(metadata["phase_order"]) == [
        "BURST", "DENSE", "NORMAL", "RECOVERY"]
    assert (metadata["phase_order"].index("BURST") <
            metadata["phase_order"].index("RECOVERY"))
    assert (metadata["phase_order"] != second_metadata["phase_order"] or
            metadata["phase_lengths"] != second_metadata["phase_lengths"] or
            [row["switch_gap_s"] for row in metadata["phase_rows"]] !=
            [row["switch_gap_s"] for row in second_metadata["phase_rows"]])
    assert len({item.task.task_id for item, _ in first}) == 80
    assert len({item.task.cargo_id for item, _ in first}) == 80
    assert all(first[index][0].arrival_time < first[index + 1][0].arrival_time
               for index in range(79))
    counts = Counter(kind.split(":", 1)[0] for _, kind in first)
    assert counts == dict(zip(metadata["phase_order"], metadata["phase_lengths"]))


def test_stage20_mixed_curriculum_environment_preserves_tensor_contract():
    env = WarehouseDispatchGymEnv(
        seed=20269001, execution_mode="CONCURRENT",
        handover_sampling="TASK_KEYED", arrival_schedule="MIXED_CURRICULUM")
    observation, info = env.reset(seed=20269001)
    assert env.dispatch.task_limit == 80
    assert env.dispatch.time_limit == 6000
    assert info["arrival_schedule"] == "MIXED_CURRICULUM"
    metadata = info["arrival_schedule_metadata"]
    assert metadata["state_reset_between_phases"] is False
    assert metadata["total_tasks"] == 80
    assert observation["state"].shape == (284,)
    assert observation["action_features"].shape == (
        1536, Stage12Encoder.ACTION_WIDTH)
    assert info["pending_arrivals"] == 79


def _run_stage14_rule_episode(seed):
    env = WarehouseDispatchGymEnv(seed=seed)
    observation, info = env.reset(seed=seed)
    rows = []
    terminated = truncated = False
    while not (terminated or truncated):
        assert env.observation_space.contains(observation)
        action = env.rule_action()
        assert info["action_mask"][action]
        observation, reward, terminated, truncated, info = env.step(action)
        selected = info["selected_action"]
        rows.append((
            selected.get("type", "ASSIGN"), selected.get("task_id", ""),
            selected.get("transport_mode", ""), selected.get("dog_id", ""),
            selected.get("stair_id", ""), round(reward, 9)))
    return env, rows


def test_stage14_gym_rule_baseline_completes_balanced_episode():
    env, rows = _run_stage14_rule_episode(20260835)
    dispatch = env.dispatch
    assert dispatch is not None
    assignments = [row for row in rows if row[0] == "ASSIGN"]
    waits = [row for row in rows if row[0] == "WAIT"]
    task_types = Counter(env.task_types[row[1]] for row in assignments)
    modes = Counter(row[2] for row in assignments)
    assert dispatch.completed == 20 and dispatch.failed == 0
    assert len(assignments) == 20 and waits
    assert sum(name.startswith("cross_") * count
               for name, count in task_types.items()) == 10
    assert modes == {"SINGLE_CAR": 10, "CAR_DOG_CAR": 7,
                     "SINGLE_DOG": 3}
    assert not dispatch.resource_claims
    assert all(not runtime.robot.task_id and not runtime.robot.cargo_id
               for runtime in dispatch.robots.values())
    assert all(stair["state"] == "FREE" for stair in dispatch.stairs.values())


def test_stage14_rule_replay_is_deterministic_through_gym_api():
    first_env, first = _run_stage14_rule_episode(20260836)
    second_env, second = _run_stage14_rule_episode(20260836)
    assert first == second
    assert first_env.dispatch is not None and second_env.dispatch is not None
    assert first_env.dispatch.now == second_env.dispatch.now


def test_stage14_wait_mask_aborts_unrecoverable_no_candidate_state():
    env = WarehouseDispatchGymEnv()
    env.reset(seed=20260837)
    assert env.dispatch is not None
    for runtime in env.dispatch.robots.values():
        runtime.robot.failure_code = "FAULT"
    env.decision = env.encoder.encode(env.dispatch)
    assert env.action_masks().sum() == 1
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and reward < 0
    assert info["termination_reason"] == "NO_FEASIBLE_CHAIN"


def test_stage14_candidate_scorer_masks_and_has_no_index_parameters():
    env = WarehouseDispatchGymEnv()
    observation, info = env.reset(seed=20260838)
    model = MaskedCandidateActorCritic(hidden_width=32)
    state = torch.as_tensor(observation["state"]).float()
    features = torch.as_tensor(observation["action_features"]).float()
    mask = torch.as_tensor(info["action_mask"]).bool()
    logits, value = model(state, features, mask)
    assert logits.shape == (1536,) and value.ndim == 0
    assert mask[model.select_action(observation, info["action_mask"])]
    assert torch.all(logits[~mask] == torch.finfo(logits.dtype).min)

    first, second = torch.nonzero(mask, as_tuple=False).flatten()[:2]
    swapped = features.clone()
    swapped[first], swapped[second] = (
        features[second].clone(), features[first].clone())
    swapped_logits, _ = model(state, swapped, mask)
    assert torch.allclose(swapped_logits[first], logits[second])
    assert torch.allclose(swapped_logits[second], logits[first])
    assert not model.metadata()["index_specific_parameters"]
    assert model.metadata()["architecture"] == \
        "contextual_masked_candidate_actor_critic_v2"
    assert model.metadata()["action_width"] == Stage12Encoder.ACTION_WIDTH


def test_stage14_evaluation_reports_context_conditioned_mode_choices():
    model = MaskedCandidateActorCritic(hidden_width=32)
    result = evaluate(
        model, [20260839], "CONCURRENT", "MEDIUM",
        torch.device("cpu"))
    assert result["cost_reference_decision_count"] == result["task_count"]
    assert result["selected_mode_by_cost_reference"]
    assert set(result["selected_mode_by_floor_relation"]) == {
        "SAME_FLOOR", "CROSS_FLOOR"}


def _two_independent_task_environment(concurrent: bool):
    points = {
        "A": Point("A", 1, "R_A", (0, 0, .45)),
        "B": Point("B", 1, "R_B", (8, 0, .45)),
        "C": Point("C", 1, "R_C", (0, 8, .45)),
        "D": Point("D", 1, "R_D", (8, 8, .45)),
    }
    tasks = [
        QueuedTask(Task("TA", "CA", points["A"], points["B"], 2, 100), 0),
        QueuedTask(Task("TB", "CB", points["C"], points["D"], 1, 100), 0),
    ]
    runtimes = [
        RobotRuntime(robot("car_a", "car", 1, "R_A"), points["A"].xyz),
        RobotRuntime(robot("car_b", "car", 1, "R_C"), points["C"].xyz),
    ]
    slots = [
        StandbySlot("SA", "car", 1, (9, 0, .45)),
        StandbySlot("SB", "car", 1, (9, 8, .45)),
    ]
    cls = (ConcurrentPersistentDispatchEnvironment if concurrent
           else PersistentDispatchEnvironment)
    return cls(runtimes, {}, AsyncTaskQueue(tasks), task_limit=2,
               time_limit=100, decision_gap=.1, standby_slots=slots)


def _run_two_task_environment(env):
    peak = 0
    while env.completed + env.failed < env.task_limit:
        candidates = env.enumerate_candidate_actions()
        legal = env.build_action_mask(candidates)
        if any(legal):
            env.step(env.select_rule_action(candidates))
        else:
            env.wait_for_next_event()
        peak = max(peak, len(getattr(env, "active_tasks", {})))
    return peak


def test_concurrent_mode_overlaps_independent_tasks_and_cleans_resources():
    serial = _two_independent_task_environment(False)
    concurrent = _two_independent_task_environment(True)
    assert _run_two_task_environment(serial) == 0
    assert _run_two_task_environment(concurrent) == 2
    assert concurrent.now < serial.now
    assert concurrent.completed == serial.completed == 2
    assert not concurrent.resource_claims
    assert not concurrent.active_standby_claims
    assert all(not runtime.robot.task_id and not runtime.robot.cargo_id
               for runtime in concurrent.robots.values())


def test_mode_conditioned_observation_and_mixed_episode_selection():
    serial = WarehouseDispatchGymEnv(seed=20260834, execution_mode="MIXED")
    serial_observation, serial_info = serial.reset(seed=20260834)
    concurrent = WarehouseDispatchGymEnv(seed=20260835, execution_mode="MIXED")
    concurrent_observation, concurrent_info = concurrent.reset(seed=20260835)
    assert serial_info["execution_mode"] == "SERIAL"
    assert concurrent_info["execution_mode"] == "CONCURRENT"
    assert serial_observation["state"].shape == (284,)
    assert concurrent_observation["state"].shape == (284,)
    assert serial_observation["state"][4] == 0
    assert concurrent_observation["state"][4] == 1


def test_smdp_gae_uses_per_transition_discount():
    advantages, returns = compute_smdp_gae(
        rewards=[1.0, 1.0], values=[0.0, 0.0],
        discounts=[0.9, 0.5], episode_ends=[False, True],
        gae_lambda=1.0)
    assert np.allclose(advantages, [1.9, 1.0])
    assert np.allclose(returns, advantages)


def test_smdp_gae_does_not_bootstrap_across_episode_boundary():
    advantages, _ = compute_smdp_gae(
        rewards=[1.0, 20.0], values=[0.0, 0.0],
        discounts=[0.99, 0.99], episode_ends=[True, True],
        gae_lambda=1.0)
    assert np.allclose(advantages, [1.0, 20.0])


def test_realistic_concurrent_course_exposes_three_transport_modes_together():
    env = WarehouseDispatchGymEnv(
        seed=20260835, execution_mode="CONCURRENT")
    _, info = env.reset(seed=20260835)
    exposed = set()
    done = False
    while not done:
        exposed.update(
            candidate.transport_mode
            for candidate, legal in zip(
                env.decision.action_ids, info["action_mask"])
            if legal and isinstance(candidate, Assignment))
        action = env.rule_action()
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    assert exposed == {"SINGLE_CAR", "SINGLE_DOG", "CAR_DOG_CAR"}


def test_stage14_reward_v2_has_no_transport_mode_bonus():
    assert STAGE14_REWARD_CONFIG.outstanding_time_scale == 30.0
    assert STAGE14_REWARD_CONFIG.timeout_penalty == 2.0
    assert STAGE14_REWARD_CONFIG.active_robot_time_penalty == .0002
    assert not any("mode" in name or "dog" in name or "car" in name
                   for name in STAGE14_REWARD_CONFIG.__dataclass_fields__)


def test_running_return_normalizer_round_trips_checkpoint_state():
    steps = [
        RolloutStep(np.zeros(1), np.zeros((1, 1)), np.ones(1, dtype=bool),
                    0, 0.0, 0.0, 2.0, .9, False),
        RolloutStep(np.zeros(1), np.zeros((1, 1)), np.ones(1, dtype=bool),
                    0, 0.0, 0.0, -1.0, .8, True),
    ]
    normalizer = RunningReturnNormalizer()
    normalized = normalizer.normalize_rollout(steps, update=True)
    restored = RunningReturnNormalizer.from_state_dict(
        normalizer.state_dict())
    assert np.all(np.isfinite(normalized))
    assert normalizer.count > 2 and normalizer.variance > 0
    assert np.allclose(
        normalized, restored.normalize_rollout(steps, update=False))
