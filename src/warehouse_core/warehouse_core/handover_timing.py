"""Auditable handover-duration calibration and timeout policy.

The current workspace does not yet contain timestamped RViz or hardware
handover observations.  The provisional calibration below is therefore a
reproducible logic-simulation model, not a claim about physical performance.
Its stable-alignment floor (1.5 s) and historical nominal handover time
(2.0 s) come from the existing system configuration.  A later measured
dataset can replace the samples without changing the timeout semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from math import ceil
import random
from typing import Iterable


CAR_TO_DOG = "CAR_TO_DOG"
DOG_TO_CAR = "DOG_TO_CAR"
HANDOVER_KINDS = (CAR_TO_DOG, DOG_TO_CAR)
CALIBRATION_SOURCE = "logic_simulation_provisional_v1"
CALIBRATION_SEED = 20260809
CALIBRATION_SAMPLES_PER_KIND = 500
DEFAULT_PERCENTILE = 0.95


@dataclass(frozen=True)
class HandoverTimingSample:
    sample_id: str
    handover_kind: str
    stable_alignment_s: float
    transfer_verify_s: float
    coordination_delay_s: float
    command_jitter_s: float
    duration_s: float
    source: str = CALIBRATION_SOURCE

    def to_dict(self) -> dict:
        return asdict(self)


def nearest_rank_percentile(values: Iterable[float], percentile: float) -> float:
    """Return an auditable nearest-rank percentile.

    The P95 threshold is the value at rank ``ceil(0.95 * n)``.  Timeout uses
    a strict greater-than comparison, so a sample equal to the threshold is
    accepted.
    """
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one timing sample is required")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    if any(value < 0 for value in ordered):
        raise ValueError("handover duration must be non-negative")
    rank = max(1, ceil(percentile * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class HandoverTimeoutPolicy:
    percentile: float
    timeout_s: float
    sample_count: int
    source: str
    comparison: str = "STRICT_GREATER_THAN"
    failure_code: str = "HANDOVER_TIMEOUT"

    @classmethod
    def from_samples(cls, samples: Iterable[HandoverTimingSample],
                     percentile: float = DEFAULT_PERCENTILE):
        samples = tuple(samples)
        kinds = {sample.handover_kind for sample in samples}
        if not kinds.issubset(HANDOVER_KINDS):
            raise ValueError(f"unsupported handover kinds: {sorted(kinds)}")
        sources = {sample.source for sample in samples}
        if len(sources) != 1:
            raise ValueError("calibration samples must have one declared source")
        return cls(
            percentile=percentile,
            timeout_s=nearest_rank_percentile(
                (sample.duration_s for sample in samples), percentile),
            sample_count=len(samples),
            source=next(iter(sources)),
        )

    def timed_out(self, elapsed_s: float) -> bool:
        if elapsed_s < 0:
            raise ValueError("handover elapsed time must be non-negative")
        return elapsed_s > self.timeout_s


def build_provisional_calibration(
        seed: int = CALIBRATION_SEED,
        samples_per_kind: int = CALIBRATION_SAMPLES_PER_KIND,
) -> tuple[HandoverTimingSample, ...]:
    """Build deterministic provisional samples for the pre-RL logic gate.

    Both kinds share the configured 1.5 s alignment floor.  Transfer and
    verification take 0.45 s for car-to-dog and 0.55 s for dog-to-car.  The
    remaining positive delays represent coordination and command/observation
    timing in the logical simulator.  These assumptions are saved per row so
    they cannot be mistaken for measurements.
    """
    if samples_per_kind <= 0:
        raise ValueError("samples_per_kind must be positive")
    rng = random.Random(seed)
    samples = []
    for kind in HANDOVER_KINDS:
        transfer_verify = .45 if kind == CAR_TO_DOG else .55
        for index in range(1, samples_per_kind + 1):
            stable = 1.5
            coordination = rng.lognormvariate(-1.5, .55)
            jitter = rng.uniform(.05, .15)
            duration = stable + transfer_verify + coordination + jitter
            samples.append(HandoverTimingSample(
                sample_id=f"{kind}_{index:04d}",
                handover_kind=kind,
                stable_alignment_s=stable,
                transfer_verify_s=transfer_verify,
                coordination_delay_s=coordination,
                command_jitter_s=jitter,
                duration_s=duration,
            ))
    return tuple(samples)


class EmpiricalHandoverDurationProvider:
    """Seeded bootstrap sampler used by the logical training environment."""

    def __init__(self, samples: Iterable[HandoverTimingSample], seed: int):
        grouped = {kind: [] for kind in HANDOVER_KINDS}
        for sample in samples:
            if sample.handover_kind not in grouped:
                raise ValueError(
                    f"unsupported handover kind: {sample.handover_kind}")
            grouped[sample.handover_kind].append(sample.duration_s)
        if any(not values for values in grouped.values()):
            raise ValueError("both current handover kinds require samples")
        self._grouped = grouped
        all_values = [value for values in grouped.values() for value in values]
        self.mean_duration_s = sum(all_values) / len(all_values)
        self._rng = random.Random(seed)

    def __call__(self, handover_kind: str,
                 task_id: str | None = None) -> float:
        try:
            values = self._grouped[handover_kind]
        except KeyError as exc:
            raise ValueError(
                f"unsupported handover kind: {handover_kind}") from exc
        return self._rng.choice(values)


class KeyedEmpiricalHandoverDurationProvider:
    """Bind a reproducible sample to (seed, task, handover kind).

    This is used by paired policy comparisons so policies that perform a
    different number of handovers do not shift each other's random stream.
    """

    def __init__(self, samples: Iterable[HandoverTimingSample], seed: int):
        grouped = {kind: [] for kind in HANDOVER_KINDS}
        for sample in samples:
            if sample.handover_kind not in grouped:
                raise ValueError(
                    f"unsupported handover kind: {sample.handover_kind}")
            grouped[sample.handover_kind].append(sample.duration_s)
        if any(not values for values in grouped.values()):
            raise ValueError("both current handover kinds require samples")
        self._grouped = grouped
        self._seed = seed
        all_values = [value for values in grouped.values() for value in values]
        self.mean_duration_s = sum(all_values) / len(all_values)

    def __call__(self, handover_kind: str,
                 task_id: str | None = None) -> float:
        if not task_id:
            raise ValueError("task-keyed handover sampling requires task_id")
        try:
            values = self._grouped[handover_kind]
        except KeyError as exc:
            raise ValueError(
                f"unsupported handover kind: {handover_kind}") from exc
        digest = hashlib.sha256(
            f"{self._seed}:{task_id}:{handover_kind}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(values)
        return values[index]


def build_provisional_policy_and_provider(seed: int):
    samples = build_provisional_calibration()
    policy = HandoverTimeoutPolicy.from_samples(samples)
    provider = EmpiricalHandoverDurationProvider(samples, seed)
    return policy, provider


def build_provisional_policy_and_keyed_provider(seed: int):
    samples = build_provisional_calibration()
    policy = HandoverTimeoutPolicy.from_samples(samples)
    provider = KeyedEmpiricalHandoverDurationProvider(samples, seed)
    return policy, provider
