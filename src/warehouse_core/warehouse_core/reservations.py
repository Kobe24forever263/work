from dataclasses import dataclass
from time import monotonic
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True)
class Lease:
    reservation_id: str
    resource_id: str
    requester_id: str
    task_id: str
    expires_at: float


class ReservationBook:
    """Exclusive leases with deterministic expiry and owner-checked release."""

    def __init__(self, clock=monotonic):
        self._clock = clock
        self._leases: dict[str, Lease] = {}

    def _purge(self) -> None:
        now = self._clock()
        self._leases = {key: lease for key, lease in self._leases.items()
                        if lease.expires_at > now}

    def reserve(self, resource_id: str, requester_id: str, task_id: str,
                lease_seconds: float) -> Optional[Lease]:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._purge()
        existing = self._leases.get(resource_id)
        if existing and (existing.requester_id, existing.task_id) != (requester_id, task_id):
            return None
        lease = Lease(existing.reservation_id if existing else str(uuid4()),
                      resource_id, requester_id, task_id,
                      self._clock() + lease_seconds)
        self._leases[resource_id] = lease
        return lease

    def release(self, reservation_id: str, requester_id: str) -> bool:
        self._purge()
        for resource_id, lease in tuple(self._leases.items()):
            if lease.reservation_id == reservation_id and lease.requester_id == requester_id:
                del self._leases[resource_id]
                return True
        return False

    def release_task(self, task_id: str) -> int:
        targets = [key for key, lease in self._leases.items() if lease.task_id == task_id]
        for key in targets:
            del self._leases[key]
        return len(targets)
