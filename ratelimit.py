"""Sliding-window attempt limiter.

WHY: /chat calls a paid model on every request and is unauthenticated, so an
open endpoint is a direct spend exposure. This bounds requests per source
address without requiring anyone to sign in, which keeps the public demo usable.

SCOPE, stated precisely rather than optimistically:

  * In-memory and single-process. State resets on restart and it does not
    coordinate across workers. Correct for the single-process demo this serves;
    it is NOT a general rate limiter, and a multi-worker deployment needs Redis.
  * Keyed on the address the WSGI server reports. Behind a proxy that is the
    proxy unless the app is configured to trust forwarded headers, in which case
    every client collapses into one bucket. This app runs directly, so
    `remote_addr` is the client.
  * Distributed request floods across many source addresses are not limited.

Reads do not create entries, and fully expired keys are swept at most once per
window, so an unauthenticated caller cannot grow the dict without bound within
a window. Peak is still (arrival rate x window); there is no hard key cap.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._last_sweep = 0.0
        self._lock = threading.Lock()

    def _sweep_locked(self, now: float) -> None:
        if now - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = now
        cutoff = now - self.window_seconds
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] < cutoff]:
            del self._hits[key]

    def _prune_locked(self, key: str, now: float) -> int:
        hits = self._hits.get(key)
        if hits is None:
            return 0
        cutoff = now - self.window_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()
        if not hits:
            del self._hits[key]
            return 0
        return len(hits)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Record a request and return whether it is within the limit.

        Counts every request, not only rejected ones — this limits usage and
        cost, it is not an authentication failure counter.
        """
        if self.max_requests <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            self._sweep_locked(now)
            count = self._prune_locked(key, now)
            if count >= self.max_requests:
                return False
            self._hits.setdefault(key, deque()).append(now)
            return True

    def size(self) -> int:
        with self._lock:
            return len(self._hits)
