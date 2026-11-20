"""M6.1 — in-memory session registry for the driver-feedback gateway.

Owns the lifecycle of ``DriverSession`` objects created through the gateway:

* unique session ids,
* expiry (TTL) + cleanup,
* feature-set isolation (a session's feature set is fixed at creation),
* safe handling of invalid/expired ids,
* a per-session lock so concurrent requests to the same session are never
  interleaved (the ``DriverSession`` type itself is not thread-safe).

No persistence, no database, no network — sessions live only in this process.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.agent.driver_session import DriverSession


class SessionRegistryError(RuntimeError):
    """Base error for session-registry failures."""


class SessionNotFoundError(SessionRegistryError):
    """Raised when a session id is unknown or has been cleaned up."""


class SessionExpiredError(SessionNotFoundError):
    """Raised when a session id exists but its TTL has elapsed."""


@dataclass
class SessionEntry:
    """One live session plus its per-session synchronization and bookkeeping."""

    session: DriverSession
    lock: threading.RLock = field(default_factory=threading.RLock)
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_accessed = time.monotonic()


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


class SessionRegistry:
    """Thread-safe in-memory store of active driver sessions."""

    def __init__(self, ttl_seconds: float = 1800.0) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self._entries: dict[str, SessionEntry] = {}
        self._guard = threading.RLock()  # guards the dict itself
        self._closed = False

    # -- creation -------------------------------------------------------
    def create(self, feature_set: str = "full_operational") -> str:
        """Create a fresh session pinned to ``feature_set`` and return its id."""
        session_id = _new_session_id()
        session = DriverSession(feature_set=feature_set, session_id=session_id)
        with self._guard:
            self._entries[session_id] = SessionEntry(session=session)
        return session_id

    # -- lookup ---------------------------------------------------------
    def _get_entry_locked(self, session_id: str) -> SessionEntry:
        entry = self._entries.get(session_id)
        if entry is None:
            raise SessionNotFoundError(f"unknown session id: {session_id!r}")
        if self._is_expired(entry):
            raise SessionExpiredError(f"session expired: {session_id!r}")
        return entry

    def get(self, session_id: str) -> SessionEntry:
        """Return the live entry for ``session_id`` (raises on missing/expired)."""
        with self._guard:
            entry = self._get_entry_locked(session_id)
            entry.touch()
            return entry

    def get_session(self, session_id: str) -> DriverSession:
        return self.get(session_id).session

    def feature_set_of(self, session_id: str) -> str:
        return self.get_session(session_id).feature_set

    # -- lifecycle ------------------------------------------------------
    def _is_expired(self, entry: SessionEntry) -> bool:
        return (time.monotonic() - entry.last_accessed) > self.ttl_seconds

    def cleanup(self, *, force: bool = False) -> int:
        """Remove expired sessions; returns the number removed."""
        removed = 0
        with self._guard:
            expired = [
                sid for sid, e in self._entries.items()
                if force or self._is_expired(e)
            ]
            for sid in expired:
                self._entries.pop(sid, None)
                removed += 1
        return removed

    def is_live(self, session_id: str) -> bool:
        try:
            self._get_entry_locked(session_id)
            return True
        except SessionNotFoundError:
            return False

    def active_count(self) -> int:
        with self._guard:
            return len(self._entries)

    def close(self) -> None:
        """Drop all sessions (test/teardown helper)."""
        with self._guard:
            self._entries.clear()
            self._closed = True

    # -- background sweeper (optional) ---------------------------------
    def start_sweeper(self, interval_seconds: float = 60.0) -> threading.Thread:
        thread = threading.Thread(
            target=self._sweep_loop,
            args=(interval_seconds,),
            daemon=True,
            name="session-registry-sweeper",
        )
        thread.start()
        return thread

    def _sweep_loop(self, interval_seconds: float) -> None:
        while not self._closed:
            try:
                self.cleanup()
            except Exception:  # pragma: no cover - defensive
                pass
            time.sleep(interval_seconds)
