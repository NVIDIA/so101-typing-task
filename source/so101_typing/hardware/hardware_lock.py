"""Process-level locks for SO-101 hardware access."""

from __future__ import annotations

import fcntl
import time
from pathlib import Path

ROBOT_BUS_LOCK_PATH = Path("/tmp/so101_homing_robot_bus.lock")


class RobotBusLock:
    """Exclusive lock around the shared SO-101 serial bus."""

    def __init__(self, path: Path = ROBOT_BUS_LOCK_PATH, timeout_s: float = 0.0):
        self.path = path
        self.timeout_s = timeout_s
        self._file = None

    def acquire(self) -> "RobotBusLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("w")
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_file.write(f"{time.time():.6f}\n")
                lock_file.flush()
                self._file = lock_file
                return self
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    lock_file.close()
                    raise RuntimeError(
                        f"SO-101 serial bus is already in use. Wait for the other robot command to finish. "
                        f"Lock file: {self.path}"
                    ) from exc
                time.sleep(0.05)

    def release(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "RobotBusLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
