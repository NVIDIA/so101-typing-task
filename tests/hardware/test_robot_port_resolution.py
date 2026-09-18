from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from so101_typing.hardware import robot


def test_resolve_robot_port_requires_explicit_port_when_multiple_controllers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports_by_pattern = {
        "/dev/serial/by-id/*": [
            "/dev/serial/by-id/usb-SO101_TEST_A-if00",
            "/dev/serial/by-id/usb-SO101_TEST_B-if00",
        ],
        "/dev/ttyACM*": ["/dev/ttyACM0", "/dev/ttyACM1"],
        "/dev/ttyUSB*": [],
    }
    real_paths = {
        "/dev/serial/by-id/usb-SO101_TEST_A-if00": "/dev/ttyACM0",
        "/dev/serial/by-id/usb-SO101_TEST_B-if00": "/dev/ttyACM1",
    }

    monkeypatch.setattr(robot.glob, "glob", lambda pattern: ports_by_pattern[pattern])
    monkeypatch.setattr(robot.os.path, "realpath", lambda path: real_paths.get(path, path))

    with pytest.raises(FileNotFoundError, match="multiple robot serial ports"):
        robot.resolve_robot_port()


def test_resolve_robot_port_returns_only_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    port = "/dev/serial/by-id/usb-SO101_TEST_B-if00"
    ports_by_pattern = {
        "/dev/serial/by-id/*": [port],
        "/dev/ttyACM*": ["/dev/ttyACM1"],
        "/dev/ttyUSB*": [],
    }

    monkeypatch.setattr(robot.glob, "glob", lambda pattern: ports_by_pattern[pattern])
    monkeypatch.setattr(robot.os.path, "realpath", lambda path: "/dev/ttyACM1" if path == port else path)

    assert robot.resolve_robot_port() == port
