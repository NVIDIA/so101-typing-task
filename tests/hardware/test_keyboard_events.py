from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from so101_typing.hardware.keyboard_events import (
    CapturedSession,
    KeyEvent,
    _normalize_key_name,
    levenshtein,
    score_typed,
)


def test_score_exact_phrase() -> None:
    score = score_typed("newton is the best", "newton is the best")

    assert score["exact_match"] is True
    assert score["position_accuracy"] == 1.0
    assert score["levenshtein"] == 0


def test_score_empty_capture_is_failure() -> None:
    score = score_typed("", "newton is the best")

    assert score["exact_match"] is False
    assert score["correct_chars"] == 0
    assert score["levenshtein"] == 18


def test_levenshtein() -> None:
    assert levenshtein("newton is best", "newton is the best") == 4


def test_captured_session_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "typed.json"
    session = CapturedSession(
        events=[
            KeyEvent(char="n", code=49, key_name="KEY_N", t_press=1.0, t_release=1.1),
            KeyEvent(char="e", code=18, key_name="KEY_E", t_press=1.2, t_release=1.3),
        ],
        device_path="/dev/input/event99",
        device_name="test keyboard",
        stop_reason="stop_key:KEY_ESC",
        t_start_monotonic=10.0,
        t_end_monotonic=11.0,
    )

    session.write(path)
    loaded = CapturedSession.load(path)

    assert loaded.typed_string() == "ne"
    assert loaded.device_path == "/dev/input/event99"
    assert loaded.device_name == "test keyboard"
    assert loaded.stop_reason == "stop_key:KEY_ESC"


def test_normalize_stop_key_name() -> None:
    assert _normalize_key_name("esc") == "KEY_ESC"
    assert _normalize_key_name("KEY_F12") == "KEY_F12"
