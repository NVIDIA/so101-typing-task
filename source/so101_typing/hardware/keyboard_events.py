"""Physical USB keyboard event capture and scoring.

This replaces the old preflight logger behavior that silently converted
capture failures into empty typing sessions. Wrong device, missing evdev, and
permission errors should fail before a robot replay starts.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import select
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import evdev
    from evdev import ecodes

    _HAS_EVDEV = True
except (ImportError, OSError):
    evdev = None  # type: ignore[assignment]
    ecodes = None  # type: ignore[assignment]
    _HAS_EVDEV = False

KEY_NAME_TO_CHAR = {
    **{f"KEY_{chr(code)}": chr(code).lower() for code in range(ord("A"), ord("Z") + 1)},
    **{f"KEY_{digit}": digit for digit in range(0, 10)},
    "KEY_SPACE": " ",
    "KEY_MINUS": "-",
    "KEY_EQUAL": "=",
    "KEY_LEFTBRACE": "[",
    "KEY_RIGHTBRACE": "]",
    "KEY_SEMICOLON": ";",
    "KEY_APOSTROPHE": "'",
    "KEY_GRAVE": "`",
    "KEY_BACKSLASH": "\\",
    "KEY_COMMA": ",",
    "KEY_DOT": ".",
    "KEY_SLASH": "/",
    "KEY_ENTER": "\n",
    "KEY_BACKSPACE": "\b",
}


@dataclass(frozen=True)
class KeyboardDeviceInfo:
    path: str
    name: str
    aliases: list[str] = field(default_factory=list)
    phys: str | None = None
    readable: bool = True
    error: str | None = None


@dataclass
class KeyEvent:
    char: str
    code: int
    key_name: str
    t_press: float
    t_release: float | None = None


@dataclass
class CapturedSession:
    events: list[KeyEvent] = field(default_factory=list)
    device_path: str | None = None
    device_name: str | None = None
    stop_reason: str | None = None
    t_start_monotonic: float = 0.0
    t_end_monotonic: float = 0.0

    def typed_string(self) -> str:
        return "".join(event.char for event in self.events if event.char not in {"\b", "\n"})

    def wpm(self) -> float:
        if len(self.events) < 2:
            return 0.0
        elapsed_s = self.events[-1].t_press - self.events[0].t_press
        if elapsed_s <= 0.0:
            return 0.0
        return (len(self.typed_string()) / 5.0) / (elapsed_s / 60.0)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "device_path": self.device_path,
            "device_name": self.device_name,
            "stop_reason": self.stop_reason,
            "events": [asdict(event) for event in self.events],
            "typed": self.typed_string(),
            "wpm": self.wpm(),
            "t_start_monotonic": self.t_start_monotonic,
            "t_end_monotonic": self.t_end_monotonic,
        }

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_jsonable(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "CapturedSession":
        data = json.loads(Path(path).read_text())
        return cls(
            events=[KeyEvent(**event) for event in data.get("events", [])],
            device_path=data.get("device_path"),
            device_name=data.get("device_name"),
            stop_reason=data.get("stop_reason"),
            t_start_monotonic=float(data.get("t_start_monotonic", 0.0)),
            t_end_monotonic=float(data.get("t_end_monotonic", 0.0)),
        )


def _require_evdev() -> None:
    if not _HAS_EVDEV:
        raise RuntimeError(
            "Python package 'evdev' is not available; install it in the environment used for hardware tests."
        )


def _keycode_to_char() -> dict[int, tuple[str, str]]:
    _require_evdev()
    out: dict[int, tuple[str, str]] = {}
    for key_name, char in KEY_NAME_TO_CHAR.items():
        if hasattr(ecodes, key_name):
            out[getattr(ecodes, key_name)] = (key_name, char)
    return out


def _normalize_key_name(key_name: str) -> str:
    normalized = key_name.strip().upper().replace("-", "_")
    if not normalized:
        raise ValueError("stop key name must not be empty")
    if not normalized.startswith("KEY_"):
        normalized = f"KEY_{normalized}"
    return normalized


def _key_name_to_code(key_name: str) -> tuple[str, int]:
    _require_evdev()
    normalized = _normalize_key_name(key_name)
    if not hasattr(ecodes, normalized):
        raise ValueError(f"unknown evdev key name: {key_name!r}")
    return normalized, int(getattr(ecodes, normalized))


@contextlib.contextmanager
def _suppress_terminal_echo(enabled: bool = True):
    if not enabled or not sys.stdin.isatty():
        yield
        return

    import termios

    fd = sys.stdin.fileno()
    original_attrs = termios.tcgetattr(fd)
    muted_attrs = original_attrs[:]
    muted_attrs[3] = muted_attrs[3] & ~termios.ECHO
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, muted_attrs)
        yield
    finally:
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original_attrs)


def list_keyboard_devices() -> list[KeyboardDeviceInfo]:
    """List Linux input event devices, including unreadable ones.

    `evdev.list_devices()` only returns readable devices. That is exactly the
    failure mode we need to surface during hardware setup, so this function
    scans `/dev/input/event*` directly and annotates permission errors.
    """

    _require_evdev()
    aliases_by_target: dict[str, list[str]] = {}
    for alias_pattern in ("/dev/input/by-id/*", "/dev/input/by-path/*"):
        for alias in glob.glob(alias_pattern):
            try:
                target = os.path.realpath(alias)
            except OSError:
                continue
            aliases_by_target.setdefault(target, []).append(alias)

    devices: list[KeyboardDeviceInfo] = []
    for path in sorted(glob.glob("/dev/input/event*")):
        aliases = sorted(aliases_by_target.get(os.path.realpath(path), []))
        try:
            dev = evdev.InputDevice(path)
        except OSError as exc:
            devices.append(
                KeyboardDeviceInfo(path=path, name="<unreadable>", aliases=aliases, readable=False, error=str(exc))
            )
            continue
        try:
            devices.append(
                KeyboardDeviceInfo(path=dev.path, name=dev.name, aliases=aliases, phys=dev.phys, readable=True)
            )
        finally:
            dev.close()
    return devices


def _open_device(device_path: str):
    _require_evdev()
    if not os.path.exists(device_path):
        raise FileNotFoundError(f"input device does not exist: {device_path}")
    if not os.access(device_path, os.R_OK):
        raise PermissionError(
            f"input device is not readable: {device_path}. "
            "For a quick test run `sudo chmod o+r /dev/input/eventX`, or add your user to the input group."
        )
    try:
        return evdev.InputDevice(device_path)
    except PermissionError as exc:
        raise PermissionError(
            f"failed to open {device_path}: {exc}. "
            "For a quick test run `sudo chmod o+r /dev/input/eventX`, or add your user to the input group."
        ) from exc
    except OSError as exc:
        raise OSError(f"failed to open input device {device_path}: {exc}") from exc


def capture(
    device_path: str,
    duration_s: float,
    allowed_chars: set[str] | None = None,
    exclusive: bool = True,
    stop_after: str | None = None,
    stop_after_exact: str | None = None,
    stop_key: str | None = "KEY_ESC",
    stop_event: Any | None = None,
) -> CapturedSession:
    """Capture key-down/up events from a physical keyboard.

    Args:
        device_path: Linux `/dev/input/eventX` path.
        duration_s: Capture duration in seconds.
        allowed_chars: Optional printable character filter.
        exclusive: Grab the device so key presses do not leak into the shell.
        stop_after: Optional target string. Capture stops once this many
            accepted characters have been recorded.
        stop_after_exact: Optional target string. Capture stops only once the
            accepted text exactly matches this string.
        stop_key: Optional evdev key name that stops capture without recording
            that key. Examples: `esc`, `KEY_ESC`, `f12`, `KEY_F12`.
        stop_event: Optional threading.Event-like object. When set, capture
            exits and preserves the partial session.
    """

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    mapping = _keycode_to_char()
    stop_key_name: str | None = None
    stop_key_code: int | None = None
    if stop_key:
        stop_key_name, stop_key_code = _key_name_to_code(stop_key)

    dev = _open_device(device_path)
    session = CapturedSession(device_path=dev.path, device_name=dev.name)
    grabbed = False
    pending: dict[int, int] = {}
    try:
        if exclusive:
            try:
                dev.grab()
                grabbed = True
            except OSError as exc:
                raise OSError(f"failed to exclusively grab {device_path}; is another process using it? {exc}") from exc

        session.t_start_monotonic = time.monotonic()
        deadline = session.t_start_monotonic + duration_s
        while True:
            if stop_event is not None and stop_event.is_set():
                session.stop_reason = "external_stop"
                break
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                break
            readable, _, _ = select.select([dev.fd], [], [], min(0.05, remaining_s))
            if not readable:
                continue
            for event in dev.read():
                if event.type != ecodes.EV_KEY:
                    continue
                if event.value == 1 and stop_key_code is not None and event.code == stop_key_code:
                    session.stop_reason = f"stop_key:{stop_key_name}"
                    return session
                key_info = mapping.get(event.code)
                if key_info is None:
                    continue
                key_name, char = key_info
                if allowed_chars is not None and char not in allowed_chars:
                    continue
                if event.value == 1:
                    pending[event.code] = len(session.events)
                    session.events.append(
                        KeyEvent(char=char, code=int(event.code), key_name=key_name, t_press=event.timestamp())
                    )
                elif event.value == 0 and event.code in pending:
                    event_index = pending.pop(event.code)
                    session.events[event_index].t_release = event.timestamp()
            if stop_after is not None and len(session.typed_string()) >= len(stop_after):
                session.stop_reason = "stop_after"
                break
            if stop_after_exact is not None and session.typed_string() == stop_after_exact:
                session.stop_reason = "stop_after_exact"
                break
        if session.stop_reason is None:
            session.stop_reason = "timeout"
    except KeyboardInterrupt:
        session.stop_reason = "keyboard_interrupt"
    finally:
        session.t_end_monotonic = time.monotonic()
        if grabbed:
            try:
                dev.ungrab()
            except OSError:
                pass
        dev.close()
    return session


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            substitution_cost = 0 if char_a == char_b else 1
            current.append(
                min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + substitution_cost)
            )
        previous = current
    return previous[-1]


def score_typed(typed: str, target: str) -> dict[str, Any]:
    correct_by_position = sum(1 for got, expected in zip(typed, target) if got == expected)
    return {
        "typed": typed,
        "target": target,
        "exact_match": typed == target,
        "correct_chars": correct_by_position,
        "total_chars": len(target),
        "position_accuracy": correct_by_position / len(target) if target else 0.0,
        "levenshtein": levenshtein(typed, target),
        "edit_accuracy": 1.0 - (levenshtein(typed, target) / max(len(target), 1)),
    }


def _cmd_list(_args: argparse.Namespace) -> int:
    for info in list_keyboard_devices():
        status = "readable" if info.readable else f"unreadable: {info.error}"
        aliases = f"\taliases: {', '.join(info.aliases)}" if info.aliases else ""
        print(f"{info.path}\t{info.name}\t{status}{aliases}")
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    allowed_chars = set(args.phrase) if args.phrase_only else None
    stop_after = args.phrase if args.stop_after_phrase else None
    stop_after_exact = args.phrase if args.stop_after_exact_phrase else None
    with _suppress_terminal_echo(enabled=not args.keep_terminal_echo):
        session = capture(
            device_path=args.device,
            duration_s=args.duration,
            allowed_chars=allowed_chars,
            exclusive=not args.no_grab,
            stop_after=stop_after,
            stop_after_exact=stop_after_exact,
            stop_key=args.stop_key,
        )
    session.write(args.out)
    score = score_typed(session.typed_string(), args.phrase)
    score["wpm"] = session.wpm()
    print(f"Captured {len(session.events)} events from {session.device_name} ({session.device_path})")
    print(f"Stop reason: {session.stop_reason}")
    print(f"Typed: {session.typed_string()!r}")
    print(f"Exact match: {score['exact_match']}")
    print(f"Position accuracy: {score['position_accuracy']:.2%}")
    print(f"Levenshtein: {score['levenshtein']}")
    print(f"WPM: {score['wpm']:.1f}")
    print(f"Wrote {args.out}")
    return 0 if score["exact_match"] else 2


def _cmd_score(args: argparse.Namespace) -> int:
    session = CapturedSession.load(args.input)
    score = score_typed(session.typed_string(), args.phrase)
    score["wpm"] = session.wpm()
    print(json.dumps(score, indent=2))
    return 0 if score["exact_match"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List readable Linux input event devices.")
    list_parser.set_defaults(func=_cmd_list)

    capture_parser = subparsers.add_parser("capture", help="Capture physical keyboard events.")
    capture_parser.add_argument("--device", required=True, help="Linux input event path, e.g. /dev/input/event3.")
    capture_parser.add_argument("--duration", type=float, default=20.0)
    capture_parser.add_argument("--phrase", default="newton is the best")
    capture_parser.add_argument("--out", type=Path, required=True)
    capture_parser.add_argument("--no-grab", action="store_true", help="Do not exclusively grab the keyboard device.")
    capture_parser.add_argument("--keep-terminal-echo", action="store_true", help="Do not suppress terminal echo.")
    capture_parser.add_argument("--phrase-only", action="store_true", help="Ignore characters outside --phrase.")
    capture_parser.add_argument(
        "--stop-key",
        default="esc",
        help="Key that stops capture early without being recorded. Use 'none' to disable.",
    )
    capture_parser.add_argument(
        "--stop-after-phrase", action="store_true", help="Stop after the phrase length is captured."
    )
    capture_parser.add_argument(
        "--stop-after-exact-phrase",
        action="store_true",
        help="Stop only after the captured text exactly matches --phrase.",
    )
    capture_parser.set_defaults(func=_cmd_capture)

    score_parser = subparsers.add_parser("score", help="Score a captured session JSON file.")
    score_parser.add_argument("--input", type=Path, required=True)
    score_parser.add_argument("--phrase", default="newton is the best")
    score_parser.set_defaults(func=_cmd_score)

    args = parser.parse_args()
    if hasattr(args, "stop_key") and args.stop_key.lower() == "none":
        args.stop_key = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
