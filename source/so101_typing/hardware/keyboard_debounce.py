# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Release debouncing shared by recording, replay and physical typing feedback.

Preserve the original first-down and final-up timestamps. A same-key re-press
before the release window expires cancels the pending release. Different-key
presses finalize earlier releases immediately, preserving real event ordering.
"""
from __future__ import annotations

import math

DEFAULT_KEYBOARD_DEBOUNCE_MS = 50.0


def event_timestamp(event: dict) -> float:
    """Prefer device time: separate edges may arrive in the same polling cycle."""
    for field in ('evdev_timestamp_s', 'time_s', 'received_monotonic_s'):
        if field in event:
            stamp = float(event[field])
            if not math.isfinite(stamp):
                raise ValueError('keyboard event timestamp must be finite')
            return stamp
    raise ValueError('keyboard event has no timestamp')


class KeyboardDebouncer:
    """Streaming filter; ``now_s`` must use the input events' clock domain."""

    def __init__(self, window_ms: float = DEFAULT_KEYBOARD_DEBOUNCE_MS):
        if not math.isfinite(window_ms) or not 0 <= window_ms <= 1000:
            raise ValueError('keyboard debounce must be finite and between 0 and 1000 ms')
        self.window_s = window_ms / 1000
        self.held: set[int] = set()
        self.pending: dict[int, dict] = {}

    def flush(self, now_s: float, *, final: bool = False) -> list[dict]:
        """Emit confirmed releases; final retains pending actual ups at end of log."""
        out = []
        for code, event in sorted(self.pending.items(), key=lambda item: event_timestamp(item[1])):
            if final or event_timestamp(event) + self.window_s <= now_s:
                out.append(event.copy())
                self.held.discard(code)
                del self.pending[code]
        return out

    def process(self, events: list[dict], *, now_s: float) -> list[dict]:
        if self.window_s == 0:
            return [event.copy() for event in events]
        out = []
        for event in events:
            stamp = event_timestamp(event)
            out.extend(self.flush(stamp))
            code, value = int(event['code']), int(event['value'])
            if value == 1:
                if code in self.pending:
                    del self.pending[code]
                elif code not in self.held:
                    # Another key intervening is a distinct press, not chatter.
                    out.extend(self.flush(stamp, final=True))
                    self.held.add(code)
                    out.append(event.copy())
            elif value == 0:
                if code in self.held:
                    self.pending.setdefault(code, event.copy())
                else:
                    out.append(event.copy())  # retain an orphan up in a trimmed take
            else:
                raise ValueError(f'unexpected key value {value}; filter repeat events first')
        out.extend(self.flush(now_s))
        return out


def debounce_events(events: list[dict], window_ms: float = DEFAULT_KEYBOARD_DEBOUNCE_MS) -> list[dict]:
    """Derive a filtered sequence without modifying any source event."""
    debouncer = KeyboardDebouncer(window_ms)
    out = debouncer.process(events, now_s=event_timestamp(events[-1]) if events else 0.)
    out.extend(debouncer.flush(0., final=True))
    return out
