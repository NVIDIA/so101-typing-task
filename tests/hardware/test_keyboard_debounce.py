# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical event filtering: preserve timing and raw evidence, never hide other keys."""
from copy import deepcopy

import pytest

from so101_typing.hardware.keyboard_debounce import KeyboardDebouncer, debounce_events
from so101_typing.hardware.policy_runtime import PressGate


def edge(t, value, code=18, char='e'):
    return dict(code=code, char=char, key_name='KEY_'+char.upper(), value=value,
                edge='down' if value else 'up', evdev_timestamp_s=t,
                received_monotonic_s=100., time_s=t)


def test_short_release_merges_across_polls_and_preserves_endpoints():
    d = KeyboardDebouncer(50)
    first, bounce_up, bounce_down, last = edge(1,1), edge(1.045,0), edge(1.067,1), edge(1.435,0)
    assert d.process([first],now_s=1)==[first]
    assert d.process([bounce_up],now_s=1.05)==[]
    assert d.process([bounce_down],now_s=1.07)==[]
    assert d.process([last],now_s=1.45)==[]
    assert d.process([],now_s=1.486)==[last]


def test_device_timing_preserves_real_repeated_letters_in_one_poll():
    events=[edge(1,1),edge(1.1,0),edge(1.2,1),edge(1.3,0)]
    assert debounce_events(events)==events  # identical receipt times must not collapse them


def test_intervening_key_is_distinct_and_keeps_order():
    events=[edge(1,1),edge(1.1,0),edge(1.12,1,17,'w'),edge(1.13,0,17,'w'),edge(1.14,1),edge(1.5,0)]
    assert debounce_events(events)==events


def test_overlapping_wrong_key_is_immediate_and_raw_guard_still_rejects():
    events=[edge(1,1),edge(1.01,1,23,'i')]
    assert KeyboardDebouncer().process(events,now_s=1.01)==events
    gate=PressGate()
    assert not gate.process(events[0])
    assert gate.process(events[1])


def test_filter_does_not_mutate_raw_or_fabricate_missing_release():
    events=[edge(1,1),edge(1.05,0),edge(1.07,1)]
    before=deepcopy(events)
    assert debounce_events(events)==[events[0]]
    assert events==before
    assert debounce_events(events,0)==events


@pytest.mark.parametrize('window',[-1,1001,float('nan'),float('inf')])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError,match='debounce'):
        KeyboardDebouncer(window)


def test_boundary_is_not_merged():
    events=[edge(1,1),edge(2,0),edge(2.05,1),edge(3,0)]
    assert debounce_events(events)==events


def test_trailing_and_orphan_releases_retained_without_synthesizing():
    events=[edge(.1,0),edge(1,1),edge(1.1,0)]
    assert debounce_events(events)==events


def test_latest_recorded_chatter_pattern_becomes_newton():
    events=[]
    for index,(char,gap) in enumerate(zip('newton',[None,.022,.008,.030,.037,None])):
        t=index*2.
        events.append(edge(t,1,index,char))
        if gap is not None:
            events.extend([edge(t+.045,0,index,char),edge(t+.045+gap,1,index,char)])
        events.append(edge(t+.5,0,index,char))
    filtered=debounce_events(events)
    assert ''.join(e['char'] for e in filtered if e['value']==1)=='newton'
    assert len(filtered)==12


def test_longer_gap_requires_explicit_larger_window():
    events=[edge(1,1),edge(1.037,0),edge(1.112,1),edge(1.7,0)]
    assert debounce_events(events,50)==events
    assert debounce_events(events,100)==[events[0],events[-1]]
