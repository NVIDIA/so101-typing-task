# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Derived playback filtering must preserve original contact evidence."""
import hashlib
import json
from pathlib import Path

from scripts.typing_calibration_replay import build_assets


def test_replay_labels_override_preserves_raw_and_includes_external(tmp_path):
    events=[dict(code=18,char='e',key_name='KEY_E',edge='down' if value else 'up',
                 value=value,time_s=t,evdev_timestamp_s=1000+t)
            for t,value in [(1,1),(1.037,0),(1.112,1),(1.7,0)]]
    data=dict(joint_names=['joint'],samples=[dict(time_s=0,lerobot_position=[0],leader=dict(lerobot_position=[0]))],
              events=events,telemetry=[],recording_start_monotonic_s=10.,started_utc='test')
    recording=tmp_path/'recording.json';recording.write_text(json.dumps(data))
    raw=tmp_path/'keyboard_events.jsonl';raw.write_text(''.join(json.dumps(e)+'\n' for e in events))
    before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (recording,raw)]
    for name in ('typing_calibration','external_camera'):
        (tmp_path/f'{name}_timestamps.jsonl').write_text(json.dumps({'received_monotonic_ns':10_000_000_000})+'\n')
        (tmp_path/f'{name}.mp4').touch()
        (tmp_path/f'{name}.webm').touch()  # conversion already exists; test only event/metadata building
    build_assets(recording,keyboard_debounce_ms=100)
    result=json.loads((tmp_path/'playback_data.json').read_text())
    assert result['key_presses']==[[1.,'E',1.7]]
    assert result['raw_key_down_count']==2 and result['debounced_key_down_count']==1
    assert result['keyboard_debounce_ms']==100
    assert result['video_start_s']==dict(arm=0.,external=0.)
    assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in (recording,raw)]
    build_assets(recording,keyboard_debounce_ms=0)
    assert json.loads((tmp_path/'playback_data.json').read_text())['debounced_key_down_count']==2
