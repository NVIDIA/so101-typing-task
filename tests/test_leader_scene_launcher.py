"""Exercise launcher identity routing with executable stubs, never serial I/O."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    follower, leader = tmp_path/'follower', tmp_path/'leader'
    follower.touch()
    leader.touch()
    log = tmp_path/'args.json'
    stub = '#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["TEST_LAUNCH_LOG"],"w").write(json.dumps(sys.argv[1:]))\n'
    for name in ('python', 'lerobot-calibrate'):
        path = tmp_path/name
        path.write_text(stub)
        path.chmod(0o755)
    env = {**os.environ, 'SO101_LEROBOT_PYTHON': str(tmp_path/'python'),
           'TEST_LAUNCH_LOG': str(log), 'SO101_LEADER_PORT': '', 'SO101_LEADER_ID': ''}

    def run(mode, *args):
        recorder_args = (['--robot-id', 'test_follower', '--keyboard-device', '/dev/test_keyboard',
                          '--task-config', 'configs/tasks/keyboard-pose.json']
                         if mode in ('check', 'inspect', 'record') else [])
        return subprocess.run([str(ROOT/'scripts/leader_scene_check.sh'), mode,
                               '--leader-port', str(leader), '--leader-id', 'test_leader',
                               '--port', str(follower), *recorder_args, *args], env=env, text=True, capture_output=True)

    return run, log, follower, leader


def test_calibration_routes_only_to_leader(launcher):
    run, log, follower, leader = launcher
    assert run('calibrate').returncode == 0
    assert json.loads(log.read_text()) == ['--teleop.type=so101_leader',
        f'--teleop.port={leader}', '--teleop.id=test_leader']
    assert str(follower) not in log.read_text()


@pytest.mark.parametrize('mode,flag', [('check', '--check'), ('inspect', '--inspect'), ('record', '--execute')])
def test_modes_explicitly_route_to_recorder(launcher, mode, flag):
    run, log, follower, leader = launcher
    assert run(mode).returncode == 0
    argv = json.loads(log.read_text())
    assert argv[:2] == ['-m', 'so101_typing.hardware.teleop_scene_record']
    assert argv[argv.index('--leader-port')+1] == str(leader)
    assert argv[argv.index('--port')+1] == str(follower)
    assert argv[-1] == flag
    assert argv[argv.index('--task-config')+1] == 'configs/tasks/keyboard-pose.json'
    assert argv[argv.index('--robot-id')+1] == 'test_follower'


def test_same_port_alias_refused_before_calibration(launcher):
    run, log, follower, leader = launcher
    leader.unlink()
    leader.symlink_to(follower)
    result = run('calibrate')
    assert result.returncode == 2 and 'different serial devices' in result.stderr
    assert not log.exists()


def test_calibration_refuses_extra_arguments_and_invalid_id(launcher):
    run, log, _, _ = launcher
    assert run('calibrate', '--robot.type=so101_follower').returncode == 2
    assert run('calibrate', '--leader-id', '../follower').returncode == 2
    assert not log.exists()


def test_missing_identity_refused_before_calibration(launcher):
    run, log, _, _ = launcher
    result = run('calibrate', '--port', '')
    assert result.returncode == 2 and 'no serial device is selected automatically' in result.stderr
    assert not log.exists()
