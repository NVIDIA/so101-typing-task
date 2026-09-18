"""The short workflow cannot report success or suggest training after wrong contacts."""
import copy
import json
from pathlib import Path

import pytest

from so101_typing.hardware import real2sim as workflow


@pytest.mark.parametrize('change', ['wrong_key', 'multiple', 'missing', 'empty', 'wrong_count'])
def test_per_key_check_requires_actual_matching_contacts(change):
    result = {'contacts': [{'key': 'Q'}, {'key': 'G'}],
              'best': {'total': 2, 'correct_first_contacts': 2, 'first_keys': ['Q', 'G']}}
    assert workflow.contact_pass(result)
    if change == 'wrong_key': result['best']['first_keys'][1] = 'H'
    if change == 'multiple': result['best']['first_keys'][1] = 'MULTIPLE'
    if change == 'missing': result['best']['first_keys'].pop()
    if change == 'empty': result = {'contacts': [], 'best': {'total': 0, 'correct_first_contacts': 0, 'first_keys': []}}
    if change == 'wrong_count': result['best']['correct_first_contacts'] = 1
    assert not workflow.contact_pass(result)


@pytest.mark.parametrize('passed', [False, True])
def test_preserves_reference_and_only_prints_training_on_pass(tmp_path, monkeypatch, capsys, passed):
    paths = [tmp_path/'capture-a.json', tmp_path/'capture-b.json']
    for i, path in enumerate(paths): path.write_text(str(i))
    records = [{'kind': 'so101_torque_off_recording', 'task_config': {'schema_version': 1}}]*2
    original = copy.deepcopy(records)
    monkeypatch.setattr(workflow, 'calibrate', lambda *a, **k: None)
    def refinement(command, **kwargs):
        assert command[1:3] == ['-m', 'so101_typing.hardware.refine_scene_contacts']
        folder = Path(command[command.index('--out-dir')+1]); folder.mkdir()
        result = {'contacts': [{'key': 'Q'}], 'best': {'total': 1,
                  'correct_first_contacts': int(passed), 'first_keys': ['Q' if passed else 'P']}}
        for name, content in [('contact-result.json', result), ('task.json', records[0]['task_config']), ('mapping-report.json', {'offsets': 'synthetic'})]:
            (folder/name).write_text(json.dumps(content))
    monkeypatch.setattr(workflow.subprocess, 'run', refinement)
    out = tmp_path/'scene'
    assert workflow.run(paths, records, out, 2) == (0 if passed else 1)
    text = capsys.readouterr().out
    assert ('./so101 train' in text) == passed
    status = json.loads((out/'scene-calibration.json').read_text())
    assert status['all_recordings_used_for_fitting']
    assert not status['physical_deployment_validated']
    assert status['task_sha256'] == workflow.digest(out/'task.json')
    assert records == original
    with pytest.raises(ValueError, match='overwrite'):
        workflow.run(paths, records, out, 2)
