"""Released policy pairing checks require no policy execution or simulation."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101_typing.hardware import checkpoint_identity as identity

ROOT = Path(__file__).resolve().parents[1]


def test_published_pairs_match_downloader_and_reject_crossed_environments():
    downloader = (ROOT/'scripts/download_benchmark_artifacts.sh').read_text()
    assert len(identity.RELEASED_ENVIRONMENTS) == 5
    for checkpoint, environment in identity.RELEASED_ENVIRONMENTS.items():
        assert checkpoint in downloader and environment in downloader
        identity.validate_pair(checkpoint, environment)
        for other in set(identity.RELEASED_ENVIRONMENTS.values())-{environment}:
            with pytest.raises(ValueError, match='published checkpoint requires its matching environment'):
                identity.validate_pair(checkpoint, other)
    # Actor format alone cannot establish training provenance for a new policy.
    identity.validate_pair('unknown-checkpoint', 'caller-supplied-environment')


@pytest.mark.parametrize('entrypoint', ['evaluation', 'legacy_handoff'])
def test_runtime_entrypoints_reject_known_mismatch_before_environment_validation(tmp_path, monkeypatch, entrypoint):
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'synthetic-known-actor')
    environment = tmp_path/'env.yaml'
    environment.write_text('not a training environment')
    monkeypatch.setitem(identity.RELEASED_ENVIRONMENTS,
                        hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'expected-environment-hash')
    if entrypoint == 'legacy_handoff':
        from so101_typing.hardware.run_fixed_cartesian_policy_handoff import _validated_training_environment
        invoke = lambda: _validated_training_environment(checkpoint=checkpoint, env_config=environment,
            profile=SimpleNamespace(name='physical-calibrated-20260718'), stage='p1d')
    else:
        # Load the actual pure validation function without importing Isaac Lab or
        # parsing the simulator entrypoint's global command-line arguments.
        source = ROOT/'scripts/reinforcement_learning/rsl_rl/evaluate_so101_fixed_cartesian.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_validate_artifact_contract')
        namespace = {'Path': Path, 'ManagerBasedRLEnvCfg': object,
                     'args_cli': SimpleNamespace(checkpoint=checkpoint, env_config=environment)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        invoke = lambda: namespace['_validate_artifact_contract'](None)
    with pytest.raises(ValueError, match='published checkpoint requires its matching environment'):
        invoke()
