"""Host-only contract checks: never start Docker or connect to hardware."""
import importlib.machinery
import importlib.util
import contextlib
import io
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader('public_cli', str(ROOT / 'so101'))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


class PublicCliTests(unittest.TestCase):
    def test_runtime_fingerprint_ignores_install_metadata_but_tracks_compatibility_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text('[project]\nname = "example"\n')
            (root / 'source').mkdir()
            (root / 'source' / 'module.py').write_text('value = 1\n')
            alias = root / 'source' / 'compat'
            alias.symlink_to('target-one', target_is_directory=True)
            with patch.object(cli, 'ROOT', root):
                baseline = cli.source_fingerprint()
                metadata = root / 'source' / 'example.egg-info'
                metadata.mkdir()
                (metadata / 'PKG-INFO').write_text('Generated install metadata')
                self.assertEqual(cli.source_fingerprint(), baseline)
                alias.unlink()
                alias.symlink_to('target-two', target_is_directory=True)
                self.assertNotEqual(cli.source_fingerprint(), baseline)
                baseline = cli.source_fingerprint()
                (root / 'source' / 'module.py').write_text('value = 2\n')
                self.assertNotEqual(cli.source_fingerprint(), baseline)

    def test_export_defaults_to_auto_and_passes_measured_mapping_without_override(self):
        args = ['so101', 'prepare-deployment', '--checkpoint', 'model.pt', '--env-config', 'env.yaml',
                '--robot-id', 'robot', '--out-dir', 'bundle']
        for extra in ([], ['--joint-calibration', 'measured.json']):
            with patch.object(sys, 'argv', args+extra), patch.object(Path, 'is_file', return_value=True), \
                 patch.object(cli.subprocess, 'call', return_value=0) as run:
                self.assertEqual(cli.main(), 0)
                command = run.call_args.args[0]
                if extra:
                    self.assertNotIn('--encoder-convention', command)
                    self.assertIn(str((Path.cwd()/'measured.json').resolve()), command)
                else:
                    self.assertEqual(command[command.index('--encoder-convention')+1], 'auto')

    def test_archive_hygiene_scan_checks_source_without_parent_git_or_outputs(self):
        from scripts import check_public_release
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('source')
            (root / 'output').mkdir()
            (root / 'output' / 'run.log').write_text('local run')
            with patch.object(check_public_release, 'ROOT', root), \
                 patch.object(check_public_release.subprocess, 'check_output') as git:
                files = check_public_release.source_files()
            git.assert_not_called()
            self.assertEqual([name for _, name in files], ['README.md'])

    def test_container_uses_host_identity_and_writable_home(self):
        command = self.plan('train', '--plan')
        self.assertIn(f'--user {os.getuid()}:{os.getgid()}', command)
        self.assertIn('HOME=/tmp/so101-home', command)
        self.assertNotIn('dst=/root/', command)

    def test_missing_image_has_setup_error_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            stderr = io.StringIO()
            with patch.object(sys, 'argv', ['so101', 'probe', '--output-dir', str(out)]), \
                 patch.object(cli.subprocess, 'check_output', side_effect=cli.subprocess.CalledProcessError(1, 'docker')), \
                 contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn('run ./so101 build', stderr.getvalue())
            self.assertFalse(out.exists())

    def test_source_archive_runs_without_git_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'so101').write_text('archive launcher')
            task = root / 'configs/tasks/replay-calibrated-20260917.json'
            task.parent.mkdir(parents=True)
            task.write_bytes((ROOT / 'configs/tasks/replay-calibrated-20260917.json').read_bytes())
            out = root / 'run'
            process = Mock(stdout=[], wait=Mock(return_value=0))
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'source_fingerprint', return_value='fingerprint'), \
                 patch.dict(os.environ, {'XDG_CACHE_HOME': str(root / 'cache')}), \
                 patch.object(sys, 'argv', ['so101', 'probe', '--output-dir', str(out)]), \
                 patch.object(cli.subprocess, 'check_output', side_effect=['fingerprint', 'image-id']) as metadata, \
                 patch.object(cli.subprocess, 'Popen', return_value=process):
                self.assertEqual(cli.main(), 0)
            saved = json.loads((out / 'request.json').read_text())
            self.assertIsNone(saved['launcher_revision'])
            self.assertEqual(saved['source_distribution'], 'archive')
            self.assertEqual(saved['runtime_source_sha256'], 'fingerprint')
            self.assertEqual(saved['artifacts']['task_config']['path'], str(task))
            self.assertTrue(all(call.args[0][0] == 'docker' for call in metadata.call_args_list))

    def test_download_forwards_default_and_explicit_selections(self):
        for extra in ([], ['--all'], ['--calibrated'], ['--historical']):
            with patch.object(sys, 'argv', ['so101', 'download', *extra]), \
                 patch.object(cli.subprocess, 'call', return_value=0) as execute:
                self.assertEqual(cli.main(), 0)
                self.assertEqual(execute.call_args.args[0],
                                 [str(ROOT / 'scripts/download_benchmark_artifacts.sh'), *extra])

    def test_download_rejects_conflicting_selections_before_execution(self):
        for flags in combinations(('--all', '--calibrated', '--historical'), 2):
            with self.subTest(flags=flags), \
                 patch.object(sys, 'argv', ['so101', 'download', *flags]), \
                 patch.object(cli.subprocess, 'call') as execute, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 2)
            execute.assert_not_called()

    def plan(self, *args):
        with patch.object(sys, 'argv', ['so101', *args]), patch('builtins.print') as out, \
             patch.dict(os.environ, {'DISPLAY': ':test'}), \
             patch.object(cli.subprocess, 'call') as execute, \
             patch.object(cli.subprocess, 'check_output') as inspect, \
             patch.object(cli.subprocess, 'Popen') as launch:
            self.assertEqual(cli.main(), 0)
            execute.assert_not_called()
            inspect.assert_not_called()
            launch.assert_not_called()
            return out.call_args.args[0]

    def test_fresh_training_probe_and_scene_default_to_september_fitted_task(self):
        task = ROOT / 'configs/tasks/replay-calibrated-20260917.json'
        for operation in ('train', 'probe', 'scene'):
            with self.subTest(operation=operation):
                command = self.plan(operation, '--plan')
                self.assertIn(f'src={task},dst=/inputs/task.json,readonly', command)
                self.assertIn('--task-config /inputs/task.json', command)
                if operation == 'probe':
                    self.assertIn('--reset-audit-count 3', command)

    def test_explicit_task_overrides_fresh_command_default(self):
        task = ROOT / 'configs/tasks/physical-reference-20260915.json'
        for operation in ('train', 'probe', 'scene'):
            with self.subTest(operation=operation):
                command = self.plan(operation, '--task-config', str(task), '--plan')
                self.assertIn(f'src={task},dst=/inputs/task.json,readonly', command)
                self.assertNotIn('replay-calibrated-20260917.json', command)

    def test_configuration_paths_expand_home_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.pt').write_bytes(b'fixture')
            (root / 'env.yaml').write_text('fixture: true')
            cases = {
                'task': {'schema_version': 1},
                'physics': {'schema_version': 1, 'solver': 'mjwarp'},
                'perturbation': {'schema_version': 1, 'domain_randomization': {'reset_joint_noise_rad': 0.01}},
            }
            for kind, data in cases.items():
                source = root / f'{kind}.json'
                source.write_text(json.dumps(data))
                with self.subTest(kind=kind), patch.dict(os.environ, {'HOME': str(root)}):
                    command = self.plan('evaluate', '--checkpoint', str(root / 'model.pt'),
                                        '--env-config', str(root / 'env.yaml'),
                                        '--' + kind + '-config', f'~/{kind}.json', '--plan')
                    self.assertIn(f'src={source},dst=/inputs/{kind}.json,readonly', command)

    def test_checkpoint_commands_preserve_historical_scene_without_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.pt'
            checkpoint.write_bytes(b'fixture')
            config = root / 'params/env.yaml'
            config.parent.mkdir()
            config.write_text('fixture: true')
            for operation in ('train', 'evaluate', 'play', 'video'):
                with self.subTest(operation=operation):
                    args = [operation, '--checkpoint', str(checkpoint), '--plan']
                    if operation != 'train':
                        args += ['--env-config', str(config)]
                    command = self.plan(*args)
                    self.assertNotIn('--task-config', command)
                    self.assertNotIn('dst=/inputs/task.json', command)
                    self.assertNotIn('--physics-config', command)

    def test_checkpoint_sidecars_are_restored_and_explicit_task_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.pt'
            checkpoint.write_bytes(b'fixture')
            config = root / 'params/env.yaml'
            config.parent.mkdir()
            config.write_text('fixture: true')
            saved_task = config.parent / 'public_task.json'
            saved_task.write_text(json.dumps({'schema_version': 1, 'keyboard': {'yaw_offset_deg': 2}}))
            saved_physics = config.parent / 'public_physics.json'
            saved_physics.write_text(json.dumps({'schema_version': 1, 'solver': 'mjwarp'}))
            explicit_task = ROOT / 'configs/tasks/physical-reference-20260915.json'
            for operation in ('train', 'evaluate', 'play', 'video'):
                for override in (False, True):
                    with self.subTest(operation=operation, override=override):
                        args = [operation, '--checkpoint', str(checkpoint), '--plan']
                        if operation != 'train':
                            args += ['--env-config', str(config)]
                        if override:
                            args += ['--task-config', str(explicit_task)]
                        command = self.plan(*args)
                        expected_task = explicit_task if override else saved_task
                        self.assertIn(f'src={expected_task},dst=/inputs/task.json,readonly', command)
                        self.assertIn(f'src={saved_physics},dst=/inputs/physics.json,readonly', command)
                        self.assertIn('--task-config /inputs/task.json', command)
                        self.assertIn('--physics-config /inputs/physics.json', command)
                        self.assertNotIn('replay-calibrated-20260917.json', command)
                        if override:
                            self.assertNotIn(f'src={saved_task},', command)

    def test_scene_zero_mapping_still_requires_explicit_matching_task(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory) / 'mapping.json'
            mapping.write_text('{}')
            with patch.object(sys, 'argv', ['so101', 'scene', '--zero-mapping', str(mapping), '--plan']), \
                 patch.dict(os.environ, {'DISPLAY': ':test'}), \
                 patch.object(cli.subprocess, 'call') as execute, \
                 contextlib.redirect_stderr(io.StringIO()) as stderr, \
                 self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn('--zero-mapping requires the matching --task-config', stderr.getvalue())
            execute.assert_not_called()
            task = ROOT / 'configs/tasks/physical-reference-20260915.json'
            command = self.plan('scene', '--zero-mapping', str(mapping), '--task-config', str(task), '--plan')
            self.assertIn(f'src={task},dst=/inputs/task.json,readonly', command)

    def test_playback_target_defaults_to_newton_and_preserves_p1a_and_explicit_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            checkpoint.write_bytes(b'fixture')
            config = Path(directory) / 'env.yaml'
            config.write_text('fixture: true')
            for operation in ('play', 'video'):
                for extra, target in (([], 'NEWTON'), (['--stage', 'p1a'], 'HE'),
                                      (['--target', 'nvidia'], 'NVIDIA')):
                    with self.subTest(operation=operation, target=target):
                        command = self.plan(operation, '--checkpoint', str(checkpoint),
                                            '--env-config', str(config), '--plan', *extra)
                        self.assertIn(f'--target {target}', command)

    def test_fresh_training_defaults_to_gtc_berlin(self):
        command = self.plan('train', '--plan')
        self.assertIn('FixedCartesian-GTCBerlin-P1A', command)

    def test_profiles_map_to_distinct_registered_tasks(self):
        for profile, registered in cli.PROFILES.items():
            command = self.plan('train', '--actuator', profile, '--plan')
            self.assertIn(f'FixedCartesian-{registered}-P1A', command)
            self.assertIn('physics=newton_mjwarp', command)
            self.assertNotIn('--device=/dev', command)

    def test_rejects_unqualified_solver_and_invalid_budget(self):
        for args in [('train', '--solver', 'physx'), ('train', '--iterations', '0'),
                     ('train', '--actuator', 'workshop')]:
            with patch.object(sys, 'argv', ['so101', *args]), self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 2)

    def test_artifacts_are_mounted_read_only_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='typing test ') as directory:
            checkpoint = Path(directory) / 'checkpoint.pt'
            config = Path(directory) / 'env.yaml'
            checkpoint.write_bytes(b'fixture')
            config.write_text('fixture: true')
            command = self.plan('evaluate', '--checkpoint', str(checkpoint), '--env-config', str(config), '--plan')
            self.assertIn('dst=/inputs/checkpoint.pt,readonly', command)
            self.assertIn('dst=/inputs/env.yaml,readonly', command)

    def test_resume_preserves_explicit_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model_0.pt'
            checkpoint.write_bytes(b'fixture')
            command = self.plan('train', '--stage', 'transit15', '--checkpoint', str(checkpoint), '--plan')
            self.assertIn('--resume --checkpoint /inputs/checkpoint.pt', command)
            self.assertIn('dst=/inputs/checkpoint.pt,readonly', command)

    def test_probe_obeys_selected_profile(self):
        command = self.plan('probe', '--actuator', 'usd', '--seed', '42', '--plan')
        self.assertIn('FixedCartesian-USDDrive-P1D-Transit15', command)
        self.assertIn('--seed 42', command)


if __name__ == '__main__':
    unittest.main()
