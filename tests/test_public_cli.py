"""Host-only contract checks: never start Docker or connect to hardware."""
import importlib.machinery
import importlib.util
import contextlib
import io
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
            self.assertTrue(all(call.args[0][0] == 'docker' for call in metadata.call_args_list))

    def test_download_archive_requires_explicit_option(self):
        for extra in ([], ['--all'], ['--calibrated']):
            with patch.object(sys, 'argv', ['so101', 'download', *extra]), \
                 patch.object(cli.subprocess, 'call', return_value=0) as execute:
                self.assertEqual(cli.main(), 0)
                self.assertEqual(execute.call_args.args[0],
                                 [str(ROOT / 'scripts/download_benchmark_artifacts.sh'), *extra])

    def test_download_rejects_conflicting_selections_before_execution(self):
        with patch.object(sys, 'argv', ['so101', 'download', '--all', '--calibrated']), \
             patch.object(cli.subprocess, 'call') as execute, self.assertRaises(SystemExit) as error:
            cli.main()
        self.assertEqual(error.exception.code, 2)
        execute.assert_not_called()

    def plan(self, *args):
        with patch.object(sys, 'argv', ['so101', *args]), patch('builtins.print') as out, patch.object(cli.subprocess, 'call') as execute:
            self.assertEqual(cli.main(), 0)
            execute.assert_not_called()
            return out.call_args.args[0]

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
