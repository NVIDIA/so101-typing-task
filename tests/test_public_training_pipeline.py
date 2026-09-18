import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.run_public_training_pipeline import Pipeline, passes_gate
from scripts import run_public_training_pipeline as supervisor
from scripts.so101_homing.joint_calibration import ARM_JOINT_NAMES, fit


class TrainingPipelineTests(unittest.TestCase):
    def test_default_budget_matches_september_recipe(self):
        with patch('sys.argv', ['pipeline', '--out-dir', 'unused']), \
             patch.object(supervisor, 'Pipeline') as pipeline:
            pipeline.return_value.execute.return_value = 0
            self.assertEqual(supervisor.main(), 0)
        args = pipeline.call_args.args[0]
        self.assertEqual((args.p1a_iterations, args.transit_iterations), (500, 2000))

    def test_source_archive_does_not_require_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(supervisor, 'ROOT', root), \
                 patch.object(supervisor.subprocess, 'check_output') as git:
                pipeline = self.pipeline(root / 'run')
            git.assert_not_called()
            self.assertIsNone(pipeline.status['source_commit'])
            self.assertEqual(pipeline.status['source_distribution'], 'archive')

    def test_gate_uses_strict_success_and_rejects_nonfinite_scores(self):
        self.assertFalse(passes_gate({'typed_exact_rate': 1.0, 'overall_success_rate': 0.2}, 0.9))
        self.assertFalse(passes_gate({'overall_success_rate': float('nan')}, 0.9))
        self.assertTrue(passes_gate({'overall_success_rate': 0.95}, 0.9))

    def pipeline(self, root, **overrides):
        args = argparse.Namespace(out_dir=root, num_envs=4, eval_envs=4, p1a_iterations=1,
                                  transit_iterations=1, seed=1307, minimum_success=0.9,
                                  prepare_deployment=True, robot_id='test_robot',
                                  encoder_convention='benchmark', joint_calibration=None,
                                  task_config=None, physics_config=None)
        vars(args).update(overrides)
        pipeline = Pipeline(args)
        pipeline.monitor = Mock()
        return pipeline

    def test_invalid_custom_inputs_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [('task', '{'), ('task', '{"schema_version": 1, "keyboard": {"position_m": [0, 0]}}'),
                     ('physics', '{"schema_version": 1, "solver": "other"}'), ('physics', None)]
            for index, (kind, contents) in enumerate(cases):
                with self.subTest(kind=kind, contents=contents):
                    source = root / f'input-{index}.json'
                    if contents is not None:
                        source.write_text(contents)
                    output = root / f'run-{index}'
                    with patch('sys.argv', ['pipeline', '--out-dir', str(output),
                                             '--' + kind + '-config', str(source)]), \
                         patch.object(Pipeline, 'execute') as execute, self.assertRaises(SystemExit) as error:
                        supervisor.main()
                    self.assertEqual(error.exception.code, 2)
                    execute.assert_not_called()
                    self.assertFalse(output.exists())

    def test_custom_configs_are_snapshotted_and_forwarded_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / 'task.json'
            physics = root / 'physics.json'
            task.write_text('{"schema_version": 1, "keyboard": {"position_m": [0.2, 0.01, 0.008]}}\n')
            physics.write_text('{"schema_version": 1, "solver": "mjwarp", "num_substeps": 2}\n')
            pipeline = self.pipeline(root / 'run', task_config=task, physics_config=physics)
            for kind, source in (('task', task), ('physics', physics)):
                metadata = pipeline.status['inputs'][kind + '_config']
                snapshot = Path(metadata['snapshot_path'])
                self.assertEqual(metadata['source_path'], str(source.resolve()))
                self.assertEqual(snapshot.parent, pipeline.root / 'inputs')
                self.assertEqual(snapshot.read_bytes(), source.read_bytes())
                self.assertEqual(metadata['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
                source.write_text('original config changed after training started')
            with patch.object(pipeline, 'run', return_value=root / 'stage') as run, \
                 patch.object(supervisor, 'checkpoint_artifacts', return_value=('checkpoint', 'environment')):
                pipeline.train('p1a', 'p1a', 1)
                pipeline.train('transit', 'transit15', 2, checkpoint='previous.pt')
            for call in run.call_args_list:
                command = call.args[1]
                for kind in ('task', 'physics'):
                    supplied = command[command.index('--' + kind + '-config') + 1]
                    self.assertEqual(supplied, pipeline.status['inputs'][kind + '_config']['snapshot_path'])
            command = run.call_args_list[-1].args[1]
            self.assertEqual(command[command.index('--checkpoint') + 1], 'previous.pt')

    def calibration(self):
        return fit({'robot_id': 'test_robot', 'arm_joint_names': list(ARM_JOINT_NAMES),
                    'samples': [{'measurement_reference': f'synthetic fixture {i}',
                                 'model_deg': [i + 1] * 5, 'encoder_deg': [i] * 5}
                                for i in range(3)]})

    def test_invalid_joint_calibration_fails_before_creating_output(self):
        wrong_robot = self.calibration()
        wrong_robot['robot_id'] = 'another_robot'
        inconsistent = self.calibration()
        inconsistent['joint_zero_offset_deg'][0] += 1
        missing_evidence = self.calibration()
        del missing_evidence['samples'][0]['model_deg']
        cases = [None, '{', json.dumps(wrong_robot), json.dumps(inconsistent), json.dumps(missing_evidence)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, contents in enumerate(cases):
                with self.subTest(contents=contents):
                    source = root / f'fit-{index}.json'
                    if contents is not None:
                        source.write_text(contents)
                    output = root / f'run-{index}'
                    with patch('sys.argv', ['pipeline', '--out-dir', str(output), '--prepare-deployment',
                                           '--robot-id', 'test_robot', '--joint-calibration', str(source)]), \
                         patch.object(Pipeline, 'execute') as execute, self.assertRaises(SystemExit) as error:
                        supervisor.main()
                    self.assertEqual(error.exception.code, 2)
                    execute.assert_not_called()
                    self.assertFalse(output.exists())
            with patch.object(Path, 'read_bytes', side_effect=PermissionError('unreadable calibration')), \
                 self.assertRaises(PermissionError):
                self.pipeline(root / 'unreadable-run', joint_calibration=root / 'fit.json')
            self.assertFalse((root / 'unreadable-run').exists())

    def test_relative_joint_calibration_is_snapshotted_and_export_uses_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'fit.json'
            source.write_text(json.dumps(self.calibration()))
            payload = source.read_bytes()
            original_directory = Path.cwd()
            try:
                os.chdir(root)
                pipeline = self.pipeline(root / 'run', joint_calibration=Path('fit.json'), encoder_convention=None)
            finally:
                os.chdir(original_directory)
            metadata = pipeline.status['inputs']['joint_calibration']
            snapshot = Path(metadata['snapshot_path'])
            self.assertEqual(metadata['source_path'], str(source))
            self.assertEqual(metadata['sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(snapshot.read_bytes(), payload)
            source.write_text('changed after training started')
            with patch.object(pipeline, 'train', return_value=('checkpoint', 'config')), \
                 patch.object(pipeline, 'evaluate', return_value=True), \
                 patch.object(pipeline, 'video'), patch.object(pipeline, 'run') as run:
                self.assertEqual(pipeline.execute(), 0)
            exported = run.call_args_list[0].args[1]
            self.assertEqual(exported[exported.index('--joint-calibration') + 1], str(snapshot))
            self.assertEqual(snapshot.read_bytes(), payload)

    def test_archive_calibration_validation_uses_only_standard_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ['scripts/run_public_training_pipeline.py',
                     'scripts/reinforcement_learning/rsl_rl/public_task.py',
                     'scripts/reinforcement_learning/rsl_rl/public_physics.py',
                     'source/so101_typing/__init__.py', 'source/so101_typing/hardware/__init__.py',
                     'source/so101_typing/hardware/constants.py',
                     'source/so101_typing/hardware/joint_calibration.py']
            for name in paths:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(supervisor.ROOT / name, destination)
            (root / 'fit.json').write_text(json.dumps(self.calibration()))
            code = (
                'import argparse, pathlib, runpy, sys; '
                'module = runpy.run_path(sys.argv[1]); '
                'args = argparse.Namespace(out_dir=pathlib.Path("run"), '
                'joint_calibration=pathlib.Path("fit.json"), robot_id="test_robot"); '
                'pipeline = module["Pipeline"](args); '
                'assert pipeline.status["source_distribution"] == "archive"; '
                'assert "numpy" not in sys.modules and "torch" not in sys.modules'
            )
            subprocess.run([sys.executable, '-I', '-S', '-c', code,
                            str(root / paths[0])], cwd=root, check=True, capture_output=True, text=True)
            saved = json.loads((root / 'run/status.json').read_text())
            self.assertEqual(saved['inputs']['joint_calibration']['source_path'], str(root / 'fit.json'))

    def test_evaluation_and_video_inherit_saved_checkpoint_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / 'task.json'
            task.write_text('{"schema_version": 1}')
            pipeline = self.pipeline(root / 'run', task_config=task)
            report_dir = root / 'evaluation'
            report_dir.mkdir()
            (report_dir / 'evaluation.json').write_text(json.dumps({
                'overall_success_rate': 1.0, 'typed_exact_rate': 1.0}))

            def run_stage(*args, **kwargs):
                pipeline.status['stages'].append({})
                return report_dir

            with patch.object(pipeline, 'run', side_effect=run_stage) as run:
                pipeline.evaluate('evaluation', 'transit15', ('model.pt', 'saved/params/env.yaml'), 2307)
                pipeline.video('video', 'transit15', ('model.pt', 'saved/params/env.yaml'), 'NVIDIA')
            for call in run.call_args_list:
                command = call.args[1]
                self.assertEqual(command[command.index('--env-config') + 1], 'saved/params/env.yaml')
                self.assertNotIn('--task-config', command)
                self.assertNotIn('--physics-config', command)

    def test_pinned_image_is_used_for_simulation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(Path(directory) / 'run', image='so101:test-frozen')
            with patch.object(supervisor.subprocess, 'run', return_value=Mock(returncode=0)) as run:
                pipeline.run('train', ['train'])
                pipeline.run('evaluate', ['evaluate'])
                pipeline.run('video', ['video'])
                pipeline.run('export', ['prepare-deployment'], simulation=False)
            for call in run.call_args_list[:3]:
                command = call.args[0]
                self.assertEqual(command[command.index('--image') + 1], 'so101:test-frozen')
            self.assertNotIn('--image', run.call_args_list[-1].args[0])

    def test_failed_p1a_gate_keeps_evidence_and_prevents_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(Path(directory) / 'run')
            with patch.object(pipeline, 'train', return_value=('checkpoint', 'config')) as train, \
                 patch.object(pipeline, 'evaluate', return_value=False), \
                 patch.object(pipeline, 'video') as video, patch.object(pipeline, 'run') as run:
                self.assertEqual(pipeline.execute(), 2)
            train.assert_called_once()
            video.assert_called_once()
            run.assert_not_called()
            saved = json.loads((pipeline.root / 'status.json').read_text())
            self.assertEqual(saved['status'], 'p1a_quality_gate_failed')
            self.assertFalse(saved['hardware_motion_enabled'])

    def test_successful_pipeline_only_dry_runs_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(Path(directory) / 'run')
            with patch.object(pipeline, 'train', return_value=('checkpoint', 'config')) as train, \
                 patch.object(pipeline, 'evaluate', return_value=True) as evaluate, \
                 patch.object(pipeline, 'video') as video, patch.object(pipeline, 'run') as run:
                self.assertEqual(pipeline.execute(), 0)
            self.assertEqual(train.call_count, 2)
            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(run.call_count, 2)
            exported = run.call_args_list[0].args[1]
            self.assertEqual(exported[exported.index('--encoder-convention')+1], 'benchmark')
            self.assertEqual(run.call_args.args[1][0], 'deploy')
            self.assertEqual(video.call_args.args[-1], 'NEWTON')
            self.assertEqual(run.call_args.args[1][-1], 'NEWTON')
            self.assertNotIn('--execute', run.call_args.args[1])

    def test_export_requires_mapping_before_spending_training_budget(self):
        with patch('sys.argv', ['pipeline', '--out-dir', 'unused', '--prepare-deployment', '--robot-id', 'robot']), \
             patch.object(supervisor, 'Pipeline') as pipeline, self.assertRaises(SystemExit) as error:
            supervisor.main()
        self.assertEqual(error.exception.code, 2)
        pipeline.assert_not_called()

    def test_failed_six_letter_gate_prevents_export(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(Path(directory) / 'run')
            with patch.object(pipeline, 'train', return_value=('checkpoint', 'config')), \
                 patch.object(pipeline, 'evaluate', side_effect=[True, True, False]), \
                 patch.object(pipeline, 'video') as video, patch.object(pipeline, 'run') as run:
                self.assertEqual(pipeline.execute(), 2)
            self.assertEqual(video.call_count, 2)
            run.assert_not_called()
            self.assertEqual(pipeline.status['status'], 'transit15_quality_gate_failed')


if __name__ == '__main__':
    unittest.main()
