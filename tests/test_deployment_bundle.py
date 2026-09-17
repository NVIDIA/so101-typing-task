from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
import sys
import types
from dataclasses import make_dataclass
from unittest.mock import patch
import yaml
from scripts.so101_homing.constants import ARM_JOINT_NAMES
from scripts.so101_homing.fixed_cartesian_policy import JOINT_RATE_LIMITS_RAD_S
from scripts.so101_homing.deployment_bundle import profile_from_environment, sha256, validate_manifest
from scripts.so101_homing import deployment_bundle, joint_calibration


class DeploymentBundleTests(unittest.TestCase):
    def test_missing_calibration_stops_export_before_writing_bundle(self):
        from scripts.so101_homing import deployment_bundle
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'bundle'
            with patch.object(deployment_bundle, 'require_calibration', side_effect=FileNotFoundError('calibrate first')), \
                 patch.object(deployment_bundle, 'FixedCartesianPolicy') as policy:
                with self.assertRaises(FileNotFoundError):
                    deployment_bundle.prepare(Path('unused'), Path('unused'), 'robot', 'lerobot', out)
                policy.assert_not_called()
            self.assertFalse(out.exists())

    def test_missing_calibration_stops_dry_run_before_runner(self):
        from scripts.so101_homing import deploy_bundle
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'deployment.json').write_text(json.dumps({'robot_id': 'missing'}))
            with patch('sys.argv', ['deploy_bundle', directory, 'NVIDIA']), \
                 patch.object(deploy_bundle, 'validate_manifest', return_value=(None, {'robot_id': 'missing', 'actuator_profile': 'anchorbench'})), \
                 patch.object(deploy_bundle, 'require_calibration', side_effect=FileNotFoundError('calibrate first')), \
                 patch.object(deploy_bundle.subprocess, 'call') as run, self.assertRaises(SystemExit) as error:
                deploy_bundle.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_malformed_bundle_is_rejected_without_runner_or_traceback(self):
        from scripts.so101_homing import deploy_bundle
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'deployment.json'
            for contents in (None, '{broken', '[]', '{}'):
                if contents is not None:
                    manifest.write_text(contents)
                stderr = io.StringIO()
                with patch('sys.argv', ['deploy_bundle', directory, 'NVIDIA']), \
                     patch.object(deploy_bundle.subprocess, 'call') as run, \
                     contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    deploy_bundle.main()
                self.assertEqual(error.exception.code, 2)
                self.assertNotIn('Traceback', stderr.getvalue())
                run.assert_not_called()

    def test_calibration_checks_numeric_fields_and_ranges_offline(self):
        from scripts.so101_homing import robot
        motors = types.ModuleType('lerobot.motors')
        motors.MotorCalibration = make_dataclass('MotorCalibration',
            [('id', int), ('drive_mode', int), ('homing_offset', int), ('range_min', int), ('range_max', int)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'robot.json'
            valid = {name: dict(id=i+1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)
                     for i, name in enumerate((*ARM_JOINT_NAMES, 'gripper'))}
            with patch.object(robot, 'calibration_directory', return_value=root), \
                 patch.dict(sys.modules, {'lerobot.motors': motors}):
                path.write_text(json.dumps(valid))
                self.assertEqual(robot.require_calibration('robot'), path)
                for field, value in [('homing_offset', '0'), ('id', True), ('range_max', 0)]:
                    data = json.loads(json.dumps(valid))
                    data['shoulder_pan'][field] = value
                    path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ValueError, 'shoulder_pan'):
                        robot.require_calibration('robot')

    def test_calibration_lookup_lists_available_ids_without_hardware(self):
        from scripts.so101_homing import robot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'another_robot.json').write_text('{}')
            with patch.object(robot, 'calibration_directory', return_value=root), \
                 patch.object(robot, 'load_lerobot') as hardware:
                with self.assertRaisesRegex(FileNotFoundError, 'another_robot.*lerobot-calibrate'):
                    robot.require_calibration('missing')
                hardware.assert_not_called()

    def environment(self):
        return {'sim':{'dt':.01},'decimation':4, 'keyboard_profile':'mx_keys_powered_az_20260718', 'actuator_profile':'anchorbench',
                'task_contract':'test_task','target_reference_sha256':'map','target_manifest_sha256':'source',
                'commands':{'typing':{'q_reset_ref':[0]*5,'letter_xyz_b_m':[[.2+i*.001,.1,.01] for i in range(26)],
                    'target_xyz_mean_b_m':[.2,.1,.01],'target_xyz_std_b_m':[.01,.02,.001],'letter_length':[2,2]}},
                'actions':{'action':{'joint_names':list(ARM_JOINT_NAMES),'velocity_limits_rad_s':list(JOINT_RATE_LIMITS_RAD_S),'q_reset_ref':[0]*5}}}

    def test_geometry_and_convention_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'env.yaml';path.write_text(yaml.safe_dump(self.environment()))
            profile,_,stage=profile_from_environment(path,'lerobot')
            self.assertEqual(profile.joint_zero_offset_deg,(0.,)*5)
            self.assertEqual(profile.az_xyz_b_m[0],(.2,.1,.01))
            self.assertEqual(stage,'p1a')
            benchmark,_,_=profile_from_environment(path,'benchmark')
            self.assertNotEqual(profile.joint_zero_offset_deg,benchmark.joint_zero_offset_deg)

    def test_identity_hashes_and_profile_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=root/'env.yaml';env.write_text(yaml.safe_dump(self.environment()))
            cp=root/'checkpoint.pt';cp.write_bytes(b'test-checkpoint')
            profile,_,stage=profile_from_environment(env,'lerobot')
            data={'schema_version':1,'kind':'so101_typing_deployment_bundle','robot_id':'robot',
                  'encoder_convention':'lerobot','profile':asdict(profile),'stage':stage,'task_contract':'test_task',
                  'checkpoint_sha256':sha256(cp),'env_sha256':sha256(env)}
            manifest=root/'deployment.json';manifest.write_text(json.dumps(data))
            validate_manifest(manifest,cp,env,'robot')
            with self.assertRaises(ValueError):validate_manifest(manifest,cp,env,'other')
            data['profile']['az_xyz_b_m']=[[0,0,0]]*26;manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):validate_manifest(manifest,cp,env,'robot')
            cp.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_manifest(manifest,cp,env,'robot')

    def test_execute_requires_explicit_user_devices_before_loading_bundle(self):
        from scripts.so101_homing import deploy_bundle
        with patch('sys.argv', ['deploy_bundle', '/nonexistent', 'HE', '--execute']), \
             patch.object(deploy_bundle.subprocess, 'call') as run, self.assertRaises(SystemExit) as error:
            deploy_bundle.main()
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_unsupported_control_timing_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data=self.environment();data['sim']['dt']=.02
            path=Path(directory)/'env.yaml';path.write_text(yaml.safe_dump(data))
            with self.assertRaises(ValueError):profile_from_environment(path,'lerobot')

    def test_auto_mapping_requires_exact_checkpoint_and_environment_pair(self):
        checkpoint, environment = deployment_bundle.REFERENCE_PAIR
        self.assertEqual(deployment_bundle.resolve_convention(checkpoint, environment, 'auto'), 'benchmark')
        for pair in [('different', environment), (checkpoint, 'different')]:
            with self.assertRaisesRegex(ValueError, 'No automatic joint mapping'):
                deployment_bundle.resolve_convention(*pair, 'auto')
        # Explicit zero-offset comparisons remain available; existing bundles are not rewritten.
        self.assertEqual(deployment_bundle.resolve_convention(checkpoint, environment, 'lerobot'), 'lerobot')

    def export_fixture(self, root, convention='lerobot', measured=None):
        checkpoint = root/'model.pt'; checkpoint.write_bytes(b'test-checkpoint')
        env = root/'env.yaml'; env.write_text(yaml.safe_dump(self.environment()))
        calibration = root/'robot.json'; calibration.write_text('test-local-calibration')
        with patch.object(deployment_bundle, 'require_calibration', return_value=calibration), \
             patch.object(deployment_bundle, 'FixedCartesianPolicy') as policy:
            policy.return_value.sha256 = sha256(checkpoint)
            result = deployment_bundle.prepare(checkpoint, env, 'robot', convention, root/'bundle', measured)
        return result, calibration

    def test_unknown_auto_export_stops_before_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'No automatic joint mapping'):
                self.export_fixture(root, 'auto')
            self.assertFalse((root/'bundle').exists())

    def test_changed_motor_calibration_invalidates_new_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, calibration = self.export_fixture(root)
            bundle = root/'bundle'
            with patch.object(deployment_bundle, 'require_calibration', return_value=calibration):
                validate_manifest(bundle/'deployment.json', bundle/'checkpoint.pt', bundle/'env.yaml', 'robot')
                calibration.write_text('recalibrated')
                with self.assertRaisesRegex(ValueError, 'LeRobot calibration changed'):
                    validate_manifest(bundle/'deployment.json', bundle/'checkpoint.pt', bundle/'env.yaml', 'robot')
            self.assertFalse(data['hardware_validated'])

    def test_measured_mapping_exports_reset_and_revalidates_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offsets = [0, 2, -11, 1, 0]
            fit = joint_calibration.fit({'robot_id':'robot', 'arm_joint_names':list(ARM_JOINT_NAMES),
                'samples':[{'encoder_deg':[q]*5, 'model_deg':[q+d for d in offsets],
                            'measurement_reference':f'test-pose-{q}'} for q in (-15, 0, 20)]})
            source = root/'measured.json'; source.write_text(json.dumps(fit))
            data, calibration = self.export_fixture(root, 'auto', source)
            bundle = root/'bundle'
            self.assertEqual(data['encoder_convention'], 'measured')
            rest = json.loads((bundle/'rest_pose.json').read_text())
            self.assertEqual([rest['pose_deg'][j] for j in ARM_JOINT_NAMES], [-d for d in offsets])
            with patch.object(deployment_bundle, 'require_calibration', return_value=calibration):
                profile, _ = validate_manifest(bundle/'deployment.json', bundle/'checkpoint.pt', bundle/'env.yaml', 'robot')
                self.assertEqual(profile.joint_zero_offset_deg, tuple(offsets))
                # Altered measurement evidence is rejected even if the fitted offsets still agree.
                copied = bundle/'joint_calibration.json'
                fit['samples'][0]['measurement_reference'] = 'different-evidence'
                copied.write_text(json.dumps(fit))
                with self.assertRaisesRegex(ValueError, 'measured joint calibration differs'):
                    validate_manifest(bundle/'deployment.json', bundle/'checkpoint.pt', bundle/'env.yaml', 'robot')
