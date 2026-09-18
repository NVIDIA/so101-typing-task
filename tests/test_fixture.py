"""Physical fixture datums, task geometry and physics configuration."""

import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import fixture_layout as fixture
from scripts.reinforcement_learning.rsl_rl.public_physics import (
    apply_config as apply_physics_config,
    fingerprint,
    load_config as load_physics_config,
)
from scripts.reinforcement_learning.rsl_rl.public_task import (
    apply_config,
    evaluation_reset_metadata,
    load_config,
    load_perturbation,
    nominal_geometry,
    rotate_xyzw,
)


# Physical datums and fixture drawings

ROOT = Path(__file__).resolve().parents[1]


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.geometry = fixture.load_geometry()

    def test_exact_reference_measurements_recover_saved_pose(self):
        task, report = fixture.fit(self.geometry, fixture.landmarks(self.geometry, {}))
        np.testing.assert_allclose(task['keyboard']['position_m'], self.geometry['reference_position_m'], atol=1e-14)
        for axis in ([1,0,0], [0,1,0], [0,0,1]):
            np.testing.assert_allclose(rotate_xyzw(task['keyboard']['rotation_xyzw'],axis),
                                      rotate_xyzw(self.geometry['reference_rotation_xyzw'],axis),atol=1e-14)
        self.assertLess(report['max_residual_mm'], 1e-10)

    def test_fit_matches_training_transform_including_height_and_yaw_pivot(self):
        position = self.geometry['reference_position_m']
        original = [k['rim_center_m'] for k in self.geometry['keys'] if k['slot'] in ('main_q','main_p','main_z')]
        labels = [k['label'] for k in self.geometry['keys'] if k['slot'] in ('main_q','main_p','main_z')]
        for yaw in (-37, 2, 91, 179):
            target = [position[0]+.015, position[1]-.024, position[2]+.012]
            _, _, moved, _ = nominal_geometry(position,self.geometry['reference_rotation_xyzw'],original,
                                               {'position_m':target,'yaw_offset_deg':yaw})
            measured = {k:[(v-self.geometry['datum_m'][i])*1000 for i,v in enumerate(p)]
                        for k,p in zip(labels,moved)}
            fitted, _ = fixture.fit(self.geometry,measured)
            np.testing.assert_allclose(fitted['keyboard']['position_m'],target,atol=1e-13)
            for key,point in fixture.landmarks(self.geometry,fitted).items():
                np.testing.assert_allclose(point,measured[key],atol=1e-10)

    def test_full_orientation_fit_and_drawing_preserve_native_key_geometry(self):
        # A level case retains the asset's native deck slope, replacing the old fitted tilt.
        for angle in (-90, 0, 180):
            task = {'keyboard': {'position_m':[.27,.02,.008],
                                 'rotation_xyzw':[0,0,float(np.sin(np.radians(angle)/2)),float(np.cos(np.radians(angle)/2))]}}
            measured = fixture.landmarks(self.geometry,task)
            fitted,report = fixture.fit(self.geometry,measured)
            np.testing.assert_allclose(fitted['keyboard']['position_m'],task['keyboard']['position_m'],atol=1e-14)
            self.assertLess(report['max_residual_mm'],1e-10)
            self.assertAlmostEqual(measured['Q'][2]-measured['Z'][2],4.332618525,places=6)
            for key in self.geometry['keys']:
                for point in key['outline_m']:
                    np.testing.assert_allclose(fixture.transform(point,self.geometry,fitted),
                                               fixture.transform(point,self.geometry,task),atol=1e-14)
            self.assertIn('<svg',fixture.drawing(self.geometry,fitted))

    def test_reject_units_mirrored_axis_and_inconsistent_dimensions(self):
        reference = fixture.landmarks(self.geometry,{})
        cases = [{k:[v/1000 for v in p] for k,p in reference.items()},
                 {k:[p[0],-p[1],p[2]] for k,p in reference.items()},
                 {k:[p[0]+(5 if k=='Z' else 0),p[1],p[2]] for k,p in reference.items()},
                 {k:reference['Q'] for k in reference}]
        for measured in cases:
            with self.assertRaisesRegex(ValueError,'residual|mirrored|non-collinear'):
                fixture.fit(self.geometry,measured)

    def test_blank_or_malformed_measurements_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.json'; out=Path(folder)/'output'
            for value in ([], {}, {k:[None,None,None] for k in fixture.LANDMARKS},
                          {k:[True,1,2] for k in fixture.LANDMARKS},
                          {k:[float('nan'),1,2] for k in fixture.LANDMARKS},
                          {k:[1e300,1,2] for k in fixture.LANDMARKS}):
                source.write_text(json.dumps(value))
                with self.assertRaises(ValueError): fixture.create(out,measurements=source)
                self.assertFalse(out.exists())

    def test_export_is_valid_task_and_never_overwrites_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'layout'
            fixture.create(out)
            config=load_config(out/'task.json')
            self.assertEqual(config['schema_version'],1)
            self.assertEqual(json.loads((out/'measurement-template.json').read_text())['Q'],[None]*3)
            before={p.name:p.read_bytes() for p in out.iterdir()}
            with self.assertRaises(FileExistsError): fixture.create(out)
            self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})

    def test_default_draws_september_physical_reference_and_preserves_geometry_provenance(self):
        reference = ROOT / 'configs/tasks/physical-reference-20260915.json'
        with tempfile.TemporaryDirectory() as folder:
            default = Path(folder) / 'default'
            explicit = Path(folder) / 'explicit'
            fixture.create(default)
            fixture.create(explicit, task_config=reference)
            self.assertEqual(load_config(default / 'task.json'), load_config(reference))
            for name in ('task.json', 'expected-landmarks.json', 'birdseye.svg'):
                self.assertEqual((default / name).read_bytes(), (explicit / name).read_bytes())
            report = json.loads((default / 'measurement-report.json').read_text())
            self.assertEqual(report['source_sha256'], self.geometry['source_sha256'])
            self.assertEqual(report['robot_usd_sha256'], self.geometry['robot_usd_sha256'])
            self.assertFalse(report['hardware_access'])
            self.assertFalse(report['physical_alignment_validated'])

    def test_explicit_and_measured_fixtures_do_not_depend_on_default_reference_file(self):
        task_path = ROOT / 'configs/tasks/replay-calibrated-20260917.json'
        task = load_config(task_path)
        measured = fixture.landmarks(self.geometry, task)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            measurement_path = root / 'measurements.json'
            measurement_path.write_text(json.dumps(measured))
            with patch.object(fixture, 'ROOT', root), \
                 patch.object(fixture, 'load_geometry', return_value=self.geometry):
                fixture.create(root / 'explicit', task_config=task_path)
                fixture.create(root / 'measured', measurements=measurement_path)
            self.assertEqual(load_config(root / 'explicit/task.json'), task)
            recovered = json.loads((root / 'measured/expected-landmarks.json').read_text())
            for key in fixture.LANDMARKS:
                np.testing.assert_allclose(recovered[key], measured[key], atol=1e-10)

    def test_derived_cap_landmarks_match_actual_npz_geometry(self):
        asset_path=next(name for name in self.geometry['source_sha256'] if name.endswith('.npz'))
        asset=np.load(ROOT/asset_path)
        manifest=json.loads(asset['manifest_json'].tobytes().decode())
        for key in self.geometry['keys']:
            slot=manifest['slot_names'].index(key['slot'])
            offset=rotate_xyzw(asset['deck_quat_xyzw'],[0,0,float(asset['keys_size'][slot,2])/2])
            local=[float(asset['keys_center'][slot,i])+offset[i] for i in range(3)]
            rotated=rotate_xyzw(self.geometry['reference_rotation_xyzw'],local)
            expected=[v+self.geometry['reference_position_m'][i] for i,v in enumerate(rotated)]
            np.testing.assert_allclose(key['rim_center_m'],expected,atol=1e-14)
        self.assertIn(self.geometry['robot_usd_sha256'],(ROOT/'docker/typing-public.Dockerfile').read_text())

    def test_stale_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'docs/fixtures').mkdir(parents=True)
            (root/'changed.py').write_text('changed')
            data={'source_sha256':{'changed.py':'invalid'}}
            (root/'docs/fixtures/model_geometry.json').write_text(json.dumps(data))
            with patch.object(fixture,'ROOT',root),self.assertRaisesRegex(ValueError,'stale'):
                fixture.load_geometry()

    def test_cli_fixture_requires_no_subprocess(self):
        loader=importlib.machinery.SourceFileLoader('fixture_cli',str(ROOT/'so101'))
        spec=importlib.util.spec_from_loader(loader.name,loader)
        cli=importlib.util.module_from_spec(spec);loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as folder, \
             patch('sys.argv',['so101','fixture','--out-dir',str(Path(folder)/'layout')]), \
             patch.object(cli.subprocess,'call') as call,patch.object(cli.subprocess,'check_output') as check:
            self.assertEqual(cli.main(),0)
            call.assert_not_called();check.assert_not_called()


# Task geometry and randomization

class PublicTaskTests(unittest.TestCase):
    def test_robustness_cannot_move_the_nominal_actor_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'perturbation.json'
            path.write_text(json.dumps({'schema_version': 1, 'keyboard': {'yaw_offset_deg': 1}}))
            with self.assertRaises(ValueError):
                load_perturbation(path)
            path.write_text(json.dumps({'schema_version': 1, 'domain_randomization': {'keyboard_x_m': [-.001, .001]}}))
            self.assertEqual(load_perturbation(path)['domain_randomization']['keyboard_x_m'], [-.001, .001])

    def test_evaluation_reports_resolved_randomization(self):
        cfg = SimpleNamespace(events=SimpleNamespace(
            reset_robot_rest=SimpleNamespace(params={'position_range': (-.005, .005), 'velocity_range': (0, 0)}),
            reset_keyboard=SimpleNamespace(params={'x_range_m': (-.003, .003), 'yaw_range_rad': (0, 0)})))
        report = evaluation_reset_metadata(cfg)
        self.assertFalse(report['exact_model_rest'])
        self.assertTrue(report['zero_joint_velocity'])
        self.assertTrue(report['keyboard_pose_randomization'])
        cfg.events.reset_robot_rest.params['position_range'] = (0, 0)
        cfg.events.reset_keyboard.params['x_range_m'] = (0, 0)
        report = evaluation_reset_metadata(cfg)
        self.assertTrue(report['exact_model_rest'])
        self.assertFalse(report['keyboard_pose_randomization'])

    def test_world_yaw_and_translation_move_map_with_fixture(self):
        pos, rot, points, local = nominal_geometry([1,2,3], [0,0,0,1], [[2,2,3]],
                                                   {'position_m':[4,5,6], 'yaw_offset_deg':90})
        for actual, expected in zip(points[0], [4,6,6]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(local, [[1,0,0]])
        restored = [a+b for a,b in zip(rotate_xyzw(rot,local[0]),pos)]
        for a,b in zip(restored,points[0]):self.assertAlmostEqual(a,b)

    def test_tilted_keyboard_round_trip_and_identity(self):
        q=[math.sin(.2),0,0,math.cos(.2)]
        pos, rot, points, local = nominal_geometry([.2,.1,.01],q,[[.21,.13,.02]],{})
        restored = [a+b for a,b in zip(rotate_xyzw(rot,local[0]),pos)]
        for a,b in zip(restored,points[0]):self.assertAlmostEqual(a,b)

    def test_absolute_rotation_replaces_reference_tilt_and_keeps_local_map(self):
        old = [math.sin(.2),0,0,math.cos(.2)]
        origin = [.2,.1,.01]
        point = [a+b for a,b in zip(origin,rotate_xyzw(old,[0,.1,.02]))]
        pos,rot,points,local = nominal_geometry(origin,old,[point],
            {'position_m':[.3,.2,.03], 'rotation_xyzw':[0,0,0,1]})
        self.assertEqual(rot,[0,0,0,1])
        for a,b in zip(points[0],[.3,.3,.05]):self.assertAlmostEqual(a,b)
        for a,b in zip(local[0],[0,.1,.02]):self.assertAlmostEqual(a,b)

    def test_absolute_pose_places_support_under_case_and_retains_reset_local_geometry(self):
        import numpy as np
        root=Path(__file__).resolve().parents[1]
        asset_path=root/'source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/keyboards/data/mx_keys/mx_keys_asset.npz'
        with np.load(asset_path) as asset:
            case=asset['case_points_0']
        cfg=SimpleNamespace(scene=SimpleNamespace(
            keyboard=SimpleNamespace(init_state=SimpleNamespace(pos=(.2,.1,.01),rot=(0,0,0,1))),
            plane=SimpleNamespace(init_state=SimpleNamespace(pos=(0,0,-.01)))),
            commands=SimpleNamespace(typing=SimpleNamespace(letter_xyz_b_m=((.2,.11,.02),))),
            events=SimpleNamespace(reset_keyboard=SimpleNamespace(),reset_robot_rest=SimpleNamespace(params={})),
            task_contract='test')
        rotation=[0,0,-math.sqrt(.5),math.sqrt(.5)]
        task={'schema_version':1,'keyboard':{'position_m':[.27,.01,.03],'rotation_xyzw':rotation}}
        module='isaaclab_tasks.core.dexsuite.config.so101.mdp.public_events'
        with patch.dict(sys.modules,{module:SimpleNamespace(reset_public_keyboard=object())}):
            apply_config(cfg,task)
        self.assertAlmostEqual(cfg.scene.plane.init_state.pos[2],.03+float(case[:,2].min()))
        for a,b in zip(cfg.events.reset_keyboard.params['letter_xyz_keyboard_m'][0],[0,.01,.01]):
            self.assertAlmostEqual(a,b)
        for a,b in zip(cfg.commands.typing.letter_xyz_b_m[0],[.28,.01,.04]):
            self.assertAlmostEqual(a,b)

    def test_bad_pose_and_randomization_rejected(self):
        for extra in [{'keyboard':{'position_m':[0,0]}}, {'keyboard':{'yaw_offset_deg':float('nan')}},
                      *({'keyboard':{'rotation_xyzw':q}} for q in ([0,0,0], [0,0,0,0], [0,0,0,2],
                          [0,0,0,True], [0,0,0,float('nan')])),
                      {'keyboard':{'rotation_xyzw':[0,0,0,1], 'yaw_offset_deg':0}},
                      {'domain_randomization':{'keyboard_x_m':[1,-1]}},
                      {'domain_randomization':{'reset_joint_noise_rad':-1}},
                      {'domain_randomization':{'keyboard_z_m':[-.01,.01]}}]:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'task.json';path.write_text(json.dumps({'schema_version':1,**extra}))
                with self.assertRaises(ValueError):load_config(path)


# Physics configuration

class PhysicsTests(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'physics.json'
            path.write_text(json.dumps(data))
            return load_physics_config(path)

    def test_invalid_physics_fails_closed(self):
        for extra in [{'solver': 'physx'}, {'num_substeps': 0}, {'num_substeps': True},
                      {'solver_parameters': {'iterations': 2.5}},
                      {'actuator_parameters': {'stiffness': {'shoulder_pan': 10}}},
                      {'actuator_parameters': {'friction': float('nan')}},
                      {'unrecognized': 5}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.load({'schema_version': 1, 'solver': 'mjwarp', **extra})

    def test_overrides_do_not_change_unrequested_fields(self):
        data = self.load({'schema_version': 1, 'solver': 'mjwarp', 'num_substeps': 4,
                          'solver_parameters': {'iterations': 50}, 'actuator_parameters': {'stiffness': 80}})
        actuator = SimpleNamespace(stiffness=100, damping=1)
        env = SimpleNamespace(task_contract='original', sim=SimpleNamespace(physics=SimpleNamespace(num_substeps=2, solver_cfg=SimpleNamespace(iterations=100))),
                  scene=SimpleNamespace(robot=SimpleNamespace(actuators={'all': actuator})))
        apply_physics_config(env, data)
        self.assertEqual(env.sim.physics.num_substeps, 4)
        self.assertEqual(env.sim.physics.solver_cfg.iterations, 50)
        self.assertEqual(actuator.stiffness, 80)
        self.assertEqual(actuator.damping, 1)
        self.assertEqual(env.task_contract, 'original_physics_' + fingerprint(data))

    def test_fingerprint_ignores_json_key_order(self):
        self.assertEqual(fingerprint({'a': 1, 'b': 2}), fingerprint({'b': 2, 'a': 1}))
