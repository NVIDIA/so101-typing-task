import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/reinforcement_learning/rsl_rl'))
from public_task import load_config, load_perturbation, nominal_geometry, rotate_xyzw, evaluation_reset_metadata, apply_config


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
