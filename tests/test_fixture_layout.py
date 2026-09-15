"""Mechanical datum conversion checks, independent of robot access and simulation."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import fixture_layout as fixture
from scripts.reinforcement_learning.rsl_rl.public_task import load_config, nominal_geometry, rotate_xyzw

ROOT = Path(__file__).resolve().parents[1]


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.geometry = fixture.load_geometry()

    def test_exact_reference_measurements_recover_saved_pose(self):
        task, report = fixture.fit(self.geometry, fixture.landmarks(self.geometry, {}))
        np.testing.assert_allclose(task['keyboard']['position_m'], self.geometry['reference_position_m'], atol=1e-14)
        self.assertAlmostEqual(task['keyboard']['yaw_offset_deg'], 0)
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
            self.assertAlmostEqual(fitted['keyboard']['yaw_offset_deg'],yaw)

    def test_reject_units_mirrored_axis_and_inconsistent_height(self):
        reference = fixture.landmarks(self.geometry,{})
        cases = [{k:[v/1000 for v in p] for k,p in reference.items()},
                 {k:[p[0],-p[1],p[2]] for k,p in reference.items()},
                 {k:[p[0],p[1],p[2]+(5 if k=='Z' else 0)] for k,p in reference.items()}]
        for measured in cases:
            with self.assertRaisesRegex(ValueError,'residual'):
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
