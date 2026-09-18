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
