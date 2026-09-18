from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from so101_typing.hardware.joint_calibration import fit, load
from so101_typing.hardware.constants import ARM_JOINT_NAMES


class JointCalibrationTests(unittest.TestCase):
    def measurements(self):
        return {'robot_id': 'test_robot', 'arm_joint_names': list(ARM_JOINT_NAMES),
                'samples': [{'model_deg': [angle+offset for offset in [1,2,3,4,5]],
                             'encoder_deg': [angle]*5, 'measurement_reference': f'test-jig-{angle}'}
                            for angle in [-10, 0, 10]]}

    def test_offset_direction_and_identity(self):
        result = fit(self.measurements())
        self.assertEqual(result['joint_zero_offset_deg'], [1,2,3,4,5])
        self.assertFalse(result['hardware_geometry_validated'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calibration.json'
            path.write_text(json.dumps(result))
            self.assertEqual(load(path, 'test_robot')[0], (1,2,3,4,5))
            with self.assertRaises(ValueError):
                load(path, 'different_robot')
            result['joint_zero_offset_deg'][0] += 1
            path.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                load(path, 'test_robot')

    def test_nonconstant_mapping_and_missing_evidence_rejected(self):
        for bad in ['angle', 'evidence']:
            data = deepcopy(self.measurements())
            if bad == 'angle':
                data['samples'][0]['model_deg'][0] += 5
            else:
                del data['samples'][0]['measurement_reference']
            with self.assertRaises(ValueError):
                fit(data)

    def test_malformed_measurements_and_artifacts_raise_readable_errors(self):
        for data in (None, [], {**self.measurements(), 'samples': None},
                     {**self.measurements(), 'samples': [None]*3}):
            with self.assertRaises(ValueError):
                fit(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calibration.json'
            for data in (None, []):
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'JSON object'):
                    load(path, 'test_robot')
