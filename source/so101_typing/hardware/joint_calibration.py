# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline encoder-to-model calibration fitting. Never connects to a robot."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from .constants import ARM_JOINT_NAMES


def vector(value):
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError('expected five joint angles in degrees')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('joint angles must be finite numbers')
    return value


def fit(data, maximum_residual_deg=0.5):
    if not isinstance(data, dict):
        raise ValueError('joint measurements must be a JSON object')
    if not isinstance(data.get('robot_id'), str) or not data['robot_id'].strip():
        raise ValueError('robot_id is required')
    if data.get('arm_joint_names') != list(ARM_JOINT_NAMES):
        raise ValueError('arm joint order must match the deployment contract')
    samples = data.get('samples', [])
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError('at least three independently measured correspondence poses are required')
    differences = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError('each correspondence pose must be a JSON object')
        if not sample.get('measurement_reference'):
            raise ValueError('each pose needs a measurement_reference identifying the jig/metrology evidence')
        model, encoder = vector(sample['model_deg']), vector(sample['encoder_deg'])
        differences.append([m - e for m, e in zip(model, encoder)])
    offsets = [mean(row[i] for row in differences) for i in range(5)]
    residual = max(abs(row[i] - offsets[i]) for row in differences for i in range(5))
    if residual > maximum_residual_deg:
        raise ValueError(f'offset-only mapping rejected: residual {residual:.6f} deg exceeds {maximum_residual_deg}')
    return {'schema_version': 1, 'kind': 'so101_encoder_to_model_calibration',
            'robot_id': data['robot_id'], 'arm_joint_names': list(ARM_JOINT_NAMES),
            'joint_zero_offset_deg': offsets, 'max_residual_deg': residual,
            'maximum_residual_deg': maximum_residual_deg, 'samples': samples,
            'mapping': 'q_model_deg = q_encoder_deg + joint_zero_offset_deg',
            'hardware_geometry_validated': False}


def load(path, robot_id):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError('joint calibration must be a JSON object')
    if data.get('kind') != 'so101_encoder_to_model_calibration' or data.get('schema_version') != 1:
        raise ValueError('unsupported joint calibration artifact')
    if data.get('robot_id') != robot_id:
        raise ValueError('joint calibration robot_id does not match selected robot')
    verified = fit(data)
    offsets = vector(data['joint_zero_offset_deg'])
    if any(abs(a-b) > 1e-9 for a,b in zip(offsets,verified['joint_zero_offset_deg'])):
        raise ValueError('stored offsets do not match measurement evidence')
    return tuple(offsets), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = fit(json.loads(args.measurements.read_text()))
        # Never replace an existing calibration silently.
        with args.out.open('x') as output:
            json.dump(result, output, indent=2)
            output.write('\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(args.out.resolve())


if __name__ == '__main__':
    main()
