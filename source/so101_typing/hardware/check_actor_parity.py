"""Compare recorded simulator actions to the CPU deployment actor; no robot connection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .fixed_cartesian_policy import FixedCartesianPolicy


def check(checkpoint, report):
    policy = FixedCartesianPolicy(checkpoint)
    data = json.loads(report.read_text())
    if data['checkpoint_sha256'] != policy.sha256:
        raise ValueError('evaluation and deployment checkpoint hashes differ')
    rows = data.get('raw_policy_action_trace', [])
    if not rows:
        raise ValueError('evaluation needs --trace-policy-actions to record parity evidence')
    errors = []
    for row in rows:
        reference = np.asarray(row['action_env0'], dtype=np.float64)
        if reference.shape != (5,) or not np.isfinite(reference).all():
            raise ValueError('invalid recorded simulator action')
        actual = policy.action(np.asarray(row['observation_env0']))['action']
        errors.append(float(np.max(np.abs(actual - reference))))
    return {'kind': 'simulation_to_cpu_actor_parity', 'checkpoint_sha256': policy.sha256,
            'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
            'samples': len(rows), 'maximum_absolute_action_error': max(errors),
            'tolerance': 1e-5, 'passed': max(errors) <= 1e-5, 'hardware_connected': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = check(args.checkpoint, args.report)
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
