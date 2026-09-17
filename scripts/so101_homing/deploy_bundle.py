"""Run a prepared typing bundle. Defaults to software dry-run; operator runs --execute."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml
from .robot import require_calibration
from .deployment_bundle import validate_manifest, require_qualified_alignment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path)
    p.add_argument('target')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--port')
    p.add_argument('--keyboard-device')
    a=p.parse_args()
    if a.execute and (not a.port or not a.keyboard_device):
        p.error('--execute requires your --port and --keyboard-device explicitly')
    bundle=a.bundle.resolve()
    try:
        data=json.loads((bundle/'deployment.json').read_text())
        if not isinstance(data, dict):
            raise ValueError('deployment.json must be a mapping')
        _, data = validate_manifest(bundle/'deployment.json', bundle/'checkpoint.pt',
                                    bundle/'env.yaml', data['robot_id'])
        if not isinstance(data.get('actuator_profile'), str):
            raise ValueError('deployment.json requires an actuator_profile')
        require_calibration(data['robot_id'])
        if a.execute:
            require_qualified_alignment(data)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        p.error(str(error))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    command=[sys.executable,'-m','scripts.so101_homing.run_fixed_cartesian_policy_handoff',
             '--deployment-config',str(bundle/'deployment.json'),'--checkpoint',str(bundle/'checkpoint.pt'),
             '--env-config',str(bundle/'env.yaml'),'--rest-pose',str(bundle/'rest_pose.json'),
             '--stage',data['stage'],'--id',data['robot_id'],'--target',a.target,
             '--expected-actuator-profile',data['actuator_profile'],
             '--out',str(bundle/'runs'/(stamp + ('_hardware.jsonl' if a.execute else '_dry_run.jsonl'))),
             '--enable-robot' if a.execute else '--dry-run']
    if a.port:command+=['--port',a.port]
    if a.keyboard_device:command+=['--device',a.keyboard_device]
    return subprocess.call(command)


if __name__=='__main__':sys.exit(main())
