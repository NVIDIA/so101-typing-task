"""Prepare and validate portable typing deployment inputs without opening hardware."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil

import yaml
from .constants import ARM_JOINT_NAMES
from .robot import require_calibration
from .fixed_cartesian_policy import (PHYSICAL_CALIBRATED_20260718_PROFILE, JOINT_RATE_LIMITS_RAD_S,
                                     FixedCartesianPolicy)
from . import joint_calibration

# Exact released checkpoint/environment pair; names and task labels are not identity.
REFERENCE_PAIR = (
    '239d93c99cd282e9add019f4ac952407c97fde2879821103a2aa7bd7a307a15f',
    'd4766dacb28a4ffc0fc75efe8e5abed162e656b338b110e8d498cc808e31ef8b',
)


def resolve_convention(checkpoint_hash, environment_hash, requested):
    if requested in ('lerobot', 'benchmark'):
        return requested
    if requested != 'auto':
        raise ValueError('unknown encoder convention')
    if (checkpoint_hash, environment_hash) == REFERENCE_PAIR:
        return 'benchmark'
    raise ValueError('No automatic joint mapping for this checkpoint/environment pair. '
                     'After checking physical/model alignment, supply --joint-calibration FILE '
                     'or explicitly select --encoder-convention lerobot or benchmark. '
                     'Training does not measure your robot alignment.')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def floats(value, length):
    result = tuple(float(x) for x in value)
    if len(result) != length or not all(math.isfinite(x) for x in result):
        raise ValueError(f'expected {length} finite values')
    return result


def profile_from_environment(path, encoder_convention, measured_offsets=None):
    env = yaml.load(Path(path).read_text(), Loader=yaml.BaseLoader)
    if not isinstance(env, dict):
        raise ValueError('environment configuration must be a mapping')
    cfg = env['commands']['typing']
    action = env['actions']['action']
    if not math.isclose(float(env['sim']['dt'])*int(env['decimation']), .04, abs_tol=1e-10):
        raise ValueError('deployment requires the trained 25 Hz control contract')
    if floats(action['velocity_limits_rad_s'],5) != tuple(JOINT_RATE_LIMITS_RAD_S):
        raise ValueError('unsupported joint-rate envelope')
    if tuple(action['joint_names']) != tuple(ARM_JOINT_NAMES):
        raise ValueError('unsupported controlled joint order')
    q_reset = floats(cfg['q_reset_ref'],5)
    if q_reset != floats(action['q_reset_ref'],5):
        raise ValueError('action and observation reset references differ')
    if encoder_convention not in ('lerobot', 'benchmark', 'measured', 'aligned'):
        raise ValueError('unknown encoder convention')
    if encoder_convention in ('measured', 'aligned'):
        offsets = floats(measured_offsets, 5)
    else:
        offsets = (0.,)*5 if encoder_convention == 'lerobot' else PHYSICAL_CALIBRATED_20260718_PROFILE.joint_zero_offset_deg
    points = tuple(floats(row,3) for row in cfg['letter_xyz_b_m'])
    if len(points) != 26:
        raise ValueError('requires the A-Z Cartesian map')
    std = floats(cfg['target_xyz_std_b_m'],3)
    if min(std) <= 0:
        raise ValueError('target normalization scales must be positive')
    length = tuple(int(x) for x in cfg['letter_length'])
    stage = {(1,1):'p0',(2,2):'p1a',(3,3):'p1b',(4,4):'p1c',(6,6):'p1d'}.get(length)
    if stage is None:
        raise ValueError('unsupported target length')
    profile = replace(PHYSICAL_CALIBRATED_20260718_PROFILE,
        calibration_id=env['keyboard_profile'], q_reset_rad=q_reset, az_xyz_b_m=points,
        az_xyz_mean_b_m=floats(cfg['target_xyz_mean_b_m'],3), az_xyz_std_b_m=std,
        target_reference_sha256=env['target_reference_sha256'],
        target_manifest_sha256=env['target_manifest_sha256'], joint_zero_offset_deg=offsets)
    return profile, env, stage


def validate_manifest(path, checkpoint, env_config, robot_id):
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') != 1 or data.get('kind') != 'so101_typing_deployment_bundle':
        raise ValueError('unsupported deployment bundle')
    if data.get('robot_id') != robot_id:
        raise ValueError('bundle robot_id differs from the selected LeRobot calibration ID')
    if sha256(checkpoint) != data.get('checkpoint_sha256') or sha256(env_config) != data.get('env_sha256'):
        raise ValueError('checkpoint/environment hash differs from prepared deployment bundle')
    offsets = None
    alignment = None
    if data['encoder_convention'] == 'aligned':
        from .scene_alignment import load_qualified
        qualification = Path(path).parent/'alignment/qualification.json'
        if sha256(qualification) != data.get('alignment_sha256'):
            raise ValueError('alignment qualification differs from prepared bundle')
        alignment, _, _ = load_qualified(qualification, robot_id)
        offsets = alignment['joint_zero_offset_deg']
    if data['encoder_convention'] == 'measured':
        offsets, digest = joint_calibration.load(Path(path).parent/'joint_calibration.json', robot_id)
        if digest != data.get('joint_calibration_sha256'):
            raise ValueError('measured joint calibration differs from prepared bundle')
    if 'lerobot_calibration_sha256' in data:
        if sha256(require_calibration(robot_id)) != data['lerobot_calibration_sha256']:
            raise ValueError('LeRobot calibration changed since export; verify joint alignment and prepare a new bundle')
    profile, env, stage = profile_from_environment(env_config, data['encoder_convention'], offsets)
    if json.loads(json.dumps(asdict(profile))) != data.get('profile'):
        raise ValueError('bundle geometry differs from the archived training environment')
    if env['task_contract'] != data.get('task_contract') or stage != data.get('stage'):
        raise ValueError('bundle task or stage differs from training')
    if alignment:
        from .scene_alignment import validate_environment
        validate_environment(alignment, profile, env)
    return profile, data


def require_qualified_alignment(data):
    if data.get('encoder_convention') != 'aligned' or not data.get('alignment_sha256'):
        raise ValueError('Physical execution requires a qualified scene alignment. Complete reserved replay and Q/G/P probes, '
                         'then re-export with --alignment qualification.json. Software dry runs remain available.')


def prepare(checkpoint, env_config, robot_id, encoder_convention, out_dir, joint_calibration_path=None, alignment_path=None):
    if not robot_id.strip():
        raise ValueError('robot_id must be nonempty and match the LeRobot calibration ID')
    calibration_hash = sha256(require_calibration(robot_id))
    offsets = calibration_digest = None
    alignment_data = evidence = None
    if alignment_path:
        if encoder_convention != 'auto' or joint_calibration_path:
            raise ValueError('--alignment cannot be combined with other mapping selections')
        from .scene_alignment import load_qualified, validate_environment
        contract, alignment_data, evidence = load_qualified(alignment_path, robot_id)
        offsets = contract['joint_zero_offset_deg']
        convention = 'aligned'
    elif joint_calibration_path:
        if encoder_convention != 'auto':
            raise ValueError('--joint-calibration and --encoder-convention cannot be combined')
        offsets, calibration_digest = joint_calibration.load(joint_calibration_path, robot_id)
        convention = 'measured'
    else:
        convention = resolve_convention(sha256(checkpoint), sha256(env_config), encoder_convention)
    profile, env, stage = profile_from_environment(env_config, convention, offsets)
    if alignment_data:
        validate_environment(contract, profile, env)
    policy = FixedCartesianPolicy(checkpoint)  # Validate supported checkpoint schema on CPU.
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint, out_dir/'checkpoint.pt')
    shutil.copy2(env_config, out_dir/'env.yaml')
    if joint_calibration_path:
        shutil.copy2(joint_calibration_path, out_dir/'joint_calibration.json')
    if alignment_data:
        (out_dir/'alignment').mkdir()
        shutil.copy2(alignment_path, out_dir/'alignment/qualification.json')
        for name, source in evidence.items():
            destination = out_dir/'alignment'/alignment_data['evidence'][name]['file']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for name in ('public_task.json','public_physics.json'):
        candidate = env_config.parent/name
        if candidate.is_file():shutil.copy2(candidate,out_dir/name)
    manifest = {'schema_version':1,'kind':'so101_typing_deployment_bundle','robot_id':robot_id,
                'encoder_convention':convention,'profile':asdict(profile), 'stage':stage,
                'task_contract':env['task_contract'],'actuator_profile':env['actuator_profile'],
                'checkpoint_sha256':policy.sha256,'env_sha256':sha256(env_config),
                'lerobot_calibration_sha256':calibration_hash,
                'mapping_selection':'qualified_alignment' if alignment_data else 'measured' if joint_calibration_path else encoder_convention,
                'hardware_validated':False}
    if alignment_data:
        manifest['alignment_sha256'] = sha256(alignment_path)
        manifest['alignment_qualified'] = True
        manifest['alignment_scope'] = alignment_data['scope']
    if calibration_digest:
        manifest['joint_calibration_sha256'] = calibration_digest
    (out_dir/'deployment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rest = {'schema_version':1,'kind':'so101_policy_rest_pose','robot_id':robot_id,
            'arm_joint_names':list(ARM_JOINT_NAMES),'closed_fixed_jaw_required':True,'contact_required':False,
            'pose_deg':{joint:math.degrees(q)-offset for joint,q,offset in
                        zip(ARM_JOINT_NAMES,profile.q_reset_rad,profile.joint_zero_offset_deg)},
            'source':'software-derived reset target; not a measured hardware pose'}
    (out_dir/'rest_pose.json').write_text(json.dumps(rest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--env-config',type=Path,required=True)
    parser.add_argument('--robot-id',required=True)
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument('--encoder-convention',choices=('auto','lerobot','benchmark'),default='auto')
    mapping.add_argument('--joint-calibration',type=Path,help='Measured encoder-to-model fit from joint_calibration')
    mapping.add_argument('--alignment', type=Path, help='qualified alignment with portable Q/G/P and replay evidence')
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    try:
        manifest = prepare(args.checkpoint,args.env_config,args.robot_id,args.encoder_convention,args.out_dir,
                           args.joint_calibration, args.alignment)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Joint mapping: {manifest['encoder_convention']} ({manifest['mapping_selection']})")
    print(f"Offsets (degrees): {manifest['profile']['joint_zero_offset_deg']}")
    print(f'Deployment bundle: {args.out_dir.resolve()}')


if __name__=='__main__': main()
