"""Validated, data-only Newton MJWarp overrides for public simulation experiments."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

JOINTS = {'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'}
SOLVER_FIELDS = {'iterations', 'ls_iterations', 'tolerance'}
ACTUATOR_FIELDS = {'stiffness', 'damping', 'armature', 'friction', 'dynamic_friction', 'viscous_friction',
                   'effort_limit_sim', 'velocity_limit_sim'}


def number(value, name, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < 0 or (positive and value == 0) or (integer and not isinstance(value, int)):
        raise ValueError(f'{name} must be a {"positive" if positive else "nonnegative"} {"integer" if integer else "number"}')


def load_config(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) - {'schema_version', 'solver', 'num_substeps', 'solver_parameters', 'actuator_parameters'}:
        raise ValueError('unknown physics configuration fields')
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1 or data.get('solver') != 'mjwarp':
        raise ValueError('requires schema_version=1 and solver=mjwarp')
    if 'num_substeps' in data:
        number(data['num_substeps'], 'num_substeps', positive=True, integer=True)
    for section, allowed in [('solver_parameters', SOLVER_FIELDS), ('actuator_parameters', ACTUATOR_FIELDS)]:
        values = data.get(section, {})
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f'unknown fields in {section}')
        for key, value in values.items():
            if isinstance(value, dict) and section == 'actuator_parameters':
                if set(value) != JOINTS:
                    raise ValueError(f'{key} must specify all six joint names')
                for joint, scalar in value.items():
                    number(scalar, f'{key}.{joint}', positive=key.endswith('_limit_sim'))
            else:
                number(value, key, positive=section == 'solver_parameters' or key.endswith('_limit_sim'),
                       integer=section == 'solver_parameters' and key != 'tolerance')
    return data


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def apply_config(env_cfg, data):
    """Apply validated overrides and give the experiment a distinct task contract."""
    if 'num_substeps' in data:
        env_cfg.sim.physics.num_substeps = data['num_substeps']
    for key, value in data.get('solver_parameters', {}).items():
        if not hasattr(env_cfg.sim.physics.solver_cfg, key):
            raise ValueError(f'active solver does not expose {key}')
        setattr(env_cfg.sim.physics.solver_cfg, key, value)
    parameters = data.get('actuator_parameters', {})
    actuators = env_cfg.scene.robot.actuators
    if parameters and len(actuators) != 1:
        raise ValueError('custom actuator parameters require one all-joint actuator group')
    for actuator in actuators.values():
        for key, value in parameters.items():
            setattr(actuator, key, deepcopy(value))
    env_cfg.task_contract += '_physics_' + fingerprint(data)
    return env_cfg
