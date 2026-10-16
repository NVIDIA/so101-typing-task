"""Machine-local device selection and calibration for tutorial deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def load_profile(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported hardware profile schema')
    for name in ('robot_id','port','keyboard','wrist_camera'):
        if not isinstance(data.get(name),str) or not data[name]:
            raise ValueError(f'Missing profile field {name}')
    for name in ('mapping_dir','post_eval_home'):
        value = data.get(name)
        if value:
            data[name] = str((path.parent / value).resolve())
    return data


def keyboard_identity(path):
    from evdev import InputDevice
    with InputDevice(path) as device:
        return {'name':device.name,'vendor':device.info.vendor,'product':device.info.product,'uniq':device.uniq}


def apply_profile(args):
    data = load_profile(args.hardware_profile)
    if not data.get('mapping_dir') or not data.get('post_eval_home'):
        raise ValueError('Profile needs a calibrated mapping_dir and saved post_eval_home before deployment')
    for key in ('port','keyboard','wrist_camera'):
        if not Path(data[key]).exists():
            raise ValueError(f'Profile device missing: {key}={data[key]}')
    if keyboard_identity(data['keyboard']) != data['keyboard_identity']:
        raise ValueError('Selected keyboard identity changed; rerun hardware configure')
    args.id, args.port = data['robot_id'], data['port']
    args.device, args.wrist_device = data['keyboard'], data['wrist_camera']
    args.external_url = data.get('external_camera') or ''
    args.no_external_camera = not bool(args.external_url)
    args.mapping_dir = Path(data['mapping_dir'])
    args.post_eval_home = Path(data['post_eval_home'])
    # A saved node requirement is optional; this workstation can pin event9.
    args.required_keyboard_node = data.get('required_keyboard_node')


def select_path(label, candidates, optional=False):
    print(f'\n{label}:')
    for index, path in enumerate(candidates,1):
        print(f'  {index}: {path}')
    answer = input('Number or full device path' + (' (blank to omit)' if optional else '') + ': ').strip()
    if not answer and optional:
        return ''
    value = str(candidates[int(answer)-1]) if answer.isdigit() and 1 <= int(answer) <= len(candidates) else answer
    if not value or not Path(value).exists():
        raise ValueError(f'Device does not exist: {value}')
    return value


def configure(path):
    from .robot import discover_robot_ports
    from evdev import InputDevice
    import cv2
    path = Path(path).resolve()
    if path.exists() and input(f'Replace {path}? Type replace: ').strip() != 'replace':
        raise ValueError('Profile left unchanged')
    robot_id = input('LeRobot follower calibration ID: ').strip()
    if not robot_id or Path(robot_id).name != robot_id or robot_id in ('.','..'):
        raise ValueError('Use a simple robot calibration ID')
    port = select_path('Follower serial port',discover_robot_ports())
    cameras = sorted(Path('/dev/v4l/by-id').glob('*video-index0'))
    if not cameras:
        cameras = sorted(Path('/dev/v4l/by-path').glob('*video-index0'))
    wrist = select_path('Wrist camera (must show the gripper)',cameras)
    external = select_path('External camera',cameras,optional=True)
    if external and os.path.samefile(wrist,external):
        raise ValueError('Wrist and external cameras must be different devices')
    keyboards = sorted(Path('/dev/input/by-id').glob('*event-kbd'))
    keyboards += sorted(Path('/dev/input/by-path').glob('*event-kbd'))
    keyboard = select_path('Keyboard the robot will press',keyboards)
    identity = keyboard_identity(keyboard)
    print(f"Press a key on {identity['name']} within 30 seconds to confirm selection.",flush=True)
    with InputDevice(keyboard) as device:
        deadline = time.monotonic()+30
        confirmed=False
        while time.monotonic()<deadline:
            ready,_,_=select.select([device.fd],[],[],1)
            if ready and any(e.type==1 and e.value==1 for e in device.read()):
                confirmed=True
                break
        if not confirmed:
            raise ValueError('No key detected on selected keyboard')
    path.parent.mkdir(parents=True,exist_ok=True)
    previews=[]
    for name,source in (('wrist',wrist),('external',external)):
        if not source:
            continue
        capture=cv2.VideoCapture(source,cv2.CAP_V4L2)
        try:
            ok,frame=capture.read()
            if not ok:
                raise ValueError(f'No frame from {source}')
            preview=path.parent/f'{name}_setup.jpg'
            cv2.imwrite(str(preview),frame)
            previews.append(preview)
        finally:
            capture.release()
    html=path.parent/'camera_setup.html'
    html.write_text('<html><h1>Confirm camera identities</h1>'+''.join(f'<h2>{p.stem}</h2><img src="{p.name}" width="640">' for p in previews)+'</html>')
    if os.environ.get('DISPLAY'):
        subprocess.run(['xdg-open',str(html)],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print(f'Camera previews: {html}')
    if input('Type yes if the wrist preview is from the arm: ').strip().lower() != 'yes':
        raise ValueError('Camera identities not confirmed')
    mapping=input('Existing calibrated deployment mapping directory (blank to configure later): ').strip()
    data={'schema_version':1,'robot_id':robot_id,'port':port,'keyboard':keyboard,'keyboard_identity':identity,
          'wrist_camera':wrist,'external_camera':external,'mapping_dir':str(Path(mapping).resolve()) if mapping else None,
          'post_eval_home':'post_eval_home.json','required_keyboard_node':None}
    path.write_text(json.dumps(data,indent=2)+'\n')
    print(f'Saved {path}. Next: hardware calibrate, calibrate the scene/mapping, then hardware save-rest.')


def save_rest(path):
    from .robot import connect_robot, require_calibration
    from .constants import ARM_JOINT_NAMES
    data=load_profile(path)
    if input('Place follower at the desired rest pose with torque OFF. Type save to read it: ').strip()!='save':
        raise ValueError('Rest pose not saved')
    robot=connect_robot(port=data['port'],robot_id=data['robot_id'],enable_torque=False)
    try:
        names=[*ARM_JOINT_NAMES,'gripper']
        torque=robot.bus.sync_read('Torque_Enable',names,normalize=False,num_retry=3)
        if any(torque.values()):
            raise ValueError('Torque must already be off; no position commands were sent')
        pose=robot.bus.sync_read('Present_Position',list(ARM_JOINT_NAMES),normalize=True,num_retry=3)
    finally:
        robot.disconnect()
    calibration=require_calibration(data['robot_id'])
    home={'schema_version':1,'kind':'so101_post_eval_home','robot_id':data['robot_id'],
          'follower_calibration_sha256':hashlib.sha256(calibration.read_bytes()).hexdigest(),
          'arm_joint_names':list(ARM_JOINT_NAMES),'pose_deg':pose,
          'typing_keyboard_device':data['keyboard'],'typing_keyboard_event_node_at_measurement':os.path.realpath(data['keyboard'])}
    Path(data['post_eval_home']).write_text(json.dumps(home,indent=2)+'\n')
    print(f"Saved torque-off rest pose: {data['post_eval_home']}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('configure','calibrate','save-rest'))
    p.add_argument('--profile',type=Path,required=True)
    args=p.parse_args()
    if args.operation=='configure':
        configure(args.profile)
    elif args.operation=='save-rest':
        save_rest(args.profile)
    else:
        data=load_profile(args.profile)
        command=[str(Path(sys.executable).parent/'lerobot-calibrate'),'--robot.type=so101_follower',
                 '--robot.port='+data['port'],'--robot.id='+data['robot_id']]
        subprocess.run(command,check=True)
        print('Motor calibration finished. Keyboard pose and joint-to-simulation mapping still require scene calibration; see docs/tutorial.md.')


if __name__=='__main__':
    main()
