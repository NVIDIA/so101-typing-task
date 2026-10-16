"""Recorder integration: separate event streams and verified torque-off cleanup."""
from dataclasses import make_dataclass
from types import SimpleNamespace

from so101_typing.hardware import typing_calibration_record as recorder


class Clock:
    now = 100.

    def monotonic(self):
        self.now += .000001
        return self.now

    def time(self):
        return self.now + 1000.

    def sleep(self, seconds):
        self.now += seconds


Cal = make_dataclass('Cal', [(k,int) for k in ('id','drive_mode','homing_offset','range_min','range_max')])


class Bus:
    def __init__(self):
        self.is_connected = False
        self.motors = {n: SimpleNamespace(id=i, model='sts3215', norm_mode='degrees' if i<6 else 'range_0_100')
                       for i,n in enumerate(recorder.JOINT_NAMES,1)}
        self.calibration = {n: Cal(i,0,0,1024 if i<6 else 1000,3072 if i<6 else 3000)
                            for i,n in enumerate(recorder.JOINT_NAMES,1)}
        self.regs = {r:dict.fromkeys(recorder.JOINT_NAMES,0)
                     for r in (*recorder.REGISTERS,*recorder.TELEMETRY_REGISTERS)}
        self.regs['Present_Position']={n:2048 if i<6 else 1000 for i,n in enumerate(recorder.JOINT_NAMES,1)}
        self.writes=[]

    def connect(self): self.is_connected=True
    def disconnect(self, **kw): self.is_connected=False
    def read_calibration(self): return self.calibration
    def sync_read(self, reg, names, **kw): return {n:self.regs[reg][n] for n in names}
    def sync_write(self, reg, values, **kw):
        self.regs[reg].update(values)
        self.writes.append((reg,values.copy()))


class Keyboard:
    def __init__(self,clock): self.clock=clock; self.polls=0
    def poll(self):
        self.polls+=1
        value={3:1,4:0,5:1,9:0}.get(self.polls)
        events=[] if value is None else [dict(code=18,char='e',key_name='KEY_E',value=value,
                    edge='down' if value else 'up',evdev_timestamp_s=self.clock.time(),
                    received_monotonic_s=self.clock.monotonic())]
        return events,self.polls>=14


def test_recorder_retains_raw_filters_display_and_disables_torque():
    follower,leader,clock=Bus(),Bus(),Clock()
    report=dict(port='/dev/test-follower',calibration={k:vars(v) for k,v in follower.calibration.items()},
                leader=dict(port='/dev/test-leader',calibration={k:vars(v) for k,v in leader.calibration.items()}))
    emitted=[]
    recorder.run_session(follower,leader,report,Keyboard(clock),execute=True,seconds=1,
                         clock=clock,emit=emitted.append,mirror_gripper=False,telemetry_hz=5)
    assert len(report['events'])==4
    assert len(report['debounced_events'])==2
    assert report['debounced_events']==[report['events'][0],report['events'][-1]]
    assert [r['kind'] for r in emitted].count('key')==4
    assert [r['kind'] for r in emitted].count('key_debounced')==2
    assert report['telemetry']
    assert report['follower_torque_off_verified']
    assert set(report['live']['follower']['registers_after']['Torque_Enable'].values())=={0}
    assert leader.writes==[]
