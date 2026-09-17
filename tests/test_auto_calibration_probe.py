"""Fault and coordinate tests for the experimental automatic endpoint probe."""
import unittest
from pathlib import Path
import tempfile
import json

from scripts.so101_homing.auto_calibration_probe import (
    ARM, CONFIG, REGISTERS, Probe, Settings, bounds, snapshot, summarize, validate_baseline,
    require_fresh_camera,
)


class FakeClock:
    def __init__(self): self.t = 0.
    def __call__(self): return self.t
    def sleep(self, dt): self.t += dt


class FakeBus:
    def __init__(self, clock):
        self.clock = clock
        self.last = 0.
        self.motors = dict.fromkeys((*ARM, 'gripper'))
        self.registers = {m: dict.fromkeys((*REGISTERS, 'Moving'), 0) for m in self.motors}
        for row in self.registers.values():
            row.update(Homing_Offset=23, Min_Position_Limit=1005, Max_Position_Limit=3195,
                       Present_Position=2100, Goal_Position=1600, Torque_Limit=800,
                       Max_Torque_Limit=1000, P_Coefficient=16, D_Coefficient=32,
                       Acceleration=50, Goal_Velocity=217, Present_Temperature=30, Present_Voltage=120)
        self.writes = []
        self.low, self.high = 1000, 3200
        self.fail_read = None
        self.never_stall = False
        self.ignore_homing_in_velocity = False

    def advance(self):
        dt = self.clock()-self.last
        self.last = self.clock()
        for row in self.registers.values():
            before = row['Present_Position']
            if row['Torque_Enable']:
                if row['Operating_Mode'] == 1:
                    row['Present_Position'] = min(self.high, max(self.low, before + dt*row['Goal_Velocity']))
                else:
                    goal = row['Goal_Position']
                    step = 400*dt
                    row['Present_Position'] += max(-step, min(step, goal-before))
            if dt > 0:
                row['Present_Velocity'] = round((row['Present_Position']-before)/dt)
                row['Moving'] = int(row['Present_Position'] != before or self.never_stall)

    def read(self, reg, motor, **kwargs):
        self.advance()
        if reg == self.fail_read:
            self.fail_read = None
            raise ConnectionError('injected read failure')
        value = round(self.registers[motor][reg])
        if reg == 'Present_Position' and self.ignore_homing_in_velocity and self.registers[motor]['Operating_Mode'] == 1:
            value = (value+self.registers[motor]['Homing_Offset']) % 4096
        return value

    def write(self, reg, motor, value, **kwargs):
        self.advance()
        self.writes.append((reg, motor, value))
        self.registers[motor][reg] = value


def setup():
    clock = FakeClock()
    bus = FakeBus(clock)
    calibration = {m: {'id': i+1, 'drive_mode': 0, 'homing_offset': 23,
                       'range_min': 1005, 'range_max': 3195} for i,m in enumerate(bus.motors)}
    before = snapshot(bus)
    events = []
    probe = Probe(bus, before, calibration, 'elbow_flex', Settings(), events.append, clock, clock.sleep)
    return bus, before, calibration, probe, events


class AutoCalibrationProbeTests(unittest.TestCase):
    def test_feedback_refreshes_after_mode_acknowledgement(self):
        bus,before,cal,probe,events=setup()
        bus.ignore_homing_in_velocity=True
        original_read, original_write=bus.read,bus.write
        pending={}
        def delayed_read(reg,motor,**kwargs):
            if reg=='Present_Position' and motor in pending and bus.clock()<pending[motor][0]:
                return pending[motor][1]
            return original_read(reg,motor,**kwargs)
        def delayed_write(reg,motor,value,**kwargs):
            if reg=='Operating_Mode':
                pending[motor]=(bus.clock()+.015,original_read('Present_Position',motor))
            return original_write(reg,motor,value,**kwargs)
        bus.read,bus.write=delayed_read,delayed_write
        self.assertEqual(probe.run(),[{'range_min':1000,'range_max':3200}]*3)
        self.assertEqual(probe.velocity_frame_shift,-23)
        self.assertTrue(probe.restore()['configuration_restored'])

    def test_velocity_mode_ignoring_homing_is_converted_before_bounds_and_fit(self):
        for homing in (-267,1200):  # Negative real homing; positive offset exercises wrap.
            with self.subTest(homing=homing):
                bus,before,cal,probe,events=setup()
                bus.ignore_homing_in_velocity=True
                bus.registers['elbow_flex']['Homing_Offset']=homing
                before['elbow_flex']['Homing_Offset']=homing
                cal['elbow_flex']['homing_offset']=homing
                runs=probe.run()
                self.assertEqual(probe.velocity_frame_shift,-homing)
                self.assertEqual(runs,[{'range_min':1000,'range_max':3200}]*3)
                self.assertTrue(probe.restore()['configuration_restored'])

    def test_mode_check_rejects_unexplained_frame_shift(self):
        bus,before,cal,probe,events=setup()
        original_read=bus.read
        def read_shift(reg,motor,**kwargs):
            value=original_read(reg,motor,**kwargs)
            if reg=='Present_Position' and bus.registers[motor]['Operating_Mode']==1:
                return value+100
            return value
        bus.read=read_shift
        with self.assertRaisesRegex(RuntimeError,'mode convention check inconclusive'):
            probe.run()
        self.assertFalse(any(reg=='Goal_Velocity' and value!=0 for reg,_,value in bus.writes))
        self.assertTrue(probe.restore()['configuration_restored'])

    def test_camera_requires_fresh_successful_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'camera.json'
            for state, ok in [({'status':'streaming','host_monotonic_received_s':9.8},True),
                              ({'status':'streaming','host_monotonic_received_s':8.},False),
                              ({'status':'failed','host_monotonic_received_s':9.8},False),
                              ({'status':'streaming','host_monotonic_received_s':11.},False)]:
                path.write_text(json.dumps(state))
                if ok:
                    require_fresh_camera(path,clock=lambda:10.)
                else:
                    with self.assertRaises(RuntimeError): require_fresh_camera(path,clock=lambda:10.)

    def test_camera_loss_aborts_velocity_and_restores(self):
        bus,before,cal,probe,events=setup()
        def check():
            if bus.read('Operating_Mode','elbow_flex') == 1 and bus.read('Goal_Velocity','elbow_flex'):
                raise RuntimeError('camera lost')
        probe.check_camera=check
        with self.assertRaisesRegex(RuntimeError, 'camera lost'): probe.run()
        self.assertEqual(bus.read('Goal_Velocity','elbow_flex'),0)
        self.assertTrue(probe.restore()['configuration_restored'])

    def test_snapshot_and_validation_write_nothing(self):
        bus,before,cal,probe,_ = setup()
        validate_baseline(before, cal, 'elbow_flex', Settings())
        self.assertEqual(bus.writes, [])

    def test_three_repeats_restore_configuration_without_stale_goals(self):
        bus,before,cal,probe,events = setup()
        runs = probe.run()
        self.assertEqual(runs, [{'range_min':1000, 'range_max':3200}]*3)
        result = probe.restore()
        self.assertTrue(result['configuration_restored'], result)
        self.assertTrue(summarize(runs, cal['elbow_flex'], Settings())['repeatable'])
        for motor in ARM:
            for reg in CONFIG:
                self.assertEqual(bus.read(reg,motor), before[motor][reg], (motor,reg))
            self.assertEqual(bus.read('Torque_Enable',motor), 1)
            self.assertAlmostEqual(bus.read('Goal_Position',motor), bus.read('Present_Position',motor), delta=1)
        self.assertFalse(any(m == 'gripper' for _,m,_ in bus.writes))
        self.assertFalse(any(r in ('P_Coefficient','I_Coefficient','D_Coefficient','Homing_Offset','Phase','Max_Torque_Limit') for r,_,_ in bus.writes))
        for m in ARM:
            enable_index = next(i for i,w in enumerate(bus.writes) if w == ('Torque_Enable',m,1))
            self.assertIn(('Goal_Position',m,2100), bus.writes[:enable_index])

    def test_early_obstruction_is_not_accepted_as_endpoint(self):
        bus,before,cal,probe,events = setup()
        bus.low = 1800
        with self.assertRaisesRegex(RuntimeError, 'early stall'):
            probe.run()
        self.assertTrue(probe.restore()['configuration_restored'])
        self.assertFalse(any(e['event']=='endpoint' for e in events))

    def test_bad_status_aborts_and_restores(self):
        bus,before,cal,probe,events = setup()
        bus.registers['elbow_flex']['Status']=32
        with self.assertRaisesRegex(RuntimeError, 'controller fault'):
            probe.run()
        self.assertTrue(probe.restore()['configuration_restored'])

    def test_communication_failure_is_not_a_stop(self):
        bus,before,cal,probe,events = setup()
        bus.fail_read='Present_Temperature'
        with self.assertRaisesRegex(ConnectionError, 'injected'):
            probe.run()
        self.assertTrue(probe.restore()['configuration_restored'])
        self.assertFalse(any(e['event']=='endpoint' for e in events))

    def test_timeout_is_not_a_stop(self):
        bus,before,cal,probe,events = setup()
        bus.never_stall=True
        with self.assertRaisesRegex(RuntimeError, 'endpoint timeout'):
            probe.run()
        result = probe.restore()
        self.assertFalse(result['configuration_restored'])
        self.assertEqual(bus.read('Torque_Enable','elbow_flex'), 0)

    def test_ctrl_c_and_failed_velocity_stop_do_not_leave_running_motor(self):
        bus,before,cal,probe,events = setup()
        original_sleep = probe.sleep
        original_write = bus.write
        interrupted = []
        def interrupt(dt):
            original_sleep(dt)
            if bus.read('Operating_Mode','elbow_flex') == 1 and bus.read('Goal_Velocity','elbow_flex'):
                interrupted.append(True)
                raise KeyboardInterrupt('operator abort')
        probe.sleep = interrupt
        failed = []
        def fail_stop(reg, motor, value, **kwargs):
            # Fail just the abort's stop packet, after motion was commanded.
            if reg == 'Goal_Velocity' and value == 0 and interrupted and not failed:
                failed.append(True)
                raise ConnectionError('stop packet failed')
            return original_write(reg,motor,value,**kwargs)
        bus.write=fail_stop
        with self.assertRaisesRegex(ConnectionError, 'stop packet failed'):
            probe.run()
        self.assertEqual(bus.read('Torque_Enable','elbow_flex'),0)
        self.assertTrue(probe.restore()['configuration_restored'])

    def test_failed_configuration_restore_is_reported_and_torque_disabled(self):
        bus,before,cal,probe,events = setup()
        probe.hold_arm()
        bus.registers['elbow_flex']['Operating_Mode']=1
        original_write=bus.write
        def fail_mode(reg,motor,value,**kwargs):
            if reg=='Operating_Mode':
                raise ConnectionError('restore write failure')
            return original_write(reg,motor,value,**kwargs)
        bus.write=fail_mode
        result=probe.restore()
        self.assertFalse(result['configuration_restored'])
        self.assertIn('restore write failure',result['errors']['elbow_flex'])
        self.assertEqual(bus.read('Torque_Enable','elbow_flex'),0)

    def test_rejects_changed_homing_and_encoder_wrap_before_any_write(self):
        bus,before,cal,probe,events = setup()
        before['elbow_flex']['Homing_Offset']=24
        with self.assertRaisesRegex(ValueError, 'file/register mismatch'):
            validate_baseline(before, cal, 'elbow_flex', Settings())
        with self.assertRaisesRegex(ValueError, 'encoder wrap'):
            bounds({'range_min':20,'range_max':3000}, Settings())
        self.assertEqual(bus.writes, [])

    def test_conversion_direction_and_endpoint_repeatability(self):
        settings = Settings()
        original = {'range_min':1000,'range_max':3000}
        runs=[{'range_min':1100,'range_max':3100}]*3
        result=summarize(runs, original, settings)
        self.assertAlmostEqual(result['q_auto_minus_q_manual_deg'], -100*360/4095)
        # Identical midpoints must not conceal endpoints with changing spans.
        runs=[{'range_min':1000+d,'range_max':3000-d} for d in (0,20,40)]
        result=summarize(runs, original, settings)
        self.assertEqual(result['spread_deg']['midpoint'],0)
        self.assertFalse(result['repeatable'])


if __name__ == '__main__':
    unittest.main()
