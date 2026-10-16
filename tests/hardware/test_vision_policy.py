# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""No-motion checks for the physical wrist-vision observation contract."""
import dataclasses
from pathlib import Path
import time
import unittest

import numpy as np

from so101_typing.hardware.fixed_cartesian_policy import (
    FixedCartesianTypingState, PHYSICAL_CALIBRATED_20260718_PROFILE,
)
from so101_typing.hardware.vision_policy import CameraRecorder, vision_observation


class VisionObservationTests(unittest.TestCase):
    def setUp(self):
        base=PHYSICAL_CALIBRATED_20260718_PROFILE
        self.profile=dataclasses.replace(base,q_reset_rad=(*base.q_reset_rad[:4],base.q_reset_rad[4]-np.pi/2))
        self.state=FixedCartesianTypingState.from_text('NEWTON',stage='p1d',profile=self.profile)

    def test_exact_order_and_goal(self):
        reset=np.asarray(self.profile.q_reset_rad)
        measured=reset+np.asarray([.01,.02,.03,.04,.05])
        command=reset+np.asarray([.05,.04,.03,.02,.01])
        velocity=np.asarray([.1,.2,.3,.4,.5])
        obs=vision_observation(measured,velocity,command,self.state,1.25)
        self.assertEqual(obs.shape,(45,))
        np.testing.assert_allclose(obs[:5],measured-reset,rtol=0,atol=2e-7)
        np.testing.assert_allclose(obs[5:10],velocity,rtol=0,atol=2e-7)
        np.testing.assert_allclose(obs[10:15],command-reset,rtol=0,atol=2e-7)
        self.assertEqual(np.flatnonzero(obs[15:41]).tolist(),[ord('N')-ord('A')])
        np.testing.assert_array_equal(obs[41:44],[1,0,0])
        self.assertEqual(float(obs[44]),1.25)

    def test_phase_and_invalid_input(self):
        self.state.key_down('N',now_s=.5)
        reset=np.asarray(self.profile.q_reset_rad)
        obs=vision_observation(reset,np.zeros(5),reset,self.state,1.0)
        np.testing.assert_array_equal(obs[41:44],[0,1,0])
        self.assertEqual(float(obs[44]),.5)
        with self.assertRaises(ValueError):
            vision_observation(reset,np.full(5,np.nan),reset,self.state,1.0)

    def test_wrist_history_order_and_normalization(self):
        recorder=CameraRecorder(Path('/tmp/unused-vision-recorder'))
        previous=np.zeros((120,160,3),dtype=np.uint8)
        previous[...,2]=255
        current=np.zeros((120,160,3),dtype=np.uint8)
        current[...,0]=255
        recorder.latest_wrist=(7,time.monotonic_ns(),current)
        history,new_previous,info=recorder.wrist_rgb_history(previous)
        self.assertEqual(history.shape,(6,120,160))
        self.assertEqual(float(history[2,0,0]),1.)
        self.assertEqual(float(history[3,0,0]),1.)
        self.assertEqual(float(history[0,0,0]),0.)
        self.assertEqual(info['frame'],7)
        np.testing.assert_array_equal(new_previous,current)


if __name__=='__main__':
    unittest.main()

class RepeatKeyGateTests(unittest.TestCase):
    def test_only_current_target_repeat_is_tolerated(self):
        from so101_typing.hardware.run_vision_policy_handoff import _is_expected_repeat
        state=FixedCartesianTypingState.from_text('NEWTON',stage='p1d')
        state.key_down('N',now_s=.1)
        self.assertTrue(_is_expected_repeat(state,'n','press_before_release'))
        state.key_up('N',now_s=.2)
        self.assertTrue(_is_expected_repeat(state,'N','press_before_clearance'))
        self.assertFalse(_is_expected_repeat(state,'E','press_before_clearance'))
        self.assertFalse(_is_expected_repeat(state,'N','wrong_key'))
        self.assertFalse(_is_expected_repeat(state,None,'press_before_clearance'))


class PhraseVisionTests(unittest.TestCase):
    def make_state(self):
        from so101_typing.hardware.vision_policy import VisionTypingState, PHRASE_CHARACTERS
        profile = PHYSICAL_CALIBRATED_20260718_PROFILE
        labels = list(PHRASE_CHARACTERS)
        labels[36] = 'Space'
        points = list(profile.az_xyz_b_m) + [(0., 0., .03)] * 12
        env = {'scene': {'robot': {'init_state': {'pos': [0,0,0], 'rot': [0,0,.707,.707]}}},
               'commands': {'typing': {'slot_labels': labels, 'letter_slots': list(range(38)),
               'letter_xyz_b_m': points, 'fixed_target_slots': [PHRASE_CHARACTERS.index(c) for c in 'ISAAC LAB 3.0'],
               'letter_length': [13,13], 'clearance_m': .015, 'clearance_control_ticks': 2,
               'phase_timeouts_s': [6,2,2]}}}
        return VisionTypingState.from_environment('ISAAC LAB 3.0', profile, env, PHRASE_CHARACTERS), env

    def test_full_phrase_and_observations(self):
        state, _ = self.make_state()
        q = np.asarray(state.profile.q_reset_rad)
        now = 0.
        for char in 'ISAAC LAB 3.0':
            obs = vision_observation(q,np.zeros(5),q,state,now)
            self.assertEqual(obs.shape, (57,))
            self.assertEqual(np.flatnonzero(obs[15:53]).tolist(), [state.goal_characters.index(char)])
            self.assertEqual(state.key_down(char,now_s=now+.1), 'target_down')
            self.assertEqual(state.key_up(char,now_s=now+.2), 'target_up')
            z=state.active_target_z()
            self.assertIsNone(state.clearance_tick(tip_z_b_m=z+.005,any_key_held=False,now_s=now+.24))
            self.assertIsNone(state.clearance_tick(tip_z_b_m=z+.016,any_key_held=False,now_s=now+.28))
            transition=state.clearance_tick(tip_z_b_m=z+.016,any_key_held=False,now_s=now+.32)
            now += .4
        self.assertEqual(transition, 'target_complete')
        self.assertTrue(state.complete)
        self.assertEqual(state.typed, 'ISAAC LAB 3.0')

    def test_reject_vocabulary_and_target_mismatch(self):
        from so101_typing.hardware.vision_policy import VisionTypingState, PHRASE_CHARACTERS
        state, env = self.make_state()
        with self.assertRaises(ValueError):
            VisionTypingState.from_environment('NEWTON', state.profile, env, PHRASE_CHARACTERS)
        env['commands']['typing']['letter_slots'][0:2] = [1,0]
        with self.assertRaises(ValueError):
            VisionTypingState.from_environment('ISAAC LAB 3.0', state.profile, env, PHRASE_CHARACTERS)
