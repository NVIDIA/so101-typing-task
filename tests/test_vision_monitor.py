"""Regression tests for the expensive-run stopping and restart decisions."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('vision_state',Path(__file__).parents[1]/'scripts/vision/state.py')
state=importlib.util.module_from_spec(spec);spec.loader.exec_module(state)

class MonitorRules(unittest.TestCase):
    def report(self,i,rate=.99):
        return {'checkpoint_sha256':str(i),'success_rate':rate,'episodes':1024,
                'domain_strength':1,'target':'NEWTON','finished_at':i*1000}
    def test_strict_streak(self):
        good=[self.report(i) for i in range(3)]
        self.assertTrue(state.converged(good))
        self.assertFalse(state.converged(good[:2]))
        self.assertFalse(state.converged([good[0]]*3))
        self.assertFalse(state.converged(good[:2]+[self.report(3,.98)]))
        self.assertFalse(state.converged(good[:2]+[dict(good[2],domain_strength=0)]))
        self.assertFalse(state.converged(good[:2]+[dict(good[2],episodes=100)]))
    def test_stagnation_resets_on_improvement_and_restart(self):
        reports=[self.report(1,.2),self.report(2,.8)]
        self.assertFalse(state.stalled(reports,0,now=15000))
        self.assertTrue(state.stalled(reports,0,now=17000))
        self.assertFalse(state.stalled(reports,16000,now=17000))
    def test_partial_progress_counts_before_full_success(self):
        reports=[dict(self.report(1,0),mean_completed_letters=1),
                 dict(self.report(10,0),mean_completed_letters=3)]
        self.assertFalse(state.stalled(reports,0,now=17000))

    def test_adjustment_retains_domain_gate(self):
        change=state.adjustment([self.report(1)],3,current_learning_rate=2.5e-5)
        self.assertLess(change['learning_rate'],1e-4)
        self.assertGreater(change['curriculum_iterations'],1000)
        self.assertNotIn('threshold',change)

    def test_recovery_halves_actual_rate_not_hardcoded_baseline(self):
        first=state.adjustment([],1,current_learning_rate=2.5e-5)
        self.assertEqual(first['learning_rate'],1.25e-5)
        self.assertEqual(first['previous_learning_rate'],2.5e-5)
        self.assertEqual(state.adjustment([],2,current_learning_rate=5e-5)['learning_rate'],2.5e-5)
        self.assertEqual(state.adjustment([],3,current_learning_rate=first['learning_rate'])['learning_rate'],1e-5)

    def test_floor_never_increases_a_lower_rate(self):
        self.assertEqual(state.adjustment([],1,current_learning_rate=5e-6)['learning_rate'],5e-6)
        for rate in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                state.adjustment([],1,current_learning_rate=rate)

if __name__=='__main__':unittest.main()
