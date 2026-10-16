"""Keep presentation labels tied to complete, unambiguous evaluation evidence."""
from copy import deepcopy
import unittest

from scripts.rendering.compose_replay_video import extract_presses


def example_report():
    return {
        'num_envs': 1, 'successes': 1, 'steps_executed': 8,
        'target_descriptor': {'target': 'NN'},
        'event_trace_env0': [
            dict(step=step, character_index=index, target_down=down, target_up=not down,
                 wrong_key=False, overlap=False)
            for step, index, down in ((1, 0, True), (3, 0, False), (4, 1, True), (7, 1, False))
        ],
    }


class ReplayCompositorTests(unittest.TestCase):
    def test_repeated_letters_retain_distinct_press_and_release_frames(self):
        report = example_report()
        before = deepcopy(report)
        self.assertEqual(extract_presses(report), ('NN', [
            {'key': 'N', 'down_frame': 1, 'up_frame': 3},
            {'key': 'N', 'down_frame': 4, 'up_frame': 7},
        ]))
        self.assertEqual(report, before)

    def test_wrong_or_overlapping_keys_cannot_be_labeled_as_target(self):
        for flag in ('wrong_key', 'overlap'):
            with self.subTest(flag=flag):
                report = example_report()
                report['event_trace_env0'][1][flag] = True
                with self.assertRaisesRegex(ValueError, 'target-only'):
                    extract_presses(report)

    def test_incomplete_or_reordered_events_are_rejected(self):
        for change in ('missing_up', 'duplicate_down', 'reversed_time', 'wrong_character'):
            with self.subTest(change=change):
                report = example_report()
                events = report['event_trace_env0']
                if change == 'missing_up':
                    events.pop()
                elif change == 'duplicate_down':
                    events[1].update(target_down=True, target_up=False)
                elif change == 'reversed_time':
                    events[1]['step'] = 0
                else:
                    events[2]['character_index'] = 0
                with self.assertRaises(ValueError):
                    extract_presses(report)

    def test_failed_or_batched_evaluations_are_rejected(self):
        for key, value in (('successes', 0), ('num_envs', 2)):
            report = example_report()
            report[key] = value
            with self.assertRaisesRegex(ValueError, 'single-environment'):
                extract_presses(report)


if __name__ == '__main__':
    unittest.main()
