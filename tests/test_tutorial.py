"""Tutorial contracts without GPU execution, network access or robot motion."""
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.tutorial import artifacts, cli


class TutorialTests(unittest.TestCase):
    def run_files(self, root):
        root.mkdir(parents=True,exist_ok=True)
        for name in artifacts.SIDECARS:
            (root/name).write_text('{}')
        chosen=root/'model_1.pt'
        chosen.write_bytes(b'fake weights for archive tests; never loaded')
        return chosen

    def test_pack_import_preserves_metadata_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            chosen=self.run_files(root/'run')
            camera=root/'camera.json';camera.write_text('{"eye_m":[1,2,3]}')
            archive=root/'model.zip'
            artifacts.pack(chosen,archive,camera)
            loaded=artifacts.unpack(archive,root/'import',artifacts.digest(archive))
            self.assertEqual(loaded.read_bytes(),chosen.read_bytes())
            self.assertTrue((loaded.parent/'evaluation_replay_camera.json').is_file())
            with self.assertRaisesRegex(ValueError,'SHA256'):
                artifacts.unpack(archive,root/'bad','0'*64)

    def test_unsafe_archive_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('../escape','bad')
            with self.assertRaisesRegex(ValueError,'paths'):
                artifacts.unpack(archive,Path(tmp)/'out',artifacts.digest(archive))
            self.assertFalse((Path(tmp)/'escape').exists())

    def test_best_requires_evaluation_and_checks_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);chosen=self.run_files(run)
            with self.assertRaisesRegex(ValueError,'No phrase evaluation'):
                artifacts.checkpoint(run)
            report={'success_rate':.8,'target':'ISAAC LAB 3.0','episodes':100,'seed':9,'domain_strength':1}
            artifacts.record_evaluation(run,chosen,report,'phrase')
            self.assertEqual(artifacts.checkpoint(run),chosen)
            with self.assertRaisesRegex(ValueError,'protocol'):
                artifacts.record_evaluation(run,chosen,report|{'seed':10},'phrase')
            chosen.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                artifacts.checkpoint(run)

    def test_random_and_phrase_selections_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);first=self.run_files(run)
            second=run/'model_2.pt';second.write_bytes(b'second')
            artifacts.record_evaluation(run,first,{'success_rate':.9,'target':'phrase'},'phrase')
            artifacts.record_evaluation(run,second,{'success_rate':.6,'target':'random13'},'random')
            self.assertEqual(artifacts.checkpoint(run,suite='phrase'),first)
            self.assertEqual(artifacts.checkpoint(run,suite='random'),second)

    def test_smoke_and_default_config_keep_random_training(self):
        for name in ('default','smoke'):
            config=cli.configuration(cli.ROOT/f'configs/tutorial/{name}.json')
            self.assertEqual(config['training']['sequence_length'],13)
            self.assertTrue(config['training']['random_sequences'])
            self.assertEqual(config['evaluation']['target'],'ISAAC LAB 3.0')

    def test_teacher_plan_has_no_process_or_filesystem_mutation(self):
        output=cli.ROOT/'output/test-tutorial-plan-uncreated'
        with patch('subprocess.run',side_effect=AssertionError('unexpected process')), patch('builtins.print'):
            self.assertEqual(cli.main(['train','teacher','--plan','--output',str(output)]),0)
        self.assertFalse(output.exists())

    def test_legacy_routes_remain_available(self):
        self.assertFalse(cli.handles(['train','--stage','p1a']))
        self.assertFalse(cli.handles(['deploy','legacy-bundle','NEWTON']))
        self.assertTrue(cli.handles(['train','vision']))
        self.assertTrue(cli.handles(['deploy','--model','demo']))
        self.assertTrue(cli.handles(['deploy','--model=demo']))

    def test_unpublished_model_has_actionable_error(self):
        with patch('sys.stderr',new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as exit:
                cli.main(['download','--model','vision-demo'])
        self.assertEqual(exit.exception.code,2)
        self.assertIn('No published URL',stderr.getvalue())

    def test_state_checkpoint_is_rejected_by_vision_hardware_runner(self):
        from scripts.tutorial.hardware import dispatch
        args = argparse.Namespace(command='deploy', model='state', plan=True)
        with patch('scripts.tutorial.cli.selected', return_value=Path('/tmp/state/checkpoint.pt')), \
             patch('scripts.tutorial.artifacts.read', return_value={'kind':'state'}), \
             patch('subprocess.run', side_effect=AssertionError('must not launch hardware')):
            with self.assertRaisesRegex(ValueError, 'requires a vision checkpoint'):
                dispatch(args)

    def test_hardware_profile_rejects_keyboard_identity_change(self):
        from so101_typing.hardware.tutorial_profile import apply_profile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);device=root/'device';device.touch()
            profile=root/'profile.json'
            data={'schema_version':1,'robot_id':'test','port':str(device),'keyboard':str(device),
                  'keyboard_identity':{'name':'expected'},'wrist_camera':str(device),
                  'mapping_dir':'mapping','post_eval_home':'home.json'}
            profile.write_text(json.dumps(data))
            with patch('so101_typing.hardware.tutorial_profile.keyboard_identity',return_value={'name':'wrong'}):
                with self.assertRaisesRegex(ValueError,'identity changed'):
                    apply_profile(argparse.Namespace(hardware_profile=profile))


if __name__=='__main__':
    unittest.main()
