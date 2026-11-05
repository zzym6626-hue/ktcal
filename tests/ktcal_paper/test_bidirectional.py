import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import h5py
import numpy as np
import torch

from ktcal_paper.experiment import train, load_deployed, evaluate, sha256, write_json


class BidirectionalExperimentTests(unittest.TestCase):
    def test_warm_start_bidir_selection_and_cosine_resume_preserve_predictions(self):
        # Catches accidental forward export, wrong selection head, reset schedule,
        # lost warm-start provenance and mutation of the source experiment.
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            rng = np.random.default_rng(20)
            with h5py.File(root / 'frames.h5', 'w') as f:
                for name in ('1_1', '2_1', '3_1'):
                    group = f.create_group(name)
                    group['frames'] = rng.integers(0, 16, (2, 2, 60, 80), dtype=np.uint8)
                    group['labels'] = np.array([[.4, .3, 0], [.5, .4, 1]], np.float32)
                    group['valid'] = np.ones(2, bool)
            manifest = {'schema': 1, 'dataset': 'TEST FIXTURE', 'h5': 'frames.h5',
                'labels_are_placeholders': False, 'bin_ms': 10., 'label_policy': 'fixture',
                'splits': {'train': ['1_1'], 'val': ['2_1'], 'holdout': ['3_1']},
                'records': [{'recording': f'{n}_1', 'frames': 2} for n in (1, 2, 3)]}
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            source = train(path, root / 'source', epochs=1, batch_size=2,
                           threads=2, device='cpu', stop_after_epochs=1)
            digest = sha256(source)
            with self.assertRaises(ValueError): load_deployed(source, torch.device('cpu'))
            config = dict(epochs=3, batch_size=2, threads=2, device='cpu',
                          inference_mode='bidirectional', scheduler='cosine',
                          lr=.0002, min_lr=.00002, readout='final_hidden')
            full = train(path, root / 'full', initialize_from=str(source), **config)
            interrupted = train(path, root / 'resumed', initialize_from=str(source),
                                stop_after_epochs=1, **config)
            resumed = train(resume=interrupted)
            self.assertEqual(sha256(source), digest)
            a, _, state = load_deployed(full, torch.device('cpu'))
            b, _, resumed_state = load_deployed(resumed, torch.device('cpu'))
            self.assertFalse(a.deployed)
            self.assertTrue(a.gru.bidirectional)
            self.assertEqual(a.readout,'final_hidden')
            self.assertEqual(state['config']['initialization']['source_readout'],'last')
            self.assertEqual(state['config']['initialization']['checkpoint_sha256'], digest)
            torch.manual_seed(4)
            frames = torch.rand(1, 16, 2, 60, 80)
            with torch.no_grad():
                torch.testing.assert_close(a(frames), b(frames), rtol=0, atol=0)
            for key in state['model_state']:
                torch.testing.assert_close(state['model_state'][key], resumed_state['model_state'][key], rtol=0, atol=0)
            history = json.loads((root / 'full/history.json').read_text())
            self.assertEqual(len(history), 3)
            self.assertTrue(all(r['phase'] == 'bidir' for r in history))
            np.testing.assert_allclose([r['learning_rate'] for r in history], [.0002, .00011, .00002])
            summary = json.loads((root / 'full/training_summary.json').read_text())
            self.assertEqual(summary['finetune_epochs'], 0)
            self.assertEqual(summary['selection_head'], 'primary')
            initial = summary['initial_validation']['primary']['mean_error_px']
            self.assertAlmostEqual(state['best_val'], min([initial] + [r['val']['primary']['mean_error_px'] for r in history]))
            self.assertEqual(train(resume=root / 'resumed/last.pt'), resumed)
            with self.assertRaises(ValueError): train(resume=full)
            report = evaluate(full, path, root / 'evaluation', split='val', threads=2)
            rows = json.loads(report.read_text())['rows']
            self.assertTrue(all(r['inference_mode'] == 'bidirectional' for r in rows))
            table=(root / 'evaluation/table.md').read_text()
            self.assertIn('| stnet_only | bidirectional |', table)
            with self.assertRaises(ValueError):
                train(path, root / 'invalid', initialize_from=str(source), variant='no_gru', **config)

            def fail_summary(file, value):
                if Path(file).name == 'training_summary.json': raise OSError('simulated full disk')
                return write_json(file, value)

            with patch('ktcal_paper.experiment.write_json', side_effect=fail_summary):
                with self.assertRaises(OSError):
                    train(path, root / 'interrupted_finalization', initialize_from=str(source), **config)
            last=root / 'interrupted_finalization/last.pt'
            self.assertNotEqual(torch.load(last, weights_only=False)['phase'], 'complete')
            recovered=train(resume=last)
            self.assertTrue(recovered.is_file())
            self.assertEqual(json.loads((last.parent/'training_summary.json').read_text())['status'], 'complete')


if __name__ == '__main__': unittest.main()
