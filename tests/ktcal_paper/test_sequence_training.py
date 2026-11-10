import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from ktcal_paper.experiment import evaluate, load_deployed, train_sequence


def fixture(root, frames=3):
    rng = np.random.default_rng(31)
    with h5py.File(root / 'frames.h5', 'w') as handle:
        for name in ('1_1', '2_1', '3_1'):
            group = handle.create_group(name)
            group['frames'] = rng.integers(0, 16, (frames, 2, 60, 80), dtype=np.uint8)
            labels = np.tile(np.array([[.4, .3, 0]], np.float32), (frames, 1))
            group['labels'] = labels
            group['valid'] = np.ones(frames, bool)
    manifest = {'schema': 1, 'dataset': 'TEST FIXTURE', 'h5': 'frames.h5',
        'labels_are_placeholders': False, 'bin_ms': 10., 'label_policy': 'fixture',
        'splits': {'train': ['1_1'], 'val': ['2_1'], 'holdout': ['3_1']},
        'records': [{'recording': f'{n}_1', 'frames': frames} for n in (1, 2, 3)]}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


class SequenceTrainingTests(unittest.TestCase):
    def test_all_timesteps_are_supervised_and_resume_reproduces_weights(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = fixture(root)
            config = dict(epochs=2, chunk_window=4, batch_size=1, threads=2,
                          device='cpu', match_weight=.3)
            full = train_sequence(path, root / 'full', **config)
            partial = train_sequence(path, root / 'partial', stop_after_epochs=1, **config)
            resumed = train_sequence(resume=partial)
            a = torch.load(full, map_location='cpu', weights_only=False)
            b = torch.load(resumed, map_location='cpu', weights_only=False)
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
            for key in a['matcher_state']:
                torch.testing.assert_close(a['matcher_state'][key], b['matcher_state'][key], rtol=0, atol=0)
            history = json.loads((root / 'full/history.json').read_text())
            self.assertEqual(len(history), 2)
            self.assertTrue(all(row['phase'] == 'sequence' for row in history))
            # Three real frames per chunk; padding frames never provide supervision.
            self.assertEqual(history[0]['valid_timesteps'], 3)
            summary = json.loads((root / 'full/training_summary.json').read_text())
            self.assertEqual(summary['inference_mode'], 'bidirectional')
            self.assertFalse(summary['holdout_used_for_model_selection'])
            model, matcher, state = load_deployed(full, torch.device('cpu'))
            self.assertTrue(model.gru.bidirectional)
            self.assertFalse(model.deployed)
            self.assertTrue(state['training_completed'])
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.zeros(1, 16, 2, 60, 80)).shape), (1, 2))
            report = evaluate(full, path, root / 'evaluation', split='val', threads=2,
                              capture='coarse')
            payload = json.loads(report.read_text())
            self.assertTrue(all(row['inference_mode'] == 'bidirectional' for row in payload['rows']))
            self.assertTrue(all(row['capture'] == 'coarse' for row in payload['rows']))
            self.assertTrue(payload['rows'][0]['checkpoint_finalized'])

    def test_unfinalized_bidirectional_checkpoint_is_rejected_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = fixture(root)
            config = dict(epochs=3, chunk_window=4, batch_size=1, threads=2,
                          device='cpu', match_weight=0.)
            last = train_sequence(path, root / 'run', stop_after_epochs=1, **config)
            state = torch.load(last, map_location='cpu', weights_only=False)
            self.assertNotEqual(state['phase'], 'complete')
            with self.assertRaises(ValueError):
                load_deployed(last, torch.device('cpu'))
            model, _, state = load_deployed(last, torch.device('cpu'), allow_partial=True)
            self.assertFalse(state.get('training_completed'))
            report = evaluate(last, path, root / 'partial', split='val', threads=2,
                              allow_partial=True)
            payload = json.loads(report.read_text())
            self.assertFalse(payload['rows'][0]['checkpoint_finalized'])
            self.assertTrue(any('not marked complete' in text for text in payload['limitations']))


if __name__ == '__main__':
    unittest.main()
