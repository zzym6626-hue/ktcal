import unittest
import tempfile
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from ktcal_paper.gate_diagnostics import (
    counterfactual_bad_fast_frames, diagnose_online_gate, summarize_gate_trace,
)
from ktcal_paper.matching import PaperMatcher


class GateDiagnosticsTests(unittest.TestCase):
    def test_oracle_recovery_checks_only_failed_fast_frames_without_future(self):
        class LastFrameModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.windows = []

            def forward(self, windows):
                self.windows.extend(windows.cpu().numpy())
                x = windows[:, -1, 0, 0, 0]
                return torch.stack((x / 80., x * 0), dim=-1)

        frames = torch.stack([torch.full((2, 60, 80), float(i)) for i in range(4)])
        truth = np.zeros((4, 2), dtype=np.float32)
        trace = [
            {'route': 'stnet_reset', 'output_px': [0., 0.]},
            {'route': 'prediction_only', 'output_px': [4., 0.]},
            {'route': 'matched_update', 'output_px': [4., 0.]},
            {'route': 'prediction_only', 'output_px': [4., 0.]},
        ]
        model = LastFrameModel()
        result = counterfactual_bad_fast_frames(trace, truth, frames,
                                                model, torch.device('cpu'))
        self.assertEqual(result['failed_fast_frames'], 3)
        self.assertEqual(result['stnet_would_recover'], 2)
        self.assertEqual(result['by_route']['prediction_only']['failed'], 2)
        self.assertEqual(result['by_route']['prediction_only']['recoverable'], 1)
        self.assertEqual(result['by_route']['matched_update']['recoverable'], 1)
        self.assertEqual(len(model.windows), 3)
        for target, window in zip((1, 2, 3), model.windows):
            self.assertEqual(window.shape, (16, 2, 60, 80))
            self.assertEqual(window[-1, 0, 0, 0], target)
            self.assertLessEqual(window[:, 0, 0, 0].max(), target)

    def test_reports_missed_errors_and_match_consensus_not_ground_truth(self):
        def row(route, output, prediction=None, match=None):
            return {'route': route, 'output_px': [output, 0.],
                    'prediction_px': None if prediction is None else [prediction, 0.],
                    'match_px': None if match is None else [match, 0.],
                    'confidence': None, 'border_peak': False}

        trace = [
            row('stnet_reset', 2),
            row('prediction_only', 4, 4),
            row('matched_update', 2, 4, 2),  # Match recovers a bad prediction.
            row('matched_update', 4, 2, 4),  # Match spoils a good prediction.
            row('matched_update', 4, 4, 4),  # Both agree but both are wrong.
            row('stnet_reset', 0, 0, 0),
        ]
        result = summarize_gate_trace(trace, np.zeros((6, 2)))
        self.assertEqual(result['frames'], 6)
        self.assertAlmostEqual(result['p3'], .5)
        self.assertEqual(result['routes']['prediction_only']['bad_3px'], 1)
        self.assertEqual(result['routes']['matched_update']['frames'], 3)
        self.assertAlmostEqual(result['routes']['matched_update']['p3'], 1 / 3)
        self.assertEqual(result['routes']['stnet_reset']['frames'], 2)
        self.assertEqual(result['matched_update']['spoiled_good_prediction'], 1)
        self.assertEqual(result['matched_update']['recovered_bad_prediction'], 1)
        self.assertEqual(result['matched_update']['prediction_and_match_wrong_but_agree'], 1)

    def test_rejects_missing_or_misaligned_trace_instead_of_silent_miscounts(self):
        with self.assertRaises(ValueError):
            summarize_gate_trace([], np.zeros((1, 2)))

    def test_diagnosis_uses_real_tracking_on_validation_only(self):
        class FixedLogits(torch.nn.Module):
            def forward(self, frames):
                x = torch.zeros(*frames.shape[:2], 80)
                y = torch.zeros(*frames.shape[:2], 60)
                x[:, -1, 40] = 100
                y[:, -1, 30] = 100
                return x, y

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_aug.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((1, 3, 2, 60, 80), dtype=np.uint8)
                truth = np.zeros((1, 3, 3), dtype=np.float32)
                truth[..., :2] = [.5, .5]
                handle['label'] = truth
            student, matcher = root / 'student.pt', root / 'matcher.pt'
            student.write_bytes(b'student')
            matcher.write_bytes(b'matcher')
            with mock.patch('ktcal_paper.gate_diagnostics.load_stnet_checkpoint',
                            return_value=(FixedLogits(), {}, 1.)), \
                 mock.patch('ktcal_paper.gate_diagnostics.load_matcher_checkpoint',
                            return_value=(PaperMatcher(), {})):
                report = diagnose_online_gate(h5, student, matcher,
                                              root / 'gate.json', device='cpu')
            self.assertEqual(report['diagnostics']['frames'], 3)
            self.assertEqual(sum(group['frames'] for group in
                                 report['diagnostics']['routes'].values()), 3)
            self.assertEqual(report['metrics']['ktcal_online']['p3'], 1.)
            self.assertEqual(report['diagnostics']['p3'], 1.)
            self.assertEqual(report['diagnostics']['counterfactual_bad_fast_frames']
                             ['failed_fast_frames'], 0)
            self.assertFalse(report['final_holdout_used'])
            self.assertFalse(report['latency_comparable'])
            self.assertTrue((root / 'gate.json').is_file())
            with self.assertRaisesRegex(ValueError, 'final holdout'):
                diagnose_online_gate(root / 'test_200_200.h5', student, matcher,
                                     root / 'holdout.json', device='cpu')


if __name__ == '__main__':
    unittest.main()
