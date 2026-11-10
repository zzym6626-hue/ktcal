import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from ktcal_paper.adaptive_offline import (
    adaptive_smooth, diagnose_frame_groups, select_reliable_match,
)
from scripts.diagnose_3et_adaptive_offline import evaluate_adaptive_offline


class AdaptiveOfflineTests(unittest.TestCase):
    def test_slow_frames_use_ma11_and_motion_uses_ma3_without_labels(self):
        reference = np.zeros((21, 2), dtype=np.float32)
        reference[8:, 0] = np.arange(13, dtype=np.float32) * 4 / 80
        candidate = reference.copy()
        candidate[10, 0] += 6 / 80
        result, speed = adaptive_smooth(candidate, reference, .5, 1.5)
        from ktcal_paper.combined_3et import moving_average_sequence_predictions
        ma3 = moving_average_sequence_predictions([candidate], 3)[0]
        ma11 = moving_average_sequence_predictions([candidate], 11)[0]
        np.testing.assert_allclose(result[2], ma11[2])
        np.testing.assert_allclose(result[12], ma3[12])
        self.assertGreater(speed[12], 1.5)
        np.testing.assert_array_equal(candidate[0], reference[0])

    def test_match_fallback_only_when_high_disagreement_and_low_confidence(self):
        stnet = np.array([[.5, .5], [.5, .5]], dtype=np.float32)
        matched = np.array([[.54, .5], [.54, .5]], dtype=np.float32)
        confidence = np.array([.55, .95], dtype=np.float32)
        selected, mask = select_reliable_match(
            stnet, matched, confidence, min_confidence=.7,
            max_disagreement_px=2.)
        np.testing.assert_array_equal(mask, [True, False])
        np.testing.assert_array_equal(selected[0], stnet[0])
        np.testing.assert_array_equal(selected[1], matched[1])

    def test_group_diagnostics_do_not_use_truth_for_membership(self):
        raw = np.array([[0., 0.], [0., 0.], [4 / 80., 0.]], dtype=np.float32)
        labels = np.zeros_like(raw)
        one = diagnose_frame_groups(raw, raw, labels, np.array([0, 10, 100]))
        other = diagnose_frame_groups(raw, raw, labels + [.5, .5],
                                      np.array([0, 10, 100]))
        self.assertEqual(one['speed']['moving']['frames'],
                         other['speed']['moving']['frames'])
        self.assertEqual(one['event_density']['sparse']['frames'],
                         other['event_density']['sparse']['frames'])

    def test_runner_rejects_holdout_without_reading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test_200_200.h5'
            path.write_bytes(b'not an hdf5 file')
            with mock.patch('scripts.diagnose_3et_adaptive_offline.load_stnet_checkpoint') as load:
                with self.assertRaisesRegex(ValueError, 'test_aug'):
                    evaluate_adaptive_offline(path, 'stnet.pt', 'matcher.pt',
                                              Path(directory) / 'result.json', device='cpu')
                load.assert_not_called()

    def test_runner_produces_validation_only_report(self):
        class FixedLogits(torch.nn.Module):
            def forward(self, frames):
                x = torch.zeros(*frames.shape[:2], 80)
                y = torch.zeros(*frames.shape[:2], 60)
                x[..., 40] = 100
                y[..., 30] = 100
                return x, y

        from ktcal_paper.matching import PaperMatcher
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_aug.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((1, 3, 2, 60, 80), dtype=np.uint8)
                labels = np.zeros((1, 3, 3), dtype=np.float32)
                labels[..., :2] = [.5, .5]
                handle['label'] = labels
            stnet, matcher, report_path = (root / n for n in
                    ('stnet.pt', 'matcher.pt', 'report.json'))
            stnet.write_bytes(b'stnet')
            matcher.write_bytes(b'matcher')
            with mock.patch('scripts.diagnose_3et_adaptive_offline.load_stnet_checkpoint',
                            return_value=(FixedLogits(), {'readout': 'attention'}, 1.)), \
                 mock.patch('scripts.diagnose_3et_adaptive_offline.load_matcher_checkpoint',
                            return_value=(PaperMatcher(), {})):
                report = evaluate_adaptive_offline(
                    h5, stnet, matcher, report_path, device='cpu')
            self.assertFalse(report['final_holdout_used'])
            self.assertEqual(report['metrics']['ktcal_ma11']['frames'], 3)
            self.assertEqual(report['metrics']['ktcal_adaptive']['frames'], 3)
            self.assertTrue(report['per_frame_npz'])
            self.assertTrue(Path(report['per_frame_npz']).is_file())
            self.assertGreaterEqual(report['postprocess_ms_per_frame'], 0.)


if __name__ == '__main__':
    unittest.main()
