"""Validation-only deferred medium corrections, independent of test labels."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from ktcal_paper.tracker import PaperTracker, PixelKalman
from scripts.diagnose_3et_deferred_medium import (
    evaluate_deferred_medium, simulate_deferred_sequence,
)


class ScriptedMatcherTracker(PaperTracker):
    def __init__(self, matches):
        super().__init__(torch.nn.Identity(), torch.nn.Identity(),
                         tau1=.5, tau2=3., border_tier=2)
        self.matches = matches

    def _match(self, template, coarse_template, frame, coarse_frame, center):
        return np.array(self.matches[int(frame[0, 0, 0])], dtype=float), .9, {
            'border_peak': False, 'psr': 10.}

    def _template_at(self, images, center):
        return None, None


class DeferredMediumTests(unittest.TestCase):
    def setUp(self):
        self.frames = torch.stack([torch.full((2, 60, 80), float(i))
                                   for i in range(3)])

    def test_medium_issues_matched_kalman_then_rolls_back_when_over_1px(self):
        tracker = ScriptedMatcherTracker({1: [42., 30.], 2: [46., 30.]})
        global_px = np.array([[40., 30.], [47., 30.], [47., 30.]])
        issued, final, stats = simulate_deferred_sequence(
            self.frames, global_px, tracker, tolerance_px=1.)
        self.assertEqual(stats['medium_rollbacks'], 1)
        self.assertEqual(stats['medium_refinements'], 0)
        self.assertEqual(stats['tier2'], 1)
        self.assertGreater(issued[1, 0], 40.)  # matched KF, not raw prediction
        self.assertLess(issued[1, 0], 42.)
        np.testing.assert_allclose(final[1], [47., 30.])
        np.testing.assert_allclose(issued[0], [40., 30.])
        self.assertGreater(stats['medium_disagreement_px'][0], 1.)

    def test_equal_1px_is_refinement_without_reset_and_issued_is_immutable(self):
        kf = PixelKalman()
        kf.initialize(np.array([40., 30.]))
        kf.predict()
        matched = kf.update(np.array([42., 30.]))
        globals_px = np.array([[40., 30.], matched + [1., 0.], matched + [1., 0.]])
        tracker = ScriptedMatcherTracker({1: [42., 30.], 2: matched + [1., 0.]})
        issued, final, stats = simulate_deferred_sequence(
            self.frames[:2], globals_px[:2], tracker, tolerance_px=1.)
        self.assertEqual(stats['medium_refinements'], 1)
        self.assertEqual(stats['medium_rollbacks'], 0)
        self.assertAlmostEqual(stats['medium_disagreement_px'][0], 1., places=7)
        np.testing.assert_allclose(issued[1], matched)
        self.assertGreater(final[1, 0], issued[1, 0])
        self.assertLess(final[1, 0], globals_px[1, 0])

    def test_large_disagreement_resets_immediately_and_no_future_labels_used(self):
        tracker = ScriptedMatcherTracker({1: [48., 30.], 2: [48., 30.]})
        global_px = np.array([[40., 30.], [44., 30.], [45., 30.]])
        issued, final, stats = simulate_deferred_sequence(self.frames, global_px, tracker)
        self.assertEqual(stats['tier3'], 2)
        self.assertEqual(stats['tier2'], 0)
        np.testing.assert_array_equal(issued[1], global_px[1])
        np.testing.assert_array_equal(final[1], global_px[1])

    def test_bad_input_and_no_accidental_fusion(self):
        tracker = ScriptedMatcherTracker({1: [40., 30.], 2: [40., 30.]})
        globals_px = np.zeros((3, 2), dtype=float)
        with self.assertRaises(ValueError):
            simulate_deferred_sequence(self.frames, globals_px, tracker,
                                       tolerance_px=0.)
        with self.assertRaises(ValueError):
            simulate_deferred_sequence(self.frames, globals_px[:2], tracker)
        tracker.fusion_alpha = .75
        with self.assertRaises(ValueError):
            simulate_deferred_sequence(self.frames, globals_px, tracker)

    def test_validation_report_rejects_holdout_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            h5 = Path(directory) / 'test_200_200.h5'
            h5.write_bytes(b'not h5')
            with mock.patch('scripts.diagnose_3et_deferred_medium.load_stnet_checkpoint') as load:
                with self.assertRaisesRegex(ValueError, 'test_aug'):
                    evaluate_deferred_medium(h5, 'stnet.pt', 'matcher.pt',
                                             Path(directory) / 'report.json', device='cpu')
                load.assert_not_called()

    def test_validation_report_counts_all_frames_without_claiming_online_speed(self):
        class FixedLogits(torch.nn.Module):
            def forward(self, frames):
                x = torch.full((*frames.shape[:2], 80), -100.)
                y = torch.full((*frames.shape[:2], 60), -100.)
                x[..., 40] = 100.
                y[..., 30] = 100.
                return x, y

        from ktcal_paper.matching import PaperMatcher
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_aug.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((1, 2, 2, 60, 80), dtype=np.uint8)
                labels = np.zeros((1, 2, 3), dtype=np.float32)
                labels[..., :2] = [.5, .5]
                handle['label'] = labels
            stnet = root / 'stnet.pt'
            matcher = root / 'matcher.pt'
            stnet.write_bytes(b'mock')
            matcher.write_bytes(b'mock')
            path = root / 'report.json'
            with mock.patch('scripts.diagnose_3et_deferred_medium.load_stnet_checkpoint',
                            return_value=(FixedLogits(), {'readout': 'attention'}, 1.)), \
                 mock.patch('scripts.diagnose_3et_deferred_medium.load_matcher_checkpoint',
                            return_value=(PaperMatcher(), {})):
                report = evaluate_deferred_medium(
                    h5, stnet, matcher, path, device='cpu')
            self.assertEqual(report['frames'], 2)
            self.assertEqual(sum(report['stats']['tier%d' % i] for i in (1, 2, 3)), 2)
            self.assertEqual(report['stats']['global_predictions_precomputed'], 2)
            self.assertEqual(report['metrics']['proposed_corrected_ma11']['frames'], 2)
            self.assertFalse(report['latency_comparable'])
            self.assertFalse(report['final_holdout_used'])
            self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
