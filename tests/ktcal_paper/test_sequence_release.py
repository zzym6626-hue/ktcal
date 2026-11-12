"""Exercise the published sequence CLI and its real default gate behavior."""
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


class SequenceReleaseTests(unittest.TestCase):
    def release(self):
        # Absence is a failed release contract rather than an import-time error.
        self.assertIsNotNone(importlib.util.find_spec('scripts.evaluate_3et'),
                             'The documented public evaluator is missing')
        from scripts import evaluate_3et
        return evaluate_3et

    def test_documented_command_has_a_complete_default_configuration(self):
        module = self.release()
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                module.main(['--help'])
        self.assertEqual(result.exception.code, 0)
        args = module.build_parser().parse_args([])
        self.assertTrue(Path(args.stnet).is_file())
        self.assertTrue(Path(args.matcher).is_file())

    def test_default_gate_updates_kf_without_reading_low_tier_stnet(self):
        module = self.release()
        from ktcal_paper.offline_batch import OfflineBatchTracker
        args = module.build_parser().parse_args(['--device', 'cpu'])
        frames = torch.zeros(1, 2, 2, 60, 80)
        globals_px = np.array([[[40., 30.], [1000., -1000.]]])
        tracker = OfflineBatchTracker(DisplacementMatcher(1., .3, 2.6),
                                      **module.tracker_options(args))
        result, stats = tracker.track(frames, globals_px)
        # Pxx after prediction = 10 + 10 + .05; measurement covariance = 32.
        np.testing.assert_allclose(result[0, 1], [40. + 20.05 / 52.05, 30.], atol=1e-6)
        self.assertEqual(stats['tier1'], 1)
        self.assertEqual(stats['small_fusion_calls'], 0)
        self.assertEqual(stats['stnet_requests'], 1)

    def test_default_reliability_gate_rejects_low_confidence_and_psr(self):
        module = self.release()
        from ktcal_paper.offline_batch import OfflineBatchTracker
        args = module.build_parser().parse_args(['--device', 'cpu'])
        frames = torch.zeros(1, 2, 2, 60, 80)
        globals_px = np.array([[[40., 30.], [50., 30.]]])
        for confidence, psr in ((.279, 3.), (.3, 2.54)):
            with self.subTest(confidence=confidence, psr=psr):
                tracker = OfflineBatchTracker(DisplacementMatcher(1., confidence, psr),
                                              **module.tracker_options(args))
                result, stats = tracker.track(frames, globals_px)
                np.testing.assert_allclose(result[0, 1], [50., 30.])
                self.assertEqual(stats['tier3'], 2)
                self.assertEqual(stats['fusion_calls'], 0)

    def test_invalid_config_rejected_before_dataset_or_weights_are_opened(self):
        module = self.release()
        for flag, value in (('--temperature', 'nan'), ('--confidence', 'nan'),
                            ('--q-pos', '0'), ('--q-vel', 'inf'), ('--r-px', '-1'),
                            ('--fusion-alpha', 'nan'), ('--min-psr', 'nan'),
                            ('--tolerance', '0'), ('--batch-size', '0')):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                module.main([flag, value, '--h5', '/missing/test_aug.h5'])

    def test_holdout_refused_before_open_and_existing_report_preserved(self):
        module = self.release()
        with self.assertRaisesRegex(ValueError, 'final holdout'):
            module.main(['--h5', '/missing/test_200_200.h5'])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'report.json'
            output.write_text('existing result')
            with self.assertRaises(FileExistsError):
                module.main(['--output', str(output)])
            self.assertEqual(output.read_text(), 'existing result')

    def test_terminal_summary_includes_scores_timing_and_raw_gate_ratios(self):
        module = self.release()
        metric = {'frames': 100, 'mean_error_px': 1.6, 'p3': .91, 'p5': .96, 'p10': .99}
        report = {'metrics': {'stnet': metric, 'ktcal': metric},
                  'latency': [{'stnet_ms': 30., 'stnet_filtered_ms': 31.}],
                  'median_ms_per_frame': 1.05,
                  'tracker_stats': {'frames': 100, 'tier1': 60, 'tier2': 1, 'tier3': 39,
                                    'small_fusion_calls': 0}}
        with redirect_stdout(io.StringIO()) as out:
            module.print_summary(report, 'result.json')
        self.assertIn('91.00%', out.getvalue())
        self.assertIn('96.00%', out.getvalue())
        self.assertIn('99.00%', out.getvalue())
        self.assertIn('1.050', out.getvalue())
        self.assertIn('60.00%', out.getvalue())
        self.assertIn('result.json', out.getvalue())


class DisplacementMatcher(torch.nn.Module):
    """Controlled matcher observation; tracker/KF/defaults remain real."""
    def __init__(self, distance, confidence=.9, psr=10., border=False):
        super().__init__()
        self.distance, self.confidence, self.psr, self.border = distance, confidence, psr, border

    def match_with_diagnostics(self, template, search, subpixel=False):
        batch = search.shape[0]
        span = search.shape[-1] - template.shape[-1]
        local = search.new_tensor([.5 + self.distance / span, .5]).expand(batch, 2)
        return local, search.new_full((batch,), self.confidence), None, {
            'border_peak': torch.full((batch,), self.border, dtype=torch.bool, device=search.device),
            'psr': search.new_full((batch,), self.psr),
        }

class SequenceStatePolicyTests(unittest.TestCase):
    def tracker(self, distance, confidence=.9):
        from ktcal_paper.offline_batch import OfflineBatchTracker
        from scripts.evaluate_3et import build_parser, tracker_options
        return OfflineBatchTracker(DisplacementMatcher(distance, confidence),
            **tracker_options(build_parser().parse_args(['--device', 'cpu'])))

    def test_medium_refines_only_within_one_pixel_of_updated_kf(self):
        frames = torch.zeros(1, 2, 2, 60, 80)
        kf_x = 40. + 3.5 * 20.05 / 52.05
        for global_x, rollbacks, refinements in ((kf_x + .5, 0, 1), (50., 1, 0)):
            globals_px = np.array([[[40., 30.], [global_x, 30.]]])
            result, stats = self.tracker(3.5).track(frames, globals_px)
            self.assertEqual(stats['tier2'], 1)
            self.assertEqual(stats['medium_rollbacks'], rollbacks)
            self.assertEqual(stats['medium_refinements'], refinements)
            self.assertEqual(stats['fusion_calls'], refinements)
            self.assertEqual(stats['small_fusion_calls'], 0)
            if rollbacks:
                np.testing.assert_allclose(result[0, 1], globals_px[0, 1])
            else:
                self.assertLess(kf_x, result[0, 1, 0])
                self.assertLess(result[0, 1, 0], global_x)

    def test_low_global_perturbations_cannot_change_entire_raw_trajectory(self):
        frames = torch.zeros(1, 10, 2, 60, 80)
        globals_px = np.ones((1, 10, 2)) * np.array([40., 30.])
        tracker = self.tracker(.5)
        raw, stats = tracker.track(frames, globals_px, return_trace=True)
        changed = globals_px.copy()
        changed[stats['tiers'] == 1] = [1000., -1000.]
        altered, other = tracker.track(frames, changed, return_trace=True)
        np.testing.assert_array_equal(raw, altered)
        np.testing.assert_array_equal(stats['tiers'], other['tiers'])
        self.assertEqual(other['small_fusion_calls'], 0)


if __name__ == '__main__':
    unittest.main()
