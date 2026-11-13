import inspect
import unittest

import numpy as np
import torch

from ktcal_paper.matching import PaperMatcher
from ktcal_paper.offline_batch import OfflineBatchTracker


class ActivityOnlineGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def tracker(self, threshold=10., device='cpu', graph=False):
        self.assertIn('activity_reset_threshold', inspect.signature(OfflineBatchTracker).parameters)
        return OfflineBatchTracker(PaperMatcher(4).to(device).eval(), device=device,
                                   activity_reset_threshold=threshold, cuda_graph=graph)

    def test_current_activity_forces_large_reset_and_skips_unneeded_matcher(self):
        tracker = self.tracker()
        matched = []
        class Reliable:
            def __call__(self, template, image, center):
                matched.append(image.clone())
                return torch.cat((center, torch.ones(len(center), 1),
                                  torch.full((len(center), 1), 10.), torch.zeros(len(center), 1)), -1)
        tracker.matcher = Reliable()
        frames = torch.zeros(1, 4, 2, 60, 80)
        frames[:, 1, 0, 0, 0] = 20.
        frames[:, 3, 1, 0, 0] = 20.
        calls = []
        coordinates = {1: [40., 30.], 2: [45., 30.], 4: [55., 30.]}
        def predict(history, mask):
            calls.append(history.shape[1])
            return np.tile(coordinates[history.shape[1]], (int(mask.sum()), 1))
        output, stats = tracker.track(frames, predict, return_trace=True)
        np.testing.assert_array_equal(output[0, [0, 1, 3]], [[40., 30.], [45., 30.], [55., 30.]])
        self.assertEqual(calls, [1, 2, 4])
        self.assertEqual(len(matched), 1)
        self.assertEqual(stats['matcher_requests'], 1)
        self.assertEqual(stats['activity_forced_large_frames'], 2)
        self.assertEqual(stats['stnet_requests'], 3)
        self.assertEqual(stats['small_fusion_calls'], 0)
        self.assertEqual(stats['tier1'], 1)
        self.assertTrue(np.isnan(stats['innovations'][0, [1, 3]]).all())

    def test_mixed_batches_match_only_low_activity_and_remain_independent(self):
        tracker = self.tracker()
        seen = []
        class Reliable:
            def __call__(self, template, image, center):
                seen.append(image[:, 0, 0, 0].cpu().numpy().copy())
                return torch.cat((center, torch.ones(len(center), 1, device=center.device),
                                  torch.full((len(center), 1), 10., device=center.device),
                                  torch.zeros(len(center), 1, device=center.device)), -1)
        tracker.matcher = Reliable()
        frames = torch.zeros(2, 3, 2, 60, 80)
        frames[0, 1, 0, 0, 0] = 20.
        frames[1, 1, 0, 0, 0] = 1.
        frames[0, 2, 0, 0, 0] = 2.
        frames[1, 2, 0, 0, 0] = 30.
        requests = []
        def predict(history, mask):
            requests.append(mask.copy())
            return np.array([[40., 30.], [50., 25.]])[mask] + history.shape[1] - 1
        batch, stats = tracker.track(frames, predict)
        np.testing.assert_array_equal(requests, [[True, True], [True, False], [False, True]])
        np.testing.assert_array_equal(seen, [[1.], [2.]])
        self.assertEqual(stats['matcher_requests'], 2)
        for record in range(2):
            single = self.tracker()
            single.matcher = Reliable()
            expected = single.track(frames[record:record + 1], lambda history, mask:
                np.array([[40., 30.], [50., 25.]])[record:record + 1] + history.shape[1] - 1)[0]
            np.testing.assert_array_equal(batch[record:record + 1], expected)

    def test_default_and_high_threshold_preserve_existing_causal_prefix_and_reset(self):
        self.tracker()
        torch.manual_seed(19)
        frames = torch.rand(1, 6, 2, 60, 80)
        matcher = PaperMatcher(4).eval()
        def predict(history, mask):
            return np.tile([40. + float(history[0, -1, 0, 0, 0]), 30.], (int(mask.sum()), 1))
        baseline = OfflineBatchTracker(matcher).track(frames, predict)[0]
        high = OfflineBatchTracker(matcher, activity_reset_threshold=1e9).track(frames, predict)[0]
        np.testing.assert_array_equal(baseline, high)
        tracker = OfflineBatchTracker(matcher, activity_reset_threshold=4000.)
        original = tracker.track(frames, predict)[0]
        changed = frames.clone()
        changed[:, 3:] *= 30
        np.testing.assert_array_equal(tracker.track(changed, predict)[0][:, :3], original[:, :3])
        np.testing.assert_array_equal(tracker.track(frames, predict)[0], original)

    def test_nonfinite_or_nonpositive_activity_threshold_is_rejected(self):
        self.tracker()
        for threshold in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.tracker(threshold)

    def test_cli_and_json_config_accept_current_activity_gate_and_validate_threshold(self):
        import json
        from pathlib import Path
        import tempfile
        from scripts.evaluate_3et_online import parse_args, validate_args
        args = parse_args(['--output', 'unused.json', '--activity-reset-threshold', '500'])
        validate_args(args)
        self.assertEqual(args.activity_reset_threshold, 500.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(dict(activity_reset_threshold=200.)))
            args = parse_args(['--config', str(path), '--output', 'unused.json'])
            self.assertEqual(args.activity_reset_threshold, 200.)
        for threshold in ('0', '-1', 'nan', 'inf'):
            args = parse_args(['--output', 'unused.json', '--activity-reset-threshold', threshold])
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_all_forced_warm_record_initializes_matcher_before_measurement(self):
        from scripts import evaluate_3et_online as evaluator
        self.assertTrue(hasattr(evaluator, 'warm_up_online_tracker'),
                        'All-forced warm records must independently initialize the matcher')
        tracker = self.tracker()
        original_matcher, phases = tracker.matcher, []
        phase = ['warm']
        class RecordingMatcher:
            def __call__(self, *args):
                phases.append(phase[0])
                return original_matcher(*args)
        tracker.matcher = RecordingMatcher()
        def predict(history, mask):
            return np.tile([40., 30.], (int(mask.sum()), 1))
        high = torch.ones(1, 3, 2, 60, 80)
        class Predictor:
            def reset(self):
                pass
            __call__ = staticmethod(predict)
        evaluator.warm_up_online_tracker(tracker, Predictor(), high)
        self.assertEqual(phases, ['warm'])
        phase[0] = 'measured'
        tracker.track(torch.zeros_like(high), predict)
        self.assertEqual(phases[0], 'warm')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_all_forced_warm_record_has_matcher_graph_before_low_record(self):
        from scripts.evaluate_3et_online import warm_up_online_tracker
        tracker = self.tracker(device='cuda', graph=True)
        class Predictor:
            def reset(self):
                pass
            def __call__(self, history, mask):
                return np.tile([40., 30.], (int(mask.sum()), 1))
        high = torch.ones(1, 3, 2, 60, 80)
        warm_up_online_tracker(tracker, Predictor(), high)
        self.assertIsNotNone(tracker.matcher.graph)
        graph = tracker.matcher.graph
        _, stats = tracker.track(torch.zeros_like(high), Predictor())
        self.assertIs(tracker.matcher.graph, graph)
        self.assertEqual(stats['matcher_requests'], 2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_graph_mixed_activity_batches_match_eager(self):
        torch.manual_seed(37)
        matcher = PaperMatcher(4).cuda().eval()
        frames = torch.zeros(2, 6, 2, 60, 80)
        frames[0, 1::2] = 1.
        frames[1, 2::2] = 1.
        def predict(history, mask):
            return np.array([[40., 30.], [30., 20.]])[mask] + history.shape[1] * .1
        eager = OfflineBatchTracker(matcher, device='cuda', activity_reset_threshold=10.)
        graph = OfflineBatchTracker(matcher, device='cuda', activity_reset_threshold=10.,
            cuda_graph=True, template_cuda_graph=True, synchronize_frames=True)
        a, sa = eager.track(frames, predict)
        b, sb = graph.track(frames, predict)
        np.testing.assert_allclose(b, a, atol=1e-5, rtol=0)
        self.assertEqual(sa['matcher_requests'], sb['matcher_requests'])
        self.assertEqual(sa['stnet_requests'], sb['stnet_requests'])
        self.assertEqual(sa['activity_forced_large_frames'], sb['activity_forced_large_frames'])


if __name__ == '__main__':
    unittest.main()
