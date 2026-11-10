import threading
import unittest
import numpy as np
import torch
from ktcal_paper.tracker import PaperTracker


class ScriptedMatcher(torch.nn.Module):
    def __init__(self,release,stale):
        super().__init__(); self.calls=0; self.release=release; self.stale=stale

    def forward(self,template,search):
        self.calls+=1
        if self.calls==1: return torch.tensor([[0.,0.]]),torch.tensor([.9]),None
        if not self.stale: self.release.set()
        return torch.tensor([[.5,.5]]),torch.tensor([.1 if self.stale else .9]),None


class ScriptedTracker(PaperTracker):
    def __init__(self,stale):
        self.release=threading.Event(); self.worker_thread=None
        super().__init__(torch.nn.Identity(),ScriptedMatcher(self.release,stale))

    def _infer(self,window):
        index=int(window[-1,0,0,0])
        if index==1:
            self.worker_thread=threading.current_thread().ident
            if not self.release.wait(2): raise AssertionError('Fast path blocked waiting for worker')
        if index==2: self.release.set()
        return np.array([40.+10*index,30.]),0.


class TrackerTests(unittest.TestCase):
    def test_gate_trace_distinguishes_skip_match_and_reset_without_changing_output(self):
        class ConfidenceMatcher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, template, search):
                self.calls += 1
                confidence = .9 if self.calls == 1 else .1
                return torch.tensor([[.5, .5]]), torch.tensor([confidence]), None

        frames = torch.zeros(5, 2, 60, 80)
        global_px = np.tile(np.array([[40., 30.]]), (5, 1))
        global_px[-1] = [42., 30.]
        baseline, _, normal_stats = PaperTracker(
            torch.nn.Identity(), ConfidenceMatcher(),
            max_prediction_variance_px2=25.).track(
                frames, global_predictions=global_px)
        online, _, stats = PaperTracker(
            torch.nn.Identity(), ConfidenceMatcher(),
            max_prediction_variance_px2=25., record_gate_trace=True).track(
                frames, global_predictions=global_px)
        np.testing.assert_array_equal(online, baseline)
        self.assertNotIn('gate_trace', normal_stats)
        trace = stats['gate_trace']
        self.assertEqual([row['route'] for row in trace], [
            'stnet_reset', 'prediction_only', 'matched_update',
            'prediction_only', 'stnet_reset'])
        self.assertIsNone(trace[1]['match_px'])
        np.testing.assert_allclose(trace[2]['prediction_px'], [40., 30.])
        np.testing.assert_allclose(trace[2]['match_px'], [40., 30.])
        self.assertLess(trace[4]['confidence'], .5)
        np.testing.assert_allclose(trace[4]['output_px'], [42., 30.])

    def test_prediction_only_fast_path_is_opt_in_and_preserves_replay(self):
        class CenterMatcher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, template, search):
                self.calls += 1
                return torch.tensor([[.5, .5]]), torch.tensor([.9]), None

        frames = torch.zeros(4, 2, 60, 80)
        globals_px = np.tile(np.array([[40., 30.]]), (4, 1))
        normal_matcher = CenterMatcher()
        normal, _, normal_stats = PaperTracker(
            torch.nn.Identity(), normal_matcher).track(frames, global_predictions=globals_px)
        fast_matcher = CenterMatcher()
        fast, corrected, stats = PaperTracker(
            torch.nn.Identity(), fast_matcher, max_prediction_variance_px2=25.,
            max_consecutive_prediction_skips=1).track(
                frames, global_predictions=globals_px)

        np.testing.assert_array_equal(fast, normal)
        np.testing.assert_array_equal(corrected, fast)
        self.assertEqual(normal_matcher.calls, 3)
        self.assertEqual(normal_stats['skipped_matches'], 0)
        self.assertEqual(stats['frame_tiers'], [3, 1, 1, 1])
        self.assertEqual(stats['skipped_matches'], 2)
        self.assertEqual(fast_matcher.calls, 1)
        self.assertEqual(stats['stnet_calls'], 1)

    def test_prediction_fast_path_rejects_bad_limits_and_matches_when_uncertain(self):
        class CenterMatcher(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.calls = 0

            def forward(self, template, search):
                self.calls += 1
                return torch.tensor([[.5, .5]]), torch.tensor([.9]), None

        for limit, count in [(-1., 1), (float('nan'), 1), (25., 0)]:
            with self.subTest(limit=limit, count=count), self.assertRaises(ValueError):
                PaperTracker(torch.nn.Identity(), CenterMatcher(),
                             max_prediction_variance_px2=limit,
                             max_consecutive_prediction_skips=count)
        matcher = CenterMatcher()
        _, _, stats = PaperTracker(
            torch.nn.Identity(), matcher, max_prediction_variance_px2=1.).track(
                torch.zeros(3, 2, 60, 80),
                global_predictions=np.tile(np.array([[40., 30.]]), (3, 1)))
        self.assertEqual(stats['skipped_matches'], 0)
        self.assertEqual(matcher.calls, 2)

    def test_external_global_predictions_drive_resets_without_calling_model(self):
        class RaisingModel(torch.nn.Module):
            def forward(self, frames):
                raise AssertionError('model must not run for precomputed predictions')

        class LowConfidenceMatcher(torch.nn.Module):
            def forward(self, template, search):
                return torch.tensor([[.5, .5]]), torch.tensor([.1]), None

        frames = torch.zeros(3, 2, 60, 80)
        predictions = np.array([[10., 20.], [11., 21.], [12., 22.]])
        online, corrected, stats = PaperTracker(
            RaisingModel(), LowConfidenceMatcher()).track(
                frames, global_predictions=predictions)

        np.testing.assert_allclose(online, predictions)
        np.testing.assert_allclose(corrected, predictions)
        self.assertEqual(stats['tier3'], 3)
        self.assertEqual(stats['stnet_calls'], 3)

    def test_external_global_predictions_are_validated_before_tracking(self):
        tracker = PaperTracker(torch.nn.Identity(), torch.nn.Identity())
        frames = torch.zeros(3, 2, 60, 80)
        invalid = (
            np.zeros((2, 2)),
            np.zeros((3, 3)),
            np.array([[1., 2.], [3., np.nan], [5., 6.]]),
        )
        for predictions in invalid:
            with self.subTest(shape=predictions.shape):
                with self.assertRaisesRegex(ValueError, 'global predictions'):
                    tracker.track(frames, global_predictions=predictions)

    def test_accuracy_fusion_blends_stnet_with_the_kalman_output_only(self):
        class FixedModel(torch.nn.Module):
            def forward(self,frames):
                index=float(frames[0,-1,0,0,0])
                return torch.tensor([[(40.+10.*index)/80.,.5]])

        class CenterMatcher(torch.nn.Module):
            def forward(self,template,search):
                return torch.tensor([[.5,.5]]),torch.tensor([.9]),None

        frames=torch.stack([torch.full((2,60,80),float(i)) for i in range(3)])
        plain,_,stats=PaperTracker(FixedModel(),CenterMatcher()).track(frames)
        fused,_,fused_stats=PaperTracker(FixedModel(),CenterMatcher(),fusion_alpha=.5).track(frames)
        np.testing.assert_allclose(plain[1],[40.,30.])
        np.testing.assert_allclose(fused[1],[45.,30.])
        np.testing.assert_allclose(fused[2],[50.,30.])
        self.assertEqual(fused_stats['fusion_calls'],2)
        self.assertEqual(fused_stats['stnet_calls'],3)
        self.assertEqual(stats['stnet_calls'],1)

    def test_medium_correction_is_asynchronous_and_online_output_is_immutable(self):
        frames=torch.stack([torch.full((2,60,80),float(i)) for i in range(3)])
        tracker=ScriptedTracker(False)
        online,corrected,stats=tracker.track(frames)
        self.assertEqual(stats['tier2'],1)
        self.assertNotEqual(tracker.worker_thread,threading.current_thread().ident)
        np.testing.assert_array_equal(online[1],[40,30])
        np.testing.assert_array_equal(corrected[1],[50,30])
        self.assertEqual(stats['async_applied'],1)
        self.assertEqual(stats['async_events'][0]['source_frame'],1)
        self.assertGreaterEqual(stats['correction_application_ms'][0],stats['correction_completion_ms'][0])
        self.assertTrue(np.isfinite(online).all() and np.isfinite(corrected).all())

    def test_sync_reset_discards_older_async_observation(self):
        frames=torch.stack([torch.full((2,60,80),float(i)) for i in range(3)])
        online,corrected,stats=ScriptedTracker(True).track(frames)
        self.assertEqual(stats['tier3'],2)
        self.assertEqual(stats['async_discarded'],1)
        self.assertEqual(stats['async_applied'],0)
        np.testing.assert_array_equal(online[2],[60,30])
        np.testing.assert_array_equal(corrected[2],[60,30])


if __name__=='__main__': unittest.main()
