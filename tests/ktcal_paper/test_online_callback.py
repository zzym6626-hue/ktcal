import unittest

import numpy as np
import torch

from ktcal_paper.matching import PaperMatcher
from ktcal_paper.offline_batch import OfflineBatchTracker, BatchedTemplateCrop
from ktcal_paper.matching import TEMPLATE, crop_rectangle
from ktcal_paper.online_inference import CausalPredictor, CausalViewDecoder, CausalOutputFilter


class OnlineCallbackTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_template_graph_crops_match_eager_and_do_not_alias_previous_template(self):
        crop = BatchedTemplateCrop(cuda_graph=True)
        frame = torch.rand(2, 2, 60, 80, device='cuda')
        centers = torch.tensor([[40., 30.], [10., 20.]], device='cuda')
        first = crop(frame, centers)
        snapshot = first.clone()
        second = crop(frame + 1., centers + 1.)
        torch.testing.assert_close(second, crop_rectangle(frame + 1., centers + 1., TEMPLATE))
        torch.testing.assert_close(first, snapshot)

    def test_complete_tracker_prefix_is_invariant_to_future_frame_changes(self):
        class CurrentPixel(torch.nn.Module):
            def forward(self, frames):
                current = frames[:, -1, 0, 0, 0]
                return torch.stack(((40. + current) / 80.,
                                    (30. + current) / 60.), -1)

        torch.manual_seed(17)
        frames = torch.rand(1, 8, 2, 60, 80)
        changed = frames.clone()
        changed[:, 5:] = torch.rand_like(changed[:, 5:]) * 100.
        matcher = PaperMatcher(4).eval()

        def run(images):
            predictor = CausalPredictor(CausalViewDecoder(CurrentPixel()), window=3, stride=2)
            tracker = OfflineBatchTracker(matcher, confidence_threshold=1.)
            return tracker.track(images, predictor, output_transform=CausalOutputFilter(.5))[0]

        np.testing.assert_array_equal(run(frames)[:, :5], run(changed)[:, :5])

    def test_callback_receives_only_past_frames_and_matches_precomputed_resets(self):
        torch.manual_seed(13)
        frames = torch.zeros(1, 5, 2, 60, 80)
        for step in range(5):
            frames[:, step] = step
        predictions = np.array([[[40. + step, 30.] for step in range(5)]])
        matcher = PaperMatcher(4).eval()
        options = dict(confidence_threshold=1., search_size=(37, 32))
        expected, offline_stats = OfflineBatchTracker(matcher, **options).track(frames, predictions)
        received = []

        def predict(history, mask):
            step = history.shape[1] - 1
            self.assertEqual(history[0, :, 0, 0, 0].tolist(), list(range(step + 1)))
            received.append(step)
            return predictions[mask, step]

        actual, stats = OfflineBatchTracker(matcher, **options).track(frames, predict)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(received, list(range(5)))
        self.assertEqual(stats['stnet_requests'], offline_stats['stnet_requests'])
        self.assertEqual(stats['global_predictions_precomputed'], 0)
        self.assertEqual(stats['model_forward_passes'], len(received))

    def test_small_tier_never_calls_global_predictor(self):
        class ReliableMatch:
            def __call__(self, template, image, center):
                return torch.cat((center, torch.ones(len(center), 1),
                                  torch.full((len(center), 1), 10.),
                                  torch.zeros(len(center), 1)), -1)

        calls = []
        tracker = OfflineBatchTracker(PaperMatcher(4))
        tracker.matcher = ReliableMatch()

        def predict(history, mask):
            calls.append(history.shape[1])
            return np.tile([40., 30.], (int(mask.sum()), 1))

        output, stats = tracker.track(torch.zeros(1, 5, 2, 60, 80), predict)
        np.testing.assert_array_equal(output, np.tile([40., 30.], (1, 5, 1)))
        self.assertEqual(calls, [1])
        self.assertEqual(stats['tier1'], 4)
        self.assertEqual(stats['model_forward_passes'], 1)


if __name__ == '__main__':
    unittest.main()
