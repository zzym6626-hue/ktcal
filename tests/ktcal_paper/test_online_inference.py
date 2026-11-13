import unittest

import numpy as np
import torch

from ktcal_paper.online_inference import CausalPredictor, CausalViewDecoder, GraphDecoder, LastStepSTNet, causal_smooth


class CurrentPixel(torch.nn.Module):
    def forward(self, frames):
        values = frames[:, -1, 0, 0, 0]
        return torch.stack((values / 80., values / 60.), -1)


class OnlineInferenceTests(unittest.TestCase):
    def test_synthetic_right_context_repeats_received_current_frame_and_reads_original_index(self):
        class Inspect(torch.nn.Module):
            def forward(self, frames):
                self.received = frames[0, :, 0, 0, 0].tolist()
                x = torch.full((len(frames), frames.shape[1], 80), -100.)
                y = torch.full((len(frames), frames.shape[1], 60), -100.)
                for step in range(frames.shape[1]):
                    x[:, step, step] = 100.
                    y[:, step, step] = 100.
                return x, y

        model = Inspect()
        frames = torch.zeros(1, 3, 2, 60, 80)
        frames[:, 1] = 1.
        frames[:, 2] = 2.
        decoded = CausalViewDecoder(model, right_context=2)(frames)
        self.assertEqual(model.received, [0., 1., 2., 2., 2.])
        np.testing.assert_array_equal(decoded, [[2., 2.]])

    def test_last_step_optimization_preserves_all_supported_readouts(self):
        from models.STNet import STNet
        from types import SimpleNamespace

        torch.set_num_threads(1)
        torch.manual_seed(11)
        with torch.inference_mode():
            for readout in ('last', 'attention', 'final_hidden'):
                source = STNet(SimpleNamespace(stnet_readout=readout)).eval()
                optimized = LastStepSTNet(source).eval()
                for length in (1, 4, 16):
                    frames = torch.rand(2, length, 2, 60, 80)
                    expected = source(frames)
                    actual = optimized(frames)
                    for reference, value in zip(expected, actual):
                        torch.testing.assert_close(value[:, -1], reference[:, -1],
                                                   atol=1e-5, rtol=1e-5)

    def test_current_step_optimization_preserves_synthetic_context_predictions(self):
        from models.STNet import STNet
        from types import SimpleNamespace

        torch.set_num_threads(1)
        with torch.inference_mode():
            for readout in ('last', 'attention', 'final_hidden'):
                source = STNet(SimpleNamespace(stnet_readout=readout)).eval()
                for context in (1, 2, 6):
                    optimized = LastStepSTNet(source, current_offset=context)
                    frames = torch.rand(1, 8, 2, 60, 80)
                    eager = CausalViewDecoder(source, right_context=context)(frames)
                    fast = CausalViewDecoder(optimized, right_context=context)(frames)
                    torch.testing.assert_close(fast, eager, atol=1e-4, rtol=1e-5)

    def test_soft_decoder_uses_current_logits_only(self):
        class Logits(torch.nn.Module):
            def forward(self, frames):
                x = torch.full((len(frames), frames.shape[1], 80), -100.)
                y = torch.full((len(frames), frames.shape[1], 60), -100.)
                x[:, :-1, 10] = 100.
                y[:, :-1, 15] = 100.
                x[:, -1, 30] = 100.
                y[:, -1, 20] = 100.
                return x, y

        decoder = CausalViewDecoder(Logits())
        np.testing.assert_allclose(decoder(torch.zeros(2, 3, 2, 60, 80)),
                                   [[30., 20.], [30., 20.]])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_graph_matches_eager_and_replays_new_input(self):
        model = CurrentPixel().cuda()
        graph = GraphDecoder(model)
        for value in (20., 35., 40.):
            frames = torch.full((1, 4, 2, 60, 80), value, device='cuda')
            torch.testing.assert_close(graph(frames), model(frames))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_predictor_return_is_not_aliased_by_next_graph_replay(self):
        predictor = CausalPredictor(GraphDecoder(CurrentPixel().cuda()), window=4, stride=2)
        history = torch.zeros(1, 2, 2, 60, 80)
        predictor.observe(torch.full((1, 2, 60, 80), 20., device='cuda'))
        first = predictor(history[:, :1], np.array([True]))
        snapshot = first.copy()
        predictor.observe(torch.full((1, 2, 60, 80), 35., device='cuda'))
        second = predictor(history, np.array([True]))
        np.testing.assert_array_equal(first, snapshot)
        self.assertFalse(np.array_equal(first, second))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_real_stnet_graph_matches_eager_for_changed_histories(self):
        from models.STNet import STNet
        from types import SimpleNamespace

        torch.manual_seed(20261113)
        model = STNet(SimpleNamespace(stnet_readout='attention')).cuda().eval()
        decoder = CausalViewDecoder(model, views=((0, 0, False), (0, 0, True))).cuda()
        graph = GraphDecoder(decoder)
        with torch.inference_mode():
            for _ in range(3):
                frames = torch.rand(1, 16, 2, 60, 80, device='cuda')
                torch.testing.assert_close(graph(frames).clone(), decoder(frames),
                                           atol=1e-4, rtol=1e-5)

    def test_ring_contains_only_received_strided_frames(self):
        class ReadHistory(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.history = []

            def forward(self, frames):
                self.history.append(frames[0, :, 0, 0, 0].tolist())
                return torch.zeros(len(frames), 2)

        model = ReadHistory()
        predictor = CausalPredictor(model, window=4, stride=2)
        frames = torch.empty(1, 15, 2, 60, 80)
        for step in range(15):
            frames[:, step] = step
            predictor.observe(frames[:, step])
            if step % 2 == 0:
                predictor(frames[:, :step + 1], np.array([True]))
                self.assertEqual(model.history[-1], [max(0, step - gap) for gap in (6, 4, 2, 0)])
        predictor.reset()
        predictor.observe(torch.full((1, 2, 60, 80), 90.))
        predictor(frames[:, :1], np.array([True]))
        self.assertEqual(model.history[-1], [90.] * 4)

    def test_decoder_undoes_shift_and_flip_before_even_view_median(self):
        class Peak(torch.nn.Module):
            def forward(self, frames):
                flat = frames[:, -1, 0].flatten(1)
                peak = flat.argmax(-1)
                return torch.stack(((peak % 80).float() / 80.,
                                    (peak // 80).float() / 60.), -1)

        frames = torch.zeros(1, 3, 2, 60, 80)
        frames[:, :, 0, 20, 30] = 1.
        decoder = CausalViewDecoder(Peak(), views=((0, 0, False), (2, 0, True)),
                                    ensemble='median')
        np.testing.assert_allclose(decoder(frames).numpy(), [[30., 20.]], atol=1e-5)

    def test_output_smoothing_is_invariant_to_future_changes(self):
        original = np.array([[10., 10.], [11., 10.], [12., 10.], [30., 30.]])
        changed = original.copy()
        changed[3] = [-1000., 1000.]
        a = causal_smooth(original, alpha=.5, max_step=2.5)
        b = causal_smooth(changed, alpha=.5, max_step=2.5)
        np.testing.assert_array_equal(a[:3], b[:3])
        np.testing.assert_array_equal(a[0], original[0])
        np.testing.assert_array_equal(a[3], original[3])


if __name__ == '__main__':
    unittest.main()
