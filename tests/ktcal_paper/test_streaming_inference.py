import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from ktcal_paper.combined_3et import load_stnet_checkpoint
from ktcal_paper.online_inference import CausalViewDecoder
from models.streaming_stnet import StreamingSTNet


class StreamingInferenceTests(unittest.TestCase):
    def setUp(self):
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        torch.manual_seed(29)

    def predictor(self, model, graph=False):
        self.assertIsNotNone(importlib.util.find_spec('ktcal_paper.streaming_inference'),
                             'Streaming inference is missing')
        from ktcal_paper.streaming_inference import StreamingPredictor
        return StreamingPredictor(model, views=((0, 0, False), (0, 0, True)),
                                  temperature=1., ensemble='mean', cuda_graph=graph)

    @torch.inference_mode()
    def test_requested_coordinates_match_full_causal_view_history(self):
        model = StreamingSTNet(hidden_size=128).eval()
        predictor = self.predictor(model)
        reference = CausalViewDecoder(model, views=((0, 0, False), (0, 0, True)),
                                     temperature=1., ensemble='mean')
        frames = torch.rand(1, 7, 2, 60, 80)
        requests = 0
        for index in range(7):
            predictor.observe(frames[:, index])
            if index % 3 == 0:
                actual = predictor(frames[:, :index + 1], np.array([True]))
                np.testing.assert_allclose(actual, reference(frames[:, :index + 1]).numpy(),
                                           atol=2e-5, rtol=1e-5)
                requests += 1
        self.assertEqual(predictor.feature_updates, 7)
        self.assertEqual(predictor.coordinate_head_requests, requests)

    @torch.inference_mode()
    def test_reset_prevents_record_leakage_and_results_do_not_alias(self):
        model = StreamingSTNet(hidden_size=128).eval()
        predictor = self.predictor(model)
        frames = torch.rand(1, 4, 2, 60, 80)
        predictor.observe(frames[:, 0])
        first = predictor(frames[:, :1], np.array([True]))
        snapshot = first.copy()
        for index in range(1, 4):
            predictor.observe(frames[:, index])
            predictor(frames[:, :index + 1], np.array([True]))
        np.testing.assert_array_equal(first, snapshot)
        predictor.reset()
        predictor.observe(frames[:, 0])
        np.testing.assert_allclose(predictor(frames[:, :1], np.array([True])), first, atol=1e-6)
        self.assertEqual(predictor.feature_updates, 1)

    @torch.inference_mode()
    def test_request_must_match_received_prefix(self):
        predictor = self.predictor(StreamingSTNet(hidden_size=128).eval())
        frames = torch.rand(1, 2, 2, 60, 80)
        predictor.observe(frames[:, 0])
        with self.assertRaisesRegex(ValueError, 'received'):
            predictor(frames, np.array([True]))
        empty = predictor(frames[:, :1], np.array([False]))
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(predictor.coordinate_head_requests, 0)

    def test_checkpoint_loader_detects_streaming_architecture(self):
        model = StreamingSTNet(hidden_size=128).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'streaming.pt'
            torch.save(dict(model_state_dict=model.state_dict(),
                            config=dict(temporal_mode='causal_streaming', hidden_size=128,
                                        unidirectional=True, readout='last', flip_axis_px=80.)), path)
            actual, config, divisor = load_stnet_checkpoint(path, 'cpu')
            self.assertIsInstance(actual, StreamingSTNet)
            self.assertEqual(divisor, 1.)
            self.assertEqual(config['temporal_mode'], 'causal_streaming')
            self.assertEqual(actual.args.stnet_flip_axis_px, 80.)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, actual.state_dict()[key])

    def test_loader_rejects_invalid_flip_axis_before_selection_can_use_it(self):
        model = StreamingSTNet(hidden_size=128).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.pt'
            for axis in (81., 78., float('nan'), float('inf'), None):
                torch.save(dict(model_state_dict=model.state_dict(), config=dict(
                    temporal_mode='causal_streaming', hidden_size=128, flip_axis_px=axis)), path)
                with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, 'flip'):
                    load_stnet_checkpoint(path, 'cpu')

    @torch.inference_mode()
    def test_selection_cache_retains_full_history_and_agrees_with_streaming(self):
        from ktcal_paper.streaming_inference import predict_streaming_sequences
        model = StreamingSTNet(hidden_size=128).eval()
        frames = torch.rand(2, 6, 2, 60, 80)
        views = ((0, 0, False), (0, 0, True))
        cache = predict_streaming_sequences(model, frames, views, temperature=1.,
                                            ensemble='mean', batch_size=1)
        predictor = self.predictor(model)
        actual = []
        for i in range(6):
            predictor.observe(frames[:, i])
            actual.append(predictor(frames[:, :i + 1], np.ones(2, dtype=bool)))
        np.testing.assert_allclose(cache, np.stack(actual, 1), atol=3e-5, rtol=1e-5)
        changed = frames.clone()
        changed[:, 3:] = torch.rand_like(changed[:, 3:]) * 20
        changed_cache = predict_streaming_sequences(model, changed, views, temperature=1.,
                                                    ensemble='mean', batch_size=1)
        np.testing.assert_allclose(cache[:, :3], changed_cache[:, :3], atol=1e-5, rtol=1e-6)

    @torch.inference_mode()
    def test_explicit_checkpoint_flip_axis_matches_cache_and_stateful_requests(self):
        from ktcal_paper.streaming_inference import StreamingPredictor, predict_streaming_sequences
        import inspect
        self.assertIn('flip_axis_px', inspect.signature(StreamingPredictor).parameters)
        model = StreamingSTNet(hidden_size=128).eval()
        model.args.stnet_flip_axis_px = 80.
        frames = torch.rand(2, 5, 2, 60, 80)
        views = ((0, 0, False), (2, -1, True))
        legacy = predict_streaming_sequences(model, frames, views, flip_axis_px=79., warm_start_steps=3)
        expected = predict_streaming_sequences(model, frames, views, warm_start_steps=3)
        np.testing.assert_allclose(expected - legacy, np.broadcast_to([.5, 0.], legacy.shape), atol=1e-5)
        predictor = StreamingPredictor(model, views=views, cuda_graph=False, warm_start_steps=3)
        self.assertEqual(predictor.flip_axis_px, 80.)
        for index in range(5):
            predictor.observe(frames[:, index])
            mask = np.array([True, index % 2 == 0])
            np.testing.assert_allclose(predictor(frames[:, :index + 1], mask), expected[mask, index],
                                       atol=3e-5, rtol=1e-5)
        for axis in (78., 81., float('nan')):
            with self.assertRaisesRegex(ValueError, 'flip'):
                StreamingPredictor(model, flip_axis_px=axis)

    @torch.inference_mode()
    def test_warm_start_repeats_only_first_received_frame_and_counts_its_work(self):
        from ktcal_paper.streaming_inference import StreamingPredictor, predict_streaming_sequences
        model = StreamingSTNet(hidden_size=128).eval()
        frames = torch.rand(1, 4, 2, 60, 80)
        views = ((0, 0, False), (0, 0, True))
        predictor = StreamingPredictor(model, views=views, temperature=1., ensemble='mean',
                                       cuda_graph=False, warm_start_steps=3)
        expected = predict_streaming_sequences(model, frames, views, temperature=1.,
                                                ensemble='mean', warm_start_steps=3)
        actual = []
        for i in range(4):
            predictor.observe(frames[:, i])
            actual.append(predictor(frames[:, :i + 1], np.array([True])))
        np.testing.assert_allclose(np.stack(actual, 1), expected, atol=3e-5, rtol=1e-5)
        self.assertEqual(predictor.feature_forward_passes, 7)
        self.assertEqual(predictor.feature_updates, 7)
        self.assertEqual(predictor.received_frame_updates, 4)
        self.assertEqual(predictor.synthetic_feature_updates, 3)
        first = actual[0].copy()
        predictor.reset()
        predictor.observe(frames[:, 0])
        np.testing.assert_allclose(predictor(frames[:, :1], np.array([True])), first, atol=1e-5)

    @torch.inference_mode()
    def test_graph_step_can_include_shift_flip_and_normalization(self):
        from ktcal_paper.streaming_inference import ViewedStreamingStep
        from ktcal_paper.offline_experiment import spatial_view
        model = StreamingSTNet(hidden_size=128).eval()
        frame = torch.rand(2, 2, 60, 80)
        views = ((1, -1, False), (-1, 1, True))
        adapter = ViewedStreamingStep(model, views, divisor=2.)
        state = adapter.initial_state(frame)
        explicit = torch.cat([spatial_view(frame, v) for v in views], 0) / 2.
        expected, expected_state = model.stream_step(explicit, state)
        actual, actual_state = adapter.stream_step(frame, state)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(actual_state, expected_state):
            torch.testing.assert_close(a, b)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    @torch.inference_mode()
    def test_cuda_replay_includes_preprocessing_without_new_python_view_calls(self):
        from ktcal_paper import streaming_inference
        model = StreamingSTNet(hidden_size=128).cuda().eval()
        predictor = self.predictor(model, graph=True)
        frames = torch.rand(1, 2, 2, 60, 80, device='cuda')
        with mock.patch.object(streaming_inference, 'spatial_view', wraps=streaming_inference.spatial_view) as views:
            predictor.observe(frames[:, 0])
            calls = views.call_count
            self.assertGreater(calls, 0)
            predictor.observe(frames[:, 1])
            self.assertEqual(views.call_count, calls)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    @torch.inference_mode()
    def test_cuda_graph_state_replay_and_reset_matches_eager(self):
        model = StreamingSTNet(hidden_size=128).cuda().eval()
        eager = self.predictor(model)
        graph = self.predictor(model, graph=True)
        frames = torch.rand(1, 10, 2, 60, 80, device='cuda')
        for start in (0, 2):
            eager.reset()
            graph.reset()
            for index in range(7):
                frame = frames[:, start + index]
                eager.observe(frame)
                graph.observe(frame)
                prefix = frames[:, start:start + index + 1]
                np.testing.assert_allclose(graph(prefix, np.array([True])),
                                           eager(prefix, np.array([True])), atol=1e-4, rtol=1e-5)
