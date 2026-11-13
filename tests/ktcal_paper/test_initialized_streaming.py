import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import h5py


class InitializedStreamingTests(unittest.TestCase):
    def model_class(self):
        self.assertIsNotNone(importlib.util.find_spec('models.initialized_streaming_stnet'),
                             'Learned current-frame GRU initialization is missing')
        from models.initialized_streaming_stnet import InitializedCurrentReadoutStreamingSTNet
        return InitializedCurrentReadoutStreamingSTNet

    def model(self, hidden=256):
        torch.set_num_threads(1)
        from models.current_readout_streaming_stnet import CurrentReadoutStreamingSTNet
        args = SimpleNamespace(stnet_decode='soft')
        original = CurrentReadoutStreamingSTNet(args, hidden_size=hidden).eval()
        with torch.no_grad():
            original.current_readout_scale.fill_(.7)
        model = self.model_class()(args, hidden_size=hidden).eval()
        model.initialize_from_current_readout(original.state_dict())
        return original, model

    @torch.inference_mode()
    def test_zero_initial_scale_reproduces_current_checkpoint_both_hidden_sizes(self):
        for hidden in (128, 256):
            with self.subTest(hidden=hidden):
                source, model = self.model(hidden)
                frames = torch.rand(2, 7, 2, 60, 80)
                for expected, actual in zip(source(frames), model(frames)):
                    # The cached latent changes Linear's input strides. Its
                    # equivalent GEMM differs by <1e-6 in FP32 logits on CPU.
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(float(model.initial_hidden_scale), 0.)
                torch.testing.assert_close(source.extract_temporal(frames)[..., :hidden],
                    model.extract_temporal(frames)[..., :hidden], rtol=0., atol=0.)
                self.assertEqual(model.extract_temporal(frames).shape, (2, 7, hidden + 256))

    @torch.inference_mode()
    def test_active_initialization_full_step_future_and_last_decoder(self):
        _, model = self.model()
        model.initial_hidden_scale.fill_(.8)
        frames = torch.rand(2, 12, 2, 60, 80)
        full = model(frames)
        state, xs, ys = None, [], []
        for index in range(12):
            features, state = model.stream_step(frames[:, index], state)
            self.assertEqual(len(state), 5)
            self.assertTrue(bool((state[4] == 1).all()))
            x, y = model.coordinate_logits(features)
            xs.append(x)
            ys.append(y)
        torch.testing.assert_close(torch.stack(xs, 1), full[0], rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(torch.stack(ys, 1), full[1], rtol=2e-5, atol=3e-6)
        changed = frames.clone()
        changed[:, 7:] *= 20
        for before, after in zip(full, model(changed)):
            torch.testing.assert_close(before[:, :7], after[:, :7], rtol=2e-5, atol=3e-6)
        from ktcal_paper.stnet_sequence import decode_soft_expectation
        torch.testing.assert_close(model.last_step(frames),
            decode_soft_expectation(full[0][:, -1], full[1][:, -1]))

    @torch.inference_mode()
    def test_marker_seeds_hidden_once_and_reset_isolates_stream(self):
        _, model = self.model(128)
        model.initial_hidden_scale.fill_(.9)
        frame = torch.rand(2, 2, 60, 80)
        initial = model.initial_state(frame)
        self.assertEqual(initial[4].shape, (1, 2, 1))
        self.assertTrue(bool((initial[4] == 0).all()))
        first, state = model.stream_step(frame, initial)
        second, _ = model.stream_step(frame, state)
        reseeded, _ = model.stream_step(frame, state[:4] + (initial[4],))
        self.assertGreater(float((second - reseeded).abs().max()), 1e-4)
        reset, _ = model.stream_step(frame)
        torch.testing.assert_close(first, reset, rtol=0., atol=0.)

    @torch.inference_mode()
    def test_partial_views_warm_reset_agree_with_cache(self):
        from ktcal_paper.streaming_inference import StreamingPredictor, predict_streaming_sequences
        _, model = self.model(128)
        model.initial_hidden_scale.fill_(.6)
        frames = torch.rand(2, 6, 2, 60, 80)
        views = ((0, 0, False), (0, 0, True))
        expected = predict_streaming_sequences(model, frames, views, batch_size=2, warm_start_steps=3)
        predictor = StreamingPredictor(model, views=views, cuda_graph=False, warm_start_steps=3)
        for index in range(6):
            predictor.observe(frames[:, index])
            actual = predictor(frames[:, :index + 1], np.array([False, True]))
            np.testing.assert_allclose(actual, expected[1:2, index], rtol=2e-5, atol=3e-5)
        predictor.reset()
        predictor.observe(frames[:, 0])
        np.testing.assert_allclose(predictor(frames[:, :1], np.array([True, True])),
                                   expected[:, 0], rtol=2e-5, atol=3e-5)

    def test_loader_selects_initialized_architecture_and_rejects_invalid_flags(self):
        from ktcal_paper.combined_3et import load_stnet_checkpoint
        _, model = self.model(128)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            config = dict(temporal_mode='causal_streaming', hidden_size=128,
                          current_readout=True, learned_initial_state=True)
            torch.save(dict(model_state_dict=model.state_dict(), config=config), path)
            loaded, _, _ = load_stnet_checkpoint(path, 'cpu')
            self.assertIsInstance(loaded, self.model_class())
            frame = torch.rand(1, 2, 60, 80)
            self.assertEqual(len(loaded.initial_state(frame)), 5)
            config['current_readout'] = False
            torch.save(dict(model_state_dict=model.state_dict(), config=config), path)
            with self.assertRaisesRegex(ValueError, 'initial'):
                load_stnet_checkpoint(path, 'cpu')

    def test_initialized_training_resumes_current_head_and_records_lr_groups(self):
        from ktcal_paper.streaming_training import train_streaming_3et
        from ktcal_paper.combined_3et import load_stnet_checkpoint
        self.assertIn('learned_initial_state', inspect.signature(train_streaming_3et).parameters)
        source, _ = self.model(128)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = np.zeros((2, 5, 2, 60, 80), dtype=np.float32)
            frames[..., 20, 30] = 1.
            labels = np.full((2, 5, 2), [.3, .3], dtype=np.float32)
            for name in ('train_aug.h5', 'test_aug.h5'):
                with h5py.File(root / name, 'w') as handle:
                    handle['frames'] = frames
                    current_labels = labels.copy()
                    if name == 'train_aug.h5':
                        current_labels[:, 1] = np.nan
                    handle['label'] = current_labels
            torch.save(dict(model_state_dict=source.state_dict(),
                config=dict(temporal_mode='causal_streaming', hidden_size=128, current_readout=True)),
                root / 'source.pt')
            result = train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                output=root / 'run', initialize_from=root / 'source.pt', current_readout=True,
                learned_initial_state=True, warm_start_steps=3, decode_temperature=.65,
                cold_boost=1., epochs=1, patience=1, batch_size=2, hidden_size=128,
                device='cpu', threads=1)
            last = torch.load(root / 'run/last.pt', weights_only=False)
            history = json.loads((root / 'run/history.json').read_text())
            self.assertTrue(last['config']['learned_initial_state'])
            self.assertEqual(last['config']['initial_hidden_lr_multiplier'], 10.)
            self.assertEqual(last['config']['initial_hidden_scale_lr_multiplier'], 20.)
            self.assertEqual(history[-1]['training_valid_timesteps'], 8)
            self.assertEqual(result['metrics']['frames'], 10)
            self.assertGreater(history[-1]['train']['current_readout_loss'], 0.)
            self.assertNotEqual(float(last['model_state_dict']['initial_hidden_scale']), 0.)
            ids = [i for group in last['optimizer_state_dict']['param_groups'] for i in group['params']]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), len(list(self.model_class()(hidden_size=128).parameters())))
            loaded, _, _ = load_stnet_checkpoint(root / 'run/last.pt', 'cpu')
            self.assertIsInstance(loaded, self.model_class())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    @torch.inference_mode()
    def test_cuda_five_states_active_init_partial_views_warm_and_reset(self):
        from ktcal_paper.streaming_inference import StreamingPredictor, predict_streaming_sequences
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                                       benchmark=torch.backends.cudnn.benchmark,
                                       deterministic=torch.backends.cudnn.deterministic,
                                       allow_tf32=False):
            _, model = self.model()
            model = model.cuda()
            model.initial_hidden_scale.fill_(.8)
            model.args.stnet_flip_axis_px = 80.
            views = ((0, 0, False), (1, -1, True))
            frames = torch.rand(2, 7, 2, 60, 80, device='cuda')
            expected = predict_streaming_sequences(model, frames, views, batch_size=2, warm_start_steps=3)
            eager = StreamingPredictor(model, views=views, cuda_graph=False, warm_start_steps=3)
            graph = StreamingPredictor(model, views=views, cuda_graph=True, warm_start_steps=3)
            mask = np.array([False, True])
            for _ in range(2):
                eager.reset()
                graph.reset()
                for index in range(7):
                    eager.observe(frames[:, index])
                    graph.observe(frames[:, index])
                    prefix = frames[:, :index + 1]
                    actual = graph(prefix, mask)
                    np.testing.assert_allclose(actual, eager(prefix, mask), atol=1e-4, rtol=1e-5)
                    np.testing.assert_allclose(actual, expected[1:2, index], atol=1e-4, rtol=1e-5)
            self.assertEqual(len(graph.state), 5)
