import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from ktcal_paper.combined_3et import _sha256, load_stnet_checkpoint
from ktcal_paper.stnet_sequence import decode_soft_expectation, masked_official_simdr_loss
from models.streaming_stnet import StreamingSTNet


class CurrentReadoutStreamingTests(unittest.TestCase):
    def setUp(self):
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        torch.manual_seed(87)

    def model(self):
        self.assertIsNotNone(importlib.util.find_spec('models.current_readout_streaming_stnet'),
                             'The direct current-spatial readout is missing')
        from models.current_readout_streaming_stnet import CurrentReadoutStreamingSTNet
        return CurrentReadoutStreamingSTNet(hidden_size=128).eval()

    @torch.inference_mode()
    def test_zero_scale_preserves_compatible_streaming_checkpoint(self):
        baseline = StreamingSTNet(hidden_size=128).eval()
        model = self.model()
        model.initialize_from_streaming(baseline.state_dict())
        frames = torch.rand(2, 6, 2, 60, 80)
        for actual, expected in zip(model(frames), baseline(frames)):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)

    @torch.inference_mode()
    def test_active_head_is_causal_and_matches_stream_steps_and_reset(self):
        model = self.model()
        model.current_readout_scale.fill_(.5)
        frames = torch.rand(1, 7, 2, 60, 80)
        expected = model(frames)
        x, y, state = [], [], None
        for frame in frames.unbind(1):
            features, state = model.stream_step(frame, state)
            self.assertEqual(features.shape, (1, 128 + 2048))
            a, b = model.coordinate_logits(features)
            x.append(a)
            y.append(b)
        self.assertEqual(len(state), 4)
        for actual, wanted in zip((torch.stack(x, 1), torch.stack(y, 1)), expected):
            torch.testing.assert_close(actual, wanted, atol=3e-5, rtol=1e-5)
        changed = frames.clone()
        changed[:, 3:] *= 30
        for a, b in zip(model(changed), expected):
            torch.testing.assert_close(a[:, :3], b[:, :3], atol=2e-6, rtol=1e-6)
        fresh, _ = model.stream_step(frames[:, 0], None)
        for a, b in zip(model.coordinate_logits(fresh), expected):
            torch.testing.assert_close(a, b[:, 0], atol=3e-5, rtol=1e-5)
        model.args.stnet_decode = 'soft'
        torch.testing.assert_close(model.last_step(frames),
                                   decode_soft_expectation(expected[0][:, -1], expected[1][:, -1]))

    def test_auxiliary_logits_train_even_at_zero_scale_and_invalid_frames_have_zero_gradient(self):
        model = self.model().train()
        frames = torch.rand(1, 4, 2, 60, 80)
        x, y, current_x, current_y = model.forward_with_current_logits(frames)
        self.assertEqual(current_x.shape, (1, 4, 80))
        self.assertEqual(current_y.shape, (1, 4, 60))
        current_x.retain_grad()
        current_y.retain_grad()
        labels = torch.full((1, 4, 2), .4)
        valid = torch.tensor([[True, True, True, False]])
        loss = masked_official_simdr_loss(current_x, current_y, labels, valid)
        loss.backward()
        self.assertGreater(float(model.current_readout[0].weight.grad.abs().sum()), 0)
        self.assertEqual(float(current_x.grad[:, 3].abs().sum()), 0)
        self.assertEqual(float(current_y.grad[:, 3].abs().sum()), 0)
        self.assertEqual(float(model.current_readout_scale), 0)

    def test_loader_restores_direct_readout_checkpoint(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save(dict(model_state_dict=model.state_dict(), config=dict(
                temporal_mode='causal_streaming', hidden_size=128, current_readout=True)), path)
            loaded, _, _ = load_stnet_checkpoint(path, 'cpu')
            self.assertIsInstance(loaded, type(model))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])

    def test_warm_teacher_augmented_training_counts_real_frames_and_updates_branch(self):
        self.model()
        from ktcal_paper.streaming_training import train_streaming_3et
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = np.random.RandomState(42).rand(2, 5, 2, 60, 80).astype(np.float32)
            labels = np.full((2, 5, 2), .4, dtype=np.float32)
            labels[:, 2, 0] = -.1
            for name in ('train_aug.h5', 'test_aug.h5'):
                with h5py.File(root / name, 'w') as handle:
                    handle['frames'], handle['label'] = frames, labels
            source = root / 'source.pt'
            torch.save(dict(model_state_dict=StreamingSTNet(hidden_size=128).state_dict(),
                            config=dict(temporal_mode='causal_streaming', hidden_size=128)), source)
            teacher = root / 'teacher.npz'
            np.savez(teacher, coordinates=labels, metadata=json.dumps(dict(split='train',
                source_h5_sha256=_sha256(root / 'train_aug.h5'), teacher_checkpoint_sha256='a' * 64)))
            train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5', output=root / 'run',
                initialize_from=source, current_readout=True, current_loss_weight=.25,
                flip_axis_px=80.,
                teacher_cache=teacher, teacher_weight=.2, warm_start_steps=3, augment=True,
                epochs=1, patience=1, batch_size=2, hidden_size=128, threads=1, device='cpu',
                lr=1e-4, min_lr=1e-4)
            history = json.loads((root / 'run/history.json').read_text())
            self.assertEqual(history[-1]['training_valid_timesteps'], 8)
            self.assertEqual(history[-1]['validation']['frames'], 10)
            self.assertGreater(history[-1]['train']['current_readout_loss'], 0)
            last = torch.load(root / 'run/last.pt', weights_only=False)
            self.assertTrue(last['config']['current_readout'])
            self.assertEqual(last['config']['warm_start_steps'], 3)
            self.assertEqual(last['config']['flip_axis_px'], 80.)
            self.assertNotEqual(float(last['model_state_dict']['current_readout_scale']), 0)
            np.testing.assert_allclose([g['lr'] for g in last['optimizer_state_dict']['param_groups']],
                                       [1e-4, 1e-3, 2e-3])
            with self.assertRaisesRegex(ValueError, 'exclusive'):
                train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                    output=root / 'invalid', current_readout=True, spatial_localizer=True, device='cpu')
            self.assertFalse((root / 'invalid').exists())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    @torch.inference_mode()
    def test_cuda_graph_active_branch_and_reset_match_eager_and_cache(self):
        from ktcal_paper.streaming_inference import StreamingPredictor, predict_streaming_sequences
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                                       benchmark=torch.backends.cudnn.benchmark,
                                       deterministic=torch.backends.cudnn.deterministic,
                                       allow_tf32=False):
            model = self.model().cuda()
            model.args.stnet_flip_axis_px = 80.
            model.current_readout_scale.fill_(.5)
            views = ((0, 0, False), (1, -1, True))
            eager = StreamingPredictor(model, views=views, cuda_graph=False, warm_start_steps=3)
            graph = StreamingPredictor(model, views=views, cuda_graph=True, warm_start_steps=3)
            frames = torch.rand(1, 7, 2, 60, 80, device='cuda')
            expected = predict_streaming_sequences(model, frames, views, warm_start_steps=3)
            for _ in range(2):
                eager.reset()
                graph.reset()
                for index in range(7):
                    eager.observe(frames[:, index])
                    graph.observe(frames[:, index])
                    prefix = frames[:, :index + 1]
                    actual = graph(prefix, np.array([True]))
                    np.testing.assert_allclose(actual, eager(prefix, np.array([True])), atol=1e-4, rtol=1e-5)
                    np.testing.assert_allclose(actual, expected[:, index], atol=1e-4, rtol=1e-5)
