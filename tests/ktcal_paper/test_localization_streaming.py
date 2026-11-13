import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import h5py

import torch
import numpy as np

from models.streaming_stnet import StreamingSTNet
from ktcal_paper.combined_3et import load_stnet_checkpoint


class LocalizationStreamingTests(unittest.TestCase):
    def setUp(self):
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        torch.manual_seed(73)

    def model(self):
        self.assertIsNotNone(importlib.util.find_spec('models.localization_streaming_stnet'))
        from models.localization_streaming_stnet import LocalizationStreamingSTNet
        return LocalizationStreamingSTNet(hidden_size=128).eval()

    @torch.inference_mode()
    def test_zero_residual_scale_reproduces_pretrained_causal_model(self):
        base = StreamingSTNet(hidden_size=128).eval()
        model = self.model()
        model.initialize_from_streaming(base.state_dict())
        frames = torch.rand(1, 7, 2, 60, 80)
        for a, b in zip(model(frames), base(frames)):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-6)

    @torch.inference_mode()
    def test_full_sequence_steps_and_future_prefix_agree_with_active_spatial_head(self):
        model = self.model()
        model.localizer_scale.fill_(.5)
        frames = torch.rand(1, 7, 2, 60, 80)
        expected = model(frames)
        state, x, y = None, [], []
        for frame in frames.unbind(1):
            features, state = model.stream_step(frame, state)
            a, b = model.coordinate_logits(features)
            x.append(a)
            y.append(b)
        for a, b in zip(expected, (torch.stack(x, 1), torch.stack(y, 1))):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=1e-5)
        changed = frames.clone()
        changed[:, 3:] *= 20
        for a, b in zip(expected, model(changed)):
            torch.testing.assert_close(a[:, :3], b[:, :3], atol=2e-6, rtol=1e-6)
        self.assertEqual(len(state), 4)
        fresh, _ = model.stream_step(frames[:, 0], None)
        torch.testing.assert_close(fresh, model.extract_temporal(frames)[:, 0], atol=3e-5, rtol=1e-5)

    def test_dense_spatial_loss_backpropagates_with_zero_residual_scale(self):
        from ktcal_paper.streaming_training import spatial_heatmap_loss
        model = self.model().train()
        frames = torch.rand(1, 4, 2, 60, 80)
        x, y, heatmap = model.forward_with_spatial_logits(frames)
        self.assertEqual(heatmap.shape, (1, 4, 20, 26))
        labels = torch.full((1, 4, 2), .4)
        valid = torch.tensor([[True, True, True, False]])
        heatmap.retain_grad()
        loss = spatial_heatmap_loss(heatmap, labels, valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.localizer[0].weight.grad.abs().sum()), 0)
        self.assertEqual(float(heatmap.grad[:, 3].abs().sum()), 0)

    def test_loader_selects_localization_architecture(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(dict(model_state_dict=model.state_dict(), config=dict(
                temporal_mode='causal_streaming', hidden_size=128, spatial_localizer=True)), path)
            loaded, _, _ = load_stnet_checkpoint(path, 'cpu')
            self.assertIsInstance(loaded, type(model))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])

    @torch.inference_mode()
    def test_last_step_decodes_active_localization_head(self):
        from ktcal_paper.stnet_sequence import decode_soft_expectation
        model = self.model()
        model.localizer_scale.fill_(.5)
        frames = torch.rand(2, 5, 2, 60, 80)
        x, y = model(frames)
        torch.testing.assert_close(model.last_step(frames),
                                   decode_soft_expectation(x[:, -1], y[:, -1]))
        model.args.stnet_decode = 'argmax'
        expected = torch.stack((x[:, -1].argmax(-1) / 80., y[:, -1].argmax(-1) / 60.), -1)
        torch.testing.assert_close(model.last_step(frames), expected)

    def test_trainer_updates_spatial_head_and_records_learning_rates(self):
        from ktcal_paper.streaming_training import train_streaming_3et
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = np.random.RandomState(42).rand(2, 5, 2, 60, 80).astype(np.float32)
            labels = np.full((2, 5, 2), .4, dtype=np.float32)
            for name in ('train_aug.h5', 'test_aug.h5'):
                with h5py.File(root / name, 'w') as handle:
                    handle['frames'], handle['label'] = frames, labels
            source = root / 'source.pt'
            torch.save(dict(model_state_dict=StreamingSTNet(hidden_size=128).state_dict(),
                            config=dict(temporal_mode='causal_streaming', hidden_size=128)), source)
            train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5', output=root / 'run',
                                initialize_from=source, hidden_size=128, spatial_localizer=True,
                                epochs=1, patience=1, batch_size=2, threads=1, device='cpu',
                                lr=1e-4, min_lr=1e-4)
            history = json.loads((root / 'run/history.json').read_text())
            self.assertGreater(history[-1]['train']['spatial_heatmap_loss'], 0)
            self.assertEqual(history[-1]['training_valid_timesteps'], 10)
            last = torch.load(root / 'run/last.pt', weights_only=False)
            self.assertTrue(last['config']['spatial_localizer'])
            self.assertNotEqual(float(last['model_state_dict']['localizer_scale']), 0)
            rates = [group['lr'] for group in last['optimizer_state_dict']['param_groups']]
            np.testing.assert_allclose(rates, [1e-4, 1e-3, 2e-3])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    @torch.inference_mode()
    def test_cuda_graph_localization_features_and_reset_match_eager(self):
        from ktcal_paper.streaming_inference import StreamingPredictor, predict_streaming_sequences
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                                       benchmark=torch.backends.cudnn.benchmark,
                                       deterministic=torch.backends.cudnn.deterministic,
                                       allow_tf32=False):
            model = self.model().cuda()
            model.localizer_scale.fill_(.5)
            views = ((0, 0, False), (1, -1, True))
            eager = StreamingPredictor(model, views=views, cuda_graph=False)
            graph = StreamingPredictor(model, views=views, cuda_graph=True)
            frames = torch.rand(1, 8, 2, 60, 80, device='cuda')
            expected = predict_streaming_sequences(model, frames, views)
            for _ in range(2):
                eager.reset()
                graph.reset()
                for index in range(8):
                    eager.observe(frames[:, index])
                    graph.observe(frames[:, index])
                    prefix = frames[:, :index + 1]
                    actual = graph(prefix, np.array([True]))
                    np.testing.assert_allclose(actual, eager(prefix, np.array([True])), atol=1e-4, rtol=1e-5)
                    np.testing.assert_allclose(actual, expected[:, index], atol=1e-4, rtol=1e-5)
