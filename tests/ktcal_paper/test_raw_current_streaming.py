import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from models.current_readout_streaming_stnet import CurrentReadoutStreamingSTNet
from ktcal_paper.stnet_sequence import decode_soft_expectation, masked_official_simdr_loss


class RawCurrentStreamingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(97)

    def model(self, hidden=128):
        self.assertIsNotNone(importlib.util.find_spec('models.raw_current_streaming_stnet'),
                             'Shared raw-current spatial branch is missing')
        from models.raw_current_streaming_stnet import RawCurrentStreamingSTNet
        return RawCurrentStreamingSTNet(hidden_size=hidden).eval()

    @torch.inference_mode()
    def test_zero_raw_scale_preserves_current_checkpoint_for_both_hidden_sizes(self):
        for hidden in (128, 256):
            baseline = CurrentReadoutStreamingSTNet(hidden_size=hidden).eval()
            baseline.current_readout_scale.fill_(.4)
            model = self.model(hidden)
            model.initialize_from_current_readout(baseline.state_dict())
            frames = torch.rand(2, 4, 2, 60, 80)
            for actual, expected in zip(model(frames), baseline(frames)):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            bad = dict(baseline.state_dict())
            del bad['gru.bias_hh_l0']
            with self.assertRaisesRegex(ValueError, 'compatible'):
                model.initialize_from_current_readout(bad)

    @torch.inference_mode()
    def test_active_residual_full_step_future_reset_and_last_step_agree(self):
        model = self.model()
        model.current_readout_scale.fill_(.4)
        model.raw_current_scale.fill_(.7)
        frames = torch.rand(2, 6, 2, 60, 80)
        expected = model(frames)
        state, xs, ys = None, [], []
        for frame in frames.unbind(1):
            features, state = model.stream_step(frame, state)
            self.assertEqual(features.shape, (2, 128 + 2048 + 128))
            x, y = model.coordinate_logits(features)
            xs.append(x)
            ys.append(y)
        self.assertEqual(len(state), 4)
        for actual, wanted in zip((torch.stack(xs, 1), torch.stack(ys, 1)), expected):
            torch.testing.assert_close(actual, wanted, atol=3e-5, rtol=1e-5)
        changed = frames.clone()
        changed[:, 3:] *= 40
        for actual, wanted in zip(model(changed), expected):
            torch.testing.assert_close(actual[:, :3], wanted[:, :3], atol=2e-6, rtol=1e-5)
        fresh, _ = model.stream_step(frames[:, 0])
        for actual, wanted in zip(model.coordinate_logits(fresh), expected):
            torch.testing.assert_close(actual, wanted[:, 0], atol=3e-5, rtol=1e-5)
        torch.testing.assert_close(model.last_step(frames),
            decode_soft_expectation(expected[0][:, -1], expected[1][:, -1]))

    @torch.inference_mode()
    def test_raw_encoder_reuses_one_spatial_call_and_only_the_received_map(self):
        model = self.model()
        first = torch.rand(2, 2, 60, 80)
        calls, maps = [], []
        hook = model.block1.spatial.register_forward_hook(lambda _, inputs, output: calls.append(output))
        raw_hook = model.raw_current_encoder.register_forward_pre_hook(lambda _, inputs: maps.append(inputs[0]))
        try:
            features, state = model.stream_step(first)
            self.assertEqual(len(calls), 1)
            torch.testing.assert_close(maps[0], calls[0][:, :, 0])
            calls.clear()
            maps.clear()
            full = model.extract_temporal(first[:, None].expand(-1, 3, -1, -1, -1))
            self.assertEqual(len(calls), 1)
            self.assertEqual(full.shape, (2, 3, 2304))
            torch.testing.assert_close(maps[0], calls[0].permute(0, 2, 1, 3, 4).reshape(6, 32, 60, 80))
        finally:
            hook.remove()
            raw_hook.remove()

    def test_raw_auxiliary_trains_at_zero_scale_and_masks_invalid_targets(self):
        model = self.model().train()
        output = model.forward_with_raw_logits(torch.rand(1, 4, 2, 60, 80))
        x, y, current_x, current_y, raw_x, raw_y = output
        raw_x.retain_grad()
        raw_y.retain_grad()
        labels = torch.full((1, 4, 2), .4)
        labels[:, 3] = float('nan')
        valid = torch.tensor([[True, True, True, False]])
        loss = masked_official_simdr_loss(raw_x, raw_y, labels, valid)
        loss.backward()
        self.assertGreater(float(model.raw_current_encoder[0].weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(model.raw_current_encoder[0].weight.grad).all())
        self.assertEqual(float(raw_x.grad[:, 3].abs().sum()), 0)
        self.assertEqual(float(raw_y.grad[:, 3].abs().sum()), 0)
        self.assertEqual(float(model.raw_current_scale), 0)

    def test_loader_restores_raw_architecture_and_rejects_incompatible_flags(self):
        from ktcal_paper.combined_3et import load_stnet_checkpoint
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            config = dict(temporal_mode='causal_streaming', hidden_size=128,
                          current_readout=True, raw_current_readout=True)
            torch.save(dict(model_state_dict=model.state_dict(), config=config), path)
            loaded, _, _ = load_stnet_checkpoint(path, 'cpu')
            self.assertIsInstance(loaded, type(model))
            for changes in (dict(current_readout=False), dict(learned_initial_state=True),
                            dict(spatial_localizer=True), dict(temporal_mode='bidirectional')):
                torch.save(dict(model_state_dict=model.state_dict(), config=dict(config, **changes)), path)
                with self.assertRaises(ValueError):
                    load_stnet_checkpoint(path, 'cpu')

    def _training_case(self, device='cpu', cli=False):
        from ktcal_paper.combined_3et import _sha256, load_stnet_checkpoint
        from ktcal_paper.streaming_training import train_streaming_3et
        self.assertIn('raw_current_readout', inspect.signature(train_streaming_3et).parameters)
        self.assertIn('raw_current_loss_weight', inspect.signature(train_streaming_3et).parameters)
        model_class = type(self.model())
        captured = []
        original = model_class.forward_with_raw_logits
        def record(instance, frames):
            result = original(instance, frames)
            if instance.training:
                for logits in result[-2:]:
                    logits.retain_grad()
                captured.append(result[-2:])
            return result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = np.random.RandomState(97).rand(2, 5, 2, 60, 80).astype(np.float32)
            labels = np.full((2, 5, 2), .4, dtype=np.float32)
            for name in ('train_aug.h5', 'test_aug.h5'):
                with h5py.File(root / name, 'w') as handle:
                    handle['frames'] = frames
                    target = labels.copy()
                    if name == 'train_aug.h5':
                        target[:, 1] = np.nan
                    handle['label'] = target
            source = CurrentReadoutStreamingSTNet(hidden_size=128)
            torch.save(dict(model_state_dict=source.state_dict(), config=dict(
                temporal_mode='causal_streaming', hidden_size=128, current_readout=True)), root / 'source.pt')
            np.savez(root / 'teacher.npz', coordinates=labels, metadata=json.dumps(dict(
                split='train', source_h5_sha256=_sha256(root / 'train_aug.h5'),
                teacher_checkpoint_sha256='a' * 64)))
            options = dict(output=root / 'run', initialize_from=root / 'source.pt',
                teacher_cache=root / 'teacher.npz', teacher_weight=.2, current_readout=True,
                raw_current_readout=True, raw_current_loss_weight=.3, warm_start_steps=3,
                epochs=1, patience=1, batch_size=2, hidden_size=128, threads=1,
                lr=1e-4, min_lr=1e-4, device=device)
            with patch.object(model_class, 'forward_with_raw_logits', record):
                if cli:
                    import sys
                    from scripts.train_streaming_3et import main
                    arguments = ['train_streaming_3et', '--train-h5', str(root / 'train_aug.h5'),
                                 '--val-h5', str(root / 'test_aug.h5')]
                    for key, value in options.items():
                        arguments.append('--' + key.replace('_', '-'))
                        if not isinstance(value, bool):
                            arguments.append(str(value))
                    with patch.object(sys, 'argv', arguments):
                        main()
                else:
                    train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5', **options)
            checkpoint = torch.load(root / 'run/last.pt', weights_only=False)
            history = json.loads((root / 'run/history.json').read_text())
            self.assertEqual(history[-1]['training_valid_timesteps'], 8)
            self.assertGreater(history[-1]['train']['raw_current_readout_loss'], 0)
            self.assertTrue(checkpoint['config']['raw_current_readout'])
            self.assertEqual(checkpoint['config']['raw_current_loss_weight'], .3)
            self.assertNotEqual(float(checkpoint['model_state_dict']['raw_current_scale']), 0)
            ids = [i for group in checkpoint['optimizer_state_dict']['param_groups'] for i in group['params']]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), len(list(model_class(hidden_size=128).parameters())))
            np.testing.assert_allclose([g['lr'] for g in checkpoint['optimizer_state_dict']['param_groups']],
                                       [1e-4, 1e-3, 2e-3])
            self.assertEqual(len(captured), 1)
            for logits in captured[0]:
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertEqual(float(logits.grad[:, :3].abs().sum()), 0)
                self.assertEqual(float(logits.grad[:, 4].abs().sum()), 0)
                self.assertGreater(float(logits.grad[:, 3].abs().sum()), 0)
            loaded, _, _ = load_stnet_checkpoint(root / 'run/last.pt', 'cpu')
            self.assertIsInstance(loaded, model_class)
            for changes in (dict(current_readout=False), dict(learned_initial_state=True),
                            dict(spatial_localizer=True), dict(raw_current_loss_weight=-1)):
                with self.assertRaises(ValueError):
                    train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                                       **dict(options, output=root / 'invalid', **changes))
                self.assertFalse((root / 'invalid').exists())

    def test_teacher_warm_training_masks_raw_logits_and_records_all_parameter_groups(self):
        self._training_case()

    def test_actual_cli_enables_raw_training_and_auxiliary_weight(self):
        self._training_case(cli=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_raw_teacher_training_has_finite_gradients(self):
        self._training_case(device='cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    @torch.inference_mode()
    def test_cuda_graph_warm_multiview_partial_reset_matches_causal_sequence(self):
        from ktcal_paper.streaming_inference import StreamingPredictor, predict_streaming_sequences
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                                       benchmark=torch.backends.cudnn.benchmark,
                                       deterministic=torch.backends.cudnn.deterministic,
                                       allow_tf32=False):
            model = self.model().cuda()
            model.args.stnet_flip_axis_px = 80.
            model.current_readout_scale.fill_(.4)
            model.raw_current_scale.fill_(.7)
            views = ((0, 0, False), (2, -1, True))
            graph = StreamingPredictor(model, views=views, cuda_graph=True, warm_start_steps=3)
            frames = torch.rand(2, 5, 2, 60, 80, device='cuda')
            expected = predict_streaming_sequences(model, frames, views, warm_start_steps=3)
            for _ in range(2):
                graph.reset()
                for index in range(5):
                    graph.observe(frames[:, index])
                    mask = np.array([index % 2 == 0, True])
                    actual = graph(frames[:, :index + 1], mask)
                    np.testing.assert_allclose(actual, expected[mask, index], atol=2e-4, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
