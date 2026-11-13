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


class MemoryReadoutStreamingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(113)

    def model(self, hidden=128, window=8):
        self.assertIsNotNone(importlib.util.find_spec('models.memory_readout_streaming_stnet'),
                             'Received-only memory coordinate readout is missing')
        from models.memory_readout_streaming_stnet import MemoryReadoutStreamingSTNet
        return MemoryReadoutStreamingSTNet(hidden_size=hidden, memory_window=window).eval()

    @torch.inference_mode()
    def test_zero_memory_scale_preserves_current_checkpoint_and_strict_initialization(self):
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
    def test_active_memory_full_step_future_reset_last_step_and_window_agree(self):
        for hidden, window in ((128, 2), (128, 8), (256, 4)):
            model = self.model(hidden, window)
            model.current_readout_scale.fill_(.4)
            model.memory_scale.fill_(.7)
            frames = torch.rand(2, 9, 2, 60, 80)
            expected = model(frames)
            state, xs, ys = None, [], []
            for frame in frames.unbind(1):
                features, state = model.stream_step(frame, state)
                self.assertEqual(features.shape, (2, hidden + 2048 + window * (hidden + 1)))
                x, y = model.coordinate_logits(features)
                xs.append(x)
                ys.append(y)
            self.assertEqual(len(state), 5)
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
    def test_left_history_is_invalid_and_only_received_hidden_is_appended_once(self):
        model = self.model(window=4)
        image = torch.rand(2, 2, 60, 80)
        initial = model.initial_state(image)
        self.assertEqual(len(initial), 5)
        self.assertEqual(initial[-1].shape, (2, 4, 129))
        self.assertEqual(float(initial[-1].abs().sum()), 0)
        calls = []
        hook = model.block1.spatial.register_forward_hook(lambda *args: calls.append(1))
        try:
            first, state = model.stream_step(image, initial)
            self.assertEqual(len(calls), 1)
            torch.testing.assert_close(state[-1][..., -1], torch.tensor([[0., 0., 0., 1.]]).expand(2, -1))
            torch.testing.assert_close(state[-1][:, -1, :-1], first[:, :128])
            self.assertEqual(float(initial[-1].abs().sum()), 0)
            saved = state[-1].clone()
            second, next_state = model.stream_step(image * 2, state)
            torch.testing.assert_close(state[-1], saved)
            torch.testing.assert_close(next_state[-1][:, -2, :-1], first[:, :128])
            torch.testing.assert_close(next_state[-1][:, -1, :-1], second[:, :128])
            features = model.extract_temporal(torch.stack((image, image * 2), 1))
            torch.testing.assert_close(features[:, 0], first, atol=3e-5, rtol=1e-5)
            torch.testing.assert_close(features[:, 1], second, atol=3e-5, rtol=1e-5)
            # Invalid history content must never affect attention.
            changed = first.clone()
            memory = changed[:, 128 + 2048:].reshape(2, 4, 129)
            memory[:, :-1, :-1] = 1000
            for actual, wanted in zip(model.memory_logits(changed), model.memory_logits(first)):
                torch.testing.assert_close(actual, wanted)
        finally:
            hook.remove()

    def test_memory_auxiliary_trains_at_zero_scale_with_finite_nan_masked_gradients(self):
        model = self.model().train()
        output = model.forward_with_memory_logits(torch.rand(1, 4, 2, 60, 80))
        memory_x, memory_y = output[-2:]
        memory_x.retain_grad()
        memory_y.retain_grad()
        labels = torch.full((1, 4, 2), .4)
        labels[:, 3] = float('nan')
        valid = torch.tensor([[True, True, True, False]])
        masked_official_simdr_loss(memory_x, memory_y, labels, valid).backward()
        for parameter in (model.memory_query.weight, model.memory_key.weight,
                          model.memory_value.weight, model.memory_head[0].weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(float(memory_x.grad[:, 3].abs().sum()), 0)
        self.assertEqual(float(memory_y.grad[:, 3].abs().sum()), 0)
        self.assertEqual(float(model.memory_scale), 0)

    def test_invalid_memory_windows_and_states_are_rejected(self):
        model = self.model()
        from models.memory_readout_streaming_stnet import MemoryReadoutStreamingSTNet
        for window in (True, 1, 33, 2.5):
            with self.assertRaises(ValueError):
                MemoryReadoutStreamingSTNet(memory_window=window)
        frame = torch.rand(1, 2, 60, 80)
        with self.assertRaises(ValueError):
            model.stream_step(frame, model.initial_state(frame)[:4])

    def test_loader_restores_memory_window_and_rejects_incompatible_architectures(self):
        from ktcal_paper.combined_3et import load_stnet_checkpoint
        model = self.model(window=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            config = dict(temporal_mode='causal_streaming', hidden_size=128,
                          current_readout=True, memory_readout=True, memory_window=4)
            torch.save(dict(model_state_dict=model.state_dict(), config=config), path)
            loaded, _, _ = load_stnet_checkpoint(path, 'cpu')
            self.assertIsInstance(loaded, type(model))
            self.assertEqual(loaded.memory_window, 4)
            for changes in (dict(current_readout=False), dict(learned_initial_state=True),
                            dict(raw_current_readout=True), dict(spatial_localizer=True),
                            dict(temporal_mode='bidirectional'), dict(memory_window=1),
                            dict(memory_window=2.5)):
                torch.save(dict(model_state_dict=model.state_dict(), config=dict(config, **changes)), path)
                with self.assertRaises(ValueError):
                    load_stnet_checkpoint(path, 'cpu')

    def _training_case(self, device='cpu', cli=False):
        from ktcal_paper.combined_3et import _sha256, load_stnet_checkpoint
        from ktcal_paper.streaming_training import train_streaming_3et
        for flag in ('memory_readout', 'memory_window', 'memory_loss_weight'):
            self.assertIn(flag, inspect.signature(train_streaming_3et).parameters)
        model_class = type(self.model())
        captured, models, groups = [], [], []
        original_forward, original_optimizer = model_class.forward_with_memory_logits, torch.optim.AdamW
        def record(instance, frames):
            result = original_forward(instance, frames)
            if instance.training:
                models.append(instance)
                for logits in result[-2:]:
                    logits.retain_grad()
                captured.append(result[-2:])
            return result
        def optimizer(params, *args, **kwargs):
            result = original_optimizer(params, *args, **kwargs)
            groups.extend(result.param_groups)
            return result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = np.random.RandomState(113).rand(2, 5, 2, 60, 80).astype(np.float32)
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
                split='train', source_h5_sha256=_sha256(root / 'train_aug.h5'), teacher_checkpoint_sha256='b' * 64)))
            options = dict(output=root / 'run', initialize_from=root / 'source.pt',
                teacher_cache=root / 'teacher.npz', teacher_weight=.2, current_readout=True,
                memory_readout=True, memory_window=4, memory_loss_weight=.3, warm_start_steps=3,
                epochs=1, patience=1, batch_size=2, hidden_size=128, threads=1,
                lr=1e-4, min_lr=1e-4, device=device)
            with patch.object(model_class, 'forward_with_memory_logits', record), patch('torch.optim.AdamW', optimizer):
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
            self.assertEqual(len(captured), 1)
            for logits in captured[0]:
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertEqual(float(logits.grad[:, :3].abs().sum()), 0)
                self.assertEqual(float(logits.grad[:, 4].abs().sum()), 0)
                self.assertGreater(float(logits.grad[:, 3].abs().sum()), 0)
            parameters = [p for group in groups for p in group['params']]
            self.assertEqual(len(parameters), len({id(p) for p in parameters}))
            self.assertEqual({id(p) for p in parameters}, {id(p) for p in models[0].parameters()})
            multipliers = {id(p): g['lr_multiplier'] for g in groups for p in g['params']}
            for name, parameter in models[0].named_parameters():
                expected = (20. if name in ('memory_scale', 'current_readout_scale') else
                            10. if name.startswith(('memory_', 'current_readout.')) else 1.)
                self.assertEqual(multipliers[id(parameter)], expected)
            checkpoint = torch.load(root / 'run/last.pt', weights_only=False)
            self.assertTrue(checkpoint['config']['memory_readout'])
            self.assertEqual(checkpoint['config']['memory_window'], 4)
            self.assertEqual(checkpoint['config']['memory_loss_weight'], .3)
            self.assertFalse(checkpoint['final_holdout_used'])
            loaded, _, _ = load_stnet_checkpoint(root / 'run/last.pt', 'cpu')
            self.assertIsInstance(loaded, model_class)
            history = json.loads((root / 'run/history.json').read_text())
            self.assertEqual(history[-1]['training_valid_timesteps'], 8)
            self.assertGreater(history[-1]['train']['memory_readout_loss'], 0)

    def test_actual_training_masks_prefix_nan_teacher_and_groups_parameters(self):
        self._training_case()

    def test_cli_exposes_memory_training_flags(self):
        self._training_case(cli=True)

    def test_trainer_rejects_invalid_memory_options_before_reading_data(self):
        from ktcal_paper.streaming_training import train_streaming_3et
        self.assertIn('memory_readout', inspect.signature(train_streaming_3et).parameters)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train_aug.h5', 'test_aug.h5'):
                with h5py.File(root / name, 'w') as handle:
                    handle['frames'] = np.zeros((1, 1, 2, 60, 80), dtype=np.float32)
                    handle['label'] = np.full((1, 1, 2), .4, dtype=np.float32)
            base = dict(output=root / 'run', current_readout=True, memory_readout=True)
            for changes in (dict(current_readout=False), dict(raw_current_readout=True),
                            dict(learned_initial_state=True), dict(spatial_localizer=True),
                            dict(memory_window=1), dict(memory_window=True),
                            dict(memory_loss_weight=-1), dict(memory_loss_weight=np.nan)):
                with self.assertRaises(ValueError):
                    train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5', **dict(base, **changes))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_memory_replay_warm_partial_requests_and_record_reset_agree_with_full(self):
        from ktcal_paper.streaming_inference import StreamingPredictor, predict_streaming_sequences
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                                       benchmark=torch.backends.cudnn.benchmark,
                                       deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
            model = self.model(window=4).cuda()
            with torch.inference_mode():
                model.memory_scale.fill_(.7)
                model.current_readout_scale.fill_(.4)
            frames = torch.rand(2, 6, 2, 60, 80)
            views = ((0, 0, False), (-1, 0, True))
            expected = predict_streaming_sequences(model, frames, views, warm_start_steps=3,
                                                    temperature=.65, ensemble='median', batch_size=2)
            predictor = StreamingPredictor(model, views=views, warm_start_steps=3,
                                            temperature=.65, ensemble='median', cuda_graph=True)
            for _ in range(2):
                predictor.reset()
                for i in range(6):
                    predictor.observe(frames[:, i].cuda())
                    mask = np.array([True, i % 2 == 0])
                    actual = predictor(frames[:, :i + 1], mask)
                    np.testing.assert_allclose(actual, expected[mask, i], atol=3e-4, rtol=1e-5)
                self.assertEqual(len(predictor.state), 5)
                self.assertEqual(predictor.feature_updates, 18)
                self.assertEqual(predictor.received_frame_updates, 12)
                self.assertEqual(predictor.synthetic_feature_updates, 6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_actual_cuda_memory_teacher_mini_epoch(self):
        self._training_case(device='cuda')


if __name__ == '__main__':
    unittest.main()
