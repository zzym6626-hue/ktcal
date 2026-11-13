import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from ktcal_paper.combined_3et import _sha256
from ktcal_paper.offline_experiment import spatial_view


class StreamingTrainingTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('ktcal_paper.streaming_training'),
                             'Streaming sequence training is not implemented')
        from ktcal_paper import streaming_training
        return streaming_training

    def make_data(self, path):
        frames = np.zeros((2, 5, 2, 60, 80), dtype=np.float32)
        frames[0, :, 0, 12, 24] = 1
        labels = np.zeros((2, 5, 3), dtype=np.float32)
        labels[:, :, :2] = [.3, .2]
        with h5py.File(path, 'w') as handle:
            handle['frames'] = frames
            handle['label'] = labels

    def test_training_dataset_refuses_validation_or_final_holdout(self):
        module = self.module()
        for name in ('test_aug.h5', 'test_200_200.h5'):
            with self.assertRaisesRegex(ValueError, 'train_aug'):
                module.StreamingSequenceDataset(name, split='train')
            with self.assertRaisesRegex(ValueError, 'train_aug'):
                module.build_teacher_cache(name, 'unused.pt', 'unused.npz')

    def test_full_record_dataset_preload_preserves_frames_and_labels(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train_aug.h5'
            self.make_data(path)
            lazy = module.StreamingSequenceDataset(path, split='train')
            ram = module.StreamingSequenceDataset(path, split='train', preload=True)
            try:
                self.assertEqual(len(lazy), 2)
                for a, b in zip(lazy[0], ram[0]):
                    torch.testing.assert_close(a, b)
                frames, labels, teacher = ram[0]
                self.assertEqual(frames.shape, (5, 2, 60, 80))
                torch.testing.assert_close(labels, teacher)
            finally:
                lazy.close()

    def test_training_spatial_view_transforms_coordinates_consistently(self):
        module = self.module()
        frame = torch.zeros(1, 1, 2, 60, 80)
        frame[..., 12, 24] = 1
        point = torch.tensor([[[24 / 80., 12 / 60.]]])
        for view in ((2, -1, False), (-2, 1, True), (0, 0, True)):
            transformed = spatial_view(frame, view)
            coordinate = module.transform_coordinates(point, view)
            y, x = transformed[0, 0, 0].nonzero()[0].tolist()
            torch.testing.assert_close(coordinate, torch.tensor([[[x / 80., y / 60.]]]))

    def test_explicit_flip_convention_matches_released_normalized_labels(self):
        point = torch.tensor([[[.3, .2]]])
        transform = self.module().transform_coordinates
        self.assertIn('flip_axis_px', __import__('inspect').signature(transform).parameters)
        actual = transform(point, (2, -1, True), flip_axis_px=80.)
        torch.testing.assert_close(actual, torch.tensor([[[1. - (.3 + 2 / 80), .2 - 1 / 60]]]))
        legacy = transform(point, (2, -1, True))
        torch.testing.assert_close(actual - legacy, torch.tensor([[[1 / 80., 0.]]]))

    def test_teacher_cache_requires_training_hash_and_split(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train_aug.h5'
            cache = Path(directory) / 'teacher.npz'
            self.make_data(path)
            metadata = dict(split='train', source_h5_sha256=_sha256(path),
                            teacher_checkpoint_sha256='a' * 64)
            coordinates = np.ones((2, 5, 2), dtype=np.float32) * .25
            np.savez(cache, coordinates=coordinates, metadata=json.dumps(metadata))
            dataset = module.StreamingSequenceDataset(path, split='train', teacher_cache=cache)
            try:
                torch.testing.assert_close(dataset[0][-1], torch.from_numpy(coordinates[0]))
            finally:
                dataset.close()
            metadata['split'] = 'validation'
            np.savez(cache, coordinates=coordinates, metadata=json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'teacher'):
                module.StreamingSequenceDataset(path, split='train', teacher_cache=cache)
            metadata['split'] = 'train'
            metadata['source_h5_sha256'] = 'b' * 64
            np.savez(cache, coordinates=coordinates, metadata=json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'teacher'):
                module.StreamingSequenceDataset(path, split='train', teacher_cache=cache)

    def test_pixel_teacher_loss_ignores_bad_training_teacher_points(self):
        module = self.module()
        logits_x = torch.zeros(1, 2, 80, requires_grad=True)
        logits_y = torch.zeros(1, 2, 60, requires_grad=True)
        truth = torch.tensor([[[39.5 / 80, 29.5 / 60], [39.5 / 80, 29.5 / 60]]])
        teacher = truth.clone()
        teacher[:, 0, 0] += .02
        teacher[:, 1, 0] += .3
        loss = module.teacher_coordinate_loss(logits_x, logits_y, teacher, truth,
                                               max_error_px=5.)
        expected = torch.nn.functional.smooth_l1_loss(
            torch.tensor([39.5, 29.5]), teacher[0, 0] * torch.tensor([80., 60.]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(float(logits_x.grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(logits_x.grad[0, 1].abs().sum()), 0)

    def test_one_epoch_trains_all_timesteps_and_saves_causal_metadata(self):
        module = self.module()
        self.assertTrue(hasattr(module, 'train_streaming_3et'), 'Streaming trainer is missing')
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root / 'train_aug.h5')
            self.make_data(root / 'test_aug.h5')
            output = root / 'run'
            result = module.train_streaming_3et(
                root / 'train_aug.h5', root / 'test_aug.h5', output=output,
                epochs=1, patience=1, batch_size=2, hidden_size=128,
                lr=1e-4, min_lr=1e-4, device='cpu', threads=1)
            self.assertTrue((output / 'best.pt').exists())
            history = json.loads((output / 'history.json').read_text())
            self.assertEqual(history[-1]['training_valid_timesteps'], 10)
            self.assertEqual(result['metrics']['frames'], 10)
            configuration = json.loads((output / 'config.json').read_text())
            self.assertEqual(configuration['temporal_mode'], 'causal_streaming')
            self.assertFalse(configuration['final_holdout_used'])
            self.assertEqual(configuration['teacher_weight'], 0.)

    def test_teacher_loss_ignores_coordinates_shifted_outside_image(self):
        module = self.module()
        x = torch.zeros(1, 2, 80, requires_grad=True)
        y = torch.zeros(1, 2, 60, requires_grad=True)
        labels = torch.tensor([[[-.01, .4], [.4, .4]]])
        loss = module.teacher_coordinate_loss(x, y, labels.clone(), labels)
        loss.backward()
        self.assertEqual(float(x.grad[0, 0].abs().sum()), 0)
        self.assertGreater(float(x.grad[0, 1].abs().sum()), 0)

    @torch.inference_mode()
    def test_warmed_validation_agrees_with_received_only_streaming(self):
        from torch.utils.data import DataLoader, TensorDataset
        from models.streaming_stnet import StreamingSTNet
        from ktcal_paper.streaming_inference import StreamingPredictor
        from ktcal_paper.evaluation_protocol import summarize_pixel_errors
        torch.set_num_threads(1)
        model = StreamingSTNet(hidden_size=128).eval()
        frames = torch.rand(2, 6, 2, 60, 80)
        labels = torch.full((2, 6, 2), .4)
        loader = DataLoader(TensorDataset(frames, labels, labels), batch_size=2)
        evaluated = self.module().evaluate_streaming(model, loader, 'cpu', warm_start_steps=3)
        predictor = StreamingPredictor(model, cuda_graph=False, warm_start_steps=3)
        outputs = []
        for index in range(6):
            predictor.observe(frames[:, index])
            outputs.append(predictor(frames[:, :index + 1], np.ones(2, dtype=bool)))
        expected = summarize_pixel_errors(np.stack(outputs, 1) / np.array([80., 60.]), labels.numpy())
        self.assertAlmostEqual(evaluated['mean_error_px'], expected['mean_error_px'], places=4)
        self.assertEqual(evaluated['frames'], 12)
        self.assertEqual(predictor.feature_updates, 18)

    def test_warmed_spatial_training_supervises_only_real_timesteps(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root / 'train_aug.h5')
            self.make_data(root / 'test_aug.h5')
            self.module().train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                output=root / 'run', warm_start_steps=3, spatial_localizer=True,
                epochs=1, patience=1, batch_size=2, hidden_size=128, device='cpu', threads=1)
            history = json.loads((root / 'run/history.json').read_text())
            config = json.loads((root / 'run/config.json').read_text())
            self.assertEqual(history[-1]['training_valid_timesteps'], 10)
            self.assertEqual(history[-1]['validation']['frames'], 10)
            self.assertEqual(config['warm_start_steps'], 3)
            self.assertGreater(history[-1]['train']['spatial_heatmap_loss'], 0)
            for invalid in (-1, 1.5, True):
                with self.assertRaisesRegex(ValueError, 'warm'):
                    self.module().train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                        output=root / 'invalid', warm_start_steps=invalid, device='cpu')

    def test_difficulty_training_masks_prefix_and_records_temperature(self):
        import inspect
        module = self.module()
        self.assertIn('decode_temperature', inspect.signature(module.train_streaming_3et).parameters)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root / 'train_aug.h5')
            self.make_data(root / 'test_aug.h5')
            teacher = np.full((2, 5, 2), [.3, .2], dtype=np.float32)
            metadata = dict(split='train', source_h5_sha256=_sha256(root / 'train_aug.h5'),
                            teacher_checkpoint_sha256='b' * 64)
            np.savez(root / 'teacher.npz', coordinates=teacher, metadata=json.dumps(metadata))
            result = module.train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                output=root / 'run', warm_start_steps=3, teacher_cache=root / 'teacher.npz',
                teacher_weight=.2, decode_temperature=.65, persistent_boost=1.,
                persistent_min_run=3, cold_boost=1., cold_frames=2,
                epochs=1, patience=1, batch_size=2, hidden_size=128,
                device='cpu', threads=1, augment=True)
            history = json.loads((root / 'run/history.json').read_text())
            config = json.loads((root / 'run/config.json').read_text())
            self.assertEqual(history[-1]['training_valid_timesteps'], 10)
            self.assertEqual(result['metrics']['frames'], 10)
            self.assertEqual(config['decode_temperature'], .65)
            self.assertEqual(config['persistent_boost'], 1.)
            self.assertEqual(config['cold_frames'], 2)
            self.assertFalse(config['teacher_used_for_validation'])
            self.assertEqual(config['difficulty_weights_split'], 'train')

    @torch.inference_mode()
    def test_temperature_validation_matches_stateful_soft_decode(self):
        import inspect
        from torch.utils.data import DataLoader, TensorDataset
        from models.streaming_stnet import StreamingSTNet
        from ktcal_paper.streaming_inference import StreamingPredictor
        from ktcal_paper.evaluation_protocol import summarize_pixel_errors
        module = self.module()
        self.assertIn('decode_temperature', inspect.signature(module.evaluate_streaming).parameters)
        torch.set_num_threads(1)
        model = StreamingSTNet(hidden_size=128).eval()
        frames = torch.rand(2, 5, 2, 60, 80)
        labels = torch.full((2, 5, 2), .4)
        loader = DataLoader(TensorDataset(frames, labels, labels), batch_size=2)
        evaluated = module.evaluate_streaming(model, loader, 'cpu',
            warm_start_steps=3, decode_temperature=.65)
        predictor = StreamingPredictor(model, cuda_graph=False, warm_start_steps=3, temperature=.65)
        outputs = []
        for index in range(5):
            predictor.observe(frames[:, index])
            outputs.append(predictor(frames[:, :index + 1], np.ones(2, dtype=bool)))
        expected = summarize_pixel_errors(np.stack(outputs, 1) / np.array([80., 60.]), labels.numpy())
        self.assertAlmostEqual(evaluated['mean_error_px'], expected['mean_error_px'], places=4)

    def test_weighted_teacher_loss_uses_only_trusted_targets(self):
        import inspect
        module = self.module()
        self.assertIn('weights', inspect.signature(module.teacher_coordinate_loss).parameters)
        x = torch.zeros(1, 3, 80, requires_grad=True)
        y = torch.zeros(1, 3, 60, requires_grad=True)
        labels = torch.tensor([[[.5, .5], [.5, .5], [.5, .5]]])
        teacher = labels.clone()
        teacher[0, 0, 0] += 1 / 80.
        teacher[0, 1, 0] += 2 / 80.
        teacher[0, 2, 0] += .3
        weights = torch.tensor([[1., 3., 2.]])
        loss = module.teacher_coordinate_loss(x, y, teacher, labels, weights=weights, temperature=.65)
        per = torch.nn.functional.smooth_l1_loss(torch.tensor([39.5, 29.5]).expand(2, -1),
                    teacher[0, :2] * torch.tensor([80., 60.]), reduction='none').mean(-1)
        torch.testing.assert_close(loss, (per * weights[0, :2]).sum() / 4)
        loss.backward()
        self.assertEqual(float(x.grad[0, 2].abs().sum()), 0.)

    def test_official_simdr_invalid_nan_targets_have_finite_zero_gradients(self):
        from ktcal_paper.stnet_sequence import masked_official_simdr_loss
        x = torch.zeros(1, 3, 80, requires_grad=True)
        y = torch.zeros(1, 3, 60, requires_grad=True)
        labels = torch.tensor([[[.3, .2], [float('nan'), float('nan')], [.4, .3]]])
        valid = torch.tensor([[True, False, True]])
        loss = masked_official_simdr_loss(x, y, labels, valid, weights=valid.float())
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool(torch.isfinite(y.grad).all()))
        self.assertEqual(float(x.grad[0, 1].abs().sum()), 0.)

    def test_weighted_main_and_current_training_accept_invalid_nan_labels(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root / 'train_aug.h5')
            self.make_data(root / 'test_aug.h5')
            with h5py.File(root / 'train_aug.h5', 'r+') as handle:
                labels = handle['label'][:]
                labels[:, 1, :2] = np.nan
                handle['label'][:] = labels
            for current in (False, True):
                with self.subTest(current=current):
                    output = root / ('current' if current else 'main')
                    result = self.module().train_streaming_3et(
                        root / 'train_aug.h5', root / 'test_aug.h5', output=output,
                        warm_start_steps=3, current_readout=current,
                        decode_temperature=.65, persistent_boost=1., cold_boost=1.,
                        epochs=1, patience=1, batch_size=2, hidden_size=128,
                        device='cpu', threads=1)
                    history = json.loads((output / 'history.json').read_text())
                    self.assertEqual(history[-1]['training_valid_timesteps'], 8)
                    self.assertEqual(result['metrics']['frames'], 10)
                    self.assertTrue(np.isfinite(history[-1]['train']['loss']))
