import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch


class StreamingAugmentationTests(unittest.TestCase):
    def warp(self):
        self.assertIsNotNone(importlib.util.find_spec('ktcal_paper.streaming_augmentation'),
                             'Train-only record augmentation is missing')
        from ktcal_paper.streaming_augmentation import warp_streaming_records
        return warp_streaming_records

    def fixture(self):
        frames = torch.zeros(2, 3, 2, 60, 80)
        frames[:, :, :, 12, 24] = 1.
        labels = torch.tensor([24.5 / 80., 12.5 / 60.]).expand(2, 3, 2).clone()
        teacher = torch.tensor([25.5 / 80., 13.5 / 60.]).expand(2, 3, 2).clone()
        return frames, labels, teacher

    def test_identity_is_exact_and_preserves_invalid_nan(self):
        frames, labels, teacher = self.fixture()
        labels[0, 1] = float('nan')
        actual = self.warp()(frames, labels, teacher, torch.zeros(2), torch.ones(2), torch.ones(2))
        for value, original in zip(actual, (frames, labels, teacher)):
            torch.testing.assert_close(value, original, atol=0, rtol=0, equal_nan=True)

    def test_rotation_inverse_image_and_forward_points_use_physical_centers(self):
        frames, labels, teacher = self.fixture()
        transformed, points, targets = self.warp()(frames, labels, teacher,
                                                  torch.tensor([90., -90.]), torch.ones(2), torch.ones(2))
        for record, (x, y) in enumerate(((57, 14), (22, 45))):
            flat = transformed[record].reshape(6, 60, 80)
            for frame in flat:
                self.assertEqual(int(frame.argmax()), y * 80 + x)
                self.assertAlmostEqual(float(frame[y, x]), 1., places=4)
        expected = torch.tensor([[[57.5 / 80., 14.5 / 60.]], [[22.5 / 80., 45.5 / 60.]]]).expand(2, 3, 2)
        torch.testing.assert_close(points, expected, atol=1e-6, rtol=0)
        expected_teacher = torch.tensor([[[56.5 / 80., 15.5 / 60.]],
                                         [[23.5 / 80., 44.5 / 60.]]]).expand(2, 3, 2)
        torch.testing.assert_close(targets, expected_teacher, atol=1e-6, rtol=0)

    def test_scale_and_gain_apply_to_both_polarities_with_constant_record_transform(self):
        frames, labels, teacher = self.fixture()
        transformed, points, targets = self.warp()(frames, labels, teacher,
            torch.zeros(2), torch.tensor([2., .5]), torch.tensor([1.5, .75]))
        torch.testing.assert_close(points[0, 0], torch.tensor([9. / 80., -5. / 60.]))
        torch.testing.assert_close(targets[0, 0], torch.tensor([11. / 80., -3. / 60.]))
        torch.testing.assert_close(points[1, 0], torch.tensor([32.25 / 80., 21.25 / 60.]))
        for record in range(2):
            for step in range(1, 3):
                torch.testing.assert_close(transformed[record, step], transformed[record, 0], atol=0, rtol=0)
            torch.testing.assert_close(transformed[record, :, 0], transformed[record, :, 1], atol=0, rtol=0)
        gained, labels_gained, _ = self.warp()(frames, labels, teacher, torch.zeros(2),
                                               torch.ones(2), torch.tensor([1.5, .75]))
        torch.testing.assert_close(gained[:, 0, 0, 12, 24], torch.tensor([1.5, .75]), atol=0, rtol=0)
        torch.testing.assert_close(labels_gained, labels, atol=0, rtol=0)

    def test_invalid_parameters_and_mismatched_record_shapes_are_rejected(self):
        frames, labels, teacher = self.fixture()
        warp = self.warp()
        for angle, scale, gain in (([float('nan'), 0.], [1., 1.], [1., 1.]),
                                    ([0., 0.], [0., 1.], [1., 1.]),
                                    ([0., 0.], [1., float('inf')], [1., 1.]),
                                    ([0., 0.], [1., 1.], [-1., 1.]),
                                    ([0.], [1., 1.], [1., 1.])):
            with self.subTest(angle=angle, scale=scale, gain=gain):
                with self.assertRaises(ValueError):
                    warp(frames, labels, teacher, torch.tensor(angle), torch.tensor(scale), torch.tensor(gain))
        for wrong_frames, wrong_labels, wrong_teacher in ((frames[:, :0], labels[:, :0], teacher[:, :0]),
                (frames, labels[:, :1], teacher), (frames, labels, teacher[:, :, :1])):
            with self.assertRaises(ValueError):
                warp(wrong_frames, wrong_labels, wrong_teacher, torch.zeros(2), torch.ones(2), torch.ones(2))

    def test_active_warp_preserves_invalid_labels_and_finite_image_gradients(self):
        frames, labels, teacher = self.fixture()
        frames.requires_grad_(True)
        labels[0, 1] = float('nan')
        teacher[0, 1] = float('nan')
        transformed, points, targets = self.warp()(frames, labels, teacher,
            torch.tensor([5., -3.]), torch.tensor([1.1, .9]), torch.tensor([.8, 1.2]))
        self.assertTrue(torch.isnan(points[0, 1]).all())
        self.assertTrue(torch.isnan(targets[0, 1]).all())
        self.assertTrue(torch.isfinite(transformed).all())
        transformed.square().sum().backward()
        self.assertTrue(torch.isfinite(frames.grad).all())
        self.assertGreater(float(frames.grad.abs().sum()), 0.)

    def test_actual_training_epoch_augments_only_train_before_startup_and_excludes_original_invalid(self):
        from ktcal_paper.streaming_training import train_streaming_3et
        from tests.ktcal_paper.test_streaming_training import StreamingTrainingTests
        warp = self.warp()
        calls = []

        def recorded(*args):
            calls.append([torch.as_tensor(value).detach().clone() for value in args])
            return warp(*args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train_aug.h5', 'test_aug.h5'):
                StreamingTrainingTests().make_data(root / name)
            with h5py.File(root / 'train_aug.h5', 'r+') as handle:
                coordinates = handle['label'][:]
                coordinates[:, 1, :2] = np.nan
                coordinates[0, 0, 0] = -.005
                handle['label'][:] = coordinates
            output = root / 'trial'
            with patch('ktcal_paper.streaming_augmentation.warp_streaming_records', side_effect=recorded):
                result = train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                    output=output, epochs=1, patience=1, batch_size=2, hidden_size=128,
                    device='cpu', threads=1, augment=True, warm_start_steps=3,
                    current_readout=True, rotation_degrees=5., scale_jitter=.1, count_gain_jitter=.15)
            self.assertEqual(len(calls), 1, 'Validation must not be augmented')
            self.assertEqual(calls[0][0].shape[1], 5, 'Warp must precede synthetic prefix')
            self.assertTrue(bool((calls[0][3].abs() <= 5.).all()))
            self.assertTrue(bool(((calls[0][4] >= .9) & (calls[0][4] <= 1.1)).all()))
            self.assertTrue(bool(((calls[0][5] >= .85) & (calls[0][5] <= 1.15)).all()))
            config = json.loads((output / 'config.json').read_text())
            self.assertEqual((config['rotation_degrees'], config['scale_jitter'], config['count_gain_jitter']),
                             (5., .1, .15))
            self.assertEqual(config['record_augmentation_split'], 'train')
            history = json.loads((output / 'history.json').read_text())
            self.assertEqual(history[-1]['training_valid_timesteps'], 7)
            self.assertTrue(np.isfinite(history[-1]['train']['loss']))
            self.assertEqual(result['metrics']['frames'], 10)

    def test_training_rejects_illegal_augmentation_before_creating_output(self):
        from ktcal_paper.streaming_training import train_streaming_3et
        from tests.ktcal_paper.test_streaming_training import StreamingTrainingTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train_aug.h5', 'test_aug.h5'):
                StreamingTrainingTests().make_data(root / name)
            for flags in (dict(rotation_degrees=float('nan')), dict(rotation_degrees=16.),
                          dict(scale_jitter=-.01), dict(scale_jitter=.31),
                          dict(count_gain_jitter=float('inf')), dict(count_gain_jitter=.31)):
                with self.subTest(flags=flags):
                    with self.assertRaises(ValueError):
                        train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                            output=root / 'forbidden', augment=True, **flags)
                    self.assertFalse((root / 'forbidden').exists())
            with self.assertRaises(ValueError):
                train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                                    output=root / 'disabled', rotation_degrees=5.)

    def test_cli_passes_optional_augmentation_and_preserves_zero_defaults(self):
        from scripts import train_streaming_3et as cli
        base = ['train_streaming_3et', '--train-h5', 'train_aug.h5', '--val-h5', 'test_aug.h5',
                '--output', 'unused']
        for flags, expected in (([], (0., 0., 0.)),
              (['--augment', '--rotation-degrees', '5', '--scale-jitter', '.1',
                '--count-gain-jitter', '.15'], (5., .1, .15))):
            with self.subTest(flags=flags):
                with patch.object(sys, 'argv', base + flags), patch.object(cli, 'train_streaming_3et', return_value={}) as train:
                    cli.main()
                kwargs = train.call_args.kwargs
                self.assertEqual(tuple(kwargs[name] for name in
                    ('rotation_degrees', 'scale_jitter', 'count_gain_jitter')), expected)

    def test_teacher_excludes_original_invalid_target_moved_into_image(self):
        from ktcal_paper.streaming_training import teacher_coordinate_loss
        frames = torch.zeros(1, 1, 2, 60, 80)
        labels = torch.tensor([[[-.005, .5]]])
        teacher = torch.tensor([[[.01, .5]]])
        valid = ((labels >= 0) & (labels <= 1)).all(-1)
        _, labels, teacher = self.warp()(frames, labels, teacher, torch.zeros(1),
                                        torch.tensor([.9]), torch.ones(1))
        self.assertTrue(bool(((labels >= 0) & (labels <= 1)).all()))
        x, y = torch.zeros(1, 1, 80, requires_grad=True), torch.zeros(1, 1, 60, requires_grad=True)
        loss = teacher_coordinate_loss(x, y, teacher, labels, weights=valid.float(), valid=valid)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(x.grad.abs().sum()), 0.)
        self.assertEqual(float(y.grad.abs().sum()), 0.)

    def test_actual_teacher_augmented_epoch_does_not_revalidate_out_of_bounds_target(self):
        from ktcal_paper.combined_3et import _sha256
        from ktcal_paper.streaming_training import train_streaming_3et
        from tests.ktcal_paper.test_streaming_training import StreamingTrainingTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train_aug.h5', 'test_aug.h5'):
                StreamingTrainingTests().make_data(root / name)
            with h5py.File(root / 'train_aug.h5', 'r+') as handle:
                coordinates = handle['label'][:]
                coordinates[0, 0, :2] = [-.005, .5]
                handle['label'][:] = coordinates
            teacher = coordinates[:, :, :2].copy()
            teacher[0, 0] = [.01, .5]
            cache = root / 'teacher.npz'
            metadata = dict(split='train', source_h5_sha256=_sha256(root / 'train_aug.h5'),
                            teacher_checkpoint_sha256='a' * 64)
            np.savez(cache, coordinates=teacher, metadata=json.dumps(metadata))
            result = train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                output=root / 'result', teacher_cache=cache, teacher_weight=.2,
                augment=True, scale_jitter=.3, warm_start_steps=3, seed=0,
                epochs=1, patience=1, batch_size=2, hidden_size=128, device='cpu', threads=1)
            history = json.loads((root / 'result' / 'history.json').read_text())
            self.assertEqual(history[-1]['training_valid_timesteps'], 9)
            self.assertTrue(np.isfinite(history[-1]['train']['teacher_huber_loss']))
            self.assertEqual(result['metrics']['frames'], 10)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_teacher_augmented_training_has_finite_gradients_with_invalid_targets(self):
        from ktcal_paper.combined_3et import _sha256
        from ktcal_paper.streaming_training import train_streaming_3et
        from tests.ktcal_paper.test_streaming_training import StreamingTrainingTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train_aug.h5', 'test_aug.h5'):
                StreamingTrainingTests().make_data(root / name)
            with h5py.File(root / 'train_aug.h5', 'r+') as handle:
                coordinates = handle['label'][:]
                coordinates[:, 1, :2] = np.nan
                coordinates[0, 0, :2] = [-.005, .5]
                handle['label'][:] = coordinates
            teacher = np.nan_to_num(coordinates[:, :, :2], nan=.3)
            teacher[0, 0] = [.01, .5]
            cache = root / 'teacher.npz'
            metadata = dict(split='train', source_h5_sha256=_sha256(root / 'train_aug.h5'),
                            teacher_checkpoint_sha256='a' * 64)
            np.savez(cache, coordinates=teacher, metadata=json.dumps(metadata))
            result = train_streaming_3et(root / 'train_aug.h5', root / 'test_aug.h5',
                output=root / 'result', teacher_cache=cache, teacher_weight=.2, augment=True,
                rotation_degrees=5., scale_jitter=.1, count_gain_jitter=.15, warm_start_steps=3,
                current_readout=True, learned_initial_state=True, epochs=1, patience=1,
                batch_size=2, hidden_size=128, device='cuda', threads=1)
            self.assertEqual(result['metrics']['frames'], 10)
            history = json.loads((root / 'result' / 'history.json').read_text())
            self.assertEqual(history[-1]['training_valid_timesteps'], 7)
            self.assertTrue(np.isfinite(history[-1]['train']['loss']))


if __name__ == '__main__':
    unittest.main()
