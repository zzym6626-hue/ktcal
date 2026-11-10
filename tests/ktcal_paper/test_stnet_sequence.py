import json
import hashlib
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from models.STNet import STNet
from scripts.train_stnet_3et import build_parser as build_3et_train_parser

from ktcal_paper.stnet_sequence import (
    FlatH5CausalWindowDataset,
    FlatH5ChunkDataset,
    RecordChunkDataset,
    decode_soft_expectation,
    masked_distillation_loss,
    masked_feature_distillation_loss,
    masked_official_simdr_loss,
    masked_pixel_loss,
    masked_simdr_loss,
    random_chunk_split,
    simdr_targets,
    select_3et_checkpoint,
    train_stnet,
    train_stnet_3et,
)


class STNetSequenceTests(unittest.TestCase):
    def test_legacy_last_readout_preserves_old_weights_and_predictions(self):
        torch.manual_seed(9)
        legacy = STNet().eval()
        explicit = STNet(SimpleNamespace(stnet_readout='last')).eval()
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        frames = torch.randn(1, 3, 2, 60, 80)
        with torch.no_grad():
            old_x, old_y = legacy(frames)
            new_x, new_y = explicit(frames)
        torch.testing.assert_close(new_x, old_x)
        torch.testing.assert_close(new_y, old_y)

    def test_final_hidden_readout_uses_reverse_state_from_complete_history(self):
        torch.manual_seed(7)
        model = STNet(SimpleNamespace(stnet_readout='final_hidden')).eval()
        frames = torch.randn(1, 4, 2, 60, 80)
        recurrent = []
        hook = model.gru.register_forward_hook(
            lambda _module, _input, output: recurrent.append(output))
        try:
            with torch.no_grad():
                logits_x, logits_y = model(frames)
        finally:
            hook.remove()
        raw_output, final_states = recurrent[0]
        expected_feature = torch.cat((final_states[-2], final_states[-1]), dim=-1)
        torch.testing.assert_close(logits_x[:, -1], model.fc_w(expected_feature))
        torch.testing.assert_close(logits_y[:, -1], model.fc_h(expected_feature))
        self.assertFalse(torch.allclose(
            raw_output[:, -1, 128:], final_states[-1]))

    def test_final_hidden_current_frame_loss_trains_reverse_recurrence(self):
        torch.manual_seed(7)
        model = STNet(SimpleNamespace(stnet_readout='final_hidden')).train()
        logits_x, logits_y = model(torch.randn(1, 4, 2, 60, 80))
        (logits_x[:, -1].square().mean() + logits_y[:, -1].square().mean()).backward()
        gradient = model.gru.weight_hh_l0_reverse.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_attention_readout_is_parameter_free_and_uses_temporal_context(self):
        torch.manual_seed(11)
        legacy = STNet(SimpleNamespace(stnet_readout='last'))
        attention = STNet(SimpleNamespace(stnet_readout='attention'))
        attention.load_state_dict(legacy.state_dict(), strict=True)
        attention.eval()
        self.assertEqual(
            sum(parameter.numel() for parameter in legacy.parameters()),
            sum(parameter.numel() for parameter in attention.parameters()))

        recurrent = []
        hook = attention.gru.register_forward_hook(
            lambda _module, _input, output: recurrent.append(output[0]))
        frames = torch.randn(2, 5, 2, 60, 80)
        try:
            logits_x, logits_y = attention(frames)
        finally:
            hook.remove()
        features = recurrent[0]
        radius = 5
        length = features.shape[1]
        offsets = torch.arange(-radius, radius + 1)
        indices = (torch.arange(length).unsqueeze(1) + offsets).clamp(
            0, length - 1)
        neighbors = features[:, indices]
        scores = (
            features.unsqueeze(2) * neighbors).sum(-1) / (
            features.shape[-1] ** 0.5)
        attended = (torch.softmax(scores, dim=-1).unsqueeze(-1)
                    * neighbors).sum(2)
        torch.testing.assert_close(logits_x, attention.fc_w(attended))
        torch.testing.assert_close(logits_y, attention.fc_h(attended))

    def test_causal_checkpoint_prefers_p3_over_mean_error(self):
        self.assertTrue(select_3et_checkpoint(.70, 5., .65, 2.,
                                                causal_last_step=True))
        self.assertFalse(select_3et_checkpoint(.60, 1., .65, 2.,
                                                 causal_last_step=True))
        self.assertTrue(select_3et_checkpoint(.65, 1.9, .65, 2.,
                                                causal_last_step=True))
        self.assertFalse(select_3et_checkpoint(.65, 1.9999, .65, 2.,
                                                causal_last_step=True))
        self.assertTrue(select_3et_checkpoint(.70, 2.1, .65, 2.,
                                               causal_last_step=False))
        self.assertFalse(select_3et_checkpoint(.60, 1.9, .65, 2.,
                                                causal_last_step=False))

    def test_causal_training_refuses_final_holdout_before_making_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self.make_flat_h5(root, 'train.h5')
            holdout = root / 'test_200_200.h5'
            output = root / 'run'
            with self.assertRaisesRegex(ValueError, 'final holdout'):
                train_stnet_3et(train, holdout, root / 'missing.pt', output,
                                causal_last_step=True, distill_weight=0)
            self.assertFalse(output.exists())

    def test_3et_causal_windows_match_online_history_without_future_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_flat_h5(root, sequences=2, timesteps=5)
            with h5py.File(path, 'r+') as handle:
                for sequence in range(2):
                    for time in range(5):
                        handle['frames'][sequence, time] = sequence * 10 + time
            data = FlatH5CausalWindowDataset(path, window=3, stride=2)
            self.assertEqual(len(data), 6)
            self.assertEqual(data.targets, [(s, t) for s in range(2)
                                            for t in (0, 2, 4)])
            expected = ([0, 0, 0], [0, 1, 2], [2, 3, 4],
                        [10, 10, 10], [10, 11, 12], [12, 13, 14])
            for index, values in enumerate(expected):
                frames, labels, valid = data[index]
                self.assertEqual(frames.shape, (3, 2, 60, 80))
                self.assertEqual(frames[:, 0, 0, 0].tolist(), values)
                self.assertEqual(labels.shape, (1, 2))
                self.assertEqual(valid.tolist(), [True])
                self.assertAlmostEqual(labels[0, 0].item(),
                                       data.targets[index][1] / 80)
            data.close()

    def test_3et_causal_validation_scores_only_current_frame(self):
        from ktcal_paper.stnet_sequence import evaluate_flat_stnet

        class PredictLast(torch.nn.Module):
            def forward(self, frames):
                batch, steps = frames.shape[:2]
                x = torch.zeros(batch, steps, 80)
                y = torch.zeros(batch, steps, 60)
                x[:, :-1, 79] = 100  # Wrong historical predictions are ignored.
                x[:, -1, 0] = 100
                y[:, -1, 0] = 100
                return x, y

        with tempfile.TemporaryDirectory() as directory:
            path = self.make_flat_h5(Path(directory), sequences=2, timesteps=5)
            data = FlatH5CausalWindowDataset(path, window=3, stride=1)
            loader = torch.utils.data.DataLoader(data, batch_size=4)
            metrics = evaluate_flat_stnet(PredictLast(), loader,
                                          torch.device('cpu'), causal_last_step=True)
            self.assertEqual(metrics['frames'], 10)
            self.assertAlmostEqual(metrics['p3'], .6)
            self.assertAlmostEqual(metrics['mean_error_px'],
                                   sum((2 ** .5) * time for time in range(5)) / 5,
                                   places=5)
            data.close()

    def test_3et_causal_training_uses_all_validation_frames_and_p3_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self.make_flat_h5(root, 'train.h5', sequences=1, timesteps=3)
            val = self.make_flat_h5(root, 'val.h5', sequences=2, timesteps=3)
            source = root / 'student.pt'
            torch.save({'model_state_dict': STNet().state_dict(),
                        'config': {'input_divisor': 1.0}}, source)
            output = root / 'causal'
            train_stnet_3et(train, val, source, output, epochs=1, patience=1,
                            window=2, train_stride=2, causal_last_step=True,
                            batch_size=2, lr=1e-5, min_lr=1e-6,
                            distill_weight=0, threads=1, device='cpu')
            config = json.loads((output / 'config.json').read_text())
            history = json.loads((output / 'history.json').read_text())
            self.assertEqual(config['train_windows'], 2)
            self.assertEqual(config['validation_windows'], 6)
            self.assertEqual(config['training_stride'], 2)
            self.assertEqual(config['validation_stride'], 1)
            self.assertEqual(history[0]['validation']['frames'], 6)
            self.assertEqual(history[1]['validation']['frames'], 6)
            self.assertEqual(history[1]['training_valid_timesteps'], 2)
            self.assertEqual(config['selection_metric'], 'hard_argmax_last_step_p3')
            self.assertIn('last step', config['training_supervision'])
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['best_validation_p3'],
                             max(row['validation']['p3'] for row in history))
            with self.assertRaises(FileExistsError):
                train_stnet_3et(train, val, source, output,
                                causal_last_step=True, distill_weight=0)

    def test_3et_cli_defaults_to_raw_100_frame_training(self):
        args = build_3et_train_parser().parse_args([
            '--train-h5', 'train.h5', '--val-h5', 'val.h5', '--output', 'run',
        ])
        self.assertIsNone(args.initialize_from)
        self.assertEqual(args.window, 100)
        self.assertEqual(args.input_divisor, 1.0)
        self.assertEqual(args.distill_weight, 0.)
        self.assertEqual(args.temporal_spatial_kernel, 1)
        self.assertFalse(args.causal_last_step)
        self.assertEqual(args.train_stride, 8)
        self.assertEqual(args.readout, 'last')

    def test_3et_training_cli_maps_short_selection_name_to_existing_metric(self):
        from scripts.train_stnet_3et import main

        args = ['train_stnet_3et', '--train-h5', 'train.h5', '--val-h5',
                'test_aug.h5', '--output', 'run', '--selection-metric', 'soft_ma11']
        with mock.patch('sys.argv', args), \
             mock.patch('scripts.train_stnet_3et.train_stnet_3et', return_value='ok') as train:
            main()
        self.assertEqual(train.call_args.kwargs['selection_metric'], 'offline_soft_ma11')

    def test_3et_attention_readout_is_offline_only(self):
        args = build_3et_train_parser().parse_args([
            '--train-h5', 'train.h5', '--val-h5', 'val.h5',
            '--initialize-from', 'best.pth', '--output', 'run',
            '--readout', 'attention',
        ])
        self.assertEqual(args.readout, 'attention')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self.make_flat_h5(root, 'train.h5')
            val = self.make_flat_h5(root, 'val.h5')
            source = root / 'source.pt'
            torch.save({'model_state_dict': STNet().state_dict(),
                        'config': {'input_divisor': 1.}}, source)
            with self.assertRaisesRegex(ValueError, 'causal'):
                train_stnet_3et(train, val, source, root / 'run',
                                epochs=1, patience=1, window=2,
                                causal_last_step=True, readout='attention',
                                distill_weight=0, threads=1, device='cpu')

    def test_3et_causal_training_records_final_hidden_readout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self.make_flat_h5(root, 'train.h5', sequences=1, timesteps=2)
            val = self.make_flat_h5(root, 'test_aug.h5', sequences=1, timesteps=2)
            source = root / 'source.pt'
            torch.save({'model_state_dict': STNet().state_dict(),
                        'config': {'input_divisor': 1.}}, source)
            output = root / 'final_hidden'
            train_stnet_3et(
                train, val, source, output, epochs=1, patience=1,
                window=2, train_stride=1, causal_last_step=True,
                readout='final_hidden', batch_size=2, lr=1e-5,
                min_lr=1e-6, distill_weight=0, threads=1, device='cpu')
            self.assertEqual(json.loads((output / 'config.json').read_text())['readout'],
                             'final_hidden')
            self.assertEqual(torch.load(output / 'best.pt', map_location='cpu',
                                        weights_only=False)['config']['readout'],
                             'final_hidden')

    def test_3et_training_rejects_external_teacher(self):
        with self.assertRaisesRegex(ValueError, 'External teacher'):
            train_stnet_3et('train.h5', 'val.h5', output='run',
                            distill_weight=.1)

    def test_3et_training_supports_random_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self.make_flat_h5(root, 'train.h5', sequences=1, timesteps=2)
            val = self.make_flat_h5(root, 'val.h5', sequences=1, timesteps=2)
            output = root / 'fresh'
            train_stnet_3et(train, val, output=output, epochs=1, patience=1,
                            window=2, batch_size=1, lr=1e-5, min_lr=1e-6,
                            threads=1, device='cpu')
            config = json.loads((output / 'config.json').read_text())
            self.assertIsNone(config['initialization']['source'])

    def make_flat_h5(self, root, name='flat.h5', sequences=2, timesteps=5):
        path = root / name
        frames = np.arange(
            sequences * timesteps * 2 * 60 * 80, dtype=np.uint32,
        ).reshape(sequences, timesteps, 2, 60, 80) % 16
        labels = np.zeros((sequences, timesteps, 3), dtype=np.float32)
        labels[..., 0] = np.arange(timesteps, dtype=np.float32) / 80
        labels[..., 1] = np.arange(timesteps, dtype=np.float32) / 60
        with h5py.File(path, 'w') as handle:
            handle['frames'] = frames.astype(np.uint8)
            handle['label'] = labels
        return path

    def make_manifest(self, root):
        records = {
            '1_1': (5, 1, np.array([1, 1, 1, 1, 1], dtype=bool)),
            '1_2': (3, 10, np.array([1, 0, 1], dtype=bool)),
            '2_1': (1, 20, np.array([1], dtype=bool)),
            '3_1': (1, 30, np.array([1], dtype=bool)),
        }
        with h5py.File(root / 'frames.h5', 'w') as handle:
            for name, (count, base, valid) in records.items():
                group = handle.create_group(name)
                frames = np.empty((count, 2, 60, 80), dtype=np.uint8)
                for index in range(count):
                    frames[index].fill(base + index)
                group['frames'] = frames
                labels = np.zeros((count, 3), dtype=np.float32)
                labels[:, 0] = np.arange(count, dtype=np.float32) / 80
                labels[:, 1] = np.arange(count, dtype=np.float32) / 60
                group['labels'] = labels
                group['valid'] = valid
        manifest = {
            'schema': 1,
            'dataset': 'tiny',
            'h5': 'frames.h5',
            'labels_are_placeholders': False,
            'splits': {
                'train': ['1_1', '1_2'],
                'val': ['2_1'],
                'holdout': ['3_1'],
            },
            'records': [{'recording': name} for name in records],
        }
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        return path

    def test_chunks_stay_in_record_and_tail_padding_is_not_supervised(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_manifest(Path(directory))
            dataset = RecordChunkDataset(path, 'train', window=4)
            self.assertEqual(
                dataset.chunks,
                [('1_1', 0, 4), ('1_1', 4, 1), ('1_2', 0, 3)],
            )

            frames, labels = dataset[1]
            self.assertEqual(tuple(frames.shape), (4, 2, 60, 80))
            self.assertTrue(torch.allclose(frames[:, 0, 0, 0], torch.full((4,), 5 / 15)))
            self.assertEqual(labels[:, 3].tolist(), [1.0, 0.0, 0.0, 0.0])

            frames, labels = dataset[2]
            self.assertTrue(torch.allclose(
                frames[:, 0, 0, 0], torch.tensor([10, 11, 12, 12]) / 15))
            self.assertEqual(labels[:, 3].tolist(), [1.0, 0.0, 1.0, 0.0])
            dataset.close()

    def test_flat_h5_chunks_preserve_raw_inputs_and_mask_padded_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_flat_h5(root)
            dataset = FlatH5ChunkDataset(path, window=3, input_divisor=1.0)

            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.chunks,
                             [(0, 0, 3), (0, 3, 2), (1, 0, 3), (1, 3, 2)])
            frames, labels, valid = dataset[1]
            self.assertEqual(tuple(frames.shape), (3, 2, 60, 80))
            self.assertEqual(tuple(labels.shape), (3, 2))
            self.assertEqual(valid.tolist(), [True, True, False])
            with h5py.File(path, 'r') as handle:
                expected = torch.from_numpy(handle['frames'][0, 3:5].astype(np.float32))
            torch.testing.assert_close(frames[:2], expected)
            torch.testing.assert_close(frames[2], expected[-1])
            dataset.close()

    def test_flat_h5_rejects_incompatible_shapes_and_divisor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_flat_h5(root)
            with self.assertRaisesRegex(ValueError, 'input divisor'):
                FlatH5ChunkDataset(path, window=3, input_divisor=0)
            with h5py.File(root / 'bad.h5', 'w') as handle:
                handle['frames'] = np.zeros((2, 5, 60, 80), dtype=np.uint8)
                handle['label'] = np.zeros((2, 5, 3), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, 'frames'):
                FlatH5ChunkDataset(root / 'bad.h5', window=3)

    def test_simdr_targets_peak_at_pixel_coordinate(self):
        coordinates = torch.tensor([[[0.25, 0.5]]])
        target_x = simdr_targets(coordinates[..., 0], bins=80, sigma=1.0)
        target_y = simdr_targets(coordinates[..., 1], bins=60, sigma=1.0)
        self.assertEqual(target_x.argmax(-1).item(), 20)
        self.assertEqual(target_y.argmax(-1).item(), 30)
        self.assertTrue(torch.allclose(target_x.sum(-1), torch.ones(1, 1)))
        self.assertTrue(torch.allclose(target_y.sum(-1), torch.ones(1, 1)))

    def test_masked_losses_ignore_invalid_timesteps(self):
        torch.manual_seed(4)
        logits_x = torch.randn(1, 2, 80)
        logits_y = torch.randn(1, 2, 60)
        labels = torch.tensor([[[0.2, 0.3], [0.8, 0.7]]])
        valid = torch.tensor([[True, False]])

        simdr = masked_simdr_loss(logits_x, logits_y, labels, valid)
        pixel = masked_pixel_loss(logits_x, logits_y, labels, valid)

        changed_x = logits_x.clone(); changed_x[:, 1] = 1000
        changed_y = logits_y.clone(); changed_y[:, 1] = -1000
        changed_labels = labels.clone(); changed_labels[:, 1] = torch.tensor([0.0, 0.0])
        self.assertTrue(torch.allclose(
            simdr,
            masked_simdr_loss(changed_x, changed_y, changed_labels, valid),
        ))
        self.assertTrue(torch.allclose(
            pixel,
            masked_pixel_loss(changed_x, changed_y, changed_labels, valid),
        ))

        decoded = decode_soft_expectation(logits_x[:, :1], logits_y[:, :1])
        self.assertEqual(tuple(decoded.shape), (1, 1, 2))
        self.assertTrue(torch.isfinite(decoded).all())

    def test_official_simdr_and_distillation_losses_respect_mask(self):
        torch.manual_seed(14)
        logits_x = torch.randn(1, 2, 80)
        logits_y = torch.randn(1, 2, 60)
        teacher_x = torch.randn(1, 2, 80)
        teacher_y = torch.randn(1, 2, 60)
        labels = torch.tensor([[[0.25, 0.5], [0.75, 0.2]]])
        valid = torch.tensor([[True, False]])

        supervised = masked_official_simdr_loss(
            logits_x, logits_y, labels, valid)
        distilled = masked_distillation_loss(
            logits_x, logits_y, teacher_x, teacher_y, valid,
            temperature=2.0)
        changed_x = logits_x.clone(); changed_x[:, 1] = 1000
        changed_y = logits_y.clone(); changed_y[:, 1] = -1000
        changed_tx = teacher_x.clone(); changed_tx[:, 1] = -1000
        changed_ty = teacher_y.clone(); changed_ty[:, 1] = 1000
        changed_labels = labels.clone(); changed_labels[:, 1] = 0

        torch.testing.assert_close(
            supervised,
            masked_official_simdr_loss(
                changed_x, changed_y, changed_labels, valid))
        torch.testing.assert_close(
            distilled,
            masked_distillation_loss(
                changed_x, changed_y, changed_tx, changed_ty, valid,
                temperature=2.0))
        self.assertGreaterEqual(distilled.item(), 0)
        self.assertTrue(torch.isfinite(supervised))

    def test_distillation_is_zero_for_identical_logits_and_validates_temperature(self):
        logits_x = torch.randn(1, 2, 80)
        logits_y = torch.randn(1, 2, 60)
        valid = torch.ones(1, 2, dtype=torch.bool)
        loss = masked_distillation_loss(
            logits_x, logits_y, logits_x, logits_y, valid,
            temperature=1.5)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        with self.assertRaisesRegex(ValueError, 'temperature'):
            masked_distillation_loss(
                logits_x, logits_y, logits_x, logits_y, valid,
                temperature=0)

    def test_feature_distillation_masks_padding_and_detaches_teacher(self):
        student = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
        teacher = torch.tensor([[[1., 0.], [1., 0.]]], requires_grad=True)
        valid = torch.tensor([[True, False]])
        loss = masked_feature_distillation_loss(student, teacher, valid)
        torch.testing.assert_close(loss, torch.zeros(()))
        changed = student.detach().clone().requires_grad_()
        with torch.no_grad():
            changed[0, 1] = torch.tensor([100., -100.])
        torch.testing.assert_close(
            loss, masked_feature_distillation_loss(changed, teacher, valid))
        shifted = torch.tensor([[[0., 1.], [0., 1.]]], requires_grad=True)
        gradient_loss = masked_feature_distillation_loss(shifted, teacher, valid)
        gradient_loss.backward()
        self.assertGreater(shifted.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(shifted.grad[0, 1].abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)
        with self.assertRaisesRegex(ValueError, 'feature shapes'):
            masked_feature_distillation_loss(student, teacher[..., :1], valid)
        with self.assertRaisesRegex(ValueError, 'valid timestep'):
            masked_feature_distillation_loss(student, teacher,
                                             torch.zeros_like(valid))

    def test_feature_taps_preserve_stnet_predictions(self):
        torch.manual_seed(18)
        frames = torch.randn(1, 3, 2, 60, 80)
        model = STNet().eval()
        with torch.no_grad():
            old_x, old_y = model(frames)
            features, new_x, new_y = model.forward_with_features(frames)
        self.assertEqual(features.shape, (1, 3, 256))
        torch.testing.assert_close(new_x, old_x)
        torch.testing.assert_close(new_y, old_y)

    def test_masked_losses_reject_an_empty_supervision_mask(self):
        logits_x = torch.zeros(1, 1, 80)
        logits_y = torch.zeros(1, 1, 60)
        labels = torch.zeros(1, 1, 2)
        valid = torch.zeros(1, 1, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, 'valid timestep'):
            masked_simdr_loss(logits_x, logits_y, labels, valid)
        with self.assertRaisesRegex(ValueError, 'valid timestep'):
            masked_pixel_loss(logits_x, logits_y, labels, valid)

    def test_full_temporal_convolution_variant_uses_three_by_three_spatial_support(self):
        args = SimpleNamespace(
            stnet_block3_kernel=5,
            stnet_unidirectional=False,
            stnet_temporal_spatial_kernel=3,
        )
        model = STNet(args)
        temporal_convolutions = [
            block.temporal[0]
            for block in (model.block1, model.block2, model.block3)
        ]
        self.assertEqual(
            [layer.kernel_size for layer in temporal_convolutions],
            [(3, 3, 3)] * 3,
        )
        self.assertEqual(
            [layer.padding for layer in temporal_convolutions],
            [(1, 1, 1)] * 3,
        )
        with self.assertRaisesRegex(ValueError, 'temporal spatial kernel'):
            STNet(SimpleNamespace(
                stnet_block3_kernel=5,
                stnet_unidirectional=False,
                stnet_temporal_spatial_kernel=2,
            ))

    def test_training_selects_on_validation_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            output = root / 'run'
            checkpoint = train_stnet(
                manifest, output, epochs=1, patience=1, window=2,
                batch_size=2, val_stride=1, lr=1e-4,
                seed=7, threads=torch.get_num_threads(), device='cpu',
                temporal_spatial_kernel=3)

            self.assertEqual(checkpoint, (output / 'best.pt').resolve())
            for name in ('config.json', 'history.json', 'best.pt', 'summary.json'):
                self.assertTrue((output / name).is_file(), name)
            history = json.loads((output / 'history.json').read_text())
            summary = json.loads((output / 'summary.json').read_text())
            state = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)

            self.assertEqual(summary['best_validation_error_px'],
                             min(row['validation_error_px'] for row in history))
            self.assertEqual(state['best_validation_error_px'],
                             summary['best_validation_error_px'])
            self.assertEqual(state['selection_split'], 'val')
            self.assertFalse(state['holdout_used'])
            self.assertEqual(
                state['manifest_sha256'],
                hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertNotIn('3_1', state['training_recordings'])
            self.assertNotIn('3_1', state['validation_recordings'])
            self.assertEqual(state['config']['temporal_spatial_kernel'], 3)
            self.assertIn('3x3x3', state['config']['architecture'])

    def test_3et_finetuning_uses_validation_only_and_records_raw_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_h5 = self.make_flat_h5(
                root, 'train.h5', sequences=1, timesteps=2)
            val_h5 = self.make_flat_h5(
                root, 'val.h5', sequences=1, timesteps=2)
            source = root / 'stnet.pt'
            torch.save({'model_state_dict': STNet().state_dict(),
                        'config': {'input_divisor': 1.}}, source)
            output = root / 'finetune'

            checkpoint = train_stnet_3et(
                train_h5, val_h5, source, output,
                epochs=1, patience=1, window=2, batch_size=1,
                lr=1e-5, min_lr=1e-6, distill_weight=0,
                seed=9, threads=torch.get_num_threads(), device='cpu')

            self.assertEqual(checkpoint, (output / 'best.pt').resolve())
            for name in ('config.json', 'history.json', 'best.pt',
                         'last.pt', 'summary.json'):
                self.assertTrue((output / name).is_file(), name)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            config = state['config']
            self.assertEqual(config['dataset'], '3ET+2025')
            self.assertEqual(config['input_divisor'], 1.0)
            self.assertEqual(config['window'], 2)
            self.assertEqual(config['initialization']['source_architecture'],
                             'STNet')
            self.assertTrue(config['initialization']['retrained'])
            self.assertEqual(config['training_h5'], str(train_h5.resolve()))
            self.assertEqual(config['validation_h5'], str(val_h5.resolve()))
            self.assertNotIn('test_200_200', json.dumps(config))
            self.assertEqual(state['selection_split'], 'validation_h5')
            self.assertFalse(state['holdout_used'])
            summary = json.loads((output / 'summary.json').read_text())
            self.assertFalse(summary['holdout_used_for_model_selection'])


    def test_random_chunk_split_is_seeded_disjoint_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_manifest(Path(directory))
            recordings, train_indices, val_indices, chunks = random_chunk_split(
                path, window=2, val_ratio=0.2, seed=3)

            self.assertEqual(recordings, ['1_1', '1_2', '2_1', '3_1'])
            self.assertEqual(len(chunks), 7)
            self.assertEqual(len(val_indices), 1)
            self.assertEqual(
                sorted(train_indices + val_indices), list(range(len(chunks))))
            self.assertFalse(set(train_indices) & set(val_indices))

            repeated = random_chunk_split(path, window=2, val_ratio=0.2, seed=3)
            self.assertEqual(repeated[1], train_indices)
            self.assertEqual(repeated[2], val_indices)

    def test_random_split_trains_on_every_recording_and_records_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            output = root / 'run_random'
            train_stnet(
                manifest, output, epochs=1, patience=1, window=2,
                batch_size=2, val_stride=1, lr=1e-4, split_mode='random',
                val_ratio=0.2, seed=3, threads=torch.get_num_threads(),
                device='cpu')

            state = torch.load(
                output / 'best.pt', map_location='cpu', weights_only=False)
            config = state['config']
            self.assertEqual(config['split_mode'], 'random')
            self.assertEqual(config['random_val_ratio'], 0.2)
            self.assertEqual(config['split_policy'],
                             'random chunk holdout across all recordings')
            self.assertEqual(config['training_recordings'],
                             ['1_1', '1_2', '2_1', '3_1'])
            self.assertTrue(config['validation_recordings'])
            self.assertEqual(config['train_chunks'], 6)
            self.assertEqual(config['validation_windows'], 1)

    def test_cosine_schedule_decays_learning_rate_and_records_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            output = root / 'run_cosine'
            train_stnet(
                manifest, output, epochs=3, patience=3, window=2,
                batch_size=2, val_stride=1, lr=0.01, min_lr=0.001,
                scheduler='cosine', split_mode='random', val_ratio=0.2,
                seed=5, threads=torch.get_num_threads(), device='cpu')

            history = json.loads((output / 'history.json').read_text())
            state = torch.load(
                output / 'best.pt', map_location='cpu', weights_only=False)
            rates = [row['learning_rate'] for row in history[1:]]
            self.assertEqual(len(rates), 3)
            self.assertAlmostEqual(rates[0], 0.01)
            self.assertAlmostEqual(rates[-1], 0.001, places=6)
            self.assertEqual(rates, sorted(rates, reverse=True))
            self.assertEqual(state['config']['scheduler'], 'cosine')
            self.assertAlmostEqual(state['config']['min_lr'], 0.001)

    def test_all_steps_selection_scores_every_valid_timestep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            output = root / 'run_all_steps'
            train_stnet(
                manifest, output, epochs=1, patience=1, window=2,
                batch_size=2, val_stride=1, lr=1e-4,
                split_mode='random', val_ratio=0.2, seed=3,
                selection_metric='all_steps',
                threads=torch.get_num_threads(), device='cpu')

            history = json.loads((output / 'history.json').read_text())
            state = torch.load(
                output / 'best.pt', map_location='cpu', weights_only=False)
            self.assertEqual(state['config']['selection_metric'], 'all_steps')
            self.assertGreaterEqual(history[0]['validation']['frames'], 1)
            self.assertLessEqual(history[0]['validation']['frames'], 2)


if __name__ == '__main__':
    unittest.main()
