"""Training-only hard cases and deployment-equivalent offline selection."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ktcal_paper.stnet_sequence import (
    evaluate_flat_stnet, masked_official_simdr_loss, select_3et_checkpoint,
    train_stnet_3et,
)
from ktcal_paper.train_hard_frames import hard_frame_weights
from ktcal_paper.combined_3et import SCALE, moving_average_sequence_predictions
from ktcal_paper.evaluation_protocol import soft_expectation_decode, summarize_pixel_errors


class HardFrameWeightsTests(unittest.TestCase):
    @staticmethod
    def example():
        frames = torch.zeros(2, 8, 2, 60, 80)
        labels = torch.tensor([.5, .5]).expand(2, 8, 2).clone()
        valid = torch.ones(2, 8, dtype=torch.bool)
        x = torch.full((2, 8, 8), -100.)
        y = torch.full((2, 8, 6), -100.)
        x[..., 4] = 100.
        y[..., 3] = 100.
        return frames, x, y, labels, valid

    def test_persistent_errors_do_not_cross_sequences_or_invalid_frames(self):
        frames, x, y, labels, valid = self.example()
        x[0, :5, 4] = -100.
        x[0, :5, 0] = 100.
        x[1, :4, 4] = -100.
        x[1, :4, 0] = 100.
        x[1, 5, 4] = -100.
        x[1, 5, 0] = 100.
        weights = hard_frame_weights(frames, x, y, labels, valid,
                                     persistent_boost=.5, sparse_boost=0)
        self.assertEqual(weights[0].tolist(), [1.5]*5 + [1.]*3)
        self.assertEqual(weights[1].tolist(), [1.]*8)

        valid[0, 2] = False
        weights = hard_frame_weights(frames, x, y, labels, valid,
                                     persistent_boost=.5, sparse_boost=0)
        self.assertEqual(weights[0].tolist(), [1., 1., 0., 1., 1., 1., 1., 1.])

    def test_sparse_quartile_is_computed_per_record(self):
        frames, x, y, labels, valid = self.example()
        for i in range(2):
            for t in range(8):
                frames[i, t, 0, 0, :i*10 + t+1] = 1
        weights = hard_frame_weights(frames, x, y, labels, valid,
                                     persistent_boost=0, sparse_boost=.25)
        self.assertEqual(weights.tolist(), [[1.25, 1.25] + [1.]*6]*2)

    def test_weighted_simdr_is_normalized_and_masks_padding(self):
        labels = torch.tensor([[[.25, .5], [.75, .2]]])
        x = torch.zeros(1, 2, 8)
        y = torch.zeros(1, 2, 6)
        x[0, 0, 2] = 10
        x[0, 1, 0] = 10
        valid = torch.tensor([[True, True]])
        first = masked_official_simdr_loss(x[:, :1], y[:, :1],
                                           labels[:, :1], valid[:, :1])
        second = masked_official_simdr_loss(x[:, 1:], y[:, 1:],
                                            labels[:, 1:], valid[:, 1:])
        weighted = masked_official_simdr_loss(x, y, labels, valid,
                                             weights=torch.tensor([[1., 3.]]))
        torch.testing.assert_close(weighted, (first + 3*second)/4)
        torch.testing.assert_close(
            masked_official_simdr_loss(x, y, labels,
                                       torch.tensor([[True, False]]),
                                       weights=torch.tensor([[1., 1e6]])),
            first)
        torch.testing.assert_close(
            masked_official_simdr_loss(x, y, labels, valid),
            masked_official_simdr_loss(x, y, labels, valid,
                                       weights=torch.ones(1, 2)))


class OfflineSelectionTests(unittest.TestCase):
    def test_offline_selection_uses_strict_mean_error_tiebreak(self):
        self.assertTrue(select_3et_checkpoint(
            .88, 1.9999, .88, 2., selection_metric='offline_soft_ma11'))
        self.assertFalse(select_3et_checkpoint(.88, 1.9999, .88, 2.))

    def test_offline_metrics_use_the_same_pixel_roundtrip_and_precision_as_combined(self):
        class ConstantLogits(torch.nn.Module):
            def forward(self, frames):
                x = torch.full((*frames.shape[:2], 80), -100.)
                y = torch.full((*frames.shape[:2], 60), -100.)
                x[..., 1] = 100.
                x[..., 2] = 99.1 - torch.arange(frames.shape[1]) * .06
                y[..., 11] = 100.
                y[..., 12] = 99.3
                return x, y

        model = ConstantLogits()
        frames = torch.zeros(1, 11, 2, 60, 80)
        labels = torch.full((1, 11, 2), .1234567, dtype=torch.float32)
        valid = torch.ones(1, 11, dtype=torch.bool)
        valid[0, 2] = False
        loader = DataLoader([(frames[0], labels[0], valid[0])], batch_size=1)
        observed = evaluate_flat_stnet(model, loader, torch.device('cpu'),
                                       offline_ma_window=11)

        logits_x, logits_y = model(frames)
        pixel_predictions = (soft_expectation_decode(logits_x, logits_y)
                             .astype(np.float32) * SCALE)
        normalized = pixel_predictions[0] / SCALE
        filtered = moving_average_sequence_predictions([normalized], 11)[0]
        expected = summarize_pixel_errors(filtered[valid[0].numpy()],
                                          labels[0, valid[0]].numpy())
        for metric in ('frames', 'mean_error_px', 'median_error_px', 'p3', 'p5', 'p10'):
            self.assertEqual(observed[metric], expected[metric], metric)

    def test_soft_ma11_is_independent_for_each_chunk_and_differs_from_argmax(self):
        class ReadFrameIndex(torch.nn.Module):
            def forward(self, frames):
                b, t = frames.shape[:2]
                x = torch.full((b, t, 80), -100., device=frames.device)
                y = torch.full((b, t, 60), -100., device=frames.device)
                idx = frames[..., 0, 0, 0].long()
                x.scatter_(-1, idx[..., None], 100.)
                y[..., 0] = 100.
                return x, y

        frames = torch.zeros(2, 11, 2, 60, 80)
        frames[0, 5, 0, 0, 0] = 20
        labels = torch.zeros(2, 11, 2)
        loader = DataLoader([(frames[i], labels[i], torch.ones(11, dtype=torch.bool))
                             for i in range(2)], batch_size=2)
        raw = evaluate_flat_stnet(ReadFrameIndex(), loader, torch.device('cpu'))
        offline = evaluate_flat_stnet(ReadFrameIndex(), loader,
                                      torch.device('cpu'), offline_ma_window=11)
        self.assertEqual(raw['frames'], 22)
        self.assertAlmostEqual(raw['p3'], 21/22)
        self.assertEqual(offline['frames'], 22)
        self.assertAlmostEqual(offline['p3'], 1.)
        self.assertAlmostEqual(offline['mean_error_px'], 10/11, places=4)

    def test_offline_model_selection_rejects_final_holdout_before_any_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            with self.assertRaisesRegex(ValueError, 'final holdout'):
                train_stnet_3et(root / 'train_aug.h5',
                                root / 'test_200_200.h5', root / 'best.pt', output,
                                selection_metric='offline_soft_ma11',
                                persistent_boost=.5, sparse_boost=.25)
            self.assertFalse(output.exists())

    def test_training_input_rejects_named_final_holdout_before_any_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with h5py.File(root / 'test_aug.h5', 'w') as handle:
                handle.create_dataset('frames', shape=(1, 100, 2, 60, 80), dtype='float32')
                handle.create_dataset('label', shape=(1, 100, 3), dtype='float32')
            output = root / 'output'
            with self.assertRaisesRegex(ValueError, 'final holdout'):
                train_stnet_3et(root / 'test_200_200.h5', root / 'test_aug.h5',
                                root / 'best.pt', output,
                                selection_metric='offline_soft_ma11')
            self.assertFalse(output.exists())

    def test_training_input_rejects_renamed_holdout_shape_before_any_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'renamed.h5'
            with h5py.File(source, 'w') as handle:
                handle.create_dataset('frames', shape=(50, 200, 2, 60, 80),
                                      dtype='float32', chunks=(1, 1, 2, 60, 80))
                handle.create_dataset('label', shape=(50, 200, 3), dtype='float32')
            with h5py.File(root / 'test_aug.h5', 'w') as handle:
                handle.create_dataset('frames', shape=(1, 100, 2, 60, 80), dtype='float32')
                handle.create_dataset('label', shape=(1, 100, 3), dtype='float32')
            output = root / 'output'
            with self.assertRaisesRegex(ValueError, 'final holdout'):
                train_stnet_3et(source, root / 'test_aug.h5',
                                root / 'best.pt', output,
                                selection_metric='offline_soft_ma11')
            self.assertFalse(output.exists())
