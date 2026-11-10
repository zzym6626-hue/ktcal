import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from ktcal_paper.combined_3et import (
    CausalLastStepSTNet,
    evaluate_combined_3et,
    evaluate_online_causal_3et,
    load_matcher_checkpoint,
    load_stnet_checkpoint,
    moving_average_sequence_predictions,
    precompute_global_predictions,
    summarize_sequence_predictions,
    validate_validation_h5,
)
from ktcal_paper.matching import PaperMatcher
from models.STNet import STNet


class Combined3ETTests(unittest.TestCase):
    def test_moving_average_filters_each_sequence_with_edge_padding(self):
        values = np.array([
            [0., 0.],
            [2., 2.],
            [4., 4.],
            [6., 6.],
        ], dtype=np.float32)
        filtered = moving_average_sequence_predictions([values], window=3)
        expected = np.array([
            [2. / 3., 2. / 3.],
            [2., 2.],
            [4., 4.],
            [16. / 3., 16. / 3.],
        ], dtype=np.float32)
        self.assertEqual(len(filtered), 1)
        np.testing.assert_allclose(filtered[0], expected, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, 'positive odd'):
            moving_average_sequence_predictions([values], window=2)

    def test_combined_report_can_add_offline_moving_average_metrics(self):
        class FixedLogits(torch.nn.Module):
            def forward(self, frames):
                batch, length = frames.shape[:2]
                x = torch.zeros(batch, length, 80)
                y = torch.zeros(batch, length, 60)
                x[..., 40] = 1.
                y[..., 30] = 1.
                return x, y

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_aug.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((1, 3, 2, 60, 80), dtype=np.uint8)
                labels = np.zeros((1, 3, 3), dtype=np.float32)
                labels[..., :2] = [.5, .5]
                handle['label'] = labels
            stnet = root / 'stnet.pt'
            matcher = root / 'matcher.pt'
            stnet.write_bytes(b'stnet')
            matcher.write_bytes(b'matcher')
            with mock.patch('ktcal_paper.combined_3et.load_stnet_checkpoint',
                            return_value=(FixedLogits(), {}, 1.)), \
                 mock.patch('ktcal_paper.combined_3et.load_matcher_checkpoint',
                            return_value=(PaperMatcher(), {})):
                report = evaluate_combined_3et(
                    h5, stnet, matcher, root / 'report.json', device='cpu',
                    offline_filter='ma11')
            self.assertEqual(report['offline_filter'], 'ma11')
            self.assertEqual(
                report['metrics']['stnet_offline']['frames'], 3)
            self.assertEqual(
                report['metrics']['ktcal_offline']['frames'], 3)
            self.assertFalse(report['final_holdout_used'])

    def test_online_cli_is_validation_only_with_default_gate(self):
        from scripts.evaluate_3et_causal_online import build_parser

        args = build_parser().parse_args([
            '--h5', 'test_aug.h5', '--stnet-checkpoint', 'student.pt',
            '--matcher-checkpoint', 'matcher.pt', '--output', 'report.json',
        ])
        self.assertEqual(args.max_prediction_variance_px2, 10.)
        self.assertEqual(args.max_consecutive_prediction_skips, 1)

    def test_causal_adapter_decodes_only_the_last_step_and_scales_input(self):
        class FixedLogits(torch.nn.Module):
            def forward(self, frames):
                self.inputs = frames.detach().clone()
                x = torch.zeros(frames.shape[0], frames.shape[1], 80)
                y = torch.zeros(frames.shape[0], frames.shape[1], 60)
                x[:, :-1, 50] = 100
                x[:, -1, 12] = 100
                y[:, -1, 23] = 100
                return x, y

        net = FixedLogits()
        model = CausalLastStepSTNet(net, divisor=2.)
        output = model(torch.full((2, 16, 2, 60, 80), 4.))
        torch.testing.assert_close(net.inputs,
                                   torch.full_like(net.inputs, 2.))
        torch.testing.assert_close(output, torch.tensor([[12 / 80, 23 / 60]] * 2))

    def test_online_causal_report_uses_on_demand_model_without_holdout(self):
        class FixedLogits(torch.nn.Module):
            def forward(self, frames):
                x = torch.zeros(*frames.shape[:2], 80)
                y = torch.zeros(*frames.shape[:2], 60)
                x[:, -1, 40] = 100
                y[:, -1, 30] = 100
                return x, y

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_aug.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((1, 3, 2, 60, 80), dtype=np.uint8)
                labels = np.zeros((1, 3, 3), dtype=np.float32)
                labels[..., :2] = [.5, .5]
                handle['label'] = labels
            student = root / 'student.pt'
            matcher = root / 'matcher.pt'
            student.write_bytes(b'student')
            matcher.write_bytes(b'matcher')
            with mock.patch('ktcal_paper.combined_3et.load_stnet_checkpoint',
                            return_value=(FixedLogits(), {}, 1.)), \
                 mock.patch('ktcal_paper.combined_3et.load_matcher_checkpoint',
                            return_value=(PaperMatcher(), {})):
                report = evaluate_online_causal_3et(
                    h5, student, matcher, root / 'online.json', device='cpu',
                    max_prediction_variance_px2=10.)
            self.assertEqual(report['metrics']['ktcal_online']['frames'], 3)
            self.assertEqual(report['tracker_stats']['frames'], 3)
            self.assertGreaterEqual(report['tracker_stats']['stnet_calls'], 1)
            self.assertEqual(report['tracker_stats']['global_prediction_source'],
                             'on_demand_causal_16_step')
            self.assertEqual(report['stnet_readout'], 'last')
            self.assertIsNone(report['matching']['fusion_alpha'])
            self.assertFalse(report['final_holdout_used'])
            self.assertGreater(report['latency']['average_frame_ms'], 0)
            with self.assertRaisesRegex(ValueError, 'final holdout'):
                evaluate_online_causal_3et(root / 'test_200_200.h5',
                    student, matcher, root / 'holdout_report.json', device='cpu')

    def test_evaluation_records_opt_in_prediction_gate_without_using_holdout(self):
        class FixedLogits(torch.nn.Module):
            def forward(self, frames):
                batch, length = frames.shape[:2]
                x = torch.zeros(batch, length, 80)
                y = torch.zeros(batch, length, 60)
                x[..., 40] = 1.
                y[..., 30] = 1.
                return x, y

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_aug.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((1, 3, 2, 60, 80), dtype=np.uint8)
                labels = np.zeros((1, 3, 3), dtype=np.float32)
                labels[..., :2] = [.5, .5]
                handle['label'] = labels
            stnet = root / 'stnet.pt'
            matcher = root / 'matcher.pt'
            stnet.write_bytes(b'stnet')
            matcher.write_bytes(b'matcher')
            with mock.patch('ktcal_paper.combined_3et.load_stnet_checkpoint',
                            return_value=(FixedLogits(), {}, 1.)), \
                 mock.patch('ktcal_paper.combined_3et.load_matcher_checkpoint',
                            return_value=(PaperMatcher(), {})):
                report = evaluate_combined_3et(
                    h5, stnet, matcher, root / 'report.json', device='cpu',
                    max_prediction_variance_px2=25., max_consecutive_prediction_skips=1)
            self.assertEqual(report['tracker_stats']['frames'], 3)
            self.assertEqual(report['tracker_stats']['skipped_matches'], 1)
            self.assertEqual(report['matching']['max_prediction_variance_px2'], 25.)
            self.assertFalse(report['final_holdout_used'])

    def test_precompute_uses_hard_argmax_pixels_and_checkpoint_divisor(self):
        class FixedLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = None

            def forward(self, frames):
                self.seen = frames.detach().cpu()
                batch, length = frames.shape[:2]
                x = torch.zeros(batch, length, 80, device=frames.device)
                y = torch.zeros(batch, length, 60, device=frames.device)
                x[..., 17] = 4
                y[..., 23] = 5
                return x, y

        model = FixedLogits()
        frames = torch.full((2, 3, 2, 4, 5), 10.)
        result = precompute_global_predictions(
            model, frames, torch.device('cpu'), input_divisor=2.)

        self.assertEqual(result.shape, (2, 3, 2))
        np.testing.assert_array_equal(
            result, np.broadcast_to(np.array([17., 23.]), result.shape))
        torch.testing.assert_close(model.seen, torch.full_like(frames, 5.))

    def test_precompute_soft_decode_uses_expectation(self):
        class SpreadLogits(torch.nn.Module):
            def forward(self, frames):
                batch, length = frames.shape[:2]
                x = torch.full((batch, length, 80), -100.)
                y = torch.full((batch, length, 60), -100.)
                x[..., 40] = 2.
                x[..., 41] = 1.
                y[..., 30] = 2.
                y[..., 31] = 1.
                return x, y

        result = precompute_global_predictions(
            SpreadLogits(), torch.ones(1, 2, 2, 4, 5), torch.device('cpu'),
            input_divisor=1., decode='soft')
        # softmax([2,1,0...]) gives expectation slightly above argmax 40
        self.assertEqual(result.shape, (1, 2, 2))
        self.assertTrue((result[..., 0] > 40).all())
        self.assertTrue((result[..., 0] < 41).all())
        self.assertTrue((result[..., 1] > 30).all())
        self.assertTrue((result[..., 1] < 31).all())

    def test_checkpoint_loaders_keep_stnet_and_matcher_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                stnet_block3_kernel=5,
                stnet_unidirectional=False,
                stnet_decode='argmax',
                stnet_temporal_spatial_kernel=1,
            )
            stnet = STNet(args)
            stnet_path = root / 'stnet.pt'
            torch.save({
                'model_state_dict': stnet.state_dict(),
                'best_epoch': 36,
                'config': {
                    'input_divisor': 1.0,
                    'temporal_spatial_kernel': 1,
                },
            }, stnet_path)
            matcher = PaperMatcher()
            matcher_path = root / 'matcher.pt'
            torch.save({
                'matcher_state': matcher.state_dict(),
                'model_state': {'incompatible': torch.ones(1)},
                'config': {'ncc_channels': 16, 'temperature': .1},
                'matcher_finetune': {'selected_epoch': 5},
            }, matcher_path)

            loaded_stnet, stnet_config, divisor = load_stnet_checkpoint(
                stnet_path, torch.device('cpu'))
            loaded_matcher, matcher_config = load_matcher_checkpoint(
                matcher_path, torch.device('cpu'))

            self.assertIsInstance(loaded_stnet, STNet)
            self.assertIsInstance(loaded_matcher, PaperMatcher)
            self.assertEqual(divisor, 1.)
            self.assertEqual(stnet_config['temporal_spatial_kernel'], 1)
            self.assertEqual(stnet_config['best_epoch'], 36)
            self.assertEqual(loaded_stnet.readout, 'last')
            self.assertEqual(matcher_config['ncc_channels'], 16)
            self.assertEqual(matcher_config['selected_epoch'], 5)

    def test_checkpoint_loader_uses_opt_in_final_hidden_readout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'final_hidden.pt'
            source = STNet(SimpleNamespace(stnet_readout='final_hidden')).eval()
            torch.save({'model_state_dict': source.state_dict(),
                        'config': {'readout': 'final_hidden', 'input_divisor': 1.}}, path)
            loaded, config, _ = load_stnet_checkpoint(path, torch.device('cpu'))
            self.assertEqual(config['readout'], 'final_hidden')
            self.assertEqual(loaded.readout, 'final_hidden')
            frames = torch.randn(1, 2, 2, 60, 80)
            with torch.no_grad():
                reference = source(frames)
                actual = loaded(frames)
            for actual_logits, expected_logits in zip(actual, reference):
                torch.testing.assert_close(actual_logits, expected_logits)

    def test_sequence_metrics_are_weighted_by_frames(self):
        labels = [np.zeros((1, 2)), np.zeros((3, 2))]
        predictions = [
            np.zeros((1, 2)),
            np.tile(np.array([[2 / 80, 0.]]), (3, 1)),
        ]
        metrics = summarize_sequence_predictions(predictions, labels)
        self.assertEqual(metrics['frames'], 4)
        self.assertAlmostEqual(metrics['mean_error_px'], 1.5)

    def test_holdout_is_rejected_by_name_before_file_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test_200_200.h5'
            with self.assertRaisesRegex(ValueError, 'final holdout'):
                validate_validation_h5(path)

    def test_holdout_is_rejected_by_official_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'renamed.h5'
            with h5py.File(path, 'w') as handle:
                handle.create_dataset(
                    'frames', shape=(50, 200, 2, 60, 80), dtype='u1')
                handle.create_dataset(
                    'label', shape=(50, 200, 3), dtype='f4')
            with self.assertRaisesRegex(ValueError, 'final holdout'):
                validate_validation_h5(path)


if __name__ == '__main__':
    unittest.main()
