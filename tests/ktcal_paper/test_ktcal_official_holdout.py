import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from ktcal_paper.official_holdout import evaluate_ktcal_official_holdout
from ktcal_paper.matching import PaperMatcher


class FixedLogits(torch.nn.Module):
    def forward(self, frames):
        batch, length = frames.shape[:2]
        x = torch.zeros(batch, length, 80)
        y = torch.zeros(batch, length, 60)
        x[..., 40] = 1.
        y[..., 30] = 1.
        return x, y


class PaddingAwareLogits(torch.nn.Module):
    def forward(self, frames):
        batch, length = frames.shape[:2]
        x = torch.zeros(batch, length, 80)
        y = torch.zeros(batch, length, 60)
        positions = (10, 12, 14, 79)
        for time, position in enumerate(positions[:length]):
            x[:, time, position] = 100.
        y[..., 30] = 100.
        return x, y


class KTCalOfficialHoldoutTests(unittest.TestCase):
    def test_offline_filter_ignores_record_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_200_200.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((1, 4, 2, 60, 80), dtype=np.uint8)
                labels = np.zeros((1, 4, 3), dtype=np.float32)
                expected_x = ((10 + 2 / 3) / 80, 12 / 80,
                              (13 + 1 / 3) / 80)
                labels[0, :3, 0] = expected_x
                labels[0, :3, 1] = .5
                handle['labels'] = labels
            stnet = root / 'stnet.pt'
            matcher = root / 'matcher.pt'
            stnet.write_bytes(b'stnet')
            matcher.write_bytes(b'matcher')

            with mock.patch(
                'ktcal_paper.official_holdout.load_stnet_checkpoint',
                return_value=(PaddingAwareLogits(), {'best_epoch': 36}, 1.),
            ), mock.patch(
                'ktcal_paper.official_holdout.load_matcher_checkpoint',
                return_value=(PaperMatcher(), {'selected_epoch': 5}),
            ):
                report = evaluate_ktcal_official_holdout(
                    h5,
                    stnet,
                    matcher,
                    root / 'report.json',
                    recording_lengths=[3],
                    chunk_size=4,
                    device='cpu',
                    stnet_decode='soft',
                    offline_filter='ma3',
                )

            self.assertLess(
                report['metrics']['stnet_offline']['mean_error_px'], 1e-5)

    def test_offline_filter_and_selected_decode_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_200_200.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((1, 5, 2, 60, 80), dtype=np.uint8)
                labels = np.zeros((1, 5, 3), dtype=np.float32)
                labels[..., :2] = [.5, .5]
                handle['labels'] = labels
            stnet = root / 'stnet.pt'
            matcher = root / 'matcher.pt'
            stnet.write_bytes(b'stnet')
            matcher.write_bytes(b'matcher')

            with mock.patch(
                'ktcal_paper.official_holdout.load_stnet_checkpoint',
                return_value=(FixedLogits(), {'best_epoch': 36}, 1.),
            ), mock.patch(
                'ktcal_paper.official_holdout.load_matcher_checkpoint',
                return_value=(PaperMatcher(), {'selected_epoch': 5}),
            ):
                report = evaluate_ktcal_official_holdout(
                    h5,
                    stnet,
                    matcher,
                    root / 'report.json',
                    recording_lengths=[5],
                    chunk_size=5,
                    device='cpu',
                    stnet_decode='soft',
                    offline_filter='ma11',
                )

            self.assertEqual(report['stnet_decode'], 'soft')
            self.assertEqual(report['offline_filter'], 'ma11')
            self.assertIn('stnet_offline', report['metrics'])
            self.assertIn('ktcal_offline', report['metrics'])
            self.assertEqual(report['metrics']['stnet_offline']['frames'], 5)
            self.assertEqual(report['metrics']['ktcal_offline']['frames'], 5)

    def test_cli_exposes_final_offline_options(self):
        from scripts.evaluate_ktcal_official_holdout import build_parser

        args = build_parser().parse_args([
            '--h5', 'test_200_200.h5',
            '--stnet-checkpoint', 'student.pt',
            '--matcher-checkpoint', 'matcher.pt',
            '--events', 'num_events.txt',
            '--output', 'report.json',
        ])

        self.assertIsNone(args.stnet_reference)
        self.assertEqual(args.stnet_decode, 'hard')
        self.assertEqual(args.offline_filter, 'none')

    def test_evaluation_trims_padding_and_records_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_200_200.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((2, 4, 2, 60, 80), dtype=np.uint8)
                labels = np.zeros((2, 4, 3), dtype=np.float32)
                labels[..., :2] = [.5, .5]
                handle['labels'] = labels
            stnet = root / 'stnet.pt'
            matcher = root / 'matcher.pt'
            stnet.write_bytes(b'stnet')
            matcher.write_bytes(b'matcher')

            with mock.patch(
                'ktcal_paper.official_holdout.load_stnet_checkpoint',
                return_value=(FixedLogits(), {'best_epoch': 36}, 1.),
            ), mock.patch(
                'ktcal_paper.official_holdout.load_matcher_checkpoint',
                return_value=(PaperMatcher(), {'selected_epoch': 5}),
            ):
                report = evaluate_ktcal_official_holdout(
                    h5,
                    stnet,
                    matcher,
                    root / 'report.json',
                    recording_lengths=[7],
                    chunk_size=4,
                    device='cpu',
                )

            self.assertEqual(report['valid_frames'], 7)
            self.assertEqual(report['metrics']['stnet_only']['frames'], 7)
            self.assertEqual(report['metrics']['ktcal_online']['frames'], 7)
            self.assertFalse(report['final_holdout_used'])
            self.assertEqual(report['metrics']['stnet_only']['p3'], 1.)
            self.assertEqual(report['matching']['fusion_alpha'], .75)
            self.assertEqual(report['matching']['max_prediction_variance_px2'], 10.)

    def test_evaluation_refuses_to_overwrite_an_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5 = root / 'test_200_200.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((1, 1, 2, 60, 80), dtype=np.uint8)
                handle['labels'] = np.zeros((1, 1, 3), dtype=np.float32)
            output = root / 'report.json'
            output.write_text('{}')

            with self.assertRaisesRegex(FileExistsError, 'overwrite'):
                evaluate_ktcal_official_holdout(
                    h5,
                    'stnet.pt',
                    'matcher.pt',
                    output,
                    recording_lengths=[1],
                    device='cpu',
                )


if __name__ == '__main__':
    unittest.main()
