import struct
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hardware.rv1106.export_matcher import export_matcher
from hardware.rv1106.export_stnet import LastFrameLogits, validate_board_config
from ktcal_paper.matching import PaperMatcher, crop_rectangle, response_to_pixel


ROOT = Path(__file__).resolve().parents[2]


class HardwareExportTests(unittest.TestCase):
    def test_board_export_rejects_offline_and_incompatible_windows(self):
        for config in ({'window': 100, 'causal_last_step': False},
                       {'window': 32, 'causal_last_step': True},
                       {'window': 16, 'causal_last_step': False}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, '16-frame causal'):
                validate_board_config(config)
        validate_board_config({'window': 16, 'causal_last_step': True})

    def test_board_logits_are_last_frame_only_with_training_divisor(self):
        class ToyModel(torch.nn.Module):
            def forward(self, frames):
                x = frames[:, :, 0, 0, 0]
                y = frames[:, :, 1, 0, 0]
                return x.unsqueeze(-1).expand(-1, -1, 80), y.unsqueeze(-1).expand(-1, -1, 60)
        frames = torch.zeros(1, 16, 2, 60, 80)
        frames[0, -1, 0, 0, 0] = 6
        frames[0, -1, 1, 0, 0] = 10
        x, y = LastFrameLogits(ToyModel(), input_divisor=2)(frames)
        self.assertEqual(tuple(x.shape), (1, 80))
        self.assertEqual(tuple(y.shape), (1, 60))
        self.assertEqual((x[0, 0].item(), y[0, 0].item()), (3., 5.))

    def test_export_preserves_matcher_weights_and_channel_count(self):
        model = PaperMatcher(channels=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            checkpoint = path / 'best.pt'
            output = path / 'matcher.bin'
            torch.save({'config': {'ncc_channels': 3},
                        'matcher_state': model.state_dict()}, checkpoint)
            export_matcher(checkpoint, output)
            raw = output.read_bytes()
            self.assertEqual(raw[:8], b'KTMATCH1')
            self.assertEqual(struct.unpack('<I', raw[8:12]), (3,))
            self.assertEqual(len(raw), 12 + (18 + 6) * 4)
            self.assertEqual(struct.unpack('<18f', raw[12:84]),
                             tuple(model.extractor[0].weight.flatten().tolist()))

    def test_rejects_missing_matcher_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'bad.pt'
            torch.save({}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'matcher_state'):
                export_matcher(checkpoint, Path(folder) / 'output.bin')
            self.assertFalse((Path(folder) / 'output.bin').exists())

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler unavailable')
    def test_cpp_matcher_matches_python_reference_on_identical_frames(self):
        torch.manual_seed(112)
        matcher = PaperMatcher(channels=3).eval()
        frames = torch.rand(2, 2, 60, 80)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, weights, binary, executable = (
                root / 'matcher.pt', root / 'matcher.bin',
                root / 'frames.bin', root / 'parity')
            torch.save({'config': {'ncc_channels': 3},
                        'matcher_state': matcher.state_dict()}, checkpoint)
            export_matcher(checkpoint, weights)
            np.asarray(frames.numpy(), dtype='<f4').tofile(binary)
            subprocess.run(['c++', '-std=c++17', '-O2',
                            str(ROOT / 'hardware/rv1106/fast_path.cc'),
                            str(ROOT / 'hardware/rv1106/parity_cli.cc'),
                            '-o', str(executable)], check=True)
            x, y, confidence, border = map(float, subprocess.check_output(
                [executable, weights, binary], text=True).split())
            template = crop_rectangle(frames[:1], torch.tensor([[40.1, 30.2]]),
                                      (30, 25))
            search = crop_rectangle(frames[1:], torch.tensor([[40., 30.]]),
                                    (35, 30))
            local, score, _, details = matcher.match_with_diagnostics(template, search)
            expected = response_to_pixel(local, torch.tensor([[40., 30.]]))[0]
            self.assertAlmostEqual(x, float(expected[0]), places=4)
            self.assertAlmostEqual(y, float(expected[1]), places=4)
            self.assertAlmostEqual(confidence, float(score[0]), places=4)
            self.assertEqual(bool(border), bool(details['border_peak'][0]))


if __name__ == '__main__':
    unittest.main()
