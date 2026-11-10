"""Evaluation entrypoints must restore STNet inference behavior, not just weights."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from models.STNet import STNet
from scripts import evaluate_official_3et_protocol as official
from scripts import evaluate_stnet_checkpoint as chunked
from scripts import evaluate_stnet_sequences as sequences


class EvaluationReadoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def checkpoint(self, readout='final_hidden'):
        path = self.root / 'best.pt'
        torch.save({
            'model_state_dict': STNet().state_dict(),
            'config': {'window': 2, 'input_divisor': 1.0,
                       'temporal_spatial_kernel': 1, 'readout': readout},
        }, path)
        return path

    def manifest(self):
        with h5py.File(self.root / 'frames.h5', 'w') as handle:
            group = handle.create_group('1_1')
            group['frames'] = np.ones((2, 2, 60, 80), dtype=np.uint8)
            group['labels'] = np.full((2, 3), 0.5, dtype=np.float32)
            group['valid'] = np.ones(2, dtype=bool)
        path = self.root / 'manifest.json'
        path.write_text(json.dumps({
            'schema': 1, 'h5': 'frames.h5', 'labels_are_placeholders': False,
            'splits': {'train': [], 'val': ['1_1'], 'holdout': []},
            'records': [{'recording': '1_1'}],
        }))
        return path

    def run_evaluator(self, module, args):
        """Run the real CLI and capture the real STNet constructed by it."""
        created = []

        def build(arguments):
            model = STNet(arguments)
            created.append(model)
            return model

        with patch.object(module, 'STNet', side_effect=build), \
                patch.object(sys, 'argv', ['evaluation', *args]), \
                contextlib.redirect_stdout(io.StringIO()):
            module.main()
        self.assertEqual(len(created), 1)
        return created[0]

    def test_official_checkpoint_restores_final_hidden_and_legacy_last(self):
        weights = STNet().state_dict()
        for config, expected in (({'readout': 'final_hidden'}, 'final_hidden'),
                                 ({}, 'last')):
            with self.subTest(config=config):
                model, *_ = official._build_model(
                    'stnet', {'model_state_dict': weights, 'config': config})
                self.assertEqual(model.readout, expected)

    def test_chunked_checkpoint_cli_restores_final_hidden(self):
        model = self.run_evaluator(chunked, [
            '--checkpoint', str(self.checkpoint()),
            '--manifest', str(self.manifest()), '--mode', 'record',
            '--batch-size', '1', '--threads', '1', '--device', 'cpu',
        ])
        self.assertEqual(model.readout, 'final_hidden')

    def test_full_sequence_cli_restores_final_hidden(self):
        model = self.run_evaluator(sequences, [
            '--checkpoint', str(self.checkpoint()),
            '--manifest', str(self.manifest()), '--window-sizes', '2',
            '--threads', '1', '--device', 'cpu',
        ])
        self.assertEqual(model.readout, 'final_hidden')


if __name__ == '__main__':
    unittest.main()
