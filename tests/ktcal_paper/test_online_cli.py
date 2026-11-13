import unittest
import json
from pathlib import Path
import tempfile

from scripts.evaluate_3et_online import build_parser, validate_args, parse_args
import numpy as np
import torch


class OnlineCLITests(unittest.TestCase):
    def test_persistent_state_control_is_forwarded_without_changing_work_counters(self):
        from scripts.evaluate_3et_online import parse_args, build_predictor
        from models.streaming_stnet import StreamingSTNet
        model = StreamingSTNet(hidden_size=128).eval()
        for flags, expected in (([], False), (['--persistent-state'], True),
                                (['--no-persistent-state'], False)):
            args = parse_args(['--output', 'unused', '--history-stride', '1'] + flags)
            predictor = build_predictor(model, 1., [(0, 0, False)], args, torch.device('cpu'))
            self.assertEqual(predictor.step.persistent_state, expected)
            predictor.observe(torch.zeros(1, 2, 60, 80))
            predictor(torch.zeros(1, 1, 2, 60, 80), np.ones(1, dtype=bool))
            self.assertEqual(predictor.received_frame_updates, 1)
            self.assertEqual(predictor.coordinate_head_requests, 1)

    def test_streaming_cli_builds_stateful_predictor_and_counts_all_updates(self):
        from scripts.evaluate_3et_online import build_predictor, predictor_work_stats
        from models.streaming_stnet import StreamingSTNet
        from ktcal_paper.streaming_inference import StreamingPredictor
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        args = build_parser().parse_args(['--output', 'unused.json', '--history-stride', '1'])
        model = StreamingSTNet(hidden_size=128).eval()
        predictor = build_predictor(model, 1., [(0, 0, False)], args, torch.device('cpu'))
        self.assertIsInstance(predictor, StreamingPredictor)
        frames = torch.rand(1, 3, 2, 60, 80)
        with torch.inference_mode():
            for i in range(3):
                predictor.observe(frames[:, i])
                if i == 0:
                    predictor(frames[:, :1], np.array([True]))
        stats = predictor_work_stats(dict(stnet_requests=1, model_forward_passes=1), predictor)
        self.assertEqual(stats['model_forward_passes'], 3)
        self.assertEqual(stats['feature_updates'], 3)
        self.assertEqual(stats['coordinate_head_forward_passes'], 1)
        self.assertEqual(stats['coordinate_head_requests'], 1)

    def test_streaming_cli_rejects_stride_or_synthetic_future(self):
        from scripts.evaluate_3et_online import build_predictor
        from models.streaming_stnet import StreamingSTNet
        model = StreamingSTNet(hidden_size=128).eval()
        for option, value in (('--history-stride', '2'), ('--right-context', '1')):
            args = build_parser().parse_args(['--output', 'unused.json', '--history-stride', '1', option, value])
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'Streaming'):
                build_predictor(model, 1., [(0, 0, False)], args, torch.device('cpu'))

    def test_reviewable_json_config_and_explicit_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'online.json'
            path.write_text(json.dumps(dict(confidence=.82, right_context=2)))
            args = parse_args(['--config', str(path), '--output', 'unused.json',
                               '--confidence', '.75'])
            self.assertEqual(args.confidence, .75)
            self.assertEqual(args.right_context, 2)
            path.write_text(json.dumps(dict(unreviewed_future_heuristic=True)))
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                parse_args(['--config', str(path), '--output', 'unused.json'])
            for field, value in (('no_psr', 'false'), ('no_graph', 'false'),
                                  ('no_optimize_stnet', 'false'), ('window', 16.5),
                                  ('threads', True), ('ensemble', 'unverified')):
                path.write_text(json.dumps({field: value}))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Invalid'):
                    parse_args(['--config', str(path), '--output', 'unused.json'])

    def test_defaults_are_causal_kf60(self):
        args = build_parser().parse_args(['--output', 'unused.json'])
        size, views = validate_args(args)
        self.assertEqual(size, (37, 32))
        self.assertEqual(len(views), 2)
        self.assertEqual((args.window, args.history_stride), (16, 2))
        self.assertEqual(args.confidence, .28)
        self.assertEqual(args.r_px, 32.)
        self.assertIsNone(args.ema_alpha)

    def test_invalid_online_configuration_rejected(self):
        for option, value in (('--window', '0'), ('--history-stride', '0'),
                              ('--ema-alpha', '0'), ('--views', '0:0:2'),
                              ('--r-px', 'nan'), ('--search-size', '29x25')):
            args = build_parser().parse_args(['--output', 'unused.json', option, value])
            with self.subTest(option=option), self.assertRaises(ValueError):
                validate_args(args)
