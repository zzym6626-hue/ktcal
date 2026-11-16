"""Published 100-frame hardware contract, independently checked on the host."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import importlib.util

from ktcal_paper.matching import PaperMatcher, crop_rectangle
from ktcal_paper.offline_batch import BatchKalman, OfflineBatchTracker
from ktcal_paper.offline_experiment import spatial_view, undo_view, smooth_pixels
from ktcal_paper.evaluation_protocol import soft_expectation_decode
from hardware.rv1106.export_matcher import export_matcher

from hardware.rv1106 import export_stnet as exporter
from models.STNet import STNet

ROOT = Path(__file__).resolve().parents[2]


class SequenceExportTests(unittest.TestCase):
    def test_full_frame_bridge_preserves_all_steps_and_normalizes_once(self):
        class Toy(torch.nn.Module):
            def forward(self, frames):
                return (frames[:, :, 0, 0, 0, None].expand(-1, -1, 80),
                        frames[:, :, 1, 0, 0, None].expand(-1, -1, 60))
        frames = torch.zeros(1, 100, 2, 60, 80)
        frames[0, :, 0, 0, 0] = torch.arange(100) * 2
        bridge = getattr(exporter, 'FullFrameLogits', None)
        self.assertIsNotNone(bridge, '100-frame export bridge is missing')
        x, y = bridge(Toy(), 2)(frames)
        self.assertEqual(tuple(x.shape), (1, 100, 80))
        self.assertEqual(tuple(y.shape), (1, 100, 60))
        torch.testing.assert_close(x[0, :, 0], torch.arange(100).float())

    def test_fixed_pool_matches_original_adaptive_pool(self):
        pool = getattr(exporter, 'StaticSpatialPool', None)
        self.assertIsNotNone(pool, 'export-safe spatial pooling is missing')
        values = torch.randn(1, 100, 128, 6, 8)
        expected = torch.nn.AdaptiveAvgPool3d((128, 4, 4))(values)
        torch.testing.assert_close(pool()(values), expected, atol=1e-6, rtol=1e-6)

    def test_sequence_config_accepts_release_rejects_other_architectures(self):
        validate = getattr(exporter, 'validate_sequence_config', None)
        self.assertIsNotNone(validate, 'sequence config validator is missing')
        config = dict(window=100, readout='attention', temporal_variant='dense',
                      temporal_spatial_kernel=1, block3_kernel=5,
                      unidirectional=False, input_divisor=1.)
        validate(config)
        for change in (dict(window=16), dict(unidirectional=True),
                       dict(readout='last'), dict(input_divisor=float('nan'))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(dict(config, **change))

    def test_verifier_rejects_wrong_model_hash_before_loading_runtime(self):
        import importlib
        module = 'hardware.rv1106.verify_stnet'
        self.assertIsNotNone(importlib.util.find_spec(module), 'ONNX verification command is missing')
        verifier = importlib.import_module(module)
        with tempfile.TemporaryDirectory() as folder:
            import json
            path = Path(folder)
            checkpoint, onnx = path / 'model.pt', path / 'model.onnx'
            checkpoint.write_bytes(b'checkpoint')
            onnx.write_bytes(b'onnx')
            onnx.with_suffix('.onnx.json').write_text(json.dumps(dict(
                mode='sequence100', checkpoint_sha256='wrong', onnx_sha256='wrong')))
            with self.assertRaisesRegex(ValueError, 'hash'):
                verifier.validate_export_manifest(checkpoint, onnx)

    def test_sequence_sdk_adapter_and_bad_input_rejection(self):
        test = ROOT / 'hardware/rv1106/test_sequence_backend.cc'
        self.assertTrue(test.exists(), 'sequence RKNN adapter test is missing')
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / 'backend_test'
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT / 'tests/fixtures/rknn'),
                            '-I', str(ROOT / 'hardware/rv1106'),
                            str(ROOT / 'hardware/rv1106/fast_path.cc'),
                            str(ROOT / 'hardware/rv1106/sequence_path.cc'),
                            str(ROOT / 'hardware/rv1106/rknn_sequence_backend.cc'),
                            str(test), '-o', str(executable)], check=True)
            subprocess.run([str(executable)], check=True)

    def test_rknn_conversion_refuses_unverified_onnx_and_preserves_existing_output(self):
        import importlib
        import json
        module = 'hardware.rv1106.convert_rknn'
        self.assertIsNotNone(importlib.util.find_spec(module), 'RKNN conversion command missing')
        converter = importlib.import_module(module)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            onnx = path / 'model.onnx'
            onnx.write_bytes(b'onnx')
            verified = path / 'verified.json'
            verified.write_text(json.dumps(dict(passed=False)))
            with self.assertRaises(ValueError):
                converter.convert(onnx, verified, path / 'output.rknn')
            existing = path / 'old.rknn'
            existing.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                converter.convert(onnx, verified, existing)
            self.assertEqual(existing.read_bytes(), b'preserve')

    def test_board_input_and_scoring_cover_every_record_without_holdout(self):
        import importlib
        import h5py
        module = 'hardware.rv1106.records'
        self.assertIsNotNone(importlib.util.find_spec(module), 'board input/scoring command missing')
        records = importlib.import_module(module)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            h5 = path / 'test_aug.h5'
            with h5py.File(h5, 'w') as handle:
                handle['frames'] = np.zeros((2, 100, 2, 60, 80), np.float32)
                handle['label'] = np.tile([.5, .5, 1.], (2, 100, 1)).astype(np.float32)
            binary = path / 'records.bin'
            records.export_records(h5, binary)
            self.assertEqual(binary.stat().st_size, 2*100*2*60*80*4)
            csv = path / 'predictions.csv'
            header = 'record,frame,x_px,y_px,raw_x_px,raw_y_px,tier\n'
            rows = [f'{record},{frame},40,30,40,30,1\n' for record in range(2) for frame in range(100)]
            csv.write_text(header + ''.join(rows))
            result = records.score_records(h5, csv, path / 'score.json')
            self.assertEqual(result['metrics']['frames'], 200)
            self.assertEqual(result['metrics']['p3'], 1.)
            csv.write_text(header + ''.join(rows[:-1]))
            with self.assertRaises(ValueError):
                records.score_records(h5, csv, path / 'missing.json')
            with self.assertRaises(ValueError):
                records.export_records(path / 'test_200_200.h5', path / 'forbidden.bin')

    def test_rknn_conversion_requires_explicit_success_and_releases_sdk(self):
        import json
        import sys
        from unittest.mock import patch
        from hardware.rv1106.convert_rknn import convert
        from ktcal_paper.combined_3et import _sha256

        # The SDK is not installed on this host. Exercise the real converter
        # against a dependency double, including an export with a partial file.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            onnx, verified = path / 'model.onnx', path / 'verified.json'
            onnx.write_bytes(b'onnx')
            metadata = dict(mode='sequence100', onnx_sha256=_sha256(onnx),
                            checkpoint_sha256='checkpoint', runtime_input_divisor=1.)
            onnx.with_suffix('.onnx.json').write_text(json.dumps(metadata))
            verified.write_text(json.dumps(dict(metadata, passed=True)))

            class SDK:
                stage = None
                status = 0
                released = False

                def config(self, **kwargs):
                    return None  # config alone has no required status code.

                def load_onnx(self, **kwargs):
                    return self.status if self.stage == 'load_onnx' else 0

                def build(self, **kwargs):
                    return self.status if self.stage == 'build' else 0

                def export_rknn(self, output):
                    Path(output).write_bytes(b'sdk-output')
                    return self.status if self.stage == 'export_rknn' else 0

                def release(self):
                    self.released = True

            for stage in ('load_onnx', 'build', 'export_rknn'):
                for status in (None, -1):
                    with self.subTest(stage=stage, status=status):
                        sdk = SDK()
                        sdk.stage, sdk.status = stage, status
                        output = path / f'{stage}_{status}.rknn'
                        modules = {'rknn': SimpleNamespace(),
                                   'rknn.api': SimpleNamespace(RKNN=lambda **kwargs: sdk)}
                        with patch.dict(sys.modules, modules):
                            with self.assertRaisesRegex(RuntimeError, stage):
                                convert(onnx, verified, output)
                        self.assertTrue(sdk.released)
                        self.assertFalse(output.with_suffix('.rknn.json').exists())
            sdk = SDK()
            with patch.dict(sys.modules, {'rknn': SimpleNamespace(),
                                          'rknn.api': SimpleNamespace(RKNN=lambda **kwargs: sdk)}):
                report = convert(onnx, verified, path / 'success.rknn')
            self.assertTrue(sdk.released)
            self.assertTrue(report['conversion_completed'])
            self.assertFalse(report['board_verified'])

    def test_board_make_uses_rv1106_runtime_and_accepts_sdk_override(self):
        args = ['make', '-n', '-C', str(ROOT / 'hardware/rv1106'), 'board',
                'RKNN_INC=/sdk/include', 'RKNN_LIB=/sdk/lib']
        default = subprocess.check_output(args, text=True)
        self.assertIn('-lrknnmrt', default)
        other = subprocess.check_output(args + ['RKNN_RUNTIME=rknnrt'], text=True)
        self.assertIn('-lrknnrt', other)


@unittest.skipUnless(shutil.which('c++'), 'C++ compiler unavailable')
class SequenceCppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix='ktcal-sequence-test-')
        cls.path = Path(cls.folder.name)
        cls.executable = cls.path / 'sequence_parity'

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def run_cpp(self, mode, *inputs):
        source = ROOT / 'hardware/rv1106/sequence_path.cc'
        self.assertTrue(source.exists(), '100-frame C++ pipeline is missing')
        if not self.executable.exists():
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT / 'hardware/rv1106'),
                            str(ROOT / 'hardware/rv1106/fast_path.cc'), str(source),
                            str(ROOT / 'hardware/rv1106/sequence_parity_cli.cc'),
                            '-o', str(self.executable)], check=True)
        return subprocess.check_output([str(self.executable), mode, *map(str, inputs)], text=True)

    def binary(self, name, values):
        path = self.path / name
        np.asarray(values, dtype='<f4').tofile(path)
        return path

    def test_spatial_views_soft_decode_even_median_and_ma7(self):
        rng = np.random.RandomState(20261116)
        frame = rng.rand(2, 60, 80).astype(np.float32)
        logits = rng.randn(10, 100, 140).astype(np.float32)
        frame_file = self.binary('frame.bin', frame)
        logits_file = self.binary('logits.bin', logits)
        actual = np.fromstring(self.run_cpp('postprocess', frame_file, logits_file), sep=' ')
        views = [(0, 0, False), (0, 0, True), (-2, 0, False), (-2, 0, True),
                 (2, 0, False), (2, 0, True), (0, -2, False), (0, -2, True),
                 (0, 2, False), (0, 2, True)]
        expected_views = np.stack([spatial_view(torch.from_numpy(frame), view).numpy()
                                   for view in views])
        cache = []
        for value, view in zip(logits, views):
            decoded = soft_expectation_decode(value[None, :, :80] / .65,
                                              value[None, :, 80:] / .65)[0]
            cache.append(undo_view(decoded.astype(np.float32) * [80., 60.], view))
        expected = np.median(cache, axis=0).astype(np.float32)
        self.assertEqual(len(actual), expected_views.size + 400)
        np.testing.assert_allclose(actual[:expected_views.size].reshape(expected_views.shape),
                                   expected_views, atol=1e-7, rtol=0)
        coordinates = actual[expected_views.size:expected_views.size + 200].reshape(100, 2)
        np.testing.assert_allclose(coordinates, expected, atol=2e-5, rtol=0)
        np.testing.assert_allclose(actual[-200:].reshape(100, 2),
                                   smooth_pixels(expected[None], 'ma7')[0], atol=2e-5, rtol=0)

    def test_subpixel_matcher_and_psr_match_python(self):
        torch.manual_seed(112)
        matcher = PaperMatcher(channels=3).eval()
        frames = torch.rand(2, 2, 60, 80)
        checkpoint, weights = self.path / 'matcher.pt', self.path / 'matcher.bin'
        torch.save({'config': {'ncc_channels': 3}, 'matcher_state': matcher.state_dict()}, checkpoint)
        export_matcher(checkpoint, weights)
        actual = np.fromstring(self.run_cpp('match', weights, self.binary('pair.bin', frames)), sep=' ')
        template = crop_rectangle(frames[:1], torch.tensor([[40.1, 30.2]]), (30, 25))
        search = crop_rectangle(frames[1:], torch.tensor([[40., 30.]]), (37, 32))
        with torch.no_grad():
            local, confidence, _, details = matcher.match_with_diagnostics(template, search, subpixel=True)
        expected = (torch.tensor([[40., 30.]]) + (local - .5) * 7)[0].numpy()
        np.testing.assert_allclose(actual[:2], expected, atol=2e-4, rtol=0)
        self.assertAlmostEqual(actual[2], float(confidence[0]), places=5)
        self.assertAlmostEqual(actual[3], float(details['psr'][0]), places=3)
        self.assertEqual(bool(actual[4]), bool(details['border_peak'][0]))

    def test_controlled_gates_kalman_and_low_global_invariance(self):
        # Fixture spans exact thresholds, accepted medium, rollback, reliability resets.
        for perturb in (False, True):
            actual = np.fromstring(self.run_cpp('gates', int(perturb)), sep=' ').reshape(-1, 3)
            kalman = BatchKalman(1, .05, .2, 32)
            kalman.initialize(np.array([[40., 30.]]))
            expected = [[40., 30., 3]]
            # dx, confidence, psr, border, global displacement from match-updated posterior
            fixtures = [(1., .8, 3., False, 0.), (3., .8, 3., False, .5),
                        (4., .8, 3., False, 2.), (5., .8, 3., False, 0.),
                        (0., .27, 3., False, 0.), (0., .8, 2.54, False, 0.),
                        (0., .8, 3., True, 0.)]
            for index, (dx, confidence, psr, border, sigma) in enumerate(fixtures):
                previous = kalman.x[:, :2].copy()
                prediction = kalman.predict()
                matched = np.array([[np.float32(prediction[0, 0]) + np.float32(dx),
                                     np.float32(prediction[0, 1])]], dtype=np.float32)
                difference = np.linalg.norm(prediction - matched, axis=-1)[0]
                tier = 3 if confidence < .28 or psr < 2.55 or border or difference >= 5 else 2 if difference >= 3 else 1
                if tier < 3:
                    kalman.update(matched)
                global_px = (kalman.x[:, :2] + [sigma, 0]).astype(np.float32)
                if tier == 1 and perturb:
                    global_px += 1000
                if tier == 2:
                    if np.linalg.norm(kalman.x[:, :2] - global_px) > 1:
                        kalman.reset(global_px, previous, np.array([True]))
                        result = kalman.x[0, :2].copy()
                    else:
                        kalman.update(global_px)
                        result = .75 * global_px[0] + .25 * kalman.x[0, :2]
                elif tier == 3:
                    kalman.reset(global_px, previous, np.array([True]))
                    result = kalman.x[0, :2].copy()
                else:
                    result = kalman.x[0, :2].copy()
                expected.append([*result, tier])
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)

    def test_full_record_tracker_matches_python_and_resets_at_record_boundary(self):
        torch.manual_seed(113)
        matcher = PaperMatcher(channels=2).eval()
        frames = torch.rand(1, 100, 2, 60, 80)
        globals_px = np.tile([40., 30.], (1, 100, 1)).astype(np.float32)
        checkpoint, weights = self.path / 'full.pt', self.path / 'full.bin'
        torch.save({'config': {'ncc_channels': 2}, 'matcher_state': matcher.state_dict()}, checkpoint)
        export_matcher(checkpoint, weights)
        tracker = OfflineBatchTracker(matcher, confidence_threshold=.28, min_psr=2.55,
                                      q_pos=.05, q_vel=.2, r_px=32., fusion_alpha=.75)
        expected, stats = tracker.track(frames, globals_px, return_trace=True)
        raw = np.fromstring(self.run_cpp('track', weights, self.binary('record.bin', frames),
                                        self.binary('global.bin', globals_px)), sep=' ').reshape(2, 100, 5)
        for actual in raw:
            np.testing.assert_allclose(actual[:, :2], expected[0], atol=4e-4, rtol=0)
            np.testing.assert_array_equal(actual[:, 2], stats['tiers'][0])
            np.testing.assert_allclose(actual[:, 3:], smooth_pixels(expected, 'ma7')[0], atol=4e-4, rtol=0)


if __name__ == '__main__':
    unittest.main()
