"""Reject invalid or inconsistent release evidence before calling it verified."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np


class SequenceVerificationTests(unittest.TestCase):
    def verifier(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.verify_3et'),
                             'The independent release verifier is missing')
        from scripts import verify_3et
        return verify_3et

    def test_independent_pixel_score_uses_strict_thresholds_and_record_filter(self):
        module = self.verifier()
        raw = np.zeros((2, 7, 2))
        raw[1, :, 0] = 5.
        np.testing.assert_allclose(module.filter_records(raw, 'ma7'), raw, atol=1e-6)
        metric = module.score_pixels(raw, np.zeros_like(raw))
        self.assertEqual(metric['frames'], 14)
        self.assertEqual(metric['p3'], .5)
        self.assertEqual(metric['p5'], .5)
        self.assertEqual(metric['p10'], 1.)
        self.assertEqual(metric['mean_error_px'], 2.5)

    def test_reproduction_must_match_counts_and_forbid_low_tier_fusion(self):
        module = self.verifier()
        metrics = dict(frames=100, p3=.91, p5=.96, p10=.99, mean_error_px=1.6)
        counts = dict(frames=100, tier1=60, tier2=1, tier3=39, small_fusion_calls=0)
        module.validate_reproduction(metrics, metrics, counts, counts)
        for bad in (dict(counts, tier1=59, tier3=40), dict(counts, small_fusion_calls=1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                module.validate_reproduction(metrics, metrics, counts, bad)
        with self.assertRaises(ValueError):
            module.validate_reproduction(metrics, dict(metrics, p3=float('nan')), counts, counts)

    def test_wrong_hdf5_sha_rejected_before_loading_frames_or_model(self):
        module = self.verifier()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5 = root / 'test_aug.h5'
            with h5py.File(h5, 'w') as handle:
                handle.create_dataset('frames', shape=(93, 100, 2, 60, 80), dtype='f4')
                handle.create_dataset('label', shape=(93, 100, 3), dtype='f4')
            expected = root / 'expected.json'
            expected.write_text(json.dumps(dict(config=dict(h5=str(h5)), h5_sha256='bad')))
            with self.assertRaisesRegex(ValueError, 'HDF5 SHA mismatch'):
                module.main(['--expected-report', str(expected), '--output', str(root / 'verify.json')])
            self.assertFalse((root / 'verify.json').exists())
