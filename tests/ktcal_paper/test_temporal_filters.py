import unittest

from scripts.evaluate_stnet_temporal_filters import build_filter_variants


class TemporalFilterTests(unittest.TestCase):
    def test_longer_moving_average_windows_are_available(self):
        variants = build_filter_variants()
        for name in ('ma11', 'ma13', 'ma15'):
            self.assertIn(name, variants)
            self.assertTrue(callable(variants[name]))


if __name__ == '__main__':
    unittest.main()
