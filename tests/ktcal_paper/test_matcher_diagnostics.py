import unittest

import numpy as np
import torch

from ktcal_paper.matcher_diagnostics import (
    response_map,
    summarize_matches,
)
from scripts.diagnose_3et_template_matching import parse_search_sizes


class MatcherDiagnosticTests(unittest.TestCase):
    def test_zero_mean_response_is_invariant_to_patch_offset(self):
        template = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        search = template + 7.0

        response = response_map(template, search, metric="zero_mean_ncc")

        torch.testing.assert_close(response, torch.ones(1, 1, 1))

    def test_summary_reports_geometry_and_matching_failures_separately(self):
        summary = summarize_matches(
            errors=np.array([0.5, 2.0, 6.0]),
            target_offsets=np.array([[1.0, 1.0], [3.0, 0.0], [0.0, 4.0]]),
            border=np.array([False, True, False]),
            confidence=np.array([0.8, 0.7, 0.4]),
            psr=np.array([5.0, 2.0, 1.0]),
        )

        self.assertEqual(summary["frames"], 3)
        self.assertAlmostEqual(summary["geometry_capture_rate"], 1 / 3)
        self.assertAlmostEqual(summary["border_peak_rate"], 1 / 3)
        self.assertAlmostEqual(summary["low_confidence_rate"], 1 / 3)
        self.assertAlmostEqual(summary["mean_error_inside_capture_px"], 0.5)
        self.assertAlmostEqual(summary["mean_error_outside_capture_px"], 4.0)

    def test_summary_accepts_the_actual_search_capture_span(self):
        summary = summarize_matches(
            errors=np.array([1.0, 3.0]),
            target_offsets=np.array([[4.0, 0.0], [6.0, 0.0]]),
            border=np.array([False, True]),
            confidence=np.array([0.8, 0.7]),
            psr=np.array([5.0, 2.0]),
            capture_half_span_px=(5.0, 5.0),
        )

        self.assertAlmostEqual(summary["geometry_capture_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_error_inside_capture_px"], 1.0)
        self.assertAlmostEqual(summary["mean_error_outside_capture_px"], 3.0)

    def test_search_size_parser_preserves_width_height_order(self):
        self.assertEqual(
            parse_search_sizes("35x30,40x35,42x37"),
            [(35, 30), (40, 35), (42, 37)],
        )

    def test_search_size_parser_rejects_a_patch_smaller_than_template(self):
        with self.assertRaisesRegex(ValueError, "smaller than template"):
            parse_search_sizes("29x25")


if __name__ == "__main__":
    unittest.main()
