import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from ktcal_paper.evaluation_protocol import (
    hard_argmax_decode,
    official_postprocess,
    summarize_pixel_errors,
    trim_record_chunks,
)
from scripts.evaluate_official_3et_protocol import (
    build_parser,
    describe_temporal_protocol,
    load_checkpoint_state,
    load_event_statistics,
    resolve_preprocessing,
    validate_official_h5,
)


class DecoderAndMetricTests(unittest.TestCase):
    def test_hard_argmax_decode_returns_normalized_bin_coordinates(self):
        logits_x = np.zeros((1, 2, 80), dtype=np.float32)
        logits_y = np.zeros((1, 2, 60), dtype=np.float32)
        logits_x[0, 0, 12] = 4.0
        logits_x[0, 1, 79] = 5.0
        logits_y[0, 0, 7] = 6.0
        logits_y[0, 1, 59] = 7.0

        decoded = hard_argmax_decode(logits_x, logits_y)

        np.testing.assert_allclose(
            decoded,
            [[[12 / 80, 7 / 60], [79 / 80, 59 / 60]]],
        )

    def test_pixel_summary_scales_normalized_coordinates_by_80_and_60(self):
        labels = np.array([[[0.0, 0.0], [0.5, 0.5]]], dtype=np.float32)
        predictions = np.array(
            [[[3 / 80, 4 / 60], [0.5, 0.5]]],
            dtype=np.float32,
        )

        summary = summarize_pixel_errors(predictions, labels)

        self.assertEqual(summary["frames"], 2)
        self.assertAlmostEqual(summary["mean_error_px"], 2.5, places=6)
        self.assertAlmostEqual(summary["median_error_px"], 2.5, places=6)

    def test_pixel_tolerance_is_strictly_less_than_three_pixels(self):
        labels = np.zeros((1, 2, 2), dtype=np.float32)
        predictions = np.array(
            [[[2 / 80, 0.0], [3 / 80, 0.0]]],
            dtype=np.float32,
        )

        summary = summarize_pixel_errors(
            predictions,
            labels,
            pixel_tolerances=(3,),
        )

        self.assertEqual(summary["p3"], 0.5)


class RecordBoundaryTests(unittest.TestCase):
    def test_all_full_chunks_are_flattened_in_order(self):
        chunks = np.arange(8).reshape(2, 4, 1)

        trimmed = trim_record_chunks(chunks, [4, 4], chunk_size=4)

        np.testing.assert_array_equal(trimmed[:, 0], np.arange(8))

    def test_partial_final_chunk_drops_only_its_padding(self):
        chunks = np.array(
            [[[0], [1], [2], [3]], [[4], [5], [99], [99]]],
        )

        trimmed = trim_record_chunks(chunks, [6], chunk_size=4)

        np.testing.assert_array_equal(trimmed[:, 0], [0, 1, 2, 3, 4, 5])

    def test_adjacent_recordings_do_not_leak_padding_across_boundary(self):
        chunks = np.array(
            [[[10], [11], [12], [999]], [[20], [21], [22], [888]]],
        )

        trimmed = trim_record_chunks(chunks, [3, 3], chunk_size=4)

        np.testing.assert_array_equal(trimmed[:, 0], [10, 11, 12, 20, 21, 22])

    def test_inconsistent_chunk_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "chunk count"):
            trim_record_chunks(np.zeros((1, 4, 2)), [3, 3], chunk_size=4)


class OfficialPostprocessTests(unittest.TestCase):
    def test_low_confidence_prediction_uses_previous_valid_prediction(self):
        predictions = np.array([[0.0, 0.0], [10.0, 1.0], [20.0, 2.0]])

        processed = official_postprocess(
            predictions,
            confidence=[1.0, 0.1, 1.0],
            num_events_y=[1.0, 1.0, 1.0],
            recording_lengths=[3],
        )

        np.testing.assert_array_equal(
            processed,
            [[0.0, 0.0], [0.0, 0.0], [20.0, 2.0]],
        )

    def test_low_event_mask_is_dilated_twice(self):
        predictions = np.stack([np.arange(7), np.zeros(7)], axis=1)

        processed = official_postprocess(
            predictions,
            confidence=np.ones(7),
            num_events_y=[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            recording_lengths=[7],
        )

        np.testing.assert_array_equal(processed[:, 0], [0, 0, 0, 0, 0, 0, 6])

    def test_masked_first_frame_does_not_wrap_to_last_prediction(self):
        predictions = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

        processed = official_postprocess(
            predictions,
            confidence=[0.0, 1.0, 1.0],
            num_events_y=[1.0, 1.0, 1.0],
            recording_lengths=[3],
        )

        np.testing.assert_array_equal(processed[:, 0], [1.0, 2.0, 3.0])

    def test_replacement_and_dilation_do_not_cross_record_boundaries(self):
        predictions = np.stack([np.array([0, 1, 2, 10, 11, 12]), np.zeros(6)], axis=1)

        processed = official_postprocess(
            predictions,
            confidence=[1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            num_events_y=[1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            recording_lengths=[3, 3],
        )

        self.assertEqual(processed[3, 0], 10)
        self.assertEqual(processed[4, 0], 11)


class OfficialProtocolCliTests(unittest.TestCase):
    def test_stnet_preprocessing_uses_checkpoint_divisor_with_legacy_default(self):
        raw = resolve_preprocessing('stnet', {'input_divisor': 1.0})
        legacy = resolve_preprocessing('stnet', {})

        self.assertEqual(raw, (1.0, 'raw_event_counts'))
        self.assertEqual(legacy, (15.0, 'event_counts_divided_by_15'))
        with self.assertRaisesRegex(ValueError, 'input divisor'):
            resolve_preprocessing('stnet', {'input_divisor': 0})

    def test_temporal_protocol_reports_unidirectional_stnet_truthfully(self):
        protocol = describe_temporal_protocol(
            "stnet",
            {"window": 16, "unidirectional": True},
        )

        self.assertFalse(protocol["bidirectional"])

    def test_official_singular_label_dataset_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "official.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "frames", shape=(50, 200, 2, 60, 80), dtype=np.uint8
                )
                handle.create_dataset("label", shape=(50, 200, 3), dtype=np.float32)

            result = validate_official_h5(path)

        self.assertEqual(result["label_key"], "label")

    def test_invalid_h5_shape_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.h5"
            with h5py.File(path, "w") as handle:
                handle["frames"] = np.zeros((1, 16, 2, 60, 80), dtype=np.uint8)
                handle["labels"] = np.zeros((1, 16, 2), dtype=np.float32)

            with self.assertRaisesRegex(ValueError, "official frames shape"):
                validate_official_h5(path)

    def test_stnet_checkpoint_mismatch_fails_loudly(self):
        model = torch.nn.Linear(2, 2)
        incomplete = {"weight": torch.zeros_like(model.weight)}

        with self.assertRaisesRegex(ValueError, "checkpoint/model mismatch"):
            load_checkpoint_state(model, incomplete, architecture="stnet")

    def test_missing_event_statistics_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "num_events.txt"

            with self.assertRaisesRegex(FileNotFoundError, "event statistics"):
                load_event_statistics(missing, expected_frames=3)

    def test_unsupported_architecture_is_rejected_by_parser(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--h5",
                    "test.h5",
                    "--checkpoint",
                    "best.pt",
                    "--architecture",
                    "transformer",
                    "--output",
                    "report.json",
                ]
            )

        self.assertIn("invalid choice", error.getvalue())


if __name__ == "__main__":
    unittest.main()
