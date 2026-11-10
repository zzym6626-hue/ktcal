import json
import tempfile
import unittest
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

from ktcal_paper.matcher_finetune import (
    _h5_description,
    checkpoint_with_matcher,
    precompute_stnet_predictions,
    train_matcher,
    validate_prediction_shape,
    weighted_validation_error,
)
from ktcal_paper.matching import PaperMatcher
from scripts.finetune_3et_matcher import build_parser


class MatcherFinetuneUtilityTests(unittest.TestCase):
    def test_cli_defaults_to_recommended_matcher_only_recipe(self):
        arguments = build_parser().parse_args(
            [
                "--train-h5", "train_aug.h5",
                "--validation-h5", "test_aug.h5",
                "--stnet-checkpoint", "stnet.pt",
                "--source-checkpoint", "ktcal.pt",
                "--output", "run",
            ]
        )

        self.assertEqual(arguments.epochs, 30)
        self.assertEqual(arguments.patience, 6)
        self.assertEqual(arguments.validation_gaps, "1,4,8")
        self.assertEqual(arguments.seed, 42)

    def test_stnet_predictions_are_cached_as_pixel_coordinates(self):
        class FakeSTNet(torch.nn.Module):
            def forward(self, frames):
                batch, length = frames.shape[:2]
                x = torch.zeros(batch, length, 80, device=frames.device)
                y = torch.zeros(batch, length, 60, device=frames.device)
                x[..., 12] = 1.0
                y[..., 7] = 1.0
                return x, y

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h5_path = root / "train_aug.h5"
            with h5py.File(h5_path, "w") as handle:
                handle["frames"] = np.zeros((3, 5, 2, 60, 80), dtype=np.uint8)
                handle["labels"] = np.zeros((3, 5, 3), dtype=np.float32)

            cache = precompute_stnet_predictions(
                h5_path,
                FakeSTNet(),
                root / "predictions.npy",
                batch_size=2,
                device=torch.device("cpu"),
            )

            predictions = np.load(cache)
            self.assertEqual(predictions.shape, (3, 5, 2))
            np.testing.assert_allclose(predictions[..., 0], 12.0)
            np.testing.assert_allclose(predictions[..., 1], 7.0)

    def test_stnet_prediction_cache_respects_checkpoint_input_divisor(self):
        class CaptureInput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = None

            def forward(self, frames):
                self.value = frames.mean().item()
                batch, length = frames.shape[:2]
                return (torch.zeros(batch, length, 80, device=frames.device),
                        torch.zeros(batch, length, 60, device=frames.device))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with h5py.File(root / 'input.h5', 'w') as handle:
                handle['frames'] = np.full((1, 2, 2, 60, 80), 15, dtype=np.uint8)
                handle['labels'] = np.zeros((1, 2, 3), dtype=np.float32)
            model = CaptureInput()
            precompute_stnet_predictions(root / 'input.h5', model,
                                          root / 'predictions.npy', batch_size=1,
                                          device=torch.device('cpu'),
                                          input_divisor=15.)
            self.assertEqual(model.value, 1.)

    def test_validation_error_is_weighted_by_scored_frames(self):
        rows = [
            {"frames": 10, "mean_error_px": 1.0},
            {"frames": 30, "mean_error_px": 3.0},
        ]

        self.assertAlmostEqual(weighted_validation_error(rows), 2.5)

    def test_checkpoint_replaces_only_matcher_state(self):
        source = {
            "model_state": {
                "weight": torch.tensor([1.0, 2.0]),
                "counter": torch.tensor(7),
            },
            "matcher_state": {"old": torch.tensor([-1.0])},
            "config": {"ncc_channels": 16, "temperature": 0.1},
        }
        original_model = {
            name: value.clone() for name, value in source["model_state"].items()
        }
        matcher = PaperMatcher(channels=4)
        with torch.no_grad():
            for parameter in matcher.parameters():
                parameter.fill_(0.25)

        result = checkpoint_with_matcher(
            source,
            matcher,
            {"selected_epoch": 3, "validation_error_px": 1.75},
        )

        self.assertEqual(set(result["matcher_state"]), set(matcher.state_dict()))
        for name, expected in matcher.state_dict().items():
            torch.testing.assert_close(result["matcher_state"][name], expected)
        for name, expected in original_model.items():
            torch.testing.assert_close(result["model_state"][name], expected, rtol=0, atol=0)
            torch.testing.assert_close(source["model_state"][name], expected, rtol=0, atol=0)
        self.assertEqual(
            result["matcher_finetune"],
            {"selected_epoch": 3, "validation_error_px": 1.75},
        )
        self.assertNotIn("old", result["matcher_state"])

    def test_checkpoint_requires_full_ktcal_state(self):
        with self.assertRaisesRegex(ValueError, "model_state"):
            checkpoint_with_matcher(
                {"matcher_state": {}}, PaperMatcher(), {"selected_epoch": 1}
            )

    def test_prediction_cache_shape_must_match_h5_sequences_and_timesteps(self):
        predictions = np.zeros((3, 5, 2), dtype=np.float32)
        validated = validate_prediction_shape(predictions, 3, 5, "train")
        self.assertIs(validated, predictions)

        with self.assertRaisesRegex(ValueError, "train prediction cache shape"):
            validate_prediction_shape(predictions, 4, 5, "train")
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = predictions.copy()
            invalid[0, 0, 0] = np.nan
            validate_prediction_shape(invalid, 3, 5, "train")

    def test_renamed_official_holdout_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "renamed_validation.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "frames", shape=(50, 200, 2, 60, 80), dtype=np.uint8
                )
                handle.create_dataset("labels", shape=(50, 200, 3), dtype=np.float32)

            with self.assertRaisesRegex(ValueError, "final holdout shape"):
                _h5_description(path, "validation")

    def test_empty_capture_batches_do_not_apply_optimizer_weight_decay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_h5 = root / "train_aug.h5"
            val_h5 = root / "test_aug.h5"
            frames = np.zeros((2, 3, 2, 60, 80), dtype=np.uint8)
            labels = np.tile(
                np.array([[[0.9, 0.9, 1.0]]], dtype=np.float32), (2, 3, 1)
            )
            for path in (train_h5, val_h5):
                with h5py.File(path, "w") as handle:
                    handle["frames"] = frames
                    handle["labels"] = labels
            predictions = np.tile(
                np.array([[[1.0, 1.0]]], dtype=np.float32), (2, 3, 1)
            )
            train_predictions = root / "train.npy"
            val_predictions = root / "val.npy"
            np.save(train_predictions, predictions)
            np.save(val_predictions, predictions)
            matcher = PaperMatcher(channels=4)
            source = {
                "model_state": {"weight": torch.tensor([1.0])},
                "matcher_state": matcher.state_dict(),
                "config": {"ncc_channels": 4, "temperature": 0.1},
            }
            source_path = root / "source.pt"
            torch.save(source, source_path)

            best = train_matcher(
                train_h5, val_h5, source_path, root / "run",
                train_predictions, val_predictions, epochs=1, patience=1,
                batch_size=2, sequence_batch_size=1, samples_per_sequence=1,
                lr=1e-3, weight_decay=0.1, device="cpu", threads=2,
                validation_gaps=(1,)
            )

            selected = torch.load(best, map_location="cpu", weights_only=False)
            for name, expected in source["matcher_state"].items():
                torch.testing.assert_close(
                    selected["matcher_state"][name], expected, rtol=0, atol=0
                )

    def test_one_epoch_training_writes_matcher_only_checkpoint_and_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_h5 = root / "train_aug.h5"
            val_h5 = root / "test_aug.h5"
            rng = np.random.default_rng(9)
            labels = np.tile(
                np.array([[[0.5, 0.5, 1.0]]], dtype=np.float32), (3, 5, 1)
            )
            for path in (train_h5, val_h5):
                with h5py.File(path, "w") as handle:
                    handle["frames"] = rng.integers(
                        0, 16, (3, 5, 2, 60, 80), dtype=np.uint8
                    )
                    handle["labels"] = labels
            predictions = np.tile(
                np.array([[[40.0, 30.0]]], dtype=np.float32), (3, 5, 1)
            )
            train_predictions = root / "train_predictions.npy"
            val_predictions = root / "val_predictions.npy"
            np.save(train_predictions, predictions)
            np.save(val_predictions, predictions)
            matcher = PaperMatcher(channels=4)
            source = {
                "model_state": {"weight": torch.tensor([1.0, 2.0])},
                "matcher_state": matcher.state_dict(),
                "config": {"ncc_channels": 4, "temperature": 0.1},
            }
            source_path = root / "source.pt"
            torch.save(source, source_path)

            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                best = train_matcher(
                    train_h5,
                    val_h5,
                    source_path,
                    root / "run",
                    train_predictions,
                    val_predictions,
                    epochs=1,
                    patience=1,
                    batch_size=4,
                    sequence_batch_size=2,
                    samples_per_sequence=2,
                    lr=1e-3,
                    device="cpu",
                    threads=2,
                    seed=42,
                    validation_gaps=(1, 4, 8),
                )

            selected = torch.load(best, map_location="cpu", weights_only=False)
            torch.testing.assert_close(
                selected["model_state"]["weight"],
                source["model_state"]["weight"],
                rtol=0,
                atol=0,
            )
            self.assertEqual(selected["matcher_finetune"]["selected_epoch"], 1)
            self.assertFalse(selected["matcher_finetune"]["final_holdout_used"])
            history = json.loads((root / "run" / "history.json").read_text())
            self.assertEqual(len(history), 1)
            self.assertEqual([row["gap"] for row in history[0]["validation"]], [1, 4])
            self.assertGreater(history[0]["training_valid_samples"], 0)

            with self.assertRaises(FileExistsError):
                train_matcher(
                    train_h5,
                    val_h5,
                    source_path,
                    root / "run",
                    train_predictions,
                    val_predictions,
                    epochs=1,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
