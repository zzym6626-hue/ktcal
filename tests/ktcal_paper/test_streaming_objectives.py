import importlib.util
import unittest

import torch


class StreamingObjectiveTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('ktcal_paper.streaming_objectives'),
                             'Soft decoder difficulty objectives are missing')
        from ktcal_paper import streaming_objectives
        return streaming_objectives

    def fixture(self, length=10):
        x = torch.full((2, length, 80), -30., requires_grad=True)
        y = torch.full((2, length, 60), -30., requires_grad=True)
        with torch.no_grad():
            x[..., 40] = 30.
            y[..., 30] = 30.
        labels = torch.tensor([.5, .5]).expand(2, length, 2).clone()
        return x, y, labels, torch.ones(2, length, dtype=torch.bool)

    def test_soft_persistent_cold_invalid_and_record_boundaries(self):
        module = self.module()
        x, y, labels, valid = self.fixture()
        labels[0, 2:7, 0] -= 4 / 80.
        labels[0, 8:, 0] -= 4 / 80.
        labels[1, :3, 0] -= 4 / 80.
        valid[0, 5] = False  # splits the five-frame run
        valid[1, 1] = False
        weights = module.streaming_frame_weights(x, y, labels, valid,
            persistent_boost=2., min_run=5, cold_boost=1., cold_frames=2)
        expected = valid.float()
        expected[:, :2] *= 2
        torch.testing.assert_close(weights, expected)
        self.assertFalse(weights.requires_grad)
        valid[0, 5] = True
        weights = module.streaming_frame_weights(x, y, labels, valid,
            persistent_boost=2., min_run=5, cold_boost=1., cold_frames=2)
        expected = valid.float()
        expected[:, :2] *= 2
        expected[0, 2:7] = 3
        torch.testing.assert_close(weights, expected)

    def test_persistent_errors_use_soft_decoder_instead_of_argmax(self):
        x, y, labels, valid = self.fixture(5)
        with torch.no_grad():
            x.fill_(-30.)
            x[..., 40] = 2.
            x[..., 48] = 1.9
        self.assertTrue(bool((x.argmax(-1) == 40).all()))
        weights = self.module().streaming_frame_weights(x, y, labels, valid,
            temperature=1., persistent_boost=2., min_run=5)
        torch.testing.assert_close(weights, torch.full_like(valid.float(), 3.))
        sharpened = self.module().streaming_frame_weights(x, y, labels, valid,
            temperature=.1, persistent_boost=2., min_run=5)
        torch.testing.assert_close(sharpened, torch.ones_like(valid.float()))

    def test_weighted_huber_masks_invalid_targets_and_gradients(self):
        module = self.module()
        x = torch.zeros(1, 3, 80, requires_grad=True)
        y = torch.zeros(1, 3, 60, requires_grad=True)
        labels = torch.tensor([[[.5, .5], [.6, .5], [float('nan'), float('nan')]]])
        valid = torch.tensor([[True, True, False]])
        weights = torch.tensor([[1., 3., 0.]])
        loss = module.weighted_coordinate_huber(x, y, labels, valid, weights=weights)
        prediction = torch.tensor([39.5, 29.5])
        per_frame = torch.nn.functional.smooth_l1_loss(
            prediction.expand(2, -1), labels[0, :2] * torch.tensor([80., 60.]),
            reduction='none').mean(-1)
        torch.testing.assert_close(loss, (per_frame * weights[0, :2]).sum() / 4)
        loss.backward()
        self.assertGreater(float(x.grad[0, :2].abs().sum()), 0)
        self.assertEqual(float(x.grad[0, 2].abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_default_huber_matches_existing_pixel_objective(self):
        from ktcal_paper.stnet_sequence import masked_pixel_loss
        x, y, labels, valid = self.fixture(5)
        labels[..., 0] += .02
        expected = masked_pixel_loss(x, y, labels, valid)
        actual = self.module().weighted_coordinate_huber(x, y, labels, valid)
        torch.testing.assert_close(actual, expected)
        weights = self.module().streaming_frame_weights(x, y, labels, valid)
        torch.testing.assert_close(weights, valid.float())

    def test_empty_trust_mask_retains_zero_gradient_graph(self):
        x, y, labels, valid = self.fixture(5)
        labels.fill_(float('nan'))
        valid.fill_(False)
        loss = self.module().weighted_coordinate_huber(x, y, labels, valid)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(x.grad.abs().sum()), 0.)

    def test_configuration_and_selected_weights_are_validated(self):
        module = self.module()
        x, y, labels, valid = self.fixture(5)
        for arguments in ({'temperature': 0.}, {'temperature': float('nan')},
                          {'persistent_boost': -1.}, {'cold_boost': float('inf')},
                          {'min_run': True}, {'min_run': 0}, {'cold_frames': 1.5}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                module.streaming_frame_weights(x, y, labels, valid, **arguments)
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.weighted_coordinate_huber(x, y, labels, valid,
                    weights=torch.full_like(valid.float(), value))
