"""Tests for an isolated export spike, not a replacement board backend."""
import importlib.util
import unittest
from types import SimpleNamespace

import torch
from models.STNet import STNet


class SplitProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('split_probe'),
                             'equivalent split probe is not implemented')
        import split_probe
        return split_probe

    def model(self):
        torch.manual_seed(719)
        model = STNet(SimpleNamespace(stnet_readout='attention')).eval()
        for block in (model.block1, model.block2, model.block3):
            for bn in (block.spatial[1], block.temporal[1]):
                bn.running_mean.copy_(torch.randn_like(bn.running_mean))
                bn.running_var.copy_(torch.rand_like(bn.running_var) + .5)
        return model

    def test_each_spatial_stage_matches_original_nontrivial_bn(self):
        # Catches wrong kernel squeeze, padding, bias, BN stats or NCHW layout.
        module, model = self.module(), self.model()
        for block, shape in zip((model.block1, model.block2, model.block3),
                                ((2, 60, 80), (32, 20, 26), (64, 6, 8))):
            spatial = module.Spatial2D(block).eval()
            value = torch.randn(1, *shape)
            with torch.inference_mode():
                expected = block.spatial(value.unsqueeze(2)).squeeze(2)
                actual = spatial(value)
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
            self.assertTrue(all(layer.weight.ndim == 4 for layer in spatial.modules()
                                if isinstance(layer, torch.nn.Conv2d)))

    def test_full_100_frame_split_keeps_temporal_edges_and_applies_divisor_once(self):
        # Catches slicing records into short windows, wrong boundary padding,
        # doubled divisor, reversed GRU or missing attention/FC/pooling.
        module, model = self.module(), self.model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        split = module.SplitSTNet(model, 7.)
        raw = torch.randn(1, 100, 2, 60, 80) * .2
        raw[:, 0] += 3.
        raw[:, -1] -= 2.
        with torch.inference_mode():
            reference = model(raw / 7.)
            actual = split(raw)
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertEqual(actual[0].shape, (1, 100, 80))
        self.assertEqual(actual[1].shape, (1, 100, 60))

    def test_split_rejects_non_100_frame_raw_contract(self):
        module, model = self.module(), self.model()
        split = module.SplitSTNet(model, 1.)
        for value in (torch.zeros(1, 16, 2, 60, 80),
                      torch.zeros(1, 100, 2, 60, 79)):
            with self.assertRaises(ValueError):
                split(value)
        for divisor in (0., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                module.SplitSTNet(model, divisor)

    def test_spatial_probe_refuses_temporal_or_depthwise_variant(self):
        module = self.module()
        model = self.model()
        # Construct an unsupported grouped kernel directly: the release model
        # deliberately does not include the experimental variant constructor.
        model.block1.temporal[0] = torch.nn.Conv3d(
            32, 32, (3, 1, 1), padding=(1, 0, 0), groups=32, bias=False)
        with self.assertRaises(ValueError):
            module.SplitSTNet(model, 1.)


if __name__ == '__main__':
    unittest.main()
