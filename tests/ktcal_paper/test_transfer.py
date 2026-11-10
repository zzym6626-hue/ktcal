import unittest
import torch

from ktcal_paper.model import PaperSTNet
from ktcal_paper.transfer import apply_sequence_state, map_sequence_state
from models.STNet import STNet


class TransferTests(unittest.TestCase):
    def test_sequence_weights_map_into_itd_and_gru_without_touching_the_head(self):
        torch.manual_seed(3)
        source = STNet()
        target = PaperSTNet()
        report = apply_sequence_state(target, source.state_dict())
        for index, block in enumerate(('block1', 'block2', 'block3')):
            spatial = getattr(source, block).spatial
            temporal = getattr(source, block).temporal
            layers = target.itd[index].layers
            torch.testing.assert_close(layers[0].weight, spatial[0].weight)
            torch.testing.assert_close(layers[0].bias, spatial[0].bias)
            torch.testing.assert_close(layers[1].running_mean, spatial[1].running_mean)
            torch.testing.assert_close(layers[3].weight, temporal[0].weight)
            torch.testing.assert_close(layers[4].running_var, temporal[1].running_var)
        for name in ('weight_ih_l0', 'weight_hh_l0_reverse', 'bias_ih_l0'):
            torch.testing.assert_close(getattr(target.gru, name), getattr(source.gru, name))
        torch.testing.assert_close(target.norm.weight, source.layernorm.weight)
        self.assertIn('head.weight', report['untouched_parameters'])
        self.assertGreater(report['mapped_parameters'], 40)

    def test_mapping_rejects_other_variants_and_shape_mismatches(self):
        with self.assertRaises(ValueError):
            map_sequence_state({}, variant='no_gru')
        state = {'gru.weight_ih_l0': torch.zeros(3, 7)}
        with self.assertRaises(ValueError):
            apply_sequence_state(PaperSTNet(), state)
        with self.assertRaises(ValueError):
            apply_sequence_state(PaperSTNet(), {})


if __name__ == '__main__':
    unittest.main()
