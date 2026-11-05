import unittest
import torch
from ktcal_paper.model import PaperSTNet


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_export_preserves_trained_forward_prediction(self):
        model = PaperSTNet().eval()
        frames = torch.rand(2, 3, 2, 60, 80)
        with torch.no_grad():
            expected = model(frames, return_forward=True)[1]
            deployed = model.export_forward().eval()
            actual = deployed(frames)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(actual.shape, (2, 2))
        self.assertFalse(any('reverse' in key for key in deployed.state_dict()))
        self.assertEqual(deployed.head.in_features, 128)
        self.assertLess(sum(p.numel() for p in deployed.parameters()),
                        sum(p.numel() for p in model.parameters()))

    def test_both_coordinate_heads_and_directions_receive_gradient(self):
        model = PaperSTNet()
        bidir, forward = model(torch.rand(2, 3, 2, 60, 80), return_forward=True)
        target = torch.tensor([[.3, .4], [.6, .5]])
        loss = ((bidir-target)**2).mean() + ((forward-target)**2).mean()
        loss.backward()
        for parameter in [model.head.weight, model.forward_head.weight,
                          model.gru.weight_ih_l0, model.gru.weight_ih_l0_reverse]:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_ablation_variants_keep_single_coordinate_interface(self):
        for variant in ['no_temporal', 'no_gru', 'full_conv']:
            model = PaperSTNet(variant=variant).eval()
            with torch.no_grad():
                prediction = model.export_forward()(torch.rand(1, 2, 2, 60, 80))
            self.assertEqual(prediction.shape, (1, 2))
            self.assertTrue(torch.isfinite(prediction).all())

    def test_final_hidden_readout_trains_reverse_recurrence_and_preserves_forward_export(self):
        model=PaperSTNet(readout='final_hidden')
        frames=torch.rand(2,5,2,60,80)
        # Isolate the reverse branch. Taking out[:, -1] makes this recurrent
        # matrix's gradient zero because its preceding hidden state is zero.
        with torch.no_grad(): model.head.weight[:,:128].zero_()
        model(frames).square().mean().backward()
        self.assertGreater(model.gru.weight_hh_l0_reverse.grad.abs().sum().item(),0)
        model.eval()
        with torch.no_grad():
            forward=model(frames,return_forward=True)[1]
            exported=model.export_forward().eval()(frames)
        torch.testing.assert_close(exported,forward,atol=1e-6,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
