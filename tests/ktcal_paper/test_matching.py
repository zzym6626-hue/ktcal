import unittest
import numpy as np
import torch
from ktcal_paper.matching import PaperMatcher, crop_rectangle, response_to_pixel
from ktcal_paper.tracker import PixelKalman, gate


class MatchingTests(unittest.TestCase):
    def test_even_rectangle_center_and_response_coordinates(self):
        image = torch.arange(60*80, dtype=torch.float32).reshape(1, 1, 60, 80)
        patch = crop_rectangle(image, torch.tensor([[40., 30.]]), (30, 25))
        self.assertEqual(tuple(patch.shape), (1, 1, 25, 30))
        self.assertAlmostEqual(patch[0,0,12,14:16].mean().item(), 30*80+40, places=3)
        for local, expected in [([0.,0.], [37.5,27.5]), ([1.,1.], [42.5,32.5])]:
            actual = response_to_pixel(torch.tensor([local]), torch.tensor([[40.,30.]]), (30,25),(35,30))
            torch.testing.assert_close(actual, torch.tensor([expected]))

    def test_local_cosine_matches_independent_manual_calculation(self):
        matcher = PaperMatcher()
        matcher.extractor = torch.nn.Identity()
        torch.manual_seed(4)
        template = torch.randn(2,2,3,4,requires_grad=True)
        search = torch.randn(2,2,5,7,requires_grad=True)
        response = matcher.ncc_response(template,search)
        for b,y,x in [(0,0,0),(0,2,3),(1,1,2)]:
            a=template[b].flatten(); c=search[b,:,y:y+3,x:x+4].flatten()
            expected=torch.dot(a,c)/(a.norm()*c.norm())
            torch.testing.assert_close(response[b,y,x],expected)
        response.sum().backward()
        self.assertTrue(torch.isfinite(template.grad).all())
        self.assertGreater(search.grad.abs().sum().item(),0)

    def test_empty_events_remain_finite(self):
        response = PaperMatcher().ncc_response(torch.zeros(1,2,25,30),torch.zeros(1,2,30,35))
        self.assertTrue(torch.isfinite(response).all())
        self.assertTrue((response == 0).all())

    def test_matching_loss_trains_depthwise_and_pointwise_features(self):
        torch.manual_seed(42)
        matcher=PaperMatcher()
        frames=torch.rand(2,3,2,60,80)
        labels=torch.ones(2,3,4); labels[:,:,:2]=.5
        loss=matcher.training_loss(frames,labels)
        loss.backward()
        for layer in matcher.extractor[:2]:
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(),0)

    def test_pixel_covariances_and_gating_boundaries(self):
        kf=PixelKalman()
        kf.initialize([40,30])
        np.testing.assert_array_equal(kf.Q,np.diag([.1,.1,.5,.5]))
        np.testing.assert_array_equal(kf.R,np.diag([2.,2.]))
        kf.reset([41,31],previous_pos=[40,30])
        np.testing.assert_array_equal(kf.x,[41,31,1,1])
        np.testing.assert_array_equal(kf.P,np.diag([10.]*4))
        self.assertEqual([gate(e,.9) for e in [0,2.99,3,4.99,5]], [1,1,2,2,3])
        self.assertEqual(gate(0,.49),3)


if __name__=='__main__': unittest.main()
