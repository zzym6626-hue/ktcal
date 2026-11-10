import unittest
from unittest import mock
import numpy as np
import torch
from ktcal_paper.matching import (COARSE_SEARCH, COARSE_TEMPLATE, SEARCH,
    PaperMatcher, crop_rectangle, downsample, response_to_pixel)
from ktcal_paper.tracker import PaperTracker, PixelKalman, gate


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

    def test_predicted_center_loss_trains_only_geometry_captured_samples(self):
        torch.manual_seed(17)
        matcher=PaperMatcher()
        previous=torch.rand(2,2,60,80)
        current=torch.rand(2,2,60,80)
        template_centers=torch.tensor([[39.,30.],[40.,30.]])
        search_centers=torch.tensor([[40.,30.],[40.,30.]])
        targets=torch.tensor([[41.,30.],[46.,30.]])

        loss,count=matcher.predicted_center_loss(
            previous,current,template_centers,search_centers,targets)
        first_loss,first_count=matcher.predicted_center_loss(
            previous[:1],current[:1],template_centers[:1],
            search_centers[:1],targets[:1])

        self.assertEqual(count,1)
        self.assertEqual(first_count,1)
        torch.testing.assert_close(loss,first_loss)
        loss.backward()
        for layer in matcher.extractor[:2]:
            self.assertTrue(torch.isfinite(layer.weight.grad).all())

    def test_predicted_center_loss_empty_capture_is_differentiable_zero(self):
        matcher=PaperMatcher()
        frames=torch.rand(2,2,60,80)
        centers=torch.tensor([[40.,30.],[40.,30.]])
        targets=torch.tensor([[46.,30.],[40.,36.]])

        loss,count=matcher.predicted_center_loss(
            frames,frames,centers,centers,targets)

        self.assertEqual(count,0)
        self.assertEqual(float(loss),0.)
        loss.backward()
        for parameter in matcher.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(float(parameter.grad.abs().sum()),0.)

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

    def test_kalman_covariances_are_tunable_without_changing_defaults(self):
        tuned=PixelKalman(q_pos=.4,q_vel=2.,r_px=8.)
        np.testing.assert_array_equal(tuned.Q,np.diag([.4,.4,2.,2.]))
        np.testing.assert_array_equal(tuned.R,np.diag([8.,8.]))
        with self.assertRaises(ValueError): PixelKalman(r_px=0)

    def test_border_peak_and_low_psr_escalate_instead_of_updating_kalman(self):
        response=torch.zeros(1,6,6)
        response[0,0,2]=1.
        _,_,details=PaperMatcher().locate_diagnostics(response)
        self.assertTrue(bool(details['border_peak'][0]))
        response=torch.zeros(1,6,6); response[0,3,3]=1.
        _,_,details=PaperMatcher().locate_diagnostics(response)
        self.assertFalse(bool(details['border_peak'][0]))
        self.assertEqual(gate(0.,.9,border_peak=True),3)
        self.assertEqual(gate(0.,.9,border_peak=True,escalate_tier=2),2)
        self.assertEqual(gate(0.,.9,psr=.4,min_psr=2.),3)
        self.assertEqual(gate(0.,.9,psr=5.,min_psr=2.),1)
        self.assertEqual(gate(0.,.9),1)

    def test_coarse_proposal_recovers_motion_outside_the_fine_capture_range(self):
        yy,xx=torch.meshgrid(torch.arange(60.),torch.arange(80.),indexing='ij')
        blob=torch.exp(-((xx-40)**2+(yy-30)**2)/(2*6.**2))
        base=blob[None,None].repeat(1,2,1,1)
        shifted=torch.roll(base,shifts=(6,),dims=(3,))
        center=torch.tensor([[40.,30.]])
        tracker=PaperTracker(torch.nn.Identity(),PaperMatcher(),capture='coarse')
        template,coarse_template=tracker._template_at(base,center[0])
        match,_,details=tracker._match(template,coarse_template,shifted,
                                      downsample(shifted),center)
        self.assertLess(abs(match[0]-46.),1.2)
        self.assertLess(abs(match[1]-30.),1.5)
        self.assertFalse(details['border_peak'])
        fine=PaperTracker(torch.nn.Identity(),PaperMatcher(),capture='fine')
        template,_=fine._template_at(base,center[0])
        clamped,_,_=fine._match(template,None,shifted,None,center)
        self.assertGreater(abs(clamped[0]-46.),3.)
        self.assertTrue(details['psr']>0)

    def test_subpixel_peak_recovers_offset_the_argmax_lattice_misses(self):
        response=torch.zeros(1,6,6)
        # A parabola whose true vertex sits 0.0 and 0.3 cells away from the
        # argmax cell (2,2): argmax can only report whole cells here.
        yy,xx=torch.meshgrid(torch.arange(6.),torch.arange(6.),indexing='ij')
        response[0]=torch.exp(-((xx-2.3)**2+(yy-2.0)**2)/(2*.8**2))
        matcher=PaperMatcher()
        hard,_=matcher.locate(response)
        refined,_=matcher.locate(response,subpixel=True)
        self.assertAlmostEqual(float(hard[0,0])*5,2.,places=6)
        # A parabola fit on a Gaussian is biased toward the peak cell, so the
        # contract is "recovers most of the sub-cell offset", not exactness.
        self.assertGreater(float(refined[0,0])*5,2.15)
        self.assertLess(float(refined[0,0])*5,2.3)
        self.assertAlmostEqual(float(hard[0,1])*5,2.,places=6)
        self.assertAlmostEqual(float(refined[0,1])*5,2.,places=6)

    def test_subpixel_leaves_border_peaks_alone_and_stays_inside_one_cell(self):
        border=torch.zeros(1,6,6); border[0,0,2]=1.
        local,_=PaperMatcher().locate(border,subpixel=True)
        torch.testing.assert_close(local,torch.tensor([[2/5,0.]]))
        slope=torch.zeros(1,6,6); slope[0,2,2]=1.; slope[0,2,3]=2.
        local,_=PaperMatcher().locate(slope,subpixel=True)
        self.assertLessEqual(abs(float(local[0,0])*5-3.),.5)
        self.assertAlmostEqual(float(local[0,1])*5,2.,places=6)

    def test_subpixel_handles_mixed_border_and_interior_peaks_in_one_batch(self):
        response=torch.zeros(2,6,6)
        response[0,0,2]=1.
        response[1,2,2]=1.; response[1,2,1]=.4; response[1,2,3]=.8

        local,_=PaperMatcher().locate(response,subpixel=True)

        torch.testing.assert_close(local[0],torch.tensor([2/5,0.]))
        self.assertGreater(float(local[1,0]),2/5)
        self.assertAlmostEqual(float(local[1,1]),2/5,places=6)

    def test_subpixel_does_not_read_tensor_predicates_for_batched_or_thin_maps(self):
        response = torch.zeros(3, 6, 6)
        response[0, 0, 2] = 1.
        response[1, 3, 3] = 1.
        response[1, 2, 3] = .4
        response[1, 4, 3] = .8
        response[2, 2, 0] = 1.
        thin = torch.tensor([[[.3], [1.], [.8]]])
        with mock.patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('GPU sync')):
            local, _ = PaperMatcher().locate(response, subpixel=True)
            narrow, _ = PaperMatcher().locate(thin, subpixel=True)
        self.assertEqual(tuple(local.shape), (3, 2))
        self.assertEqual(tuple(narrow.shape), (1, 2))
        self.assertAlmostEqual(float(local[0, 1]), 0.)
        self.assertGreater(float(local[1, 1]), 3/5)
        self.assertAlmostEqual(float(local[2, 0]), 0.)
        self.assertAlmostEqual(float(narrow[0, 0]), 0.)

    def test_match_fetches_fine_and_coarse_diagnostics_with_one_host_transfer(self):
        class FixedMatcher:
            def match_with_diagnostics(self, template, search, subpixel=False):
                is_coarse = search.shape[-1] == 24
                return (torch.tensor([[.5, .5]]), torch.tensor([.9]), None,
                        {'border_peak': torch.tensor([is_coarse]),
                         'psr': torch.tensor([2.]) if is_coarse else torch.tensor([3.])})

        frame = torch.zeros(1, 2, 60, 80)
        center = torch.tensor([[40., 30.]])
        original_cpu = torch.Tensor.cpu
        for capture, expected_border, expected_psr in [('fine', False, 3.),
                                                         ('coarse', True, 2.)]:
            tracker = PaperTracker(torch.nn.Identity(), PaperMatcher(), capture=capture)
            tracker.matcher = FixedMatcher()
            template, coarse_template = tracker._template_at(frame, center[0])
            transfers = []

            def count_cpu(tensor, *args, **kwargs):
                transfers.append(tensor.shape)
                return original_cpu(tensor, *args, **kwargs)

            with mock.patch.object(torch.Tensor, 'cpu', count_cpu), \
                    mock.patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('GPU sync')), \
                    mock.patch.object(torch.Tensor, '__float__', side_effect=AssertionError('GPU sync')):
                match, confidence, details = tracker._match(
                    template, coarse_template, frame, downsample(frame), center)
            np.testing.assert_allclose(match, [40., 30.])
            self.assertAlmostEqual(confidence, .9)
            self.assertEqual(details['border_peak'], expected_border)
            self.assertAlmostEqual(details['psr'], expected_psr)
            self.assertEqual(len(transfers), 1)

    def test_tracker_forwards_subpixel_flag_and_keeps_paper_default(self):
        tracker=PaperTracker(torch.nn.Identity(),PaperMatcher(),subpixel=True)
        self.assertTrue(tracker.subpixel)
        self.assertFalse(PaperTracker(torch.nn.Identity(),PaperMatcher()).subpixel)
        seen={}
        class Recorder:
            def match_with_diagnostics(self,template,search,subpixel=False):
                seen['subpixel']=subpixel
                ones=torch.ones(1)
                return torch.zeros(1,2),ones,None,{'border_peak':torch.zeros(1,dtype=torch.bool),
                                                   'psr':torch.ones(1)}
        tracker.matcher=Recorder()
        tracker._locate(None,None)
        self.assertTrue(seen['subpixel'])


if __name__=='__main__': unittest.main()
