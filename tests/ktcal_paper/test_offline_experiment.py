import unittest
import numpy as np
import torch
from ktcal_paper.offline_experiment import spatial_view, undo_view, average_views, smooth_pixels

class ViewTests(unittest.TestCase):
    def test_shift_and_mirror_restore_coordinate_system(self):
        frames = torch.zeros(1, 2, 2, 60, 80)
        frames[..., 25, 30] = 1.
        view = (2, -1, True)
        image = spatial_view(frames, view)
        self.assertEqual(float(image[..., 24, 47].sum()), 4.)
        predicted = np.array([[[47., 24.], [47., 24.]]], dtype=np.float32)
        np.testing.assert_allclose(undo_view(predicted, view), [[[30.,25.],[30.,25.]]])
        self.assertEqual(float(image[..., -1,:].sum()), 0.)
    def test_ensemble_never_uses_labels_and_filter_preserves_records(self):
        cache = np.stack([np.zeros((2,100,2)), np.ones((2,100,2))*2.])
        output = average_views(cache, [0,1])
        np.testing.assert_allclose(output, 1.)
        output[1] = 50.
        smoothed = smooth_pixels(output, 'ma11')
        np.testing.assert_allclose(smoothed[0], 1., atol=1e-5)
        np.testing.assert_allclose(smoothed[1], 50., atol=1e-5)
    def test_invalid_views_or_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            spatial_view(torch.zeros(1,100,2,60,80), (80,0,False))
        with self.assertRaises(ValueError):
            average_views(np.zeros((2,1,100,2)), [0,1], weights=[1.,-1.])

class RobustFilterTests(unittest.TestCase):
    def test_robust_ensemble_rejects_one_outlier(self):
        cache = np.stack([np.ones((2,100,2))*x for x in (1.,1.,2.,2.,100.)])
        np.testing.assert_allclose(average_views(cache,list(range(5)),method='median'),2.)
        np.testing.assert_allclose(average_views(cache,list(range(5)),method='trimmed'),5./3.)
    def test_alternative_filters_do_not_cross_record_boundaries(self):
        values=np.stack([np.ones((100,2))*x for x in (1.,50.)])
        for kind in ('gauss1','gauss2','median5','savgol9','rts_q1_r4'):
            np.testing.assert_allclose(smooth_pixels(values,kind),values,atol=1e-5)
