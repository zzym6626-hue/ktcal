import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.current_readout_streaming_stnet import CurrentReadoutStreamingSTNet
from models.initialized_streaming_stnet import InitializedCurrentReadoutStreamingSTNet


class PersistentStreamingStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(51)

    def adapter(self, model):
        from ktcal_paper import streaming_inference as inference
        self.assertTrue(hasattr(inference, 'PersistentStreamingStep'), 'Persistent state adapter is missing')
        viewed = inference.ViewedStreamingStep(model, ((0, 0, False), (2, -1, True))).eval()
        return inference.PersistentStreamingStep(viewed).eval(), viewed

    @torch.inference_mode()
    def test_four_and_five_states_update_supplied_storage_and_match_reference(self):
        frames = torch.rand(2, 6, 2, 60, 80)
        for model_class in (CurrentReadoutStreamingSTNet, InitializedCurrentReadoutStreamingSTNet):
            with self.subTest(model=model_class.__name__):
                model = model_class(hidden_size=128).eval()
                model.current_readout_scale.fill_(.3)
                if isinstance(model, InitializedCurrentReadoutStreamingSTNet):
                    model.initial_hidden_scale.fill_(.4)
                adapter, viewed = self.adapter(model)
                actual_state = adapter.initial_state(frames[:, 0])
                expected_state = viewed.initial_state(frames[:, 0])
                pointers = [value.data_ptr() for value in actual_state]
                for index in range(6):
                    expected, expected_state = viewed.stream_step(frames[:, index], expected_state)
                    actual, new_state = adapter.stream_step(frames[:, index], actual_state)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                    self.assertEqual([value.data_ptr() for value in new_state], pointers)
                    for original, returned, want in zip(actual_state, new_state, expected_state):
                        self.assertIs(original, returned)
                        torch.testing.assert_close(returned, want, atol=1e-6, rtol=1e-5)
                    actual_state = new_state
                reset = adapter.initial_state(frames[:, 0])
                actual, _ = adapter.stream_step(frames[:, 0], reset)
                expected, _ = viewed.stream_step(frames[:, 0], viewed.initial_state(frames[:, 0]))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    @torch.inference_mode()
    def test_cuda_replay_copies_only_current_frame_and_honors_external_reset(self):
        from ktcal_paper.streaming_inference import StatefulGraphStep, ViewedStreamingStep
        frames = torch.rand(2, 4, 2, 60, 80, device='cuda')
        model = InitializedCurrentReadoutStreamingSTNet(hidden_size=128).cuda().eval()
        model.current_readout_scale.fill_(.3)
        model.initial_hidden_scale.fill_(.4)
        viewed = ViewedStreamingStep(model, ((0, 0, False), (0, 0, True))).eval()
        persistent = StatefulGraphStep(viewed, persistent_state=True)
        control = StatefulGraphStep(viewed, persistent_state=False)
        actual, state = persistent(frames[:, 0], None)
        expected, control_state = control(frames[:, 0], None)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
        copied = []
        original_copy = torch.Tensor.copy_

        def observed_copy(destination, source, *args, **kwargs):
            copied.append(tuple(destination.shape))
            return original_copy(destination, source, *args, **kwargs)

        with patch.object(torch.Tensor, 'copy_', new=observed_copy):
            actual, state = persistent(frames[:, 1], state)
        self.assertEqual(copied, [(2, 2, 60, 80)])
        expected, control_state = control(frames[:, 1], control_state)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
        for index in range(2, 4):
            actual, state = persistent(frames[:, index], state)
            expected, control_state = control(frames[:, index], control_state)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
        actual, _ = persistent(frames[:, 0], viewed.initial_state(frames[:, 0]))
        expected, _ = control(frames[:, 0], None)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
        # Different record batch sizes must recapture without inheriting history.
        actual, _ = persistent(frames[:1, 0], None)
        expected, _ = control(frames[:1, 0], None)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    @torch.inference_mode()
    def test_predictor_persistent_and_control_warm_partial_requests_reset_are_equivalent(self):
        from ktcal_paper.streaming_inference import StreamingPredictor
        model = CurrentReadoutStreamingSTNet(hidden_size=128).cuda().eval()
        model.current_readout_scale.fill_(.3)
        frames = torch.rand(2, 8, 2, 60, 80, device='cuda')
        options = dict(views=((0, 0, False), (2, -1, True)), temperature=.65,
                       ensemble='median', warm_start_steps=3, flip_axis_px=80.)
        fast = StreamingPredictor(model, persistent_state=True, **options)
        control = StreamingPredictor(model, persistent_state=False, **options)
        saved = []
        for index in range(8):
            fast.observe(frames[:, index])
            control.observe(frames[:, index])
            mask = np.array([True, index % 3 == 0])
            actual = fast(frames[:, :index + 1], mask)
            expected = control(frames[:, :index + 1], mask)
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-5)
            saved.append((actual, actual.copy()))
        for actual, snapshot in saved:
            np.testing.assert_array_equal(actual, snapshot)
        self.assertEqual(fast.feature_updates, 22)
        self.assertEqual(fast.coordinate_head_requests, control.coordinate_head_requests)
        fast.reset()
        control.reset()
        fast.observe(frames[:, 0])
        control.observe(frames[:, 0])
        np.testing.assert_allclose(fast(frames[:, :1], np.ones(2, bool)),
                                   control(frames[:, :1], np.ones(2, bool)), atol=3e-5, rtol=1e-5)
