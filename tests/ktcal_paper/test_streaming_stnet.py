import importlib.util
from types import SimpleNamespace
import unittest

import torch

from models.STNet import STNet


class StreamingSTNetTests(unittest.TestCase):
    def setUp(self):
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        torch.manual_seed(23)

    def model(self, hidden_size=256):
        self.assertIsNotNone(importlib.util.find_spec('models.streaming_stnet'),
                             'The causal streaming model is not implemented')
        from models.streaming_stnet import StreamingSTNet
        return StreamingSTNet(hidden_size=hidden_size).eval()

    @torch.inference_mode()
    def test_full_sequence_and_stateful_steps_agree(self):
        for hidden in (128, 256):
            model = self.model(hidden)
            frames = torch.rand(2, 7, 2, 60, 80)
            expected = model(frames)
            state, x, y = None, [], []
            for frame in frames.unbind(1):
                feature, state = model.stream_step(frame, state)
                x.append(model.fc_w(feature))
                y.append(model.fc_h(feature))
            for reference, output in zip(expected, (torch.stack(x, 1), torch.stack(y, 1))):
                torch.testing.assert_close(output, reference, atol=2e-5, rtol=1e-5)
            self.assertEqual(len(state), 4)
            self.assertEqual(state[-1].shape, (1, 2, hidden))

    @torch.inference_mode()
    def test_perturbing_future_frames_preserves_every_prefix_logit(self):
        model = self.model()
        frames = torch.rand(1, 9, 2, 60, 80)
        changed = frames.clone()
        changed[:, 4:] = torch.rand_like(changed[:, 4:]) * 20
        for old, new in zip(model(frames), model(changed)):
            torch.testing.assert_close(old[:, :4], new[:, :4], atol=1e-6, rtol=1e-6)

    @torch.inference_mode()
    def test_reset_removes_prior_record_information(self):
        model = self.model()
        first = torch.rand(1, 4, 2, 60, 80)
        second = torch.rand(1, 3, 2, 60, 80)
        state = None
        for frame in first.unbind(1):
            _, state = model.stream_step(frame, state)
        fresh, _ = model.stream_step(second[:, 0], None)
        expected = model.extract_temporal(second)[:, 0]
        torch.testing.assert_close(fresh, expected, atol=2e-5, rtol=1e-5)
        carried, _ = model.stream_step(second[:, 0], state)
        self.assertGreater(float((carried - fresh).abs().max()), 1e-5)

    @torch.inference_mode()
    def test_bidirectional_initialization_preserves_gate_groups(self):
        source = STNet(SimpleNamespace(stnet_readout='last')).eval()
        inputs = torch.randn(2, 5, 2048)
        direction_outputs = []
        for suffix in ('', '_reverse'):
            gru = torch.nn.GRU(2048, 128, batch_first=True)
            gru.load_state_dict({key: source.gru.state_dict()[key + suffix]
                                 for key in gru.state_dict()})
            direction_outputs.append(gru(inputs)[0])
        for hidden in (128, 256):
            model = self.model(hidden)
            metadata = model.initialize_from_bidirectional(source)
            output = model.gru(inputs)[0]
            expected = (direction_outputs[0] if hidden == 128
                        else torch.cat(direction_outputs, -1))
            torch.testing.assert_close(output, expected, atol=2e-5, rtol=1e-5)
            self.assertEqual(metadata['gru_hidden_size'], hidden)
            torch.testing.assert_close(model.fc_w[0].weight,
                                       source.fc_w[0].weight[:, :hidden])

    def test_stream_step_rejects_training_mode(self):
        model = self.model().train()
        with self.assertRaisesRegex(ValueError, 'eval'):
            model.stream_step(torch.rand(1, 2, 60, 80))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    @torch.inference_mode()
    def test_cuda_stateful_sequence_equivalence(self):
        # Different temporal batch sizes can choose different TF32 kernels.
        # Check structural equivalence under strict FP32, without changing
        # the process-wide precision used by later latency tests.
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                                       benchmark=torch.backends.cudnn.benchmark,
                                       deterministic=torch.backends.cudnn.deterministic,
                                       allow_tf32=False):
            model = self.model().cuda()
            frames = torch.rand(1, 12, 2, 60, 80, device='cuda')
            reference = model(frames)
            state, x, y = None, [], []
            for frame in frames.unbind(1):
                feature, state = model.stream_step(frame, state)
                x.append(model.fc_w(feature))
                y.append(model.fc_h(feature))
            for expected, actual in zip(reference, (torch.stack(x, 1), torch.stack(y, 1))):
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
