import threading
import unittest
import numpy as np
import torch
from ktcal_paper.tracker import PaperTracker


class ScriptedMatcher(torch.nn.Module):
    def __init__(self,release,stale):
        super().__init__(); self.calls=0; self.release=release; self.stale=stale

    def forward(self,template,search):
        self.calls+=1
        if self.calls==1: return torch.tensor([[0.,0.]]),torch.tensor([.9]),None
        if not self.stale: self.release.set()
        return torch.tensor([[.5,.5]]),torch.tensor([.1 if self.stale else .9]),None


class ScriptedTracker(PaperTracker):
    def __init__(self,stale):
        self.release=threading.Event(); self.worker_thread=None
        super().__init__(torch.nn.Identity(),ScriptedMatcher(self.release,stale))

    def _infer(self,window):
        index=int(window[-1,0,0,0])
        if index==1:
            self.worker_thread=threading.current_thread().ident
            if not self.release.wait(2): raise AssertionError('Fast path blocked waiting for worker')
        if index==2: self.release.set()
        return np.array([40.+10*index,30.]),0.


class TrackerTests(unittest.TestCase):
    def test_medium_correction_is_asynchronous_and_online_output_is_immutable(self):
        frames=torch.stack([torch.full((2,60,80),float(i)) for i in range(3)])
        tracker=ScriptedTracker(False)
        online,corrected,stats=tracker.track(frames)
        self.assertEqual(stats['tier2'],1)
        self.assertNotEqual(tracker.worker_thread,threading.current_thread().ident)
        np.testing.assert_array_equal(online[1],[40,30])
        np.testing.assert_array_equal(corrected[1],[50,30])
        self.assertEqual(stats['async_applied'],1)
        self.assertEqual(stats['async_events'][0]['source_frame'],1)
        self.assertGreaterEqual(stats['correction_application_ms'][0],stats['correction_completion_ms'][0])
        self.assertTrue(np.isfinite(online).all() and np.isfinite(corrected).all())

    def test_sync_reset_discards_older_async_observation(self):
        frames=torch.stack([torch.full((2,60,80),float(i)) for i in range(3)])
        online,corrected,stats=ScriptedTracker(True).track(frames)
        self.assertEqual(stats['tier3'],2)
        self.assertEqual(stats['async_discarded'],1)
        self.assertEqual(stats['async_applied'],0)
        np.testing.assert_array_equal(online[2],[60,30])
        np.testing.assert_array_equal(corrected[2],[60,30])


if __name__=='__main__': unittest.main()
