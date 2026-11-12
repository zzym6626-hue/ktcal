import unittest
import torch
from ktcal_paper.matching import PaperMatcher, TEMPLATE, crop_rectangle
from ktcal_paper.offline_batch import BatchedFineMatcher

@unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
class GraphTests(unittest.TestCase):
    def test_graph_matches_eager_with_changed_input_and_centers(self):
        torch.manual_seed(12)
        matcher = PaperMatcher().cuda().eval()
        eager = BatchedFineMatcher(matcher,(37,32),cuda_graph=False)
        graph = BatchedFineMatcher(matcher,(37,32),cuda_graph=True)
        with torch.inference_mode():
            frames=torch.randn(2,2,60,80,device='cuda')
            centers=torch.tensor([[40.,30.],[30.,20.]],device='cuda')
            template=crop_rectangle(frames,centers,TEMPLATE)
            for delta in (0.,.35):
                value=frames+delta
                moved=centers+delta
                reference=eager(template,value,moved)
                result=graph(template,value,moved).clone()
                torch.testing.assert_close(result,reference,atol=1e-5,rtol=1e-5)
            # Template shape is unchanged, but the graph must recapture the frame shape.
            resized=torch.randn(2,2,61,81,device='cuda')
            reference=eager(template,resized,centers)
            result=graph(template,resized,centers).clone()
            torch.testing.assert_close(result,reference,atol=1e-5,rtol=1e-5)
