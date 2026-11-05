import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import h5py
import numpy as np
import torch

from ktcal_paper import diagnostics
from ktcal_paper.experiment import sha256
from ktcal_paper.model import PaperSTNet


class DiagnosticTests(unittest.TestCase):
    def test_diagnostics_keep_subjects_and_heads_separate_and_refuse_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with h5py.File(root/'frames.h5','w') as f:
                for n,xy in [('1_1',[.5,.5]),('2_1',[.6,.5])]:
                    g=f.create_group(n)
                    g['frames']=np.zeros((2,2,60,80),np.uint8)
                    g['labels']=np.array([xy+[0],xy+[1]],np.float32)
                    g['valid']=np.ones(2,bool)
                # No holdout data group: diagnose must not open it.
            manifest={'schema':1,'labels_are_placeholders':False,'dataset':'TEST FIXTURE','h5':'frames.h5',
                'splits':{'train':['1_1'],'val':['2_1'],'holdout':['3_1']},
                'records':[{'recording':f'{i}_1','frames':2} for i in (1,2,3)]}
            path=root/'manifest.json'; path.write_text(json.dumps(manifest))
            model=PaperSTNet()
            with torch.no_grad():
                for p in model.parameters(): p.zero_()
                model.head.bias.copy_(torch.tensor([.5,.5]))
                model.forward_head.bias.copy_(torch.tensor([.6,.5]))
            checkpoint=root/'weights.pt'
            torch.save({'schema':1,'model_state':model.state_dict(),'model_config':{'deployed':False,'variant':'full'},
                'config':{'window':16},'manifest_sha256':sha256(path),'phase':'bidir','epoch':1},checkpoint)
            report=diagnostics.diagnose(checkpoint,path,root/'diagnosis',stride=1,threads=2)
            result=json.loads(report.read_text())
            self.assertEqual(result['split'],'val')
            self.assertEqual(result['primary']['overall']['frames'],2)
            self.assertAlmostEqual(result['primary']['overall']['mean_error_px'],8.,places=4)
            self.assertAlmostEqual(result['forward']['overall']['mean_error_px'],0.,places=4)
            self.assertEqual(set(result['primary']['by_record']),{'2_1'})
            self.assertEqual(set(result['primary']['by_original_state']),{'0','1'})
            self.assertFalse(result['holdout_used'])
            self.assertFalse(result['deployment_timing_measured'])
            self.assertAlmostEqual(result['train_mean_baseline']['mean_error_px'],8.,places=4)
            with np.load(root/'diagnosis/predictions.npz') as arrays:
                self.assertEqual(arrays['recording'].tolist(),['2_1','2_1'])
                self.assertEqual(arrays['frame'].tolist(),[0,1])
            with self.assertRaises(ValueError):
                diagnostics.diagnose(checkpoint,path,root/'forbidden',split='holdout')
            original_load=torch.load
            expected_hash=sha256(checkpoint)
            def replace_after_load(*args,**kwargs):
                value=original_load(*args,**kwargs)
                checkpoint.write_bytes(b'replaced by a training checkpoint')
                return value
            with patch('ktcal_paper.diagnostics.torch.load',side_effect=replace_after_load):
                concurrent=diagnostics.diagnose(checkpoint,path,root/'concurrent',stride=1,threads=2)
            self.assertEqual(json.loads(concurrent.read_text())['checkpoint_sha256'],expected_hash)


if __name__=='__main__': unittest.main()
