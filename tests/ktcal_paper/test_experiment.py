import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from ktcal_paper.experiment import train,evaluate,metrics


class ExperimentTests(unittest.TestCase):
    def test_metrics_are_pixel_euclidean_and_do_not_clip_predictions(self):
        result=metrics([[3,4],[80,60]],[[0,0],[74,52]])
        self.assertEqual(result['mean_error_px'],7.5)
        self.assertEqual(result['p3'],0)
        self.assertEqual(result['p5'],.5)
        self.assertEqual(result['p10'],1.)
        self.assertEqual(result['mse_px2'],31.25)
        self.assertEqual(result['out_of_bounds_ratio'],.5)

    def test_resume_matches_uninterrupted_training_and_evaluation_writes_real_rows(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); rng=np.random.default_rng(10)
            with h5py.File(root/'frames.h5','w') as f:
                for name in ('1_1','2_1','3_1'):
                    group=f.create_group(name)
                    group['frames']=rng.integers(0,16,(2,2,60,80),dtype=np.uint8)
                    group['labels']=np.array([[.4,.3,0],[.5,.4,1]],np.float32)
                    group['valid']=np.ones(2,bool)
            manifest={'schema':1,'dataset':'TEST FIXTURE','h5':'frames.h5','labels_are_placeholders':False,
                'splits':{'train':['1_1'],'val':['2_1'],'holdout':['3_1']},
                'records':[{'recording':f'{n}_1','frames':2} for n in (1,2,3)],'bin_ms':10.,'label_policy':'fixture'}
            path=root/'manifest.json'; path.write_text(json.dumps(manifest))
            config={'epochs':2,'batch_size':2,'threads':2,'device':'cpu'}
            full=train(path,root/'full',**config)
            last=train(path,root/'resume',stop_after_epochs=1,**config)
            resumed=train(resume=last)
            a=torch.load(full,map_location='cpu',weights_only=False)
            b=torch.load(resumed,map_location='cpu',weights_only=False)
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key],b['model_state'][key],rtol=0,atol=0)
            for key in a['matcher_state']:
                torch.testing.assert_close(a['matcher_state'][key],b['matcher_state'][key],rtol=0,atol=0)
            summary=json.loads((root/'resume/training_summary.json').read_text())
            self.assertEqual(summary['finetune_epochs'],2)
            self.assertFalse(summary['holdout_used_for_model_selection'])
            report=evaluate(resumed,path,root/'evaluation',threads=2)
            result=json.loads(report.read_text())
            self.assertEqual(len(result['rows']),2)
            self.assertEqual(result['rows'][0]['frames'],2)
            self.assertFalse(result['paper_results_reproduced'])
            self.assertEqual(result['rows'][0]['trigger_ratio'],1.)
            self.assertTrue((root/'evaluation/metrics.csv').exists())
            with self.assertRaises(FileExistsError): train(path,root/'full',**config)


if __name__=='__main__': unittest.main()
