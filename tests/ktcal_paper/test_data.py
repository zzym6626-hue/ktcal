import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from ktcal_paper.data import subject_split, read_labels, WindowDataset, load_manifest


class DataTests(unittest.TestCase):
    def test_subjects_are_disjoint_and_split_is_stable(self):
        names = [f'{s}_{r}' for s in range(1,14) for r in (1,2)]
        a = subject_split(names,42)
        self.assertEqual(a,subject_split(list(reversed(names)),42))
        sets = [{n.split('_')[0] for n in a[k]} for k in ('train','val','holdout')]
        self.assertEqual([len(s) for s in sets],[9,2,2])
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_original_label_state_is_preserved_and_placeholder_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'label.txt'
            p.write_text('(320, 240, 0)\n(320, 240, 1)\n')
            labels,valid=read_labels(p)
            np.testing.assert_array_equal(labels[:,2],[0,1])
            np.testing.assert_allclose(labels[:,:2],[[.5,.5],[.5,.5]])
            self.assertTrue(valid.all())
            p.write_text('(0,0,0)\n(0,0,0)\n')
            with self.assertRaisesRegex(ValueError,'placeholder'): read_labels(p)

    def test_window_padding_no_future_and_no_recording_crossing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with h5py.File(root/'frames.h5','w') as f:
                for name,values in [('1_1',[1,2,3]),('2_1',[8,9])]:
                    g=f.create_group(name)
                    g['frames']=np.stack([np.full((2,60,80),v,np.uint8) for v in values])
                    g['labels']=np.tile([.5,.5,0],(len(values),1)).astype(np.float32)
                    g['valid']=np.ones(len(values),bool)
            manifest={'schema':1,'dataset':'fixture','h5':'frames.h5','labels_are_placeholders':False,
                'splits':{'train':['1_1','2_1'],'val':[],'holdout':[]},
                'records':[{'recording':'1_1','frames':3},{'recording':'2_1','frames':2}]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            ds=WindowDataset(root/'manifest.json','train',window=3)
            np.testing.assert_allclose(ds[1][0][:,0,0,0].numpy(),np.array([1,1,2])/15)
            np.testing.assert_allclose(ds[3][0][:,0,0,0].numpy(),np.array([8,8,8])/15)
            ds.close()
            manifest['labels_are_placeholders']=True
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'placeholder'): load_manifest(root/'manifest.json')


if __name__=='__main__': unittest.main()
