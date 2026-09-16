import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.multimodal import assemble, predictions, train_model, MultimodalFusionClassifier, validation_loss
ROOT=Path(__file__).resolve().parents[1]

def fixture():
    keys=pd.Index(['a','b','c','d'],name='user_key')
    masks=pd.DataFrame({'has_graph':[1,1,0,1],'has_tweet_text':[1,1,1,0],'has_temporal':[1,0,1,1]},index=keys)
    pop={'users':pd.DataFrame({'label_binary':[0,1,0,1]},index=keys),'masks':masks,'x':np.arange(160).reshape(4,40)/160,'y':np.array([0,1,0,1])}
    emb={'keys':pd.Index(['c','b','a']),'x':np.arange(2304).reshape(3,768)/2304}
    mapping=pd.DataFrame({'user_key':['b','a','d'],'node_index':[0,1,2]})
    return pop,emb,mapping,torch.tensor([[2,2,4],[1,1,2],[3,3,6]],dtype=torch.float32)

class Tests(unittest.TestCase):
    def test_alignment_missingness_and_registry(self):
        pop,e,m,d=fixture();s=assemble(pop,e,m,d)
        self.assertEqual(s['keys'].tolist(),['a','b'])
        np.testing.assert_array_equal(s['dataset'].tensors[1][0],e['x'][2].astype('float32'))
        torch.testing.assert_close(s['dataset'].tensors[2][0],d[1])
        c=json.loads((ROOT/'configs/multimodal_v2.json').read_text())
        p=predictions(MultimodalFusionClassifier(),pop,s,'test',c).set_index('user_key')
        self.assertEqual(len(p),4)
        self.assertEqual(p.loc['b','status'],'predicted')
        self.assertEqual(p.loc['c','status'],'missing_graph')
        self.assertEqual(p.loc['d','status'],'missing_tweet_text')
        self.assertTrue(p.loc[['c','d'],'bot_probability'].isna().all())
        self.assertTrue(p.loc[['c','d'],'prediction'].isna().all())
        e['keys']=pd.Index(['c','b','b'])
        with self.assertRaises(ValueError):assemble(pop,e,m,d)

    def test_archived_architecture_identical_logits(self):
        n=json.loads((ROOT/'analysis/archive/06_multimodal_fusion_pre_v2.ipynb').read_text(encoding='utf8'))
        names={'TabularEncoder','TextEncoder','GraphEncoder','MultimodalFusionClassifier'}
        defs=[node for c in n['cells'] if c['cell_type']=='code' for node in ast.parse(''.join(c['source'])).body if isinstance(node,ast.ClassDef) and node.name in names]
        self.assertEqual(len(defs),4)
        scope={'nn':nn,'torch':torch}
        exec(compile(ast.Module(body=defs,type_ignores=[]),'archived','exec'),scope)
        old=scope['MultimodalFusionClassifier']().eval();new=MultimodalFusionClassifier().eval()
        new.load_state_dict(old.state_dict())
        x=[torch.randn(5,n) for n in (40,768,3)]
        torch.testing.assert_close(new(*x),old(*x),rtol=0,atol=0)

    def test_repeatable_training_and_deepcopy(self):
        s=assemble(*fixture());c=json.loads((ROOT/'configs/multimodal_v2.json').read_text());c['training']['epochs']=3
        states,h,info=train_model(s,s,c);again,h2,i2=train_model(s,s,c)
        pd.testing.assert_frame_equal(h,h2);self.assertEqual(info,i2)
        for v in states:
            for k in states[v]:torch.testing.assert_close(states[v][k],again[v][k],rtol=0,atol=0)
        best=deepcopy(states['best_validation_loss'])
        for k in states['last']:
            states['last'][k].add_(10)
            torch.testing.assert_close(best[k],states['best_validation_loss'][k],rtol=0,atol=0)
        model=MultimodalFusionClassifier();model.load_state_dict(best)
        self.assertAlmostEqual(validation_loss(model,s,c,torch.tensor(info['class_weights'])),h.validation_loss.min(),places=7)

if __name__=='__main__':unittest.main()
