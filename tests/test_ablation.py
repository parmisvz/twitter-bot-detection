import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.ablation import AblationClassifier, VARIANTS, predictions, train_model
from src.multimodal import assemble, validation_loss
ROOT=Path(__file__).resolve().parents[1]

def fixture():
    keys=pd.Index(['a','b','c','d'],name='user_key')
    masks=pd.DataFrame({'has_graph':[1,1,0,1],'has_tweet_text':[1,1,1,0],'has_temporal':[1,0,1,1]},index=keys)
    pop={'users':pd.DataFrame({'label_binary':[0,1,0,1]},index=keys),'masks':masks,'x':np.arange(160).reshape(4,40)/160,'y':np.array([0,1,0,1])}
    emb={'keys':pd.Index(['c','b','a']),'x':np.arange(2304).reshape(3,768)/2304}
    mapping=pd.DataFrame({'user_key':['b','a','d'],'node_index':[0,1,2]})
    return pop,assemble(pop,emb,mapping,torch.tensor([[2,2,4],[1,1,2],[3,3,6]],dtype=torch.float32))

class Tests(unittest.TestCase):
    def test_all_architectures_match_archive(self):
        notebook=json.loads((ROOT/'analysis/archive/07_ablation_study_pre_v2.ipynb').read_text(encoding='utf8'))
        scope={'nn':nn,'torch':torch}
        classes={}
        for cell in notebook['cells']:
            if cell['cell_type']!='code':continue
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node,ast.ClassDef) and node.name.endswith(('Encoder','FusionClassifier')):
                    exec(compile(ast.Module(body=[node],type_ignores=[]),'archive','exec'),scope)
                    if node.name.endswith('FusionClassifier'):classes[node.name]=scope[node.name]().eval()
        names={'tabular_text':'TabularTextFusionClassifier','text_graph':'TextGraphFusionClassifier','tabular_graph':'TabularGraphFusionClassifier'}
        args=[torch.randn(5,n) for n in (40,768,3)]
        indices={'tabular':0,'text':1,'graph':2}
        for v in VARIANTS:
            current=AblationClassifier(v).eval();old=classes[names[v]]
            current.load_state_dict(old.state_dict(),strict=True)
            torch.testing.assert_close(current(*args),old(*(args[indices[n]] for n in VARIANTS[v])),rtol=0,atol=0)
            removed=next(i for n,i in indices.items() if n not in VARIANTS[v])
            altered=list(args);altered[removed]=torch.full_like(args[removed],float('nan'))
            torch.testing.assert_close(current(*args),current(*altered),rtol=0,atol=0)
            self.assertFalse(hasattr(current,list(indices)[removed]+'_encoder'))

    def test_registry_distinguishes_design_exclusion_from_missingness(self):
        pop,sample=fixture();config=json.loads((ROOT/'configs/ablation_v2.json').read_text())
        for v in VARIANTS:
            frame=predictions(AblationClassifier(v),pop,sample,'test',config).set_index('user_key')
            self.assertEqual(len(frame),4)
            self.assertEqual(frame.loc['b','status'],'predicted') # Missing temporal does not filter.
            self.assertTrue(frame.loc[['c','d'],'bot_probability'].isna().all())
            self.assertTrue(frame.loc[['c','d'],'prediction'].isna().all())
            self.assertEqual(frame.loc['c','status'],'outside_fixed_ablation_cohort' if v=='tabular_text' else 'missing_required_modality')
            self.assertEqual(frame.loc['d','status'],'outside_fixed_ablation_cohort' if v=='tabular_graph' else 'missing_required_modality')

    def test_seeded_training_and_independent_best_states(self):
        _,s=fixture();c=json.loads((ROOT/'configs/ablation_v2.json').read_text());c['training']['epochs']=3
        for v in VARIANTS:
            states,h,info=train_model(s,s,c,v);again,h2,i2=train_model(s,s,c,v)
            pd.testing.assert_frame_equal(h,h2);self.assertEqual(info,i2)
            best=deepcopy(states['best_validation_loss'])
            for snapshot in states:
                for k in states[snapshot]:torch.testing.assert_close(states[snapshot][k],again[snapshot][k],rtol=0,atol=0)
            for k in states['last']:
                states['last'][k].add_(10)
                torch.testing.assert_close(best[k],states['best_validation_loss'][k],rtol=0,atol=0)
            model=AblationClassifier(v);model.load_state_dict(best)
            self.assertAlmostEqual(validation_loss(model,s,c,torch.tensor(info['class_weights'])),h.validation_loss.min(),places=7)

if __name__=='__main__':unittest.main()
