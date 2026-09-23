import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from tests import test_temporal_statistics as fixtures
from src.text_aggregation import aggregate, TemporalTextModel, train_one, ARMS, cohort_rows


class TestTextAggregation(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.TestTemporalStatistics();fixture.setUp();self.mapping=fixture.mapping
        self.config=json.loads(Path('configs/text_aggregation_v1.json').read_text())

    def test_key_alignment_single_tweet_and_absence(self):
        text=pd.DataFrame({'user_key':['b','a','b']})
        vectors=np.stack([np.full(768,-3,dtype=np.float32),np.full(768,7,dtype=np.float32),np.full(768,-1,dtype=np.float32)])
        reps,count=aggregate(text,vectors,pd.Index(['a','b','c']),np.array([True,True,False]))
        np.testing.assert_array_equal(count,[1,2,0])
        np.testing.assert_array_equal(reps['T0_mean'][0],reps['T1_max'][0])
        self.assertTrue((reps['T0_mean'][1]==-2).all());self.assertTrue((reps['T1_max'][1]==-1).all())
        self.assertTrue(np.isnan(reps['T0_mean'][2]).all())
        with self.assertRaises(ValueError): aggregate(text,vectors,pd.Index(['a','b','c']),np.array([True,False,False]))

    def test_missing_text_and_temporal_invariance(self):
        model=TemporalTextModel('T0_mean',self.mapping).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()),16897)
        x=torch.randn(4,816);x[:,40:47]=1;x[:,815]=0
        x[0,40:47]=0
        corrupt=x.clone();corrupt[:,47:815]=float('nan');corrupt[0,self.mapping['temporal_indices']]=float('nan')
        with torch.no_grad():
            torch.testing.assert_close(model(x),model(corrupt),atol=0,rtol=0)
            self.assertTrue(torch.isfinite(model(corrupt)).all())

    def test_paired_initialization_batches_and_reload(self):
        config={**self.config,'training':{**self.config['training'],'epochs':2,'batch_size':2}}
        x=np.random.default_rng(7).normal(size=(4,816)).astype(np.float32);x[:,40:47]=1;x[:,815]=1
        pop={'x':x,'y':np.array([0,1,0,1],dtype=np.float32)};data={'train':pop,'validation':pop}
        states,h,info=train_one(data,self.mapping,config,ARMS[0],42,1.)
        other={s:{**p,'x':p['x'].copy()} for s,p in data.items()}
        for p in other.values():p['x'][:,47:815]+=1
        _,_,info2=train_one(other,self.mapping,config,ARMS[1],42,1.)
        for k in ['initial_state_sha256','batch_order_sha256','epoch_order_sha256','parameter_count']:
            self.assertEqual(info[k],info2[k])
        model=TemporalTextModel(ARMS[0],self.mapping);model.load_state_dict(states['last']);model.eval()
        with torch.no_grad():self.assertTrue(torch.isfinite(model(torch.tensor(x))).all())
        saved=states['last']['text_projection.0.weight'].clone()
        with torch.no_grad():model.text_projection[0].weight.add_(1)
        torch.testing.assert_close(saved,states['last']['text_projection.0.weight'],atol=0,rtol=0)

    def test_empty_and_single_class_cohort_metrics(self):
        f=pd.DataFrame({'arm':['T0_mean']*2,'seed':[42]*2,'policy':['last']*2,'epoch':[50]*2,
                        'tweet_count':[1,2],'has_tweet_text':[1,1],'label_binary':[0,1],'bot_probability':[.2,.7]})
        rows={r['cohort']:r for r in cohort_rows(f)}
        self.assertTrue(np.isnan(rows['one_tweet']['roc_auc']))
        self.assertEqual(rows['missing_text']['metric_status'],'empty_cohort')


if __name__=='__main__':unittest.main()
