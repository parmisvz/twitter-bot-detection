import unittest
import numpy as np
import torch
from src.xlmr_pooling import masks,pool
from src.token_pooling import TemporalTextModel,train_one,ARMS
from tests import test_text_aggregation as fixtures

class TestTokenPooling(unittest.TestCase):
    def test_excludes_interior_special_ids_and_padding(self):
        ids=torch.tensor([[0,10,3,250001,11,2,1]])
        attention=torch.tensor([[1,1,1,1,1,1,0]])
        returned=torch.tensor([[1,0,0,0,0,1,1]])
        p0,p1,special=masks(ids,attention,returned,[0,1,2,3,250001])
        self.assertEqual(p0.sum().item(),6);self.assertEqual(p1.tolist(),[[False,True,False,False,True,False,False]])
        hidden=torch.arange(14,dtype=torch.float32).reshape(1,7,2)
        torch.testing.assert_close(pool(hidden,p1),hidden[:,[1,4]].mean(1),atol=0,rtol=0)
        torch.testing.assert_close(pool(hidden,p0),hidden[:,:6].mean(1),atol=0,rtol=0)
        with self.assertRaises(ValueError):masks(torch.tensor([[0,3,2,1]]),torch.tensor([[1,1,1,0]]),torch.tensor([[1,0,1,1]]),[0,1,2,3])

    def test_paired_factory_training_and_missing_text(self):
        f=fixtures.TestTextAggregation();f.setUp()
        config={**f.config,'arms':ARMS,'expected_parameters':dict.fromkeys(ARMS,16897),'training':{**f.config['training'],'epochs':2,'batch_size':2}}
        x=np.random.default_rng(7).normal(size=(4,816)).astype(np.float32);x[:,40:47]=1;x[:,815]=1
        data={s:{'x':x.copy(),'y':np.array([0,1,0,1],dtype=np.float32)} for s in ['train','validation']}
        a,_,i=train_one(data,f.mapping,config,'P0',42,1.)
        for p in data.values():p['x'][:,47:815]+=1
        _,_,j=train_one(data,f.mapping,config,'P1',42,1.)
        for k in ['initial_state_sha256','batch_order_sha256','epoch_order_sha256','parameter_count']:self.assertEqual(i[k],j[k])
        model=TemporalTextModel('P0',f.mapping).eval();model.load_state_dict(a['last']);self.assertEqual(i['parameter_count'],16897)
        clean=torch.tensor(x);clean[:,815]=0;corrupt=clean.clone();corrupt[:,47:815]=float('nan')
        with torch.no_grad():torch.testing.assert_close(model(clean),model(corrupt),atol=0,rtol=0)

if __name__=='__main__':unittest.main()
