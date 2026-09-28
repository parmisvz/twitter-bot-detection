import unittest
import torch
import numpy as np
from tests import test_text_aggregation as fixtures
from src.modality_ablation import build, initialization, invariance, ARMS, COUNTS


class TestModalityAblation(unittest.TestCase):
    def setUp(self):
        f=fixtures.TestTextAggregation();f.setUp();self.mapping=f.mapping

    def test_counts_pairing_fanin_and_rng(self):
        for seed in [42,43,44]:
            m=build(self.mapping,'M4',seed);reference=initialization(m,'M4',seed);rng=torch.get_rng_state()
            for arm in ARMS:
                model=build(self.mapping,arm,seed);info=initialization(model,arm,seed)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertEqual(sum(p.numel() for p in model.parameters()),COUNTS[arm])
                self.assertEqual(info['final_classifier'],reference['final_classifier'])
                for b,h in info['encoders'].items():self.assertEqual(h,reference['encoders'][b])
                if arm!='M4':
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(300000+seed);expected=torch.nn.Linear(info['first_fan_in'],32)
                    for k,v in expected.state_dict().items():torch.testing.assert_close(v,model.classifier[0].state_dict()[k],atol=0,rtol=0)

    def test_excluded_inputs_missingness_and_reproducibility(self):
        x=np.random.default_rng(4).normal(size=(32,816)).astype(np.float32);x[:,40:47]=1;x[:,815]=1
        for arm in ARMS:
            model=build(self.mapping,arm,42);invariance(model,arm,{'x':x},self.mapping)
            a=initialization(model,arm,42);b=initialization(build(self.mapping,arm,42),arm,42)
            self.assertEqual(a,b)

if __name__=='__main__':unittest.main()
