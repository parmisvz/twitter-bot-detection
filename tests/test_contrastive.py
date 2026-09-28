import unittest
import numpy as np
import torch
from torch import nn
from src.contrastive_core import (backbone, head, primary, fused, auxiliary, projected, supcon,
                                 pair_masks, equivalence, representation_stats, rng_hash)
from tests.test_text_aggregation import TestTextAggregation as Fixture


class TestContrastive(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        f=Fixture();f.setUp();self.mapping=f.mapping
        self.x=torch.tensor(np.random.default_rng(17).normal(size=(8,816)).astype(np.float32))
        self.x[:,40:47]=1;self.x[:,815]=1

    def test_lambda_zero_multiple_updates_including_single_class(self):
        batches=[(self.x,torch.tensor(y,dtype=torch.float32).reshape(-1,1)) for y in
                 [[0,0,0,0,0,0,0,1],[0]*8,[0,1,0,1,0,1,0,1]]]
        result=equivalence(self.mapping,batches*2,42,540/131)
        self.assertEqual(result['max_logit_difference'],0.)

    def test_auxiliary_bn_rng_and_primary_capture(self):
        torch.manual_seed(42);m=backbone(self.mapping).train()
        state=torch.get_rng_state();expected=m(self.x)
        torch.set_rng_state(state);actual,h=primary(m,self.x)
        torch.testing.assert_close(expected,actual,atol=0,rtol=0)
        before={k:v.clone() for k,v in m.named_buffers()};rng=rng_hash()
        p=head(42);self.assertEqual(rng_hash(),rng)
        h2=auxiliary(m,self.x,42,0);self.assertEqual(rng_hash(),rng)
        for k,v in m.named_buffers():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        self.assertFalse(torch.equal(h,h2))
        self.assertEqual(sum(v.numel() for v in p.parameters()),3120)
        m.eval()
        with torch.no_grad():torch.testing.assert_close(primary(m,self.x)[1],fused(m,self.x),atol=0,rtol=0)

    def test_loss_against_independent_scalar_definition(self):
        y=torch.tensor([0.,0.,0.,1.])
        a=nn.functional.normalize(torch.randn(4,16,dtype=torch.float64),dim=1).requires_grad_()
        b=nn.functional.normalize(torch.randn(4,16,dtype=torch.float64),dim=1).requires_grad_()
        z=torch.cat([a,b]);labels=y.repeat(2);expected=[]
        for c in [0,1]:
            losses=[]
            for i in range(8):
                if labels[i]!=c:continue
                candidates=[j for j in range(8) if j!=i]
                positives=[j for j in candidates if labels[j]==labels[i]]
                denom=torch.logsumexp(torch.stack([z[i].dot(z[j])/.1 for j in candidates]),0)
                losses.append(torch.stack([denom-z[i].dot(z[j])/.1 for j in positives]).mean())
            expected.append(torch.stack(losses).mean())
        loss=supcon(a,b,y)
        torch.testing.assert_close(loss,torch.stack(expected).mean(),atol=1e-12,rtol=1e-12)
        loss.backward();self.assertGreater(float(a.grad.abs().sum()),0);self.assertGreater(float(b.grad.abs().sum()),0)
        labels,other,pos,neg=pair_masks(y)
        self.assertFalse(pos.diagonal().any());self.assertEqual(pos[3].sum().item(),1)
        self.assertEqual(neg[3].sum().item(),6)
        self.assertEqual(float(supcon(a,b,torch.zeros(4)).detach()),0.)
        permutation=torch.tensor([3,0,2,1])
        torch.testing.assert_close(supcon(a[permutation],b[permutation],y[permutation]),loss,atol=1e-12,rtol=1e-12)

    def test_missingness_and_inference_head_removal(self):
        m=backbone(self.mapping).eval();p=head(42)
        x=self.x.clone();x[:,815]=0;x[:,40:47]=0
        corrupt=x.clone();corrupt[:,47:815]=float('nan');corrupt[:,self.mapping['temporal_indices']]=float('nan')
        with torch.no_grad():
            torch.testing.assert_close(m(x),m(corrupt),atol=0,rtol=0)
            logits,h=primary(m,corrupt);projected(p,h)
            torch.testing.assert_close(logits,m(corrupt),atol=0,rtol=0)
            self.assertTrue(torch.isfinite(h).all())
            self.assertEqual(h[:,48:].abs().sum().item(),0)

    def test_collapse_and_class_separation_diagnostics(self):
        y=np.array([0,0,1,1]);a=np.ones((4,3))
        s,coords,spectrum=representation_stats(a,y)
        self.assertEqual(s['effective_rank'],0);self.assertEqual(s['total_variance'],0)
        a[2:]*=-1;s,_,_=representation_stats(a,y)
        self.assertAlmostEqual(s['effective_rank'],1.)
        self.assertEqual(s['balanced_within_scatter'],0.)
        self.assertGreater(s['centroid_distance'],0.)

if __name__=='__main__':unittest.main()
