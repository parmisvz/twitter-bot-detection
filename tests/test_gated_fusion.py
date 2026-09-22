from copy import deepcopy
import unittest
import torch
import pandas as pd
from tests import test_gcn_integration as fixtures
from src.gated_fusion import initialized_model, train_one
from src.gcn_integration import initialized_model as baseline, train_one as baseline_train
from src.checkpoint_weight_experiment import state_hash


class TestGatedFusion(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.TestGCNIntegration()
        fixture.setUp()
        fixture.pop['users'] = pd.DataFrame(index=pd.Index(['a', 'b', 'c', 'd'], name='user_key'))
        self.f = fixture

    def test_identity_initialization_and_rng_pairing(self):
        f = self.f
        concat = initialized_model('concat', f.mapping, 42)
        rng = torch.get_rng_state().clone()
        gated = initialized_model('gated', f.mapping, 42)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(state_hash(concat.model.state_dict()), state_hash(gated.model.state_dict()))
        self.assertEqual(sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in concat.parameters()), 300)
        concat.eval(); gated.eval()
        x = torch.tensor(f.pop['x'], dtype=torch.float32)
        torch.testing.assert_close(concat(x, torch.arange(4), f.graph), gated(x, torch.arange(4), f.graph), atol=0, rtol=0)

    def test_absent_modalities_and_gateless_population(self):
        f = self.f
        model = initialized_model('gated', f.mapping, 42).eval()
        with torch.no_grad():
            model.gate.weight.fill_(.07)
            model.gate.bias.fill_(.1)
        x = torch.tensor(f.pop['x'], dtype=torch.float32)
        x[:, 40:47] = 0
        graph = deepcopy(f.graph)
        graph['available'][:] = False
        _, weights = model.components(x, torch.arange(4), graph)
        self.assertTrue(weights[:, 1:].eq(0).all())
        before = model(x, torch.arange(4), graph)
        graph['data'].x *= 1000
        x[:, model.model.base.temporal_indices] = 1e8
        torch.testing.assert_close(before, model(x, torch.arange(4), graph), atol=0, rtol=0)

    def test_concat_trainer_reproduces_frozen_implementation(self):
        f = self.f
        data = {'train': f.pop, 'validation': f.pop}
        graphs = {'train': f.graph, 'validation': f.graph}
        spec = {'epochs': 2, 'batch_size': 2, 'lr': .001, 'weight_decay': .0001}
        states, _, info = baseline_train(data, graphs, f.mapping, spec, 'gcn', 42, 1.)
        actual, _, paired = train_one(data, graphs, f.mapping, spec, 'concat', 42, 1.)
        self.assertEqual(info['epoch_order_sha256'], paired['epoch_order_sha256'])
        for policy, state in states.items():
            for key, value in state.items():
                torch.testing.assert_close(value, actual[policy]['model.' + key], atol=0, rtol=0)
        gated, _, treatment = train_one(data, graphs, f.mapping, spec, 'gated', 42, 1.)
        self.assertEqual(paired['epoch_order_sha256'], treatment['epoch_order_sha256'])
        self.assertTrue(gated['last']['gate.weight'].ne(0).any())


if __name__ == '__main__':
    unittest.main()
