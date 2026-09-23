import unittest
import numpy as np
import pandas as pd
from scripts.pilot_xlmr_provenance import differences, agreement, select


class TestProvenancePilot(unittest.TestCase):
    def test_comparison_metrics_and_predeclared_tolerance(self):
        base=np.ones((2,768),dtype=np.float32)
        self.assertEqual(agreement(base,base)['classification'],'exact')
        self.assertEqual(agreement(base+1e-6,base)['classification'],'near-exact')
        self.assertIn('materially different',agreement(base+.1,base)['classification'])
        f=differences(base+.1,base)
        np.testing.assert_allclose(f.l2_distance,np.sqrt(768)*.1,rtol=1e-5)
        np.testing.assert_allclose(f.cosine_similarity,1.,atol=1e-12)

    def test_selection_is_deterministic_complete_and_order_preserving(self):
        keys=['a','b']+['c']*2+['d']*2+['e']*11+['f']*11
        lengths=np.array([20,129,112,40,120,33]+[20,40,120,150,20,40,120,150,20,40,120]*2)
        text=pd.DataFrame({'user_key':keys})
        rows,chosen,reasons=select(text,lengths)
        again=select(text,lengths)
        np.testing.assert_array_equal(rows,again[0]);self.assertEqual(chosen,again[1]);self.assertEqual(reasons,again[2])
        np.testing.assert_array_equal(rows,np.flatnonzero(text.user_key.isin(chosen)))
        self.assertEqual(len(chosen),6)


if __name__=='__main__':unittest.main()
