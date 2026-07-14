import unittest
import tempfile
from pathlib import Path
import torch
from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import HypothesisScorer
from projects.mmp_tracker.mmp_tracker.routeD_selector import RouteDRiskSelector

class TestRouteDSelector(unittest.TestCase):
    def test_load_and_select(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"s.pt"
            m=HypothesisScorer(12,8)
            torch.save({"feature_dim":12,"model_state":m.state_dict(),"feature_mean":torch.zeros(12),"feature_std":torch.ones(12)},p)
            s=RouteDRiskSelector(p)
            out=s.select(torch.randn(2,3,12),torch.randn(2,3,2))
            self.assertEqual(out["points"].shape,(2,2))

if __name__=='__main__': unittest.main()
