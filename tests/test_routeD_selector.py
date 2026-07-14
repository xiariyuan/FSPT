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


    def test_fallback_and_distance_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"s.pt"
            m=HypothesisScorer(12,8)
            with torch.no_grad():
                for parameter in m.parameters():
                    parameter.zero_()
                m.network[-1].bias.fill_(1.0)
            torch.save({"feature_dim":12,"model_state":m.state_dict(),"feature_mean":torch.zeros(12),"feature_std":torch.ones(12)},p)
            s=RouteDRiskSelector(p, threshold=0.0)
            features=torch.randn(1,3,12)
            candidates=torch.tensor([[[0.0,0.0],[0.5,0.5],[0.8,0.8]]])
            fallback=torch.tensor([[0.1,0.1]])
            out=s.select(
                features,
                candidates,
                fallback_points=fallback,
                max_switch_distance_px=5.0,
                pixel_scale=torch.tensor([255.0,255.0]),
            )
            self.assertTrue(torch.equal(out["index"], torch.zeros(1,dtype=torch.long)))
            self.assertTrue(torch.allclose(out["points"], fallback))


    def test_soft_fusion_interpolates_from_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"s.pt"
            m=HypothesisScorer(12,8)
            with torch.no_grad():
                for parameter in m.parameters():
                    parameter.zero_()
            torch.save({"feature_dim":12,"model_state":m.state_dict(),"feature_mean":torch.zeros(12),"feature_std":torch.ones(12)},p)
            s=RouteDRiskSelector(p, threshold=0.0)
            features=torch.randn(1,3,12)
            candidates=torch.tensor([[[0.0,0.0],[1.0,1.0],[0.5,0.5]]])
            fallback=torch.tensor([[0.0,0.0]])
            out=s.select(
                features,
                candidates,
                fallback_points=fallback,
                fusion_strength=0.5,
            )
            self.assertTrue(torch.allclose(out["fusion_alpha"], torch.tensor([0.25])))
            self.assertTrue(torch.allclose(out["points"], torch.tensor([[0.25,0.25]])))

if __name__=='__main__': unittest.main()
