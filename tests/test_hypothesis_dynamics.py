import unittest

import torch

from projects.mmp_tracker.mmp_tracker.hypothesis_dynamics import (
    GLOBAL_SOURCE_ID,
    LOCAL_SOURCE_ID,
    generate_local_global_hypotheses,
    update_temporal_belief,
)
from projects.mmp_tracker.mmp_tracker.multi_hypothesis_belief import (
    belief_from_probabilities,
)


class TestHypothesisDynamics(unittest.TestCase):
    def test_generator_concatenates_local_and_global_sources(self):
        local_points = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        local_evidence = torch.tensor([[0.8, 0.7]])
        global_points = torch.tensor(
            [[[[0.5, 0.6], [0.7, 0.8]], [[0.2, 0.3], [0.9, 0.9]]]]
        )
        global_evidence = torch.tensor([[[0.4, 0.2], [0.5, 0.1]]])
        active = torch.tensor([[True, False]])
        candidates = generate_local_global_hypotheses(
            local_points,
            local_evidence,
            global_points,
            global_evidence,
            active_mask=active,
        )
        self.assertEqual(candidates.points.shape, (1, 2, 3, 2))
        self.assertTrue(torch.equal(candidates.source_ids[..., 0], torch.full((1, 2), LOCAL_SOURCE_ID)))
        self.assertTrue(torch.equal(candidates.source_ids[..., 1:], torch.full((1, 2, 2), GLOBAL_SOURCE_ID)))
        self.assertTrue(candidates.valid_mask[0, 0].all())
        self.assertFalse(candidates.valid_mask[0, 1].any())

    def test_first_update_matches_current_evidence(self):
        local_points = torch.tensor([[[0.1, 0.1]]])
        local_evidence = torch.tensor([[0.25]])
        global_points = torch.tensor([[[[0.9, 0.9]]]])
        global_evidence = torch.tensor([[[0.75]]])
        candidates = generate_local_global_hypotheses(
            local_points, local_evidence, global_points, global_evidence
        )
        update = update_temporal_belief(candidates)
        self.assertTrue(torch.allclose(update.belief.weights, torch.tensor([[[0.25, 0.75]]]), atol=1e-6))

    def test_temporal_prior_favors_spatially_persistent_mode(self):
        previous_points = torch.tensor([[[[0.1, 0.1], [0.9, 0.9]]]])
        previous = belief_from_probabilities(
            previous_points,
            torch.tensor([[[0.9, 0.1]]]),
        )
        local_points = torch.tensor([[[0.12, 0.11]]])
        local_evidence = torch.tensor([[0.5]])
        global_points = torch.tensor([[[[0.88, 0.9]]]])
        global_evidence = torch.tensor([[[0.5]]])
        candidates = generate_local_global_hypotheses(
            local_points, local_evidence, global_points, global_evidence
        )
        update = update_temporal_belief(
            candidates,
            previous,
            transition_sigma=0.1,
            birth_mass=0.0,
        )
        self.assertGreater(float(update.belief.weights[0, 0, 0]), 0.8)

    def test_balanced_modes_survive_temporal_update(self):
        previous_points = torch.tensor([[[[0.1, 0.1], [0.9, 0.9]]]])
        previous = belief_from_probabilities(
            previous_points,
            torch.tensor([[[0.5, 0.5]]]),
        )
        candidates = generate_local_global_hypotheses(
            torch.tensor([[[0.11, 0.1]]]),
            torch.tensor([[0.5]]),
            torch.tensor([[[[0.89, 0.91]]]]),
            torch.tensor([[[0.5]]]),
        )
        update = update_temporal_belief(
            candidates,
            previous,
            transition_sigma=0.1,
            birth_mass=0.0,
        )
        self.assertTrue(torch.allclose(update.belief.weights, torch.tensor([[[0.5, 0.5]]]), atol=1e-4))

    def test_inactive_candidate_row_stays_empty(self):
        candidates = generate_local_global_hypotheses(
            torch.tensor([[[0.1, 0.1]]]),
            torch.tensor([[1.0]]),
            torch.tensor([[[[0.9, 0.9]]]]),
            torch.tensor([[[1.0]]]),
            active_mask=torch.tensor([[False]]),
        )
        update = update_temporal_belief(candidates)
        self.assertEqual(float(update.belief.weights.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
