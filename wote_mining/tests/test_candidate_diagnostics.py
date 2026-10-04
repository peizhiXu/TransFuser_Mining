"""Candidate telemetry tests without CARLA or checkpoint dependencies."""

import json
import unittest

import torch

from wote_mining.WoTE_diagnostics import reward_topk_diagnostics
from wote_mining.WoTE_model import METRIC_NAMES


class CandidateDiagnosticsTests(unittest.TestCase):
    def make_outputs(self, count=6):
        trajectories = torch.zeros(1, count, 8, 3)
        trajectories[0, :, 1, 0] = torch.arange(count).float()
        trajectories[0, :, 1, 1] = torch.arange(count).float()
        return {
            "world_trajectories": trajectories.requires_grad_(),
            "final_rewards": torch.arange(count).float()[None].requires_grad_(),
            "metric_scores": torch.arange(count * 5).float().reshape(
                1, count, 5
            ).requires_grad_(),
            "selected_index": torch.tensor([count - 1]),
        }

    def test_ranking_metric_alignment_speed_and_serialization(self):
        outputs = self.make_outputs()
        candidates = reward_topk_diagnostics(outputs)["reward_topk_candidates"]
        self.assertEqual([item["anchor_index"] for item in candidates], [5, 4, 3, 2])
        self.assertEqual([item["selected"] for item in candidates], [True, False, False, False])
        for rank, item in enumerate(candidates):
            index = item["anchor_index"]
            self.assertEqual(item["rank"], rank + 1)
            self.assertEqual(item["reward"], float(index))
            self.assertEqual(item["trajectory"], outputs["world_trajectories"][0, index].tolist())
            self.assertAlmostEqual(item["learned_desired_speed_mps"], 2 * (2 ** 0.5) * index, places=5)
            self.assertEqual(item["metric_scores"], dict(zip(METRIC_NAMES, range(index * 5, index * 5 + 5))))
        json.dumps(candidates, allow_nan=False)

    def test_ties_include_actual_selected_candidate(self):
        outputs = self.make_outputs()
        outputs["final_rewards"] = torch.ones(1, 6)
        candidates = reward_topk_diagnostics(outputs)["reward_topk_candidates"]
        self.assertEqual([item["anchor_index"] for item in candidates], [5, 0, 1, 2])

    def test_small_bank_and_empty_request(self):
        outputs = self.make_outputs(2)
        self.assertEqual(len(reward_topk_diagnostics(outputs)["reward_topk_candidates"]), 2)
        self.assertEqual(reward_topk_diagnostics(outputs, topk=0)["reward_topk_candidates"], [])

    def test_no_output_mutation_or_gradient_changes(self):
        outputs = self.make_outputs()
        snapshots = {key: value.detach().clone() for key, value in outputs.items()}
        reward_topk_diagnostics(outputs)
        for key, value in outputs.items():
            self.assertTrue(torch.equal(value, snapshots[key]))
        self.assertTrue(outputs["world_trajectories"].requires_grad)
        outputs["world_trajectories"].sum().backward()
        self.assertTrue(torch.equal(outputs["world_trajectories"].grad, torch.ones_like(outputs["world_trajectories"])))
        self.assertIsNone(outputs["final_rewards"].grad)


if __name__ == "__main__":
    unittest.main()
