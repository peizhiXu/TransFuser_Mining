"""Verify the inference ablation with real planner/world/reward modules on CPU."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from wote_mining.WoTE_model import WoTEMiningPlanner


class WorldPassTests(unittest.TestCase):
    def test_calls_candidates_selection_and_training_compatibility(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.manual_seed(17)

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(lidar_pos=[3.5])
                for name in ('change_channel_conv_image', 'c5_conv', 'up_conv5',
                             'up_conv4', 'up_conv3'):
                    setattr(self, name, nn.Identity())

            def forward(self, rgb, lidar, speed, return_fused_lidar=False):
                return lidar

        anchors = np.zeros((8, 8, 3), dtype=np.float32)
        anchors[:, :, 0] = np.arange(8)[:, None] + np.arange(8)[None] * .5
        with patch('wote_mining.WoTE_model.np.load', return_value=anchors):
            planner = WoTEMiningPlanner(Backbone(), 'synthetic.npy').eval()
        # Force decoded trajectories to differ from anchors: a wrong return
        # of the fixed anchor must not accidentally pass this test.
        with torch.no_grad():
            planner.trajectory_head.offset_head.bias.fill_(.25)
        args = (torch.zeros(1, 3, 8, 8), torch.randn(1, 512, 8, 8),
                torch.ones(1, 1), torch.tensor([[10., 0.]]))
        kw = dict(predict_auxiliary=False)
        with torch.no_grad(), patch.object(planner.world_model, 'forward',
                                           wraps=planner.world_model.forward) as call:
            two = planner(*args, **kw)
            self.assertEqual(call.call_count, 2)
            call.reset_mock()
            one = planner(*args, reuse_anchor_rewards=True, **kw)
            self.assertEqual(call.call_count, 1)
            torch.testing.assert_close(one['trajectories'], two['trajectories'], rtol=0, atol=0)
            winner = one['final_rewards'].argmax(1).item()
            torch.testing.assert_close(one['selected_trajectory'], one['trajectories'][:, winner])
            self.assertFalse(torch.equal(one['selected_trajectory'], one['anchors'][:, winner]))
            training_path = planner(*args, use_fused_world=False, **kw)
            torch.testing.assert_close(one['final_rewards'], training_path['final_rewards'])
            torch.testing.assert_close(training_path['selected_trajectory'], one['anchors'][:, winner])
            ids = torch.tensor([[6, 2, 4]])
            subset = planner(*args, reuse_anchor_rewards=True, world_candidate_indices=ids, **kw)
            chosen = ids[0, subset['selected_index'].item()].item()
            self.assertEqual(subset['selected_anchor_index'].item(), chosen)
            torch.testing.assert_close(subset['selected_trajectory'], one['trajectories'][:, chosen])
        with self.assertRaises(ValueError):
            planner(*args, use_fused_world=False, reuse_anchor_rewards=True, **kw)
        planner.train()
        with self.assertRaises(ValueError):
            planner(*args, reuse_anchor_rewards=True, **kw)


if __name__ == '__main__':
    unittest.main()
