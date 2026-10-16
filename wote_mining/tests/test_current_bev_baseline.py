"""Current-only training and selection without world/reward modules or labels."""
import tempfile
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from wote_mining.WoTE_config import WoTEMiningConfig
from wote_mining.WoTE_loss import source_style_core_losses
from wote_mining.WoTE_model import WoTEMiningPlanner
from wote_mining.WoTE_train import (
    WoTEMiningTrainingModule, build_optimizer, run_epoch, save_checkpoint,
)


class CurrentBEVBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make_module(self):
        config = WoTEMiningConfig(setting='eval')
        config.wote_amp = False
        config.use_target_point_image = True

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.projection = nn.Conv2d(3, 512, 1)
                for name in ('change_channel_conv_image', 'c5_conv', 'up_conv5',
                             'up_conv4', 'up_conv3'):
                    setattr(self, name, nn.Identity())

            def forward(self, rgb, lidar, speed, return_fused_lidar=False):
                return self.projection(lidar)

        anchors = np.zeros((256, 8, 3), dtype=np.float32)
        anchors[:, :, 0] = np.arange(8)[None] * .5 + np.arange(256)[:, None] * .05
        with patch('wote_mining.WoTE_model.np.load', return_value=anchors):
            planner = WoTEMiningPlanner(Backbone(), 'synthetic.npy')
        module = WoTEMiningTrainingModule(planner, config)
        batch = dict(
            rgb=torch.randn(1, 3, 8, 8), lidar=torch.randn(1, 2, 8, 8),
            target_point_image=torch.zeros(1, 1, 8, 8),
            speed=torch.ones(1, 1), target_point=torch.tensor([[10., 0.]]),
            wote_future_poses=torch.from_numpy(anchors[:1].copy()) + .1,
        )
        return module, batch

    def test_no_future_parameters_or_label_dependencies(self):
        module, batch = self.make_module()
        config = module.config
        self.assertFalse(config.wote_future_scene_targets)
        self.assertIsNone(config.wote_metric_cache_dir)
        self.assertIsNone(config.wote_dense_routes_dir)
        self.assertEqual(config.wote_reward_topk_loss_weight, 0)
        names = list(module.state_dict())
        for forbidden in ('world_model', 'reward_head', 'future_', 'current_agent_head'):
            self.assertFalse(any(forbidden in name for name in names), forbidden)
        outputs = module(batch)
        self.assertEqual(outputs['trajectories'].shape, (1, 256, 8, 3))
        losses = module.compute_losses(batch, outputs)
        self.assertEqual(set(losses), {'loss_traj_offset', 'loss_offset_imitation', 'loss_total'})
        losses['loss_total'].backward()
        # Every retained trainable parameter participates (DDP needs no unused search).
        self.assertEqual([name for name, p in module.named_parameters()
                          if p.requires_grad and p.grad is None], [])
        for name in ('offset_head', 'score_head'):
            self.assertGreater(getattr(module.planner.trajectory_head, name).weight.grad.abs().sum(), 0)

    def test_selection_executes_decoded_trajectory_not_fixed_anchor(self):
        module, batch = self.make_module()
        module.eval()
        head = module.planner.trajectory_head
        with torch.no_grad():
            head.offset_head.bias.fill_(.25)
            output = module(batch)
        index = output['scores'].argmax(dim=1)
        torch.testing.assert_close(output['selected_index'], index)
        torch.testing.assert_close(output['selected_trajectory'], output['trajectories'][torch.arange(1), index])
        self.assertFalse(torch.equal(output['selected_trajectory'], output['anchors'][torch.arange(1), index]))
        diagnostics, _ = module.compute_diagnostics(batch, output)
        self.assertIn('traj_selected_ade_m', diagnostics)

    def test_losses_match_existing_wta_and_score_objectives(self):
        module, batch = self.make_module()
        outputs = module(batch)
        outputs['imitation_logits'] = torch.randn_like(outputs['scores'])
        full = source_style_core_losses(outputs, batch['wote_future_poses'])
        baseline = module.compute_losses(batch, outputs)
        torch.testing.assert_close(baseline['loss_traj_offset'], full['loss_traj_offset'])
        torch.testing.assert_close(baseline['loss_offset_imitation'], full['loss_offset_imitation'] * .1)

    def test_train_validation_and_checkpoint_roundtrip(self):
        module, batch = self.make_module()
        optimizer = build_optimizer(module, module.config)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        for training in (True, False):
            result = run_epoch(module, [batch], optimizer, scaler, torch.device('cpu'),
                               module.config, 0, training, False, 1, 0, 5.)
            self.assertTrue(np.isfinite(result['loss_total']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            save_checkpoint(path, module, optimizer, scaler, 0,
                            SimpleNamespace(planner_mode='current_only'), module.config, 1.)
            checkpoint = torch.load(path, map_location='cpu')
        restored, _ = self.make_module()
        restored.load_state_dict(checkpoint['model'], strict=True)
        self.assertEqual(checkpoint['config']['planner_mode'], 'current_only')

    def test_full_mode_can_be_explicitly_restored(self):
        config = WoTEMiningConfig(setting='eval')
        config.set_planner_mode('future_reward')
        self.assertTrue(config.wote_future_scene_targets)
        self.assertIsNotNone(config.wote_metric_cache_dir)
        self.assertEqual(config.wote_reward_topk, 4)

    def test_closed_loop_wrapper_uses_scores_and_zero_world_passes(self):
        # Stub only the CARLA leaderboard base; run the real inference wrapper.
        module, batch = self.make_module()
        base = types.ModuleType('team_code_transfuser.submission_agent')
        base.HybridAgent = object
        path = Path(__file__).resolve().parents[1] / 'WoTE_agent.py'
        spec = importlib.util.spec_from_file_location('current_agent_test', str(path))
        agent_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'team_code_transfuser.submission_agent': base}):
            spec.loader.exec_module(agent_module)
        with patch.dict('os.environ', {'WOTE_WORLD_PASSES': '2'}):
            wrapper = agent_module.WoTEMiningInferenceModel(
                module.config, 'unused.npy', planner=module.planner
            ).eval()
        with torch.no_grad():
            expected = module.eval()(batch)['selected_trajectory'][..., :2]
            actual, _ = wrapper.forward_ego(
                batch['rgb'], batch['lidar'], batch['target_point'],
                batch['target_point_image'], batch['speed'],
            )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(wrapper.last_diagnostics['world_passes'], 0)
        self.assertEqual(wrapper.last_diagnostics['selection_source'], 'score_head')
        self.assertNotIn('selected_reward', wrapper.last_diagnostics)


if __name__ == '__main__':
    unittest.main()
