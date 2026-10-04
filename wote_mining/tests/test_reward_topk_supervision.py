"""CPU regression tests; no CARLA, dataset, or checkpoint download required.

Run: python -m unittest discover -s wote_mining/tests -v
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from wote_mining.WoTE_loss import (
    reward_topk_trajectory_loss,
    select_reward_topk_training_candidates,
)


class RewardTopKSelectionTests(unittest.TestCase):
    def setUp(self):
        self.future = torch.zeros(1, 8, 3)
        self.anchors = torch.zeros(1, 8, 8, 3)
        self.anchors[0, :, :, 0] = torch.arange(8)[:, None].float()
        self.trajectories = self.anchors.clone()
        self.trajectories[0, 1, :, 0] = 0.2
        self.trajectories[0, 2, :, 0] = 0.3
        self.trajectories[0, 3, :, 0] = 0.1
        self.trajectories[0, 3, -1, 0] = 2.1  # ADE passes, endpoint fails.
        self.trajectories[0, 4, :, 0] = 1.1  # Endpoint passes, ADE fails.
        self.trajectories[0, 5, :, 0] = 0.4
        self.trajectories[0, 6, :, 0] = 0.5
        self.trajectories[0, 7, :, 0] = 0.6
        self.rewards = torch.tensor([[100., 1., 4., 99., 98., 3., -float('inf'), 2.]])

    def select(self, **kwargs):
        return select_reward_topk_training_candidates(
            self.rewards, self.anchors, self.trajectories, self.future, **kwargs
        )

    def test_ranks_reward_not_geometry_and_excludes_oracle(self):
        selected = self.select(reward_topk=4)
        self.assertEqual(selected['reward_topk_training_indices'].tolist(), [[2, 5, 7, 1]])
        self.assertEqual(selected['reward_topk_training_valid'].tolist(), [[True] * 4])
        self.assertEqual(selected['reward_topk_oracle_index'].tolist(), [0])
        self.assertEqual(selected['reward_topk_eligible_count'].tolist(), [4])
        self.assertTrue(selected['reward_topk_selected_supervised'].item())

    def test_oracle_uses_fixed_anchor_even_if_decoded_oracle_is_far(self):
        self.trajectories[0, 0, :, 0] = 10.
        selected = self.select()
        self.assertEqual(selected['reward_topk_oracle_index'].item(), 0)
        self.assertNotIn(0, selected['reward_topk_training_indices'][0].tolist())

    def test_insufficient_compatible_candidates_are_masked_not_replaced(self):
        self.trajectories[0, 7, :, 0] = 8.
        selected = self.select()
        valid = selected['reward_topk_training_valid'][0]
        ids = selected['reward_topk_training_indices'][0]
        self.assertEqual(ids[valid].tolist(), [2, 5, 1])
        self.assertEqual(valid.sum().item(), 3)
        self.assertEqual(ids[~valid].tolist(), [0])

    def test_reward_winner_outside_expert_mode_is_not_forced_to_expert(self):
        self.rewards[0, 3] = 101.
        selected = self.select(reward_topk=2)
        self.assertFalse(selected['reward_topk_selected_supervised'].item())
        self.assertNotIn(3, selected['reward_topk_training_indices'][0].tolist())

    def test_nonoracle_reward_winner_is_supervised_when_eligible(self):
        self.rewards[0, 2] = 101.
        self.assertTrue(self.select(reward_topk=1)['reward_topk_selected_supervised'].item())

    def test_nonfinite_reward_and_full_pose_are_ineligible(self):
        self.rewards[0, 6] = float('nan')
        self.trajectories[0, 7, 0, 2] = float('nan')
        selected = self.select()
        valid = selected['reward_topk_training_valid'][0]
        self.assertEqual(selected['reward_topk_training_indices'][0, valid].tolist(), [2, 5, 1])

    def test_zero_topk_and_no_compatible_candidates(self):
        selected = self.select(reward_topk=0)
        self.assertEqual(selected['reward_topk_training_indices'].shape, (1, 0))
        self.trajectories[:, 1:, :, 0] = 10.
        self.assertEqual(self.select()['reward_topk_training_valid'].sum().item(), 0)

    def test_boundary_is_inclusive_and_invalid_settings_raise(self):
        self.trajectories[0, 1, :, 0] = 1.
        selected = self.select(reward_topk=1, endpoint_max_m=1., ade_max_m=1.)
        self.assertEqual(selected['reward_topk_eligible_count'].item(), 4)
        for kwargs in [dict(reward_topk=-1), dict(reward_topk=8),
                       dict(reward_topk=1.5), dict(ade_max_m=0),
                       dict(endpoint_max_m=float('nan')), dict(ade_max_m=float('inf'))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.select(**kwargs)


class RewardTopKLossTests(unittest.TestCase):
    def test_only_valid_extra_candidates_receive_direct_regression_gradients(self):
        trajectories = torch.full((1, 6, 8, 3), 0.5, requires_grad=True)
        anchors = torch.zeros_like(trajectories)
        anchors[:, 1:, :, 0] = 10.
        rewards = torch.tensor([[100., 1., 2., 3., 4., 5.]], requires_grad=True)
        future = torch.zeros(1, 8, 3)
        selection = select_reward_topk_training_candidates(
            rewards, anchors, trajectories, future, reward_topk=2
        )
        loss = reward_topk_trajectory_loss(dict(trajectories=trajectories, **selection), future)
        loss.backward()
        for candidate in range(6):
            grad = trajectories.grad[0, candidate].abs().sum().item()
            self.assertEqual(grad > 0., candidate in (4, 5))
        self.assertIsNone(rewards.grad)
        self.assertTrue(all(not tensor.requires_grad for tensor in selection.values()))

    def test_observations_are_equal_weight_despite_different_valid_counts(self):
        trajectories = torch.full((2, 5, 8, 3), 0.5, requires_grad=True)
        with torch.no_grad():
            trajectories[1].fill_(1.)
        indices = torch.tensor([[1, 2, 3], [1, 0, 0]])
        valid = torch.tensor([[True, True, True], [True, False, False]])
        outputs = dict(trajectories=trajectories, reward_topk_training_indices=indices,
                       reward_topk_training_valid=valid)
        loss = reward_topk_trajectory_loss(outputs, torch.zeros(2, 8, 3))
        # Smooth L1(0.5)=0.125, Smooth L1(1)=0.5; average samples, not slots.
        self.assertAlmostEqual(loss.item(), (0.125 + 0.5) / 2.)

    def test_empty_samples_contribute_zero_to_batch_average(self):
        trajectories = torch.full((2, 3, 8, 3), 0.5, requires_grad=True)
        outputs = dict(trajectories=trajectories,
                       reward_topk_training_indices=torch.tensor([[1], [0]]),
                       reward_topk_training_valid=torch.tensor([[True], [False]]))
        loss = reward_topk_trajectory_loss(outputs, torch.zeros(2, 8, 3))
        self.assertAlmostEqual(loss.item(), 0.125 / 2.)
        loss.backward()
        self.assertEqual(trajectories.grad[1].abs().sum().item(), 0.)

    def test_zero_slots_and_all_invalid_slots_have_connected_zero_loss(self):
        for slots in (0, 4):
            with self.subTest(slots=slots):
                trajectories = torch.randn(2, 5, 8, 3, requires_grad=True)
                outputs = dict(trajectories=trajectories,
                               reward_topk_training_indices=torch.zeros(2, slots, dtype=torch.long),
                               reward_topk_training_valid=torch.zeros(2, slots, dtype=torch.bool))
                loss = reward_topk_trajectory_loss(outputs, torch.zeros(2, 8, 3))
                self.assertEqual(loss.item(), 0.)
                loss.backward()
                self.assertIsNotNone(trajectories.grad)
                self.assertEqual(trajectories.grad.abs().sum().item(), 0.)

    def test_invalid_nonfinite_padding_cannot_poison_loss(self):
        trajectories = torch.full((1, 3, 8, 3), 0.5)
        trajectories[0, 0] = float('nan')
        trajectories.requires_grad_()
        outputs = dict(trajectories=trajectories,
                       reward_topk_training_indices=torch.tensor([[1, 0]]),
                       reward_topk_training_valid=torch.tensor([[True, False]]))
        loss = reward_topk_trajectory_loss(outputs, torch.zeros(1, 8, 3))
        self.assertAlmostEqual(loss.item(), 0.125)
        loss.backward()
        self.assertTrue(torch.isfinite(trajectories.grad).all())
        self.assertEqual(trajectories.grad[0, 0].abs().sum().item(), 0.)

    def test_loss_is_on_final_full_poses_not_anchor_or_pre_fusion_features(self):
        trajectories = torch.zeros(1, 3, 8, 3, requires_grad=True)
        with torch.no_grad():
            trajectories[0, 1, :, 2] = 0.5
        outputs = dict(trajectories=trajectories, anchors=torch.zeros_like(trajectories),
                       reward_topk_training_indices=torch.tensor([[1]]),
                       reward_topk_training_valid=torch.tensor([[True]]))
        loss = reward_topk_trajectory_loss(outputs, torch.zeros(1, 8, 3))
        self.assertAlmostEqual(loss.item(), 0.125 / 3.)
        loss.backward()
        self.assertGreater(trajectories.grad[0, 1, :, 2].abs().sum().item(), 0.)


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Model import stays out of the lightweight selection/loss tests.
        from wote_mining.WoTE_config import WoTEMiningConfig
        from wote_mining.WoTE_model import WoTEMiningPlanner
        from wote_mining.WoTE_train import WoTEMiningTrainingModule
        cls.config_type = WoTEMiningConfig
        cls.planner_type = WoTEMiningPlanner
        cls.training_type = WoTEMiningTrainingModule
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_training_module(self, loss_weight=0.25):
        config = self.config_type(setting='eval')
        config.use_target_point_image = True
        config.wote_reward_topk_loss_weight = loss_weight

        class SyntheticBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.projection = nn.Conv2d(3, 512, 1)
                for name in ('change_channel_conv_image', 'c5_conv', 'up_conv5',
                             'up_conv4', 'up_conv3'):
                    setattr(self, name, nn.Identity())

            def forward(self, rgb, lidar, speed, return_fused_lidar=False):
                return self.projection(lidar)

        anchors = np.zeros((8, 8, 3), dtype=np.float32)
        anchors[:, :, 0] = np.arange(8)[None] * 0.5 + np.arange(8)[:, None] * 0.05
        with patch('wote_mining.WoTE_model.np.load', return_value=anchors):
            planner = self.planner_type(SyntheticBackbone(), 'synthetic.npy')
        module = self.training_type(planner, config)
        batch = dict(
            rgb=torch.randn(1, 3, 8, 8), lidar=torch.randn(1, 2, 8, 8),
            target_point_image=torch.zeros(1, 1, 8, 8),
            speed=torch.ones(1, 1), target_point=torch.tensor([[10., 0.]]),
            wote_future_poses=torch.from_numpy(anchors[:1].copy()),
            wote_augmentation_degrees=torch.zeros(1),
            wote_metric_targets=torch.full((1, 8, 5), 0.5),
            wote_metric_valid=torch.ones(1, 8, 5),
            wote_current_scene=torch.zeros(1, 4, 160, 160),
            wote_current_valid=torch.ones(1, 160, 160),
            wote_current_agent_boxes=torch.zeros(1, 20, 5),
            wote_current_agent_mask=torch.zeros(1, 20, dtype=torch.bool),
            wote_future_scene=torch.zeros(1, 4, 160, 160),
            wote_future_valid=torch.ones(1, 160, 160),
        )
        return module, batch

    def test_full_training_graph_loss_diagnostics_and_all_parameters_backward(self):
        module, batch = self.make_training_module()
        outputs = module(batch)
        losses = module.compute_losses(batch, outputs)
        diagnostics, weights = module.compute_diagnostics(batch, outputs)
        expected_aux = reward_topk_trajectory_loss(outputs, batch['wote_future_poses']) * 0.25
        self.assertTrue(torch.allclose(losses['loss_reward_topk_traj'], expected_aux))
        self.assertEqual(diagnostics['traj_reward_topk_candidates_mean'].item(), 4.)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertTrue(all(torch.isfinite(value) for value in diagnostics.values()))
        self.assertEqual(set(diagnostics), set(weights))
        self.assertFalse(set(diagnostics).intersection(losses))
        losses['loss_total'].backward()
        missing = [name for name, param in module.named_parameters()
                   if param.requires_grad and param.grad is None]
        self.assertEqual(missing, [], 'all DDP parameters must remain in the graph')
        self.assertGreater(module.planner.trajectory_head.offset_head.weight.grad.abs().sum(), 0.)

    def test_disable_restores_original_total_loss_and_inference_is_unchanged(self):
        module, batch = self.make_training_module(loss_weight=0.)
        module.eval()
        outputs = module(batch)
        losses = module.compute_losses(batch, outputs)
        self.assertEqual(outputs['reward_topk_training_indices'].shape, (1, 0))
        self.assertEqual(losses['loss_reward_topk_traj'].item(), 0.)
        original_total = sum(value for name, value in losses.items()
                             if name not in ('loss_total', 'loss_reward_topk_traj'))
        self.assertTrue(torch.equal(losses['loss_total'], original_total))
        diagnostics, _ = module.compute_diagnostics(batch, outputs)
        self.assertEqual(diagnostics['traj_reward_topk_candidates_mean'].item(), 0.)
        with torch.no_grad():
            online = module.planner(
                batch['rgb'], torch.cat((batch['lidar'], batch['target_point_image']), dim=1),
                batch['speed'], batch['target_point'], predict_auxiliary=False,
            )
        self.assertFalse(any(name.startswith('reward_topk_') for name in online))
        self.assertEqual(online['trajectories'].shape, (1, 8, 8, 3))
        self.assertTrue(torch.equal(online['world_trajectories'], online['trajectories']))

    def test_trajectory_losses_stop_at_future_condition_but_train_fusion(self):
        for loss_name in ('loss_traj_offset', 'loss_reward_topk_traj'):
            with self.subTest(loss=loss_name):
                torch.manual_seed(7)
                module, batch = self.make_training_module()
                module.eval()
                head = module.planner.trajectory_head
                with torch.no_grad():
                    # Test after the zero-initialized gate has started learning.
                    head.offset_head.weight.normal_(0., 0.005)
                    head.future_adaln_modulation[-1].weight.normal_(0., 0.01)
                    head.future_adaln_modulation[-1].bias[2 * head.offset_head.in_features:].fill_(0.1)
                outputs = module(batch)
                future = outputs['future_bev_tokens']
                self.assertTrue(future.requires_grad)
                future.retain_grad()
                losses = module.compute_losses(batch, outputs)
                self.assertGreater(losses[loss_name].item(), 0.)
                losses[loss_name].backward()
                self.assertIsNone(future.grad)
                self.assertTrue(all(parameter.grad is None
                                    for parameter in module.planner.world_model.parameters()))
                for layer in (head.offset_head, head.future_adaln_modulation[-1],
                              head.future_update_mlp[-1], module.planner.backbone.projection):
                    self.assertGreater(layer.weight.grad.abs().sum().item(), 0.)
                self.assertGreater(head.future_bev_attention.in_proj_weight.grad.abs().sum().item(), 0.)

    def test_map_and_reward_losses_still_train_world_model(self):
        for loss_name in ('loss_future_map', 'loss_imitation_reward', 'loss_metric_reward'):
            with self.subTest(loss=loss_name):
                torch.manual_seed(7)
                module, batch = self.make_training_module()
                module.eval()
                outputs = module(batch)
                future = outputs['future_bev_tokens']
                future.retain_grad()
                module.compute_losses(batch, outputs)[loss_name].backward()
                self.assertGreater(future.grad.abs().sum().item(), 0.)
                gradient = module.planner.world_model.action_encoder[0].weight.grad
                self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_checkpoint_records_settings_without_adding_model_parameters(self):
        from wote_mining.WoTE_train import save_checkpoint
        module, _ = self.make_training_module()
        optimizer = torch.optim.AdamW(module.parameters())
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        with tempfile.TemporaryDirectory(prefix='wote-topk-test-') as directory:
            checkpoint_path = Path(directory) / 'checkpoint.pth'
            save_checkpoint(checkpoint_path, module, optimizer, scaler, 0,
                            SimpleNamespace(reward_topk=4), module.config, 1.)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.assertEqual(checkpoint['config']['reward_topk_supervision'],
                         dict(topk=4, ade_max_m=1., endpoint_max_m=2.))
        self.assertEqual(checkpoint['config']['loss_weights']['wote_reward_topk_loss_weight'], 0.25)
        self.assertFalse(any('topk' in name for name in checkpoint['model']))
        module.load_state_dict(checkpoint['model'], strict=True)

    def test_cli_defaults_and_overrides(self):
        from wote_mining.WoTE_train import parse_args
        with patch('sys.argv', ['WoTE_train.py', '--root-dir', '/unused']):
            args = parse_args()
        self.assertEqual((args.reward_topk, args.reward_topk_loss_weight,
                          args.reward_topk_ade_max_m, args.reward_topk_endpoint_max_m),
                         (4, 0.25, 1., 2.))
        with patch('sys.argv', ['WoTE_train.py', '--root-dir', '/unused',
                                '--reward-topk', '2', '--reward-topk-loss-weight', '0.1',
                                '--reward-topk-ade-max-m', '0.8',
                                '--reward-topk-endpoint-max-m', '1.5']):
            args = parse_args()
        self.assertEqual((args.reward_topk, args.reward_topk_loss_weight,
                          args.reward_topk_ade_max_m, args.reward_topk_endpoint_max_m),
                         (2, 0.1, 0.8, 1.5))

    def test_resume_loss_signatures_distinguish_changed_and_disabled_objectives(self):
        from wote_mining.WoTE_train import reward_topk_supervision_signature
        signature = reward_topk_supervision_signature
        default = signature(0.25, 4, 1., 2.)
        self.assertNotEqual(default, signature(0., 0, 1., 2.))
        self.assertEqual(default, signature(0.25, 4, 1., 2.))
        for values in [(0.5, 4, 1., 2.), (0.25, 2, 1., 2.),
                       (0.25, 4, 0.5, 2.), (0.25, 4, 1., 1.)]:
            self.assertNotEqual(default, signature(*values))
        self.assertIsNone(signature(0., 4, 1., 2.))
        self.assertIsNone(signature(0.25, 0, 1., 2.))

    def test_invalid_cli_settings_fail_before_dataset_or_distributed_setup(self):
        from wote_mining.WoTE_train import main
        for arguments in [('--reward-topk', '-1'), ('--reward-topk', '256'),
                          ('--reward-topk-loss-weight', '-0.1'),
                          ('--reward-topk-loss-weight', 'nan'),
                          ('--reward-topk-ade-max-m', '0'),
                          ('--reward-topk-endpoint-max-m', 'inf')]:
            with self.subTest(arguments=arguments), patch(
                    'sys.argv', ['WoTE_train.py', '--root-dir', '/unused'] + list(arguments)
            ), self.assertRaises(ValueError):
                main()


class AdaLNGradientTests(unittest.TestCase):
    def test_extra_loss_reaches_modulation_and_future_attention_for_256_candidates(self):
        from wote_mining.WoTE_model import WoTEMiningTrajectoryHead
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(7)
            anchors = np.zeros((256, 8, 3), dtype=np.float32)
            anchors[:, :, 0] = np.arange(256)[:, None] * 0.001
            with patch('wote_mining.WoTE_model.np.load', return_value=anchors):
                head = WoTEMiningTrajectoryHead('synthetic.npy', hidden_dim=32, layers=1, heads=4)
            head.eval()
            with torch.no_grad():
                # Simulate training past the identity-initialized first steps.
                head.offset_head.weight.normal_(0., 0.005)
                head.future_adaln_modulation[-1].weight.normal_(0., 0.01)
                head.future_adaln_modulation[-1].bias[64:].fill_(0.1)
            current = head(torch.randn(1, 512, 8, 8), torch.ones(1, 1), torch.ones(1, 2))
            future = torch.randn(1, 256, 64, 32, requires_grad=True)
            current.update(head.fuse_future_bev(current, future))
            rewards = torch.arange(256)[None].float().requires_grad_()
            expert = torch.zeros(1, 8, 3)
            current.update(select_reward_topk_training_candidates(
                rewards, current['anchors'], current['trajectories'], expert
            ))
            self.assertEqual(current['reward_topk_training_valid'].sum().item(), 4)
            reward_topk_trajectory_loss(current, expert).backward()
            for layer in (head.offset_head, head.future_adaln_modulation[-1],
                          head.future_update_mlp[-1]):
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0.)
            self.assertGreater(head.future_bev_attention.in_proj_weight.grad.abs().sum().item(), 0.)
            self.assertIsNone(future.grad)
            self.assertIsNone(rewards.grad)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
