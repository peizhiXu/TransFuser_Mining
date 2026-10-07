"""Stable future conditions, auxiliary label alignment, and gradient bounds."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from wote_mining.WoTE_loss import future_read_metric_loss
from wote_mining.tests import test_reward_topk_supervision as fixtures


class FutureReadLossTests(unittest.TestCase):
    def test_fixed_anchor_labels_independent_metric_normalization_and_masking(self):
        logits = torch.tensor([[[1., 2., 3.], [-1., -2., -3.]]], requires_grad=True)
        targets = torch.tensor([[[0., 1., 0.25, 0., 0.],
                                 [1., 0., 0.75, 1., 1.]]])
        valid = torch.ones_like(targets)
        valid[0, 1, 1] = 0.
        loss = future_read_metric_loss(logits, targets, valid)
        elementwise = F.binary_cross_entropy_with_logits(
            logits, targets[..., :3], reduction='none'
        )
        expected = elementwise[..., 0].mean() + elementwise[0, 0, 1] + elementwise[..., 2].mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertEqual(logits.grad[0, 1, 1].item(), 0.)
        self.assertTrue((logits.grad[valid[..., :3].bool()].abs() > 0).all())
        # TTC/comfort are deliberately not auxiliary targets.
        targets[..., 3:] = float('nan')
        self.assertTrue(torch.equal(future_read_metric_loss(logits, targets, valid), loss))

    def test_all_missing_and_nonfinite_masked_entries_have_connected_zero(self):
        logits = torch.full((1, 4, 3), float('nan'), requires_grad=True)
        targets = torch.full((1, 4, 5), float('nan'))
        loss = future_read_metric_loss(logits, targets, torch.zeros_like(targets))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_invalid_shapes_values_and_validity_are_rejected(self):
        logits, targets = torch.zeros(1, 2, 3), torch.zeros(1, 2, 5)
        for target, valid in ((targets[..., :3], torch.ones_like(targets)),
                              (targets + 2., torch.ones_like(targets)),
                              (targets, torch.full_like(targets, float('nan'))),
                              (targets, torch.full_like(targets, -1.))):
            with self.subTest(), self.assertRaises(ValueError):
                future_read_metric_loss(logits, target, valid)


class StableFutureReadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TrainingIntegrationTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.TrainingIntegrationTests.tearDownClass()

    def make_module(self):
        module, batch = fixtures.TrainingIntegrationTests().make_training_module()
        # This class exercises the optional stable-read ablation explicitly;
        # the production defaults now use the simpler residual recipe.
        module.config.wote_stable_future_condition = True
        module.config.wote_future_read_loss_weight = 0.1
        module.config.wote_decoded_imitation_loss_weight = 0.0
        module.planner.stable_future_condition = True
        return module, batch

    def head_outputs(self, module, batch):
        return module.planner.trajectory_head(
            torch.randn(1, 512, 8, 8), batch['speed'], batch['target_point']
        )

    def test_condition_ignores_rng_and_action_encoding_dropout(self):
        module, batch = self.make_module()
        planner = module.planner
        outputs = self.head_outputs(module, batch)
        # Identical candidate actions must have identical scene predictions.
        outputs['anchors'] = outputs['anchors'][:, :1].expand(-1, 8, -1, -1)
        conditions = []
        for seed in (7, 31):
            torch.manual_seed(seed)
            conditions.append(planner.planning_future_condition(
                outputs, batch['speed'], batch['target_point']
            ))
        for key in ('future_scene_tokens', 'current_scene_tokens'):
            self.assertTrue(torch.equal(conditions[0][key], conditions[1][key]))
            self.assertFalse(conditions[0][key].requires_grad)
        future = conditions[0]['future_scene_tokens']
        self.assertTrue(torch.allclose(future[:, :1].expand_as(future), future,
                                       atol=1e-6, rtol=1e-6))
        self.assertLess(future.std(dim=1, unbiased=False).mean().item(), 1e-6)
        # The normal differentiable reward/map pass remains stochastic.
        actions = planner.trajectory_head.encode_trajectory_features(
            outputs['anchors'], batch['speed'], batch['target_point']
        )
        normal = planner.world_model(outputs['bev_tokens'], actions, outputs['anchors'])
        self.assertTrue(normal['future_scene_tokens'].requires_grad)
        self.assertGreater(normal['future_scene_tokens'].std(dim=1).mean().item(), 0.01)

    def test_nested_modes_restored_after_success_and_exception(self):
        module, batch = self.make_module()
        planner = module.planner
        # Preserve an intentionally mixed subtree, not just root.train().
        planner.world_model.transition.layers[0].dropout.eval()
        outputs = self.head_outputs(module, batch)
        modes = [(part, part.training) for part in planner.modules()]
        planner.planning_future_condition(outputs, batch['speed'], batch['target_point'])
        self.assertTrue(all(part.training == state for part, state in modes))
        with patch.object(planner.world_model, 'forward', side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                planner.planning_future_condition(outputs, batch['speed'], batch['target_point'])
        self.assertTrue(all(part.training == state for part, state in modes))
        self.assertTrue(torch.is_grad_enabled())

    def test_training_adds_one_condition_pass_but_eval_and_inference_do_not(self):
        module, batch = self.make_module()
        planner = module.planner
        calls = []
        handle = planner.world_model.register_forward_pre_hook(
            lambda world, args: calls.append((world.transition.training, torch.is_grad_enabled()))
        )
        try:
            module(batch)
            self.assertEqual(calls, [(True, True), (False, False)])
            self.assertTrue(planner.world_model.transition.training)
            calls.clear()
            planner.stable_future_condition = False
            module(batch)
            self.assertEqual(calls, [(True, True)])
            calls.clear()
            planner.stable_future_condition = True
            module.eval()
            with torch.no_grad():
                module(batch)
            self.assertEqual(calls, [(False, False)])
            calls.clear()
            with torch.no_grad():
                online = planner(
                    batch['rgb'], torch.cat((batch['lidar'], batch['target_point_image']), dim=1),
                    batch['speed'], batch['target_point'], predict_auxiliary=False,
                )
            self.assertEqual(calls, [(False, False), (False, False)])
            self.assertNotIn('future_read_metric_logits', online)
            self.assertTrue(torch.equal(online['trajectories'], online['world_trajectories']))
        finally:
            handle.remove()

    def test_auxiliary_trains_reader_not_world_or_adaln_or_reward(self):
        module, batch = self.make_module()
        outputs = module(batch)
        head = module.planner.trajectory_head
        self.assertEqual(outputs['future_read_metric_logits'].shape, (1, 8, 3))
        loss = module.compute_losses(batch, outputs)['loss_future_read_metrics']
        loss.backward()
        for layer in (head.future_read_metric_head, head.anchor_encoder[0],
                      head.query_fusion[0]):
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(head.future_bev_attention.in_proj_weight.grad.abs().sum().item(), 0.)
        for branch in (module.planner.world_model, module.planner.reward_head,
                       head.future_adaln_modulation, head.future_update_mlp, head.offset_head):
            self.assertTrue(all(parameter.grad is None for parameter in branch.parameters()))

    def test_zero_weight_or_missing_labels_keeps_all_training_parameters_connected(self):
        for disabled in ('weight', 'labels'):
            with self.subTest(disabled=disabled):
                module, batch = self.make_module()
                if disabled == 'weight':
                    module.config.wote_future_read_loss_weight = 0.
                else:
                    batch['wote_metric_valid'].zero_()
                outputs = module(batch)
                losses = module.compute_losses(batch, outputs)
                self.assertEqual(losses['loss_future_read_metrics'].item(), 0.)
                losses['loss_total'].backward()
                for name, parameter in module.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertEqual(module.planner.trajectory_head.future_read_metric_head.weight.grad.abs().sum(), 0.)

    def test_full_256_candidate_training_forward_and_backward(self):
        module, batch = self.make_module()
        head = module.planner.trajectory_head
        head.anchors = head.anchors.repeat(32, 1, 1)
        head.anchor_attention_mask = torch.zeros(256, 256, dtype=torch.bool)
        for name in ('wote_metric_targets', 'wote_metric_valid'):
            batch[name] = batch[name].repeat(1, 32, 1)
        outputs = module(batch)
        self.assertEqual(outputs['trajectories'].shape, (1, 256, 8, 3))
        self.assertEqual(outputs['future_read_metric_logits'].shape, (1, 256, 3))
        self.assertEqual(sum(parameter.numel() for parameter in
                             head.future_read_metric_head.parameters()), 771)
        self.assertTrue(torch.equal(outputs['trajectories'], outputs['anchors']))
        losses = module.compute_losses(batch, outputs)
        losses['loss_total'].backward()
        self.assertGreater(head.future_read_metric_head.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(module.planner.world_model.action_encoder[0].weight.grad.abs().sum().item(), 0.)
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_diagnostics_valid_denominators_and_actual_condition_spread(self):
        module, batch = self.make_module()
        batch['wote_metric_valid'][0, 1:, 0] = 0.
        batch['wote_metric_targets'][0, 0, 0] = 0.
        seen = []
        handle = module.planner.world_model.register_forward_hook(
            lambda world, args, output: seen.append(output)
        )
        try:
            outputs = module(batch)
        finally:
            handle.remove()
        diagnostics, weights = module.compute_diagnostics(batch, outputs)
        self.assertEqual(weights['future_read_no_collision_bce'].item(), 1.)
        self.assertEqual(weights['future_read_no_collision_unsafe_mae'].item(), 1.)
        self.assertEqual(diagnostics['future_read_no_collision_unsafe_fraction'].item(), 1.)
        expected = seen[1]['future_scene_tokens'].float().std(
            dim=1, unbiased=False
        ).mean(dim=(-1, -2))
        self.assertTrue(torch.equal(outputs['future_scene_candidate_std'], expected))
        for name in ('future_scene_candidate_std', 'future_delta_candidate_std',
                     'future_read_candidate_std', 'future_read_ego_progress_bce'):
            self.assertTrue(math.isfinite(diagnostics[name].item()), name)
            self.assertGreater(weights[name].item(), 0.)

    def test_auxiliary_head_does_not_change_inference_selection(self):
        module, batch = self.make_module()
        module.eval()
        planner = module.planner
        args = (batch['rgb'], torch.cat((batch['lidar'], batch['target_point_image']), dim=1),
                batch['speed'], batch['target_point'])
        with torch.no_grad():
            with_aux = planner(*args, predict_auxiliary=True)
            without_aux = planner(*args, predict_auxiliary=False)
        for key in ('trajectories', 'final_rewards', 'selected_index', 'selected_trajectory'):
            self.assertTrue(torch.equal(with_aux[key], without_aux[key]), key)

    def test_cli_and_checkpoint_record_new_training_definition(self):
        from wote_mining.WoTE_train import main, parse_args, save_checkpoint
        with patch('sys.argv', ['train', '--root-dir', '/unused']):
            args = parse_args()
        self.assertEqual(args.future_read_loss_weight, 0.)
        self.assertEqual(args.decoded_imitation_loss_weight, 0.25)
        self.assertFalse(args.stable_future_condition)
        with patch('sys.argv', ['train', '--root-dir', '/unused',
                                '--future-read-loss-weight', '0.1',
                                '--decoded-imitation-loss-weight', '0',
                                '--stable-future-condition']):
            args = parse_args()
        self.assertEqual(args.future_read_loss_weight, 0.1)
        self.assertEqual(args.decoded_imitation_loss_weight, 0.)
        self.assertTrue(args.stable_future_condition)
        for weight in ('-1', 'nan', 'inf'):
            with patch('sys.argv', ['train', '--root-dir', '/unused',
                                    '--future-read-loss-weight', weight]), self.assertRaises(ValueError):
                main()
        module, _ = self.make_module()
        with tempfile.TemporaryDirectory(prefix='wote-future-read-') as directory:
            path = Path(directory) / 'checkpoint.pth'
            save_checkpoint(path, module, torch.optim.AdamW(module.parameters()),
                            torch.cuda.amp.GradScaler(enabled=False), 0,
                            SimpleNamespace(), module.config, 1.)
            checkpoint = torch.load(path, map_location='cpu')
        self.assertTrue(checkpoint['config']['stable_future_condition'])
        self.assertEqual(checkpoint['config']['loss_weights']['wote_future_read_loss_weight'], 0.1)
        self.assertEqual(
            checkpoint['config']['loss_weights']['wote_decoded_imitation_loss_weight'],
            0.0,
        )
        module.load_state_dict(checkpoint['model'], strict=True)


if __name__ == '__main__':
    unittest.main()
