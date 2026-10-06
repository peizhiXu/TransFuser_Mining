"""CPU tests for online/proxy selection, strict legacy loading and evaluation."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from wote_mining.WoTE_inference_eval import (
    AUXILIARY_KEYS, compare_selections, evaluate, load_planner_weights,
    snapshot, summarize_records, trajectory_metrics,
)


class InferenceMetricTests(unittest.TestCase):
    def make_outputs(self):
        expert = torch.zeros(2, 8, 3)
        expert[:, :, 0] = torch.arange(1, 9).float() * 0.5
        bank = expert[:, None].repeat(1, 3, 1, 1)
        bank[:, 1, :, 1] = 1.0
        bank[:, 2, :, 1] = 3.0
        fixed = dict(trajectories=bank, final_rewards=torch.tensor([[0., 2., 1.]]).repeat(2, 1),
                     selected_index=torch.tensor([1, 1]), selected_trajectory=torch.full_like(expert, 100.))
        online = dict(trajectories=bank, final_rewards=torch.tensor([[0., 1., 2.]]).repeat(2, 1),
                      selected_index=torch.tensor([2, 2]), selected_trajectory=bank[:, 2])
        return fixed, online, expert

    def test_proxy_gathers_decoded_bank_not_fixed_selected_trajectory(self):
        fixed, online, expert = self.make_outputs()
        values = compare_selections(fixed, online, expert)
        self.assertEqual(values['proxy_ade_m'].tolist(), [1., 1.])
        self.assertEqual(values['online_ade_m'].tolist(), [3., 3.])
        self.assertEqual(values['oracle_min_ade_m'].tolist(), [0., 0.])
        self.assertEqual(values['online_selection_regret_ade_m'].tolist(), [3., 3.])
        self.assertEqual(values['online_minus_proxy_ade_m'].tolist(), [2., 2.])
        self.assertEqual(values['selection_agreement'].tolist(), [False, False])
        self.assertEqual(values['online_oracle_reward_rank'].tolist(), [3, 3])
        self.assertEqual(values['online_reward_top4_min_ade_m'].tolist(), [0., 0.])

    def test_oracle_by_ade_fde_is_distinct_from_min_fde(self):
        fixed, online, expert = self.make_outputs()
        fixed['trajectories'][:, 0, -1, 1] = 0.5
        fixed['trajectories'][:, 1, -1, 1] = 0.0
        values = compare_selections(fixed, online, expert)
        self.assertEqual(values['oracle_by_ade_fde_m'].tolist(), [0.5, 0.5])
        self.assertEqual(values['oracle_min_fde_m'].tolist(), [0., 0.])

    def test_changed_bank_or_incorrect_execution_or_selection_is_rejected(self):
        for issue in ('bank', 'selected', 'reward', 'nonfinite'):
            fixed, online, expert = self.make_outputs()
            if issue == 'bank':
                fixed['trajectories'] = fixed['trajectories'].clone() + 0.1
            elif issue == 'selected':
                online['selected_trajectory'] = online['selected_trajectory'] + 0.1
            elif issue == 'reward':
                online['selected_index'] = torch.tensor([0, 0])
            else:
                online['final_rewards'][0, 0] = float('nan')
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                compare_selections(fixed, online, expert)

    def test_speed_uses_points_zero_and_one_at_two_hz_and_direction_wraps(self):
        truth, predicted = torch.zeros(1, 8, 3), torch.zeros(1, 8, 3)
        angle = np.deg2rad(179.)
        truth[0, 1, :2] = torch.tensor([np.cos(angle), np.sin(angle)])
        predicted[0, 1, :2] = torch.tensor([np.cos(-angle), np.sin(-angle)])
        values = trajectory_metrics(predicted, truth)
        self.assertAlmostEqual(values['desired_speed_mps'].item(), 2., places=5)
        self.assertAlmostEqual(values['first_segment_direction_error_deg'].item(), 2., places=4)

    def test_stationary_direction_is_excluded_not_a_false_zero_error(self):
        truth, predicted = torch.zeros(1, 8, 3), torch.zeros(1, 8, 3)
        truth[:, 1, 0] = 1.
        values = trajectory_metrics(predicted, truth)
        self.assertFalse(values['direction_valid'].item())
        self.assertTrue(torch.isnan(values['first_segment_direction_error_deg']).item())
        self.assertEqual(values['desired_speed_abs_error_mps'].item(), 2.)

    def test_summary_weights_samples_and_excludes_undefined_direction(self):
        summary = summarize_records([
            dict(sample_index=0, route='a', frame='0', online_ade_m=1., direction=None),
            dict(sample_index=1, route='a', frame='1', online_ade_m=1., direction=10.),
            dict(sample_index=2, route='b', frame='0', online_ade_m=4., direction=20.),
        ])
        self.assertEqual(summary['means']['online_ade_m'], 2.)
        self.assertEqual(summary['means']['direction'], 15.)
        self.assertEqual(summary['valid_counts']['direction'], 2)
        self.assertEqual(summary['samples'], 3)
        with self.assertRaises(ValueError):
            summarize_records([])


class CheckpointCompatibilityTests(unittest.TestCase):
    def make_planner(self):
        planner = nn.Module()
        planner.trajectory_head = nn.Module()
        planner.trajectory_head.register_buffer('anchors', torch.zeros(3, 8, 3))
        planner.trajectory_head.future_read_metric_head = nn.Linear(4, 3)
        planner.trajectory_head.offset_head = nn.Linear(4, 24)
        return planner

    def state(self, planner):
        return {'planner.' + key: value.clone() for key, value in planner.state_dict().items()}

    def test_current_strict_and_legacy_auxiliary_only_loading(self):
        planner = self.make_planner()
        state = self.state(planner)
        self.assertEqual(load_planner_weights(planner, state), [])
        for key in AUXILIARY_KEYS:
            del state['planner.' + key]
        self.assertEqual(set(load_planner_weights(planner, state)), AUXILIARY_KEYS)
        self.assertEqual(planner.trajectory_head.future_read_metric_head.weight.abs().sum().item(), 0.)

    def test_rejects_unrelated_missing_extra_wrong_shape_and_anchor_changes(self):
        for issue in ('missing', 'extra', 'shape', 'anchors', 'partial_aux', 'prefix'):
            planner = self.make_planner()
            state = self.state(planner)
            before = self.state(planner)
            if issue == 'missing':
                del state['planner.trajectory_head.offset_head.weight']
            elif issue == 'extra':
                state['planner.unexpected'] = torch.zeros(1)
            elif issue == 'shape':
                state['planner.trajectory_head.offset_head.weight'] = torch.zeros(5)
            elif issue == 'anchors':
                state['planner.trajectory_head.anchors'] += 1.
            elif issue == 'partial_aux':
                del state['planner.trajectory_head.future_read_metric_head.weight']
            else:
                state['wrong_prefix'] = torch.zeros(1)
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                load_planner_weights(planner, state)
            self.assertTrue(all(torch.equal(before[key], value) for key, value in self.state(planner).items()))


class InferencePlannerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_planner(self, candidates=8):
        from wote_mining.WoTE_model import WoTEMiningPlanner

        class SyntheticBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_target_point_image=True, lidar_pos=[1.3, 0., 2.5])
                self.projection = nn.Conv2d(3, 512, 1)
                for name in ('change_channel_conv_image', 'c5_conv', 'up_conv5', 'up_conv4', 'up_conv3'):
                    setattr(self, name, nn.Identity())

            def forward(self, rgb, lidar, speed, return_fused_lidar=False):
                return self.projection(lidar)

        anchors = np.zeros((candidates, 8, 3), dtype=np.float32)
        anchors[:, :, 0] = np.arange(8)[None] * 0.5 + np.arange(candidates)[:, None] * 0.05
        with patch('wote_mining.WoTE_model.np.load', return_value=anchors):
            planner = WoTEMiningPlanner(SyntheticBackbone(), 'synthetic.npy')
        with torch.no_grad():
            planner.trajectory_head.offset_head.weight.normal_(std=0.01)
        batch = dict(rgb=torch.randn(1, 3, 8, 8), lidar=torch.randn(1, 2, 8, 8),
                     target_point_image=torch.zeros(1, 1, 8, 8), speed=torch.ones(1),
                     target_point=torch.tensor([[10., 0.]]),
                     wote_future_poses=torch.from_numpy(anchors[:1].copy()))
        return planner.eval(), batch

    def test_real_planner_runs_both_paths_deterministically_without_changing_weights(self):
        torch.manual_seed(12)
        planner, batch = self.make_planner()
        original = {key: value.clone() for key, value in planner.state_dict().items()}
        fixed = snapshot(planner, batch, torch.device('cpu'), False, False)
        online = snapshot(planner, batch, torch.device('cpu'), True, False)
        again = snapshot(planner, batch, torch.device('cpu'), True, False)
        self.assertTrue(torch.equal(online['final_rewards'], again['final_rewards']))
        values = compare_selections(fixed, online, batch['wote_future_poses'])
        self.assertEqual(values['online_ade_m'].shape, (1,))
        # A nonzero decoder produces genuine final-trajectory reranking inputs.
        self.assertFalse(torch.equal(fixed['final_rewards'], online['final_rewards']))
        self.assertTrue(all(torch.equal(original[key], value) for key, value in planner.state_dict().items()))
        records = evaluate(planner, [batch], [dict(sample_index=0, route='synthetic', frame='0002')],
                           torch.device('cpu'), False)
        self.assertEqual(summarize_records(records)['samples'], 1)

    def test_cli_compares_legacy_and_current_on_identical_sorted_samples(self):
        from wote_mining import WoTE_inference_eval as evaluator
        from torch.utils.data import Dataset
        planner, batch = self.make_planner()
        state = {'planner.' + key: value.clone() for key, value in planner.state_dict().items()}
        legacy = {key: value for key, value in state.items()
                  if key[len('planner.'):] not in AUXILIARY_KEYS}

        class SyntheticDataset(Dataset):
            lidars = np.array([[b'/synthetic/z/lidar/0004.npy'],
                               [b'/synthetic/a/lidar/0003.npy'],
                               [b'/synthetic/a/lidar/0002.npy']])

            def __len__(self):
                return 3

            def __getitem__(self, index):
                return {key: value[0] for key, value in batch.items()}

        def build(config, saved_args, saved_state, anchors):
            model, _ = self.make_planner()
            missing = load_planner_weights(model, saved_state)
            return model, missing

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'val/route/lidar').mkdir(parents=True)
            for name, weights in (('old', legacy), ('new', state)):
                torch.save(dict(epoch=30, model=weights, args={}, config={}), str(root / (name + '.pth')))
            args = SimpleNamespace(root_dir=str(root), checkpoint=[
                'old=' + str(root / 'old.pth'), 'new=' + str(root / 'new.pth')],
                output_dir=str(root / 'evaluation'), anchors=None, batch_size=2,
                workers=0, device='cpu', amp=False, seed=2026, max_samples=None)
            with patch.object(evaluator, 'parse_args', return_value=args), \
                    patch.object(evaluator, 'build_planner', side_effect=build), \
                    patch('team_code_transfuser.data.CARLA_Data', return_value=SyntheticDataset()), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                evaluator.main()
                with self.assertRaises(FileExistsError):
                    evaluator.main()
            comparison = json.loads((root / 'evaluation/comparison.json').read_text())
            metadata = json.loads((root / 'evaluation/samples.json').read_text())
            self.assertEqual([row['sample_index'] for row in metadata], [2, 1, 0])
            self.assertEqual(comparison['runs']['new']['samples'], 3)
            self.assertEqual(len(comparison['runs']['new']['by_route']), 2)
            self.assertEqual(comparison['runs']['old']['legacy_auxiliary_keys'], sorted(AUXILIARY_KEYS))
            self.assertEqual(comparison['runs']['new']['legacy_auxiliary_keys'], [])
            self.assertTrue(all(value == 0. for value in comparison['mean_differences_vs_reference']['new'].values()))
            old_rows = [json.loads(line) for line in (root / 'evaluation/old_samples.jsonl').read_text().splitlines()]
            new_rows = [json.loads(line) for line in (root / 'evaluation/new_samples.jsonl').read_text().splitlines()]
            self.assertEqual(old_rows, new_rows)

    def test_full_256_candidates_online_and_proxy(self):
        planner, batch = self.make_planner(candidates=256)
        fixed = snapshot(planner, batch, torch.device('cpu'), False, False)
        online = snapshot(planner, batch, torch.device('cpu'), True, False)
        self.assertEqual(online['trajectories'].shape, (1, 256, 8, 3))
        values = compare_selections(fixed, online, batch['wote_future_poses'])
        self.assertGreaterEqual(values['online_oracle_reward_rank'].item(), 1)
        self.assertLessEqual(values['online_oracle_reward_rank'].item(), 256)

    def test_constructor_disables_pretrained_downloads_only_during_build(self):
        from team_code_transfuser import transfuser
        from wote_mining.WoTE_inference_eval import build_planner
        planner, _ = self.make_planner()
        saved = {'planner.' + key: value for key, value in planner.state_dict().items()}
        anchors = planner.trajectory_head.anchors.numpy()
        pretrained_flags = []

        def create_model(name, pretrained):
            pretrained_flags.append(pretrained)
            return nn.Identity()

        def construct(config, **kwargs):
            # Emulate the image encoder's usual request for pretrained weights.
            transfuser.timm.create_model('synthetic', pretrained=True)
            return planner.backbone

        with patch.object(transfuser.timm, 'create_model', side_effect=create_model) as factory, \
                patch.object(transfuser, 'TransfuserBackbone', side_effect=construct), \
                patch('wote_mining.WoTE_model.np.load', return_value=anchors):
            rebuilt, missing = build_planner(planner.backbone.config, {}, saved)
            self.assertEqual(pretrained_flags, [False])
            self.assertEqual(missing, [])
            self.assertIs(transfuser.timm.create_model, factory)
            self.assertTrue(all(torch.equal(value, rebuilt.state_dict()[key[len('planner.'):]])
                                for key, value in saved.items()))


if __name__ == '__main__':
    unittest.main()
