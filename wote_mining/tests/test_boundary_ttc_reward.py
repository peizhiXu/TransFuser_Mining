"""Five named mining heads: retain margin, append independently supervised TTC."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from wote_mining.WoTE_model import WoTEMiningRewardHead
from wote_mining.WoTE_reward_schema import (
    BOUNDARY_TTC_METRIC_NAMES, BOUNDARY_LABEL_VERSION, TTC_LABEL_VERSION,
    metric_names_for_scheme, restore_reward_settings, reward_settings,
)
from wote_mining.WoTE_simulator import evaluate_mining_candidates
from wote_mining.WoTE_targets import metric_cache_file_for_collection, load_metric_cache_shard


class MiningTTCRewardTests(unittest.TestCase):
    def test_append_ttc_preserves_old_formula_and_rewards_safe_candidate(self):
        old = WoTEMiningRewardHead(hidden_dim=8, reward_scheme='mining_boundary')
        new = WoTEMiningRewardHead(hidden_dim=8, reward_scheme='mining_boundary_ttc')
        imitation = torch.zeros(1, 2)
        logits = torch.zeros(1, 2, 5, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 4] = -8
            logits[0, 1, 4] = 8
        baseline = old.compose_rewards(imitation, logits[..., :4])[0]
        result, _, scores = new.compose_rewards(imitation, logits)
        torch.testing.assert_close(result, baseline + .2 * logits[..., 4].sigmoid())
        self.assertGreater(result[0, 1], result[0, 0])
        self.assertLessEqual(float(result[0, 1]-result[0, 0]), .2)
        result.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[..., 4] > 0).all())
        disabled = WoTEMiningRewardHead(hidden_dim=8, reward_scheme='mining_boundary_ttc', ttc_reward_weight=0)
        torch.testing.assert_close(disabled.compose_rewards(imitation, logits)[0], baseline)
        self.assertEqual(scores.shape[-1], 5)
        self.assertEqual(len(new.metric_heads), 5)
        self.assertEqual(metric_names_for_scheme('mining_boundary_ttc'), BOUNDARY_TTC_METRIC_NAMES)
        for weight in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                WoTEMiningRewardHead(hidden_dim=8, ttc_reward_weight=weight)

    def test_forward_and_metric_supervision_train_ttc_head(self):
        from wote_mining.WoTE_loss import source_style_core_losses
        head = WoTEMiningRewardHead(hidden_dim=8, reward_scheme='mining_boundary_ttc')
        output = head(torch.randn(2, 64, 8), torch.randn(2, 3, 64, 8),
                      torch.randn(2, 3, 8), torch.randn(2, 3, 8))
        self.assertEqual(output['metric_names'], BOUNDARY_TTC_METRIC_NAMES)
        self.assertEqual(output['metric_logits'].shape, (2, 3, 5))
        targets = torch.zeros_like(output['metric_logits'])
        valid = torch.zeros_like(targets)
        valid[..., 4] = 1
        output.update(anchors=torch.zeros(2, 3, 8, 3),
                      offsets=torch.zeros(2, 3, 8, 3),
                      scores=torch.zeros(2, 3))
        losses = source_style_core_losses(output, torch.zeros(2, 8, 3), targets, valid)
        losses['loss_metric_reward'].backward()
        grad = head.metric_heads[4][-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_labels_retain_nc_dac_progress_margin_and_source_ttc_mask(self):
        road = np.ones((500, 500), dtype=np.uint8)
        route = np.stack((np.arange(0, 90), np.zeros(90)), axis=-1)
        poses = np.zeros((2, 8, 3), dtype=np.float32)
        poses[..., 0] = np.arange(1, 9)
        # No agents, but guaranteed recording coverage supplies valid labels.
        tracks = dict(boxes=np.zeros((9, 0, 5), np.float32),
                      mask=np.zeros((9, 0), bool),
                      recorded_ego=np.zeros((9, 3), np.float32), recording_radius_m=50.)
        for recorded in [None, tracks]:
            source = evaluate_mining_candidates(poses, road, route, recorded_tracks=recorded)
            old = evaluate_mining_candidates(poses, road, route, recorded_tracks=recorded,
                                            reward_scheme='mining_boundary')
            new = evaluate_mining_candidates(poses, road, route, recorded_tracks=recorded,
                                            reward_scheme='mining_boundary_ttc')
            for key in ['metric_targets', 'metric_valid']:
                np.testing.assert_array_equal(new[key][:, :4], old[key])
                np.testing.assert_array_equal(new[key][:, 4], source[key][:, 3])
            if recorded is None:
                self.assertFalse(new['metric_valid'][:, 4].any())
            else:
                self.assertTrue(new['metric_valid'][:, 4].all())

    def test_checkpoint_settings_preserve_weight_and_reject_wrong_versions(self):
        config = SimpleNamespace(wote_reward_scheme='mining_boundary_ttc', wote_ttc_reward_weight=.3)
        saved = reward_settings(config)
        restored = SimpleNamespace()
        restore_reward_settings(restored, {}, {'reward_scheme_settings': saved})
        self.assertEqual(reward_settings(restored), saved)
        self.assertEqual(restored.wote_ttc_reward_weight, .3)
        for key in ['label_version', 'ttc_label_version']:
            bad = dict(saved, **{key: 'wrong'})
            with self.assertRaises(ValueError):
                restore_reward_settings(restored, {}, {'reward_scheme_settings': bad})
        restore_reward_settings(restored, {})
        self.assertEqual(restored.wote_reward_scheme, 'source')

    def test_ttc_can_flag_extrapolation_risk_without_actual_candidate_collision(self):
        from wote_mining.WoTE_simulator import evaluate_recorded_agent_overlap
        # Synthetic geometry deliberately separates planned displacement from
        # instantaneous heading, so NC is safe but heading extrapolation is not.
        poses = np.zeros((1, 8, 3), np.float32)
        poses[0, :, 0] = np.arange(1, 9)
        poses[0, :, 2] = np.pi/2
        boxes = np.zeros((9, 1, 5), np.float32)
        boxes[:] = [7, 7, 0, 2, 2]
        tracks = dict(boxes=boxes, mask=np.ones((9, 1), bool),
                      recorded_ego=np.zeros((9, 3), np.float32), recording_radius_m=50.)
        result = evaluate_recorded_agent_overlap(poses, tracks)
        self.assertEqual(result['no_collision'][0], 1)
        self.assertEqual(result['ttc_within_bound'][0], 0)
        self.assertEqual(result['ttc_valid'][0], 1)

    def test_ttc_telemetry_is_named_not_confused_with_margin_or_comfort(self):
        from wote_mining.WoTE_diagnostics import reward_topk_diagnostics
        outputs = dict(final_rewards=torch.tensor([[0., 1.]]), selected_index=torch.tensor([1]),
                       world_trajectories=torch.zeros(1, 2, 8, 3),
                       metric_scores=torch.tensor([[[1., 1., 1., .3, .7], [1., 1., 1., .4, .8]]]),
                       metric_names=BOUNDARY_TTC_METRIC_NAMES)
        scores = reward_topk_diagnostics(outputs)['reward_topk_candidates'][0]['metric_scores']
        self.assertEqual(tuple(scores), BOUNDARY_TTC_METRIC_NAMES)
        self.assertAlmostEqual(scores['road_boundary_margin'], .4, places=5)
        self.assertAlmostEqual(scores['time_to_collision'], .8, places=5)

    def test_cache_is_separate_and_preflight_requires_valid_ttc_labels(self):
        from wote_mining.WoTE_train import validate_reward_cache
        with tempfile.TemporaryDirectory(prefix='wote-ttc-') as temp:
            root = Path(temp)
            anchors = root/'anchors.npy'
            np.save(anchors, np.zeros((256, 8, 3), np.float32))
            manifest = dict(metric_names=list(BOUNDARY_TTC_METRIC_NAMES),
                            boundary_label_settings=dict(version=BOUNDARY_LABEL_VERSION, target_m=.6),
                            ttc_label_version=TTC_LABEL_VERSION, valid_counts=[256]*5,
                            anchors_sha256=hashlib.sha256(anchors.read_bytes()).hexdigest(),
                            routes={'route': {'file': 'route.npz'}})
            for split in ['train', 'val']:
                folder = root/split
                folder.mkdir()
                (folder/'manifest.json').write_text(json.dumps(manifest))
                np.savez(folder/'route.npz', frames=np.array([1]),
                         metric_targets=np.ones((1, 256, 5)), metric_valid=np.ones((1, 256, 5)))
            config = SimpleNamespace(wote_reward_scheme='mining_boundary_ttc',
                                     wote_boundary_target_m=.6, wote_metric_cache_dir=str(root))
            validate_reward_cache(config, anchors)
            path = metric_cache_file_for_collection('route', root/'train', BOUNDARY_TTC_METRIC_NAMES, .6)
            self.assertEqual(load_metric_cache_shard(path, metric_names=BOUNDARY_TTC_METRIC_NAMES)['metric_targets'].shape, (1, 256, 5))
            with self.assertRaises(ValueError):
                metric_cache_file_for_collection('route', root/'train', boundary_target_m=.6)
            manifest['valid_counts'][4] = 0
            (root/'val/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'no valid labels'):
                validate_reward_cache(config, anchors)


if __name__ == '__main__':
    unittest.main()
