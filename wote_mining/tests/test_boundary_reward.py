"""Road-margin geometry, reward semantics and legacy/cache compatibility."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from wote_mining.WoTE_simulator import evaluate_road_compliance
from wote_mining.WoTE_model import WoTEMiningRewardHead
from wote_mining.WoTE_reward_schema import (
    BOUNDARY_METRIC_NAMES, BOUNDARY_LABEL_VERSION, restore_reward_settings,
)
from wote_mining.WoTE_targets import metric_cache_file_for_collection, load_metric_cache_shard
from wote_mining.WoTE_diagnostics import reward_topk_diagnostics


class BoundaryGeometryTests(unittest.TestCase):
    def road(self, half_width=3.0):
        road = np.zeros((500, 500), dtype=np.uint8)
        width = int(round(half_width * 5))
        road[:, 250-width:251+width] = 1
        return road

    def trajectory(self, lateral=0.0):
        poses = np.zeros((1, 8, 3), dtype=np.float32)
        poses[0, :, 0] = np.arange(1, 9)
        poses[0, :, 1] = lateral
        return poses

    def test_same_dac_different_margin_and_saturation(self):
        poses = np.concatenate([self.trajectory(0), self.trajectory(0.4)])
        result = evaluate_road_compliance(poses, self.road(), boundary_target_m=0.6)
        np.testing.assert_array_equal(result['drivable_compliance'], [1, 1])
        self.assertGreater(result['road_boundary_margin'][0], result['road_boundary_margin'][1])
        saturated = evaluate_road_compliance(poses, self.road(6.0), boundary_target_m=0.6)
        np.testing.assert_allclose(saturated['road_boundary_margin'], [1, 1])

    def test_shared_start_does_not_lock_future_minimum_scores(self):
        poses = np.concatenate([self.trajectory(), self.trajectory()])
        # A road offset makes the shared current body close to its left edge.
        road = np.zeros((500, 500), dtype=np.uint8)
        road[:, 236:281] = 1
        poses[1, :, 1] = np.linspace(0.1, 1.5, 8)
        result = evaluate_road_compliance(poses, road)
        self.assertEqual(result['future_road_clearance_m'].shape, (2, 32))
        # The legacy global minimum remains dominated by the common start.
        self.assertAlmostEqual(float(result['min_road_clearance_m'][0]),
                               float(result['min_road_clearance_m'][1]), places=4)
        self.assertGreater(result['road_boundary_margin'][1], result['road_boundary_margin'][0])
        expected = np.clip(result['future_road_clearance_m'].min(axis=1)/0.6, 0, 1)
        np.testing.assert_allclose(result['road_boundary_margin'], expected)

    def test_brief_close_approach_is_not_averaged_away(self):
        poses = self.trajectory()
        poses[0, -1, 1] = 3.0
        result = evaluate_road_compliance(poses, self.road(6.0))
        self.assertEqual(result['drivable_compliance'][0], 1)
        minimum = float(result['future_road_clearance_m'].min())
        self.assertLess(minimum, .6)
        score = float(result['road_boundary_margin'][0])
        self.assertAlmostEqual(score, minimum/.6, places=5)
        old_mean = np.clip(result['future_road_clearance_m']/.6, 0, 1).mean()
        self.assertGreater(old_mean, score)

    def test_footprint_not_center_and_unknown_map_is_masked(self):
        result = evaluate_road_compliance(self.trajectory(), self.road(2.0))
        self.assertEqual(result['drivable_compliance'][0], 0)
        self.assertEqual(result['road_boundary_margin'][0], 0)
        outside = self.trajectory()
        outside[0, :, 0] = 70
        result = evaluate_road_compliance(outside, self.road(6.0))
        self.assertEqual(result['road_boundary_valid'][0], 0)
        self.assertEqual(result['road_boundary_margin'][0], 0)

    def test_rotation_and_invalid_threshold(self):
        poses = self.trajectory()
        angle = np.pi/6
        rotated = poses.copy()
        rotated[..., 0] = poses[..., 0]*np.cos(angle)
        # Local augmentation convention rotates coordinates by -angle.
        rotated[..., 1] = -poses[..., 0]*np.sin(angle)
        rotated[..., 2] = -angle
        base = evaluate_road_compliance(poses, self.road(), samples_per_interval=1)
        result = evaluate_road_compliance(rotated, self.road(), augmentation_degrees=30,
                                          samples_per_interval=1)
        np.testing.assert_allclose(base['road_boundary_margin'], result['road_boundary_margin'], atol=1e-5)
        for threshold in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                evaluate_road_compliance(poses, self.road(), boundary_target_m=threshold)


class BoundaryProbeTests(unittest.TestCase):
    def test_edge_selection_uses_known_dac_not_other_metrics(self):
        from wote_mining.tools.WoTE_probe_boundary_labels import select_probe_frames
        targets = np.ones((3, 4, 5), dtype=np.float32)
        valid = np.ones_like(targets, dtype=np.uint8)
        targets[0, :, 0] = 0  # NC must not control edge sampling.
        targets[1, :3, 1] = 0
        targets[2, :, 1] = 0
        valid[2, :, 1] = 0  # Unknown labels must not count as failures.
        selected = select_probe_frames(np.arange(3), targets, valid, 1, "edge")
        np.testing.assert_array_equal(selected, [1])

    def test_temporal_mean_can_hide_brief_close_approach(self):
        from wote_mining.tools.WoTE_probe_boundary_labels import compare_boundary_aggregation
        clearances = np.full((2, 32), 2.0, dtype=np.float32)
        clearances[0, -1] = 0.12
        mean, worst = compare_boundary_aggregation(clearances, .6)
        self.assertGreater(float(mean[0]), .95)
        self.assertAlmostEqual(float(worst[0]), .2, places=5)
        self.assertEqual(float(worst[1]), 1.0)


class BoundaryRewardTests(unittest.TestCase):
    def test_soft_bonus_bounded_and_safety_and_progress_remain(self):
        head = WoTEMiningRewardHead(hidden_dim=8, reward_scheme='mining_boundary', boundary_reward_weight=0.2)
        logits = torch.zeros(1, 2, 4, requires_grad=True)
        imitation = torch.zeros(1, 2)
        logits.data[0, 0, 3] = -10
        logits.data[0, 1, 3] = 10
        rewards, _, _ = head.compose_rewards(imitation, logits)
        self.assertGreater(rewards[0, 1], rewards[0, 0])
        self.assertLessEqual((rewards[0, 1]-rewards[0, 0]).item(), 0.2)
        # A tiny soft bonus cannot rescue a strongly collision-risky mode.
        logits.data[0, 1, 0] = -10
        rewards, _, _ = head.compose_rewards(imitation, logits)
        self.assertGreater(rewards[0, 0], rewards[0, 1])
        rewards.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        # Stopping is allowed: EP=0 has no log(0) singularity.
        logits.data[..., 2] = -100
        self.assertTrue(torch.isfinite(head.compose_rewards(imitation, logits)[0]).all())

    def test_source_formula_is_unchanged_and_old_settings_restore(self):
        head = WoTEMiningRewardHead(hidden_dim=8)
        imitation = torch.tensor([[0.1, 0.4]])
        logits = torch.tensor([[[0.2, 0.4, 0.6, 0.8, 1.0], [1.0, 0.8, 0.6, 0.4, 0.2]]])
        scores = logits.sigmoid()
        expected = (0.1*imitation.log_softmax(-1) + 0.5*torch.nn.functional.logsigmoid(logits[..., 0])
                    + 0.5*torch.nn.functional.logsigmoid(logits[..., 1])
                    + (5*scores[..., 3]+2*scores[..., 4]+5*scores[..., 2]).log())
        torch.testing.assert_close(head.compose_rewards(imitation, logits)[0], expected)
        config = SimpleNamespace(wote_reward_scheme='mining_boundary')
        restore_reward_settings(config, {}, {})
        self.assertEqual(config.wote_reward_scheme, 'source')

    def test_boundary_checkpoint_and_args_preserve_label_definition(self):
        from wote_mining.WoTE_reward_schema import reward_settings
        config = SimpleNamespace(wote_reward_scheme='mining_boundary')
        saved = reward_settings(config)
        self.assertEqual(saved['label_version'], BOUNDARY_LABEL_VERSION)
        restore_reward_settings(config, {}, {'reward_scheme_settings': saved})
        restore_reward_settings(config, {'reward_scheme': 'mining_boundary',
                                        'boundary_label_version': BOUNDARY_LABEL_VERSION})
        for version in [None, 'future_footprint_saturated_mean_v1']:
            with self.assertRaises(ValueError):
                restore_reward_settings(config, {'reward_scheme': 'mining_boundary',
                                                'boundary_label_version': version})

    def test_telemetry_names_boundary_instead_of_ttc(self):
        outputs = dict(final_rewards=torch.tensor([[0., 1.]]), selected_index=torch.tensor([1]),
                       world_trajectories=torch.zeros(1, 2, 8, 3),
                       metric_scores=torch.ones(1, 2, 4), metric_names=BOUNDARY_METRIC_NAMES)
        scores = reward_topk_diagnostics(outputs)['reward_topk_candidates'][0]['metric_scores']
        self.assertEqual(tuple(scores), BOUNDARY_METRIC_NAMES)


class BoundaryCacheTests(unittest.TestCase):
    def test_mean_based_cache_is_rejected_even_when_threshold_matches(self):
        with tempfile.TemporaryDirectory(prefix='wote-boundary-old-mean-') as temp:
            root = Path(temp)
            manifest = dict(metric_names=list(BOUNDARY_METRIC_NAMES),
                            boundary_label_settings=dict(version='future_footprint_saturated_mean_v1', target_m=.6),
                            routes={'route': {'file': 'route.npz'}})
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                metric_cache_file_for_collection('route', root, BOUNDARY_METRIC_NAMES, .6)

    def test_cache_rejects_old_schema_or_wrong_threshold_and_loads_four_heads(self):
        with tempfile.TemporaryDirectory(prefix='wote-boundary-cache-') as temp:
            root = Path(temp)
            manifest = dict(metric_names=list(BOUNDARY_METRIC_NAMES),
                            boundary_label_settings=dict(version=BOUNDARY_LABEL_VERSION, target_m=0.6),
                            routes={'route': {'file': 'route.npz'}})
            (root/'manifest.json').write_text(json.dumps(manifest))
            np.savez(root/'route.npz', frames=np.array([2]),
                     metric_targets=np.full((1, 256, 4), 0.5), metric_valid=np.ones((1, 256, 4)))
            with self.assertRaises(ValueError):
                metric_cache_file_for_collection('route', root)
            with self.assertRaises(ValueError):
                metric_cache_file_for_collection('route', root, BOUNDARY_METRIC_NAMES, 0.8)
            path = metric_cache_file_for_collection('route', root, BOUNDARY_METRIC_NAMES, 0.6)
            loaded = load_metric_cache_shard(path, metric_names=BOUNDARY_METRIC_NAMES)
            self.assertEqual(loaded['metric_targets'].shape, (1, 256, 4))
            with self.assertRaises(ValueError):
                load_metric_cache_shard(path)

    def test_training_preflight_checks_both_splits_and_anchor_identity(self):
        from wote_mining.WoTE_train import validate_reward_cache
        import hashlib
        with tempfile.TemporaryDirectory(prefix='wote-boundary-preflight-') as temp:
            root = Path(temp)
            anchors = root/'anchors.npy'
            np.save(anchors, np.zeros((256, 8, 3), dtype=np.float32))
            manifest = dict(metric_names=list(BOUNDARY_METRIC_NAMES),
                            anchors_sha256=hashlib.sha256(anchors.read_bytes()).hexdigest(),
                            boundary_label_settings=dict(version=BOUNDARY_LABEL_VERSION, target_m=0.6))
            config = SimpleNamespace(wote_reward_scheme='mining_boundary',
                                     wote_boundary_target_m=0.6, wote_metric_cache_dir=str(root))
            for split in ['train', 'val']:
                (root/split).mkdir()
                (root/split/'manifest.json').write_text(json.dumps(manifest))
            validate_reward_cache(config, anchors)
            config.wote_boundary_target_m = 0.8
            with self.assertRaises(ValueError):
                validate_reward_cache(config, anchors)
            config.wote_boundary_target_m = 0.6
            np.save(anchors, np.ones((256, 8, 3), dtype=np.float32))
            with self.assertRaises(ValueError):
                validate_reward_cache(config, anchors)


if __name__ == '__main__':
    unittest.main()
