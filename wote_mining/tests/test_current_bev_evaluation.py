"""Verify pointwise L2 horizons and the deployed score-selection evaluator."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from wote_mining.WoTE_current_bev_eval import main, sample_metrics
from wote_mining.tests import test_current_bev_baseline as baseline_tests


class CurrentBEVEvaluationTests(unittest.TestCase):
    def test_metrics_use_score_selection_and_exact_pointwise_seconds(self):
        bank = torch.zeros(1, 3, 8, 3)
        bank[:, 1, :, 0] = torch.arange(1, 9).float()
        bank[:, 2, :, 0] = 9
        output = dict(trajectories=bank, scores=torch.tensor([[0., 2., 1.]]),
                      selected_index=torch.tensor([1]), selected_trajectory=bank[:, 1])
        values = sample_metrics(output, torch.zeros(1, 8, 3))
        for key, expected in [('online_l2_1s_m', 2), ('online_l2_2s_m', 4),
                              ('online_l2_3s_m', 6), ('online_l2_4s_m', 8),
                              ('online_ade_2s_m', 2.5), ('online_ade_m', 4.5),
                              ('oracle_min_ade_m', 0), ('online_selection_regret_ade_m', 4.5)]:
            self.assertAlmostEqual(values[key].item(), expected)
        output['selected_index'] = torch.tensor([0])
        with self.assertRaisesRegex(ValueError, 'argmax'):
            sample_metrics(output, torch.zeros(1, 8, 3))

    def test_full_entry_preserves_all_samples_and_serializes_predictions(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        module, batch = baseline_tests.CurrentBEVBaselineTests().make_module()

        class Samples(Dataset):
            def __init__(self, root, config):
                self.lidars = [[str(Path(root[0]) / 'lidar' / ('%04d.npy' % i)).encode()]
                               for i in range(3)]

            def __len__(self):
                return 3

            def __getitem__(self, index):
                sample = {key: value[0].clone() for key, value in batch.items()}
                sample['wote_future_poses'][:, 0] += index * .2
                return sample

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train').mkdir()
            (root / 'val/route/lidar').mkdir(parents=True)
            argv = ['eval', '--root-dir', str(root), '--checkpoint', 'synthetic.pth',
                    '--output-dir', str(root / 'out'), '--batch-size', '2',
                    '--workers', '0', '--device', 'cpu']
            with patch('sys.argv', argv), patch('wote_mining.WoTE_current_bev_eval.load_planner',
                                               return_value=(module.planner, module.config, 25)), \
                    patch('team_code_transfuser.data.CARLA_Data', Samples), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            summary = json.loads((root / 'out/summary.json').read_text())
            records = [json.loads(s) for s in (root / 'out/samples.jsonl').read_text().splitlines()]
            self.assertEqual(summary['samples'], 3)
            self.assertEqual(summary['deployment_world_passes'], 0)
            self.assertEqual(summary['checkpoint_epoch_number'], 25)
            for key, value in summary['means'].items():
                self.assertAlmostEqual(value, sum(r[key] for r in records) / 3)
            self.assertEqual([r['frame'] for r in records], ['0000', '0001', '0002'])
            self.assertEqual(len(records[0]['trajectory']), 8)


if __name__ == '__main__':
    unittest.main()
