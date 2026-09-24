# WoTE 矿山版

该目录包含在 HD465 TransFuser 基础上实现的全部 WoTE 功能：

```text
wote_mining/
├── WoTE_model.py                           完整核心模型
├── WoTE_loss.py                            联合损失
├── WoTE_train.py                           训练逻辑与入口
├── WoTE_agent.py                           推理、控制与CARLA Agent
├── WoTE_config.py                          矿山配置
├── WoTE_targets.py                         训练目标构造
├── WoTE_simulator.py                       离线候选评价
├── WoTE_evaluate.sh                        闭环评测入口
├── assets/                                 anchor、路线、标签缓存
└── tools/                                  资产重新生成工具
```

训练：

```bash
cd ~/projects/transfuser-wote
conda activate tfuse
python wote_mining/WoTE_train.py \
  --root-dir '/media/ubuntu/徐培智/dataset_transfuser_hd465/raw' \
  --output-dir log/wote-mining-hd465 \
  --batch-size 1 --workers 4 --amp
```

入口根据 `assets/metric_cache/{train,val}/manifest.json` 直接从 `raw/`
解析120条训练路线和20条验证路线，不需要建立第二份数据集或恢复旧 split。

训练产生 `latest.pth` 后进行闭环评测：

```bash
TEAM_CONFIG=/path/to/log/wote-mining-hd465 \
CARLA_ROOT=/path/to/CARLA \
CHECKPOINT_ENDPOINT=results/hd465_wote_eval.json \
bash wote_mining/WoTE_evaluate.sh
```

`WoTE_agent.py` 复用矿车基线的传感器预处理、坡道控制、转向限幅、LiDAR安全检查和
卡住恢复机制。它在线评价256条候选，选择一条4秒轨迹，再输出转向、油门和制动。

离线标签顺序固定为 `NC、DAC、EP、TTC、Comfort`。需要重建标签时使用
`wote_mining/tools/WoTE_build_metric_cache.py`；其他资产重建与检查入口也都在
`wote_mining/tools/` 中。
