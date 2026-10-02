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

双卡训练（`--batch-size 4` 是每张卡4个样本，全局 batch size 为8）：

```bash
cd ~/projects/transfuser-wote
conda activate tfuse
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  wote_mining/WoTE_train.py \
  --root-dir '/media/ubuntu/徐培智/dataset_transfuser_hd465/raw' \
  --output-dir log/wote-mining-hd465 \
  --batch-size 4 --workers 4 --amp --save-every 5
```

入口根据 `assets/metric_cache/{train,val}/manifest.json` 直接从 `raw/`
解析120条训练路线和20条验证路线，不需要建立第二份数据集或恢复旧 split。

## 当前/未来 BEV 融合

本分支只在原版 WoTE 的轨迹解码位置加入未来 BEV，候选、轨迹表示和训练目标均不变：

```text
固定 anchors / 原候选 queries
            │
            ├── cross-attention 读取当前 BEV ──┐
            │                                  ├── 特征融合
世界模型①按每条 anchor 预测未来 BEV           │
            └── 同一 query 读取对应未来 BEV ──┘
                                               │
                              原 offset_head 输出 offset
                                               │
                                  trajectory = anchor + offset
                                               │
                                   世界模型②（仅在线推理）
                                               │
                                         reward 一次
```

这里的 `anchor + offset` 就是原版 WoTE 的轨迹生成方式，不是生成轨迹后的修正器。
实现中没有额外的绝对轨迹头、残差修正器、门控、top-k 或新增轨迹损失；原来的
WTA offset loss、候选 imitation loss、固定 anchor 奖励标签和地图标签都保留。

训练标签对应固定的256条 anchor，因此训练和验证阶段的世界模型①、reward head
和地图监督仍使用固定 anchors；融合后的 offset/scores 直接接受原版轨迹损失。
CARLA 在线推理时，融合后生成的完整轨迹再进入世界模型②，最后只计算一次奖励。

`latest.pth` 每个 epoch 覆盖保存，`best.pth` 保存最低验证总损失，编号 checkpoint
默认每5个 epoch及最后一个 epoch保存；可通过 `--save-every` 调整。

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

五个评价头各自在自己的有效标签上计算 BCE，再将五项相加，与原版 WoTE 的
五头损失尺度一致；缺失标签不会改变其他评价头的权重。当前与未来语义 BEV 使用
`alpha=0.5、gamma=2.0` 的 masked sigmoid focal loss：参数沿用原版 WoTE，
sigmoid 形式用于兼容矿山版可相互重叠的语义图层。

`metrics.jsonl` 除联合 loss 外，还记录五个评价头各自的 `bce`、`mae`、
`pred_mean`、`target_mean` 和 `valid_fraction`，以及轨迹的
`traj_matched_{ade,fde}_m` 与 `traj_selected_{ade,fde}_m`。这些字段均为
无梯度监控量，不参与 `loss_total`，不会改变训练目标。
