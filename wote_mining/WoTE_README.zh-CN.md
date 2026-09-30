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

训练标签对应固定的256条 anchor，因此训练和验证阶段的世界模型、reward head
都使用固定 anchor；轨迹 offset 分支仍正常学习。CARLA 在线推理时才将修正后的
轨迹送入世界模型评价，这与原版 WoTE 的训练/推理设计一致。

## Future-BEV 轨迹修正分支

`future-bev-planning` 分支在 WoTE 候选生成和最终评价之间增加一次候选级未来反馈：

```text
当前BEV → 256条粗轨迹 → 第一次世界模型 → 每条候选自己的未来BEV
       → 候选/未来交叉注意力 → 8点轨迹残差 → 第二次世界模型
       → 原WoTE reward评价一次 → 最终轨迹
```

候选特征已经通过原轨迹解码器读取过当前BEV；新增模块只让每条候选查询自己的
64个未来BEV token。融合采用拼接MLP和残差连接，最后的24维轨迹修正层为零初始化，
因此未训练时的新轨迹与原WoTE轨迹严格相同。

训练时只运行一次世界模型：前 256 个固定 anchor 正常参与原有五项 reward 和未来
语义图监督，另附加 oracle 匹配的 1 条粗轨迹，其未来 token 以停止梯度的方式指导
轨迹残差。新增
`loss_future_refinement` 在 oracle 匹配的粗轨迹上监督修正轨迹。训练时把该粗轨迹
作为第 257 个条件候选并入同一次世界模型前向，因此修正器看到的是这条粗轨迹自身
导致的未来 BEV；前 256 个固定 anchor 仍单独对应缓存的奖励和未来地图标签，不会把固定候选
reward标签错误地配给移动后的轨迹。在线推理时完整执行两次共享权重的世界模型：
第一次指导修正，第二次预测修正轨迹对应的T+4秒未来；两次不是连续预测到T+8秒。
修正损失权重可通过
`--future-refinement-loss-weight` 设置，默认值为1.0。

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
`traj_coarse_matched_{ade,fde}_m`、`traj_matched_{ade,fde}_m`、
`traj_selected_{ade,fde}_m`、`traj_refinement_mean_m` 和
`traj_refinement_max_m`。这些字段均为
无梯度监控量，不参与 `loss_total`，不会改变训练目标。
