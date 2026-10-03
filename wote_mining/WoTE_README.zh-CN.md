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

本分支在原版 WoTE 的轨迹解码位置加入未来 BEV，候选与 `anchor + offset`
轨迹表示不变。当前版本保留原有损失，并新增轻量的 reward-topk 多候选轨迹监督：

```text
固定 anchors / 原候选 queries
            │
            ├── cross-attention 读取当前 BEV ───────────┐
            │                                           │
世界模型①按每条 anchor 预测未来 BEV                    │
            └── 同一 query 读取对应未来 BEV             │
                              │                         │
             零初始化 AdaLN-Zero 残差调制 ──────────┘
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
实现中没有额外的绝对轨迹头、残差修正器或局部未来 BEV。
当前 BEV 解码特征保留为规划主路径；候选 query 对对应的整张未来 BEV 做一次
全局注意力读取。候选 query 只用于读取，不再直接拼接进调制器；读取到的
未来 BEV 特征是 AdaLN-Zero 的唯一显式条件，用它产生逐通道 `scale`、`shift`
和 `gate`：

```text
conditioned = LayerNorm(current) * (1 + scale) + shift
future_update = MLP(conditioned)
fused = current + gate * future_update
```

产生 `scale/shift/gate` 的最后一层使用全零初始化，因此训练起点严格满足
`fused == current`，未来信息随后通过轨迹损失逐渐学习影响规划。原来的 WTA
offset loss、候选 imitation loss、固定 anchor 奖励标签和地图标签都保留。

训练标签对应固定的256条 anchor，因此训练和验证阶段的世界模型①、reward head
和地图监督仍使用固定 anchors；融合后的 offset/scores 直接接受原版轨迹损失。
CARLA 在线推理时，融合后生成的完整轨迹再进入世界模型②，最后只计算一次奖励。

### Reward-topk 多候选轨迹监督（仅训练/验证）

借用旧 rewardtopk 版本的候选筛选思想，不引入它的轨迹修正器：

1. 原有 WTA offset L1 loss 不变，始终监督距离专家最近的固定 anchor 候选。
2. 对其他候选，用**当前解码出的完整轨迹**与专家轨迹的 XY 平均误差
   `ADE <= 1 m`、终点误差 `FDE <= 2 m` 筛选同一局部驾驶模式。
3. 在通过筛选的候选中，按原 `reward_head` 的 `final_rewards` 取最多4条。
   不是按 `score_head` 排序；选择过程无梯度，不重复加入 WTA 候选。
4. 对这些候选的**最终 AdaLN 融合轨迹的8个完整位姿**计算辅助 Smooth L1，
   默认权重为 **0.25**。每个样本内部按有效额外候选数量取均值，再对 batch
   取均值；没有合格额外候选时贡献零，不强行填入其他模式，也不降低原 WTA 权重。

辅助回归会训练现有 offset head、AdaLN 融合路径及其上游，不增加网络参数、修正器、
奖励头或世界模型调用。原语义/奖励标签和损失不变；在线仍完整生成并评价256条候选，
训练 top-k 不限制推理候选数量。训练时的 reward 依然评价固定 anchors，因此这里
复用的是同一候选 ID 的奖励偏好，**不等同于在线世界模型②对完整轨迹的真实评分**。
这项改变旨在减少轨迹回归与最终候选选择的监督缺口，不保证闭环一定提升。

默认开启，可显式指定：

```bash
--reward-topk 4 \
--reward-topk-loss-weight 0.25 \
--reward-topk-ade-max-m 1.0 \
--reward-topk-endpoint-max-m 2.0
```

`--reward-topk 0` 或 `--reward-topk-loss-weight 0` 会关闭辅助监督，恢复之前的
训练总损失。建议本版本使用新的输出目录重新训练，避免混合旧日志；模型参数结构
没有变化，旧 AdaLN checkpoint 仍可严格加载。设置同时写入 `args.json` 和 checkpoint。
若用 `--resume` 加载旧 checkpoint 或改变辅助监督设置，会重置最低验证损失记录，
避免用新总损失与旧训练目标下的数值比较；模型、优化器状态和 epoch 仍正常恢复。

新增 TensorBoard 指标：`loss_reward_topk_traj`（已经乘以辅助权重）、
`traj_reward_topk_candidates_mean`（实际增加的候选数）、
`traj_reward_topk_eligible_mean`（通过筛选的非 oracle 候选数）、
`traj_reward_selected_supervised_fraction`（固定-anchor reward 选中 ID 被原 WTA
或有效 top-k 覆盖的样本比例）、`traj_reward_topk_{ade,fde}_m`（有效额外候选误差）。
这些覆盖率/误差不是在线世界模型②的闭环评价结果；ADE/FDE 的统计也受筛选阈值影响。

CPU 回归测试（包含筛选、遮罩、梯度、训练图、checkpoint 与256候选测试）：

```bash
python -m unittest discover -s wote_mining/tests -v
```

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
`traj_matched_{ade,fde}_m`、`traj_selected_{ade,fde}_m`、
`future_scale_magnitude`、`future_shift_magnitude`、`future_gate_magnitude` 与
`future_residual_magnitude`。这些项表示 AdaLN-Zero 调制参数及实际残差更新的
平均绝对幅度，只用于观察模型使用未来特征的程度，不能单独证明未来信息带来了收益。这些
字段均为无梯度监控量，不参与 `loss_total`，不会改变训练目标。
