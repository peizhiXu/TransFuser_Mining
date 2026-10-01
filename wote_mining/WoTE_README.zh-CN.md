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
当前BEV → 256条粗轨迹 → 第一次世界模型 → 粗轨迹reward预评分
       → top-4候选/未来交叉注意力 → 8点轨迹残差
       → 与其余252条未修正轨迹合并 → 第二次世界模型
       → 原WoTE reward最终评分 → 最终轨迹
```

候选特征已经通过原轨迹解码器读取过当前BEV。在线阶段先对256条粗轨迹预评分，
只让reward最高的4条查询各自的64个未来BEV token；其余252条保留粗轨迹并继续参加
最终评分。修正器还直接编码候选的8个 (x,y,yaw) 点，融合后为每个轨迹点预测
残差及0--1门控；横纵向单点修正限制在2 m内，航向角修正限制在0.35 rad内。
残差层零初始化，因此未训练时的新轨迹与原WoTE轨迹严格相同。

训练时，第一次世界模型仍对 256 条固定 anchor 运行，负责原有五项 reward 和未来
语义图监督，并用模型预测的最终 WoTE reward 排序候选。每个样本固定监督 1 条与专家
轨迹在 24 维坐标中最近的 oracle 候选；另外最多选择 4 条高 reward 粗轨迹，但它们必须
同时满足终点 XY 误差不超过 2 m、8 点平均 XY 误差不超过 1 m。候选不足时保留空槽掩码，
不会用无关轨迹补满。选出的 1--5 条粗轨迹再经过一次小规模、停止梯度的世界模型推演，
因此修正器看到的是每条粗轨迹自身导致的未来 BEV。修正损失给予 oracle、最高奖励候选、
其余候选 2、1、0.25 的相对权重；对已经接近专家的粗轨迹额外惩罚无谓的横纵向修正。
修正损失不再通过候选隐藏特征反传到原轨迹解码器；原 WoTE 损失继续训练粗轨迹。
缓存的 reward 和未来地图标签始终只与固定 anchor 对齐。

在线推理使用两次共享权重的世界模型和两次 reward head：第一次 reward 只确定需要
修正的 top-4 粗轨迹，第二次 reward 对4条修正轨迹与252条原粗轨迹统一评分并选择最终轨迹。
两次世界模型都是从当前时刻预测 T+4 秒，不是连续预测到 T+8 秒。修正损失权重
由 `--future-refinement-loss-weight` 设置；接近专家时的少改约束由
`--refinement-identity-loss-weight` 设置，默认0.1。附加候选数和兼容门槛可分别通过
`--refinement-reward-topk`、`--refinement-endpoint-max-m` 和
`--refinement-ade-max-m` 调整，默认值为 4、2.0 m 和 1.0 m。
训练挑候选仍使用固定 anchor 的奖励；若要改为粗轨迹奖励，需另增一次256条候选的推演。

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
`traj_coarse_selected_{ade,fde}_m`、`traj_selected_{ade,fde}_m`、
`traj_refinement_mean_m` 和
`traj_refinement_max_m`。这些字段均为
无梯度监控量，不参与 `loss_total`，不会改变训练目标。
