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
世界模型①按每条 anchor 预测未来 scene                   │
            └── 注入动作前的未来 scene - 对齐当前 scene  │
                              │                         │
                 同一 query 读取对应变化量              │
                              │                         │
             零初始化 AdaLN-Zero 残差调制 ──────────┘
                              │
                              原 offset_head 输出 offset
                                               │
                                  trajectory = anchor + offset
                                               │
                             reward读取世界模型①的结果
                                               │
                             按奖励选中的编号取最终轨迹
```

这里的 `anchor + offset` 就是原版 WoTE 的轨迹生成方式，不是生成轨迹后的修正器。
实现中没有额外的绝对轨迹头、残差修正器或局部未来 BEV。
当前 BEV 解码特征保留为规划主路径。世界模型先输出动作条件的未来 scene，
再把动作特征显式注入候选终点格；规划融合只使用**注入前**的未来 scene，
避免候选 query 从终点格直接读回自己的动作编码。当前与未来 scene 经过同一个
LayerNorm 后相减，不增加独立投影网络；候选 query 对对应的64个变化 token 做
一次全局注意力读取。读出的变化特征是 AdaLN-Zero 的唯一显式条件，用它产生
逐通道 `scale`、`shift` 和 `gate`：

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
CARLA 默认 `WOTE_WORLD_PASSES=1`，复用世界模型①及其奖励排序，按选中的编号取
最终解码轨迹。`WOTE_WORLD_PASSES=2` 可用于对照：将完整轨迹送入世界模型②后评分。

当前参考 scene 使用 `detach()`；未来 scene 保持可微。因此 WTA 和 reward-topk
轨迹损失可以经未来残差条件训练世界模型，参考项不会沿相减路径反传。
未来注意力、AdaLN、offset head 和当前 BEV 主路径仍接受轨迹梯度；动作注入后的
未来 tokens 继续通过语义地图与 reward 损失训练。

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

辅助回归会训练现有 offset head、未来注意力、AdaLN 及当前特征路径，
也会经可微的未来 BEV 条件训练世界模型。不增加网络参数、修正器、
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

闭环的 `control_telemetry.jsonl` 额外记录 `reward_topk_candidates`：奖励最高的
4条候选各自的排名、anchor编号、是否最终选中、总奖励、具名评分、完整
`[8,3]` 轨迹及 `learned_desired_speed_mps`。轨迹为车辆坐标系的
`[x,y,yaw]`（米、米、弧度），速度按前两个轨迹点的XY间距乘2计算，与PID的
原始目标速度定义一致，尚未经过坡度、安全停车或卡住恢复调整。奖励相同的候选
优先列出实际选中的候选，再按编号排列。这些数据只用于分辨“没有生成合适轨迹”
和“奖励没有选好轨迹”，不改变选轨迹、PID或训练损失。仅日志新增不需要重训；
未来条件梯度和奖励定义的改动则需要重训后评估。

`source` 标签顺序为 `NC、DAC、EP、TTC、Comfort`。需要重建标签时使用
`wote_mining/tools/WoTE_build_metric_cache.py`；其他资产重建与检查入口也都在
`wote_mining/tools/` 中。

五个评价头各自在自己的有效标签上计算 BCE，再将五项相加，与原版 WoTE 的
五头损失尺度一致；缺失标签不会改变其他评价头的权重。当前与未来语义 BEV 使用
`alpha=0.5、gamma=2.0` 的 masked sigmoid focal loss：参数沿用原版 WoTE，
sigmoid 形式用于兼容矿山版可相互重叠的语义图层。

`metrics.jsonl` 除联合 loss 外，还记录五个评价头各自的 `bce`、`mae`、
`pred_mean`、`target_mean` 和 `valid_fraction`，以及轨迹的
`traj_matched_{ade,fde}_m`、`traj_selected_{ade,fde}_m`、
`future_delta_magnitude`、`future_delta_candidate_std`、`future_scale_magnitude`、
`future_shift_magnitude`、`future_gate_magnitude` 与
`future_residual_magnitude`。这些项表示 AdaLN-Zero 调制参数及实际残差更新的
平均绝对幅度；其中 `future_delta_magnitude` 是动作终点注入前的未来场景与
空间对齐当前场景经过共享归一化后的差值幅度，`future_delta_candidate_std`
是这份差值在全部候选之间的总体标准差。两者结合用于区分共享时间变化和候选相关
变化，但不能单独证明未来信息带来了收益。这些
字段均为无梯度监控量，不参与 `loss_total`，不会改变训练目标。

## 道路边界裕度试验（mining_boundary）

通过 `--reward-scheme mining_boundary` 启用，默认仍为 `source`，以免旧训练命令
意外更换目标。保留模仿奖励，评价头变为 **NC、DAC、EP、road_boundary_margin**
四个；不训练或使用 TTC/Comfort 头。AdaLN、Top-K、未来梯度和控制器沿用当前版本。

车身尺寸沿用 9.3969 m × 5.3645 m，每 0.125 s 检查一个未来姿态，每个姿态取
整个采样车身到非道路区域的最小距离 `c(t)`。未来标签为
`clip(min_t(c(t)) / boundary_target_m, 0, 1)`：排除当前时刻，取未来最小车身间距。
这样短暂贴边也会降低分数，避免被其他安全时刻稀释。标签版本为
`future_footprint_saturated_min_v2`，旧的时间平均标签缓存必须重新生成。
越界时刻的余量为零，地图外区域按未知掩码处理；原 DAC 包括当前时刻的定义不变。
距离来自 0.2 m/像素栅格与 0.5 m 车身采样，是几何近似，不是连续空间安全保证。

新评分为
`0.1 log(p_im) + 0.5 log(p_NC) + 0.5 log(p_DAC) + log(1 + p_EP) + lambda * p_boundary`。
前三个系数及 EP 系数仍由 `wote_reward_weights` 控制；EP=0 的分数有限，允许合法停车。
`lambda` 默认试验值 0.2，边界只是有界偏好，不作为一票否决。这里的 EP 聚合也已
变化，不能把完整方案的得分差异全部归因于新边界标签。四头 BCE 各自归一化后相加，
与旧五头的总损失不能直接横向比较。

**阈值是试验设置，不是矿车安全标准。** `0.6 m` 暂时保留，不为制造差异放大。
2026-10-12 检查训练集120条采集路线，每条按旧DAC失败比例选2个边缘约束较强帧，
共240张地图、44996条道路合规候选；这属于偏向边缘的抽样，不代表全数据发生率。
在0.6 m设置下，时间平均评分的场景内标准差均值为0.00653，未来最小间距评分
为0.09311；4.12%的合规候选进入非满分区间，合规专家均满分。
实际例子中最低间距约0.12 m，平均分约0.94，而最小间距评分约0.21。
道路标签来自OpenDRIVE道路栅格，不是实体路肩/边坡真值。已有闭环遥测缺少连续
车辆位姿，不能据此完成真实横向跟踪误差标定；仍需后续仿真验证安全间距。
CPU几何、缓存及梯度测试不替代服务器上使用真实batch的GPU短跑。

只读抽样（不生成报告、不改数据）：

```bash
python wote_mining/tools/WoTE_probe_boundary_labels.py \
  --data-root /path/to/raw \
  --cache-dir wote_mining/assets/metric_cache \
  --split train --max-routes 20 --frames-per-route 3
```

对边缘约束较强帧比较时间平均与最小间距（只读原数据，预览写入指定临时目录）：

```bash
python wote_mining/tools/WoTE_probe_boundary_labels.py \
  --data-root /path/to/raw --cache-dir wote_mining/assets/metric_cache \
  --split train --max-routes 120 --frames-per-route 2 --frame-selection edge \
  --target-m 0.6 --preview-dir /tmp/wote-boundary-preview
```

确定阈值后，分别生成新的 train/val 缓存，不覆盖原缓存。以0.6 m演示命令：

```bash
for split in train val; do
  python wote_mining/tools/WoTE_build_metric_cache.py \
    --data-root /path/to/raw \
    --assignments leaderboard/data/mining/split_v2/split_assignments.csv \
    --anchors wote_mining/assets/anchors/trajectory_anchors_256.npy \
    --routes-dir wote_mining/assets/routes \
    --output-dir "wote_mining/assets/metric_cache_boundary/$split" \
    --split "$split" --reward-scheme mining_boundary \
    --boundary-target-m 0.6 --workers 4
done
```

在原训练命令中附加 `--reward-scheme mining_boundary --boundary-target-m 0.6
--boundary-reward-weight 0.2`，使用新的训练输出目录。缓存默认定位到
`assets/metric_cache_boundary`，也可指定 `--metric-cache-dir`。旧缓存名称、阈值或
anchor哈希不匹配时会明确报错；不允许从旧五头checkpoint直接resume为四头。
新checkpoint及args.json保存奖励定义，旧checkpoint没有该字段时按source恢复。
推理仍仅使用传感器与学习的Reward，不查询真实道路地图。

缓存生成和TensorBoard记录边界标签均值、预测MAE，以及道路合规候选内的标签
标准差，后者用于确认新指标是否真的能区分DAC已经通过的候选。
