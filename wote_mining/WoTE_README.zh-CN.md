# WoTE 矿山版

该目录包含在 HD465 TransFuser 基础上实现的全部 WoTE 功能：

```text
wote_mining/
├── WoTE_model.py                           完整核心模型
├── WoTE_loss.py                            联合损失
├── WoTE_train.py                           训练逻辑与入口
├── WoTE_inference_eval.py                  录制验证集上的推理路径诊断
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
轨迹表示不变。当前版本保留原有损失及 reward-topk 多候选轨迹监督，并让原
reward imitation 分支在训练时额外评价最终解码候选。稳定未来条件和未来读取监督
保留为消融开关，但默认关闭：

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
                                   世界模型②（在线推理；训练评分对齐）
                                               │
                                         reward 一次
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
CARLA 在线推理时，融合后生成的完整轨迹再进入世界模型②，最后只计算一次奖励。

用于构造变化条件的当前与未来 scene 都使用 `detach()`，因此原 WTA 和
reward-topk 轨迹损失不会沿这条条件路径回传到世界模型或骨干。未来注意力、
AdaLN、offset head 和当前 BEV 主路径仍接受轨迹梯度；动作注入后的未来 tokens
继续通过原有语义地图与 reward 损失训练。这不是冻结世界模型或共享骨干，
也没有增加融合前轨迹的辅助监督。代码保留一个可选的读取评价头，
应使用新的输出目录从头训练并单独评估，不直接续训旧 checkpoint。

### 稳定未来条件与读取监督

训练时，原世界模型①的计算和 dropout 不变，继续接受语义地图、reward 损失。
仅对规划用的未来条件额外进行一次 `no_grad` 前向：临时关闭动作编码器
`anchor_context` 和世界模型 `transition` 的 dropout，重新编码固定 anchors，
获得动作注入前的未来 scene。之后恢复所有子模块原来的训练状态。
这样消除这两处 dropout 给候选条件引入的独立随机差异；不是把整个模型设为
eval，也不冻结世界模型。共享骨干、规划 query 和读取注意力仍按原方式训练，
因此不能要求训练/验证集的所有诊断值完全相同。验证和在线推理已经处于 eval，
直接复用已有世界模型①的输出，不增加世界模型调用。

在 `future_delta_features`（注意力读出的特征，进入 AdaLN 之前）上增加一个
`Linear(256,3)` 头，预测 **NC / DAC / EP**。监督直接使用现有
`wote_metric_targets[..., :3]` 和有效标签掩码，不使用 reward 预测当伪标签，
也不重建标签。未来条件基于固定 anchor，标签使用相同候选 ID，不能把它解释成
融合后新轨迹的安全真值。每项 BCE 独立按有效标签平均，三项相加，
默认总权重为 **0（关闭）**。没有有效标签时贡献连接计算图的零损失。

这项损失训练读取注意力、共享归一化、query 路径和新增读取头；由于 scene
操作数仍 detach，不沿条件路径训练世界模型。辅助头不直接训练 AdaLN，
AdaLN 仍由原轨迹损失和 reward-topk 轨迹损失训练。
它不是新的在线 reward head；在线关闭辅助头计算，仍由原五指标 reward
选出轨迹，PID 不变。它能促使读取结果包含候选相关信息，但不保证消除候选相似
或提升闭环表现，也不能单凭 BCE 降低证明世界预测正确。

新增头有 **771** 个参数，旧 checkpoint 缺少该头，无法在本版直接严格加载或
恢复优化器状态；旧模型闭环测试请继续使用对应旧项目代码。新版本自身的 checkpoint
支持正常 `--resume`；若改变读取权重、稳定条件或 top-k 设置，重置历史最优总损失。

该消融默认关闭；如需重新启用完整的稳定读取实验：

```bash
--future-read-loss-weight 0.1 \
--stable-future-condition
```

`--future-read-loss-weight 0` 仅关闭新增损失；`--stable-future-condition`
会启用额外确定性条件前向，兼容参数 `--no-stable-future-condition` 可显式关闭。
启用稳定条件后，训练增加一次不保存反向图的
世界模型①前向，会有额外计算成本；验证和闭环推理不增加世界模型次数。

TensorBoard 新增 `loss_future_read_metrics`（已乘权重）、
`future_read_{no_collision,drivable_compliance,ego_progress}_{bce,mae,pred_mean,target_mean,valid_fraction}`、
`future_read_no_collision_unsafe_{mae,fraction}`。NC 多数标签可能为“安全”，
应同时看不安全样本比例与误差，不能只看总体 BCE。
`future_scene_candidate_std` 衡量实际规划条件中的原始未来 scene 的候选差异，
`future_delta_candidate_std` 衡量归一化差值的候选差异，
`future_read_candidate_std` 衡量最终读取特征的候选差异。前两者来自稳定条件；
后一项仍可能包含训练态 query/attention dropout，不是纯环境差异。

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
不会经截断的未来 BEV 条件训练世界模型。不增加网络参数、修正器、
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

`--reward-topk 0` 或 `--reward-topk-loss-weight 0` 会将辅助 top-k 项置零；
新增的读取监督独立控制。建议本版本使用新的输出目录重新训练，
避免混合旧日志。top-k 本身不改变参数结构，但本次读取头改变了结构，
旧 AdaLN checkpoint 不再直接兼容。设置同时写入 `args.json` 和 checkpoint。
若用 `--resume` 加载本版 checkpoint 并改变辅助监督设置，会重置最低验证损失记录，
避免用新总损失与旧训练目标下的数值比较；模型、优化器状态和 epoch 仍正常恢复。

新增 TensorBoard 指标：`loss_reward_topk_traj`（已经乘以辅助权重）、
`traj_reward_topk_candidates_mean`（实际增加的候选数）、
`traj_reward_topk_eligible_mean`（通过筛选的非 oracle 候选数）、
`traj_reward_selected_supervised_fraction`（固定-anchor reward 选中 ID 被原 WTA
或有效 top-k 覆盖的样本比例）、`traj_reward_topk_{ade,fde}_m`（有效额外候选误差）。
这些覆盖率/误差不是在线世界模型②的闭环评价结果；ADE/FDE 的统计也受筛选阈值影响。

### 最终解码候选的评分对齐（仅训练/验证）

原固定-anchor reward 与五项指标监督全部保留。在此基础上，训练时将256条
`anchor + offset` 最终轨迹按在线路径重新编码，复用同一个世界模型和 reward head，
只对已有 imitation logits 增加专家相似度监督。软目标由完整4秒位姿距离、前1秒
位置距离和首段速度向量误差共同构成；这不是把专家轨迹当作碰撞或道路安全真值，
因此不会错误复用固定 anchors 的 NC/DAC/EP 标签。

这条评分支路对输入轨迹和几何目标执行 `detach`：它训练共享轨迹编码器、世界模型
和 reward head，但不能通过这项损失直接推动 offset 输出贴近答案。轨迹生成仍由
WTA 与 reward-topk 轨迹损失负责。它不增加模型参数和闭环推理计算，默认权重为
**0.25**：

```bash
--decoded-imitation-loss-weight 0.25
```

设为0会同时跳过这次额外世界模型前向。TensorBoard 的
`loss_decoded_imitation` 记录加权损失；`traj_decoded_selected_{ade,fde}_m` 记录
实际解码候选重新评分后的选择误差，`traj_decoded_selection_agreement` 记录它与
固定-anchor代理选择的一致率。这些仍是录制数据上的专家几何诊断，不等于闭环安全分。

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
4条候选各自的排名、anchor编号、是否最终选中、总奖励、五项具名评分、完整
`[8,3]` 轨迹及 `learned_desired_speed_mps`。轨迹为车辆坐标系的
`[x,y,yaw]`（米、米、弧度），速度按前两个轨迹点的XY间距乘2计算，与PID的
原始目标速度定义一致，尚未经过坡度、安全停车或卡住恢复调整。奖励相同的候选
优先列出实际选中的候选，再按编号排列。这些数据只用于分辨“没有生成合适轨迹”
和“奖励没有选好轨迹”，不改变选轨迹、PID或训练损失。仅日志新增不需要重训；
本次 future BEV detach 的训练影响则需要重训后评估。

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
`traj_decoded_selected_{ade,fde}_m`、`traj_decoded_selection_agreement`、
`future_delta_magnitude`、`future_delta_candidate_std`、`future_scale_magnitude`、
`future_shift_magnitude`、`future_gate_magnitude` 与
`future_residual_magnitude`。这些项表示 AdaLN-Zero 调制参数及实际残差更新的
平均绝对幅度；其中 `future_delta_magnitude` 是动作终点注入前的未来场景与
空间对齐当前场景经过共享归一化后的差值幅度，`future_delta_candidate_std`
是这份差值在全部候选之间的总体标准差。两者结合用于区分共享时间变化和候选相关
变化，但不能单独证明未来信息带来了收益。这些
字段均为无梯度监控量，不参与 `loss_total`，不会改变训练目标。

## 录制验证集上的推理路径诊断（不需要重训）

训练及原有 loss 验证保留固定 anchor 评分方式，不改训练目标。新增独立入口
`WoTE_inference_eval.py`，对同一帧进行两次 **eval/no_grad** 前向：

- `proxy`：`use_fused_world=False`，按固定 anchor 的 reward 选择 ID，再提取该
  ID 的 **融合后轨迹**；对应原 TensorBoard 的选中轨迹误差统计。
- `online`：`use_fused_world=True`，完整融合后轨迹重新编码，进入世界模型②、
  原 reward head，选择最终 reward 最大的轨迹。调用路径与 CARLA agent 一致，
  但传感器输入仍是录制验证数据，没有运行 CARLA/PID。
- `oracle_min_ade`：最终候选库中 ADE 最低的一条，仅用于诊断，不参与执行。

比较两者的 ADE/FDE、选择 ID 一致率、首秒位置误差，以及按 PID 同一公式计算的
原始目标速度误差（前两个点间距乘2）、该段速度向量和方向误差。方向误差仅统计
预测和专家速度都大于0.1m/s的样本，单独保存有效数；停止样本仍统计位置和速度。
`online_selection_regret_ade_m` 是实际选中轨迹 ADE 减去候选最小 ADE；
`online_reward_top4_min_ade_m` 检查更接近专家的轨迹是否进入 reward 前4名。
最小 FDE 和 ADE 最小那条轨迹的 FDE 分别记录，避免混为一条“最优轨迹”。

在训练服务器同时比较旧残差与 stable-read 的 epoch30：

```bash
conda activate WoTE
cd /home/kemove/xpz/projects/transfuser_wote_bevfusion_residual

CUDA_VISIBLE_DEVICES=0 python wote_mining/WoTE_inference_eval.py \
  --root-dir /home/kemove/xpz/datasets/mining_dataset_hd465/split_v2 \
  --checkpoint previous=/home/kemove/xpz/outputs/wote/wote-bevfusion-residual-adalnzero-topk-bs16-30ep/checkpoint_030.pth \
  --checkpoint stable_read=/home/kemove/xpz/outputs/wote/wote-bevfusion-residual-stableread-bs16-30ep/checkpoint_030.pth \
  --output-dir /home/kemove/xpz/outputs/wote/inference-compare-residual-stableread-ep30 \
  --batch-size 4 --workers 4
```

如需先检查读数据及权重，可另选输出目录并加 `--max-samples 16`；小样本只用于
检查脚本，不用于挑模型或判断收益。此入口不用 torchrun，也不用启动 CARLA。
输出目录必须是新的，避免覆盖已有结果。
默认使用 float32，与现有 CARLA agent 一致；可选 `--amp` 提速，但可能改变接近
平分的候选排序，正式对齐闭环选轨迹时建议保持默认精度。

脚本按 lidar 路径排序，每个 checkpoint 使用同一批验证帧，并保存
`samples.json`、每个模型的 `*_samples.jsonl` 和 `*_summary.json`，以及
`comparison.json`（总表、每条采集路线的统计、相对第一个 checkpoint 的差值）。
均值按样本计权，保留最后不足一批的数据，不做 DDP 填充重复样本。
旧残差 checkpoint 仅允许缺少后来新增、在线不使用的两个辅助读取头参数；
其他缺失、额外参数、形状差异或 anchor 内容不一致都会报错。骨干初始化不下载
预训练权重，所有在线参数从 checkpoint 加载。

这些是 **执行路径的专家距离诊断**，不是安全真值、PDMS 或 CARLA 闭环得分。
候选库最小 ADE 变差，提示生成轨迹问题；最小 ADE 稳定但选中 ADE/选择遗憾变差，
提示排序问题。不过安全绕行也可能增大专家距离，因此需要结合已有闭环日志确认，
不能仅据此断定碰撞风险或闭环分数改善。
