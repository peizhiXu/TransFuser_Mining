# HD465 矿山 TransFuser

本项目基于 TransFuser 2022 分支，加入了 HD465 矿卡、矿山传感器、专家采集、
显式路线级数据划分和闭环评测适配。

## 保留的入口

| 任务 | 入口 |
|---|---|
| 专家采集 | `leaderboard/scripts/collect_hd465_transfuser.sh` |
| 单路线采集调试 | `leaderboard/scripts/datagen_mine.sh` |
| 生成固定数据划分 | `tools/dataset/prepare_hd465_split_v2.sh` |
| 双卡训练 | `team_code_transfuser/train_hd465_split_v2.sh` |
| 闭环评测 | `leaderboard/scripts/evaluate_hd465_transfuser.sh` |

正式路线位于 `leaderboard/data/mining/routes_final/`，共 160 条。固定实验划分位于
`leaderboard/data/mining/split_v2/`：120 条训练、20 条验证、20 条测试。测试路线不进入
训练或 checkpoint 选择。

## 服务器生成数据划分

默认原始数据：

```text
/home/kemove/xpz/datasets/mining_dataset_hd465/raw
```

运行：

```bash
cd /home/kemove/xpz/projects/transfuser
bash tools/dataset/prepare_hd465_split_v2.sh
```

输出为 `/home/kemove/xpz/datasets/mining_dataset_hd465/split_v2`，只包含指向原始数据的
符号链接，不复制图片或点云。脚本会验证路线数量必须为 120/20/20。

## 双卡训练

```bash
cd /home/kemove/xpz/projects/transfuser
conda activate tfuse
bash team_code_transfuser/train_hd465_split_v2.sh
```

默认使用 GPU 0、1，每卡 batch size 4，训练 41 个 epoch，每 5 个 epoch 验证一次。
输出目录为 `/home/kemove/xpz/outputs/transfuser/hd465_transfuser_split_v2`。参数可通过
`GPU_IDS`、`BATCH_SIZE`、`EPOCHS`、`VAL_EVERY`、`DATA_ROOT`、`LOG_ROOT` 和
`EXPERIMENT_ID` 环境变量覆盖。

## 闭环路线

- 验证：`leaderboard/data/mining/split_v2/val_routes.xml`
- 测试：`leaderboard/data/mining/split_v2/test_routes.xml`

评测脚本默认使用验证路线。正式测试时显式设置：

```bash
ROUTES=/home/kemove/xpz/projects/transfuser/leaderboard/data/mining/split_v2/test_routes.xml \
bash leaderboard/scripts/evaluate_hd465_transfuser.sh
```

源文件名中的 `yutian` 是历史命名；实验应以 XML 中实际的 CARLA weather 参数为准。
