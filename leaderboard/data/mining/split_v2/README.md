# HD465 fixed split v2

- `split_assignments.csv`: 160 个原始采集路线到 train/val/test 的唯一映射。
- `train_routes.xml`: 120 条训练路线。
- `val_routes.xml`: 20 条验证路线。
- `test_routes.xml`: 20 条最终测试路线。
- `*.provenance.json`: 输出 XML ID 到原始 XML 文件和路线 ID 的映射。

划分平衡地图、实际天气/光照、长度、弯道、坡度、路口、环线和路网绕行。
训练代码只读取生成数据目录中的 `train/` 和 `val/`；`test/` 不参与训练。

