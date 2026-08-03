# VestibularFusion raw fixed-view 重新训练结果

本结果来自 `retrained_v3`：fixed-view R1/R2 分支由当前项目从原始数据重新生成，每个 fold 只使用 source-train/source-val 选择配置；不使用历史 `fixed_margins`、历史 fixed-view 配置或锁定参考 checkpoint。

| 数据集 | 状态 ACC | 高低眩晕 ACC |
|---|---:|---:|
| monifeixing | 85.10% | 88.89% |
| VRQ | 81.00% | 73.91% |
| 城市巡航 | 77.20% | 66.23% |

Temporal Encoder 在训练中保持冻结，仅 A1 和 `PairSeverityHead` 更新；`PairSeverityHead` 不参与最终预测。严重度是 no-inner 非嵌套诊断，包含无标签目标域校准，不是 strict blind test。

原来的 `results/reference/` 仍保存 `reproduce`/`verify` 对应的锁定参考结果，两套结果不能混用。
