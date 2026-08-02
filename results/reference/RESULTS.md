# VestibularFusion v1 参考结果

主模型协议：前向 `[t,t+1,t+2]` 上下文、`split_seed=42`、跨被试五折、无 inner loop、`severity_weight=0.3`、状态 `(R1+R2+2R4)/4`、严重度 R4-only。

| 数据集 | 状态 ACC | 高低眩晕 ACC |
|---|---:|---:|
| monifeixing | 88.40% | 83.33% |
| VRQ | 80.79% | 69.57% |
| 城市巡航 | 77.83% | 68.83% |

外层严重度标签不参与拟合或打分，但 source R4 evidence 在 source subjects 内生成，并包含无标签目标域校准。因此这是相对乐观的非嵌套诊断，不是 strict blind test。
