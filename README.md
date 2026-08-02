# BioFoundation v1

BioFoundation 主模型的独立整理版。仓库内包含完整 Python 源码，不会导入原始 `BioFoundation-main` 中的 Python 模块；原工作区仅作为不可公开的数据、问卷、协议清单和 checkpoint 资产来源。

## 锁定协议

- 前向上下文：`[t,t+1,t+2]`，右边界复制，不跨 session。
- `split_seed=42`，`training_seed=1001`，跨被试五折，无 inner loop。
- 联合训练损失：`L_state + 0.3 * L_severity`。
- 状态证据只计算 R1、R2、R4，融合为 `(R1 + R2 + 2R4) / 4`。
- 严重度只使用 R4，不计算或使用 R3。

| Dataset | State ACC | R4-only severity ACC |
|---|---:|---:|
| monifeixing | 88.40% | 83.33% |
| VRQ | 80.79% | 69.57% |
| City cruise | 77.83% | 68.83% |

严重度结果属于 `no-inner` 非嵌套诊断：source R4 evidence 在 source subjects 内生成，因此结果相对乐观。外层严重度标签不参与拟合或打分，但流程包含无标签目标域校准，不能描述为 strict blind test。

## 项目结构

```text
src/biofoundation_v1/
├── model/       # FEMBA Encoder、A1、PairSeverityHead
├── data/        # monifeixing、VRQ、city 数据适配器
├── training/    # 冻结 Encoder 的联合训练与 smoke
└── evaluation/  # 前向上下文、R1/R2/R4、严重度、指标与复现
```

历史实验、消融、多随机种子、随机初始化、Transformer、中心化上下文、伪在线实现和第三方 HEEGNet/geoopt vendor 均不在本项目中。

## 环境与外部资产

锁定环境为 WSL2 `Ubuntu-22.04-Bio`、Python 3.10、PyTorch `2.11.0+cu128`、CUDA 12.8、`mamba_ssm 2.3.1`。NumPy、SciPy、scikit-learn、openpyxl 和 joblib 版本见 [pyproject.toml](pyproject.toml)。

数据、问卷、约 0.7 GiB checkpoint、衍生特征和预测表不随仓库发布。复制 `configs/paths.example.json` 为 `configs/paths.local.json`，只在本机填写路径；本地配置已被 Git 忽略。

## 命令

```bash
python -m pip install -e .
python -m biofoundation_v1 preflight --config configs/paths.local.json
python -m biofoundation_v1 train --config configs/paths.local.json --dataset monifeixing --fold 1 --smoke
python -m biofoundation_v1 reproduce --config configs/paths.local.json --datasets monifeixing vrq city --device cuda
python -m biofoundation_v1 verify --actual outputs/seed42/aggregate_report.json
python scripts/generate_manifest.py --config configs/paths.local.json
```

`preflight` 会检查环境版本、数据与问卷、协议文件、五折 checkpoint、SHA-256 和被试隔离；任何缺失、哈希变化、fold 不完整或身份重叠都会直接报错。`reproduce` 拒绝非空输出目录，不支持静默 resume。

三数据集均提供按 fold 的训练入口，例如：

```bash
python -m biofoundation_v1 train --config configs/paths.local.json --dataset vrq --fold 1
python -m biofoundation_v1 train --config configs/paths.local.json --dataset city --fold 1
```

Windows 用户也可运行 `scripts/reproduce_wsl.ps1`。本地输出统一写入被忽略的 `outputs/`。

## License

本项目保留原 BioFoundation 的 Apache License 2.0。数据集、问卷和模型权重具有独立许可，公开仓库不分发这些资产。
