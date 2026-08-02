# VestibularFusion

VestibularFusion 是一个面向生物信号分析的可复现实验项目。项目使用 Temporal Encoder 提取时序特征，并通过 `DirectionalMambaKAN`（A1）完成状态建模、生成 R1、R2、R4 证据。项目同时提供两条用途不同的严重度路径：训练接口使用 `PairSeverityHead` 作为辅助任务头；锁定复现使用 R4-only 标量特征和逻辑回归生成报告结果。

1. 识别样本的状态类别；
2. 基于锁定的 R4-only 诊断协议判断高、低眩晕严重度。

项目提供统一的数据适配、模型组件、训练入口、评估流程、环境检查和结果验证工具，支持 monifeixing、VRQ 和城市巡航三个数据集。

## 主要功能

- 使用 Temporal Encoder 作为冻结的时序特征提取器；
- 使用 A1 完成状态分类并生成 R1、R2、R4 证据；
- 在训练接口中使用 PairSeverityHead 提供辅助严重度监督；
- 在锁定复现中使用 R4-only 标量特征与逻辑回归评估严重度；
- 支持三窗口前向上下文和 session 边界处理；
- 支持跨被试五折划分，避免同一被试同时出现在训练集和评估集；
- 提供训练 smoke 测试，用于验证 Encoder 冻结以及任务头参数更新；
- 提供严格的资产、环境、SHA-256 和 fold 完整性检查；
- 提供从外部数据和 checkpoint 重新运行评估的统一命令；
- 提供脱敏的参考结果和源码清单。

## 训练与复现路径

训练路径用于验证和保留联合训练能力：

```text
输入生物信号窗口
        │
        ▼
Temporal Encoder（冻结）
        │
        ▼
       DirectionalMambaKAN（A1）
        │
        ├── 状态预测 ──────────────► L_state
        │
        └── reference/task embeddings
                    │
                    ▼
             PairSeverityHead ─────► L_severity

总损失：L_state + 0.3 * L_severity
```

锁定复现路径用于生成本仓库公布的严重度结果：

```text
输入生物信号窗口
        │
        ▼
Temporal Encoder → DirectionalMambaKAN（A1） → R4 evidence
                            │
                            ▼
                均匀选择 11 个任务窗口
                            │
                            ▼
                  winsorized standard deviation
                            │
                            ▼
                    单维 severity feature
                            │
                            ▼
              StandardScaler + LogisticRegression
                            │
                            ▼
                    报告严重度 ACC
```

城市巡航在逻辑回归前还会执行累计 R4 特征和 subject contextualization。`reproduce` 只加载锁定的 Temporal Encoder/A1 评估组件，不创建或加载 `PairSeverityHead`，因此锁定报告的严重度结果不是该训练头的性能结果。

项目源码位于 `src/vestibular_fusion/`，其中 `vestibular_fusion` 是当前 Python 包和命令行入口名称。

```text
src/vestibular_fusion/
├── model/
│   ├── encoder.py       # Temporal Encoder 和 checkpoint 加载
│   ├── a1.py          # A1 状态头
│   ├── severity.py    # PairSeverityHead
│   └── main.py        # 主模型组合
├── data/
│   ├── monifeixing.py # monifeixing 数据适配
│   ├── vrq.py         # VRQ 数据适配
│   ├── city.py        # 城市巡航数据适配
│   └── features.py    # 共享特征结构
├── training/
│   └── runner.py      # 冻结 Encoder 的训练与 smoke 流程
├── evaluation/
│   ├── context.py     # 前向上下文构造
│   ├── fusion.py      # R1/R2/R4 状态融合
│   ├── severity.py    # 严重度评估
│   ├── metrics.py     # 分类指标
│   └── runner.py      # 三个数据集的复现流程
├── preflight.py       # 环境和外部资产检查
└── verify.py          # 参考结果验证
```

## 固定实验协议

项目使用统一的主模型协议：

- 前向上下文为 `[t, t+1, t+2]`；
- 到达 session 右边界时复制最后一个有效窗口；
- 上下文不会跨越 session；
- `split_seed=42`；
- `training_seed=1001`；
- 跨被试五折划分；
- 不使用 inner fold；
- `train` 的联合训练损失为 `L_state + 0.3 * L_severity`；
- `train` 会训练 PairSeverityHead，并在训练 checkpoint 中保存 `severity_head_state_dict`；
- `reproduce` 不加载 `severity_head_state_dict`，而是在每个外层 fold 的 source subjects 上重新拟合 R4-only 逻辑回归；
- 状态证据使用 R1、R2 和 R4，融合公式为：

  ```text
  (R1 + R2 + 2R4) / 4
  ```

- 锁定报告的严重度分类使用 R4-only 证据。

## 参考结果

以下结果来自项目锁定协议和五折评估流程：

| 数据集 | 状态 ACC | 高低眩晕 ACC |
|---|---:|---:|
| monifeixing | 88.40% | 83.33% |
| VRQ | 80.79% | 69.57% |
| 城市巡航 | 77.83% | 68.83% |

> **结果解释：** 表中的 83.33%、69.57% 和 68.83% 均来自 R4-only source-fit logistic severity diagnostic，不是 `PairSeverityHead` 的测试成绩。本仓库当前没有把 `PairSeverityHead` 的外层测试性能作为参考结果发布。准确地说，`PairSeverityHead` 只作为辅助严重度监督约束 A1 表示学习；正式严重度推理采用 R4 离散度特征和源域逻辑回归归类。

完整参考结果位于：

- [`results/reference/RESULTS.md`](results/reference/RESULTS.md)
- [`results/reference/aggregate_report.json`](results/reference/aggregate_report.json)

严重度结果属于 `no-inner` 非嵌套诊断。source R4 evidence 在 source subjects 内生成，流程还包含无标签目标域校准，因此该结果相对乐观，不能描述为 strict blind test。外层严重度标签不参与拟合或打分。

## 环境

推荐使用以下环境运行：

- WSL2：`Ubuntu-22.04-Bio`
- Python：`3.10`
- PyTorch：`2.11.0+cu128`
- CUDA：`12.8`
- `mamba_ssm`：`2.3.1`
- NumPy：`2.2.6`
- SciPy：`1.15.3`
- scikit-learn：`1.7.2`
- openpyxl：`3.1.5`
- joblib：`1.5.3`

Python 依赖版本记录在 [`pyproject.toml`](pyproject.toml)，CUDA 相关依赖记录在 [`requirements-cuda.txt`](requirements-cuda.txt)。

## 外部数据和模型资产

公开仓库只包含源码、配置模板、测试、参考结果和文档。原始数据、问卷、checkpoint、衍生特征和运行输出不包含在仓库中。

首次运行前，复制配置模板：

```bash
cp configs/paths.example.json configs/paths.local.json
```

然后在 `configs/paths.local.json` 中填写本机数据和模型资产路径。Windows 用户可以手动复制文件，或使用 PowerShell：

```powershell
Copy-Item configs/paths.example.json configs/paths.local.json
```

`configs/paths.local.json`、`outputs/`、本地资产清单、缓存、数据文件和模型权重均已加入 Git 忽略规则，不应提交到公开仓库。

## 安装

在仓库根目录执行：

```bash
python -m pip install -e .
```

安装完成后，项目命令模块为 `vestibular_fusion`。

## 使用方法

### 环境和资产检查

```bash
python -m vestibular_fusion preflight \
  --config configs/paths.local.json
```

`preflight` 会检查 Python、PyTorch、CUDA 和依赖版本，以及数据文件、问卷、协议文件、五折 checkpoint、SHA-256 摘要、fold 完整性和被试身份隔离。发现缺失、版本不匹配、哈希变化或身份重叠时会直接报错。

### 训练 smoke 测试

```bash
python -m vestibular_fusion train \
  --config configs/paths.local.json \
  --dataset monifeixing \
  --fold 1 \
  --smoke
```

smoke 流程只运行单 fold、单批次训练，用于确认 Temporal Encoder 参数保持不变，而 A1 和 PairSeverityHead 能够通过反向传播更新。该检查只能证明训练路径有效，不代表参考结果使用了 PairSeverityHead。

### 训练单个 fold

```bash
python -m vestibular_fusion train \
  --config configs/paths.local.json \
  --dataset monifeixing \
  --fold 1
```

数据集参数可选：`monifeixing`、`vrq`、`city`。

完整训练会保存 A1 的 `model_state_dict` 和 PairSeverityHead 的 `severity_head_state_dict`。这些训练输出与下面的锁定复现路径是不同的评估接口。

### 完整复现

```bash
python -m vestibular_fusion reproduce \
  --config configs/paths.local.json \
  --datasets monifeixing vrq city \
  --device cuda
```

复现流程会先执行 `preflight`，然后加载锁定的五折 A1 checkpoint，生成 R1、R2、R4 状态证据，并在 source subjects 上拟合 R4-only 严重度逻辑回归，最后自动验证生成的 `aggregate_report.json`。该命令不加载 PairSeverityHead。输出写入 `outputs/`，不使用静默 resume。

Windows 用户也可以运行：

```powershell
.\scripts\reproduce_wsl.ps1
```

### 验证结果

```bash
python -m vestibular_fusion verify \
  --actual outputs/seed42/aggregate_report.json
```

### 生成资产清单

```bash
python scripts/generate_manifest.py \
  --config configs/paths.local.json
```

资产清单会记录外部文件和源码的 SHA-256 摘要，便于确认复现实验使用的文件没有发生变化。

## 测试

运行项目测试：

```bash
python -m pytest -q
```

测试覆盖：

- 三窗口边界复制；
- session 隔离；
- R1/R2/R4 状态融合；
- R4-only 严重度计算；
- Lorentz 几何计算；
- 五折被试身份隔离；
- 分类指标；
- SHA-256 资产校验。

## 输出和文件边界

运行输出统一写入 `outputs/`，包括日志、预测表、缓存和复现报告。这些内容用于本机实验，不属于公开源码发布内容。公开仓库中的参考结果已经去除本机绝对路径、原始预测和衍生特征。

## 许可

源码使用 Apache License 2.0，详见 [`LICENSE`](LICENSE)。数据集、问卷和模型权重可能具有独立许可，使用前请分别确认相应授权条件。
