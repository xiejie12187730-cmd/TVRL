# TVI-Net

Trustworthy Variational Imputation Network for Incomplete Multi-view Multi-label Classification.

本工程的模型、模块及损失名称与论文及框架图统一。Python 类名使用 `TVINet`，论文和运行日志中的模型名称使用 `TVI-Net`。

## 运行

在包含 `train.py` 的项目目录中执行：

```bash
python train.py
```

训练入口使用 `from models.TVINet import TVINet`。依赖包括 Python、PyTorch、NumPy、SciPy 和 scikit-learn。当前附带 Corel5k 数据和对应的数据划分、缺失掩码。

## 框架图与代码名称

| 框架部分 | 论文名称 | 代码位置与类名 |
| --- | --- | --- |
| 整体模型 | TVI-Net | `models/TVINet.py`：`TVINet` |
| (a) | VAE-based View Imputation | `models/vae_imputation.py`：`VAEViewImputation` |
| (b) | Dual-weighted Multi-view Representation Learning | `TVINet` 中的各视图编码器、解码器、`CrossViewProjectionHead`，以及 `models/trustworthy.py` 中的 `DualWeightedReliabilityEstimator` |
| (c) | Trustworthy Multi-view Multi-label Classification with Uncertainty Quantification | `models/trustworthy.py`：`TrustworthyMultiLabelClassification`、`LabelWiseUncertaintyQuantifier`；共享分类器为 `MultiLabelClassifier` |
| 损失调度 | Dynamic Weight Scheduling | `models/dynamic_weight_scheduler.py`：`DynamicLossWeightScheduler` |

框架 (b) 的可靠性估计器在代码中由辅助可信分类模块持有。此处的模块划分用于说明与框架图的对应关系，计算顺序与原实现保持一致。

## 损失名称

```text
L = alpha * L_vae + beta * L_rec + gamma * L_con + lambda_t * L_trust
L_trust = L_bce + L_tc + L_unc
```

| 论文符号 | 代码名称 | 含义 |
| --- | --- | --- |
| L_vae | `loss_vae` | VAE 视图插补损失 |
| L_rec | `loss_rec` / `ViewReconstructionLoss` | 视图重建损失 |
| L_con | `loss_con` / `MaskedCrossViewContrastiveLoss` | 带掩码和样本邻接信息的跨视图对比损失 |
| L_bce | `loss_bce` / `MaskedBCELoss` | 各视图与主融合预测的标签掩码 BCE 汇总 |
| L_tc | `loss_tc` | 可信预测与停止梯度的主预测之间的一致性损失 |
| L_unc | `loss_unc` | 辅助分支的不确定性损失 |
| L_trust | `loss_trust` | 可信分类模块的组合损失 |

训练日志中的 `loss_vae`、`loss_rec`、`loss_con`、`loss_trust` 保留原先的加权统计方式；`loss_bce` 和 `loss_unc` 保留原先的分项统计方式。重命名没有新增单独的 `loss_tc` 日志列。

当前推理采用潜表示补偿后的平均融合预测。可信权重用于辅助可信分支的融合；现有分类 BCE 的计算方式保持不变。所有参数默认值、掩码处理、梯度截断、损失系数、训练与测试流程均保持原实现。

## 参数敏感性分析

```bash
python parameter_sensitivity_alpha_beta/run_alpha_beta_sensitivity.py
```

具体参数和输出格式见 `parameter_sensitivity_alpha_beta/README.md`。

