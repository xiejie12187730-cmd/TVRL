TVI-Net：Trustworthy Variational Imputation Network

训练入口：train.py
模型实现：models/TVINet.py，Python 类名 TVINet。

框架图对应：
(a) VAE-based View Imputation：VAEViewImputation。
(b) Dual-weighted Multi-view Representation Learning：视图编码与重建、跨视图对比学习、DualWeightedReliabilityEstimator。
(c) Trustworthy Multi-view Multi-label Classification with Uncertainty Quantification：TrustworthyMultiLabelClassification、LabelWiseUncertaintyQuantifier，以及共享的 MultiLabelClassifier。

L = alpha*L_vae + beta*L_rec + gamma*L_con + lambda_t*L_trust
L_trust = L_bce + L_tc + L_unc

本次只统一代码和说明中的名称；计算逻辑与参数保持不变。
详细名称对照、运行说明和迁移说明见 README.md。
