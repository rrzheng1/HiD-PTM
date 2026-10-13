# HiD-PTM：最终模型归档 v1

**HiD-PTM — Hierarchical Disease-conditioned PTM Association Model**，中文为“疾病条件层次交互PTM位点关联模型”。这是本项目暂定名称，不宣称名称唯一或架构已获独立验证。

固定架构为L0/H2-Lite，与F3计算路径等价：31aa ESM与全蛋白平均ESM、PTM类型和位置编码→疾病条件Cross-Attention→Protein–Disease及Site–Protein–Disease两级门控→交互门控→MLP/Sigmoid。疾病使用冻结BiomedBERT语义及两层MONDO GCN。模型没有One-hot、BLOSUM62、传统描述符CNN、BiGRU、MoE或GAT。

## 文件组织

- `model.py`、`h2_reference.py`、`baseline.py`：精简模型实现依赖，实际构造后不存在传统CNN分支。
- `models/fold_1/best.pt`至`fold_5/best.pt`：五折最佳精简权重，即L0的compact_best.pt原样复制，可直接严格加载。
- `results/cross_validation/seed_42`：完整五折结果、OOF预测、训练历史、来源配置、原始best/last及精简权重。保留F3源名称，不改写历史记录。
- `source_code/lite_experiment`：L0相关实验代码原样快照。
- `source_code/original_F3_training`：实际产生训练权重的F3训练代码快照。
- `metadata/dataset`：原数据集划分、序列、位点/疾病目录及原metadata。
- `metadata/features`：共享特征来源清单、疾病节点顺序及预训练文本/ESM来源记录。
- `archive_manifest.json`：名称、来源、训练设置及全部归档文件SHA256。

源代码快照保留原相对路径，用于追溯，不应在快照目录直接运行训练；如需复现实验，仍使用原`experiment/h2_feature_ablation_v1`或`h2_lite_validation_v1`启动脚本。原数据和8.35GiB特征缓存只读引用，没有再次复制大型缓存。应继续保留`experiment/site_disease_cross_attention_v1/cache`及`experiment/embedding/esm2_embedding`。

## 查看结果与加载模型

结果表：`results/cross_validation/seed_42/fivefold_results.md`。五种PTM＋Overall六行、七指标，五折均值±标准差；pr_auc采用AP，阈值0.5。包含Disease-only、同疾病内部AUC和位点打乱诊断，v1的疾病先验影响应如实报告。

```powershell
conda activate ptm_local
Set-Location 'D:\learn\PTM\ptm疾病相关'
python .\experiment\HiD_PTM_final_v1\load_model.py
```

程序中调用本目录`load_model.load_model(fold=1,device='cuda')`。输入次序为`model(esm_window, protein_mean, None, mask, ptm_index, disease_index)`，PTM索引依次Acetylation、Methylation、Phosphorylation、SUMOylation、Ubiquitination；疾病索引必须使用`metadata/features/disease_nodes.csv`顺序。输出logit，概率为`torch.sigmoid(logit)`。只加载可信本地检查点。

## 当前归档状态

保存的是五个交叉验证最佳模型，尚未用全部开发集重训一个部署模型。L0的独立测试尚未评估；之前DIME-PTM独立测试结果不属于HiD-PTM，不能混用。原last.pt为F3训练优化器，仅为来源记录，不能直接当成精简模型续训状态。

此次只命名和归档，未重新训练、未改动原模型或结果。早停原设置为patience5/min_delta0.0001，监控验证AP。
