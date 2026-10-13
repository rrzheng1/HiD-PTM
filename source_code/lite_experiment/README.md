# H2-Lite验证：L0–L3

独立实验目录，固定v1数据及MMseqs30蛋白质同源簇五折。H2-Lite保留31aa ESM、全蛋白平均ESM、PTM类型、BiomedBERT语义、MONDO GCN、疾病条件Cross-Attention、两级Site–Protein–Disease门控和最终交互门控/MLP。没有One-hot、BLOSUM62或传统描述符CNN。

| 实验 | 改变 |
|---|---|
| L0 | F3等价的H2-Lite，复用已完成F3结果与核心权重 |
| L1 | 删除两级显式层次门控及其归一化；仍保留全蛋白ESM加入每个局部token |
| L2 | 删除全蛋白ESM投影及所有依赖它的层次交互，只保留局部ESM的位点编码 |
| L3 | 删除MONDO GCN传播，用原BiomedBERT文本投影作为疾病表示，其余保持Lite |

L2是全蛋白背景及层次交互的联合移除；L1对L0检验显式层次机制，L2对L1进一步隔离全蛋白直接融合的信息贡献。L3仍使用MONDO名称、定义与同义词文本，不等价于移除全部MONDO知识。没有BiGRU、MoE或GAT。

## 运行

```powershell
conda activate ptm_local
Set-Location 'D:\learn\PTM\ptm疾病相关'
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiment\h2_lite_validation_v1\run_all.ps1 -Device cuda -BatchSize 64
```

默认先核验/复用L0，再训练L1–L3，共新增15次训练。单组可加`-Variant L1`、`L2`、`L3`；`-Variant L0`仅核验/复用，不重复训练。

全部采用seed42、AdamW lr0.0003、weight_decay0.0001、最大40轮、patience5/min_delta0.0001、batch64。以验证AP早停，微小改善不延长等待，最高实际AP仍保存best.pt，评价恢复最佳模型。last.pt保存优化器、早停及随机状态，重跑原命令可续跑/跳过完成折。对复用L0严格核验F3的训练参数、数据与文件SHA；若更改参数或seed，需有对应设置的F3，不能混合比较。

只读复用原特征缓存，训练输入读取ESM窗口、有效位置mask和蛋白平均表示；不读取One-hot/BLOSUM描述符输入。L2不搬运全蛋白向量到GPU。为保持与F3的共同初始化一致，构造器先初始化原H2，再删除弃用模块；**最终模型实际不包含传统CNN参数**。

## L0模型与来源

`results/L0/seed_42`逐文件复制F3原始结果，源配置中的F3名称和原完成标记不改写。`reused_from_F3.json`记录来源与SHA，原目录不修改。

每折另存`compact_best.pt`：只删除原F3中未使用的`local.*`、`local_projection.*`和`blosum`，其余训练权重不变，可用本目录`SiteDiseaseModel(..., variant='L0')`严格加载。这是权重精简导出，没有再训练，没有重新选择epoch。原`best.pt`和`last.pt`仍作为来源完整保留；原last.pt是F3优化器，不能直接作为精简模型的续训检查点。

L1–L3的每折保存本次精简结构的best.pt、last.pt、配置、历史和完整验证预测。

## 输出

- 每组`results/L0`至`L3`下的`seed_42/fivefold_results.md`和`fivefold_mean_std.csv`：五种PTM＋Overall六行、七指标、五折均值±标准差。
- 完成后`reports/L0_L3_results.md`、`all_mean.csv`、`all_std.csv`：完整四组比较。
- `paired_AP_differences.csv`：各组相对L0的同折AP差值、标准差和提高折数，不自动当成显著性检验。
- `training_audit.csv`：实际轮数、最佳epoch、L0来源复用标记。

阈值0.5，pr_auc采用AP，Overall先合并每折样本计算指标再取折均值。仍保留Disease-only、同疾病内AUC和位点打乱诊断，v1疾病先验捷径必须继续考虑。此次沿同一开发集探索结构，不声称独立验证模型选择；没有开始独立测试评价。

代码已验证Lite等价于关闭描述符分支的H2、严格权重转换、物理删除模块、各变体梯度及移除输入不影响输出。正式训练由用户运行。
