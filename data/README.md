# 位点–疾病关联：固定1:1样本

正样本来自相邻目录ptmd_site_5ptm_mmseqs30_v1，保留全部235960条五种PTM Gold+Silver关联。Label=1为已知关联，Label=0为采样的未标注关联，并非经过实验确认的真负例。

## 采样规则

固定seed=20261010。对每个“位点×PTM类型”，从固定的标准化疾病词表中均匀、无放回抽取未观察关联的疾病，负例数等于该位点×PTM的正例数。因此每个分区、每种PTM和每个既定开发折都保持1:1；也保留位点和蛋白的正负样本数平衡。疾病词表采用来源数据的既定候选词表，不按测试疾病频率、模型分数或相似度筛选负例；疾病的正负频率不刻意匹配。

排除表涵盖全部11种PTM的Gold+Silver已知关联，使用矫正后的坐标键，且跨开发/测试分区完整排除。同一负关联不重复采样。

负例的位点和PTM来自该分区中的已知PTM位点，不使用任意未修饰残基。所有样本继承位点的蛋白同源簇、分区和Fold。同一簇不跨开发/测试、每折训练/验证。一次生成开发集负例，再按固定Fold组成五折；不会每折重新采样导致评价候选变化。

## 文件

- development_1to1.csv：373520条，正负各186760；用于交叉验证和最终重训。
- independent_test_1to1.csv：98400条，正负各49200；锁定的最终测试。
- folds/fold_1至fold_5：train_1to1.csv和validation_1to1.csv。
- samples_1to1.csv：471920条全部样本，**不要对这个文件再次随机划分**。
- proteins.fasta、sites.csv、diseases.csv、protein_split.csv、cluster_mapping.csv：与原始位点版本一致。
- known_positive_exclusion_all11.csv：全量已知正关联排除表。
- sample_statistics.csv、sampling_audit.csv、metadata.json：数量、检查、种子和文件SHA256。

Label是训练标签。Evidence_level、Sample_origin和PDA_key不能作为预测特征：前两者直接标识正负样本来源，PDA_key仅作追溯索引。只取允许的序列、位点、PTM和疾病特征；图、GIP、频率、归一化等只由当前训练折构建。数据在同一蛋白内的位置使用1-based坐标，逐残基PLM索引用Sequence_position-1。

当前协议是已有疾病候选词表上的蛋白簇隔离评价，并非专门设计的疾病冷启动评价。1:1的AUPR对应50%采样阳性率，不代表真实全候选空间的发现率；Silver仍为弱证据，不能解释为位点导致疾病的因果证据。Gold结果另行分层报告时，应明确相应负例范围。

## 本地核验

```powershell
conda activate ptm_local
Set-Location 'D:\learn\PTM\ptm疾病相关'
python .\experiment\data\ptmd_site_5ptm_mmseqs30_1to1_v1\code\verify_samples.py
```

所有模型必须共用此固定版本。build_samples.py保留生成逻辑；已生成的发布目录禁止原位重新生成。
