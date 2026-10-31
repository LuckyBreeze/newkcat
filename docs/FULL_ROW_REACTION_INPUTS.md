# 保留全部记录：反应输入与 sr 转换分别处理

sr-SMILES 转换失败不等于动力学标签失效，也不要求删除这条记录。
本版新增完整 reaction-SMILES 独立准备和显式的完整反应回退，并保留原严格与底物回退选项。
模型、标签和划分不自动更换；全量基准以同一批 sample_id 为准。

## 为什么有的合法反应不能直接合并

例如以下净反应元素和形式电荷都守恒：

```text
OO.[Cl-].[H+]>>O.OCl
H2O2 + Cl- + H+ -> H2O + HOCl
```

一种常见映射是：

```text
[OH:1][OH:2].[Cl-:3].[H+]>>[OH2:1].[OH:2][Cl:3]
```

两侧的氧和氯都有对应编号，但左侧的游离质子是独立节点，右侧的氢写在氧原子的
氢计数中。严格的显式节点映射集合检查会拒绝它，普通反应字符串仍能完整表达它。
不能把“显式节点集合不同”与“原子总数不守恒”混为一谈。

本版本不自动删氢或在另一侧补造质子。游离氢、键合氢、氘和不同自由基状态也不能一概而论。
若未来研究“sr 核心 + 氢上下文”扩展，必须把原两侧氢上下文输入模型并验证完整反应可重建；
仅将被移除的氢写入日志仍然是信息丢失。本版的默认全量解决办法保留原完整 reaction-SMILES。

## 1. R 模型可以独立准备，无须映射或 sr 转换器

先按 [主使用说明](SEQUENCE_MODELS.md) 导入原始划分，随后执行：

```bash
newkcat sequence prepare \
  --samples data/sequence_import/samples.csv \
  --output data/reaction_only \
  --max-length 2048 \
  --reaction-only
```

这个入口完全跳过映射 JSONL 与 sr 转换。它保留完整原反应、规范化书写顺序、去映射号并检查长度。
不进行反应补全、去盐或化学结构替换。每行标为 `reaction_only`，准备报告和 hash 仍然保留。
底物与反应物的疑似错接会作为标记输出，不擅自修改基准中的原反应。

复制 `configs/sequence_reaction_all.yaml`，将 `data.samples` 指向上述 `samples.csv`，
`data.features` 指向现有 ProtT5 缓存，即可运行 R。这个准备目录不能直接用于 C；
没有 sr 转换结果不能假装已经得到 CGR 表示。

## 2. 全量比较：能保真转换的记录用 sr，其余仍读完整反应

为 R 和混合表示 C 使用同一个准备目录：

```bash
newkcat sequence prepare \
  --samples data/sequence_import/samples.csv \
  --mapping-jsonl /path/to/by_reaction.jsonl \
  --output data/sequence_full \
  --max-length 2048 \
  --on-error reaction
```

成功 sr 行：`reaction_status=ok`，同时保留完整 R 和经过往返验证的 C。
sr 失败行：`reaction_status=fallback_reaction`，完整 R 保留，sr 留空，动态字段为空。
原始反应本身无效、缺失或超过长度限制时会明确拒绝，不能用这种政策掩盖损坏的 R。

运行配置：

```bash
newkcat sequence train --config configs/sequence_reaction_all.yaml
newkcat sequence train --config configs/sequence_sr_hybrid.yaml
```

两个配置默认读取 `data/sequence_full/samples.csv`，特征路径需核对实际缓存位置。
R 对所有行读取同一字段 `reaction_smiles`。混合 C 在 `ok` 行读取 sr；其余读取同一条完整 R。
两者都使用一个化学 Transformer，不调用 MolT5/MACCS 来填补失败行，也不减少样本数。
原来的 98 字符词表无需改变：R 本来包含 `>>`，sr 使用自己的动态语法；格式状态同时写入审计。

混合 C 的显式配置是：

```yaml
data:
  missing_reaction: reaction
model:
  chemical_encoder: sr_smiles
  pooling: mean
```

未选择这一政策的严格 C 仍会拒绝 `fallback_reaction`，避免实验悄悄换题。
若使用 `pooling: edit_conditioned`，完整 R 回退行没有可信的 sr 动态字段，局部向量为零；
全局反应信息仍然进入同一个化学编码器。没有恢复成键/断键注释的行不能伪造局部标签。

如果某行底物与反应不相符，系统保留原始字段和问题标记，并禁止将其标为可靠 sr。
全量政策保留其原 R 只是复用基准原输入，不表示该错接已经修复。修改这类测量对应关系
应作为独立的数据更正，使所有比较模型使用同一更正版数据。

## 3. 本地 decoder 兼容补丁修复什么

固定的 sr-smiles 0.2.0 逆转换存在两类已复现问题：

- 调整四面体手性时，原正则漏掉 `[C@H:1]` 一类含氢计数及同位素前缀的原子；
  当邻居遍历顺序改变时，未正确调整 `@/@@`，会改变绝对构型。
- 去除“冗余”括号时，`[S]` 可能被改写为 `S`；两者的隐式氢和自由基状态不同，括号并不冗余。

`sr_compat.py` 在独立的函数命名空间修复这些逆转换操作，保留完整原子括号，不修改
安装包、site-packages 或进程内上游模块的全局绑定。兼容标识记录在转换结果及准备报告中。
原有带映射和无映射的完整结构/立体/同位素/电荷往返检查继续执行，未以忽略手性提高通过率。

补丁改善的是转换实现的正确性。往返通过仍不证明实验标签或原始原子映射的化学机制正确。

## 4. 论文与评测怎样表述

主比较是 **R 全量 vs sr/完整 R 混合表示全量**，共享原有训练、验证和测试 sample_id。
它检验“在相同完整反应信息下，将可可靠转换的反应显式凝聚，能否改善全量预测”。
不能将混合版本写成“全部样本均采用纯 sr-SMILES”。

`run.json`、`summary.json` 和评估输出包含 `representation_policy`；checkpoint 保存政策，
推断不会因为 YAML 被修改而自动切换。`metrics_by_input_status` 提供相同状态行的分层指标，
主指标仍以全部测试记录计算。R 的输入状态可保留 C 可用性的分层信息，尽管 R 对所有行都读完整反应。

成功转换子集上的 R/C 配对分析可以作为补充诊断，不能用该子集的指标替换全量主结果。
不要将“增加一条读取方式”的效果归因于新增信息：两种字符串表达的是同一条原反应。
若后续加入额外反应预训练，需单独控制语料、预算和划分。

当前 MLM 的 sr 输入是纯 sr 语料；它不会自动把混合 R/C 数据当作纯 sr 预训练。
可以在纯 sr MLM 权重上微调混合回归，但应明确这项输入分布变化；不能默认冻结后即可同样处理 R 回退。
