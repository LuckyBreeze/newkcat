# 用反应序列替换 CataPro 的化学编码器

本分支 `feat/reaction-sequence-encoder` 增加了独立的 `newkcat sequence` 命令组。
它实现“缓存的 ProtT5 蛋白表示 + 一个化学序列编码器 + 回归头”，允许完整替换
MolT5、MACCS 和底物图分支。原有图分支命令仍然保留，便于对照已有实验。

**这是一套可运行的研究代码和比较流程，没有实际数据上优于 CataPro 的结论。**
新的字符 Transformer 默认从随机参数开始训练；它不是 sr-SMILES 论文中大规模
预训练 BERT 的复现，也没有附带或自动下载该论文的预训练权重。

**全量数据入口更新：** 普通 R 输入不需要 sr 转换成功；新增 `prepare --reaction-only`。
需要保持全部记录的 C 实验，可显式选择 `--on-error reaction` 与 `missing_reaction: reaction`，
sr 失败时仍读取完整原反应。详见 [全量反应输入说明](FULL_ROW_REACTION_INPUTS.md)。
下文原有严格／共同底物回退选项继续保留，均不自动删除记录。

## 1. 五个比较版本分别回答什么

| 版本 | 配置文件 | 化学输入 | 要回答的问题 |
| --- | --- | --- | --- |
| B0 | `configs/sequence_baseline.yaml` | 原有 MolT5 768 维 + MACCS 167 维缓存 | 在统一训练器和协议下，CataPro 回归结构的表现如何？ |
| S | `configs/sequence_substrate.yaml` | 主底物 SMILES | 换成小型序列骨干本身有什么影响？ |
| R | `configs/sequence_reaction.yaml` | 完整 `reactants>>products` | 提供产物及其他反应组分有什么影响？ |
| C | `configs/sequence_sr.yaml` | sr-SMILES | 相同反应信息下，显式凝聚表示有什么影响？ |
| C-L | `configs/sequence_sr_local.yaml` | sr-SMILES + 酶条件化的动态字段读取 | 对动态字段单独读取是否有帮助？ |

五个版本都复用同一组样本 ID、原有划分、标签和 ProtT5 缓存。S、R、C 使用相同的
字符词表、骨干规模和全局池化；C-L 增加局部读取参数，参数量会记录在运行结果中。

B0 使用原有 `NewKcat(mode="baseline")` 回归结构，仍包含其 BatchNorm。
本分支示例对所有版本统一设置 `dropout: 0.1`，因此 **B0 是匹配当前训练协议的比较基线，
不是 CataPro 作者训练设置或历史指标的自动复现**。模型结构相容不能替代原始实验复现。

B0/S 的输入条件是“酶 + 底物”；R/C/C-L 的输入条件是“酶 + 已知净反应”。
若实际部署只有底物而没有产物，不能把 R/C 的成绩当作相同输入条件下的提升。
R 与 C 的比较更适合判断表示形式；B0 与 C 的比较包含信息范围、预训练资源和架构的共同变化。

## 2. 安装与分支

首次获取本分支：

```bash
git clone --branch feat/reaction-sequence-encoder https://github.com/LuckyBreeze/newkcat.git
cd newkcat
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[sequence,test]'
```

已克隆仓库时，先保存工作区已有改动，再切换到该分支；实验及修改不需要在 `main` 上进行。
Windows PowerShell 的环境激活命令为 `.venv\Scripts\Activate.ps1`。
CUDA 机器应先安装与驱动相符的 PyTorch，再安装项目；示例配置默认使用 CPU。

依赖分为两部分：

- 基础训练依赖为 PyTorch、NumPy、RDKit、PyYAML。训练时读取现成的 ProtT5 缓存，不下载蛋白模型。
- `sequence` 可选依赖固定 `sr-smiles==0.2.0`，用于审计后的 R/C 成对预处理。转换器版本不符会拒绝运行。

完成预处理后，序列训练不再调用 sr-SMILES 转换器；如果集群环境不适合安装该包，
可以在独立预处理环境生成数据，再完整复制准备产物到训练环境。基础 RDKit 依然用于输入校验。

## 3. 先用合成示例确认流程

```bash
python examples/make_sequence_demo.py
newkcat sequence validate --config runs/sequence_demo/sr.yaml --split train
newkcat sequence validate --config runs/sequence_demo/sr.yaml --split val
newkcat sequence train --config runs/sequence_demo/sr.yaml
newkcat sequence evaluate \
  --config runs/sequence_demo/sr.yaml \
  --checkpoint runs/sequence_demo/models/sr/best.pt \
  --split test --output runs/sequence_demo/evaluation/sr
newkcat sequence predict \
  --config runs/sequence_demo/sr.yaml \
  --checkpoint runs/sequence_demo/models/sr/best.pt \
  --split test --output runs/sequence_demo/predictions/sr
```

生成器提供 `baseline.yaml`、`substrate.yaml`、`reaction.yaml`、`sr.yaml`、`sr_local.yaml`
和可选预训练迁移的 `sr_pretrained.yaml`。各模型输出到 `runs/sequence_demo/models/<模式>`。
示例包含 4 条训练、2 条验证、2 条测试记录，含通配原子，特征和标签为合成数据。
它用于检查数据接入、转换、保存与重载，不能用于判断模型精度或生物学有效性。
示例生成器拒绝覆盖非空目录。

可选的 MLM 检查流程为：

```bash
newkcat sequence pretrain \
  --config runs/sequence_demo/sr.yaml \
  --train-corpus runs/sequence_demo/mlm/train_sr.txt \
  --val-corpus runs/sequence_demo/mlm/val_sr.txt \
  --output runs/sequence_demo/mlm_output
newkcat sequence train --config runs/sequence_demo/sr_pretrained.yaml
```

这些预训练示例文本与下游示例分开，也不能视为具有实用化学知识的预训练模型。

## 4. 真实数据的三个输入层次

### 4.1 样本表：保留 ID、标签和原 cluster40 划分

统一的 `samples.csv` 需要以下列：

| 列 | 含义 |
| --- | --- |
| `sample_id` | 全表唯一且非空，与特征缓存中的 ID 一致 |
| `sequence` | 酶序列，用于身份及跨划分检查 |
| `smiles` | 这条动力学记录对应的主底物 |
| `log10_kcat` | 已有的 `log10(kcat / s^-1)` 标签 |
| `split` | `train`、`val` 或 `test` |
| `group_id` | 可选，已有序列聚类 ID；应尽量保留 |
| `reaction_smiles` | R/C 所需的完整净反应，可含映射，格式为 `reactants>>products` |
| `mapped_rxn` | 可选，已完成原子映射的完整反应；与上一列描述的结构必须一致 |

序列路线没有 EC 输入，也不会通过 EC 猜产物。`reaction_smiles` 和 `mapped_rxn` 可以来自
经审计的外部反应数据，但必须先与具体测量记录对齐。仅有 EC 或主底物不能唯一决定完整反应。

已有三个划分 CSV 时，可按列名接入：

```bash
newkcat sequence import-splits \
  --train /path/to/train.csv \
  --val /path/to/val.csv \
  --test /path/to/test.csv \
  --id-column row_uid \
  --sequence-column sequence \
  --smiles-column substrate \
  --target-column log10kcat_max \
  --reaction-column reaction_smiles \
  --group-column cluster_id \
  --output data/sequence_import
```

命令保留原列并添加统一字段，合并为 `data/sequence_import/samples.csv`；不会重算标签、
重新编号样本或重新生成划分。`--target-column` 必须已经是所需的对数标签，不能传入原始
`kcat` 后期待它自动取对数。源表若有 `split` 列，必须与其所属文件一致。
原表的 `valid`/`validation` 可规范为 `val`。输出目录同时包含来源 hash 和计数报告。

现有标准格式表可以跳过导入。若从官方 CataPro 的原始数值和 pickle 缓存开始，
使用原仓库的 [官方格式导入流程](../README.md#3-接入你现有的-catapro-数据)，
不要混用两种导入器的标签变换语义。

### 4.2 特征缓存：序列版本只需要 ProtT5

序列版本读取安全的 NPZ 文件，按 ID 连接，不依赖表格与矩阵的行顺序：

```python
import numpy as np

np.savez_compressed(
    "data/sequence/features.npz",
    sample_ids=np.asarray(ids, dtype=str),  # [N]，Unicode，禁止 object 数组
    protein=np.asarray(prott5_vectors, dtype=np.float32),  # [N, 1024]
)
```

| 数组 | S/R/C/C-L | B0 |
| --- | --- | --- |
| `sample_ids` | 必需，唯一 Unicode ID | 必需 |
| `protein` | 必需，`[N, 1024]` 有限数值 | 必需 |
| `molt5` | 不读取、不要求存在 | 必需，`[N, 768]` |
| `maccs` | 不读取、不要求存在 | 必需，`[N, 167]`，0/1，第 0 位为 0 |

因此，已有四数组 CataPro NPZ 可以直接供五个版本共用；无需删除 MolT5/MACCS。
只有蛋白数组的 NPZ 可以训练序列版本，但不能运行 B0。各数组行数必须与 `sample_ids`
一致，所选样本都必须有唯一缓存记录。常规加载使用 `allow_pickle=False`。

保持原 ProtT5 的截断、特殊 token 处理和池化方式一致；本分支不重新提取蛋白嵌入。
`import-splits` 和 `prepare` 只处理样本/反应，不负责生成、复制或重排特征缓存。
先完成下一节的准备，再将已有 NPZ 放到 `data/sequence/features.npz`；也可以在 YAML 中
直接填写缓存的现有路径。默认准备命令要求输出目录为空，提前放入 NPZ 会触发非空目录检查。

### 4.3 完整映射反应：不是原有 centers.jsonl

旧图路线的 `centers.jsonl` 只有“某个底物原子的分数/参考标签”。它不能恢复产物、
共底物、完整键变化或 sr-SMILES，不能直接改名当作新路线的输入。

映射反应可放在样本表的 `mapped_rxn` 列，或提供单独 JSONL：

```json
{"reaction_smiles":"CC(=O)C>>C=C(C)O","mapped_rxn":"[CH3:1][C:2](=[O:3])[CH3:4]>>[CH2:1]=[C:2]([OH:3])[CH3:4]","map_status":"ok"}
```

该例只展示酮–烯醇净转化及字段格式，不是某条已验证的酶测量。JSONL 通过原始
`reaction_smiles` 字符串精确查找；同一键不能重复，非 `ok` 条目不作为有效映射。
程序会比较原始与映射反应的规范化结构，拒绝用相似但不相同的反应替换原记录。

映射输入优先顺序是：样本自身的 `mapped_rxn`；否则使用指定 JSONL 中的匹配结果；
如果没有指定 JSONL，则尝试把样本 `reaction_smiles` 自身作为已映射反应读取。
本代码不运行 RXNMapper/EnzymeMap，不自动补齐缺失原子或推断反应方向。

## 5. 成对预处理 R/C，并先看覆盖率

默认采用严格策略：

```bash
newkcat sequence prepare \
  --samples data/sequence_import/samples.csv \
  --mapping-jsonl /path/to/reaction_mappings.jsonl \
  --output data/sequence \
  --max-length 2048
```

如果映射已在表中，可省略 `--mapping-jsonl`；自定义列名使用 `--mapped-column`。
准备目录必须与源表目录分开，不能原地覆盖输入。

同一次准备产生普通 reaction-SMILES 和 sr-SMILES；二者都去掉原子映射号，不把任意编号
作为模型特征。程序记录动态字段范围并执行带映射及去映射后的往返转换。
主底物的规范化组分必须包含在映射反应物的组分多重集合中，不能仅凭指纹相似就接入。

主要产物如下：

| 文件 | 用途 |
| --- | --- |
| `samples.csv` | 原 ID/标签/split，加两种序列、状态和 `edit_spans` |
| `coverage.json` | 全量及各 split 成功/失败数、长度分布、版本、政策和 hash |
| `conversion_errors.csv` | 失败样本 ID、split 和具体原因 |
| `reaction_provenance.jsonl` | 原反应、映射反应、转换结果与对应关系审计记录 |

**默认只要有一条转换失败，就写出审计报告并停止，不生成可训练的 `samples.csv`。**
如果同目录内曾有旧的准备表，失败的严格重跑会移除旧表，避免误用上轮数据。
程序不会悄悄删除那一行，也不会修改它的标签或划分。
重新使用一个已有准备目录时需要明确传入 `--overwrite`；更稳妥的实验记录方式是使用新目录。

R/C 加载要求 `coverage.json` 和 `reaction_provenance.jsonl` 与 `samples.csv` 同目录，
并检查准备产物的 hash。手工编辑准备后的表格会导致校验失败；应修正上游输入再准备。
迁移到集群时请复制这些产物，不要只复制一个 CSV。

### 明确选择“所有记录保留，共同退回底物”时

如果实验决定保留全部记录，对无法可靠使用完整反应的条目采用底物序列，需要同时设置两处：

```bash
newkcat sequence prepare \
  --samples data/sequence_import/samples.csv \
  --mapping-jsonl /path/to/reaction_mappings.jsonl \
  --output data/sequence \
  --max-length 2048 \
  --on-error substrate --overwrite
```

这里的 `--overwrite` 允许替换刚才严格审计写出的报告；也可以去掉该选项并使用一个新的空目录。

并在参与 R/C 比较的 YAML 中写：

```yaml
data:
  samples: ../data/sequence/samples.csv
  features: ../data/sequence/features.npz
  missing_reaction: substrate
```

R 和 C 使用**完全相同的回退行**：即使 R 本身能解析，只要其对应 C 转换不可靠或任一
表示超长，这一行在两者中都使用主底物序列。C-L 在这些行没有动态字段，局部向量严格为零，
但仍保留蛋白表示和底物序列的全局读取。这里没有切回旧 MolT5/MACCS 分支。

这个策略保留样本与划分，但改变了模型实际可见的信息，应报告成功与回退覆盖率及分层表现。
它是显式实验设置，不是学习出的可靠性门控。若选择干净子集，应另外生成并记录固定子集，
在同一子集重训所有比较版本；本工具不自动替你选择或删除数据。

### `*`、显式氢和“配平”到底如何处理

- **`*`：** 底物序列允许 RDKit 通配/虚原子，字符词表也能无损表示。完整反应仍需逐条通过映射与往返校验。含 `*` 的记录不再因旧图特征器的限制而必然失败，但必须对实际数据重新审计；程序不会恢复通配原子的未知结构。
- **显式氢：** 解析时保留 `[H]` 节点；转换使用 `remove_hydrogens=False`。所有显式节点，包括游离质子和底物显式氢，都需要正且唯一的映射号，两侧必须对应。程序不把游离质子悄悄删掉，也不统一删除显式氢来“修复”对齐。
- **映射一致性：** 两侧映射号集合必须相同，对应元素和同位素必须相同。形式电荷、键和隐式氢状态可以变化，并接受往返核对。隐式氢数量可编码为动态原子状态，这不等于补足了缺失共底物。
- **配平边界：** 转换禁用自动配平。显式映射节点集合一致不等于已经验证所有隐式氢、电荷/电子或完整反应计量守恒。往返成功说明表示保真，不能证明原始原子映射、生化反应或限速步骤正确。
- **表示范围：** 不自动去盐、改变质子化、选择互变异构体或保留最大组分。完整反应采用 `reactants>>products`，不接收独立 agents 段；CXSMILES 增强立体组不在当前支持范围。超长序列明确报错，不截掉尾部的反应变化。

损坏的主底物、无法保留语义的输入或底物本身超长不属于可静默回退的问题；应在数据审计中解决。
R/C 的完整反应覆盖率必须重新计算，不能用旧的“底物与中心能对齐”比例代替。

## 6. 配置与模型定义

以 `configs/sequence_sr.yaml` 为例，默认配置结构为：

```yaml
data:
  samples: ../data/sequence/samples.csv
  features: ../data/sequence/features.npz
  missing_reaction: error
model:
  chemical_encoder: sr_smiles
  vocab_size: 98
  protein_dim: 1024
  d_model: 128
  nhead: 4
  num_layers: 3
  dim_feedforward: 512
  head_dim: 256
  dropout: 0.1
  max_length: 2048
  pooling: mean
training:
  seed: 668
  batch_size: 16
  epochs: 150
  patience: 20
  lr: 0.0001
  weight_decay: 0.0001
  min_delta: 0.0
  grad_clip: 5.0
  num_workers: 0
  num_threads: 1
  device: cpu
  deterministic: false
encoder_checkpoint: null
freeze_encoder: false
output_dir: ../runs/sequence_sr
```

YAML 中的路径相对该 YAML 所在目录解析；CLI 的 `--output` 相对当前工作目录。
`max_length` 包括 BOS/EOS，准备和模型长度限制应一致。初始超参数是实验起点，不是已验证最优设置。
蛋白输入当前固定为 1024 维；不提供其他维度缓存的自动适配。

`chemical_encoder` 可选 `catapro_features`、`substrate_smiles`、`reaction_smiles`、`sr_smiles`。
`pooling: edit_conditioned` 仅用于 `sr_smiles`。未知配置项和 YAML 重复键会报错，避免拼写错误被忽略。
`batch_size` 至少为 2，所有版本共用避免 BatchNorm 单样本尾批的采样器；每条训练记录每轮访问一次。

### 编码器实际做了什么

固定词表包含 ASCII 33–126 的 94 个非空白字符，加 PAD/BOS/EOS/MASK，共 98 个 token。
每个字符一个 token；不依赖原 MolT5 词表，不学习测试集词表，没有 `[UNK]` 替换。
不支持的字符或超长输入会报错。不同等价 SMILES 的表示不保证相同，因此准备时固定规范化流程。

化学塔是带绝对正弦位置编码的双向 Transformer。位置表示字符串顺序，**不是图距离、
反应中心距离、RoPE 或三维几何**。蛋白侧使用缓存向量，经 LayerNorm 和投影进入回归头。

设反应 token 隐状态为 `H`，蛋白向量为 `e`：

- C 的全局化学表示来自所有有效字符的均值，排除 BOS、EOS、PAD。
- C-L 保留同一全局表示，并用蛋白投影作为 query，在动态字段的字符上做注意力读取。
- 全局与局部结果来自同一个化学塔，投影后再与蛋白表示共同回归 `log10_kcat`。
- 没有动态字段时，局部结果在带偏置的投影之后再置零，避免“缺失信息”制造常量信号。

局部位置由解析 sr-SMILES 的 `{before|after}` 动态原子/键字段得到，包含这些字段内的字符。
**当前没有把端点邻接原子扩展为一至两跳邻域，也没有残基级交叉注意力。**
动态字段隐状态已经经过全局自注意力，所以局部读取并不等于隔离出来的局部子图。
注意力是酶条件化的读取权重，不能直接解释成原子–残基接触、反应机制或过渡态。

## 7. 正式训练、独立评估和无标签预测

先检查训练、验证两部分：

```bash
newkcat sequence validate --config configs/sequence_sr.yaml --split train
newkcat sequence validate --config configs/sequence_sr.yaml --split val
```

`validate` 不训练模型。省略 `--split` 时会检查全部 split，包括测试标签和输入；
如果希望训练准备阶段不访问测试标签，请按上面的方式分别检查 train/val。
完整 manifest 的 ID/序列/group 一致性和文件 hash 仍会检查，不会据此拟合测试统计量。

使用统一协议训练各版本：

```bash
newkcat sequence train --config configs/sequence_baseline.yaml
newkcat sequence train --config configs/sequence_substrate.yaml
newkcat sequence train --config configs/sequence_reaction.yaml
newkcat sequence train --config configs/sequence_sr.yaml
newkcat sequence train --config configs/sequence_sr_local.yaml
```

训练只实例化 train/val 的标签、输入及所选特征，不在训练结束时自动计算测试成绩。
各版本统一使用 AdamW、MSE、梯度裁剪、验证 RMSE 和早停；checkpoint 总是保存最低验证 RMSE，
`min_delta` 仅影响早停耐心。不同版本应使用不同输出目录。

可以显式覆盖编码器/池化/输出目录，例如：

```bash
newkcat sequence train \
  --config configs/sequence_sr.yaml \
  --encoder sr_smiles --pooling edit_conditioned \
  --output runs/my_sr_local_seed668
```

默认输出目录非空会报错，确认允许重做该实验时才传 `--overwrite`。输入样本、特征、配置和
checkpoint 所在目录不能被输出目录包含，以防覆盖输入。更换 seed 应复制配置并修改
`training.seed` 与 `output_dir`，不要复用同一目录堆叠实验。

每次成功训练保存：

| 文件 | 内容 |
| --- | --- |
| `best.pt` | 最低验证 RMSE 的权重、架构、tokenizer、信息条件和来源信息 |
| `config.resolved.yaml` | 解析后的有效配置 |
| `history.jsonl` | 每轮训练 RMSE 和验证指标 |
| `validation_predictions.csv` | 最佳权重对应的验证预测，按样本 ID 输出 |
| `summary.json` | 最佳 epoch、参数量、覆盖率、验证指标和 checkpoint hash |
| `run.json` | 运行状态、配置、数据 hash、划分身份和初始化来源 |

锁定训练结果后，单独运行测试：

```bash
newkcat sequence evaluate \
  --config configs/sequence_sr.yaml \
  --checkpoint runs/sequence_sr/best.pt \
  --split test \
  --output runs/sequence_sr_test
```

输出 `predictions.csv` 和 `metrics.json`。预测文件包含 `sample_id`、预测、输入状态以及评估目标。
评估还输出按输入状态分层的 `metrics_by_input_status`；全量指标仍为主指标。
指标在 `log10(kcat / s^-1)` 空间计算；测试集还会与 checkpoint 中的 train/val 样本 ID、
精确序列 hash 和 group ID 检查交集。精确身份检查不等于重新验证 40% 序列相似性聚类，
原 cluster40 划分仍需在上游确认。

评估时模型结构、表示、tokenizer 和回退政策以 checkpoint 为准；更改 YAML 不会把已经
训练好的 R 权重变成 C 模型。数据文件路径和计算设备可以改变，输出会记录新旧数据 hash。

无标签预测：

```bash
newkcat sequence predict \
  --config configs/sequence_sr.yaml \
  --checkpoint runs/sequence_sr/best.pt \
  --output runs/sequence_sr_predict
```

`predict` 可用 `--split test` 等选定部分；省略时预测当前 manifest 全部记录。
表中仍需保留 `log10_kcat` 列，但可为空；预测不会读取或验证其数值，也不会计算指标。
新数据仍需提供该 checkpoint 所需的完整输入与准备产物。

## 8. 可选 MLM 预训练和严格迁移

### 8.1 先准备独立的反应语料

MLM 接口接收两个独立文件：训练语料和验证语料。支持：

- UTF-8 `.txt`：每行一个去映射的化学字符串，不允许空行或字符串内空白。
- `.csv`：默认读取与 `chemical_encoder` 同名的列，可用 `--text-column` 指定。若含 `split` 列，训练文件只能为 `train`，验证文件只能为 `val`/`valid`。

程序会去除每个文件内部的完全重复文本，记录计数，拒绝训练/验证文本交集。
可用 `--exclude-corpus` 从两边显式排除给定文本，过滤后两边都必须非空。
任何原子映射编号都不得进入预训练文本。

**这个内置检查只比较完全相同的字符串，不等于唯一化学反应去重。**
不同遍历、组分顺序或等价表示可能描述同一反应；请先在上游按统一的反应身份去重、划分，
再从同一反应列表分别生成 R 与 C 语料。要报告严格归纳泛化，还需要按实验定义排除
下游验证/测试反应，必要时排除相关规则/家族，并记录筛选政策。
外部未标注预训练中是否包含下游测试化学必须披露，不能用“没有 kcat 标签”代替该说明。

### 8.2 预训练命令

```bash
newkcat sequence pretrain \
  --config configs/sequence_sr.yaml \
  --train-corpus data/pretrain/train_sr.txt \
  --val-corpus data/pretrain/val_sr.txt \
  --exclude-corpus data/pretrain/exclude_sr.txt \
  --mask-probability 0.15 \
  --output runs/pretrain_sr
```

排除文件是可选的；未使用时删除该参数。R 对照使用相同反应身份与划分生成的
`train_reaction.txt` / `val_reaction.txt` / `exclude_reaction.txt`：

```bash
newkcat sequence pretrain \
  --config configs/sequence_reaction.yaml \
  --train-corpus data/pretrain/train_reaction.txt \
  --val-corpus data/pretrain/val_reaction.txt \
  --exclude-corpus data/pretrain/exclude_reaction.txt \
  --output runs/pretrain_reaction
```

预训练使用 YAML 中的骨干和训练配置；不读取蛋白特征或 kcat 标签，虽然共用配置结构仍需有
`data.samples`/`data.features` 路径字段。预训练不接受已设置的 `encoder_checkpoint` 或
`freeze_encoder: true`，不会隐式续训旧模型。

MLM 只在化学字符上选择监督位置，BOS/EOS/PAD 不参与。每条序列至少选择一个字符，
被选位置采用 80% MASK、10% 随机合法字符、10% 保持原字符；验证集掩码按 seed 和文本固定，
训练掩码按 epoch 变化。验证损失按被监督 token 数加权，最低验证损失决定 `best_encoder.pt`。
同时保存 `history.jsonl`、`summary.json`、`run.json`，包含文本来源 hash、去重/排除计数和随机初始化声明。

### 8.3 迁移到回归

复制 `configs/sequence_sr.yaml` 为一个新的实验配置，保持编码器参数一致，并设置：

```yaml
encoder_checkpoint: ../runs/pretrain_sr/best_encoder.pt
freeze_encoder: true
output_dir: ../runs/sequence_sr_pretrained_frozen
```

然后执行：

```bash
newkcat sequence train --config configs/sequence_sr_pretrained.yaml
```

上述命令假定已将复制的配置保存为 `configs/sequence_sr_pretrained.yaml`。
`freeze_encoder: true` 会冻结化学塔并关闭其 dropout，回归部分照常训练；
`false` 会从相同预训练权重开始联合微调。冻结随机编码器会被拒绝。

迁移校验包括表示名称、tokenizer 元数据及指纹、编码器配置指纹、张量键/shape/dtype 和有限值。
R 的 MLM 权重不能误装到 C 配置；包括 `dropout`、`max_length` 在内的编码器设置必须匹配。
蛋白/回归头和局部读取不属于 MLM 编码器权重，因此同一 sr 化学塔可用于 C 或 C-L。

### 8.4 如何公平比较预训练收益

至少把以下比较分开报告：

1. R 与 C 都从随机参数开始，以相同下游训练预算比较表示。
2. R 与 C 使用同一批外部反应身份、相同预训练划分和声明的预算，再比较迁移结果。
3. 固定预训练权重后，分别比较冻结和微调；不要只给表现较好的版本额外调参预算。

相同 epoch/批量记录数不意味着相同 token 数或运算量，因为 R 与 C 的长度不同。
同时记录唯一反应数、更新次数、token 长度、训练时间和所选预算口径。
小模型从零训练输给已有充分预训练的 MolT5，不能单独否定凝聚表示；额外预训练后提升，
也不能全归因于字符串格式。

## 9. 推荐的实验顺序与本版边界

先完成样本/标签/特征身份校验和成对准备审计，再用同一协议训练 B0、R、C。
若要分离“换骨干”与“增加反应信息”，加入 S；C-L 用来检验动态字段读取，不应替代 C 对照。
先根据验证结果固定配置，再执行独立测试，不在多个测试结果间反复挑参数。

多 seed、多 fold、共同有效子集和覆盖率分层都应配对比较。若研究新化学泛化，可在已有
cluster40 之外另建按反应规则划分的评测，但该划分需单独设计并保存；代码不会自动重切数据。
对于含 `*`、显式氢、映射歧义或缺失反应的记录，保留错误原因和预先选定的处理政策。

本分支尚未实现：自动反应补全/原子映射、图距离的一至两跳局部扩展、残基级交互、
三维结构、反应生成、真实过渡态计算、条件感知动力学以及真实数据上的性能结论。
净结构变化并不确定限速步骤，`kcat` 也不只由单个化学步的活化能决定；代码的目标是
检验这种表示是否带来可泛化的预测信息，而非把字符串解释为完整催化机理。

实现入口：`src/newkcat/sequence/{chemistry,prepare,tokenizer,data,model,runner,pretrain,cli}.py`。
对应的 `tests/test_sequence_*.py` 覆盖转换往返、ID 对齐、padding、缺失局部、checkpoint、
预训练迁移及训练/测试边界。测试和合成示例验证实现行为，不能代替真实实验。

表示法和转换器的来源见 [sr-SMILES 官方仓库](https://github.com/heid-lab/sr-smiles)、
[原论文](https://doi.org/10.1039/D6DD00120C) 及本项目的
[第三方说明](../THIRD_PARTY_NOTICES.md)。
