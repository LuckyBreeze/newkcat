# newkcat

**本分支新增：用反应序列完整替换化学编码器。** `newkcat sequence` 提供
CataPro 缓存基线、底物 SMILES、完整 reaction-SMILES、sr-SMILES 及动态字段局部读取五个版本，
支持成对转换审计、独立训练/评估和可选 MLM 预训练。序列版本只需 ProtT5 缓存，
不要求 MolT5、MACCS 或底物图。详见 **[反应序列模型使用说明](docs/SEQUENCE_MODELS.md)**。
当前小型字符 Transformer 默认随机初始化，没有附带论文预训练权重，也不承诺性能提升。
序列路线的配置位于 `configs/sequence_*.yaml`；以下第 1–6 节保留原有图分支路线的说明。

**在 CataPro 上检验“相对于反应中心的化学环境是否有用”。**

这是一个可训练、可消融的研究实现。保留 CataPro 的 ProtT5 1024 维、
MolT5 768 维和 MACCS 167 维输入，在此基础上增加一个小型底物图分支：
中心概率 → 图距离分布 → GNN → 分层读取 → 酶条件化 → kcat 回归。
纯 PyTorch + RDKit，不依赖 PyG、对接软件或三维结构。

**当前实现的反应侧是“底物图 + 原子级中心分数”，不是完整 CGR 编辑图。**
已映射反应可以提供参考中心；预测中心需要外部预测器提供。
本项目不会把原子位点概率说成成键/断键标签，也不会自动猜测 EC 对应的产物。
第一阶段要检验的是中心相对位置这一假设，完整的反应编辑图可在结果支持后扩展。

本仓库包含训练、验证、独立测试、预测、中心扰动审计、严格的数据接入和自动测试。
示例是**合成数据上的流程检查**，不是实际 kcat 性能结果；没有预先声称优于 CataPro。

## 1. 安装

Python 3.10+。建议新建环境，在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
```

CUDA 环境先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)
安装匹配驱动的 PyTorch，再安装本项目。CUDA 12.8 的候选安装命令为：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'
```

本次开发验证在 CPU 上进行；没有在 RTX 5080 上实测显存或速度。
全局语言模型特征从缓存读取，训练过程不会下载或微调 ProtT5/MolT5。

## 2. 先运行一个完整检查

```bash
python examples/make_demo.py
newkcat validate --config examples/demo.yaml
newkcat train --config examples/demo.yaml
newkcat evaluate --checkpoint runs/demo/best.pt --config examples/demo.yaml --split test --output runs/demo_test
newkcat predict --checkpoint runs/demo/best.pt --config examples/demo.yaml --split test --output runs/demo_predict
newkcat audit-centers --checkpoint runs/demo/best.pt --config examples/demo.yaml --split val --output runs/demo_audit
pytest -q
```

示例包含 32 条训练、8 条验证、8 条测试记录，特征表故意打乱行顺序，部分样本缺少中心。
随机生成的蛋白特征、分子特征和标签仅用于检查流水线。再次运行到同一输出目录需要显式
`--overwrite`，以免覆盖实验。

## 3. 接入你现有的 CataPro 数据

优先沿用原来的划分和原来的特征缓存，避免把表示改动和预处理改动混在一起。
仓库提供官方格式的转换器：

```bash
python -m pip install -e '.[catapro]'
python -m newkcat.import_catapro \
  --data /path/to/kcat-data_0.4simi-10fold.csv \
  --features /path/to/kcat-datasets_features.pkl \
  --test-fold 0 --val-fold 1 \
  --output data/catapro/fold0 --trust-pickle
```

该命令保留样本 ID 和原始 fold，将 `kcat(s^-1)` 转为 `log10_kcat`，按 ID 对齐特征，
输出 `samples.csv` 和安全的 `features.npz`。仅为你信任的本地 pickle 指定
`--trust-pickle`；常规训练不加载 pickle。
指定其他测试/验证 fold 即可构建下一折，不随机重切已有划分。

准备三类输入：

| 文件 | 内容 | 作用 |
|---|---|---|
| `samples.csv` | sample_id、sequence、smiles、log10_kcat、split | 样本与既有划分 |
| `features.npz` | sample_ids、protein、molt5、maccs | CataPro 原始特征，按 ID 连接 |
| `centers.jsonl` | 带映射的底物 SMILES、映射号→分数、来源 | 可选的原子中心先验 |

详见 [数据格式](docs/DATA_FORMAT.md)。没有中心文件也能运行基线与无中心图消融；
中心分支对缺失中心严格输出零。中心全缺失的实验不能检验中心建模假设。

### 中心从哪里来

**预测中心**：接入外部 GNN-SOM 或你自己的模型，导出带原子映射号的分数。
本仓库不附送未验证的预训练中心预测器；不能把某个 SMILES 上的数组直接套到另一个
SMILES 遍历顺序上。所有原子都需要正的、唯一的映射号和对应分数。

**参考中心**：如果你已有完整、原子映射正确的反应，可运行：

```bash
newkcat centers-from-reactions --samples data/reactions.csv --output data/reference_centers.jsonl
```

输入需要 `sample_id,smiles,reaction_smiles`。这一步只提取净结构变化，不执行原子映射，
也不把参考中心称为绝对正确的催化机制。映射不完整、主底物匹配不唯一等情况直接报错。
如果推理时只有底物，使用产物提取的中心属于参考信息实验，需要设置
`training.allow_reference_centers: true` 并单独报告，不能混充可部署成绩。

## 4. 六个比较版本

| `model.mode` | 图分支 | 要回答的问题 |
|---|---|---|
| `baseline` | 无 | 原 CataPro 回归结构能达到什么水平？ |
| `global` | 普通 GNN + 全图读取 | 增益是否只来自多一个图编码器？ |
| `center` | 普通 GNN + 中心概率加权读取 | 知道哪些原子重要是否已经足够？ |
| `shell` | 中心距离位置编码 + 分层读取 | 中心周围的环境层次是否有额外作用？ |
| `conditioned` | `shell` + 酶条件化分层权重 | 不同酶是否需要关注不同范围的环境？ |
| `learned` | 无中心的五个可学习读取槽 + 酶条件化 | 增益是否只是读取槽和容量增加？ |

`baseline` 沿用官方回归层：ProtT5/MolT5 分别 BatchNorm，MACCS 直接连接，
1959 → 256 → 1。默认 dropout 为 0，真实实验可以在验证集上选择，但所有比较应采用
同等搜索预算。新增图分支用 LayerNorm，最后再与原始全局特征连接。

代码支持加载官方 CataPro `state_dict`：

```bash
newkcat train --config configs/catapro_fold0.yaml --init-catapro /path/to/catapro.pt
```

迁移时保留原回归权重，并将新增输入列置零，使扩展模型在 `eval()` 下初始化为原模型。
该操作要求特征缓存和原模型相容；不代表“新分支已经训练好”。只加载张量型权重，
不自动执行旧 checkpoint 内的 Python 对象。

## 5. 正式实验的顺序

1. 用你已有数据和缓存运行 `baseline`，确认划分、标签单位、特征及指标与之前一致。
2. 检查中心覆盖率、来源、原子对齐和是否使用 EC；对齐失败不要随意丢弃后与全量基线比较。
3. 在**同一组样本、同一组 fold、同一批随机种子**上依次比较六个版本。
4. 只在验证集选择参数/epoch；保存最好模型后，单独执行测试集评估。
5. 用固定 checkpoint 做中心打乱/清零审计，检查是否真的依赖中心；再开展训练时中心打乱消融。
6. 按 fold 报告配对差值和不确定范围，检查覆盖率分层及新反应规则泛化，再决定是否增加完整 CGR。

配置起点见 [configs/catapro_fold0.yaml](configs/catapro_fold0.yaml)。
可用 `--mode baseline` 等命令行参数覆盖模型版本；给不同实验使用不同 `output_dir`。
如果中心预测器用过 EC，配置 `model.use_ec2: true`，同时给基线相同 EC 输入。
EC 词表只由训练集构建，验证/测试未见类别映射为 unknown。

还没准备中心文件时，先用 `centers: null` 的基线配置：

```bash
newkcat validate --config configs/catapro_baseline_fold0.yaml
newkcat train --config configs/catapro_baseline_fold0.yaml
```

中心文件准备好后，同一配置逐项比较，例如：

```bash
newkcat train --config configs/catapro_fold0.yaml --mode global --output runs/fold0_global_seed17
newkcat train --config configs/catapro_fold0.yaml --mode center --output runs/fold0_center_seed17
newkcat train --config configs/catapro_fold0.yaml --mode shell --output runs/fold0_shell_seed17
newkcat train --config configs/catapro_fold0.yaml --mode conditioned --output runs/fold0_conditioned_seed17
newkcat train --config configs/catapro_fold0.yaml --mode learned --output runs/fold0_learned_seed17
```

当前训练器对所有版本统一使用 AdamW、MSE、梯度裁剪和验证集早停。
“兼容 CataPro 回归结构”不等于已复现论文全部训练设置或性能；
原始基线的数值复现仍需要核对你的缓存、划分、优化设置和随机种子。

## 6. 第一性原理与对抗性检查如何落地

- **输入身份要清楚**：样本按 ID 连接，原子按显式映射号对齐，不把 atom index 当化学特征。
- **无关编号不能改变结论**：原子重编号、边顺序变化和 padding 不应改变预测。
- **缺失先验不能制造假信号**：中心缺失/全零时，在所有带偏置的投影之后清零中心分支。
- **新模块必须有可证伪的对照**：全图、仅中心、距离分层、酶条件化、无中心可学习槽逐级比较。
- **信息必须公平**：区分预测中心和参考中心，记录 EC 来源，不通过测试标签选模型。
- **数值形状必须正确**：标签和输出都为 `[B]`，防止 `[B,1]` 与 `[B]` 广播成错误的成对损失。

这里的“对抗性”主要指主动构造反例检验方法，不额外默认加入 GAN 或对抗训练损失。
真实的生物学增益、结构可靠性和外部预测器质量仍需实验检验。

## 文档与代码入口

- [架构与公式](docs/ARCHITECTURE.md)
- [数据格式与泄漏边界](docs/DATA_FORMAT.md)
- [已执行的验证及局限](docs/VALIDATION.md)
- `src/newkcat/model.py`、`position.py`：模型与中心距离分布
- `chem.py`、`centers.py`、`data.py`：图、中心和输入校验
- `train.py`、`cli.py`：训练、评估与审计
- `tests/`：数值、数据、化学对应和命令行测试

方法借鉴与区别见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。
本项目没有宣称首次把 CGR 或反应中心用于酶动力学；完整 CGR 与三维几何也未在本版实现。
原 CataPro 的许可与署名保留在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
