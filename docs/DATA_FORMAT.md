# 数据接入与信息边界

所有 YAML 路径相对于该 YAML 所在目录解析。命令行的 `--output` 路径相对于当前目录。
常规训练不需要互联网，不自动下载数据库或语言模型。

## samples.csv

必需列：

| 列名 | 要求 |
|---|---|
| `sample_id` | 全文件唯一、非空；与缓存及中心记录一致 |
| `sequence` | 酶序列；空白移除后转大写，用于身份与防泄漏校验 |
| `smiles` | 此记录对应的底物；保留电荷、立体及同位素语义 |
| `log10_kcat` | `log10(kcat / s^-1)`，训练/评估须为有限数；无标签预测可留空 |
| `split` | `train` / `val` / `test` |

可选列：`ec`、`group_id`。EC 前两级形成类别，例如 `1.1.1.1` → `1.1`。
未注释、多重/歧义 EC 不任意取第一项，使用 unknown。
训练集之外的新 EC 也使用 unknown，不提前扩大训练词表。

同一序列、同一酶–底物组合、同一非空 `group_id` 不能跨 split。
这些检查不能替代 MMseqs2/CD-HIT 等相似性聚类；要复用已有的相似性划分。
如果你要报告 40% 序列相似性泛化，必须在上游核实该划分，而不是把通过这里的
精确字符串检查写成通过 40% 聚类检查。

## features.npz

使用 NumPy `savez` / `savez_compressed`，四个数组：

```python
np.savez_compressed(
    "features.npz",
    sample_ids=np.asarray(ids, dtype=str),  # Unicode，不能用 object 数组
    protein=prott5_vectors,                # [N, 1024]
    molt5=molt5_vectors,                   # [N, 768]
    maccs=maccs_bits,                      # [N, 167]
)
```

矩阵需要有限数值；MACCS 只能为 0/1，第 0 位为 RDKit 保留位 0。
不允许把 166 位指纹直接补成“看起来能跑”的格式。
缓存可包含多余记录，但所选样本必须各有唯一对应 ID。加载时从不依赖行顺序。
`protein/molt5` 要与之前的 CataPro 特征处理方式一致，尤其是长序列截断、
特殊 token 去除和池化方式；原缓存最适合做第一轮可控比较。

官方 DataFrame 缓存可用 `python -m newkcat.import_catapro` 导入。
转换器按 ID 校验、保留 `source_fold`，不会重新生成语言模型嵌入或更改原划分。
原 kcat 必须大于 0 且有限；转换器不会把 0、缺失值或字符串单元格悄悄改为有效标签。

## centers.jsonl

每行一个样本，示意：

```json
{"sample_id":"sample_001","mapped_smiles":"[CH3:11][CH2:27][OH:92]","scores":{"11":0.1,"27":0.7,"92":0.8},"kind":"predicted","source":"my-center-model-v1-out-of-fold","uses_ec":false}
```

- `scores` 的键必须与 `mapped_smiles` 的全部正整数映射号完全一致。
- 映射号无需连续；不能重复，也不能将原子数组索引假充映射号。
- 分数必须是 `[0,1]` 内有限数值；布尔、NaN、遗漏原子、额外原子均报错。
- `kind` 只能为 `predicted` 或 `reference`；后者必须是 0/1 标签。
- `source` 必须非空，写明来源、模型版本或反应数据版本。
- `uses_ec` 必须显式为 JSON 布尔值；外部预测器使用 EC 时写 `true`。
- 缺少该样本的整条记录代表“中心不可用”；全零记录代表“中心已观测但分数质量为零”。
  两者均使中心依赖分支回退，但覆盖率报告区分这两种情况。

原子对齐使用这条中心记录自己的映射 SMILES 构图，随后按映射号读取分数。
数据层会检查它与底物在结构、电荷、同位素、立体化学及支持的原子表示上相符。
任意 SMILES 上的裸 `scores=[...]` 不被接受，因为无法证明它与图原子顺序一致。

## 从映射反应提取参考中心

输入 CSV 必需 `sample_id,smiles,reaction_smiles`，可选 `source,substrate_maps`。
反应格式支持 `reactants>>products` 或 `reactants>agents>products`。
中心提取规则比较键邻域/键级/键立体、电荷、氢计数、自由基及立体状态变化。
立体变化使用相对于映射邻居的四面体奇偶性和双键 cis/trans 状态，避免仅因远处
取代基改变 CIP 优先级，就把未发生局部几何变化的原子误标为中心。
这是净结构变化的参考标注，不是反应机制或真实过渡态结构。

```bash
newkcat centers-from-reactions \
  --samples data/reactions.csv \
  --output data/reference_centers.jsonl
```

严格要求：

- 反应物和产物各自所有原子均映射，编号唯一，两侧映射号集合相同。
- 同一映射号不能改变元素或同位素。
- 主底物必须能与反应物中的一个完整组分精确匹配；反向反应不会自动猜测方向。
- 有多个相同组分时提供 `substrate_maps`，格式为 CSV 单元格内的 JSON 整数列表。
- 所选底物必须有检测到的净变化；无变化、配平不完整、映射歧义时直接报错。

上述限制较保守，会拒绝一些常见的不完整数据库反应。统计覆盖率时应将拒绝原因报告出来，
不能把“解析成功”称为“映射质量已经验证”。本版没有自动 EC→Rhea 匹配、反应配平或
RXNMapper 调用；这些操作要在上游单独完成和审计。

## 三种不同的信息条件

| 信息条件 | 设置 | 能报告什么 |
|---|---|---|
| 仅序列和底物 | `baseline/global/learned`，`use_ec2: false` | 无额外中心先验的比较 |
| 预测中心 | 中心模式、`kind: predicted` | 推理时可获得该预测器及所需输入时的结果 |
| 参考中心 | `kind: reference`，`allow_reference_centers: true` | 使用参考反应信息的诊断；需明确推理是否有产物 |

如果中心预测器输入了 EC，中心模型必须 `use_ec2: true`，比较基线也应开启 EC。
如果预测器在下游测试反应标签上训练过，应按你的泛化定义去除重合/做交叉拟合。
程序能记录来源与检查输入形式，但不能从一组概率数字证明外部预测器没有泄漏。

`reference` 分支并不是“真实无噪声 oracle”，也不保证形成严格性能上界；
它仍受映射、标注规则和优化影响。

## 训练、验证、测试边界

`train` 仅实例化 train 和 val 数据集，使用验证 RMSE 选择 checkpoint。
完整文件会计算 SHA256，完整 manifest 会校验 ID/序列/group 划分，
但 test 目标与数值特征不会参与训练、早停或模型选择。
`validate` 是显式的数据检查命令，会检查选定配置中的全部 split；
不要把它和训练阶段的统计量拟合混淆。

独立 `evaluate --split test` 除当前划分检查外，还会对 checkpoint 内记录的训练/验证
样本 ID、序列 hash 和 group ID 检查交集，避免换个 manifest 就把训练样本标成测试样本。
所有指标基于 log10 空间。Pearson、Spearman 或 R² 无定义时输出 JSON `null`，不伪造为 0。

推理时架构和 EC 词表以 checkpoint 为准；更改评估 YAML 的 `model.mode` 不会改变已训练架构。
输出同时记录训练与当前输入 hash，便于发现数据或标注发生了变化。

## 运行限制

每个底物图最多 512 个显式原子；超出时明确报错，避免距离矩阵的二次内存开销。
这里没有默认去盐、最大组分、互变异构体或质子化标准化。请对所有比较采用同一上游政策，
保留必要的辅因子/共底物信息时也要明确这是否仍是单底物任务。
本版为该图上所有原子计算精确图跳数，不把不连通组分伪装成一个很大的有限距离。
不支持的 CXSMILES 增强立体组、非四面体/非常规键立体化学直接拒绝，避免静默丢失信息。
中心标注也不能相对于底物增删显式氢节点；对齐不只是分子式相同或氢归一化后相同。
