# 已执行的验证

验证日期：2026-10-31。环境：Linux、Python 3.12.14、CPU。

| 组件 | 实测版本 |
|---|---|
| PyTorch | 2.7.1+cpu |
| RDKit | 2026.9.1 |
| NumPy | 2.5.4 |
| PyYAML | 6.0.3 |
| pandas（旧缓存转换） | 3.1.0 |
| pytest | 9.2.0 |
| ruff | 0.17.3 |

## 自动测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
ruff check .
```

最终完整测试结果：**211 passed**。Ruff 检查通过。

这些测试覆盖：

- 六种模型模式的 `[B]` 输出和有限反向梯度。
- 原子置换、边端点/边排列、padding、eval batch 组成不变性。
- 多中心的软距离分布、断开图的独立通道、中心分数与绝对质量统计。
- 缺失/全零中心在所有偏置和 LayerNorm 后严格清零；无中心消融不使用中心输入。
- CataPro 原始回归权重迁移；新增列为零时与独立原版回归器输出一致。
- 显式氢、同位素、手性、电荷、原子映射及中心分数的严格身份校验。
- 实际立体翻转被识别；远端取代改变 CIP/E-Z 优先级不被误作局部几何改变。
- 重复/缺失样本 ID、打乱的缓存行、打乱的旧缓存列，均按身份正确对齐或明确拒绝。
- train-only EC 词表、样本/序列/group 泄漏和外部 manifest 重标训练样本。
- 同时故意破坏 test 标签与数值特征，fit 仍完成；显式 evaluate 再拒绝损坏测试数据。
- 防止 `[B,1]` / `[B]` 广播损失、训练最后一个单样本 batch 丢失、未定义指标伪造。
- checkpoint 安全重载、坏权重/哈希/schema 拒绝、严格最小验证 RMSE 保存。
- 重复 YAML 字段拒绝，默认不覆盖已有实验，未授权 pickle 格式不反序列化。

其中有些检查是许多输入组合的参数化测试；测试数量不代表独立生物学证据数量。

## 完整命令链

以下命令在本环境实际运行成功：

```bash
python examples/make_demo.py
newkcat validate --config examples/demo.yaml
newkcat train --config examples/demo.yaml
newkcat train --config examples/demo.yaml --mode baseline --output runs/demo_baseline
newkcat evaluate --checkpoint runs/demo/best.pt --config examples/demo.yaml --split test --output runs/demo_test
newkcat predict --checkpoint runs/demo/best.pt --config examples/demo.yaml --split test --output runs/demo_predict
newkcat audit-centers --checkpoint runs/demo/best.pt --config examples/demo.yaml --split val --output runs/demo_audit
newkcat centers-from-reactions --samples examples/reference_reactions.csv --output data/demo/reference_centers.jsonl
```

合成数据共 48 条：32 train / 8 val / 8 test。`conditioned` 和 `baseline` 均完成
4 个 epoch，产生 checkpoint、配置快照、输入 hash、历史记录与预测表。
`conditioned` 的最好验证轮为第 3 轮，验证 RMSE 为 0.45824；独立测试 RMSE 为 0.43275。
**这些值仅用于核对示例运行，标签和语言模型向量是人为合成的，不能用于评价生物学性能。**

示例中的中心覆盖：训练 25/32、验证 7/8、测试 6/8；其余显式走缺失回退。
固定 checkpoint 的中心清零和打乱审计产生三组预测，不更新权重。
额外的参考反应文件是化学映射解析用例，不是酶促反应测量数据。

默认真实实验配置下，baseline 注册 506,113 个参数，五种图版本各注册 1,087,169 个参数。
这不表示它们具有相同有效容量：不同消融使用的模块子集不同。
示例使用更小的图分支，其 `conditioned` 注册 570,289 个参数。

## 已修复的具体问题

对抗复核不是只加入“鲁棒性”措辞，实际找到了并修复了以下问题：

1. 中心文件的显式氢表示可能给无中心消融增加节点；现在要求实际图身份相同。
2. 远端取代可改变 CIP/E-Z 名称而不改变局部立体状态；参考中心改用映射相对状态。
3. 原始缓存即使仍有 1959 列，也可能被重新排列；现在按官方列标签 0..1958 恢复顺序。
4. `min_delta` 不应阻止保存较好的模型；现在只控制耐心值，checkpoint 保存严格最优验证 RMSE。
5. YAML 重复键不应静默覆盖配置；现在明确报错。

## 尚未完成的实证

- 没有用户的真实 kcat 文件、既有全局特征缓存或预测中心文件，因此未运行正式交叉验证。
- 未下载或验证 GNN-SOM 预训练预测器，未计算真实 CataPro 与 EnzymeMap 的交集覆盖率。
- 没有 GPU 实测、长时间训练、显存 benchmark 或 RTX 5080 兼容性实测。
- 没有完整 CGR 编辑图、产物生成、3D 几何或残基级交叉注意力。
- 没有统计显著性结论、泛化提升结论或“首次提出”结论。

GitHub Actions 工作流配置为运行 CPU 测试和相同示例命令链。
本记录描述的是本地实际验证；远程 CI 状态以仓库 Actions 页面为准。
