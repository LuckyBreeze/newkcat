# 本分支验证记录

## 已完成

- `ruff check .`：通过。
- `pytest -q`：**309 passed**，包含原仓库与新增模型、训练器、审计测试。
- 8种读出/融合组合完成CPU合成数据的CLI训练、权重锁定、重载和独立测试：
  r0、protein1024、mean_max、queries、queries_full、conditioned、fusion、flatten。
- 合成数据的配对审计CLI、cluster bootstrap与逐行输出完成。
- 默认R0与旧R的参数键、相同seed初始化、训练态及推断态前向逐值一致。
- 真正的全量数据只执行了元数据审计：训练19194、验证2698、测试2698，均未删行。
  这些数据和任何真实标签/缓存/权重都不随代码发布。

## 科学与数据完整性检查

- 保持全部样本；ID缺失、重复、额外行及标签不一致直接报错。
- 拟合不读取测试标签和特征用于训练或选择权重。
- 选模只用全验证集RMSE；完成后锁定权重。
- 正式evaluate绑定checkpoint记录的完整split ID顺序与原samples/features哈希。
  缩小测试集合、改标签或改输入不能继续作为同一benchmark评估。
- 变长输入右侧padding不改变有效预测；mean/max/query/flatten都屏蔽非化学位置。
- batch1及不满梯度累积窗口的样本加权与数值更新已核验。
- 大flatten在分配前有显式参数预算；inspect-model使用meta tensors。
- 新数据可用predict，不计算或伪造未提供的标签指标。
- 现有文件不会被新分支覆盖；新增目录可独立使用。

## 尚未完成且不能据此声称的结果

- 未训练这组新架构的真实kcat模型。
- 没有新架构优于CataPro/MCKcat的实验结论。
- 七个旧模型的真实逐样本预测不在此次附件中；尚未得到其sr成功/回退组RMSE。
- 没有实现或验证完整Retroformer图局部编码器、原子–残基配对或物理过渡态模型。

复现CPU检查：

```bash
python -m pip install -e '.[test]'
ruff check .
pytest -q
python examples/make_readout_demo.py
python -m newkcat.readout train --config runs/readout_demo/queries_full.yaml
python -m newkcat.readout evaluate \
  --config runs/readout_demo/models/queries_full/config.resolved.yaml \
  --checkpoint runs/readout_demo/models/queries_full/best.pt \
  --split test --output runs/readout_demo/evaluate_queries_full
```
