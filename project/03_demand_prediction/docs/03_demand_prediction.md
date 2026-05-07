# 03 预测模型：区域-时段出行需求预测

## 目标定义

本模块预测某个上车区域在某个小时内的出行需求量。为了保证可复现，需求量被定义为：

> `trip_count = 按 PULocationID + pickup_hour_ts 聚合后的订单数量`

该标签来自项目清洗后的 NYC TLC 行程记录，不是 TLC 官方发布的需求预测标签。

## 输入数据

默认读取第一步生成的清洗后 Parquet 文件，例如：

```bash
data/processed/yellow_tripdata_2026-01_clean.parquet
```

要求至少包含：

- `tpep_pickup_datetime`
- `PULocationID`

如果使用其他车型或数据版本，需要通过参数修改时间列和区域列。

## 特征工程

模型使用以下类型特征：

1. 区域特征：`PULocationID`
2. 时间特征：`hour`、`day_of_week`、`is_weekend`、`is_peak_hour`、`month`、`day_of_month`
3. 历史需求特征：
   - `lag_1h_demand`：同一区域上一小时需求量
   - `lag_24h_demand`：同一区域 24 小时前需求量
   - `rolling_3h_mean_demand`：同一区域过去 3 小时平均需求量

说明：历史需求特征只使用当前预测时点之前的观测值，避免直接使用当前目标值。

## 训练/测试划分

采用 8:2 划分，但不是随机划分，而是按时间排序后进行时间序列式切分：

- 前 80% 时间片：训练集
- 后 20% 时间片：测试集

理由：需求预测本质上是面向未来时段的预测任务，随机划分可能让未来信息间接泄漏到训练集中。

## 模型

### 神经网络

使用 PyTorch 构建多层感知机 MLP：

- 输入层：one-hot 后的区域和时间特征 + 数值历史需求特征
- 隐藏层：128、64
- 激活函数：ReLU
- Dropout：0.10
- 损失函数：MSE
- 优化器：Adam

### 随机森林

使用 `RandomForestRegressor` 作为结构化表格数据基线模型。

## 输出文件

运行后会自动保存到 `outputs/modeling/`：

- `region_hour_demand_dataset.parquet`：聚合后的区域-小时监督学习数据集
- `05_nn_loss_curve.png`：神经网络训练/测试 loss 曲线
- `06_model_comparison_mae_rmse.png`：神经网络与随机森林的 MAE、RMSE 对比图
- `demand_prediction_metrics.json`：训练配置、指标、输出路径
- `test_predictions_sample.csv`：测试集预测样例
- `random_forest_feature_importance.csv`：随机森林特征重要性

## 运行方式

```bash
python -m src.tlc_qa.demand_prediction \
  --input data/processed/yellow_tripdata_2026-01_clean.parquet \
  --output-dir outputs/modeling \
  --region-col PULocationID \
  --epochs 30
```

如果只是快速验证流程，可以限制行数：

```bash
python -m src.tlc_qa.demand_prediction \
  --input data/processed/yellow_tripdata_2026-01_clean.parquet \
  --output-dir outputs/modeling \
  --max-rows 100000 \
  --epochs 10
```

## 评价指标

- MAE：平均绝对误差，单位是订单数
- RMSE：均方根误差，单位是订单数，对大误差更敏感

## 两种方法的优劣分析

| 方法 | 优点 | 局限 |
|---|---|---|
| 神经网络 MLP | 可以学习非线性关系；在数据量大、特征充分时有扩展空间；可继续接入更复杂时序模型 | 对特征缩放、超参数、训练轮数更敏感；小样本或低维表格任务上不一定优于树模型；解释性较弱 |
| 随机森林 | 强表格基线；对异常值和非线性关系较稳健；通常调参成本较低；可输出特征重要性 | 对时间外推能力有限；模型体积可能较大；无法像深度模型一样自然扩展到复杂序列结构 |

## 事实边界

- `PULocationID` 是 TLC Taxi Zone ID，不是真实经纬度。
- 本模块给出的 MAE/RMSE 只在用户实际运行脚本后才有确定数值。
- 如果训练样本、月份、车型或清洗策略变化，指标会变化。
- 模型结果是基于历史记录的统计预测，不代表官方需求预测或现实中的全部交通需求。
