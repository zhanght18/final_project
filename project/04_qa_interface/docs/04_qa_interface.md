# 04 问答接口模块说明

## 目标

第四步提供一个可交互的命令行问答循环，让用户输入自然语言问题，系统通过关键词和参数解析匹配到确定性函数，再基于 M1-M3 的数据与函数返回：

- 数字结论
- 图表或衍生数据路径
- 事实边界说明

本模块不让大模型凭空生成结论。所有结论来自输入的已清洗数据、聚合结果或已保存的分析图表。

## 运行方式

交互模式：

```bash
python -m src.tlc_qa.qa_interface \
  --input data/processed/yellow_tripdata_2026-01_clean.parquet \
  --output-dir outputs/qa
```

单问题模式：

```bash
python -m src.tlc_qa.qa_interface \
  --input data/processed/yellow_tripdata_2026-01_clean.parquet \
  --output-dir outputs/qa \
  --question "区域 132 在 8 点的需求预测"
```

JSON 输出：

```bash
python -m src.tlc_qa.qa_interface \
  --input data/processed/yellow_tripdata_2026-01_clean.parquet \
  --question "上客 TOP 10 区域" \
  --json
```

## 支持的问题类型

### 1. 时段查询

示例：

- 哪个小时订单最多？
- 工作日和周末的需求有什么差异？
- 高峰时段订单量如何？

返回内容：最高订单小时、工作日/周末订单量、图表路径。

调用逻辑：复用 `analysis_visualization.analyze_time_demand()`。

### 2. 区域排名

示例：

- 上客 TOP 10 区域
- 下客量最高的前 5 个区域
- 哪些 Taxi Zone 最热门？

返回内容：TOP Taxi Zone ID、订单量、柱状图路径。

事实边界：`PULocationID` 和 `DOLocationID` 是 TLC Taxi Zone ID，不是真实经纬度。

### 3. 需求预测 / 历史基线

示例：

- 区域 132 在 8 点的需求预测
- zone 237 hour 18 demand
- 某区域某时段预计有多少订单？

返回内容：按 `PULocationID + pickup hour` 聚合后的历史平均需求、中位数、最大值，以及区域-小时需求数据路径。

调用逻辑：复用 `demand_prediction.build_region_hour_dataset()` 的 Step 3 目标定义。

事实边界：如果未加载已训练并部署的模型，交互接口给出的是历史基线估计，不是官方预测，也不是实时神经网络服务输出。

### 4. 可能费用查询

示例：

- 距离 3 英里 18 点 1 人可能车费是多少？
- 5 miles 2 passengers fare
- 晚高峰 10 英里的可能费用

返回内容：相似历史行程的 `fare_amount` 中位数、均值、`total_amount` 中位数，以及费用分析图表路径。

匹配策略：

- 距离：按输入距离的 ±20% 过滤
- 小时：若问题中出现具体小时，则按 `pickup_hour` 过滤
- 乘客数：若问题中出现人数，则按 `passenger_count` 过滤

### 5. OD 热门线路

示例：

- OD 热门线路 TOP 15
- 哪条路线订单最多？
- 起终点组合最高频是什么？

返回内容：最高频 `PULocationID-DOLocationID` 组合、订单量、OD 排名图路径。

调用逻辑：复用 `analysis_visualization.analyze_high_value_od_routes()`。

### 6. 数据质量 / 能力边界

示例：

- 当前数据有多少行？
- 这个系统能问什么？
- 字段缺失情况如何？

返回内容：行列数、缺失率 TOP 10 图表、系统能力边界。

## 自然语言匹配策略

当前版本采用轻量级规则：

1. 关键词分类：如“车费/费用/fare”进入费用查询，“预测/需求/demand”进入需求查询。
2. 正则抽取参数：小时、区域 ID、TOP N、距离、乘客数。
3. 调用对应函数并返回结构化结果。

这种方式透明、可解释，适合课程项目和原型系统。后续可以升级为：

- 意图分类模型
- LLM 生成结构化 JSON 参数
- DuckDB SQL 查询层
- Streamlit 或 Gradio 可视化界面

## 事实边界

- 所有回答只基于输入数据文件。
- Taxi Zone ID 不等于真实经纬度。
- 历史基线需求不等于官方预测。
- 如果过滤后没有匹配样本，系统会直接说明无法确认，而不是编造数值。
