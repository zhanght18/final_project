# Step 2 分析可视化模块说明

## 目标

本模块基于 Step 1 输出的清洗后 Parquet 文件，完成 4 类分析，并自动把图表保存到 `outputs/` 目录。

事实边界：

- 图表基于清洗后的样本或全量数据，不直接代表未经清洗的官方原始数据。
- `PULocationID` 和 `DOLocationID` 是 TLC Taxi Zone ID，不是真实经纬度。
- 若需要区域名称或地图展示，应额外连接 TLC Taxi Zone lookup / shapefile；本步骤默认不伪造区域名称。

## 分析 1：出行需求时间规律

输出图表：

- `outputs/01_time_demand_avg_hourly_weekday_weekend.png`

分析逻辑：

- 先按日期、小时、工作日/周末统计订单量。
- 再计算每个小时的日均订单量。
- 这样可以避免一个月内工作日数量多于周末而导致直接总量比较失真。

可回答问题示例：

- 哪些小时需求最高？
- 工作日和周末的出行节奏是否不同？

## 分析 2：区域热度分析

输出图表：

- `outputs/02a_top_pickup_zones.png`
- `outputs/02b_top_dropoff_zones.png`
- `outputs/02c_top_pickup_zones_hour_heatmap.png`

分析逻辑：

- 分别统计上客量最高的 TOP 10 Taxi Zone ID。
- 分别统计下客量最高的 TOP 10 Taxi Zone ID。
- 对 TOP 10 上客区域绘制小时热力图，观察高峰时段分布。

注意：这里展示的是 Taxi Zone ID，不是区域名称。若要名称，需要加入官方 Taxi Zone lookup。

## 分析 3：车费影响因素分析

输出图表：

- `outputs/03a_distance_vs_fare_scatter.png`
- `outputs/03b_avg_fare_by_hour.png`
- `outputs/03c_avg_fare_by_passenger_count.png`

分析逻辑：

- 距离与车费散点图：观察车费是否随距离增长。
- 小时与平均车费折线图：观察不同时段的价格水平。
- 乘客数与平均车费柱状图：观察乘客人数是否与车费存在结构性差异。

补充输出：

- `analysis_summary.json` 中会保存 `fare_amount` 与距离、时长、速度、乘客数的相关系数。

## 分析 4：自选洞察：OD 热门线路分析

输出图表：

- `outputs/04_top_od_zone_pairs.png`

为什么选择这个分析：

单独看上客区或下客区只能说明局部热度；OD 区域对可以揭示稳定的出行走廊，更适合交通需求挖掘、车辆调度、热点运营和问答系统中的“从哪里到哪里最多”类问题。

输出内容：

- TOP OD Taxi Zone Pair
- 行程数
- 中位车费
- 中位距离
- 中位时长
- 中位速度

## 运行方式

先运行 Step 1：

```bash
python -m src.tlc_qa.data_processing \
  --source https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet \
  --output data/processed/yellow_tripdata_2026-01_clean.parquet \
  --report reports/data_quality_report_yellow_2026_01.json \
  --sample-rows 100000
```

再运行 Step 2：

```bash
python -m src.tlc_qa.analysis_visualization \
  --input data/processed/yellow_tripdata_2026-01_clean.parquet \
  --output-dir outputs
```

运行完成后，所有图片和 `analysis_summary.json` 都会保存到 `outputs/`。

## 给 M4 可视化界面的接口建议

后续 Streamlit / Plotly 页面可以直接读取：

- `outputs/*.png` 用于快速报告展示。
- `outputs/analysis_summary.json` 用于问答系统引用结构化分析结果。
- `data/processed/*_clean.parquet` 用于交互筛选和实时聚合。
