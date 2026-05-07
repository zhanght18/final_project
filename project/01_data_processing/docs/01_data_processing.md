# Step 1 数据处理模块说明

## 数据来源

数据来自 NYC Taxi & Limousine Commission 的 TLC Trip Record Data 页面。Yellow / Green Taxi 数据包含上下车时间、上下车位置 Taxi Zone、行程距离、费用、付款方式、乘客数等字段。TLC 同时说明：这些数据由授权技术服务商提交，TLC 不保证其准确性。

## 字段事实边界

当前 Yellow Taxi Parquet 文件中，上下车位置是 `PULocationID` 和 `DOLocationID`，表示 TLC Taxi Zone ID，不是实际经纬度。若需要经纬度，只能结合 Taxi Zone 空间边界计算区域中心点；这属于近似，不代表真实上下车点。

## 数据质量报告

`generate_quality_report()` 输出：

- 总行数、总列数
- 核心字段缺失数与缺失率
- 异常值统计：
  - 结束时间不晚于开始时间
  - 行程距离小于等于 0
  - 乘客数小于等于 0
  - 费用为负
  - 未知付款类型
- 数值字段描述性统计
- IQR 异常值边界

## 清洗策略

清洗函数 `clean_data()` 使用保守规则：

1. 只保留问答系统需要的核心字段，避免下游回答使用未审查字段。
2. 解析上下车时间，无法解析或行程时长小于等于 0 的记录删除。
3. 删除距离、乘客数、费用明显不适合需求分析的记录。
4. 设置宽松上限：行程时长 24 小时、乘客数 8、距离 200 英里、总金额 1000 美元、速度 100 mph。
5. 保留 Taxi Zone ID，不伪造真实经纬度。

## 特征工程

基础时间特征：

- `pickup_hour`
- `pickup_weekday`
- `pickup_weekday_name`
- `is_weekend`
- `is_peak_hour`

衍生特征：

- `duration_min`：行程时长，支持拥堵与效率分析。
- `speed_mph`：平均速度，支持识别低速时段和区域组合。
- `fare_per_mile`：每英里费用，支持价格强度分析。
- `is_airport_trip_candidate`：基于 JFK、LGA、EWR Taxi Zone ID 的机场行程候选标记。
- `route_zone_pair`：上客区-下客区组合，支持 OD 需求分析。

## 运行方式

```bash
python -m src.tlc_qa.data_processing \
  --source https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet \
  --output data/processed/yellow_tripdata_2026-01_clean.parquet \
  --report reports/data_quality_report_yellow_2026_01.json
```

快速测试可加：

```bash
--sample-rows 100000
```
