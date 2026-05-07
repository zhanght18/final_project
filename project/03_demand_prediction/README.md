# 03_demand_prediction: NYC TLC 出行数据问答系统阶段代码


## PyCharm 运行

```bash
pip install -r requirements.txt
python main.py
```

## 事实边界

- 默认数据源是 NYC TLC Yellow Taxi Parquet 月度文件。
- `PULocationID` / `DOLocationID` 是 TLC Taxi Zone ID，不是真实经纬度。
- `main.py` 默认读取前 100000 行作为演示；如需完整月度数据，把 `sample_rows=100000` 改为 `sample_rows=None`。
- 生成文件会写入 `data/processed/`、`reports/`、`outputs/`。
