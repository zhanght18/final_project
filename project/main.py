# unified_main.py
"""
一键运行入口：Stage 01-04
数据处理 → 可视化分析 → 模型训练 → QA 交互
"""

from pathlib import Path
from src.tlc_qa.data_processing import run_pipeline
from src.tlc_qa.analysis_visualization import run_all_analyses
from src.tlc_qa.demand_prediction import PredictionConfig, run_pipeline as run_model_pipeline
from src.tlc_qa.qa_interface import load_clean_dataset, interactive_loop

# 数据源和清洗数据路径
DEFAULT_SOURCE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet"
CLEAN_PATH = "data/processed/yellow_tripdata_clean.parquet"

# 输出目录统一管理
ANALYSIS_OUTPUT_DIR = "outputs/analysis"
MODEL_OUTPUT_DIR = "outputs/modeling"
QA_OUTPUT_DIR = "outputs/qa"

if __name__ == "__main__":
    print("=== Stage 01: 数据处理 ===")
    if not Path(CLEAN_PATH).exists():
        run_pipeline(
            source=DEFAULT_SOURCE,
            output_path=CLEAN_PATH,
            report_path="reports/data_quality_report.json",
            sample_rows=100000,  # 调试用，可设为 None
        )
    else:
        print(f"清洗数据已存在: {CLEAN_PATH}")

    print("=== Stage 02: 可视化分析 ===")
    run_all_analyses(clean_path=CLEAN_PATH, output_dir=ANALYSIS_OUTPUT_DIR)
    print(f"分析图表已生成: {ANALYSIS_OUTPUT_DIR}")

    print("=== Stage 03: 预测模型 ===")
    metrics_path = Path(MODEL_OUTPUT_DIR) / "demand_prediction_metrics.json"
    if not metrics_path.exists():
        cfg = PredictionConfig(
            input_path=CLEAN_PATH,
            output_dir=MODEL_OUTPUT_DIR,
            region_col="PULocationID",
            pickup_time_col="pickup_datetime",
            epochs=10,
            max_rows=None,
        )
        run_model_pipeline(cfg)
        print(f"模型训练完成，输出路径: {MODEL_OUTPUT_DIR}")
    else:
        print(f"模型输出已存在: {metrics_path}")

    print("=== Stage 04: QA 问答接口 ===")
    df = load_clean_dataset(CLEAN_PATH, max_rows=None)
    interactive_loop(df, QA_OUTPUT_DIR)