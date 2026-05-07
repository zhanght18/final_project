"""Stage 04 one-click entry: process data, draw charts, train models, then start QA loop."""
from pathlib import Path
from src.tlc_qa.data_processing import run_pipeline
from src.tlc_qa.analysis_visualization import run_all_analyses
from src.tlc_qa.demand_prediction import PredictionConfig, run_pipeline as run_model_pipeline
from src.tlc_qa.qa_interface import load_clean_dataset, interactive_loop

DEFAULT_SOURCE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet"
CLEAN_PATH = "data/processed/yellow_tripdata_clean.parquet"

if __name__ == "__main__":
    if not Path(CLEAN_PATH).exists():
        run_pipeline(
            source=DEFAULT_SOURCE,
            output_path=CLEAN_PATH,
            report_path="reports/data_quality_report.json",
            sample_rows=100000,
        )
    run_all_analyses(clean_path=CLEAN_PATH, output_dir="outputs")
    metrics_path = Path("outputs/modeling/demand_prediction_metrics.json")
    if not metrics_path.exists():
        cfg = PredictionConfig(
            input_path=CLEAN_PATH,
            output_dir="outputs/modeling",
            region_col="PULocationID",
            pickup_time_col="pickup_datetime",
            epochs=10,
            max_rows=None,
        )
        run_model_pipeline(cfg)
    df = load_clean_dataset(CLEAN_PATH, max_rows=None)
    interactive_loop(df, "outputs/qa")
