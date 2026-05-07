"""Stage 02 one-click entry: run data processing, then generate analysis charts."""
from pathlib import Path
from src.tlc_qa.data_processing import run_pipeline
from src.tlc_qa.analysis_visualization import run_all_analyses

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
