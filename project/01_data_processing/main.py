"""Stage 01 one-click entry: data loading, quality report, cleaning, feature engineering."""
from src.tlc_qa.data_processing import run_pipeline

DEFAULT_SOURCE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet"

if __name__ == "__main__":
    # sample_rows limits first run time in PyCharm. Set to None for the full monthly file.
    run_pipeline(
        source=DEFAULT_SOURCE,
        output_path="data/processed/yellow_tripdata_clean.parquet",
        report_path="reports/data_quality_report.json",
        sample_rows=100000,
    )
