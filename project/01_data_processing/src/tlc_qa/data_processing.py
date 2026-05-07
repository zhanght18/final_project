"""
Data processing module for NYC TLC Yellow Taxi trip records.

Facts and limitations:
- Yellow taxi trip records contain pickup/dropoff datetime, Taxi Zone pickup/dropoff IDs,
  trip distance, itemized fares, payment type, and driver-reported passenger count.
- Current TLC trip files do not contain exact pickup/dropoff longitude/latitude. PULocationID
  and DOLocationID identify TLC Taxi Zones. Any latitude/longitude derived from zone geometry
  is an approximate zone centroid, not the actual trip point.
- TLC states the trip data is submitted by authorized technology providers and TLC makes no
  representations as to the accuracy of these data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd


PAYMENT_TYPE_MAP = {
    0: "Flex Fare trip",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
}

REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "total_amount",
    "payment_type",
]


def load_dataset(source: Union[str, Path], columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load a TLC Parquet file from a local path or HTTPS URL."""
    return pd.read_parquet(source, columns=list(columns) if columns else None)


def validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if required fields are absent instead of silently inventing fields."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required TLC columns: {missing}")


def _iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate missing-rate and anomaly statistics before cleaning."""
    validate_schema(df)
    report: Dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing": {},
        "anomalies": {},
    }

    for col in REQUIRED_COLUMNS:
        missing_count = int(df[col].isna().sum())
        report["missing"][col] = {
            "missing_count": missing_count,
            "missing_rate": float(missing_count / len(df)) if len(df) else 0.0,
        }

    pickup = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    duration_min = (dropoff - pickup).dt.total_seconds() / 60

    report["anomalies"] = {
        "invalid_datetime": int(pickup.isna().sum() + dropoff.isna().sum()),
        "dropoff_not_after_pickup": int((duration_min <= 0).sum()),
        "trip_distance_le_zero": int((df["trip_distance"] <= 0).sum()),
        "passenger_count_le_zero": int((df["passenger_count"] <= 0).sum()),
        "fare_amount_negative": int((df["fare_amount"] < 0).sum()),
        "total_amount_negative": int((df["total_amount"] < 0).sum()),
        "unknown_payment_type": int((~df["payment_type"].isin(PAYMENT_TYPE_MAP.keys())).sum()),
    }

    numeric_cols = ["trip_distance", "passenger_count", "fare_amount", "total_amount"]
    report["numeric_summary"] = df[numeric_cols].describe(percentiles=[0.01, 0.5, 0.99]).to_dict()
    report["iqr_outlier_bounds"] = {
        col: dict(zip(["lower", "upper"], _iqr_bounds(df[col].dropna()))) for col in numeric_cols
    }
    return report


def clean_data(
    df: pd.DataFrame,
    *,
    max_trip_hours: float = 24,
    max_passenger_count: int = 8,
    max_trip_distance_miles: float = 200,
    max_total_amount: float = 1000,
    max_speed_mph: float = 100,
) -> pd.DataFrame:
    """Clean TLC trips with explicit, conservative rules.

    Strategy notes:
    1. Keep only required analytical columns so downstream QA answers stay grounded.
    2. Parse datetimes; drop unparseable or non-positive duration rows because duration-based
       features cannot be trusted for those rows.
    3. Drop non-positive distance, fare, and passenger counts. These values can represent errors,
       cancellations, or special cases, but they are not suitable for demand/speed/fare analytics.
    4. Apply broad caps for duration, passenger count, distance, total amount, and derived speed.
       The caps are intentionally wide to remove physically implausible records without overfitting
       to one month's distribution.
    5. Preserve PULocationID/DOLocationID as zone IDs. Do not fabricate exact coordinates.
    """
    validate_schema(df)
    cleaned = df[REQUIRED_COLUMNS].copy()

    cleaned["pickup_datetime"] = pd.to_datetime(cleaned.pop("tpep_pickup_datetime"), errors="coerce")
    cleaned["dropoff_datetime"] = pd.to_datetime(cleaned.pop("tpep_dropoff_datetime"), errors="coerce")
    cleaned["duration_min"] = (cleaned["dropoff_datetime"] - cleaned["pickup_datetime"]).dt.total_seconds() / 60

    mask = cleaned["pickup_datetime"].notna() & cleaned["dropoff_datetime"].notna()
    mask &= cleaned["duration_min"].between(1, max_trip_hours * 60, inclusive="both")
    mask &= cleaned["trip_distance"].between(0.01, max_trip_distance_miles, inclusive="both")
    mask &= cleaned["passenger_count"].between(1, max_passenger_count, inclusive="both")
    mask &= cleaned["fare_amount"] >= 0
    mask &= cleaned["total_amount"].between(0, max_total_amount, inclusive="both")
    mask &= cleaned["payment_type"].isin(PAYMENT_TYPE_MAP.keys())

    cleaned = cleaned.loc[mask].copy()
    cleaned["speed_mph"] = cleaned["trip_distance"] / (cleaned["duration_min"] / 60)
    cleaned = cleaned[cleaned["speed_mph"].between(0, max_speed_mph, inclusive="both")].copy()

    return cleaned.reset_index(drop=True)


def add_time_and_derived_features(df: pd.DataFrame, peak_hours: Iterable[int] = (7, 8, 9, 16, 17, 18, 19)) -> pd.DataFrame:
    """Add time features and at least two meaningful derived features.

    Derived features included:
    - duration_min: trip length in minutes, supporting time-cost and congestion analysis.
    - speed_mph: distance divided by duration, useful for identifying slow corridors/time windows.
    - fare_per_mile: total fare intensity, useful for pricing and route profitability analysis.
    - is_airport_trip_candidate: rough proxy based on known TLC zone IDs for JFK/LGA/EWR.
      This is a candidate flag only; it uses zone IDs, not exact coordinates.
    """
    featured = df.copy()
    peak_set = set(int(h) for h in peak_hours)

    featured["pickup_hour"] = featured["pickup_datetime"].dt.hour
    featured["pickup_weekday"] = featured["pickup_datetime"].dt.dayofweek  # Monday=0, Sunday=6
    featured["pickup_weekday_name"] = featured["pickup_datetime"].dt.day_name()
    featured["is_weekend"] = featured["pickup_weekday"].isin([5, 6])
    featured["is_peak_hour"] = featured["pickup_hour"].isin(peak_set)

    featured["fare_per_mile"] = np.where(
        featured["trip_distance"] > 0,
        featured["total_amount"] / featured["trip_distance"],
        np.nan,
    )
    featured["payment_type_name"] = featured["payment_type"].map(PAYMENT_TYPE_MAP)

    airport_zone_ids = {1, 132, 138}  # Newark Airport, JFK Airport, LaGuardia Airport in TLC zones.
    featured["is_airport_trip_candidate"] = (
        featured["PULocationID"].isin(airport_zone_ids) | featured["DOLocationID"].isin(airport_zone_ids)
    )
    featured["route_zone_pair"] = (
        featured["PULocationID"].astype(str) + "-" + featured["DOLocationID"].astype(str)
    )

    return featured


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def run_pipeline(
    source: str,
    output_path: str,
    report_path: str,
    sample_rows: Optional[int] = None,
) -> None:
    df = load_dataset(source)
    if sample_rows:
        df = df.head(sample_rows).copy()
    report = generate_quality_report(df)
    cleaned = clean_data(df)
    featured = add_time_and_derived_features(cleaned)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(output_path, index=False)
    save_json(report, report_path)

    print(f"raw rows: {len(df):,}")
    print(f"clean rows: {len(featured):,}")
    print(f"saved clean data: {output_path}")
    print(f"saved quality report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process NYC TLC Yellow Taxi trip records.")
    parser.add_argument("--source", required=True, help="Local parquet path or HTTPS parquet URL")
    parser.add_argument("--output", default="data/processed/yellow_clean.parquet")
    parser.add_argument("--report", default="reports/data_quality_report.json")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional sample for quick tests")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.source, args.output, args.report, args.sample_rows)
