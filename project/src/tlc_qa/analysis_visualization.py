"""
Analysis and visualization module for NYC TLC Yellow Taxi trip records.

This module reads the cleaned feature dataset produced by data_processing.py and saves
all charts under outputs/. It does not infer exact latitude/longitude because current
TLC Yellow Taxi trip records provide Taxi Zone IDs, not point coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_hour",
    "pickup_weekday",
    "is_weekend",
    "is_peak_hour",
    "trip_distance",
    "duration_min",
    "speed_mph",
    "passenger_count",
    "fare_amount",
    "total_amount",
    "PULocationID",
    "DOLocationID",
]

WEEKDAY_LABEL = {False: "Weekday", True: "Weekend"}


def load_clean_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load the cleaned feature dataset from Step 1."""
    df = pd.read_parquet(path)
    validate_schema(df)
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if Step 1 was not run or expected columns are missing."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns for analysis. Run Step 1 data_processing.py first. "
            f"Missing: {missing}"
        )


def _prepare_output_dir(output_dir: Union[str, Path]) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_current_figure(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return str(path)


def analyze_time_demand(df: pd.DataFrame, output_dir: Union[str, Path]) -> Dict[str, Any]:
    """Analyze demand by hour and weekday/weekend.

    Chart 1: average hourly order count split by weekday/weekend.
    The average is computed per calendar date and hour, then averaged by hour. This avoids
    months with more weekdays than weekends overwhelming the comparison.
    """
    out = _prepare_output_dir(output_dir)
    data = df.copy()
    data["pickup_date"] = data["pickup_datetime"].dt.date
    data["day_type"] = data["is_weekend"].map(WEEKDAY_LABEL)

    daily_hourly = (
        data.groupby(["pickup_date", "day_type", "pickup_hour"], observed=True)
        .size()
        .rename("orders")
        .reset_index()
    )
    avg_hourly = (
        daily_hourly.groupby(["day_type", "pickup_hour"], observed=True)["orders"]
        .mean()
        .reset_index()
        .sort_values(["day_type", "pickup_hour"])
    )

    plt.figure(figsize=(10, 5))
    for day_type, part in avg_hourly.groupby("day_type"):
        plt.plot(part["pickup_hour"], part["orders"], marker="o", label=day_type)
    plt.title("Average Hourly Trip Demand: Weekday vs Weekend")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Average Orders per Day")
    plt.xticks(range(0, 24))
    plt.legend()
    chart_path = _save_current_figure(out / "01_time_demand_avg_hourly_weekday_weekend.png")

    top_hour = avg_hourly.loc[avg_hourly["orders"].idxmax()].to_dict() if not avg_hourly.empty else {}
    return {
        "chart": chart_path,
        "table_preview": avg_hourly.head(10).to_dict(orient="records"),
        "top_average_hour_segment": top_hour,
    }


def analyze_zone_hotspots(df: pd.DataFrame, output_dir: Union[str, Path], top_n: int = 10) -> Dict[str, Any]:
    """Analyze top pickup/dropoff Taxi Zones and peak-hour distribution.

    Chart 2a: top pickup Taxi Zone IDs by pickup count.
    Chart 2b: top dropoff Taxi Zone IDs by dropoff count.
    Chart 2c: heatmap of pickup counts for top pickup zones by hour.

    Fact boundary: Zone IDs are TLC Taxi Zones. Without joining the Taxi Zone lookup file,
    this function does not assign borough/zone names or coordinates.
    """
    out = _prepare_output_dir(output_dir)
    top_pickups = df["PULocationID"].value_counts().head(top_n).rename_axis("PULocationID").reset_index(name="pickup_count")
    top_dropoffs = df["DOLocationID"].value_counts().head(top_n).rename_axis("DOLocationID").reset_index(name="dropoff_count")

    plt.figure(figsize=(10, 5))
    plt.bar(top_pickups["PULocationID"].astype(str), top_pickups["pickup_count"])
    plt.title(f"Top {top_n} Pickup Taxi Zones")
    plt.xlabel("Pickup Taxi Zone ID")
    plt.ylabel("Pickup Count")
    pickup_chart = _save_current_figure(out / "02a_top_pickup_zones.png")

    plt.figure(figsize=(10, 5))
    plt.bar(top_dropoffs["DOLocationID"].astype(str), top_dropoffs["dropoff_count"])
    plt.title(f"Top {top_n} Dropoff Taxi Zones")
    plt.xlabel("Dropoff Taxi Zone ID")
    plt.ylabel("Dropoff Count")
    dropoff_chart = _save_current_figure(out / "02b_top_dropoff_zones.png")

    top_zone_ids = top_pickups["PULocationID"].tolist()
    heat = (
        df[df["PULocationID"].isin(top_zone_ids)]
        .groupby(["PULocationID", "pickup_hour"], observed=True)
        .size()
        .rename("pickup_count")
        .reset_index()
        .pivot(index="PULocationID", columns="pickup_hour", values="pickup_count")
        .reindex(index=top_zone_ids, columns=range(24))
        .fillna(0)
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(heat.values, aspect="auto")
    plt.title(f"Hourly Pickup Heatmap for Top {top_n} Pickup Taxi Zones")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Pickup Taxi Zone ID")
    plt.xticks(range(24), range(24))
    plt.yticks(range(len(heat.index)), heat.index.astype(str))
    plt.colorbar(label="Pickup Count")
    heatmap_chart = _save_current_figure(out / "02c_top_pickup_zones_hour_heatmap.png")

    peak_zone_distribution = (
        df[df["is_peak_hour"]]
        .groupby("PULocationID", observed=True)
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .rename("peak_pickup_count")
        .reset_index()
    )

    return {
        "charts": [pickup_chart, dropoff_chart, heatmap_chart],
        "top_pickups": top_pickups.to_dict(orient="records"),
        "top_dropoffs": top_dropoffs.to_dict(orient="records"),
        "top_peak_pickup_zones": peak_zone_distribution.to_dict(orient="records"),
    }


def analyze_fare_factors(df: pd.DataFrame, output_dir: Union[str, Path], sample_size: int = 100_000) -> Dict[str, Any]:
    """Analyze relationships between fare and distance/time/passenger count.

    Chart 3a: trip distance vs fare scatter plot.
    Chart 3b: average fare by pickup hour.
    Chart 3c: average fare by passenger count.

    Sampling is used only for scatter plot rendering speed; aggregate charts use all rows.
    """
    out = _prepare_output_dir(output_dir)
    sample = df.sample(n=min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df

    plt.figure(figsize=(8, 6))
    plt.scatter(sample["trip_distance"], sample["fare_amount"], s=6, alpha=0.25)
    plt.title("Trip Distance vs Fare Amount")
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Fare Amount (USD)")
    scatter_chart = _save_current_figure(out / "03a_distance_vs_fare_scatter.png")

    hourly_fare = df.groupby("pickup_hour", observed=True)["fare_amount"].mean().reset_index(name="avg_fare_amount")
    plt.figure(figsize=(10, 5))
    plt.plot(hourly_fare["pickup_hour"], hourly_fare["avg_fare_amount"], marker="o")
    plt.title("Average Fare Amount by Pickup Hour")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Average Fare Amount (USD)")
    plt.xticks(range(24))
    hour_chart = _save_current_figure(out / "03b_avg_fare_by_hour.png")

    passenger_fare = (
        df.groupby("passenger_count", observed=True)["fare_amount"]
        .agg(avg_fare_amount="mean", trip_count="size")
        .reset_index()
        .sort_values("passenger_count")
    )
    plt.figure(figsize=(8, 5))
    plt.bar(passenger_fare["passenger_count"].astype(str), passenger_fare["avg_fare_amount"])
    plt.title("Average Fare Amount by Passenger Count")
    plt.xlabel("Passenger Count")
    plt.ylabel("Average Fare Amount (USD)")
    passenger_chart = _save_current_figure(out / "03c_avg_fare_by_passenger_count.png")

    correlations = df[["fare_amount", "trip_distance", "duration_min", "speed_mph", "passenger_count"]].corr(numeric_only=True)["fare_amount"].to_dict()

    return {
        "charts": [scatter_chart, hour_chart, passenger_chart],
        "fare_correlations": {k: None if pd.isna(v) else float(v) for k, v in correlations.items()},
        "avg_fare_by_hour_preview": hourly_fare.head(24).to_dict(orient="records"),
        "avg_fare_by_passenger_count": passenger_fare.to_dict(orient="records"),
    }


def analyze_high_value_od_routes(df: pd.DataFrame, output_dir: Union[str, Path], top_n: int = 15) -> Dict[str, Any]:
    """Self-selected insight: high-demand origin-destination route pairs.

    This is useful for transport demand planning because OD pairs reveal recurring movement
    corridors better than isolated pickup or dropoff zones.

    Chart 4: top OD zone pairs by trip count, with median fare and median speed in summary.
    """
    out = _prepare_output_dir(output_dir)
    data = df.copy()
    if "route_zone_pair" not in data.columns:
        data["route_zone_pair"] = data["PULocationID"].astype(str) + "-" + data["DOLocationID"].astype(str)

    od = (
        data.groupby("route_zone_pair", observed=True)
        .agg(
            trip_count=("route_zone_pair", "size"),
            median_fare_amount=("fare_amount", "median"),
            median_total_amount=("total_amount", "median"),
            median_distance_miles=("trip_distance", "median"),
            median_duration_min=("duration_min", "median"),
            median_speed_mph=("speed_mph", "median"),
        )
        .reset_index()
        .sort_values("trip_count", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(12, 6))
    plt.bar(od["route_zone_pair"], od["trip_count"])
    plt.title(f"Top {top_n} Origin-Destination Taxi Zone Pairs by Trip Count")
    plt.xlabel("Pickup-Dropoff Taxi Zone Pair")
    plt.ylabel("Trip Count")
    plt.xticks(rotation=45, ha="right")
    chart = _save_current_figure(out / "04_top_od_zone_pairs.png")

    return {
        "chart": chart,
        "top_od_pairs": od.to_dict(orient="records"),
        "interpretation_note": (
            "OD pairs use TLC Taxi Zone IDs. They identify zone-to-zone demand corridors, "
            "not exact street-level routes."
        ),
    }


def run_all_analyses(
    clean_path: Union[str, Path],
    output_dir: Union[str, Path] = "outputs",
    summary_path: Optional[Union[str, Path]] = None,
    top_n_zones: int = 10,
    top_n_routes: int = 15,
    scatter_sample_size: int = 100_000,
) -> Dict[str, Any]:
    """Run all four required analyses and save charts to outputs/."""
    df = load_clean_data(clean_path)
    out = _prepare_output_dir(output_dir)

    results = {
        "input_path": str(clean_path),
        "row_count": int(len(df)),
        "analysis_1_time_demand": analyze_time_demand(df, out),
        "analysis_2_zone_hotspots": analyze_zone_hotspots(df, out, top_n=top_n_zones),
        "analysis_3_fare_factors": analyze_fare_factors(df, out, sample_size=scatter_sample_size),
        "analysis_4_self_selected_od_routes": analyze_high_value_od_routes(df, out, top_n=top_n_routes),
        "fact_boundaries": [
            "Charts are computed from the cleaned dataset produced by Step 1, not directly from raw official records.",
            "PULocationID and DOLocationID are TLC Taxi Zone IDs, not exact longitude/latitude.",
            "If a sample is used for the scatter plot, it affects visualization density only; aggregate summaries use all cleaned rows.",
        ],
    }

    if summary_path is None:
        summary_path = out / "analysis_summary.json"
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    results["summary_path"] = str(summary_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TLC trip demand and fare visual analyses.")
    parser.add_argument("--input", required=True, help="Clean parquet generated by Step 1")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved charts and summary JSON")
    parser.add_argument("--summary", default=None, help="Optional analysis summary JSON path")
    parser.add_argument("--top-n-zones", type=int, default=10)
    parser.add_argument("--top-n-routes", type=int, default=15)
    parser.add_argument("--scatter-sample-size", type=int, default=100000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_all_analyses(
        clean_path=args.input,
        output_dir=args.output_dir,
        summary_path=args.summary,
        top_n_zones=args.top_n_zones,
        top_n_routes=args.top_n_routes,
        scatter_sample_size=args.scatter_sample_size,
    )
    print(json.dumps({"saved_summary": result["summary_path"], "row_count": result["row_count"]}, indent=2))
