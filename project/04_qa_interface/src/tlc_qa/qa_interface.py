"""Command line question-answer interface for the NYC TLC trip data project.

The interface uses deterministic keyword/parameter matching instead of inventing
answers. It reads the cleaned Step 1 dataset, can call Step 2 visualization
functions when chart regeneration is requested, and reuses the Step 3
region-hour aggregation definition for demand questions.

Supported question types include:
1. Time demand: hourly demand and weekday/weekend patterns.
2. Region ranking: top pickup/dropoff Taxi Zone IDs.
3. Demand query/prediction proxy: region-hour historical demand summary.
4. Fare query: historical fare statistics filtered by distance/hour/passengers.
5. OD route analysis: pickup-dropoff Taxi Zone pair demand.
6. Data quality/schema boundary: row count, columns, and factual limitations.

Important factual boundary:
- PULocationID and DOLocationID are TLC Taxi Zone IDs, not exact coordinates.
- Unless a trained model service is supplied, interactive demand answers are
  historical summaries or historical-baseline estimates, not official forecasts.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_visualization import (
    analyze_fare_factors,
    analyze_high_value_od_routes,
    analyze_time_demand,
    analyze_zone_hotspots,
)
from .demand_prediction import build_region_hour_dataset


FACT_BOUNDARIES = [
    "本系统只基于输入的已清洗 TLC 行程数据回答；数据外事实不会猜测。",
    "PULocationID/DOLocationID 是 TLC Taxi Zone ID，不是真实经纬度。",
    "交互式需求问题默认给出历史统计或历史基线估计；除非另行加载已部署模型，否则不是官方预测。",
]


@dataclass
class QAResult:
    question_type: str
    answer: str
    numbers: Dict[str, Any]
    chart_paths: list[str]
    fact_boundaries: list[str]

    def to_text(self) -> str:
        lines = [f"问题类型：{self.question_type}", "", self.answer]
        if self.numbers:
            lines.append("\n数字结论：")
            for k, v in self.numbers.items():
                lines.append(f"- {k}: {v}")
        if self.chart_paths:
            lines.append("\n图表路径：")
            for path in self.chart_paths:
                lines.append(f"- {path}")
        lines.append("\n事实边界：")
        for note in self.fact_boundaries:
            lines.append(f"- {note}")
        return "\n".join(lines)


def load_clean_dataset(path: Union[str, Path], max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load Step 1 cleaned data and normalize datetime columns.

    The Step 1 pipeline stores pickup/dropoff as pickup_datetime/dropoff_datetime.
    Some raw TLC files use tpep_pickup_datetime/tpep_dropoff_datetime; aliases are
    supported so the interface can fail less often during experiments.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到输入数据文件: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=max_rows)
    else:
        df = pd.read_parquet(path)
        if max_rows is not None:
            df = df.head(max_rows).copy()

    if "pickup_datetime" not in df.columns and "tpep_pickup_datetime" in df.columns:
        df = df.rename(columns={"tpep_pickup_datetime": "pickup_datetime"})
    if "dropoff_datetime" not in df.columns and "tpep_dropoff_datetime" in df.columns:
        df = df.rename(columns={"tpep_dropoff_datetime": "dropoff_datetime"})

    required = ["pickup_datetime", "PULocationID", "DOLocationID", "trip_distance", "fare_amount", "total_amount"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"问答接口缺少必要字段，无法可靠回答: {missing}")

    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    if "dropoff_datetime" in df.columns:
        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
    if "pickup_hour" not in df.columns:
        df["pickup_hour"] = df["pickup_datetime"].dt.hour
    if "pickup_weekday" not in df.columns:
        df["pickup_weekday"] = df["pickup_datetime"].dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["pickup_weekday"].isin([5, 6])
    if "is_peak_hour" not in df.columns:
        df["is_peak_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19])
    if "duration_min" not in df.columns and "dropoff_datetime" in df.columns:
        df["duration_min"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60
    if "route_zone_pair" not in df.columns:
        df["route_zone_pair"] = df["PULocationID"].astype(str) + "-" + df["DOLocationID"].astype(str)
    return df.dropna(subset=["pickup_datetime"])


def ensure_dir(path: Union[str, Path]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_int_after_patterns(question: str, patterns: Iterable[str]) -> Optional[int]:
    for pattern in patterns:
        m = re.search(pattern, question, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def parse_float_after_patterns(question: str, patterns: Iterable[str]) -> Optional[float]:
    for pattern in patterns:
        m = re.search(pattern, question, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_hour(question: str) -> Optional[int]:
    hour = parse_int_after_patterns(
        question,
        [
            r"(\d{1,2})\s*(?:点|时|:00)",
            r"hour\s*[:=]?\s*(\d{1,2})",
            r"(\d{1,2})\s*(?:am|pm)",
        ],
    )
    if hour is None:
        return None
    if 0 <= hour <= 23:
        return hour
    return None


def parse_zone(question: str, default: Optional[int] = None) -> Optional[int]:
    zone = parse_int_after_patterns(
        question,
        [
            r"(?:区域|zone|LocationID|PULocationID|DOLocationID)\s*[:=：]?\s*(\d+)",
            r"(?:上客区|下客区|pickup zone|dropoff zone)\s*[:=：]?\s*(\d+)",
        ],
    )
    return zone if zone is not None else default


def save_simple_bar(series: pd.Series, path: Path, title: str, xlabel: str, ylabel: str) -> str:
    plt.figure(figsize=(10, 5))
    plt.bar(series.index.astype(str), series.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return str(path)


def answer_time_demand(df: pd.DataFrame, question: str, output_dir: Path) -> QAResult:
    result = analyze_time_demand(df, output_dir)
    hourly = df.groupby("pickup_hour").size().sort_index()
    top_hour = int(hourly.idxmax()) if not hourly.empty else None
    top_count = int(hourly.max()) if not hourly.empty else 0
    weekend_counts = df.groupby("is_weekend").size().to_dict()
    weekday_count = int(weekend_counts.get(False, 0))
    weekend_count = int(weekend_counts.get(True, 0))
    answer = (
        f"按当前清洗数据统计，订单量最高的上车小时是 {top_hour} 点，共 {top_count:,} 单。"
        f"工作日样本订单 {weekday_count:,} 单，周末样本订单 {weekend_count:,} 单。"
    )
    return QAResult(
        "时段需求查询",
        answer,
        {"top_pickup_hour": top_hour, "top_hour_trip_count": top_count, "weekday_rows": weekday_count, "weekend_rows": weekend_count},
        [result["chart"]],
        FACT_BOUNDARIES,
    )


def answer_region_ranking(df: pd.DataFrame, question: str, output_dir: Path) -> QAResult:
    top_n = parse_int_after_patterns(question, [r"top\s*(\d+)", r"前\s*(\d+)" ]) or 10
    is_dropoff = any(k in question.lower() for k in ["下客", "dropoff", "do", "dolocation"])
    col = "DOLocationID" if is_dropoff else "PULocationID"
    label = "下客" if is_dropoff else "上客"
    counts = df[col].value_counts().head(top_n)
    chart_path = save_simple_bar(counts, output_dir / f"qa_top_{col.lower()}_{top_n}.png", f"Top {top_n} {label} Taxi Zones", col, "Trip Count")
    top_zone = int(counts.index[0]) if not counts.empty else None
    top_count = int(counts.iloc[0]) if not counts.empty else 0
    answer = f"当前清洗数据中，{label}量最高的 Taxi Zone ID 是 {top_zone}，共 {top_count:,} 单。已生成 TOP {top_n} 排名图。"
    return QAResult(
        "区域排名查询",
        answer,
        {"ranking_column": col, "top_zone_id": top_zone, "top_zone_trip_count": top_count, "top_n": top_n},
        [chart_path],
        FACT_BOUNDARIES,
    )


def answer_demand_prediction(df: pd.DataFrame, question: str, output_dir: Path) -> QAResult:
    zone = parse_zone(question)
    hour = parse_hour(question)
    # Reuse Step 3 target definition. Rename to match the Step 3 function API.
    tmp = df[["pickup_datetime", "PULocationID"]].rename(columns={"pickup_datetime": "model_pickup_datetime"})
    demand = build_region_hour_dataset(tmp, pickup_time_col="model_pickup_datetime", region_col="PULocationID")
    output_dataset = output_dir / "qa_region_hour_demand_dataset.parquet"
    demand.to_parquet(output_dataset, index=False)

    filtered = demand.copy()
    if zone is not None:
        filtered = filtered[filtered["PULocationID"] == int(zone)]
    if hour is not None:
        filtered = filtered[filtered["hour"] == int(hour)]

    if filtered.empty:
        answer = "在当前数据中没有找到匹配的区域/时段记录，因此无法给出基于事实的需求数值。"
        numbers: Dict[str, Any] = {"requested_zone": zone, "requested_hour": hour, "matched_rows": 0}
    else:
        avg_demand = float(filtered["trip_count"].mean())
        median_demand = float(filtered["trip_count"].median())
        max_demand = int(filtered["trip_count"].max())
        answer = (
            "基于 Step 3 的区域-小时需求定义，当前交互接口给出历史基线估计："
            f"平均每小时约 {avg_demand:.2f} 单，中位数 {median_demand:.2f} 单，样本最大值 {max_demand} 单。"
            "这不是官方预测，也不是已部署神经网络的实时输出。"
        )
        numbers = {
            "requested_zone": zone if zone is not None else "all_zones",
            "requested_hour": hour if hour is not None else "all_hours",
            "matched_region_hour_rows": int(len(filtered)),
            "avg_trip_count": round(avg_demand, 3),
            "median_trip_count": round(median_demand, 3),
            "max_trip_count": max_demand,
        }

    chart_series = filtered.groupby("hour")["trip_count"].mean().reindex(range(24), fill_value=0) if not filtered.empty else pd.Series(dtype=float)
    chart_path = ""
    if not chart_series.empty:
        chart_path = save_simple_bar(chart_series, output_dir / "qa_demand_by_hour_baseline.png", "Historical Baseline Demand by Hour", "Hour", "Average Trip Count")
    charts = [str(output_dataset)] + ([chart_path] if chart_path else [])
    return QAResult("需求预测/历史基线查询", answer, numbers, charts, FACT_BOUNDARIES)


def answer_fare_query(df: pd.DataFrame, question: str, output_dir: Path) -> QAResult:
    distance = parse_float_after_patterns(question, [r"(\d+(?:\.\d+)?)\s*(?:英里|mile|miles)", r"距离\s*[:=：]?\s*(\d+(?:\.\d+)?)"])
    hour = parse_hour(question)
    passengers = parse_int_after_patterns(question, [r"(\d+)\s*(?:人|位乘客|passengers?)", r"乘客\s*[:=：]?\s*(\d+)"])

    filtered = df.copy()
    if distance is not None:
        low = max(0, distance * 0.8)
        high = distance * 1.2 if distance > 0 else 0.5
        filtered = filtered[filtered["trip_distance"].between(low, high, inclusive="both")]
    if hour is not None:
        filtered = filtered[filtered["pickup_hour"] == hour]
    if passengers is not None and "passenger_count" in filtered.columns:
        filtered = filtered[filtered["passenger_count"] == passengers]

    charts = analyze_fare_factors(df, output_dir, sample_size=50_000)["charts"]
    if filtered.empty:
        answer = "当前数据中没有足够的相似历史行程，因此不能可靠估计可能费用。"
        numbers = {"matched_rows": 0, "distance_miles": distance, "hour": hour, "passengers": passengers}
    else:
        fare_median = float(filtered["fare_amount"].median())
        total_median = float(filtered["total_amount"].median())
        fare_mean = float(filtered["fare_amount"].mean())
        answer = (
            f"按相似历史行程过滤后，共匹配 {len(filtered):,} 条记录。"
            f"fare_amount 中位数为 ${fare_median:.2f}，total_amount 中位数为 ${total_median:.2f}。"
        )
        numbers = {
            "matched_rows": int(len(filtered)),
            "median_fare_amount_usd": round(fare_median, 2),
            "mean_fare_amount_usd": round(fare_mean, 2),
            "median_total_amount_usd": round(total_median, 2),
            "distance_miles_filter": distance,
            "hour_filter": hour,
            "passengers_filter": passengers,
        }
    return QAResult("可能费用查询", answer, numbers, list(charts), FACT_BOUNDARIES)


def answer_od_routes(df: pd.DataFrame, question: str, output_dir: Path) -> QAResult:
    top_n = parse_int_after_patterns(question, [r"top\s*(\d+)", r"前\s*(\d+)" ]) or 15
    result = analyze_high_value_od_routes(df, output_dir, top_n=top_n)
    top = result["top_od_pairs"][0] if result["top_od_pairs"] else {}
    answer = "当前数据中最高频 OD Taxi Zone 对为 {pair}，共 {count:,} 单。".format(
        pair=top.get("route_zone_pair", "无法确认"), count=int(top.get("trip_count", 0))
    )
    return QAResult(
        "OD 热门线路查询",
        answer,
        {"top_route_zone_pair": top.get("route_zone_pair"), "top_route_trip_count": int(top.get("trip_count", 0)), "top_n": top_n},
        [result["chart"]],
        FACT_BOUNDARIES,
    )


def answer_quality_or_scope(df: pd.DataFrame, question: str, output_dir: Path) -> QAResult:
    missing_rate = df.isna().mean().sort_values(ascending=False).head(10)
    chart_path = save_simple_bar(missing_rate, output_dir / "qa_missing_rate_top10.png", "Top Missing Rates in Loaded Dataset", "Column", "Missing Rate")
    answer = (
        f"当前加载数据共有 {len(df):,} 行、{df.shape[1]:,} 列。"
        "系统可回答时段需求、区域排名、需求历史基线/预测准备、可能费用、OD 热门线路等问题。"
    )
    return QAResult(
        "数据质量/能力边界查询",
        answer,
        {"row_count": int(len(df)), "column_count": int(df.shape[1]), "top_missing_rates": missing_rate.round(4).to_dict()},
        [chart_path],
        FACT_BOUNDARIES,
    )


def classify_question(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["费用", "车费", "fare", "amount", "多少钱", "价格"]):
        return "fare"
    if any(k in q for k in ["预测", "需求", "demand", "forecast", "预计"]):
        # OD demand questions should still route to OD if explicit route wording exists.
        if any(k in q for k in ["od", "线路", "路线", "route"]):
            return "od"
        return "demand"
    if any(k in q for k in ["区域", "上客", "下客", "zone", "排名", "top", "热点"]):
        return "region"
    if any(k in q for k in ["小时", "几点", "高峰", "周末", "工作日", "time", "hour", "peak"]):
        return "time"
    if any(k in q for k in ["od", "线路", "路线", "route", "起终点"]):
        return "od"
    if any(k in q for k in ["质量", "缺失", "字段", "能问", "帮助", "help", "schema"]):
        return "quality"
    return "quality"


def answer_question(df: pd.DataFrame, question: str, output_dir: Union[str, Path]) -> QAResult:
    out = ensure_dir(output_dir)
    qtype = classify_question(question)
    if qtype == "time":
        return answer_time_demand(df, question, out)
    if qtype == "region":
        return answer_region_ranking(df, question, out)
    if qtype == "demand":
        return answer_demand_prediction(df, question, out)
    if qtype == "fare":
        return answer_fare_query(df, question, out)
    if qtype == "od":
        return answer_od_routes(df, question, out)
    return answer_quality_or_scope(df, question, out)


def interactive_loop(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    print("NYC TLC 出行数据问答系统已启动。输入 exit/quit/退出 结束。")
    print("示例：哪个小时订单最多？ / 上客 TOP 10 区域 / 区域 132 在 8 点的需求预测 / 距离 3 英里 18 点 1 人可能车费 / OD 热门线路")
    while True:
        question = input("\n请输入问题> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q", "退出", "结束"}:
            print("已退出。")
            break
        try:
            result = answer_question(df, question, output_dir)
            print("\n" + result.to_text())
        except Exception as exc:  # Keep CLI alive while being honest about failures.
            print(f"\n无法回答该问题：{exc}")
            print("请检查字段是否存在，或换一种更明确的问题表达。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive QA over cleaned NYC TLC trip data.")
    parser.add_argument("--input", required=True, help="Step 1 cleaned parquet/csv file.")
    parser.add_argument("--output-dir", default="outputs/qa", help="Directory for QA charts and derived files.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick demos.")
    parser.add_argument("--question", default=None, help="Run one question and exit instead of interactive loop.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON for one-shot mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_clean_dataset(args.input, max_rows=args.max_rows)
    if args.question:
        result = answer_question(df, args.question, args.output_dir)
        if args.json:
            print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))
        else:
            print(result.to_text())
    else:
        interactive_loop(df, args.output_dir)


if __name__ == "__main__":
    main()
