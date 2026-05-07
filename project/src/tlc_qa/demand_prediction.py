"""Demand prediction module for NYC TLC trip data.

This module predicts trip demand for a pickup zone and time slot.
The target is explicitly defined as the number of trips after grouping by
pickup Taxi Zone ID and pickup hour. It is not an official TLC demand label.

Example:
    python -m src.tlc_qa.demand_prediction \
      --input data/processed/yellow_tripdata_2026-01_clean.parquet \
      --output-dir outputs/modeling \
      --region-col PULocationID \
      --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass
class PredictionConfig:
    input_path: str
    output_dir: str = "outputs/modeling"
    pickup_time_col: str = "pickup_datetime"
    region_col: str = "PULocationID"
    test_size: float = 0.2
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.001
    random_state: int = 42
    max_rows: Optional[int] = None
    min_region_hour_count: int = 1


class DemandMLP(nn.Module):
    """Small feed-forward neural network for tabular demand regression."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_clean_data(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load cleaned trip data.

    Parquet is preferred because TLC monthly files and the previous module use it.
    CSV support is included for debugging or small samples.
    """
    input_path = Path(path)
    if not input_path.exists() and not path.startswith("http"):
        raise FileNotFoundError(f"Input file not found: {path}")

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=max_rows)
    else:
        df = pd.read_parquet(path)
        if max_rows is not None:
            df = df.head(max_rows)
    return df


def build_region_hour_dataset(
    df: pd.DataFrame,
    pickup_time_col: str = "pickup_datetime",
    region_col: str = "PULocationID",
    min_region_hour_count: int = 1,
) -> pd.DataFrame:
    """Aggregate trips into a supervised region-hour demand dataset.

    Strategy and reasons:
    1. Convert pickup time to pandas datetime and drop invalid timestamps because
       the target depends on reliable temporal grouping.
    2. Keep records with non-null region IDs because the model predicts demand
       for specific pickup zones.
    3. Group by pickup zone and hourly timestamp. The count is the target demand.
    4. Create calendar features known before prediction time.
    5. Create lag features from prior demand. Lag features improve forecasting
       while avoiding direct use of the current target.
    """
    required = [pickup_time_col, region_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[required].copy()
    data[pickup_time_col] = pd.to_datetime(data[pickup_time_col], errors="coerce")
    data = data.dropna(subset=[pickup_time_col, region_col])
    data[region_col] = data[region_col].astype(int)
    data["pickup_hour_ts"] = data[pickup_time_col].dt.floor("h")

    grouped = (
        data.groupby([region_col, "pickup_hour_ts"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
    )

    # Optional low-volume filter. Keeping the default at 1 preserves all observed
    # region-hour cells. Higher values are useful for noisy demos.
    grouped = grouped[grouped["trip_count"] >= min_region_hour_count].copy()

    grouped["hour"] = grouped["pickup_hour_ts"].dt.hour
    grouped["day_of_week"] = grouped["pickup_hour_ts"].dt.dayofweek
    grouped["is_weekend"] = grouped["day_of_week"].isin([5, 6]).astype(int)
    grouped["is_peak_hour"] = grouped["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    grouped["month"] = grouped["pickup_hour_ts"].dt.month
    grouped["day_of_month"] = grouped["pickup_hour_ts"].dt.day

    grouped = grouped.sort_values([region_col, "pickup_hour_ts"])
    grouped["lag_1h_demand"] = grouped.groupby(region_col)["trip_count"].shift(1)
    grouped["lag_24h_demand"] = grouped.groupby(region_col)["trip_count"].shift(24)
    grouped["rolling_3h_mean_demand"] = (
        grouped.groupby(region_col)["trip_count"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Missing lag values occur at the start of each region's history. Fill with 0
    # because no prior observation exists in the available sample.
    lag_cols = ["lag_1h_demand", "lag_24h_demand", "rolling_3h_mean_demand"]
    grouped[lag_cols] = grouped[lag_cols].fillna(0.0)
    return grouped


def make_train_test_matrices(
    dataset: pd.DataFrame,
    region_col: str,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Create chronological 8:2 train/test matrices.

    We split by time rather than random rows to better mimic forecasting and to
    reduce leakage from future periods into the training set.
    """
    dataset = dataset.sort_values("pickup_hour_ts").reset_index(drop=True)
    split_idx = int(len(dataset) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(dataset):
        raise ValueError("Not enough aggregated rows for an 8:2 train/test split.")

    train_df = dataset.iloc[:split_idx].copy()
    test_df = dataset.iloc[split_idx:].copy()

    feature_cols = [
        region_col,
        "hour",
        "day_of_week",
        "is_weekend",
        "is_peak_hour",
        "month",
        "day_of_month",
        "lag_1h_demand",
        "lag_24h_demand",
        "rolling_3h_mean_demand",
    ]

    # One-hot encode region and calendar categories. The train columns define the
    # schema; test is aligned to the same schema to avoid train/test mismatch.
    categorical = [region_col, "hour", "day_of_week", "month"]
    train_x = pd.get_dummies(train_df[feature_cols], columns=categorical, drop_first=False)
    test_x = pd.get_dummies(test_df[feature_cols], columns=categorical, drop_first=False)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_x).astype(np.float32)
    X_test = scaler.transform(test_x).astype(np.float32)
    y_train = train_df["trip_count"].to_numpy(dtype=np.float32)
    y_test = test_df["trip_count"].to_numpy(dtype=np.float32)
    return X_train, X_test, y_train, y_test, list(train_x.columns), train_df, test_df, scaler


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: PredictionConfig,
) -> Tuple[DemandMLP, Dict[str, List[float]], np.ndarray]:
    if torch is None:
        raise ImportError("PyTorch is not installed. Install dependencies with: pip install -r requirements.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    history = {"train_loss": [], "test_loss": []}
    X_test_tensor = torch.tensor(X_test).to(device)
    y_test_tensor = torch.tensor(y_test).to(device)

    for _epoch in range(cfg.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = loss_fn(test_pred, y_test_tensor)
        history["train_loss"].append(float(np.mean(losses)))
        history["test_loss"].append(float(test_loss.detach().cpu().item()))

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).detach().cpu().numpy()
    preds = np.clip(preds, 0, None)  # Demand cannot be negative.
    return model, history, preds


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
) -> Tuple[RandomForestRegressor, np.ndarray]:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)
    return model, preds


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save_loss_curve(history: Dict[str, List[float]], output_dir: Path) -> Path:
    path = output_dir / "05_nn_loss_curve.png"
    plt.figure(figsize=(9, 5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["test_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Neural network training and test loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_model_comparison(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> Path:
    path = output_dir / "06_model_comparison_mae_rmse.png"
    metric_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    metric_df.plot(x="model", y=["MAE", "RMSE"], kind="bar", figsize=(8, 5))
    plt.ylabel("Trips")
    plt.title("Demand prediction model comparison on test set")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_predictions_sample(test_df: pd.DataFrame, y_true: np.ndarray, nn_pred: np.ndarray, rf_pred: np.ndarray, output_dir: Path) -> Path:
    path = output_dir / "test_predictions_sample.csv"
    out = test_df[["pickup_hour_ts", "trip_count"]].copy()
    out["actual_trip_count"] = y_true
    out["nn_pred_trip_count"] = nn_pred
    out["rf_pred_trip_count"] = rf_pred
    out.head(1000).to_csv(path, index=False)
    return path


def run_pipeline(cfg: PredictionConfig) -> Dict[str, object]:
    set_seed(cfg.random_state)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_data(cfg.input_path, cfg.max_rows)
    dataset = build_region_hour_dataset(
        df,
        pickup_time_col=cfg.pickup_time_col,
        region_col=cfg.region_col,
        min_region_hour_count=cfg.min_region_hour_count,
    )
    if dataset.empty:
        raise ValueError("No aggregated region-hour rows were produced. Check input data and column names.")

    demand_dataset_path = output_dir / "region_hour_demand_dataset.parquet"
    dataset.to_parquet(demand_dataset_path, index=False)

    X_train, X_test, y_train, y_test, feature_names, train_df, test_df, _scaler = make_train_test_matrices(
        dataset, cfg.region_col, cfg.test_size
    )

    _nn_model, history, nn_pred = train_neural_network(X_train, y_train, X_test, y_test, cfg)
    rf_model, rf_pred = train_random_forest(X_train, y_train, X_test, cfg.random_state)

    metrics = {
        "neural_network": regression_metrics(y_test, nn_pred),
        "random_forest": regression_metrics(y_test, rf_pred),
    }

    loss_path = save_loss_curve(history, output_dir)
    comparison_path = save_model_comparison(metrics, output_dir)
    sample_path = save_predictions_sample(test_df, y_test, nn_pred, rf_pred, output_dir)

    metrics_path = output_dir / "demand_prediction_metrics.json"
    result = {
        "config": asdict(cfg),
        "target_definition": "trip_count grouped by pickup Taxi Zone ID and pickup hour",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count_after_encoding": int(X_train.shape[1]),
        "metrics": metrics,
        "outputs": {
            "region_hour_dataset": str(demand_dataset_path),
            "loss_curve": str(loss_path),
            "model_comparison": str(comparison_path),
            "prediction_sample": str(sample_path),
        },
        "method_notes": {
            "split": "Chronological 8:2 split, not random row split, to better simulate forecasting.",
            "neural_network": "MLP can model nonlinear interactions but requires tuning and sufficient data.",
            "random_forest": "Strong tabular baseline; less tuning, often robust on small/medium structured data, but less extrapolative over unseen time patterns.",
        },
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save feature importances for the random forest baseline.
    fi = pd.DataFrame({"feature": feature_names, "importance": rf_model.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(output_dir / "random_forest_feature_importance.csv", index=False)

    return result


def parse_args() -> PredictionConfig:
    parser = argparse.ArgumentParser(description="Train NN and Random Forest demand prediction models.")
    parser.add_argument("--input", required=True, dest="input_path", help="Cleaned trip data parquet/csv path.")
    parser.add_argument("--output-dir", default="outputs/modeling")
    parser.add_argument("--pickup-time-col", default="pickup_datetime")
    parser.add_argument("--region-col", default="PULocationID")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-region-hour-count", type=int, default=1)
    args = parser.parse_args()
    return PredictionConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    result = run_pipeline(cfg)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
