"""
Forecasting: compares multiple models on monthly revenue and exports
the best-performing one's predictions to outputs/forecast.csv.
Models: Linear Regression, Random Forest, XGBoost (if installed).
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DB_PATH = Path(__file__).parent.parent / "data" / "ecommerce.db"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "forecast.csv"
METRICS_PATH = Path(__file__).parent.parent / "outputs" / "model_metrics.csv"

FORECAST_MONTHS = 3
FEATURE_COLS = ["t", "lag1", "lag2", "lag3", "rolling3", "order_count"]


def build_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("year_month")
        .agg(revenue=("revenue", "sum"), order_count=("order_id", "nunique"))
        .reset_index()
        .sort_values("year_month")
    )

    # Drop the last month if it looks incomplete (less than 20% of median revenue)
    median_rev = monthly["revenue"].median()
    if monthly["revenue"].iloc[-1] < median_rev * 0.2:
        print(f"  [INFO] Dropping incomplete month: {monthly['year_month'].iloc[-1]} (R${monthly['revenue'].iloc[-1]:,.0f})")
        monthly = monthly.iloc[:-1]

    monthly["t"] = np.arange(len(monthly))
    monthly["lag1"] = monthly["revenue"].shift(1)
    monthly["lag2"] = monthly["revenue"].shift(2)
    monthly["lag3"] = monthly["revenue"].shift(3)
    monthly["rolling3"] = monthly["revenue"].shift(1).rolling(3).mean()
    return monthly.dropna().reset_index(drop=True)


def evaluate_models(X_train, y_train, X_test, y_test) -> dict:
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    except ImportError:
        print("  [INFO] XGBoost not installed, skipping. Run: pip install xgboost")

    baseline_mae = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        improvement = (baseline_mae - mae) / baseline_mae * 100
        results[name] = {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "baseline_mae": baseline_mae,
            "pct_improvement": improvement,
        }
        print(f"  {name}: MAE=R${mae:,.0f}  RMSE=R${rmse:,.0f}  vs baseline: {improvement:+.1f}%")

    return results


def forecast_future(best_model, monthly: pd.DataFrame) -> pd.DataFrame:
    history_revenue = list(monthly["revenue"])
    history_orders = list(monthly["order_count"])
    last_t = monthly["t"].iloc[-1]

    # Generate real future month strings from the last known month
    last_month = pd.Period(monthly["year_month"].iloc[-1], freq="M")

    future_rows = []
    for i in range(1, FORECAST_MONTHS + 1):
        lag1 = history_revenue[-1]
        lag2 = history_revenue[-2] if len(history_revenue) >= 2 else lag1
        lag3 = history_revenue[-3] if len(history_revenue) >= 3 else lag1
        rolling3 = np.mean(history_revenue[-3:])
        order_count = int(np.mean(history_orders[-3:]))

        row = {
            "t": last_t + i,
            "lag1": lag1, "lag2": lag2, "lag3": lag3,
            "rolling3": rolling3, "order_count": order_count,
        }
        pred = max(best_model.predict(pd.DataFrame([row]))[0], 0)
        history_revenue.append(pred)
        history_orders.append(order_count)
        future_month = str(last_month + i)
        future_rows.append({"year_month": future_month, "revenue": pred, "type": "forecast"})

    return pd.DataFrame(future_rows)


def run():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT year_month, order_id, revenue FROM fact_orders", conn)
    conn.close()

    print("Building monthly features...")
    monthly = build_monthly_features(df)
    print(f"  Months available: {len(monthly)}")

    split = len(monthly) - FORECAST_MONTHS
    train, test = monthly.iloc[:split], monthly.iloc[split:]
    X_train, y_train = train[FEATURE_COLS], train["revenue"]
    X_test, y_test = test[FEATURE_COLS], test["revenue"]

    print("\nEvaluating models...")
    results = evaluate_models(X_train, y_train, X_test, y_test)

    best_name = min(results, key=lambda k: results[k]["mae"])
    best = results[best_name]
    print(f"\n  Best model: {best_name} (MAE improvement over baseline: {best['pct_improvement']:+.1f}%)")

    # Refit best model on all available data before forecasting
    best["model"].fit(monthly[FEATURE_COLS], monthly["revenue"])

    actuals = monthly[["year_month", "revenue"]].copy()
    actuals["type"] = "actual"
    future = forecast_future(best["model"], monthly)
    forecast_df = pd.concat([actuals, future], ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(OUTPUT_PATH, index=False)

    metrics_df = pd.DataFrame([
        {
            "model": k,
            "mae": v["mae"],
            "rmse": v["rmse"],
            "baseline_mae": v["baseline_mae"],
            "pct_improvement": v["pct_improvement"],
        }
        for k, v in results.items()
    ])
    metrics_df.to_csv(METRICS_PATH, index=False)

    print(f"\n  Forecast saved to: {OUTPUT_PATH}")
    print(f"  Model metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    run()
