"""
Anomaly detection: flags unusual orders using IQR and Isolation Forest.
Outputs a CSV of flagged records for review / Power BI highlighting.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

DB_PATH = Path(__file__).parent.parent / "data" / "ecommerce.db"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "anomalies.csv"


def iqr_flag(series: pd.Series, k: float = 4.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def run():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM fact_orders", conn)
    conn.close()

    features = df[["price", "freight_value", "revenue"]].fillna(0)

    # IQR on price only — freight varies too much by region to flag reliably
    df["iqr_price_flag"] = iqr_flag(df["price"])

    # Isolation Forest catches unusual combinations across all three features
    iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    df["iso_score"] = iso.fit_predict(features)
    df["iso_anomaly"] = df["iso_score"] == -1

    # A row is anomalous if caught by either method
    df["is_anomaly"] = df["iqr_price_flag"] | df["iso_anomaly"]

    anomalies = df[df["is_anomaly"]][[
        "order_id", "order_date", "price", "freight_value", "revenue",
        "customer_state", "product_category_name_english",
        "iqr_price_flag", "iso_anomaly",
    ]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anomalies.to_csv(OUTPUT_PATH, index=False)

    total = len(df)
    flagged = len(anomalies)
    print(f"  Anomalies detected: {flagged:,} / {total:,} rows ({flagged/total*100:.1f}%)")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
