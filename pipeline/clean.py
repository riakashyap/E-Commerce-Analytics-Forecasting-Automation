"""
Cleaning: joins core tables, standardizes types, drops bad rows,
and writes a master fact table back to the database.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "ecommerce.db"

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def load_and_join(conn) -> pd.DataFrame:
    orders = pd.read_sql("SELECT * FROM orders", conn)
    items = pd.read_sql("SELECT * FROM order_items", conn)
    payments = pd.read_sql(
        "SELECT order_id, SUM(payment_value) as total_payment FROM order_payments GROUP BY order_id",
        conn,
    )
    customers = pd.read_sql(
        "SELECT customer_id, customer_state, customer_city FROM customers", conn
    )
    products = pd.read_sql(
        "SELECT product_id, product_category_name FROM products", conn
    )
    translation = pd.read_sql("SELECT * FROM category_translation", conn)

    df = (
        orders.merge(items, on="order_id", how="left")
        .merge(payments, on="order_id", how="left")
        .merge(customers, on="customer_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(translation, on="product_category_name", how="left")
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["order_purchase_timestamp", "price"])
    df = df[df["order_status"].isin(["delivered", "shipped", "invoiced"])]

    df["revenue"] = df["price"] + df["freight_value"].fillna(0)
    df["order_date"] = df["order_purchase_timestamp"].dt.date
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    p999 = df["price"].quantile(0.999)
    df["price_outlier"] = df["price"] > p999

    return df.reset_index(drop=True)


def run():
    conn = sqlite3.connect(DB_PATH)
    print("Joining tables...")
    df = load_and_join(conn)
    print(f"  Raw joined rows: {len(df):,}")

    print("Cleaning...")
    df = clean(df)
    print(f"  Clean rows: {len(df):,}")

    df.to_sql("fact_orders", conn, if_exists="replace", index=False)
    conn.close()

    processed_path = Path(__file__).parent.parent / "data" / "processed" / "fact_orders.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"  Saved to: {processed_path}")


if __name__ == "__main__":
    run()
