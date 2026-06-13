"""
Ingestion: loads all Olist CSVs from archive/ into a SQLite database.
Run this once (or incrementally) before the rest of the pipeline.
"""

import sqlite3
import pandas as pd
from pathlib import Path

ARCHIVE_DIR = Path(__file__).parent.parent / "archive"
DB_PATH = Path(__file__).parent.parent / "data" / "ecommerce.db"

TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def ingest_all():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    for table, filename in TABLES.items():
        filepath = ARCHIVE_DIR / filename
        if not filepath.exists():
            print(f"  [SKIP] {filename} not found")
            continue
        df = pd.read_csv(filepath)
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"  [OK]   {table}: {len(df):,} rows")

    conn.close()
    print(f"\nDatabase written to: {DB_PATH}")


if __name__ == "__main__":
    print("Ingesting Olist CSVs into SQLite...")
    ingest_all()
