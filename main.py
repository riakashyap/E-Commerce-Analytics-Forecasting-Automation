"""
Main pipeline runner — executes all stages in order.
Usage: python main.py
"""

import time
from pipeline import ingest, clean, anomaly, forecast


def run_stage(name: str, fn):
    print(f"\n{'=' * 50}")
    print(f"STAGE: {name}")
    print("=" * 50)
    start = time.time()
    fn()
    print(f"  Completed in {time.time() - start:.1f}s")


if __name__ == "__main__":
    print("E-Commerce Analytics Pipeline")
    print("Starting...\n")
    total_start = time.time()

    run_stage("1. Ingest", ingest.ingest_all)
    run_stage("2. Clean", clean.run)
    run_stage("3. Anomaly Detection", anomaly.run)
    run_stage("4. Forecasting", forecast.run)

    print(f"\n{'=' * 50}")
    print(f"Pipeline complete in {time.time() - total_start:.1f}s")
    print("Outputs saved to outputs/")
    print("=" * 50)
