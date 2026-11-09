"""
predict.py - command-line prototype: latitude/longitude and a start month give 12 monthly land cover
maps, class percentages per month and the change in percentage points. Results are printed and
saved with the prefix given by --out (default outputs/run).
"""
import argparse
import os

import pandas as pd

from lulc_pipeline import run

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, required=True)
ap.add_argument("--lon", type=float, required=True)
ap.add_argument("--start", required=True, help="start month, YYYY-MM")
ap.add_argument("--size-km", type=float, default=10.0)
ap.add_argument("--model", default="models/lulc_unet.pt")
ap.add_argument("--out", default="outputs/run")
ap.add_argument("--scheme", default=None, help="class scheme the model was trained with (e.g. 4veg)")
ap.add_argument("--tta", action="store_true", help="test-time augmentation (flip averaging)")
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
r = run(a.lat, a.lon, a.start, a.size_km, a.model, a.out, tta=a.tta)
print("\nClass coverage per month (%):\n", r["percent"])
print("\nMonth-to-month change (percentage points):\n", r["month_to_month"])
print("\nTotal change first -> last month (percentage points):\n", r["total_change"])
print(f"\nSaved maps + CSVs with prefix: {a.out}")