"""
collect_data.py - builds the training dataset from the site lists below.

Each site gives 256 x 256 px tiles (7.68 km at 30 m) saved as .npz files with the monthly
Sentinel-1 VV/VH composites for 2021 (dB) and the ESA WorldCover 2021 labels. The split is by city,
so the test cities are never seen in training. Tiles that already exist are skipped.
"""
import argparse
import csv
import os
import random
import traceback

import numpy as np
from tqdm import tqdm

from config import TILE_PX, RESOLUTION_M, TRAIN_YEAR
from features import clean_s1
from geo_io import get_catalog, load_s1_series, load_worldcover

TRAIN_SITES = [
    ("berlin", 52.52, 13.40), ("munich", 48.14, 11.58), ("mannheim", 49.49, 8.47),
    ("paris", 48.86, 2.35), ("madrid", 40.42, -3.70), ("rome", 41.90, 12.50),
    ("warsaw", 52.23, 21.01), ("london", 51.51, -0.13), ("stockholm", 59.33, 18.07),
    ("athens", 37.98, 23.73), ("istanbul", 41.01, 28.98), ("cairo", 30.04, 31.24),
    ("lagos", 6.52, 3.38), ("nairobi", -1.29, 36.82), ("johannesburg", -26.20, 28.05),
    ("capetown", -33.92, 18.42), ("delhi", 28.61, 77.21), ("mumbai", 19.08, 72.88),
    ("bengaluru", 12.97, 77.59), ("pune", 18.52, 73.86), ("dhaka", 23.81, 90.41),
    ("bangkok", 13.76, 100.50), ("jakarta", -6.21, 106.85), ("manila", 14.60, 120.98),
    ("shanghai", 31.23, 121.47), ("beijing", 39.90, 116.41), ("tokyo", 35.68, 139.69),
    ("seoul", 37.57, 126.98), ("sydney", -33.87, 151.21), ("melbourne", -37.81, 144.96),
    ("newyork", 40.71, -74.01), ("chicago", 41.88, -87.63), ("losangeles", 34.05, -118.24),
    ("houston", 29.76, -95.37), ("mexicocity", 19.43, -99.13), ("bogota", 4.71, -74.07),
    ("saopaulo", -23.55, -46.63), ("lima", -12.05, -77.04), ("buenosaires", -34.60, -58.38),
    ("toronto", 43.65, -79.38), ("riyadh", 24.71, 46.68), ("tehran", 35.69, 51.39),
    ("hochiminh", 10.82, 106.63), ("kyiv", 50.45, 30.52),
]
TEST_SITES = [
    ("hamburg", 53.55, 9.99), ("vienna", 48.21, 16.37), ("hyderabad", 17.39, 78.48),
    ("accra", 5.60, -0.19), ("chengdu", 30.57, 104.07), ("denver", 39.74, -104.99),
    ("santiago", -33.45, -70.67), ("brisbane", -27.47, 153.03),
]


def jitter(lat, lon, km, rng):
    dlat = rng.uniform(-km, km) / 111.32
    dlon = rng.uniform(-km, km) / (111.32 * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def center_crop(a, n):
    h, w = a.shape[-2:]
    y0, x0 = (h - n) // 2, (w - n) // 2
    return a[..., y0:y0 + n, x0:x0 + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--tiles-per-site", type=int, default=4)
    ap.add_argument("--jitter-km", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-missing-months", type=int, default=2,
                    help="skip a tile if more than this many of the 12 months have no Sentinel-1 scenes")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/tiles", exist_ok=True)
    rng = random.Random(args.seed)
    cat = get_catalog()
    months = [(TRAIN_YEAR, m) for m in range(1, 13)]
    size_km = (TILE_PX + 8) * RESOLUTION_M / 1000.0

    jobs = []
    for i, (name, lat, lon) in enumerate(TRAIN_SITES):
        split = "val" if i % 5 == 4 else "train"
        for t in range(args.tiles_per_site):
            la, lo = (lat, lon) if t == 0 else jitter(lat, lon, args.jitter_km, rng)
            jobs.append((name, t, la, lo, split))
    for (name, lat, lon) in TEST_SITES:
        for t in range(args.tiles_per_site):
            la, lo = (lat, lon) if t == 0 else jitter(lat, lon, args.jitter_km, rng)
            jobs.append((name, t, la, lo, "test"))

    index_path = f"{args.out}/index.csv"
    rows = []
    reasons = {"saved": 0, "already_existed": 0, "too_many_missing_months": 0,
               "no_worldcover": 0, "tile_too_small": 0, "failed": 0}
    problems = []
    for name, t, la, lo, split in tqdm(jobs, desc="tiles"):
        fn = f"{name}_{t}.npz"
        path = f"{args.out}/tiles/{fn}"
        if os.path.exists(path):
            rows.append((fn, split, name, la, lo))
            reasons["already_existed"] += 1
            continue
        try:
            s1, gb, missing = load_s1_series(cat, la, lo, size_km, months)
            if len(missing) > args.max_missing_months:
                reasons["too_many_missing_months"] += 1
                problems.append((fn, f"{len(missing)} of 12 months have no Sentinel-1 scenes"))
                tqdm.write(f"skip {fn}: {len(missing)} months without S1")
                continue
            label = load_worldcover(cat, gb, TRAIN_YEAR)
            if label is None:
                reasons["no_worldcover"] += 1
                problems.append((fn, "no WorldCover tile found"))
                tqdm.write(f"skip {fn}: no WorldCover")
                continue
            s1 = clean_s1(s1)
            if s1.shape[-1] < TILE_PX or s1.shape[-2] < TILE_PX:
                reasons["tile_too_small"] += 1
                problems.append((fn, f"tile only {s1.shape[-2]}x{s1.shape[-1]} px (need {TILE_PX})"))
                tqdm.write(f"skip {fn}: tile too small {s1.shape[-2:]}")
                continue
            s1 = center_crop(s1, TILE_PX).astype(np.float16)
            label = center_crop(label, TILE_PX)
            np.savez(path, s1=s1, label=label)
            rows.append((fn, split, name, la, lo))
            reasons["saved"] += 1
        except Exception as e:
            reasons["failed"] += 1
            problems.append((fn, f"{type(e).__name__}: {str(e)[:200]}"))
            tqdm.write(f"FAILED {fn}\n{traceback.format_exc(limit=2)}")

    with open(index_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "site", "lat", "lon"])
        w.writerows(rows)

    lines = [f"Tiles requested: {len(jobs)}", *(f"  {k}: {v}" for k, v in reasons.items()),
             f"Index rows written: {len(rows)} -> {index_path}", "", "Problems:"]
    lines += [f"  {fn}: {why}" for fn, why in problems] or ["  none"]
    report = "\n".join(lines)
    open(f"{args.out}/collect_log.txt", "w").write(report)
    print("\n" + report)
    if len(rows) < 0.5 * len(jobs):
        print("\nWARNING: fewer than half of the tiles could be collected. Read the 'Problems' list above "
              "(also saved in collect_log.txt) - it says why each tile was skipped.")


if __name__ == "__main__":
    main()