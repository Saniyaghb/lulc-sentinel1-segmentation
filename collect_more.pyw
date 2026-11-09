"""
collect_more.py - collects extra training tiles from additional cities (53 sites, 3 tiles each).

Same tile format as collect_data.py. The new tiles are added to the train split only, so the
validation and test sets stay unchanged. Tiles with almost constant radar or a single class
(sea, lake) are skipped. With --merge-into the tiles are copied into the main dataset and the
old index.csv is backed up as index_before_merge.csv.
"""
import argparse
import csv
import os
import random
import shutil
import traceback

import numpy as np
from tqdm import tqdm

from collect_data import TEST_SITES, TRAIN_SITES, center_crop, jitter
from config import RESOLUTION_M, TILE_PX, TRAIN_YEAR
from features import clean_s1
from geo_io import get_catalog, load_s1_series, load_worldcover

EXTRA_SITES = [
    ("milan", 45.46, 9.19), ("barcelona", 41.39, 2.17), ("lisbon", 38.72, -9.14), ("amsterdam", 52.37, 4.90),
    ("prague", 50.08, 14.44), ("budapest", 47.50, 19.04), ("bucharest", 44.43, 26.10), ("helsinki", 60.17, 24.94),
    ("oslo", 59.91, 10.75), ("dublin", 53.35, -6.26), ("moscow", 55.76, 37.62), ("kolkata", 22.57, 88.36),
    ("chennai", 13.08, 80.27), ("karachi", 24.86, 67.01), ("lahore", 31.55, 74.34), ("colombo", 6.93, 79.86),
    ("singapore", 1.35, 103.82), ("kualalumpur", 3.14, 101.69), ("taipei", 25.03, 121.57), ("osaka", 34.69, 135.50),
    ("guangzhou", 23.13, 113.26), ("wuhan", 30.59, 114.31), ("xian", 34.34, 108.94), ("hanoi", 21.03, 105.85),
    ("yangon", 16.84, 96.17), ("addisababa", 9.03, 38.74), ("kinshasa", -4.44, 15.27), ("dakar", 14.72, -17.47),
    ("casablanca", 33.57, -7.59), ("algiers", 36.75, 3.06), ("daressalaam", -6.79, 39.21), ("luanda", -8.84, 13.23),
    ("abuja", 9.08, 7.40), ("kampala", 0.35, 32.58), ("perth", -31.95, 115.86), ("auckland", -36.85, 174.76),
    ("vancouver", 49.28, -123.12), ("seattle", 47.61, -122.33), ("miami", 25.76, -80.19), ("atlanta", 33.75, -84.39),
    ("phoenix", 33.45, -112.07), ("dallas", 32.78, -96.80), ("sanfrancisco", 37.77, -122.42), ("montreal", 45.50, -73.57),
    ("havana", 23.11, -82.37), ("quito", -0.18, -78.47), ("rio", -22.91, -43.17), ("brasilia", -15.79, -47.88),
    ("medellin", 6.25, -75.57), ("dubai", 25.20, 55.27), ("baghdad", 33.31, 44.37), ("tashkent", 41.30, 69.24),
    ("almaty", 43.24, 76.89),
]


def merge_into(extra_dir, main_dir, rows):
    """Copy the new tiles into the main dataset and append them to its index.csv (backup first)."""
    main_index = f"{main_dir}/index.csv"
    backup = f"{main_dir}/index_before_merge.csv"
    if not os.path.exists(backup):
        shutil.copy(main_index, backup)
    existing = {r["file"] for r in csv.DictReader(open(main_index))}
    added = 0
    with open(main_index, "a", newline="") as f:
        w = csv.writer(f)
        for fn, split, name, la, lo in rows:
            if fn in existing:
                continue
            shutil.copy(f"{extra_dir}/tiles/{fn}", f"{main_dir}/tiles/{fn}")
            w.writerow([fn, split, name, la, lo])
            added += 1
    print(f"Merged {added} new tiles into {main_dir} (backup of the old index: {backup})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_extra")
    ap.add_argument("--tiles-per-site", type=int, default=3)
    ap.add_argument("--jitter-km", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-missing-months", type=int, default=2)
    ap.add_argument("--merge-into", default=None, help="main dataset folder (e.g. data) to add the new tiles to")
    args = ap.parse_args()

    used = {n for n, _, _ in TRAIN_SITES} | {n for n, _, _ in TEST_SITES}
    sites = [s for s in EXTRA_SITES if s[0] not in used]
    os.makedirs(f"{args.out}/tiles", exist_ok=True)
    rng = random.Random(args.seed)
    cat = get_catalog()
    months = [(TRAIN_YEAR, m) for m in range(1, 13)]
    size_km = (TILE_PX + 8) * RESOLUTION_M / 1000.0

    jobs = []
    for name, lat, lon in sites:
        for t in range(args.tiles_per_site):
            la, lo = (lat, lon) if t == 0 else jitter(lat, lon, args.jitter_km, rng)
            jobs.append((name, t, la, lo, "train"))

    rows, problems = [], []
    counts = {"saved": 0, "already_existed": 0, "skipped": 0}
    for name, t, la, lo, split in tqdm(jobs, desc="extra tiles"):
        fn = f"{name}_{t}.npz"
        path = f"{args.out}/tiles/{fn}"
        if os.path.exists(path):
            rows.append((fn, split, name, la, lo))
            counts["already_existed"] += 1
            continue
        try:
            s1, gb, missing = load_s1_series(cat, la, lo, size_km, months)
            if len(missing) > args.max_missing_months:
                raise ValueError(f"{len(missing)} months without Sentinel-1")
            label = load_worldcover(cat, gb, TRAIN_YEAR)
            if label is None:
                raise ValueError("no WorldCover")
            s1 = clean_s1(s1)
            if s1.shape[-1] < TILE_PX or s1.shape[-2] < TILE_PX:
                raise ValueError("tile too small")
            s1 = center_crop(s1, TILE_PX).astype(np.float16)
            label = center_crop(label, TILE_PX)
            if float(s1.astype(np.float32).std()) < 0.5:
                raise ValueError("radar almost constant")
            if len(np.unique(label[label < 255])) < 2:
                raise ValueError("label has a single class (sea/lake)")
            np.savez(path, s1=s1, label=label)
            rows.append((fn, split, name, la, lo))
            counts["saved"] += 1
        except Exception as e:
            counts["skipped"] += 1
            problems.append((fn, f"{type(e).__name__}: {str(e)[:150]}"))
            tqdm.write(f"skip {fn}: {e}")

    with open(f"{args.out}/index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "site", "lat", "lon"])
        w.writerows(rows)
    report = [f"Tiles requested: {len(jobs)}", *(f"  {k}: {v}" for k, v in counts.items()),
              f"Index rows: {len(rows)}", "", "Problems:"] + ([f"  {a}: {b}" for a, b in problems] or ["  none"])
    open(f"{args.out}/collect_log.txt", "w").write("\n".join(report))
    print("\n" + "\n".join(report))
    if args.merge_into:
        merge_into(args.out, args.merge_into, rows)


if __name__ == "__main__":
    main()