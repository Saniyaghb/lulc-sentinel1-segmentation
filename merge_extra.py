"""
merge_extra.py - adds tiles already collected by collect_more.py to the main dataset (train split only).
Tiles already listed in index.csv are skipped, and the old index is saved as index_before_merge.csv.
"""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="data_extra")
    ap.add_argument("--main", default="data")
    a = ap.parse_args()

    index = f"{a.main}/index.csv"
    backup = f"{a.main}/index_before_merge.csv"
    if not os.path.exists(backup):
        shutil.copy(index, backup)
    have = {r["file"] for r in csv.DictReader(open(index))}
    added = 0
    with open(index, "a", newline="") as f:
        w = csv.writer(f)
        for fn in sorted(os.listdir(f"{a.extra}/tiles")):
            if not fn.endswith(".npz") or fn in have:
                continue
            shutil.copy(f"{a.extra}/tiles/{fn}", f"{a.main}/tiles/{fn}")
            w.writerow([fn, "train", fn.rsplit("_", 1)[0], 0, 0])
            added += 1
    print(f"Added {added} tiles to {a.main} (old index backed up as {backup})")


if __name__ == "__main__":
    main()