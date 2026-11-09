"""
eda.py - exploratory analysis of the collected dataset.

Reports tiles per split and city, class distribution per split, mean Sentinel-1 backscatter per
month, suspicious tiles (almost constant radar or a single class) and a few example tiles.
Results are printed and saved to outputs/eda.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from config import CLASSES, COLORS, NUM_CLASSES, SCHEME, remap_label
from metrics import class_counts

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/eda")
    ap.add_argument("--scheme", default=SCHEME, help="class scheme (see config.py)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(f"{a.data}/index.csv")))
    idx = pd.DataFrame(rows)
    print("== Tiles per split ==")
    print(idx.split.value_counts().to_string())
    print("\n== Cities per split ==")
    print(idx.groupby("split").site.nunique().to_string())
    per_city = idx.groupby(["split", "site"]).size().rename("tiles").reset_index()
    per_city.to_csv(f"{a.out}/tiles_per_city.csv", index=False)

    dist = {sp: np.zeros(NUM_CLASSES) for sp in ["train", "val", "test"]}
    vv_sum = np.zeros(12); vh_sum = np.zeros(12); vv_sq = np.zeros(12); vh_sq = np.zeros(12); n_px = 0
    suspicious = []
    lo_hi = {"VV": [np.inf, -np.inf], "VH": [np.inf, -np.inf]}
    for r in rows:
        d = np.load(f"{a.data}/tiles/{r['file']}")
        s1 = d["s1"].astype(np.float32)
        lab = remap_label(d["label"])
        dist[r["split"]] += class_counts(lab)
        vv, vh = s1[:, 0], s1[:, 1]
        vv_sum += vv.mean((1, 2)); vh_sum += vh.mean((1, 2))
        vv_sq += (vv ** 2).mean((1, 2)); vh_sq += (vh ** 2).mean((1, 2)); n_px += 1
        lo_hi["VV"] = [min(lo_hi["VV"][0], vv.min()), max(lo_hi["VV"][1], vv.max())]
        lo_hi["VH"] = [min(lo_hi["VH"][0], vh.min()), max(lo_hi["VH"][1], vh.max())]

        if s1.std() < 0.5:
            suspicious.append((r["file"], "radar almost constant (likely filled / empty data)"))
        elif len(np.unique(lab[lab < NUM_CLASSES])) == 1:
            suspicious.append((r["file"], "label contains a single class"))

    dd = pd.DataFrame({sp: 100 * v / max(v.sum(), 1) for sp, v in dist.items()}, index=CLASSES).T.round(2)
    print("\n== Class distribution (% of labelled pixels) ==")
    print(dd.to_string())
    dd.to_csv(f"{a.out}/class_distribution.csv")

    mean_vv, mean_vh = vv_sum / n_px, vh_sum / n_px
    std_vv = np.sqrt(np.maximum(vv_sq / n_px - mean_vv ** 2, 0)); std_vh = np.sqrt(np.maximum(vh_sq / n_px - mean_vh ** 2, 0))
    stats = pd.DataFrame({"VV_mean_dB": mean_vv, "VH_mean_dB": mean_vh}, index=MONTHS).round(2)
    print("\n== Mean backscatter per month (dB, over all tiles) ==")
    print(stats.to_string())
    print(f"\nValue range VV: {lo_hi['VV'][0]:.1f} .. {lo_hi['VV'][1]:.1f} dB | VH: {lo_hi['VH'][0]:.1f} .. {lo_hi['VH'][1]:.1f} dB")
    print("(typical land: VV about -20..0 dB, VH about -30..-5 dB; values pinned at the clip limits -40 / 10 mean filled or invalid data)")
    stats.to_csv(f"{a.out}/backscatter_per_month.csv")

    print(f"\n== Suspicious tiles: {len(suspicious)} ==")
    for f, why in suspicious:
        print(f"  {f}: {why}")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(NUM_CLASSES)
    for i, sp in enumerate(dd.index):
        ax.bar(x + (i - 1) * 0.27, dd.loc[sp].values, 0.27, label=sp)
    ax.set_xticks(x, CLASSES, rotation=20); ax.set_ylabel("% of labelled pixels"); ax.legend()
    ax.set_title("Class distribution per split")
    fig.tight_layout(); fig.savefig(f"{a.out}/class_distribution.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(MONTHS, mean_vv, marker="o", label="VV"); ax.plot(MONTHS, mean_vh, marker="o", label="VH")
    ax.set_ylabel("mean backscatter (dB)"); ax.set_title("Mean Sentinel-1 backscatter per month (all tiles)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{a.out}/backscatter_per_month.png", dpi=150)

    pick = idx.groupby("site").head(1).sample(n=min(6, idx.site.nunique()), random_state=0)
    fig, axes = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 6.4), squeeze=False)
    for j, (_, r) in enumerate(pick.iterrows()):
        d = np.load(f"{a.data}/tiles/{r['file']}")
        axes[0, j].imshow(d["s1"][6, 0].astype(np.float32), cmap="gray", vmin=-25, vmax=0)
        axes[0, j].set_title(f"{r['site']} VV (Jul)", fontsize=9)
        axes[1, j].imshow(remap_label(d["label"]), cmap=ListedColormap(COLORS), vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        axes[1, j].set_title("label", fontsize=9)
        axes[0, j].axis("off"); axes[1, j].axis("off")
    fig.tight_layout(); fig.savefig(f"{a.out}/eda_examples.png", dpi=150)
    print(f"\nSaved tables and figures to {a.out}/")


if __name__ == "__main__":
    main()