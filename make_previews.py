"""
make_previews.py - saves one quick-look PNG per dataset tile (July VV radar next to its label)
in <data>/previews, so the tiles can be viewed without code.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.data, "tiles", "*.npz")))
print(f"found {len(files)} tiles in {os.path.abspath(a.data)}/tiles")
if not files:
    raise SystemExit("No tiles found. Run this from the project folder (where the 'data' folder is) or pass --data <folder>.")
out = os.path.join(a.data, "previews")
os.makedirs(out, exist_ok=True)
ok = bad = 0
for i, f in enumerate(files, 1):
    try:
        d = np.load(f)
        s = d["s1"].astype("float32")[6, 0]
        lab = d["label"].astype("float32")
        lab[lab == 255] = np.nan
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
        ax[0].imshow(s, cmap="gray"); ax[0].set_title("Sentinel-1 VV (July), dB", fontsize=8)
        ax[1].imshow(lab, cmap="tab10", vmin=0, vmax=9); ax[1].set_title("Label (WorldCover)", fontsize=8)
        for x in ax:
            x.axis("off")
        fig.suptitle(os.path.basename(f), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, os.path.basename(f).replace(".npz", ".png")), dpi=70)
        plt.close(fig)
        ok += 1
    except Exception as e:
        bad += 1
        print("skip", f, type(e).__name__, e)
    if i % 50 == 0:
        print(f"{i}/{len(files)}")
print(f"done: {ok} previews written to {os.path.abspath(out)}, {bad} skipped")