"""
case_study.py - runs the prototype for one location and compares the maps with an independent
land-cover product (Esri 10 m Annual Land Cover), which was not used for training.

Results go to outputs/case_<name>/: the prototype tables and 12-month map, agreement with the
Esri map, class share comparison, and the observed total change against the false-change level
measured in analysis.py. Esri is itself a model-derived map, so agreement shows consistency, not truth.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from config import CLASSES, COLORS, NUM_CLASSES, IGNORE_INDEX, SCHEME, remap_label
from geo_io import get_catalog, load_esri_lulc
from lulc_pipeline import run
from metrics import confusion, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--size-km", type=float, default=10.0)
    ap.add_argument("--model", default="models/lulc_unet.pt")
    ap.add_argument("--noise", default="outputs/analysis/change_inflation.csv")
    ap.add_argument("--scheme", default=SCHEME, help="class scheme the model was trained with")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (flip averaging)")
    a = ap.parse_args()

    out = f"outputs/case_{a.name}"
    os.makedirs(out, exist_ok=True)
    r = run(a.lat, a.lon, a.start, a.size_km, a.model, f"{out}/{a.name}", tta=a.tta)
    maps = r["maps"]
    print("\nClass coverage per month (%):\n", r["percent"])
    print("\nMonth-to-month change (pp):\n", r["month_to_month"])
    print("\nTotal change first->last (pp):\n", r["total_change"])


    ref, year = load_esri_lulc(get_catalog(), r["geobox"], int(a.start[:4]))
    if ref is None:
        raise SystemExit("No Esri land-cover tile found for this location - try another area.")
    ref = remap_label(ref)
    if ref.shape != maps.shape[1:]:
        raise SystemExit(f"Shape mismatch {ref.shape} vs {maps.shape[1:]} (model trained with res-factor>1?)")
    print(f"\nIndependent reference: Esri 10 m Annual Land Cover, year {year}")

    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_month = []
    for i, mp in enumerate(maps):
        c = confusion(mp, ref)
        cm_total += c
        s = summarize(c)
        per_month.append({"month": f"{r['months'][i][0]}-{r['months'][i][1]:02d}",
                          "OA": s["overall_acc"], "mIoU": s["mIoU"]})
    res = summarize(cm_total)
    res.update({"reference": "Esri 10m Annual Land Cover (io-lulc-annual-v02)", "reference_year": year,
                "per_month": per_month})
    json.dump(res, open(f"{out}/independent_check.json", "w"), indent=2)
    print(f"Agreement with independent map: OA {res['overall_acc']:.3f}  mIoU {res['mIoU']:.3f}  mF1 {res['mF1']:.3f}")
    print(pd.DataFrame({"IoU": res["iou"], "F1": res["f1"]}).round(3).to_string())


    cmap = ListedColormap(COLORS)
    valid = ref != IGNORE_INDEX
    agree = np.where(valid, (maps[0] == ref).astype(int), 2)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    ax[0].imshow(maps[0], cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
    ax[0].set_title(f"Model (Sentinel-1 only), {r['months'][0][0]}-{r['months'][0][1]:02d}")
    ax[1].imshow(np.where(valid, ref, 0), cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
    ax[1].set_title(f"Independent: Esri LULC {year}")
    ax[2].imshow(agree, cmap=ListedColormap(["#263238", "#cfd8dc", "#ffffff"]), vmin=0, vmax=2,
                 interpolation="nearest")
    ax[2].set_title("Agreement (light) / disagreement (dark)")
    for x in ax:
        x.axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, CLASSES)],
               loc="lower center", ncol=NUM_CLASSES, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(f"{out}/independent_check.png", dpi=150, bbox_inches="tight")


    ref_pct = np.bincount(ref[valid], minlength=NUM_CLASSES) / valid.sum() * 100
    mod_pct = r["percent"].mean(0).values
    area = pd.DataFrame({"model_12mo_mean_%": mod_pct, "esri_%": ref_pct}, index=CLASSES).round(2)
    area["difference_pp"] = (area.iloc[:, 0] - area.iloc[:, 1]).round(2)
    area.to_csv(f"{out}/area_comparison.csv")
    print("\nArea share, model vs independent map:\n", area.to_string())
    fig, axb = plt.subplots(figsize=(8, 4))
    xx = np.arange(NUM_CLASSES)
    axb.bar(xx - 0.2, area.iloc[:, 0], 0.4, label="Model (mean of 12 months)")
    axb.bar(xx + 0.2, area.iloc[:, 1], 0.4, label=f"Esri {year}")
    axb.set_xticks(xx, CLASSES, rotation=20)
    axb.set_ylabel("% of area")
    axb.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/area_comparison.png", dpi=150)


    if os.path.exists(a.noise):
        noise = pd.read_csv(a.noise, index_col=0)
        obs = r["total_change"].iloc[0]
        cmp_ = pd.DataFrame({"observed_total_change_pp": obs.values,
                             "false_change_p90_pp": noise["p90_abs_total_change_pp"].reindex(CLASSES).values},
                            index=CLASSES)
        cmp_["exceeds_noise_level"] = cmp_.observed_total_change_pp.abs() > cmp_.false_change_p90_pp
        cmp_.to_csv(f"{out}/total_change_vs_noise.csv")
        print("\nObserved change vs false-change level (from analysis.py):\n", cmp_.round(2).to_string())
    else:
        print(f"\n({a.noise} not found - run analysis.py first to compare against the false-change level)")


if __name__ == "__main__":
    main()