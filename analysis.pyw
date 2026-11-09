"""
analysis.py - evaluation of the trained model on the held-out test cities.

Everything is saved in the output folder (default outputs/analysis):
  test_metrics.json         overall accuracy, mIoU, mean F1, per-class IoU / F1 / precision / recall, confusion matrix
  confusion_matrix.png      the same confusion matrix as a picture
  per_site.csv / .png       accuracy for each test city (how well it works on unseen regions)
  per_month.csv / .png      accuracy for each calendar month (seasonal effects)
  class_distribution.csv / .png   share of each class in train, validation and test
  change_inflation.*        the labels are one fixed map, so the real change over 12 months is 0.
                            Any change the model shows between months is therefore error or seasonal
                            radar effect. These files measure how big that false change is.
  seasonal_bias.png         predicted minus real class share for each month
  training_curves.png       loss and accuracy per epoch (taken from the training results file)
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CLASSES, COLORS, NUM_CLASSES, IGNORE_INDEX, SCHEME, remap_label
from features import downsample_label
from lulc_pipeline import load_model, predict_maps
from metrics import confusion, summarize, class_counts

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def read_index(root, split=None):
    with open(f"{root}/index.csv") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def plot_confusion(cm, path):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference (WorldCover)")
    ax.set_title("Test confusion matrix (row-normalised)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_training_curves(results_json, path):
    if not os.path.exists(results_json):
        return None
    h = pd.DataFrame(json.load(open(results_json))["history"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(h.epoch, h.train_loss, label="train (CE+Dice)")
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(h.epoch, h.val_loss, color="C1")
    ax[1].set_title("Validation cross-entropy")
    ax[1].set_xlabel("epoch")
    ax[2].plot(h.epoch, h.val_acc, label="val OA")
    ax[2].plot(h.epoch, h.val_mIoU, label="val mIoU")
    ax[2].set_title("Validation accuracy")
    ax[2].set_xlabel("epoch")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_class_distribution(df, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(NUM_CLASSES)
    for i, split in enumerate(df.index):
        ax.bar(x + (i - 1) * 0.27, df.loc[split].values, 0.27, label=split)
    ax.set_xticks(x, CLASSES, rotation=20)
    ax.set_ylabel("% of labelled pixels")
    ax.set_title("Class distribution (ESA WorldCover, grouped by class scheme)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def analyse(data, model_path, out, split="test", max_tiles=None, tta=False):
    os.makedirs(out, exist_ok=True)
    model, device, meta = load_model(model_path)
    rows = read_index(data, split)[:max_tiles]
    if not rows:
        raise SystemExit(f"No '{split}' tiles in {data}/index.csv")

    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_site = defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    cm_month = np.zeros((12, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    n_tiles_site = defaultdict(int)
    per_tile, seasonal_dev, flips, nets = [], [], [], []

    for r in tqdm(rows, desc=f"predicting {split} tiles"):
        d = np.load(f"{data}/tiles/{r['file']}")
        y = remap_label(downsample_label(d["label"], meta["factor"])).astype(np.int64)
        maps = predict_maps(model, device, d["s1"].astype(np.float32), meta, tta=tta)
        valid = (y != IGNORE_INDEX) & (y < NUM_CLASSES)
        if valid.sum() == 0:
            continue
        for m in range(12):
            c = confusion(maps[m], y)
            cm_total += c
            cm_site[r["site"]] += c
            cm_month[m] += c
        n_tiles_site[r["site"]] += 1

        # --- change-inflation bookkeeping (true change is zero: labels are one static map)
        nv = valid.sum()
        pred_pct = np.stack([np.bincount(maps[m][valid], minlength=NUM_CLASSES) / nv * 100
                             for m in range(12)])
        true_pct = np.bincount(y[valid], minlength=NUM_CLASSES) / nv * 100
        dev = pred_pct - true_pct
        seasonal_dev.append(dev)
        m2m = np.abs(np.diff(pred_pct, axis=0))
        per_tile.append({
            "bias_pp": dev.mean(0), "abs_bias_pp": np.abs(dev.mean(0)),
            "range_pp": pred_pct.max(0) - pred_pct.min(0),
            "abs_m2m_pp": m2m.mean(0),
            "abs_total_pp": np.abs(pred_pct[-1] - pred_pct[0]),
        })
        flips.append(np.mean([(maps[m + 1] != maps[m])[valid].mean() * 100 for m in range(11)]))
        nets.append(np.mean([0.5 * np.abs(pred_pct[m + 1] - pred_pct[m]).sum() for m in range(11)]))


    test = summarize(cm_total)
    test.update({"split": split, "n_tiles": len(rows), "months_evaluated": 12, "model": model_path,
                 "preprocessing": meta, "scheme": SCHEME, "tta": tta})
    json.dump(test, open(f"{out}/test_metrics.json", "w"), indent=2)
    plot_confusion(cm_total, f"{out}/confusion_matrix.png")


    pc = pd.DataFrame({"IoU": test["iou"], "F1": test["f1"], "precision": test["precision"],
                       "recall": test["recall"], "support_pixels": test["support_pixels"]})
    pc.to_csv(f"{out}/per_class_metrics.csv")


    recs = []
    for site, cm in cm_site.items():
        s = summarize(cm)
        sup = np.asarray(list(s["support_pixels"].values()), dtype=float)
        recs.append({"site": site, "tiles": n_tiles_site[site], "OA": s["overall_acc"],
                     "mIoU": s["mIoU"], "mF1": s["mF1"],
                     "dominant_class": CLASSES[int(sup.argmax())],
                     **{f"%{c}": round(100 * v / sup.sum(), 1) for c, v in zip(CLASSES, sup)}})
    site_df = pd.DataFrame(recs).sort_values("OA")
    site_df.to_csv(f"{out}/per_site.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(site_df))))
    ax.barh(site_df.site, site_df.OA, color="C0", label="OA")
    ax.barh(site_df.site, site_df.mIoU, color="C1", height=0.4, label="mIoU")
    ax.set_xlim(0, 1)
    ax.set_title("Accuracy on each held-out city")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/per_site.png", dpi=150)


    mrecs = []
    for m in range(12):
        s = summarize(cm_month[m])
        mrecs.append({"month": MONTHS[m], "OA": s["overall_acc"], "mIoU": s["mIoU"]})
    month_df = pd.DataFrame(mrecs)
    month_df.to_csv(f"{out}/per_month.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(month_df.month, month_df.OA, marker="o", label="OA")
    ax.plot(month_df.month, month_df.mIoU, marker="o", label="mIoU")
    ax.set_title("Test accuracy by calendar month")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/per_month.png", dpi=150)


    dist = {}
    for sp in ["train", "val", "test"]:
        cnt = np.zeros(NUM_CLASSES)
        for r in read_index(data, sp):
            cnt += class_counts(remap_label(np.load(f"{data}/tiles/{r['file']}")["label"]))
        dist[sp] = 100 * cnt / max(cnt.sum(), 1)
    dist_df = pd.DataFrame(dist, index=CLASSES).T.round(2)
    dist_df.to_csv(f"{out}/class_distribution.csv")
    plot_class_distribution(dist_df, f"{out}/class_distribution.png")


    agg = {k: np.mean([t[k] for t in per_tile], axis=0) for k in per_tile[0]}
    p90 = np.percentile([t["abs_total_pp"] for t in per_tile], 90, axis=0)
    ci = pd.DataFrame({"mean_bias_pp": agg["bias_pp"], "mean_abs_bias_pp": agg["abs_bias_pp"],
                       "mean_range_over_12_months_pp": agg["range_pp"],
                       "mean_abs_month_to_month_pp": agg["abs_m2m_pp"],
                       "mean_abs_total_change_pp": agg["abs_total_pp"],
                       "p90_abs_total_change_pp": p90}, index=CLASSES).round(3)
    ci.index.name = "class"
    ci.to_csv(f"{out}/change_inflation.csv")
    summary = {"true_change_pp": 0.0,
               "mean_pixels_switching_class_per_month_pct": float(np.mean(flips)),
               "mean_net_area_change_per_month_pct": float(np.mean(nets)),
               "note": "True change is zero (static reference map). Gross pixel switching is far larger "
                       "than net area change because switches partly cancel; both are error, not change."}
    json.dump(summary, open(f"{out}/change_inflation_summary.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(NUM_CLASSES)
    for i, col in enumerate(["mean_abs_month_to_month_pp", "mean_range_over_12_months_pp",
                             "mean_abs_total_change_pp"]):
        ax.bar(x + (i - 1) * 0.27, ci[col].values, 0.27, label=col.replace("_", " "))
    ax.set_xticks(x, CLASSES, rotation=20)
    ax.set_ylabel("percentage points (true value = 0)")
    ax.set_title("Apparent change where no real change exists (test tiles)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/change_inflation.png", dpi=150)

    sd = np.mean(seasonal_dev, axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    for k in range(NUM_CLASSES):
        ax.plot(MONTHS, sd[:, k], marker="o", color=COLORS[k], label=CLASSES[k])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("predicted % - reference % (pp)")
    ax.set_title("Seasonal bias of predicted class area (mean over test tiles)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/seasonal_bias.png", dpi=150)

    plot_training_curves(model_path.replace(".pt", "_results.json"), f"{out}/training_curves.png")

    print(f"\n{split.upper()} OA {test['overall_acc']:.4f} | mIoU {test['mIoU']:.4f} | mF1 {test['mF1']:.4f}")
    print(pc.round(3).to_string())
    print("\nPer-site:\n", site_df[["site", "OA", "mIoU", "dominant_class"]].round(3).to_string(index=False))
    print("\nApparent change with zero true change (pp):\n", ci.to_string())
    print("\n", json.dumps(summary, indent=2))
    return dict(test=test, per_class=pc, per_site=site_df, per_month=month_df,
                class_distribution=dist_df, change_inflation=ci, change_summary=summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="models/lulc_unet.pt")
    ap.add_argument("--out", default="outputs/analysis")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--scheme", default=SCHEME, help="class scheme the model was trained with")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (flip averaging)")
    a = ap.parse_args()
    analyse(a.data, a.model, a.out, a.split, a.max_tiles, a.tta)