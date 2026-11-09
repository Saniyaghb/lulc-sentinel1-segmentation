"""One table of every trained model (reads the *_results.json files that train.py wrote).

  python compare_runs.py                      # scans models/ recursively
  python compare_runs.py --models models --out outputs/model_comparison

Only runs that have a results file are listed, so nothing is shown that was not actually trained.
Runs with different class schemes (6 / 5 / 4 / 3 classes) are NOT directly comparable, so compare inside one scheme.
"""
import argparse
import glob
import json
import os

import pandas as pd


def load(models_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(models_dir, "**", "*_results.json"), recursive=True)):
        try:
            r = json.load(open(f))
            a = r.get("args", {})
            t = r.get("test", {})
            rows.append({
                "run": os.path.relpath(f, models_dir).replace("_results.json", ""),
                "scheme": r.get("scheme", a.get("scheme")),
                "arch": a.get("arch"), "encoder": a.get("encoder"), "features": a.get("features"),
                "speckle": a.get("speckle"), "lr": a.get("lr"), "dropout": a.get("dropout"),
                "epochs": len(r.get("history", [])),
                "val_OA": r.get("best_val_OA"), "val_mIoU": r.get("best_val_mIoU"),
                "test_OA": t.get("overall_acc"), "test_mIoU": t.get("mIoU"), "test_mF1": t.get("mF1"),
            })
        except Exception as e:
            print("skip", f, e)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="outputs/model_comparison")
    a = ap.parse_args()
    df = load(a.models)
    if df.empty:
        print("no *_results.json found in", a.models)
        return
    df["scheme"] = df["scheme"].astype(str)
    df = df.sort_values(["scheme", "val_mIoU"], ascending=[True, False])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out + ".csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(df.round(4).to_string(index=False))
    print("\nRanked by VALIDATION mIoU inside each class scheme (test scores are reported, not used to choose).")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sc = sorted(df["scheme"].unique())
        fig, axes = plt.subplots(1, len(sc), figsize=(5.5 * len(sc), 4.5), squeeze=False)
        for ax, s in zip(axes[0], sc):
            d = df[df["scheme"] == s].sort_values("val_mIoU")
            ax.barh(d["run"].str.slice(-28), d["val_mIoU"], color="#3b7dd8")
            ax.set_title(f"{s} classes: validation mIoU")
            ax.set_xlim(max(0, d["val_mIoU"].min() - 0.05), d["val_mIoU"].max() + 0.03)
            ax.tick_params(labelsize=7)
        plt.tight_layout()
        plt.savefig(a.out + ".png", dpi=150)
        print("saved", a.out + ".csv /.png")
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
