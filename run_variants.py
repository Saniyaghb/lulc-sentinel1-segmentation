"""
run_variants.py - trains ablation variants with train.py and collects their test results in one table
(outputs/variants.csv and variants.png): baseline, no temporal context, Lee speckle filter, ResNet18,
no ImageNet pretraining, and 60 m resolution (coarser pixels, so not directly comparable to the others).
Each variant is saved in models/variants/ and skipped if it already exists.
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

VARIANTS = {
    "baseline": [],
    "no_context": ["--no-context"],
    "speckle5": ["--speckle", "5"],
    "resnet18": ["--encoder", "resnet18"],
    "no_pretrained": ["--no-pretrained"],
    "res60m": ["--res-factor", "2"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out-dir", default="models/variants")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    names = a.only or list(VARIANTS)
    for n in names:
        ckpt = f"{a.out_dir}/{n}.pt"
        if os.path.exists(ckpt.replace(".pt", "_results.json")):
            print(f"[skip] {n} already trained")
            continue
        cmd = [sys.executable, "train.py", "--data", a.data, "--epochs", str(a.epochs),
               "--out", ckpt] + VARIANTS[n]
        print("\n=== ", " ".join(cmd))
        subprocess.run(cmd, check=True)

    rows = []
    for n in names:
        p = f"{a.out_dir}/{n}_results.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        t = j["test"]
        rows.append({"variant": n, "OA": t["overall_acc"], "mIoU": t["mIoU"], "mF1": t["mF1"],
                     "best_val_mIoU": j["best_val_mIoU"],
                     **{f"IoU_{k}": v for k, v in t["iou"].items()}})
    df = pd.DataFrame(rows).round(4)
    df.to_csv("outputs/variants.csv", index=False)
    print("\n", df.to_string(index=False))

    if len(df):
        fig, ax = plt.subplots(figsize=(9, 4))
        x = range(len(df))
        ax.bar([i - 0.2 for i in x], df.OA, 0.4, label="OA")
        ax.bar([i + 0.2 for i in x], df.mIoU, 0.4, label="mIoU")
        ax.set_xticks(list(x), df.variant, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title("Model variants on the test cities")
        ax.legend()
        fig.tight_layout()
        fig.savefig("outputs/variants.png", dpi=150)


if __name__ == "__main__":
    main()