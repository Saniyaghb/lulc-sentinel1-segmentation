"""
run_search.py - trains a set of configurations with train.py (architectures, rich features, speckle filter,
learning rates, class schemes) and ranks them by validation mIoU. Test scores are only reported.

Each run is saved as models/search/<name>.pt with a results JSON; finished runs are skipped.
The overview is written to outputs/search.csv.
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

ARCH_RUNS = {
    "unet_r34": ["--arch", "unet", "--encoder", "resnet34"],
    "unet_r50": ["--arch", "unet", "--encoder", "resnet50"],
    "unetpp_r34": ["--arch", "unetplusplus", "--encoder", "resnet34"],
    "fpn_r34": ["--arch", "fpn", "--encoder", "resnet34"],
    "deeplabv3p_r50": ["--arch", "deeplabv3plus", "--encoder", "resnet50"],
    "manet_r34": ["--arch", "manet", "--encoder", "resnet34"],
    "unet_effb3": ["--arch", "unet", "--encoder", "efficientnet-b3"],
}
EXTRA_RUNS = {
    "rich_features": ["--features", "rich"],
    "speckle5": ["--speckle", "5"],
    "rich_speckle5": ["--features", "rich", "--speckle", "5"],
    "lr1e-3": ["--lr", "1e-3"],
    "lr1e-4": ["--lr", "1e-4"],
    "wd1e-3": ["--weight-decay", "1e-3"],
    "samples8": ["--samples-per-tile", "8"],
}
SCHEME_RUNS = ["6", "5", "4veg", "4open", "3"]


def run_one(name, args_list, a, scheme="6"):
    ckpt = f"{a.out_dir}/{name}.pt"
    if os.path.exists(ckpt.replace(".pt", "_results.json")):
        print(f"[skip] {name} already trained")
        return
    cmd = [sys.executable, "train.py", "--data", a.data, "--epochs", str(a.epochs), "--bs", str(a.bs),
           "--scheme", scheme, "--out", ckpt] + (["--no-pretrained"] if a.no_pretrained else []) + args_list
    print("\n=== ", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"!! {name} failed (exit code {r.returncode}) - continuing with the next run")


def collect(a):
    rows = []
    for fn in sorted(os.listdir(a.out_dir)):
        if not fn.endswith("_results.json"):
            continue
        j = json.load(open(f"{a.out_dir}/{fn}"))
        t, ar = j["test"], j["args"]
        rows.append({"run": fn.replace("_results.json", ""), "scheme": j.get("scheme", "6"),
                     "arch": ar.get("arch", "unet"), "encoder": ar["encoder"], "features": ar.get("features", "base"),
                     "speckle": ar["speckle"], "lr": ar["lr"],
                     "val_OA": round(j.get("best_val_OA", float("nan")), 4), "val_mIoU": round(j["best_val_mIoU"], 4),
                     "best_epoch": j.get("best_epoch"), "test_OA": round(t["overall_acc"], 4),
                     "test_mIoU": round(t["mIoU"], 4), "test_mF1": round(t["mF1"], 4)})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["scheme", "val_mIoU"], ascending=[True, False])
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/search.csv", index=False)
    print("\nAll runs (sorted by validation mIoU within each class scheme; choose by VALIDATION):\n")
    print(df.to_string(index=False))
    for sch, g in df.groupby("scheme"):
        b = g.iloc[0]
        print(f"\nBest for scheme '{sch}' by validation mIoU: {b.run}  (val mIoU {b.val_mIoU}, "
              f"test OA {b.test_OA}, test mIoU {b.test_mIoU})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["archs", "extras", "schemes", "all"], default="archs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--arch", default="unet", help="base architecture for the extras/schemes stages")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--features", default="base")
    ap.add_argument("--out-dir", default="models/search")
    ap.add_argument("--only", nargs="*", default=None, help="run only these run names")
    ap.add_argument("--no-pretrained", action="store_true", help="random initialisation (e.g. no internet)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    base = ["--arch", a.arch, "--encoder", a.encoder, "--features", a.features]
    if a.stage in ("archs", "all"):
        for n, extra in ARCH_RUNS.items():
            if not a.only or n in a.only:
                run_one(n, extra, a)
    if a.stage in ("extras", "all"):
        for n, extra in EXTRA_RUNS.items():
            if not a.only or n in a.only:
                run_one(f"x_{a.arch}_{a.encoder}_{n}", base + extra, a)
    if a.stage in ("schemes", "all"):
        for sch in SCHEME_RUNS:
            n = f"scheme{sch}_{a.arch}_{a.encoder}_{a.features}"
            if not a.only or n in a.only:
                run_one(n, base, a, scheme=sch)
    collect(a)


if __name__ == "__main__":
    main()