"""
train.py - trains the segmentation network on Sentinel-1 inputs only.

Options cover architecture and encoder, feature set (7 or 19 channels), class scheme, speckle filter,
resolution factor, dropout on the decoder output and noise augmentation. Loss: class-weighted
cross-entropy + Dice, AdamW with a one-cycle schedule. The checkpoint with the best validation mIoU is
kept and evaluated once on the test cities; weights and a results JSON (history, test metrics, class
distribution) are saved.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import NUM_CLASSES, IGNORE_INDEX, MODEL_PATH, CLASSES, SCHEME, remap_label
from features import build_features, preprocess, downsample_label, N_CHANNELS
from metrics import confusion, summarize, class_counts
from models import build_model, ARCHS


class TileDS(Dataset):
    """Training: random month with random flip/rotation. Evaluation: every (tile, month) pair."""

    def __init__(self, root, files, train, use_context=True, speckle=0, factor=1, months=None,
                 features="base", samples_per_tile=4, aug=0.0):
        self.root, self.files, self.train = root, files, train
        self.use_context, self.speckle, self.factor = use_context, speckle, factor
        self.features, self.samples_per_tile = features, samples_per_tile
        self.aug = aug
        self.months = list(range(12)) if months is None else list(months)

    def __len__(self):
        return len(self.files) * (self.samples_per_tile if self.train else len(self.months))

    def __getitem__(self, i):
        n = len(self.files)
        f = self.files[i % n]
        d = np.load(f"{self.root}/tiles/{f}")
        s1 = preprocess(d["s1"].astype(np.float32), self.speckle, self.factor)
        y = remap_label(downsample_label(d["label"], self.factor)).astype(np.int64)
        m = np.random.randint(12) if self.train else self.months[i // n]
        x = build_features(s1, m, self.features)
        if not self.use_context:
            x[3:] = 0.0
        if self.train:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (0, 1))
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if self.aug > 0:
                x = np.ascontiguousarray(x, dtype=np.float32)
                x = x + (np.random.randn(*x.shape) * self.aug).astype(np.float32)
                x = x + np.float32(np.random.randn() * self.aug)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def read_split(root):
    sp = {"train": [], "val": [], "test": []}
    with open(f"{root}/index.csv") as f:
        for r in csv.DictReader(f):
            sp[r["split"]].append(r["file"])
    return sp


def label_distribution(root, files):
    cnt = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in files:
        cnt += class_counts(remap_label(np.load(f"{root}/tiles/{f}")["label"]))
    return cnt


def class_weights(cnt):
    freq = cnt / max(cnt.sum(), 1)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-4))
    return torch.tensor(w / w.mean(), dtype=torch.float32), freq


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loss_sum, n_valid = 0.0, 0
    for x, y in loader:
        out = model(x.to(device))
        yy = y.to(device)
        loss_sum += F.cross_entropy(out, yy, ignore_index=IGNORE_INDEX, reduction="sum").item()
        n_valid += int((yy != IGNORE_INDEX).sum().item())
        cm += confusion(out.argmax(1).cpu().numpy(), y.numpy())
    res = summarize(cm)
    res["loss"] = loss_sum / max(n_valid, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--arch", default="unet", choices=list(ARCHS), help="segmentation architecture")
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--features", default="base", choices=list(N_CHANNELS),
                    help="base = 7 channels, rich = 19 channels (min/max + seasonal means)")
    ap.add_argument("--scheme", default=SCHEME, help="class scheme: 6, 5, 4veg, 4open, 3 (see config.py)")
    ap.add_argument("--no-context", action="store_true", help="ablation: drop 12-month statistics channels")
    ap.add_argument("--speckle", type=int, default=0, help="Lee filter window (0 = off, e.g. 5)")
    ap.add_argument("--res-factor", type=int, default=1, help="2 = simulate 60 m by block-averaging the 30 m tiles")
    ap.add_argument("--samples-per-tile", type=int, default=4, help="random-month samples per tile per epoch")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout2d on the decoder output (0 = off, e.g. 0.2)")
    ap.add_argument("--aug", type=float, default=0.0, help="radar noise / brightness jitter strength (0 = off, e.g. 0.05)")
    ap.add_argument("--workers", type=int, default=0, help="0 avoids the shared-memory crash in Docker")
    ap.add_argument("--out", default=MODEL_PATH)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = read_split(args.data)
    in_ch = N_CHANNELS[args.features]
    print({k: len(v) for k, v in sp.items()}, "device:", device,
          f"| scheme {SCHEME} ({NUM_CLASSES} classes) | {args.arch}/{args.encoder} | features {args.features} ({in_ch} ch)")
    use_ctx = not args.no_context
    kw = dict(use_context=use_ctx, speckle=args.speckle, factor=args.res_factor, features=args.features)

    tr = DataLoader(TileDS(args.data, sp["train"], True, samples_per_tile=args.samples_per_tile, aug=args.aug, **kw),
                    batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True)
    va = DataLoader(TileDS(args.data, sp["val"], False, months=[0, 3, 6, 9], **kw),
                    batch_size=args.bs, num_workers=args.workers)
    te = DataLoader(TileDS(args.data, sp["test"], False, **kw),
                    batch_size=args.bs, num_workers=args.workers)

    model = build_model(args.arch, args.encoder, in_ch, NUM_CLASSES, pretrained=not args.no_pretrained).to(device)
    if args.dropout > 0:
        model.decoder.register_forward_hook(lambda m, i, o: F.dropout2d(o, args.dropout, training=m.training))

    dist = {k: label_distribution(args.data, v) for k, v in sp.items()}
    w, freq = class_weights(dist["train"])
    print("train class distribution:", {c: round(float(v), 3) for c, v in zip(CLASSES, freq)})
    ce = nn.CrossEntropyLoss(weight=w.to(device), ignore_index=IGNORE_INDEX)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE_INDEX)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    hist, best, best_ep, best_val_oa = [], -1.0, 0, 0.0
    for ep in range(args.epochs):
        model.train()
        t0, tot, correct, total = time.time(), 0.0, 0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                out = model(x)
                loss = ce(out, y) + dice(out, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            with torch.no_grad():
                valid = y != IGNORE_INDEX
                correct += ((out.argmax(1) == y) & valid).sum().item()
                total += valid.sum().item()
        train_acc = correct / max(total, 1)
        v = evaluate(model, va, device)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), "train_acc": train_acc,
                     "val_loss": v["loss"], "val_acc": v["overall_acc"], "val_mIoU": v["mIoU"],
                     "val_mF1": v["mF1"], "lr": opt.param_groups[0]["lr"]})
        print(f"ep {ep+1:02d} loss {tot/len(tr):.4f} train OA {train_acc:.4f} | val loss {v['loss']:.4f} "
              f"OA {v['overall_acc']:.4f} mIoU {v['mIoU']:.4f} ({time.time()-t0:.0f}s)")
        if v["mIoU"] > best:
            best, best_ep, best_val_oa = v["mIoU"], ep + 1, v["overall_acc"]
            torch.save({"state_dict": model.state_dict(), "arch": args.arch, "encoder": args.encoder,
                        "in_channels": in_ch, "classes": CLASSES, "scheme": SCHEME,
                        "features": args.features, "use_context": use_ctx, "speckle": args.speckle,
                        "factor": args.res_factor}, args.out)

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, te, device)
    print("TEST: OA %.4f  mIoU %.4f  mF1 %.4f" % (test["overall_acc"], test["mIoU"], test["mF1"]))
    print("TEST per-class IoU:", {k: (None if v is None else round(v, 3)) for k, v in test["iou"].items()})
    json.dump({"history": hist, "best_val_mIoU": best, "best_val_OA": best_val_oa, "best_epoch": best_ep,
               "test": test, "args": vars(args), "scheme": SCHEME,
               "class_distribution_pixels": {k: v.tolist() for k, v in dist.items()},
               "classes": CLASSES},
              open(args.out.replace(".pt", "_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()