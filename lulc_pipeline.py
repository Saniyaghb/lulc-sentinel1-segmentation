"""
lulc_pipeline.py - the prototype: latitude/longitude and a start month give 12 monthly land cover maps,
the class percentages per month, and the month-to-month and total change in percentage points.
Used by predict.py, case_study.py and analysis.py.
"""
import calendar

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import CLASSES, COLORS, NUM_CLASSES, MODEL_PATH, SCHEME
from features import preprocess, build_features
from geo_io import get_catalog, load_s1_series
from models import build_model


def month_list(start: str, n: int = 12):
    """List of n (year, month) pairs starting at 'YYYY-MM'."""
    y, m = map(int, start.split("-"))
    out = []
    for _ in range(n):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def load_model(path=MODEL_PATH, device=None):
    """Returns (model, device, meta); meta holds the preprocessing used in training."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    trained_scheme = ckpt.get("scheme", "6")
    if trained_scheme != SCHEME:
        raise SystemExit(f"This model was trained with class scheme '{trained_scheme}' but the active scheme is "
                         f"'{SCHEME}'. Add  --scheme {trained_scheme}  (or set LULC_SCHEME={trained_scheme}).")
    model = build_model(ckpt.get("arch", "unet"), ckpt["encoder"], ckpt["in_channels"], NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    meta = {"use_context": ckpt.get("use_context", True),
            "speckle": ckpt.get("speckle", 0),
            "factor": ckpt.get("factor", 1),
            "features": ckpt.get("features", "base")}
    return model.to(device).eval(), device, meta


def _probs(model, x, tta):
    """Class probabilities, averaged over 4 flips when tta is on."""
    if not tta:
        return model(x).softmax(1)
    outs = []
    for dims in [(), (3,), (2,), (2, 3)]:
        xi = torch.flip(x, dims) if dims else x
        o = model(xi).softmax(1)
        outs.append(torch.flip(o, dims) if dims else o)
    return torch.stack(outs).mean(0)


@torch.no_grad()
def predict_maps(model, device, s1, meta=None, tta=False):
    """Class map for each of the 12 months of a (12, 2, H, W) dB stack."""
    meta = meta or {}
    s1 = preprocess(s1, meta.get("speckle", 0), meta.get("factor", 1))
    T, _, H, W = s1.shape
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    maps = []
    for m in range(T):
        x = build_features(s1, m, meta.get("features", "base"))
        if not meta.get("use_context", True):
            x[3:] = 0.0
        x = torch.from_numpy(x)[None].to(device)
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        pred = _probs(model, x, tta).argmax(1)[0, :H, :W].cpu().numpy().astype(np.uint8)
        maps.append(pred)
    return np.stack(maps)


def percent_table(maps, months):
    labels = [f"{y}-{m:02d}" for y, m in months]
    total = maps[0].size
    pct = np.stack([np.bincount(mp.ravel(), minlength=NUM_CLASSES) / total * 100 for mp in maps])
    df = pd.DataFrame(pct, index=labels, columns=CLASSES)
    df.index.name = "month"
    return df


def compute_stats(maps, months):
    """Class percentages per month, month-to-month change and first-to-last change (percentage points)."""
    labels = [f"{y}-{m:02d}" for y, m in months]
    pct_df = percent_table(maps, months).round(2)
    m2m = pct_df.diff().iloc[1:]
    m2m.index = [f"{labels[i-1]} -> {labels[i]}" for i in range(1, len(labels))]
    m2m = m2m.round(2)
    total_change = (pct_df.iloc[-1] - pct_df.iloc[0]).round(2).to_frame(
        f"{labels[0]} -> {labels[-1]} (pp)").T
    return pct_df, m2m, total_change


def plot_maps(maps, months, title="", save_path=None):
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, mp, (y, m) in zip(axes.ravel(), maps, months):
        ax.imshow(mp, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        ax.set_title(f"{calendar.month_abbr[m]} {y}")
        ax.axis("off")
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, CLASSES)],
               loc="lower center", ncol=NUM_CLASSES, frameon=False)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def run(lat, lon, start, size_km=10.0, model_path=MODEL_PATH, out_prefix=None, verbose=True, tta=False):
    months = month_list(start)
    model, device, meta = load_model(model_path)
    cat = get_catalog()
    s1, gb, missing = load_s1_series(cat, lat, lon, size_km, months, verbose=verbose)
    maps = predict_maps(model, device, s1, meta, tta=tta)
    pct, m2m, tot = compute_stats(maps, months)
    fig = plot_maps(maps, months, f"LULC from Sentinel-1 | {lat:.4f}, {lon:.4f} | 30 m",
                    save_path=f"{out_prefix}_maps.png" if out_prefix else None)
    if out_prefix:
        pct.to_csv(f"{out_prefix}_percent.csv")
        m2m.to_csv(f"{out_prefix}_month_to_month_pp.csv")
        tot.to_csv(f"{out_prefix}_total_change_pp.csv")
        np.save(f"{out_prefix}_maps.npy", maps)
    return dict(maps=maps, percent=pct, month_to_month=m2m, total_change=tot,
                fig=fig, missing_months=missing, months=months, geobox=gb, s1=s1)