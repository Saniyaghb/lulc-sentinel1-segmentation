"""
features.py - Sentinel-1 preprocessing and model input features.

Cleaning (NaN fill, dB clipping), optional Lee speckle filter in linear power (--speckle),
optional block averaging, and the input channels for one target month: the month's VV, VH and
VV-VH plus 12-month statistics ("base", 7 channels), or with 12-month min/max and seasonal
means added ("rich", 19 channels, used by the final model).
"""
import numpy as np
from scipy.ndimage import uniform_filter


def clean_s1(s1: np.ndarray) -> np.ndarray:
    """Replace NaN / inf with the per-band median and clip to -40..10 dB."""
    s1 = s1.astype(np.float32).copy()
    for b in range(2):
        band = s1[:, b]
        bad = ~np.isfinite(band)
        if bad.any():
            fill = np.nanmedian(np.where(bad, np.nan, band)) if (~bad).any() else -15.0
            band[bad] = fill
            s1[:, b] = band
    return np.clip(s1, -40.0, 10.0)


def lee_filter(db: np.ndarray, size: int = 5, enl: float = 5.0) -> np.ndarray:
    """Lee speckle filter on linear power (window `size`, ENL `enl`)."""
    lin = np.power(10.0, db / 10.0).astype(np.float32)
    win = (1,) * (lin.ndim - 2) + (size, size)
    mean = uniform_filter(lin, win, mode="reflect")
    sq = uniform_filter(lin * lin, win, mode="reflect")
    var = np.maximum(sq - mean * mean, 0.0)
    noise = (mean * mean) / enl
    w = var / (var + noise + 1e-12)
    out = mean + w * (lin - mean)
    return (10.0 * np.log10(np.clip(out, 1e-4, None))).astype(np.float32)


def block_mean_db(db: np.ndarray, f: int) -> np.ndarray:
    """Average f x f pixel blocks in linear power."""
    if f <= 1:
        return db
    H, W = db.shape[-2:]
    H2, W2 = H // f * f, W // f * f
    lin = np.power(10.0, db[..., :H2, :W2] / 10.0)
    lin = lin.reshape(*lin.shape[:-2], H2 // f, f, W2 // f, f).mean(axis=(-3, -1))
    return (10.0 * np.log10(np.clip(lin, 1e-4, None))).astype(np.float32)


def downsample_label(y: np.ndarray, f: int) -> np.ndarray:
    """Label reduction matching block_mean_db (centre pixel)."""
    if f <= 1:
        return y
    H, W = y.shape[-2:]
    H2, W2 = H // f * f, W // f * f
    return y[..., f // 2:H2:f, f // 2:W2:f]


def preprocess(s1: np.ndarray, speckle: int = 0, factor: int = 1) -> np.ndarray:
    """Cleaning, optional Lee filter and optional block averaging of a (12, 2, H, W) dB stack."""
    s1 = clean_s1(s1)
    if speckle and speckle > 1:
        s1 = lee_filter(s1, size=int(speckle))
    if factor and factor > 1:
        s1 = block_mean_db(s1, int(factor))
    return np.clip(s1, -40.0, 10.0)


N_CHANNELS = {"base": 7, "rich": 19}


def build_features(s1: np.ndarray, m: int, kind: str = "base") -> np.ndarray:
    """Feature stack (C, H, W) for month index m: "base" = 7 channels, "rich" = 19 channels."""
    vv, vh = s1[m, 0], s1[m, 1]
    mean = s1.mean(0)
    std = s1.std(0)
    chans = [
        (vv + 15.0) / 10.0,
        (vh + 22.0) / 10.0,
        (vv - vh - 7.0) / 4.0,
        (mean[0] + 15.0) / 10.0,
        (mean[1] + 22.0) / 10.0,
        std[0] / 2.0,
        std[1] / 2.0,
    ]
    if kind == "rich":
        mn, mx = s1.min(0), s1.max(0)
        chans += [(mn[0] + 15.0) / 10.0, (mn[1] + 22.0) / 10.0,
                  (mx[0] + 15.0) / 10.0, (mx[1] + 22.0) / 10.0]
        for q in range(4):
            qm = s1[3 * q:3 * q + 3].mean(0)
            chans += [(qm[0] + 15.0) / 10.0, (qm[1] + 22.0) / 10.0]
    elif kind != "base":
        raise ValueError(f"unknown feature kind '{kind}' (use 'base' or 'rich')")
    return np.stack(chans).astype(np.float32)