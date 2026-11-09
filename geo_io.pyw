"""
geo_io.py - data access through the Microsoft Planetary Computer STAC API (no account or API key needed).

Monthly median Sentinel-1 RTC composites (VV, VH in dB, 30 m), ESA WorldCover labels remapped to
the project classes, and the Esri 10 m Annual Land Cover map used as the independent check.
"""
import calendar
import math
import time

import numpy as np

from config import RESOLUTION_M, WC_MAP, ESRI_MAP, IGNORE_INDEX

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def retry(fn, tries=6, wait=10):
    """Call fn(); on a temporary service error wait and try again."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  service error ({str(e)[:60]}), retry {k + 1}/{tries - 1} in {wait} s", flush=True)
            time.sleep(wait)


def get_catalog():
    import planetary_computer
    import pystac_client
    return retry(lambda: pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace))


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def bbox_around(lat: float, lon: float, size_km: float):
    """(west, south, east, north) in degrees for a square of side size_km."""
    dlat = size_km / 111.32 / 2
    dlon = size_km / (111.32 * math.cos(math.radians(lat))) / 2
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def month_range(year: int, month: int) -> str:
    last = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01/{year}-{month:02d}-{last}"


def to_db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(x, 1e-4, None))


def load_s1_month(cat, bbox, year, month, lat, lon, geobox=None, res=RESOLUTION_M):
    """Monthly median Sentinel-1 gamma0 (VV, VH) in dB; returns (array or None, geobox)."""
    import odc.stac
    items = retry(lambda: cat.search(collections=["sentinel-1-rtc"], bbox=bbox,
                                     datetime=month_range(year, month)).item_collection())
    if len(items) == 0:
        return None, geobox
    kw = dict(bands=["vv", "vh"], dtype="float32", resampling="average",
              groupby="solar_day", chunks={})
    if geobox is None:
        ds = odc.stac.load(items, bbox=bbox, crs=f"EPSG:{utm_epsg(lat, lon)}",
                           resolution=res, **kw)
    else:
        ds = odc.stac.load(items, geobox=geobox, **kw)
    ds = ds.where(ds > 0)
    vv = ds.vv.median("time", skipna=True).values
    vh = ds.vh.median("time", skipna=True).values
    gb = ds.odc.geobox
    arr = np.stack([to_db(vv), to_db(vh)]).astype(np.float32)
    arr[:, ~np.isfinite(vv) | ~np.isfinite(vh)] = np.nan
    return arr, gb


def load_s1_series(cat, lat, lon, size_km, months, res=RESOLUTION_M, verbose=False):
    """Stack of monthly composites for a list of (year, month); returns (stack, geobox, missing months)."""
    bbox = bbox_around(lat, lon, size_km)
    geobox, frames, missing = None, [], []
    for (y, m) in months:
        arr, geobox = load_s1_month(cat, bbox, y, m, lat, lon, geobox, res)
        if verbose:
            print(f"  S1 {y}-{m:02d}: {'ok' if arr is not None else 'NO DATA'}")
        frames.append(arr)
        if arr is None:
            missing.append((y, m))
    if all(f is None for f in frames):
        raise RuntimeError("No Sentinel-1 scenes found for this location/period.")
    shape = next(f.shape for f in frames if f is not None)

    for i, f in enumerate(frames):
        if f is None:
            j = min((k for k, g in enumerate(frames) if g is not None), key=lambda k: abs(k - i))
            frames[i] = frames[j].copy()
    return np.stack(frames), geobox, missing


def load_worldcover(cat, geobox, year=2021):
    """ESA WorldCover remapped to the project classes on the grid of geobox."""
    import odc.stac
    items = retry(lambda: cat.search(collections=["esa-worldcover"], bbox=geobox.geographic_extent.boundingbox.bbox,
                                     datetime=f"{year}-01-01/{year}-12-31").item_collection())
    if len(items) == 0:
        return None
    ds = odc.stac.load(items, bands=["map"], geobox=geobox, resampling="mode",
                       chunks={}, dtype="uint8")
    wc = ds["map"].max("time").values
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for k, v in WC_MAP.items():
        lut[k] = v
    return lut[wc]


def load_esri_lulc(cat, geobox, year):
    """Esri 10 m Annual Land Cover (independent map); tries `year`, then earlier years. Returns (map, year used)."""
    import odc.stac
    bbox = geobox.geographic_extent.boundingbox.bbox
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for k, v in ESRI_MAP.items():
        lut[k] = v
    for y in range(min(year, 2023), 2016, -1):
        items = retry(lambda: cat.search(collections=["io-lulc-annual-v02"], bbox=bbox,
                                         datetime=f"{y}-01-01/{y}-12-31").item_collection())
        if len(items) == 0:
            continue
        ds = odc.stac.load(items, bands=["data"], geobox=geobox, resampling="mode",
                           chunks={}, dtype="uint8")
        return lut[ds["data"].max("time").values], y
    return None, None