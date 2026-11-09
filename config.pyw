"""
config.py - shared settings: 30 m resolution, 256 px tiles, label year, class definitions and colours.

The tiles store 6 base classes. A class scheme (6, 5, 4veg, 4open or 3) groups them for training and
evaluation; the final model uses scheme "3". It is chosen with --scheme or LULC_SCHEME.
The Esri map is used only as the independent check in the case study, never for training.
"""
import os
import sys

import numpy as np

RESOLUTION_M = 30
TILE_PX = 256
TRAIN_YEAR = 2021

BASE_CLASSES = ["Trees", "Grass/Shrub", "Cropland", "Built-up", "Bare/Snow", "Water"]
BASE_COLORS = ["#2e7d32", "#aed581", "#fbc02d", "#d32f2f", "#a1887f", "#1e88e5"]
IGNORE_INDEX = 255

SCHEMES = {
    "6":     [[0], [1], [2], [3], [4], [5]],
    "5":     [[0], [1, 2], [3], [4], [5]],
    "4veg":  [[0, 1, 2], [3], [4], [5]],
    "4open": [[0], [1, 2, 4], [3], [5]],
    "3":     [[0, 1, 2, 4], [3], [5]],
}
SCHEME_NAMES = {
    "6":     BASE_CLASSES,
    "5":     ["Trees", "Low vegetation (grass/shrub/crop)", "Built-up", "Bare/Snow", "Water"],
    "4veg":  ["Vegetation (trees/grass/crop)", "Built-up", "Bare/Snow", "Water"],
    "4open": ["Trees", "Open land (grass/crop/bare)", "Built-up", "Water"],
    "3":     ["Vegetation & open land", "Built-up", "Water"],
}
SCHEME_COLORS = {
    "6":     BASE_COLORS,
    "5":     ["#2e7d32", "#cddc39", "#d32f2f", "#a1887f", "#1e88e5"],
    "4veg":  ["#2e7d32", "#d32f2f", "#a1887f", "#1e88e5"],
    "4open": ["#2e7d32", "#cddc39", "#d32f2f", "#1e88e5"],
    "3":     ["#7cb342", "#d32f2f", "#1e88e5"],
}


def _read_scheme() -> str:
    s = os.environ.get("LULC_SCHEME")
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--scheme" and i + 1 < len(argv):
            s = argv[i + 1]
        elif a.startswith("--scheme="):
            s = a.split("=", 1)[1]
    s = s or "3"
    if s not in SCHEMES:
        raise SystemExit(f"Unknown class scheme '{s}'. Choose one of: {', '.join(SCHEMES)}")
    return s


SCHEME = _read_scheme()
CLASSES = SCHEME_NAMES[SCHEME]
COLORS = SCHEME_COLORS[SCHEME]
NUM_CLASSES = len(CLASSES)

LABEL_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _new, _group in enumerate(SCHEMES[SCHEME]):
    for _old in _group:
        LABEL_LUT[_old] = _new


def remap_label(y: np.ndarray) -> np.ndarray:
    """Base-class label array (values 0..5 or 255) -> labels of the active scheme."""
    return LABEL_LUT[np.asarray(y, dtype=np.uint8)]


WC_MAP = {
    10: 0,   # Tree cover
    20: 1,   # Shrubland
    30: 1,   # Grassland
    40: 2,   # Cropland
    50: 3,   # Built-up
    60: 4,   # Bare / sparse vegetation
    70: 4,   # Snow and ice
    80: 5,   # Permanent water
    90: 1,   # Herbaceous wetland
    95: 0,   # Mangroves
    100: 4,  # Moss and lichen
}

ESRI_MAP = {
    1: 5,    # Water
    2: 0,    # Trees
    4: 1,    # Flooded vegetation
    5: 2,    # Crops
    7: 3,    # Built area
    8: 4,    # Bare ground
    9: 4,    # Snow / ice
    10: IGNORE_INDEX,  # Clouds
    11: 1,   # Rangeland
}

IN_CHANNELS = 7
MODEL_PATH = "models/lulc_unet.pt"