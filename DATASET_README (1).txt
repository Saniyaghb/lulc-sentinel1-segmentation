LULC from Sentinel-1: dataset description
==========================================

Task: semantic segmentation (land use / land cover) with Sentinel-1 radar as the ONLY input.

Contents
--------
tiles/*.npz      one file per tile (336 tiles in total)
index.csv        file, split, site, lat, lon  (split = train / val / test)
previews/*.png   quick-look pictures (July VV radar next to the label), only for viewing
index_before_merge.csv   backup of the index before the 143 extra training tiles were added

Split (by city, so test cities are never seen in training)
-----------------------------------------------------------
train 278 tiles | val 28 tiles | test 30 tiles
Test cities: hamburg, vienna, hyderabad, accra, chengdu, denver, santiago, brisbane.
The extra tiles added later (from 53 more cities) were put in the TRAIN split only.

Sources
-------
Input:  Sentinel-1 RTC (radiometric terrain corrected, VV and VH), ESA Copernicus data, accessed via the
        Microsoft Planetary Computer STAC API (no account or API key needed).
Labels: ESA WorldCover 2021 (CC BY 4.0), merged to 6 base classes and resampled to the radar grid (mode).
Period: radar year 2021, one median composite per calendar month (12 months), matching the label year.
Checks: Esri 10 m Annual Land Cover is used only as an independent check in the case study, never for training.

Tile format
-----------
Each .npz contains two arrays:
  s1     float16, shape (12, 2, 256, 256)   months x [VV, VH] x height x width, in dB, 30 m pixels
         (10 m radar area-averaged to 30 m; a tile covers 7.68 km x 7.68 km)
  label  uint8,   shape (256, 256)          base class id per pixel, 255 = no data (ignored)

Base classes stored in the tiles:
  0 Trees | 1 Grass/Shrub | 2 Cropland | 3 Built-up | 4 Bare/Snow | 5 Water
Classes used by the final model (3-class scheme, grouping classes that radar cannot separate):
  Vegetation & open land = base 0,1,2,4 | Built-up = base 3 | Water = base 5
(Other schemes with 6, 5 and 4 classes are defined in config.py.)

Open a tile
-----------
  import numpy as np
  d = np.load("tiles/vienna_0.npz")
  radar, label = d["s1"], d["label"]
