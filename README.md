# Sentinel-1 → Land Use / Land Cover (LULC): monthly maps from SAR only

Deep-learning semantic segmentation that uses **only Sentinel-1 radar** as input.
Prototype: latitude/longitude + start month → 12 monthly LULC maps, % area per class per month,
month-to-month change and total change (percentage points).

> **Problem statement.** Optical satellites cannot see through clouds; radar can, but SAR is noisy and hard to interpret.
> Can a deep-learning model that uses only low-resolution (30 m) Sentinel-1 data produce reliable monthly LULC maps and
> support measuring change over 12 months, and how much do classification errors and seasonal radar effects distort that change?

## Design
| Item | Choice |
|---|---|
| Input | Sentinel-1 RTC (VV, VH), terrain-corrected by the provider; monthly median composite; Microsoft Planetary Computer |
| Resolution | **30 m** (10 m native, area-averaged: low resolution and less speckle) |
| Speckle | averaging and monthly median; the optional Lee filter (`--speckle 5`) was tested as a variant and gave no gain, so it is not used |
| Labels | ESA WorldCover 2021, 6 base classes (Trees, Grass/Shrub, Cropland, Built-up, Bare/Snow, Water). **The final model uses 3 grouped classes:** Vegetation & open land, Built-up, Water (`--scheme 3`), because SAR cannot separate the vegetation and bare-soil classes |
| Model | U-Net, ResNet50 encoder (ImageNet-pretrained), 19 input channels (`--features rich`), dropout 0.2, noise augmentation |
| Channels | VV, VH, VV−VH of the target month + 12-month mean/std/min/max of VV and VH + four seasonal means |
| Split | By city: validation and test cities are never seen in training (278 train / 28 val / 30 test tiles) |
| Independent check | Esri 10 m Annual Land Cover (case study only, never used in training) |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
**Accounts / API keys: none.** Sentinel-1, WorldCover and Esri LULC are read anonymously from the Microsoft Planetary Computer.
Internet is required, also on the first training run (ImageNet encoder weights are downloaded).
A GPU is recommended for training; prediction runs on a CPU.

**Links:** dataset: `<ADD LINK>`  |  trained weights (`lulc_unet.pt`, place in `models/`): `<ADD LINK>`

## Run the prototype
```bash
python -W ignore predict.py --lat 48.21 --lon 16.37 --start 2023-01 --tta
python -W ignore case_study.py --name vienna --lat 48.21 --lon 16.37 --start 2023-01 --scheme 3 --model models/lulc_unet.pt
```
`predict.py` writes to `outputs/`: `*_maps.png` (12 monthly maps), `*_percent.csv` (class % per month),
`*_month_to_month_pp.csv` and `*_total_change_pp.csv`. `case_study.py` additionally compares the maps with the Esri map.

## Results (test cities never seen in training)
| Classes | Test OA | Test mIoU |
|---|---|---|
| 6 | 69.8 % | 0.543 |
| 5 | 76.4 % | 0.605 |
| 4 (vegetation merged) | 82.5 % | 0.657 |
| **3 (final model, with flip averaging)** | **88.0 %** | **0.776** |

Grouping the classes that SAR cannot separate (trees, grass, cropland, bare soil) is what raises the accuracy. Always state this when
quoting the 88 %; the 6-class result is reported as well. Per-city accuracy ranges from 73 % (Denver) to 96 % (Santiago).

## Model comparison
Every trained model is listed by `python compare_runs.py` (reads all `models/**/*_results.json`, writes `outputs/model_comparison.csv/.png`).
Models are chosen by validation score; test scores are only reported.

| What was compared | Result |
|---|---|
| Architecture (6 classes, 7 channels): U-Net, U-Net++, FPN, DeepLabV3+, MAnet with ResNet34/50 or EfficientNet-B3 encoders | all within about 2 points; **U-Net with ResNet50** was best on validation (val OA 71.5 %) |
| Class scheme (same model): 6, 5, 4, 3 classes | large effect, 69.8 % → 88.0 % |
| Features: 7 channels vs 19 channels (`rich`) | richer yearly statistics help |
| Tuning: learning rate, weight decay, dropout, augmentation | changed results by less than 1 point |
| Amount of data: 135 → 278 training tiles | about +0.5 points test OA |
| Lee speckle filter (`--speckle 5`), 3 classes | test OA 87.9 % vs 87.9 % without it: no gain, not used |

## Why there are extra data scripts
* **`collect_more.py`**: the first dataset had only 135 training tiles from about 45 cities. More cities help the model generalise
  to regions it has not seen. The script adds 143 tiles from 53 further cities. It is a separate script so that the original tiles
  and, above all, the validation and test cities stay exactly as they were: all new tiles go to the train split only, so results before
  and after are comparable. The old `index.csv` is backed up as `index_before_merge.csv`; `--merge-into data` copies the tiles into the main dataset.
* **`merge_extra.py`**: does the same merge for tiles already collected, for example when `collect_more.py` was stopped early.
  It skips tiles that are already in `index.csv` and can be run more than once.
* **`make_previews.py`**: the tiles are `.npz` arrays and cannot be opened as images. This script saves one PNG per tile
  (July VV radar next to its label) in `data/previews/` so the dataset can be viewed directly. It is not used for training.
* **`DATASET_README.txt`**: describes the contents and format of the dataset zip.

## Pipeline (commands used for the final model)
```bash
python collect_data.py --out data --tiles-per-site 4
python collect_more.py --out data_extra --tiles-per-site 3 --merge-into data
python make_previews.py
python eda.py --data data
python run_search.py --stage archs --epochs 60 --bs 32
python train.py --data data --epochs 60 --bs 32 --scheme 3 --arch unet --encoder resnet50 --features rich \
       --dropout 0.2 --aug 0.05 --weight-decay 1e-3 --samples-per-tile 8 --out models/lulc_unet.pt
python train.py ... --speckle 5 --out models/variants/speckle5.pt
python analysis.py --data data --model models/lulc_unet.pt --scheme 3 --tta
python compare_runs.py
python case_study.py --name vienna --lat 48.21 --lon 16.37 --start 2023-01 --scheme 3 --model models/lulc_unet.pt
```
A model must be evaluated and used with the class scheme it was trained with (`--scheme`, checked by the code).

## Files
| File | Purpose |
|---|---|
| `config.py` | classes, colours, resolution, label mappings, class schemes |
| `geo_io.py` | Planetary Computer access: Sentinel-1 monthly composites, WorldCover, Esri LULC |
| `features.py` | cleaning, Lee speckle filter, input channels |
| `collect_data.py`, `collect_more.py`, `merge_extra.py` | build and extend the dataset (tiles + `index.csv`) |
| `make_previews.py` | PNG preview of each tile |
| `eda.py` | dataset checks: class distribution, backscatter per month, suspicious tiles |
| `models.py` | architecture factory |
| `train.py` | training; saves the best-validation-mIoU weights and a results JSON |
| `run_search.py`, `run_variants.py` | train and rank many configurations / variants |
| `compare_runs.py` | one table and chart of all trained models |
| `metrics.py` | OA, mIoU, per-class IoU / F1 / precision / recall |
| `analysis.py` | test evaluation, per-city and per-month accuracy, class distribution, false-change experiment |
| `lulc_pipeline.py`, `predict.py` | the prototype (library and command-line tool) |
| `case_study.py` | prototype run for one location and comparison with the independent map |
| `lulc_sentinel1.ipynb` | notebook with the executed results and training curves |

## Where each report requirement is covered
| Report requirement | Where |
|---|---|
| Data source, regions, period, classes, class distribution | `collect_data.py` site lists; `DATASET_README.txt`; `outputs/analysis/class_distribution.*` |
| Calibration, terrain correction | done by the RTC product (provider) |
| dB conversion, speckle filtering, normalisation, patching | `geo_io.py`, `features.py` |
| Label matching, augmentation, splits | same 30 m grid in `geo_io.load_worldcover`; `train.TileDS`; city-based split in `collect_data.py` |
| Architecture and training setup | notebook section 2; `train.py` |
| OA, mIoU, per-class IoU/F1 on the test set | `outputs/analysis/test_metrics.json`, `per_class_metrics.csv` |
| Training curves | `outputs/analysis/training_curves.png`, notebook section 3 |
| Variant comparison | `outputs/model_comparison.csv/.png`, notebook section 4 |
| Prototype architecture, monthly compositing, class % and change | notebook section 5; `geo_io.py`, `lulc_pipeline.compute_stats` |
| Case study checked against an independent map | `outputs/case_<name>/independent_check.*` |
| Regions unlike training data | `outputs/analysis/per_site.csv/.png` |
| SAR-specific confusion | `confusion_matrix.png`, per-class F1 |
| Seasonal effects | `per_month.png`, `seasonal_bias.png` |
| Errors inflating the change percentages | `change_inflation.csv/.json/.png`, `total_change_vs_noise.csv` |

## Known limitations
- Labels are a single-year (2021) model-derived map, so "accuracy" is agreement with WorldCover. The monthly maps show how the model
  responds to SAR change, not independently verified monthly truth. The Esri map used in the case study is also model-derived.
- SAR confuses grass, crop and shrub; built-up versus trees and bare ground versus water are other typical errors. Built-up area is overestimated by about 3.5 points.
- Training uses 2021 only; other years and very different regions (for example Denver, 73 %) reduce accuracy.
- True change is zero in the reference labels, so `analysis.py` shows how much false change the pipeline produces (about 0.5 percentage points for 3 classes); real change should exceed that level to be trusted.
