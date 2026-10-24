# HyperLithoMap
### Automated Lithological Mapping from Spaceborne Hyperspectral Imagery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why This Framework Matters

### For Geologists

Traditional geological field mapping is time-consuming, expensive, and limited in spatial coverage. A single field campaign may take months and cover only a few square kilometres. Airborne hyperspectral surveys improve on this, but remain costly and logistically demanding.

**Spaceborne hyperspectral sensors** (PRISMA, EnMAP, …) now deliver hundreds of contiguous spectral bands at 30 m ground resolution across entire regions in a single pass. Every rock type, alteration zone, and mineral assemblage has a unique **spectral fingerprint** — a pattern of absorption features across wavelengths 400–2500 nm — that can be detected from orbit. This framework automates the translation of that raw spectral data into **classified geological maps** at scale, giving geologists a first-pass lithological map before fieldwork begins, or enabling regional-scale mapping that would otherwise be impossible.

Key geological applications enabled by this framework:
- **Lithological discrimination** — separating igneous, metamorphic, and sedimentary units from spectral signatures
- **Hydrothermal alteration mapping** — locating clay minerals, iron oxides, and carbonates that signal mineralization pathways and porphyry/epithermal systems
- **Structural geology** — tracing contacts, faults, and fold axes through spectral contrasts
- **Mineral exploration targeting** — delineating alteration halos around ore bodies without drilling
- **Soil and regolith characterisation** — distinguishing pedogenic horizons and alluvial sediments

### For Data Scientists

Hyperspectral images are volumetric tensors — typically `(height × width × 200+ bands)` — where each pixel is a high-dimensional spectral vector. The classification challenge is harder than RGB or multispectral imagery because:

- **Extreme dimensionality**: 200–250 bands per pixel creates Hughes phenomenon (curse of dimensionality)
- **Spatial-spectral coupling**: rock type boundaries are determined jointly by spatial context (texture, structure) AND spectral content — neither alone is sufficient
- **Label scarcity**: geological ground truth is expensive to collect; training sets are small relative to image size
- **Class imbalance**: rare units (alteration zones, serpentinites) are spectrally important but spatially rare
- **Sensor artifacts**: atmospheric absorption, mixed pixels at boundaries, and band-to-band noise require careful preprocessing

This framework addresses all of these with a modular pipeline that handles sensor-specific preprocessing, band removal, nodata-aware normalization, stratified sampling, and a suite of models from classical ensemble methods to state-of-the-art spatial-spectral transformers and state space models.

---

## Supported Sensors and Formats

The reader is chosen by the dataset's `format` in `configs/datasets.yaml`, not by its name, so any sensor delivered in one of these formats works without code changes:

| `format` | Typical source | Notes |
|----------|---------------|-------|
| `prisma_he5` | PRISMA L2D HDF5 (`.he5`) | VNIR + SWIR cubes merged; georeferenced from the lat/lon geolocation fields. HDF5 dataset keys are configurable. |
| `geotiff` | EnMAP L2A `SPECTRAL_IMAGE.TIF`, any multi-band GeoTIFF | Single stacked cube. |
| `envi` | ENVI binary (`.dat`/`.img` + `.hdr`) — PRISMA/EnMAP exports, AVIRIS-NG, HyMap, DESIS … | Read natively through GDAL/rasterio. |

Bad bands (water-vapour absorption, noisy detector edges) are removed per dataset with 0-based band lists.

### Datasets shipped in the registry

| Dataset | Sensor | Format | Region |
|---------|--------|--------|--------|
| `PRISMA` | PRISMA | `prisma_he5` | Abu Rusheid (Aboresheed), 25 Jul 2020 |
| `PRISMA_Gahelia` | PRISMA | `prisma_he5` | Bir Gahelia, full scene, 2 Oct 2020 |
| `PRISMA_Gahelia_Clipped` | PRISMA | `prisma_he5` | Bir Gahelia, clipped training set |
| `EnMap` | EnMAP | `geotiff` | Abu Rusheid (Aboresheed), 29 Aug 2024 |
| `RabshiPRISMA` | PRISMA | `envi` | Rabshi |

The imagery and field shapefiles are **not** part of the repository (see [Data](#data)).

---

## Implemented Models

| Model | Type | Input | Reference |
|-------|------|-------|-----------|
| **Random Forest** | Classical | per-pixel spectrum | — |
| **XGBoost** | Classical | per-pixel spectrum | — |
| **S²Mamba** | Spatial-spectral state space model | 3×3 patch × all bands | Wang et al., IEEE TGRS 2025 |
| **SpectralFormer (ViT)** | Transformer with grouped band patches | 3×3 patch × 3-band groups | Hong et al., IEEE TGRS 2022 |

Random Forest and XGBoost run on CPU in seconds to minutes and are the recommended starting point for a new dataset. S²Mamba and SpectralFormer need an NVIDIA GPU (S²Mamba depends on the CUDA-only `mamba-ssm` kernels). Both deep models use class-weighted cross-entropy and keep the checkpoint with the best validation OA.

---

## Pipeline

```
scene (.he5 / .tif / .dat)                training shapefile
        │                                         │
  1. convert → stacked GeoTIFF (bad bands removed, nodata = -9999)
  1a. clip to shapefile extent
  1b. append N MNF components (noise-whitened PCA)
        │                                         │
  2. rasterise polygons → label raster  ◄─────────┘
  3. colourise labels (.clr table or default palette)
  4. stratified train/test masks (N pixels/class, test fraction)
  5. train + predict: Random Forest │ XGBoost │ S²Mamba │ SpectralFormer
  6. figures (map panels, confusion-matrix panel) + DOCX evaluation report
```

Class indices are assigned from the shapefile's class-name column in alphabetical order, with `Cloud` and `Background` placed last; the same mapping is reused for figures, legends and reports.

---

## Installation

Requirements: Python 3.10+ (tested with 3.12), GDAL-compatible `rasterio`/`geopandas`, and for the deep models an NVIDIA GPU with CUDA.

```bash
git clone https://github.com/nourlaban/HyperLithoMap.git
cd HyperLithoMap
python -m venv .venv && source .venv/bin/activate

# 1. PyTorch matching your CUDA version — see https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. Everything else (mamba-ssm compiles against the installed torch)
pip install -r requirements.txt

# 3. Check
python -c "import torch; print(torch.cuda.is_available())"
python -m pytest tests -q
```

Without a GPU you can still run preprocessing, Random Forest and XGBoost; leave `mamba-ssm` out of the install if it fails to build.

---

## Data

Each dataset lives in its own folder under the **data root**:

```
<data root>/
└── RabshimamabadataPrisma/          # `base_dir` in datasets.yaml
    ├── input/                       # you provide: scene, shapefile, optional .clr
    │   ├── PRISMA_Rabshi_clip.dat
    │   ├── PRISMA_Rabshi_clip.hdr
    │   ├── geology_classes.clr
    │   └── Rabshi_PRISMA2026/Rabshi_PRISMA2026.shp
    ├── intermediate/                # written by `preprocess`
    └── output/<experiment>/         # written by `train` / `run-all` / `report`
```

The data root is `./data` by default (ignored by git). Point it elsewhere with `--data-root /path` or `export HSDL_DATA_ROOT=/path`.

**Where to get imagery** — PRISMA: [prisma.asi.it](https://prisma.asi.it) (L2D, HDF5). EnMAP: [EOWEB GeoPortal](https://eoweb.dlr.de) (L2A, GeoTIFF).

**Training shapefile** — polygons of pure lithology with a text class-name column (`Classname` by default). Optional `RED`/`GREEN`/`BLUE` columns are used as figure colours when no `.clr` file is given. Any CRS works; polygons are reprojected to the scene.

**Colour table (`.clr`, optional)** — one line per class index: `index R G B alpha [label]`, where the index follows the alphabetical class order described above.

---

## Configuring a Dataset

Add an entry to [`configs/datasets.yaml`](configs/datasets.yaml):

```yaml
datasets:
  MyRegion_EnMAP:
    format: geotiff                  # prisma_he5 | geotiff | envi
    sensor: EnMAP
    description: My region — EnMAP L2A, 2025-05-01
    region_label: My Region          # used in figure/report titles
    base_dir: MyRegionEnMAP          # folder under the data root
    experiment: experiments_001      # default output folder
    scene: ENMAP01-____L2A-..._SPECTRAL_IMAGE.TIF
    shapefile: training/labels.shp   # relative to input/
    bands_to_remove: "range(131,136)"   # 0-based; list or "range(a,b) + range(c,d) + n"
    clr_file: geology_classes.clr    # optional
    # optional overrides of the `defaults:` block:
    # class_name_field: Classname
    # num_mnf_bands: 10
    # clip_to_shapefile: true
    # samples: 1000
    # test_size: 0.70
```

For a PRISMA `.he5` scene, merge the `*prisma_l2d` anchor (HDF5 keys + band lists) as the existing PRISMA entries do. To keep private dataset definitions outside the repo, copy the file and set `HSDL_DATASETS_FILE=/path/to/datasets.yaml`.

A sensor in a new file format needs one converter in `preprocessing/converth5totiff.py` plus a branch in `HyperspectralPipeline._step_convert_satellite_data` (`src/pipeline.py`).

---

## Usage

```bash
python main.py list-datasets                         # registry + whether each scene is found

python main.py preprocess --dataset RabshiPRISMA     # steps 1–4 only

python main.py train --dataset EnMap --skip-preprocessing \
       --models random_forests xgboost               # reuse preprocessed rasters

python main.py run-all --dataset PRISMA_Gahelia_Clipped \
       --experiment experiments_010 --verbose        # everything, into a new experiment folder

python main.py report --dataset RabshiPRISMA --experiment experiments_003   # figures + DOCX only
```

Common options: `--dataset` (alias `--satellite`), `--experiment NAME`, `--data-root PATH`, `--verbose`. Training commands also take `--models {random_forests,xgboost,mamba,spectral_former}` and `--epochs N` (overrides the deep models' default of 200, handy for quick checks).

### Python API

```python
from projutils.config import config
from src.pipeline import HyperspectralPipeline
from src.model_runner import ModelRunner, ModelType

config.reinit("RabshiPRISMA", experiment="experiments_004")   # shared config used by all modules

pipeline = HyperspectralPipeline(config)
pipeline.run_preprocessing()
classes = pipeline.get_class_info()          # {class name: raster index}

ModelRunner(config).run_multiple_models([ModelType.RANDOM_FORESTS, ModelType.XGBOOST], classes)
```

---

## Outputs

```
output/<experiment>/
├── rf_final_prediction_map.tif        # georeferenced class maps, same CRS as the scene
├── xgb_final_prediction_map.tif
├── MAMBA_final_prediction_map.tif
├── VIT_final_prediction_map.tif
├── new__newfinal_class_labels.tif     # rasterised training labels
├── Evaluation_Report_<Region>.docx    # data-driven report (run-all / report)
├── figures/                           # per-model map PNGs, comparison + confusion panels
└── saved_models/
    ├── rf_model.pkl, xgb_model.pkl, vit_best_model.pt, mamba_best_model_*.pt
    └── {rf,xgb,mamba,VITransformer}_model_evaluation/   # confusion_matrix.png, *evaluation_results.txt
```

### Interpreting the Accuracy Report

Each model writes `evaluation_results.txt` containing:

```
Accuracy:  0.8234          ← Overall Accuracy (OA): fraction of test pixels correct
IoU:       0.7118          ← Mean Intersection-over-Union (Jaccard): harsher than OA

Classification Report:
                     precision  recall  f1-score  support
Acidic Metavolcanics    0.84     0.79     0.82      421
Alteration zones        0.91     0.88     0.89      700
...
```

- **Precision**: of pixels predicted as class X, how many truly are X (purity)
- **Recall**: of true class-X pixels, how many were detected (completeness)
- **F1**: harmonic mean; use this as the primary per-class metric
- **Support**: number of test pixels for that class

For geological mapping, **recall is often more important than precision** — missing an alteration zone is more costly than misclassifying a small portion of one unit into an adjacent unit.

---

---

## Repository Layout

```
main.py                     CLI (list-datasets, preprocess, train, run-all, report)
configs/datasets.yaml       dataset registry
projutils/config.py         registry → Config (paths, band lists, sampling)
projutils/class_scheme.py   class names/colours for figures, aligned with raster indices
src/pipeline.py             preprocessing orchestrator
src/model_runner.py         model dispatch
preprocessing/              converters, clipping, MNF, rasterisation, masks, figures, report
hsmodel/                    RF, XGBoost, S²Mamba, SpectralFormer implementations
scripts/figures/<region>/   publication figure scripts per study area
scripts/reports/            paper/technical-report generators (DOCX)
scripts/tools/              timing benchmarks, DEM download, vector clipping utility
tests/                      synthetic end-to-end tests (no real data needed)
wiki/                       static HTML documentation
```

The scripts under `scripts/` reproduce the figures and reports of specific published experiments (e.g. Aboresheed `experiments_008`, Bir Gahelia `experiments_004`); they expect those experiment folders to exist under the data root. Run them from anywhere, e.g. `python scripts/figures/rabshi/plot_figure1_regional_map_rabshi.py`.

---

## Testing

```bash
python -m pytest tests -q
```

The tests build a tiny synthetic scene in each format (PRISMA `.he5`, ENVI, GeoTIFF) with a labelled shapefile and run preprocessing, all four models, the figure/report step and the CLI. The S²Mamba/SpectralFormer test is skipped automatically when no CUDA GPU or `mamba-ssm` is available.

---

## References

**S2Mamba**
> Wang, G., Zhang, X., Peng, Z., Zhang, T., & Jiao, L. (2025). S2Mamba: A Spatial-spectral State Space Model for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. DOI: 10.1109/TGRS.2025.3530993

**SpectralFormer**
> Hong, D., Han, Z., Yao, J., Gao, L., Zhang, B., & Plaza, A. (2022). SpectralFormer: Rethinking Hyperspectral Image Classification With Transformers. *IEEE Transactions on Geoscience and Remote Sensing*, 60, 1–20.

**PRISMA Mission**
> Cogliati, S. et al. (2021). The PRISMA imaging spectroscopy mission: overview and first performance analysis. *Remote Sensing of Environment*, 262, 112499.

**EnMap Mission**
> Guanter, L. et al. (2015). The EnMAP spaceborne imaging spectroscopy mission for earth observation. *Remote Sensing*, 7(7), 8830–8857.

---

---

## Citation

```bibtex
@software{laban_hyperlithomap_2026,
  author = {Laban, Nour},
  title  = {HyperLithoMap: Hyperspectral Lithological Mapping Framework},
  year   = {2026},
  url    = {https://github.com/nourlaban/HyperLithoMap}
}
```

## License

MIT — see [LICENSE](LICENSE).
