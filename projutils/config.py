"""
Dataset-driven configuration.

Datasets are declared in ``configs/datasets.yaml`` (or the file named by the
``HSDL_DATASETS_FILE`` environment variable). ``Config`` turns one entry into
the flat set of absolute paths and parameters used by the pipeline, models and
figure scripts.

A module-level singleton ``config`` is exported for the model/figure modules
that read paths at call time; ``main.py`` initialises it with ``reinit``.
"""

import os
import re
from typing import List, Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REGISTRY = os.path.join(REPO_ROOT, 'configs', 'datasets.yaml')

SUPPORTED_FORMATS = ('prisma_he5', 'envi', 'geotiff')

_RANGE_RE = re.compile(r'range\(\s*(\d+)\s*,\s*(\d+)\s*\)')


def _parse_band_list(value) -> List[int]:
    """Accept a list of ints or a string such as 'range(3,6) + range(40,57) + 7'."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_parse_band_list(v))
        return out
    out = []
    for term in str(value).split('+'):
        term = term.strip()
        if not term:
            continue
        m = _RANGE_RE.fullmatch(term)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2))))
        elif term.isdigit():
            out.append(int(term))
        else:
            raise ValueError(f"Cannot parse band specification term: {term!r}")
    return out


def registry_path() -> str:
    return os.environ.get('HSDL_DATASETS_FILE', DEFAULT_REGISTRY)


def load_registry(path: Optional[str] = None) -> dict:
    with open(path or registry_path()) as f:
        return yaml.safe_load(f) or {}


def available_datasets(path: Optional[str] = None) -> List[str]:
    return list((load_registry(path).get('datasets') or {}).keys())


def data_root(override: Optional[str] = None) -> str:
    """Absolute data root (CLI override > $HSDL_DATA_ROOT > registry data_root)."""
    return _resolve_data_root(load_registry(), override)


def _resolve_data_root(registry: dict, data_root: Optional[str]) -> str:
    root = data_root or os.environ.get('HSDL_DATA_ROOT') or registry.get('data_root') or 'data'
    root = os.path.expanduser(root)
    if not os.path.isabs(root):
        root = os.path.join(REPO_ROOT, root)
    return os.path.normpath(root)


class Config:
    def __init__(self, satellite_name=None, experiment=None, data_root=None, epochs=None, create_dirs=True):
        self._init(satellite_name, experiment, data_root, epochs, create_dirs)

    def reinit(self, satellite_name, experiment=None, data_root=None, epochs=None):
        """Re-run setup in-place so all existing references see updated paths."""
        self._init(satellite_name, experiment, data_root, epochs)

    def update_from(self, other: 'Config') -> None:
        """Copy every setting from another Config (used to sync the singleton)."""
        self.__dict__.clear()
        self.__dict__.update(other.__dict__)

    def _init(self, satellite_name, experiment=None, data_root=None, epochs=None, create_dirs=True):
        self.satellite_name = satellite_name
        if satellite_name is None:
            return

        registry = load_registry()
        datasets = registry.get('datasets') or {}
        if satellite_name not in datasets:
            raise ValueError(
                f"Unknown dataset {satellite_name!r}. Available: {', '.join(datasets)} "
                f"(defined in {registry_path()})"
            )
        ds = dict(registry.get('defaults') or {})
        ds.update(datasets[satellite_name])

        self.scene_format = ds.get('format')
        if self.scene_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Dataset {satellite_name!r}: format must be one of {SUPPORTED_FORMATS}, "
                f"got {self.scene_format!r}"
            )
        self.sensor = ds.get('sensor', self.scene_format)
        self.description = ds.get('description', '')
        self.region_label = ds.get('region_label', satellite_name)

        # === Directories ===
        self.data_root = _resolve_data_root(registry, data_root)
        self.base_data_dir = os.path.join(self.data_root, ds['base_dir'])
        self.input_data_dir = os.path.join(self.base_data_dir, ds['input_dir'])
        self.intermediate_dir = os.path.join(self.base_data_dir, ds['intermediate_dir'])
        self.output_dir = os.path.join(self.base_data_dir, ds['output_dir'])
        experiments_id = experiment or os.environ.get('HSDL_EXPERIMENT') or ds.get('experiment', 'experiments_001')
        self.experiment_id = experiments_id

        # === Scene / labels ===
        self.scene_id = ds['scene']
        self.scene_path = os.path.join(self.input_data_dir, self.scene_id)
        self.shapefile_path = os.path.join(self.input_data_dir, ds['shapefile'])
        self.class_field = ds['class_field']
        self.class_name_field = ds['class_name_field']
        if ds.get('clr_file'):
            self.clr_path = os.path.join(self.input_data_dir, ds['clr_file'])

        # === Format-specific reader settings ===
        if self.scene_format == 'prisma_he5':
            self.h5_path = self.scene_id
            self.h5_vnir_key = ds['h5_vnir_key']
            self.h5_swir_key = ds['h5_swir_key']
            self.h5_lat_key = ds['h5_lat_key']
            self.h5_lon_key = ds['h5_lon_key']
            self.vnir_bands_to_remove = _parse_band_list(ds.get('vnir_bands_to_remove'))
            self.swir_bands_to_remove = _parse_band_list(ds.get('swir_bands_to_remove'))
        else:
            self.bands_to_remove = _parse_band_list(ds.get('bands_to_remove'))

        # === Preprocessing ===
        self.num_mnf_bands = int(ds['num_mnf_bands'])
        self.clip_to_shapefile = bool(ds['clip_to_shapefile'])
        self.samples = int(ds['samples'])
        self.test_size = float(ds['test_size'])

        # Training epochs override for the deep models (None = model default).
        self.epochs = epochs if epochs is not None else (
            int(os.environ['HSDL_EPOCHS']) if os.environ.get('HSDL_EPOCHS') else None
        )

        # Shared country boundary for regional-map figures; prefer a copy inside
        # the dataset's own input folder when present.
        local_boundary = os.path.join(self.input_data_dir, 'egy_boundaries', 'egy_boundaries.shp')
        shared = registry.get('boundary_shapefile')
        self.boundary_shapefile_path = (
            local_boundary if os.path.exists(local_boundary) or not shared
            else os.path.join(self.data_root, shared)
        )

        # === Input raster files (processed data) ===
        self.tifpath = os.path.join(self.intermediate_dir,  'vnir_swir_stackedd.tif')
        self.train_mask_path = os.path.join(self.intermediate_dir,  'train_mask.tif')
        self.test_mask_path = os.path.join(self.intermediate_dir,  'test_mask.tif')
        self.label_tif_path = os.path.join(self.intermediate_dir,  'colored_class_labels.tif')

        # === Output directories ===
        self.experiment_path = os.path.join(self.output_dir, experiments_id)
        self.save_model_path = os.path.join(self.output_dir, experiments_id, 'saved_models')
        self.final_log_path = os.path.join(self.output_dir, experiments_id, 'logs')
        self.outputVIT_tif_path = os.path.join(self.output_dir, experiments_id, 'VIT_final_prediction_map.tif')
        self.outputCAF_tif_path = os.path.join(self.output_dir, experiments_id, 'CAF_final_prediction_map.tif')
        self.output_rf_tif_path = os.path.join(self.output_dir, experiments_id, 'rf_final_prediction_map.tif')
        self.output_xgb_tif_path = os.path.join(self.output_dir, experiments_id, 'xgb_final_prediction_map.tif')
        self.output_MAMBA_tif_path = os.path.join(self.output_dir, experiments_id, 'MAMBA_final_prediction_map.tif')

        # === Output files ===
        self.matrix_output_path = 'matrix.mat'
        self.new_final_class_labels = os.path.join(self.output_dir, experiments_id, 'new__newfinal_class_labels.tif')
        self.colored_output_tif = os.path.join(self.output_dir, experiments_id, 'colored_class_labels.tif')

        if create_dirs:
            self._create_required_dirs()

    def _create_required_dirs(self):
        for dir_path in {
            self.intermediate_dir,
            self.experiment_path,
            self.save_model_path,
            self.final_log_path,
        }:
            os.makedirs(dir_path, exist_ok=True)


config = Config()
