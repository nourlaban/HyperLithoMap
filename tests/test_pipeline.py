import os
import subprocess
import sys

import numpy as np
import pytest
import rasterio

from conftest import CLASSES, REPO_ROOT

EXPECTED_BANDS = {           # bands after band removal + 5 MNF components
    'SynthGeoTIFF': 48 - 4 + 5,
    'SynthENVI': 40 - 2 + 5,
    'SynthPRISMA': (66 - 5) + (173 - 66) + 5,
}


def _cuda_models_available():
    try:
        import torch
        import mamba_ssm  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


# ── configuration ────────────────────────────────────────────────────────────

def test_band_spec_parsing():
    from projutils.config import _parse_band_list
    assert _parse_band_list("range(3,6) + range(40,42) + 7") == [3, 4, 5, 40, 41, 7]
    assert _parse_band_list([1, 2, "range(5,7)"]) == [1, 2, 5, 6]
    assert _parse_band_list(None) == []


def test_repository_registry_loads(monkeypatch):
    """Every dataset shipped in configs/datasets.yaml resolves to a valid config."""
    from projutils.config import Config, available_datasets, DEFAULT_REGISTRY
    names = available_datasets(DEFAULT_REGISTRY)
    assert {'PRISMA', 'PRISMA_Gahelia', 'PRISMA_Gahelia_Clipped', 'EnMap', 'RabshiPRISMA'} <= set(names)
    monkeypatch.delenv('HSDL_DATASETS_FILE', raising=False)
    for name in names:
        cfg = Config(name, create_dirs=False)
        assert os.path.isabs(cfg.tifpath)
        assert cfg.scene_format in ('prisma_he5', 'envi', 'geotiff')


def test_unknown_dataset_rejected(synthetic_env):
    from projutils.config import Config
    with pytest.raises(ValueError, match='Unknown dataset'):
        Config('NoSuchDataset', create_dirs=False)


# ── preprocessing, one test per reader ───────────────────────────────────────

@pytest.mark.parametrize('dataset', ['SynthGeoTIFF', 'SynthENVI', 'SynthPRISMA'])
def test_preprocessing(synthetic_env, dataset):
    from projutils.config import config
    from src.pipeline import HyperspectralPipeline

    config.reinit(dataset)
    pipeline = HyperspectralPipeline(config, verbose=False)
    pipeline.run_preprocessing()

    assert pipeline.get_class_info() == {'Gabbro': 1, 'Granite': 2, 'Wadi deposits': 3, 'Background': 4}
    with rasterio.open(config.tifpath) as src:
        assert src.count == EXPECTED_BANDS[dataset]
        assert src.crs is not None
    with rasterio.open(config.new_final_class_labels) as src:
        labels = src.read(1)
    assert set(np.unique(labels)) == {0, 1, 2, 3, 4}
    with rasterio.open(config.train_mask_path) as tr, rasterio.open(config.test_mask_path) as te:
        train, test = tr.read(1), te.read(1)
    assert set(np.unique(train)) - {0} == {1, 2, 3, 4}
    assert not np.any((train > 0) & (test > 0)), 'train and test pixels overlap'


def test_geotiff_band_removal_keeps_spatial_size(synthetic_env):
    """Regression: band removal must drop bands, not image columns."""
    from preprocessing.converth5totiff import convert_geotiff_to_stacked_tiff
    src_path = os.path.join(synthetic_env['root'], 'synth_geotiff', 'input', 'scene.tif')
    out = os.path.join(synthetic_env['root'], 'band_removal_check.tif')
    convert_geotiff_to_stacked_tiff(src_path, list(range(40, 44)), out)
    with rasterio.open(src_path) as a, rasterio.open(out) as b:
        assert (b.count, b.height, b.width) == (a.count - 4, a.height, a.width)


def test_colormap_uses_clr(synthetic_env):
    from projutils.config import config
    config.reinit('SynthENVI')
    with rasterio.open(config.label_tif_path) as src:
        cmap = src.colormap(1)
    assert cmap[1][:3] == CLASSES['Granite']   # .clr row 1


def test_class_scheme_matches_raster_indices(synthetic_env):
    from projutils.class_scheme import load_class_scheme
    scheme = load_class_scheme('SynthENVI')
    assert scheme.CLASS_NAMES == ['Unlabeled', 'Gabbro', 'Granite', 'Wadi deposits', 'Background']
    assert scheme.COLORS_RGB[1] == list(CLASSES['Granite'])  # .clr row 1 wins (raster index 1)


# ── models + report ──────────────────────────────────────────────────────────

def _run_models(dataset, models, epochs=None):
    from projutils.config import config
    from main import class_info_from_shapefile, MODEL_CHOICES
    from src.model_runner import ModelRunner

    config.reinit(dataset, epochs=epochs)
    runner = ModelRunner(config=config, verbose=False)
    runner.run_multiple_models([MODEL_CHOICES[m] for m in models], class_info_from_shapefile(config))
    return config


def test_classical_models(synthetic_env):
    cfg = _run_models('SynthGeoTIFF', ['random_forests', 'xgboost'])
    for path in (cfg.output_rf_tif_path, cfg.output_xgb_tif_path):
        with rasterio.open(path) as src:
            assert (src.height, src.width) == rasterio.open(cfg.tifpath).shape
    assert os.path.exists(os.path.join(cfg.save_model_path, 'rf_model_evaluation', 'evaluation_results.txt'))
    assert os.path.exists(os.path.join(cfg.save_model_path, 'xgb_model_evaluation', 'evaluation_results.txt'))


@pytest.mark.skipif(not _cuda_models_available(), reason='needs a CUDA GPU and mamba_ssm')
def test_deep_models_and_report(synthetic_env):
    cfg = _run_models('SynthGeoTIFF', ['spectral_former', 'mamba'], epochs=2)
    assert os.path.exists(cfg.outputVIT_tif_path)
    assert os.path.exists(cfg.output_MAMBA_tif_path)

    from main import run_evaluation_report
    run_evaluation_report(cfg)
    figures = os.listdir(os.path.join(cfg.experiment_path, 'figures'))
    assert any(f.endswith('.png') for f in figures)
    assert any(f.startswith('Evaluation_Report') and f.endswith('.docx') for f in os.listdir(cfg.experiment_path))


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_list_and_train_skip_preprocessing(synthetic_env):
    env = dict(os.environ, HSDL_DATASETS_FILE=str(synthetic_env['registry']), MPLBACKEND='Agg')
    out = subprocess.run([sys.executable, 'main.py', 'list-datasets'], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    assert 'SynthPRISMA' in out and 'scene found' in out

    # Needs the preprocessed rasters from test_preprocessing[SynthENVI].
    subprocess.run([sys.executable, 'main.py', 'train', '--dataset', 'SynthENVI', '--skip-preprocessing',
                    '--models', 'random_forests', '--experiment', 'cli_run'],
                   cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True)
    assert os.path.exists(os.path.join(synthetic_env['root'], 'synth_envi', 'output', 'cli_run',
                                       'rf_final_prediction_map.tif'))
