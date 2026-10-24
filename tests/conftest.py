"""
Synthetic fixtures: a tiny scene in each supported format (PRISMA .he5, ENVI,
GeoTIFF) plus a labelled training shapefile, registered in a temporary
datasets.yaml so the real pipeline code runs end-to-end without real data.
"""

import os
import sys

import geopandas as gpd
import h5py
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import box

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

H, W = 36, 40
CLASSES = {  # name -> (RED, GREEN, BLUE)
    'Granite': (220, 50, 50),
    'Gabbro': (40, 90, 200),
    'Wadi deposits': (210, 190, 140),
    'Background': (200, 200, 200),
}


def _cube(bands, seed):
    """(B, H, W) float32 cube with a distinct spectral signature per quadrant."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.1, size=(bands, H, W)).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, bands, dtype=np.float32)[:, None, None]
    base[:, : H // 2, : W // 2] += 0.5 * ramp
    base[:, : H // 2, W // 2:] += 0.5 * ramp[::-1]
    base[:, H // 2:, : W // 2] += 0.3
    return base


def _labels_shapefile(path, bounds, crs):
    """Four class polygons, one per image quadrant (inset so they stay inside)."""
    minx, miny, maxx, maxy = bounds
    midx, midy = (minx + maxx) / 2, (miny + maxy) / 2
    dx, dy = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    quads = [
        (minx + dx, midy + dy, midx - dx, maxy - dy),
        (midx + dx, midy + dy, maxx - dx, maxy - dy),
        (minx + dx, miny + dy, midx - dx, midy - dy),
        (midx + dx, miny + dy, maxx - dx, midy - dy),
    ]
    rows = []
    for (name, rgb), q in zip(CLASSES.items(), quads):
        rows.append({'Classname': name, 'Classcode': 0, 'RED': rgb[0], 'GREEN': rgb[1],
                     'BLUE': rgb[2], 'geometry': box(*q)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gpd.GeoDataFrame(rows, crs=crs).to_file(path)


def _write_geotiff(path, cube, driver='GTiff'):
    transform = from_origin(500000, 2800000, 30, 30)
    with rasterio.open(path, 'w', driver=driver, height=H, width=W, count=cube.shape[0],
                       dtype='float32', crs='EPSG:32636', transform=transform) as dst:
        dst.write(cube)
    return (500000, 2800000 - 30 * H, 500000 + 30 * W, 2800000)


def _write_prisma_he5(path):
    vnir = np.transpose(_cube(66, 1), (1, 0, 2))    # (H, 66, W) like PRISMA L2D
    swir = np.transpose(_cube(173, 2), (1, 0, 2))   # (H, 173, W)
    lon, lat = np.meshgrid(np.linspace(33.0, 33.04, W), np.linspace(25.04, 25.0, H))
    root = 'HDFEOS/SWATHS/PRS_L2D_HCO'
    with h5py.File(path, 'w') as f:
        f[f'{root}/Data Fields/VNIR_Cube'] = vnir
        f[f'{root}/Data Fields/SWIR_Cube'] = swir
        f[f'{root}/Geolocation Fields/Latitude'] = lat.astype(np.float32)
        f[f'{root}/Geolocation Fields/Longitude'] = lon.astype(np.float32)
    return (33.0, 25.0, 33.04, 25.04)


@pytest.fixture(scope='session')
def synthetic_env(tmp_path_factory):
    """Create the synthetic data root + registry and point the config at them."""
    root = tmp_path_factory.mktemp('hsdl_data')

    inp = root / 'synth_geotiff' / 'input'
    inp.mkdir(parents=True)
    bounds = _write_geotiff(str(inp / 'scene.tif'), _cube(48, 3))
    _labels_shapefile(str(inp / 'labels' / 'labels.shp'), bounds, 'EPSG:32636')

    inp = root / 'synth_envi' / 'input'
    inp.mkdir(parents=True)
    bounds = _write_geotiff(str(inp / 'scene.dat'), _cube(40, 4), driver='ENVI')
    _labels_shapefile(str(inp / 'labels' / 'labels.shp'), bounds, 'EPSG:32636')
    with open(inp / 'classes.clr', 'w') as f:
        for i, rgb in enumerate(CLASSES.values(), start=1):
            f.write(f"{i} {rgb[0]} {rgb[1]} {rgb[2]} 255\n")

    inp = root / 'synth_prisma' / 'input'
    inp.mkdir(parents=True)
    bounds = _write_prisma_he5(str(inp / 'scene.he5'))
    _labels_shapefile(str(inp / 'labels' / 'labels.shp'), bounds, 'EPSG:4326')

    with open(os.path.join(REPO_ROOT, 'configs', 'datasets.yaml')) as f:
        prisma_keys = {k: v for k, v in yaml.safe_load(f)['_prisma_l2d'].items()}

    common = {'experiment': 'test_run', 'scene': None, 'shapefile': 'labels/labels.shp',
              'num_mnf_bands': 5, 'samples': 60, 'test_size': 0.5}
    registry = {
        'data_root': str(root),
        'defaults': {'input_dir': 'input', 'intermediate_dir': 'intermediate', 'output_dir': 'output',
                     'class_field': 'Classcode', 'class_name_field': 'Classname',
                     'clip_to_shapefile': True, 'clr_file': None},
        'datasets': {
            'SynthGeoTIFF': {**common, 'format': 'geotiff', 'sensor': 'EnMAP', 'base_dir': 'synth_geotiff',
                             'scene': 'scene.tif', 'bands_to_remove': 'range(40,44)'},
            'SynthENVI': {**common, 'format': 'envi', 'sensor': 'PRISMA', 'base_dir': 'synth_envi',
                          'scene': 'scene.dat', 'bands_to_remove': [0, 1], 'clr_file': 'classes.clr'},
            'SynthPRISMA': {**common, **prisma_keys, 'base_dir': 'synth_prisma', 'scene': 'scene.he5'},
        },
    }
    reg_path = root / 'datasets.yaml'
    with open(reg_path, 'w') as f:
        yaml.safe_dump(registry, f)

    mp = pytest.MonkeyPatch()
    mp.setenv('HSDL_DATASETS_FILE', str(reg_path))
    mp.delenv('HSDL_DATA_ROOT', raising=False)
    mp.delenv('HSDL_EXPERIMENT', raising=False)
    mp.setenv('MPLBACKEND', 'Agg')
    yield {'root': root, 'registry': reg_path}
    mp.undo()
