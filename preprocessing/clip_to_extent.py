import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box


def clip_tif_to_shapefile(tifpath: str, shapefile_path: str) -> None:
    """
    Clip the TIF at *tifpath* in-place to the bounding box of the shapefile.

    The shapefile is reprojected to the raster CRS before computing the extent,
    so mixed CRS combinations (e.g. WGS-84 shapefile + UTM raster) are handled
    automatically.  Only the bounding rectangle is used — not individual polygon
    geometries — which keeps all pixels that *could* contain training labels.
    """
    gdf = gpd.read_file(shapefile_path)

    with rasterio.open(tifpath) as src:
        raster_crs = src.crs
        nodata = src.nodata if src.nodata is not None else -9999.0

        # Reproject shapefile to raster CRS
        if raster_crs and gdf.crs and not gdf.crs.equals(raster_crs):
            gdf = gdf.to_crs(raster_crs)

        # Bounding box of all features
        minx, miny, maxx, maxy = gdf.total_bounds
        clip_geom = [box(minx, miny, maxx, maxy).__geo_interface__]

        out_image, out_transform = rio_mask(
            src, clip_geom, crop=True, all_touched=True, nodata=nodata
        )
        out_profile = src.profile.copy()
        out_profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            nodata=nodata,
        )

    with rasterio.open(tifpath, 'w', **out_profile) as dst:
        dst.write(out_image)

    print(
        f"Clipped to shapefile extent: "
        f"{out_image.shape[1]} rows × {out_image.shape[2]} cols "
        f"({out_image.shape[0]} bands)"
    )
