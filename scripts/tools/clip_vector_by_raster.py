import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon
import numpy as np

def clip_vector_by_raster_footprint(raster_path, vector_path, output_path):
    """
    Clips a shapefile to the valid data area (non-nodata) of a raster.
    """
    print("1. Loading Raster...")
    with rasterio.open(raster_path) as src:
        # Read the first band
        # You can adjust this if you need a specific band or a composite mask
        band = src.read(1)
        
        # Determine the transform (to convert pixel coords to map coords)
        transform = src.transform
        
        # Get the nodata value
        nodata = src.nodata
        
        # Handle cases where nodata might be None
        if nodata is None:
            print("   Warning: NoData value not defined in raster metadata.")
            # Heuristic: Assume 0 is nodata if not defined, or require user input
            nodata = 0 

        print(f"   Raster CRS: {src.crs}")
        print(f"   NoData Value: {nodata}")

        # 2. Create a Mask of Valid Data
        # Create a boolean mask where True = Valid Data, False = NoData
        # Note: rasterio.features.shapes expects an image array. 
        # We create a simple binary array: 1 for valid, 0 for nodata.
        
        # Initialize mask with 0s
        mask_array = np.zeros(band.shape, dtype=np.uint8)
        
        # Set pixels that are NOT nodata to 1
        # Use np.isclose for float comparisons to be safe, or != for integers
        if np.issubdtype(band.dtype, np.floating):
            mask_array[~np.isclose(band, nodata)] = 1
        else:
            mask_array[band != nodata] = 1

        print("2. Extracting Footprint Polygons...")
        # rasterio.features.shapes yields (geometry, value) pairs
        # We only want shapes where the value is 1 (valid data)
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(mask_array, mask=mask_array==1, transform=transform)
            )
        )

        # Convert the generator to a list of Shapely geometries
        geoms = list(results)
        
        if not geoms:
            print("Error: No valid data found in raster.")
            return

        # Create a GeoDataFrame from the raster footprint
        footprint_gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

    # 3. Dissolve Footprint
    # The 'shapes' function might return many small polygons. 
    # We dissolve them into one unified geometry to speed up the clip.
    print("3. Dissolving footprint geometries...")
    footprint_dissolved = footprint_gdf.dissolve()

    # 4. Load Vector Data
    print("4. Loading Input Vector...")
    input_gdf = gpd.read_file(vector_path)

    # 5. CRS Check and Reprojection
    # The raster and vector must be in the same CRS for the clip to work.
    if input_gdf.crs != footprint_dissolved.crs:
        print(f"   Reprojecting vector from {input_gdf.crs} to {footprint_dissolved.crs}...")
        input_gdf = input_gdf.to_crs(footprint_dissolved.crs)

    # 6. Perform the Clip
    print("5. Clipping Vector Data...")
    # The clip function keeps parts of the input_gdf that intersect with footprint_dissolved
    clipped_gdf = input_gdf.clip(footprint_dissolved)

    # 7. Save Output
    if not clipped_gdf.empty:
        print(f"6. Saving result to {output_path}...")
        clipped_gdf.to_file(output_path)
        print("Done.")
    else:
        print("Result was empty (no overlap between raster data and vector).")

# ==========================================
# CONFIGURATION
# ==========================================
# Update these paths to your actual files
raster_file = "path/to/your_raster.tif"
shapefile_in = "path/to/your_vector.shp"
shapefile_out = "path/to/clipped_output.shp"

# Run the function
try:
    clip_vector_by_raster_footprint(raster_file, shapefile_in, shapefile_out)
except Exception as e:
    print(f"An error occurred: {e}")