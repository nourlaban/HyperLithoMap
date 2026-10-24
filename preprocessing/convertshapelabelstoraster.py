import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np


def create_code(shapefile, name_field='Classname'):
        """Map class names to 1-based raster indices (alphabetical; Cloud/Background last)."""
        all_classes = set()
        vector_data = gpd.read_file(shapefile)
        all_classes.update(vector_data[name_field].dropna().unique())
        
        sorted_classes = sorted(list(all_classes))

        if 'Cloud' in sorted_classes:
            sorted_classes.remove('Cloud')
            sorted_classes.insert(len(all_classes), 'Cloud')
    
        if 'Background' in sorted_classes:
            sorted_classes.remove('Background')
            sorted_classes.insert(len(all_classes), 'Background')
        
        # return  {i+1: class_name for i, class_name in enumerate(sorted_classes)}
        all_classes =  {class_name: i +1  for i, class_name in enumerate(sorted_classes)}
       

        return all_classes 

def load_and_preprocess_shapefile(shapefile_path, class_field, class_shift=2, class_name_field='Classname'):
    """
    Load shapefile and preprocess class values.
    
    Args:
        shapefile_path (str): Path to the shapefile
        class_field (str): Name of the class field column
        class_shift (int): Value to shift classes by (default: 2)
    
    Returns:
        gpd.GeoDataFrame: Preprocessed geodataframe
    """

    all_classes = create_code(shapefile_path, class_name_field)
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf[gdf[class_name_field].notna()]

    gdf[class_field] = gdf[class_name_field].map(all_classes)

    # gdf[class_field] = gdf[class_field].map(all_classes) + class_shift
    return gdf , all_classes


def load_reference_raster_metadata(reference_tif):
    """
    Load reference raster metadata and data mask.
    
    Args:
        reference_tif (str): Path to reference raster
    
    Returns:
        tuple: (crs, transform, shape, profile, data_mask)
    """
    with rasterio.open(reference_tif) as src:
        ref_crs = src.crs
        ref_transform = src.transform
        ref_shape = (src.height, src.width)
        ref_profile = src.profile
        data_mask = ~src.read(1, masked=True).mask
    
    return ref_crs, ref_transform, ref_shape, ref_profile, data_mask


def reproject_shapefile_if_needed(gdf, target_crs):
    """
    Reproject shapefile to target CRS if they don't match.
    
    Args:
        gdf (gpd.GeoDataFrame): Input geodataframe
        target_crs: Target coordinate reference system
    
    Returns:
        gpd.GeoDataFrame: Reprojected geodataframe
    """
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def create_label_raster(gdf, class_field, ref_shape, ref_transform, 
                       data_mask, unclassified_value=2):
    """
    Create label raster from shapefile.
    
    Args:
        gdf (gpd.GeoDataFrame): Input geodataframe
        class_field (str): Name of the class field column
        ref_shape (tuple): Shape of reference raster (height, width)
        ref_transform: Raster transform
        data_mask (np.array): Boolean mask of valid data areas
        unclassified_value (int): Value for unclassified pixels (default: 2)
        background_value (int): Value for background pixels (default: 1)
    
    Returns:
        np.array: Label raster array
    """
    # Initialize raster with unclassified value
    burned = np.full(ref_shape, unclassified_value, dtype=np.uint8)
    
    # Create shapes for rasterization
    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf[class_field]))
    
    # Rasterize shapefile
    burned_from_shapes = features.rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=unclassified_value,
        dtype=np.uint8
    )
    
    # Update burned array
    burned[:, :] = burned_from_shapes
    
    # Set background value for areas outside image data
    # burned[~data_mask] = background_value
    
    return burned


def save_label_raster(label_array, output_path, ref_profile):
    """
    Save label raster to file.
    
    Args:
        label_array (np.array): Label raster array
        output_path (str): Output file path
        ref_profile (dict): Reference raster profile
    """
    # Update profile for output
    output_profile = ref_profile.copy()
    output_profile.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "compress": "lzw"
    })
    
    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(label_array, 1)


def process_shapefile_to_raster(reference_tif, shapefile_path, output_label_tif,
                               class_field, class_shift=2, unclassified_value=2,
                               class_name_field='Classname'):
    """
    Main function to process shapefile into label raster.
    
    Args:
        reference_tif (str): Path to reference raster
        shapefile_path (str): Path to shapefile
        output_label_tif (str): Path for output label raster
        class_field (str): Name of the class field column
        class_shift (int): Value to shift classes by (default: 2)
        unclassified_value (int): Value for unclassified pixels (default: 2)
        background_value (int): Value for background pixels (default: 1)
    """
    # Load and preprocess shapefile
    print("Loading and preprocessing shapefile...")
    gdf, all_classes = load_and_preprocess_shapefile(shapefile_path, class_field, class_shift, class_name_field)
    
    # Load reference raster metadata
    print("Loading reference raster metadata...")
    ref_crs, ref_transform, ref_shape, ref_profile, data_mask = load_reference_raster_metadata(reference_tif)
    
    # Reproject shapefile if needed
    print("Checking CRS compatibility...")
    gdf = reproject_shapefile_if_needed(gdf, ref_crs)
    
    # Create label raster
    print("Creating label raster...")
    label_array = create_label_raster(
        gdf, class_field, ref_shape, ref_transform, 
        data_mask, unclassified_value
    )
    
    # Save label raster
    print("Saving label raster...")
    save_label_raster(label_array, output_label_tif, ref_profile)
    classnames = gdf[class_name_field].unique()
    # print(f"✅ Raster saved with background={background_value}, unclassified={unclassified_value}, and shifted classes from {3}+")
    print(f"Columns = {classnames}")
    return all_classes


def main():
    """
    Main execution function using config parameters.
    """
    from projutils.config import config
    # Input paths from config
    reference_tif = config.tifpath
    shapefile_path = config.shapefile_path
    output_label_tif = config.new_final_class_labels
    class_field = config.class_field
    
    # Process shapefile to raster
    process_shapefile_to_raster(
        reference_tif=reference_tif,
        shapefile_path=shapefile_path,
        output_label_tif=output_label_tif,
        class_field=class_field,
        class_shift=2,
        unclassified_value=2,
        class_name_field=config.class_name_field,
    )


if __name__ == "__main__":
    main()