"""
Main pipeline orchestrator for hyperspectral image processing.
Handles the complete workflow: data preparation, preprocessing, and model training/inference.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from projutils.config import Config
from preprocessing.convertshapelabelstoraster import process_shapefile_to_raster
from preprocessing.color_map import apply_colormap_to_tiff
from preprocessing.split_pixel import create_train_test_masks
from preprocessing.converth5totiff import convert_h5_to_stacked_tiff, convert_enmap_to_stacked_tiff, convert_dat_to_stacked_tiff

logger = logging.getLogger(__name__)


class HyperspectralPipeline:
    """
    Orchestrates the complete hyperspectral image processing pipeline.
    
    Workflow:
    1. Convert satellite data (PRISMA .he5, ENVI or GeoTIFF) to stacked TIFF
    2. Process vector labels to raster format
    3. Apply color mapping for visualization
    4. Create train/test masks for model evaluation
    5. Run machine learning models
    """
    
    def __init__(self, config: Config, verbose: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object containing paths and parameters
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self._setup_logging()
        self.all_classes = None
        
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        log_level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def run_preprocessing(self) -> None:
        """Execute the complete preprocessing pipeline."""
        logger.info("=" * 80)
        logger.info("Starting Hyperspectral Image Preprocessing Pipeline")
        logger.info(f"Dataset: {self.config.satellite_name} ({self.config.sensor}, {self.config.scene_format})")
        logger.info("=" * 80)
        
        # Step 1: Convert satellite data to stacked TIFF
        self._step_convert_satellite_data()

        # Step 1a: Clip to shapefile extent (reduces image size before MNF)
        self._step_clip_to_shapefile()

        # Step 1b: Append MNF bands to the stacked TIFF
        self._step_append_mnf()

        # Step 2: Process shapefile labels to raster
        self._step_process_labels()
        
        # Step 3: Apply color mapping
        self._step_apply_colormap()
        
        # Step 4: Create train/test masks
        self._step_create_masks()
        
        logger.info("=" * 80)
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info("=" * 80)
    
    def _step_convert_satellite_data(self) -> None:
        """Step 1: Convert raw satellite data to stacked TIFF format."""
        logger.info("\n[Step 1/4] Converting satellite data to stacked TIFF format...")
        
        fmt = self.config.scene_format
        try:
            if fmt == 'prisma_he5':
                logger.info(f"Processing {self.config.sensor} HDF5 (.he5) data...")
                convert_h5_to_stacked_tiff(
                    h5_path=self.config.scene_path,
                    vnir_key=self.config.h5_vnir_key,
                    swir_key=self.config.h5_swir_key,
                    lat_key=self.config.h5_lat_key,
                    lon_key=self.config.h5_lon_key,
                    output_tiff_path=self.config.tifpath,
                    vnir_bands_to_remove=self.config.vnir_bands_to_remove,
                    swir_bands_to_remove=self.config.swir_bands_to_remove
                )

            elif fmt == 'geotiff':
                logger.info(f"Processing {self.config.sensor} GeoTIFF data...")
                convert_enmap_to_stacked_tiff(
                    tif_path=self.config.scene_path,
                    bands_to_remove=self.config.bands_to_remove,
                    output_tiff_path=self.config.tifpath
                )

            elif fmt == 'envi':
                logger.info(f"Processing {self.config.sensor} ENVI data...")
                convert_dat_to_stacked_tiff(
                    dat_path=self.config.scene_path,
                    output_tiff_path=self.config.tifpath,
                    bands_to_remove=self.config.bands_to_remove
                )

            else:
                raise ValueError(f"Unsupported scene format: {fmt}")

            logger.info(f"✓ Stacked TIFF created: {self.config.tifpath}")

        except Exception as e:
            logger.error(f"✗ Error converting satellite data: {str(e)}")
            raise

    def _step_clip_to_shapefile(self) -> None:
        """Step 1a: Clip stacked TIF to shapefile bounding extent (in-place)."""
        if not getattr(self.config, 'clip_to_shapefile', False):
            return

        logger.info(f"\n[Step 1a] Clipping {self.config.tifpath} to shapefile extent...")
        try:
            from preprocessing.clip_to_extent import clip_tif_to_shapefile
            clip_tif_to_shapefile(self.config.tifpath, self.config.shapefile_path)
            logger.info(f"✓ TIF clipped to shapefile extent: {self.config.tifpath}")
        except Exception as e:
            logger.error(f"✗ Error clipping TIF: {str(e)}")
            raise

    def _step_append_mnf(self) -> None:
        """Step 1b: Compute MNF and append most informative bands to the stacked TIFF."""
        num = getattr(self.config, 'num_mnf_bands', 0)
        if num <= 0:
            return

        logger.info(f"\n[Step 1b] Appending {num} MNF band(s) to {self.config.tifpath}...")
        try:
            from preprocessing.mnf_transform import append_mnf_to_tif
            append_mnf_to_tif(self.config.tifpath, num_components=num)
            logger.info(f"✓ {num} MNF band(s) appended to {self.config.tifpath}")
        except Exception as e:
            logger.error(f"✗ Error computing MNF: {str(e)}")
            raise

    def _step_process_labels(self) -> None:
        """Step 2: Convert shapefile labels to raster format."""
        logger.info("\n[Step 2/4] Processing shapefile labels to raster...")
        
        try:
            self.all_classes = process_shapefile_to_raster(
                reference_tif=self.config.tifpath,
                shapefile_path=self.config.shapefile_path,
                output_label_tif=self.config.new_final_class_labels,
                class_field=self.config.class_field,
                class_name_field=self.config.class_name_field,
                class_shift=0,
                unclassified_value=0
            )
            
            logger.info(f"✓ Labels rasterized: {self.config.new_final_class_labels}")
            logger.info(f"  Number of classes: {len(self.all_classes)}")
            
        except Exception as e:
            logger.error(f"✗ Error processing labels: {str(e)}")
            raise
    
    def _step_apply_colormap(self) -> None:
        """Step 3: Apply color mapping to labels for visualization."""
        logger.info("\n[Step 3/4] Applying color mapping for visualization...")
        
        try:
            apply_colormap_to_tiff(
                self.config.new_final_class_labels,
                self.config.label_tif_path,
                clr_path=getattr(self.config, 'clr_path', None)
            )
            logger.info(f"✓ Colored labels created: {self.config.label_tif_path}")
            
        except Exception as e:
            logger.error(f"✗ Error applying colormap: {str(e)}")
            raise
    
    def _step_create_masks(self) -> None:
        """Step 4: Create train/test split masks."""
        logger.info("\n[Step 4/4] Creating train/test split masks...")
        
        try:
            create_train_test_masks(
                self.config.new_final_class_labels,
                self.config.train_mask_path,
                self.config.test_mask_path,
                self.config.samples,
                self.config.test_size
            )
            logger.info(f"✓ Train mask created: {self.config.train_mask_path}")
            logger.info(f"✓ Test mask created: {self.config.test_mask_path}")
            
        except Exception as e:
            logger.error(f"✗ Error creating masks: {str(e)}")
            raise
    
    def get_class_info(self) -> Optional[List]:
        """Return the processed class information."""
        return self.all_classes
