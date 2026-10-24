"""
Model runner module for executing different machine learning models.
Supports RandomForests, XGBoost, Mamba, and SpectralFormer models.
"""

import logging
from typing import List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    RANDOM_FORESTS = "RandomForests"
    XGBOOST = "XGBoost"
    MAMBA = "Mamba"
    SPECTRAL_FORMER = "SpectralFormer"


class ModelRunner:
    """
    Executes different machine learning models for hyperspectral image classification.
    """

    def __init__(self, config=None, verbose: bool = True):
        self.config = config
        self.verbose = verbose

    def _sync_module_config(self):
        """Point the module-level config singleton (read by the model modules) at this run's config."""
        if self.config is not None:
            from projutils.config import config as _cfg
            if _cfg is not self.config:
                _cfg.update_from(self.config)

    def run_model(self, model_type: ModelType, all_classes: List, **kwargs) -> None:
        """
        Run a specific model.
        
        Args:
            model_type: Type of model to run
            all_classes: Processed class information
            **kwargs: Additional arguments for the model
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Executing: {model_type.value}")
        logger.info(f"{'='*80}")

        self._sync_module_config()

        try:
            if model_type == ModelType.RANDOM_FORESTS:
                self._run_random_forests(all_classes, **kwargs)
                
            elif model_type == ModelType.XGBOOST:
                self._run_xgboost(all_classes, **kwargs)
                
            elif model_type == ModelType.MAMBA:
                self._run_mamba(all_classes, **kwargs)
                
            elif model_type == ModelType.SPECTRAL_FORMER:
                self._run_spectral_former(all_classes, **kwargs)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            logger.info(f"✓ {model_type.value} completed successfully")
            
        except Exception as e:
            logger.error(f"✗ Error running {model_type.value}: {str(e)}")
            raise
    
    def run_multiple_models(self, model_types: List[ModelType], all_classes: List, **kwargs) -> None:
        """
        Run multiple models in sequence.
        
        Args:
            model_types: List of model types to run
            all_classes: Processed class information
            **kwargs: Additional arguments for models
        """
        logger.info(f"\nRunning {len(model_types)} models...")
        
        for model_type in model_types:
            self.run_model(model_type, all_classes, **kwargs)
    
    def _run_random_forests(self, all_classes: List, **kwargs) -> None:
        """Execute Random Forests model."""
        try:
            from hsmodel import demo_RandomForests
            demo_RandomForests.main(all_classes)
        except ImportError:
            logger.error("Random Forests demo module not found")
            raise
    
    def _run_xgboost(self, all_classes: List, **kwargs) -> None:
        """Execute XGBoost model."""
        try:
            from hsmodel import demo_eXtremeGradientBoosting
            demo_eXtremeGradientBoosting.main(all_classes)
        except ImportError:
            logger.error("XGBoost demo module not found")
            raise
    
    def _run_mamba(self, all_classes: List, **kwargs) -> None:
        """Execute Mamba State Space model."""
        try:
            from hsmodel import demo_mambaStateSpaceModel
            demo_mambaStateSpaceModel.main(all_classes)
        except ImportError:
            logger.error("Mamba demo module not found")
            raise
    
    def _run_spectral_former(self, all_classes: List, **kwargs) -> None:
        """Execute SpectralFormer model."""
        try:
            from hsmodel import demo_SpectralFormer
            demo_SpectralFormer.main(all_classes)
        except ImportError:
            logger.error("SpectralFormer demo module not found")
            raise
