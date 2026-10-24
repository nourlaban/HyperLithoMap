#!/usr/bin/env python3
"""
Main entry point for the Hyperspectral Image Classification Framework.
Provides CLI interface for preprocessing, model training, and inference.

Datasets (scene + training shapefile) are declared in configs/datasets.yaml;
run `python main.py list-datasets` to see what is available.
"""

import argparse
import logging
import os

from projutils.config import config, available_datasets, registry_path, Config
from src.pipeline import HyperspectralPipeline
from src.model_runner import ModelRunner, ModelType

MODEL_CHOICES = {
    'random_forests': ModelType.RANDOM_FORESTS,
    'xgboost': ModelType.XGBOOST,
    'mamba': ModelType.MAMBA,
    'spectral_former': ModelType.SPECTRAL_FORMER,
}


def setup_logging(level: int = logging.INFO, log_file: str = 'hyperspectral_processing.log') -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def init_config(args) -> Config:
    """Initialise the shared config singleton from the CLI arguments."""
    config.reinit(
        args.satellite,
        experiment=args.experiment,
        data_root=args.data_root,
        epochs=getattr(args, 'epochs', None),
    )
    return config


def class_info_from_shapefile(cfg: Config) -> dict:
    """Rebuild the {class name: raster index} mapping used during preprocessing."""
    from preprocessing.convertshapelabelstoraster import create_code
    return create_code(cfg.shapefile_path, cfg.class_name_field)


def run_preprocessing(args) -> None:
    """Execute the preprocessing pipeline."""
    cfg = init_config(args)
    pipeline = HyperspectralPipeline(cfg, verbose=args.verbose)
    pipeline.run_preprocessing()


def run_models(args) -> None:
    """Execute machine learning models."""
    cfg = init_config(args)

    if not args.skip_preprocessing:
        pipeline = HyperspectralPipeline(cfg, verbose=args.verbose)
        pipeline.run_preprocessing()
        all_classes = pipeline.get_class_info()
    else:
        missing = [p for p in (cfg.tifpath, cfg.train_mask_path, cfg.test_mask_path, cfg.label_tif_path)
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "--skip-preprocessing was given but preprocessed files are missing: "
                + ", ".join(missing) + ". Run `main.py preprocess` first."
            )
        all_classes = class_info_from_shapefile(cfg)

    if not all_classes:
        raise ValueError("Could not obtain class information. Run preprocessing first.")

    model_runner = ModelRunner(config=cfg, verbose=args.verbose)
    models = [MODEL_CHOICES[m] for m in args.models]
    model_runner.run_multiple_models(models, all_classes)

    if args.report:
        run_evaluation_report(cfg)


def run_full_pipeline(args) -> None:
    """Execute the complete pipeline: preprocessing + all models + evaluation report."""
    cfg = init_config(args)

    pipeline = HyperspectralPipeline(cfg, verbose=args.verbose)
    pipeline.run_preprocessing()
    all_classes = pipeline.get_class_info()

    model_runner = ModelRunner(config=cfg, verbose=args.verbose)
    model_runner.run_multiple_models([MODEL_CHOICES[m] for m in args.models], all_classes)

    if not args.skip_report:
        run_evaluation_report(cfg)


def run_report(args) -> None:
    """Generate figures and the evaluation report for an existing experiment."""
    run_evaluation_report(init_config(args))


def run_evaluation_report(cfg) -> None:
    """Step 5: export comparison/confusion-matrix figures and a data-driven evaluation report."""
    from preprocessing.export_figures import export_thematic_maps_to_figures, export_confusion_matrix_panel
    from preprocessing.generate_evaluation_report import generate_report

    logging.info("\n[Step 5/5] Generating evaluation figures and report...")
    export_thematic_maps_to_figures(
        experiments_dir=cfg.experiment_path,
        clr_path=getattr(cfg, 'clr_path', None),
        shapefile_path=cfg.shapefile_path,
        class_name_field=cfg.class_name_field,
    )
    export_confusion_matrix_panel(
        saved_models_dir=cfg.save_model_path,
        output_dir=os.path.join(cfg.experiment_path, 'figures'),
    )
    generate_report(cfg)


def list_datasets(_args) -> None:
    """Print the datasets defined in the registry."""
    print(f"Datasets defined in {registry_path()}:\n")
    for name in available_datasets():
        try:
            cfg = Config(name, create_dirs=False)
            scene_ok = 'found' if os.path.exists(cfg.scene_path) else 'missing'
            print(f"  {name:<24} {cfg.sensor:<8} {cfg.scene_format:<11} scene {scene_ok:<8} {cfg.description}")
        except Exception as e:  # keep listing even if one entry is broken
            print(f"  {name:<24} ERROR: {e}")


def main():
    """Main entry point with argument parsing."""
    datasets = available_datasets()
    default_dataset = datasets[0] if datasets else None

    parser = argparse.ArgumentParser(
        description="Hyperspectral Image Classification Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show configured datasets
  python main.py list-datasets

  # Run preprocessing only
  python main.py preprocess --dataset RabshiPRISMA

  # Train selected models on already-preprocessed data
  python main.py train --dataset EnMap --models random_forests xgboost --skip-preprocessing

  # Run complete pipeline into a new experiment folder
  python main.py run-all --dataset PRISMA_Gahelia_Clipped --experiment experiments_010 --verbose
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    def add_common_args(p):
        p.add_argument('--dataset', '--satellite', dest='satellite', choices=datasets, default=default_dataset,
                       help=f'Dataset from configs/datasets.yaml (default: {default_dataset})')
        p.add_argument('--experiment', default=None,
                       help="Experiment folder name under the dataset's output dir "
                            "(default: the dataset's configured experiment)")
        p.add_argument('--data-root', default=None,
                       help='Root directory containing the dataset folders (default: $HSDL_DATA_ROOT or ./data)')
        p.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    def add_model_args(p, default):
        p.add_argument('--models', '-m', nargs='+', choices=list(MODEL_CHOICES), default=default,
                       help=f"Models to run (default: {' '.join(default)})")
        p.add_argument('--epochs', type=int, default=None,
                       help='Override training epochs for the deep models (mamba, spectral_former)')

    subparsers.add_parser('list-datasets', help='List datasets defined in configs/datasets.yaml')

    preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocessing pipeline only')
    add_common_args(preprocess_parser)

    train_parser = subparsers.add_parser('train', help='Run machine learning models')
    add_common_args(train_parser)
    add_model_args(train_parser, ['mamba', 'spectral_former'])
    train_parser.add_argument('--skip-preprocessing', action='store_true',
                              help='Reuse existing preprocessed rasters instead of re-running preprocessing')
    train_parser.add_argument('--report', action='store_true',
                              help='Generate evaluation figures/report after training')

    full_parser = subparsers.add_parser('run-all', help='Run complete pipeline (preprocessing + models + report)')
    add_common_args(full_parser)
    add_model_args(full_parser, list(MODEL_CHOICES))
    full_parser.add_argument('--skip-report', action='store_true',
                             help='Skip generating evaluation figures/report after training')

    report_parser = subparsers.add_parser('report', help='Generate figures + evaluation report for an experiment')
    add_common_args(report_parser)

    args = parser.parse_args()

    if args.command in (None, 'list-datasets'):
        if args.command is None:
            parser.print_help()
        else:
            list_datasets(args)
        return

    setup_logging(logging.INFO if args.verbose else logging.WARNING)

    if args.command == 'preprocess':
        run_preprocessing(args)
    elif args.command == 'train':
        run_models(args)
    elif args.command == 'run-all':
        run_full_pipeline(args)
    elif args.command == 'report':
        run_report(args)


if __name__ == '__main__':
    main()
