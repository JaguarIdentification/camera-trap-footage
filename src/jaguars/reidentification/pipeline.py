"""Re-identification pipeline orchestration.

This module orchestrates the complete re-identification pipeline:
1. Data ingestion (from FiftyOne, disk, or HuggingFace)
2. Feature extraction with backbone
3. Model training
4. Evaluation
5. Export results

Can be run as:
- Python function from notebooks
- CLI: python -m jaguars.reidentification.pipeline
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from jaguars.common.logging_utils import setup_logger
from jaguars.reidentification.config import ReidentificationConfig, get_default_config
from jaguars.reidentification.evaluation.evaluation import run_processing as run_evaluation
from jaguars.reidentification.export_results import run_processing as run_export
from jaguars.reidentification.training.train import run_processing as run_training

MODULE_NAME = "reidentification.pipeline"
logger = setup_logger(MODULE_NAME)


def run_pipeline(
    config: ReidentificationConfig | None = None,
    steps: list[str] | None = None,
    model_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run complete re-identification pipeline.

    Args:
        config: Configuration object
        steps: List of steps to run (train, evaluate, export). If None, runs all.
        model_path: Path to existing model (for evaluation/export only)
        verbose: Enable verbose logging

    Returns:
        Dictionary with results from each step
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)

    if config is None:
        config = get_default_config()

    if steps is None:
        steps = ["train", "evaluate", "export"]

    logger_instance.info("="* 70)
    logger_instance.info("Starting Re-identification Pipeline")
    logger_instance.info("=" * 70)
    logger_instance.info(f"Steps: {steps}")
    logger_instance.info(f"Dataset source: {config.dataset.source}")
    logger_instance.info(f"Backbone: {config.backbone.name}")

    results: dict[str, Any] = {}

    # Step 1: Training
    if "train" in steps:
        logger_instance.info("\n" + "=" * 70)
        logger_instance.info("STEP 1: Training")
        logger_instance.info("=" * 70)

        training_results = run_training(config=config, verbose=verbose)
        results["training"] = training_results

        # Use best model for evaluation
        if model_path is None:
            model_path = config.training.save_dir / "best_model.pt"

        logger_instance.info(f"Training completed. Best model: {model_path}")

    # Step 2: Evaluation
    if "evaluate" in steps:
        logger_instance.info("\n" + "=" * 70)
        logger_instance.info("STEP 2: Evaluation")
        logger_instance.info("=" * 70)

        if model_path is None:
            model_path = config.training.save_dir / "best_model.pt"
            logger_instance.info(f"Using model: {model_path}")

        evaluation_results = run_evaluation(config=config, model_path=model_path, verbose=verbose)
        results["evaluation"] = evaluation_results

        logger_instance.info("Evaluation completed")

    # Step 3: Export
    if "export" in steps:
        logger_instance.info("\n" + "=" * 70)
        logger_instance.info("STEP 3: Export Results")
        logger_instance.info("=" * 70)

        export_targets = []
        if config.evaluation.save_embeddings or config.evaluation.save_predictions:
            export_targets.append("disk")
        if config.evaluation.add_to_fiftyone:
            export_targets.append("fiftyone")

        if not export_targets:
            logger_instance.info("No export targets configured, skipping export")
        else:
            export_results = run_export(
                config=config, model_path=model_path, export_targets=export_targets, verbose=verbose
            )
            results["export"] = export_results

            logger_instance.info("Export completed")

    logger_instance.info("\n" + "=" * 70)
    logger_instance.info("Pipeline completed successfully!")
    logger_instance.info("=" * 70)

    if "training" in results:
        logger_instance.info(f"Best training mAP: {results['training'].get('best_val_map', 'N/A'):.4f}")
    if "evaluation" in results:
        logger_instance.info(f"Test mAP: {results['evaluation'].get('map', 'N/A'):.4f}")

    return results


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run jaguar re-identification pipeline", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Pipeline args
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["train", "evaluate", "export"],
        default=["train", "evaluate", "export"],
        help="Pipeline steps to run",
    )
    parser.add_argument("--model-path", type=Path, help="Path to existing model (for eval/export only)")

    # Dataset args
    parser.add_argument("--dataset-source", type=str, choices=["disk", "huggingface", "fiftyone"], default="fiftyone")
    parser.add_argument("--data-dir", type=Path, help="Data directory (for disk source)")
    parser.add_argument("--hf-repo", type=str, help="HuggingFace repo")
    parser.add_argument("--fo-dataset", type=str, default="JID_Master_Dataset", help="FiftyOne dataset name")

    # Training args
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    # Model args
    parser.add_argument("--backbone", type=str, default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)

    # Wandb args
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="jaguar-reidentification")
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-run-name", type=str, help="Wandb run name")

    # Export args
    parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings to disk")
    parser.add_argument("--add-to-fiftyone", action="store_true", help="Add results to FiftyOne dataset")

    # Output args
    parser.add_argument("--save-dir", type=Path, default=Path("data/models/reidentification"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/reidentification"))

    # Runtime args
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Build configuration
    config = get_default_config()

    # Dataset config
    config.dataset.source = args.dataset_source
    if args.data_dir:
        config.dataset.data_dir = args.data_dir
    if args.hf_repo:
        config.dataset.hf_repo = args.hf_repo
    config.dataset.fo_dataset_name = args.fo_dataset

    # Training config
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.num_epochs
    config.training.learning_rate = args.learning_rate
    config.training.save_dir = args.save_dir

    # Model config
    config.backbone.name = args.backbone
    config.model.embedding_dim = args.embedding_dim
    config.model.hidden_dim = args.hidden_dim

    # Wandb config
    config.wandb.enabled = args.wandb
    config.wandb.project = args.wandb_project
    if args.wandb_entity:
        config.wandb.entity = args.wandb_entity
    if args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name

    # Evaluation/Export config
    config.evaluation.save_embeddings = args.save_embeddings
    config.evaluation.add_to_fiftyone = args.add_to_fiftyone
    config.evaluation.output_dir = args.output_dir

    # Runtime
    config.verbose = args.verbose
    config.seed = args.seed

    # Set random seeds
    import random

    import numpy as np
    import torch

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    try:
        run_pipeline(config=config, steps=args.steps, model_path=args.model_path, verbose=args.verbose)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
