"""Export module for re-identification results.

Exports embeddings and predictions to:
- FiftyOne datasets
- Disk (numpy arrays, JSON)
- HuggingFace datasets
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jaguars.common.logging_utils import setup_logger
from jaguars.reidentification.backbone import get_backbone
from jaguars.reidentification.config import ReidentificationConfig, get_default_config, load_config_from_dict
from jaguars.reidentification.data_loader import load_dataset
from jaguars.reidentification.model import build_model

MODULE_NAME = "reidentification.export"
logger = setup_logger(MODULE_NAME)


def validate_resources(config: ReidentificationConfig, model_path: Path | None = None) -> None:
    """Validate resources.

    Args:
        config: Configuration
        model_path: Optional path to model checkpoint

    Raises:
        ValueError: If resources are missing
    """
    # Check dataset
    if config.dataset.source == "fiftyone":
        try:
            import fiftyone as fo

            if not fo.dataset_exists(config.dataset.fo_dataset_name):
                raise ValueError(f"FiftyOne dataset '{config.dataset.fo_dataset_name}' does not exist")
        except ImportError:
            raise ImportError("fiftyone required for FiftyOne datasets")

    # Check model
    if model_path is not None and not model_path.exists():
        raise ValueError(f"Model checkpoint not found: {model_path}")

    logger.info("Resource validation passed")


def write_summary(summary_data: dict[str, Any], summary_location: Path, to_wandb: bool = False) -> None:
    """Write export summary.

    Args:
        summary_data: Summary data
        summary_location: Path to save summary
        to_wandb: Whether to log to wandb
    """
    summary_location.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_location, "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Summary written to {summary_location}")

    if to_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(summary_data)
        except ImportError:
            pass


def export_to_disk(
    embeddings: np.ndarray,
    image_paths: list[str],
    labels: list[str],
    output_dir: Path,
    predictions: dict[str, Any] | None = None,
) -> None:
    """Export embeddings and predictions to disk.

    Args:
        embeddings: Embeddings array
        image_paths: List of image paths
        labels: List of labels
        output_dir: Output directory
        predictions: Optional predictions dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    emb_path = output_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info(f"Saved embeddings to {emb_path}")

    # Save metadata
    metadata: dict[str, Any] = {"image_paths": image_paths, "labels": labels, "embedding_shape": list(embeddings.shape)}

    if predictions is not None:
        metadata["predictions"] = predictions

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {meta_path}")


def export_to_fiftyone(
    embeddings: np.ndarray,
    image_paths: list[str],
    dataset_name: str,
    embeddings_field: str = "reid_embeddings",
    predictions_field: str = "reid_predictions",
    predictions: dict[str, Any] | None = None,
) -> None:
    """Export embeddings to FiftyOne dataset.

    Args:
        embeddings: Embeddings array
        image_paths: Image paths
        dataset_name: FiftyOne dataset name
        embeddings_field: Field name for embeddings
        predictions_field: Field name for predictions
        predictions: Optional predictions dictionary
    """
    try:
        import fiftyone as fo
    except ImportError:
        raise ImportError("fiftyone required for FiftyOne export")

    logger.info(f"Exporting to FiftyOne dataset: {dataset_name}")

    # Load dataset
    if not fo.dataset_exists(dataset_name):
        raise ValueError(f"FiftyOne dataset '{dataset_name}' does not exist")

    dataset = fo.load_dataset(dataset_name)

    # Add embeddings to samples
    num_updated = 0
    for img_path, embedding in zip(image_paths, embeddings):
        # Find matching sample
        matching = dataset.match(fo.ViewField("filepath") == img_path)
        if len(matching) == 0:
            logger.warning(f"No sample found for {img_path}")
            continue

        sample = matching.first()
        sample[embeddings_field] = embedding.tolist()

        if predictions is not None and img_path in predictions:
            sample[predictions_field] = predictions[img_path]

        sample.save()
        num_updated += 1

    logger.info(f"Updated {num_updated}/{len(image_paths)} samples in FiftyOne")


def export_to_huggingface(
    embeddings: np.ndarray, image_paths: list[str], labels: list[str], repo_name: str, private: bool = True
) -> None:
    """Export to HuggingFace dataset.

    Args:
        embeddings: Embeddings array
        image_paths: Image paths
        labels: Labels
        repo_name: HuggingFace repo name
        private: Whether to make repo private
    """
    try:
        from datasets import Dataset
        from huggingface_hub import login
    except ImportError:
        raise ImportError("datasets and huggingface_hub required for HuggingFace export")

    logger.info(f"Exporting to HuggingFace: {repo_name}")

    # Login
    login()

    # Create dataset
    data = {"image_path": image_paths, "label": labels, "embedding": embeddings.tolist()}

    hf_dataset = Dataset.from_dict(data)

    # Push to hub
    hf_dataset.push_to_hub(repo_name, private=private)

    logger.info(f"Exported to HuggingFace: {repo_name}")


def run_processing(
    config: ReidentificationConfig | None = None,
    config_dict: dict[str, Any] | None = None,
    model_path: Path | None = None,
    output_dir: Path | None = None,
    export_targets: list[str] | None = None,
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Core export logic.

    Args:
        config: Configuration
        config_dict: Configuration dictionary
        model_path: Path to model checkpoint
        output_dir: Output directory for disk export
        export_targets: List of export targets (disk, fiftyone, huggingface)
        summary_location: Path to save summary
        dry_run: Only validate
        verbose: Verbose logging

    Returns:
        Export results dictionary
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)

    # Load config
    if config is None:
        if config_dict is not None:
            config = load_config_from_dict(config_dict)
        else:
            config = get_default_config()

    if dry_run:
        config.dry_run = True

    if export_targets is None:
        export_targets = ["disk"]

    if output_dir is None:
        output_dir = config.evaluation.output_dir

    logger_instance.info("Starting export...")
    logger_instance.info(f"Export targets: {export_targets}")

    # Validate
    validate_resources(config, model_path)

    if dry_run:
        logger_instance.info("DRY RUN: Would export embeddings and predictions")
        return {"status": "dry_run"}

    # Set device
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load dataset
    logger_instance.info("Loading dataset...")
    dataset_metadata = load_dataset(config.dataset)

    # Extract embeddings if needed
    if dataset_metadata.embeddings is None:
        logger_instance.info("Extracting baseline embeddings...")
        backbone = get_backbone(config.backbone, device)
        backbone.eval()
        baseline_embeddings = backbone.extract_embeddings(dataset_metadata.image_paths, device)
    else:
        baseline_embeddings = dataset_metadata.embeddings

    # Load model and get fine-tuned embeddings
    finetuned_embeddings = None
    if model_path is not None:
        logger_instance.info(f"Loading model from {model_path}")
        input_dim = baseline_embeddings.shape[1]
        model = build_model(input_dim, dataset_metadata.num_classes, config.model)
        model.to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger_instance.info("Extracting fine-tuned embeddings...")
        with torch.no_grad():
            embeddings_tensor = torch.FloatTensor(baseline_embeddings).to(device)
            finetuned_embeddings = model.get_embeddings(embeddings_tensor).cpu().numpy()

    # Use fine-tuned if available, otherwise baseline
    embeddings = finetuned_embeddings if finetuned_embeddings is not None else baseline_embeddings

    logger_instance.info(f"Embeddings shape: {embeddings.shape}")

    # Export to targets
    results = {"status": "completed", "num_samples": len(embeddings), "embedding_dim": embeddings.shape[1]}

    if "disk" in export_targets:
        logger_instance.info("Exporting to disk...")
        export_to_disk(embeddings, dataset_metadata.image_paths, dataset_metadata.labels, output_dir)
        results["disk_export"] = str(output_dir)

    if "fiftyone" in export_targets:
        logger_instance.info("Exporting to FiftyOne...")
        export_to_fiftyone(
            embeddings,
            dataset_metadata.image_paths,
            config.dataset.fo_dataset_name,
            config.evaluation.fo_embeddings_field,
            config.evaluation.fo_predictions_field,
        )
        results["fiftyone_export"] = config.dataset.fo_dataset_name

    if "huggingface" in export_targets:
        if config.dataset.hf_repo is None:
            logger_instance.warning("HuggingFace repo not specified, skipping HF export")
        else:
            logger_instance.info("Exporting to HuggingFace...")
            export_to_huggingface(embeddings, dataset_metadata.image_paths, dataset_metadata.labels, config.dataset.hf_repo)
            results["huggingface_export"] = config.dataset.hf_repo

    # Write summary
    if summary_location is not None:
        write_summary(results, summary_location, config.wandb.enabled)

    logger_instance.info("Export completed!")

    return results


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="Export re-identification embeddings and predictions", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model args
    parser.add_argument("--model-path", type=Path, help="Path to model checkpoint (optional)")

    # Dataset args
    parser.add_argument("--dataset-source", type=str, choices=["disk", "huggingface", "fiftyone"], default="fiftyone")
    parser.add_argument("--fo-dataset", type=str, default="JID_Master_Dataset")

    # Export args
    parser.add_argument(
        "--export-to", nargs="+", choices=["disk", "fiftyone", "huggingface"], default=["disk"], help="Export targets"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/reidentification"))
    parser.add_argument("--hf-repo", type=str, help="HuggingFace repo for export")

    # Output args
    parser.add_argument("--summary-location", type=Path)

    # Runtime args
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Build config
    config = get_default_config()
    config.dataset.source = args.dataset_source
    config.dataset.fo_dataset_name = args.fo_dataset
    if args.hf_repo:
        config.dataset.hf_repo = args.hf_repo

    try:
        run_processing(
            config=config,
            model_path=args.model_path,
            output_dir=args.output_dir,
            export_targets=args.export_to,
            summary_location=args.summary_location,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Error during export: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
