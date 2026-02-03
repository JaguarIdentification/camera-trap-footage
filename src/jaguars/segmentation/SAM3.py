r"""SAM3 Segmentation Module.

Runs SAM3 segmentation on images in a FiftyOne grouped dataset.
Only processes the "image" slice of the dataset.

Usage:
    python src/jaguars/segmentation/SAM3.py \\
        --dataset JID_Master_Dataset \\
        --prompt "jaguar" \\
        --threshold 0.5 \\
        --mask-threshold 0.5

"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fiftyone as fo
import fiftyone.zoo as foz
import torch
from fiftyone import ViewField as F

from jaguars.common.config import JID_MASTER_DATASET
from jaguars.common.fiftyone_utils import get_or_create_dataset
from jaguars.common.logging_utils import setup_logger

MODULE_NAME = "segmentation.SAM3"
logger = setup_logger(MODULE_NAME)


def validate_resources(dataset_name: str) -> None:
    """Checks if dataset exists."""
    if dataset_name not in fo.list_datasets():
        raise ValueError(f"Dataset '{dataset_name}' does not exist")


def write_summary(summary_data: dict[str, Any], summary_path: Path) -> None:
    """Generates summary report."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)


def run_processing(
    dataset_name: str = JID_MASTER_DATASET,
    prompt: str = "jaguar",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    output_field: str = "sam3_segmentations",
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    device: str | None = None,
) -> fo.Dataset | None:
    """Core Logic for SAM3 segmentation on images.

    Args:
        dataset_name: Name of the FiftyOne dataset.
        prompt: Text prompt for concept segmentation (default: "jaguar").
        threshold: Confidence threshold (default: 0.5).
        mask_threshold: Mask threshold (default: 0.5).
        output_field: Field name to store segmentations (default: "sam3_segmentations").
        summary_location: Path to write summary JSON.
        dry_run: If True, only logs what would be done.
        verbose: Enable detailed logging.
        device: Device to run model on ("cuda", "mps", "cpu"). If None, auto-detect.

    Returns:
        Updated FiftyOne dataset.
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)
    logger_instance.info("Starting SAM3 segmentation for dataset: %s", dataset_name)
    logger_instance.info("Parameters: prompt='%s', threshold=%s, mask_threshold=%s", prompt, threshold, mask_threshold)

    validate_resources(dataset_name)

    if dry_run:
        logger_instance.info("DRY RUN: Would run SAM3 on images in '%s' -> field '%s'", dataset_name, output_field)
        return None

    dataset = get_or_create_dataset(dataset_name)

    # We only want to process the "image" slice of the grouped dataset
    # If it's not a grouped dataset, we assume all samples are images
    if dataset.group_field:
        logger_instance.info("Dataset is grouped. Selecting 'image' slice.")
        view = dataset.select_group_slices("image")
    else:
        logger_instance.info("Dataset is not grouped. Processing all samples.")
        view = dataset

    num_samples = len(view)
    logger_instance.info("Found %d samples to process", num_samples)

    if num_samples == 0:
        logger_instance.warning("No samples found in the image slice. Exiting.")
        return dataset

    # Register SAM3 zoo model
    logger_instance.info("Registering SAM3 zoo model...")
    foz.register_zoo_model_source("https://github.com/harpreetsahota204/sam3_images")

    # Load model
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    logger_instance.info("Loading SAM3 model on device: %s", device)
    model = foz.load_zoo_model("facebook/sam3", device=device)

    # Configure model parameters
    model.operation = "concept_segmentation"
    model.prompt = prompt
    model.threshold = threshold
    model.mask_threshold = mask_threshold

    # Apply model
    logger_instance.info("Applying model to %d samples...", num_samples)
    view.apply_model(
        model,
        label_field=output_field,
        batch_size=16,
        num_workers=0,  # > 0 fails
    )

    logger_instance.info("Saving view...")
    view.save()

    # Calculate summary
    # Count how many samples have detections
    detections = view.filter_labels(output_field, F("confidence") >= threshold)
    num_detections = len(detections)

    if summary_location:
        summary_data = {
            "dataset_name": dataset_name,
            "prompt": prompt,
            "threshold": threshold,
            "mask_threshold": mask_threshold,
            "total_samples": num_samples,
            "samples_with_detections": num_detections,
            "output_field": output_field,
        }
        write_summary(summary_data, summary_location)
        logger_instance.info("Summary written to %s", summary_location)

    logger_instance.info("SAM3 segmentation output saved to field '%s'", output_field)
    return dataset


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run SAM3 segmentation on images in a FiftyOne dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=JID_MASTER_DATASET,
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="jaguar",
        help="Text prompt for segmentation (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--output-field",
        type=str,
        default="sam3_segmentations",
        help="Field name for results (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-location",
        type=Path,
        default=None,
        help="Path to write summary JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    try:
        run_processing(
            dataset_name=args.dataset,
            prompt=args.prompt,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            output_field=args.output_field,
            summary_location=args.summary_location,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error("Failed to run SAM3 segmentation: %s", e, exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
