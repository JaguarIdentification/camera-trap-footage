"""Metadata computation module.

Computes metadata (width, height, num_channels, etc.) for all samples in the dataset.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fiftyone as fo

from jaguars.common.config import JID_MASTER_DATASET
from jaguars.common.logging_utils import setup_logger

MODULE_NAME = "ingestion.processing.compute_metadata"
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
    overwrite: bool = False,
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> fo.Dataset | None:
    """Core Logic for computing metadata.

    Args:
        dataset_name: Name of the FiftyOne dataset
        overwrite: Whether to overwrite existing metadata
        summary_location: Path to write summary JSON
        dry_run: If True, only log what would be done
        verbose: Enable detailed logging

    Returns:
        Updated FiftyOne dataset or None if dry_run
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)
    logger_instance.info("Starting metadata computation for dataset: %s", dataset_name)

    validate_resources(dataset_name)

    if dry_run:
        logger_instance.info("DRY RUN: Would compute metadata for dataset '%s' (overwrite=%s)", dataset_name, overwrite)
        return None

    dataset = fo.load_dataset(dataset_name)

    logger_instance.info("Computing metadata (overwrite=%s)...", overwrite)
    dataset.compute_metadata(overwrite=overwrite)

    if summary_location:
        summary = {
            "dataset": dataset_name,
            "total_samples": len(dataset),
            "overwrite": overwrite,
        }
        write_summary(summary, summary_location)
        logger_instance.info("Summary written to %s", summary_location)

    logger_instance.info("Metadata computation completed successfully.")
    return dataset


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compute metadata for a FiftyOne dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=JID_MASTER_DATASET,
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metadata",
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
            overwrite=args.overwrite,
            summary_location=args.summary_location,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error("Error during metadata computation: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
