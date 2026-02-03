"""Segmentation pipeline module.

Orchestrates segmentation using SAM3, performs filtering of results,
and cleaning of the dataset.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fiftyone as fo
from fiftyone import ViewField as F

from jaguars.common.config import JID_MASTER_DATASET
from jaguars.common.logging_utils import setup_logger
from jaguars.segmentation.SAM3 import run_processing as run_sam3

MODULE_NAME = "ingestion.processing.segmentation"
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


def filter_by_count(
    dataset: fo.Dataset,
    segmentation_field: str,
    expected_count: int = 1,
    tag: str = "filter_count",
) -> int:
    """Tags samples that do not have the expected number of detections."""
    # Find samples where number of detections != expected_count
    # Handle None field (0 detections)

    # Expression: field is null OR detections list len check.
    # Note: F(field) returns the field value.
    # If it's Detections, it doesn't have .detections attribute in the ViewField expression builder same as Python object.
    # In FiftyOne ViewField expressions, you access embedded fields directly or use matching.
    # Correct way to assume Detections field has 'detections' list inside.
    # But F("field") refers to the object.

    # View of samples with WRONG count
    view = dataset.match((F(segmentation_field) is None) | (F(f"{segmentation_field}.detections").length() != expected_count))

    count = len(view)
    if count > 0:
        view.tag_samples(tag)
        logger.info("Tagged %d samples with '%s' (detection count != %d)", count, tag, expected_count)

    return count


def filter_unexpected_segmentations(
    dataset: fo.Dataset,
    segmentation_field: str,
    min_confidence: float = 0.5,
    min_area_rel: float = 0.01,
    max_area_rel: float = 1.0,
    tag: str = "filter_quality",
) -> int:
    """Tags samples with segmentation quality issues (confidence, area)."""
    # We want to identify samples where ANY detection has issues?
    # Or samples where the "best" detection has issues?
    # Assuming we filtered for count=1 already, we can check the single detection.
    # But to be robust, let's tag samples where *any* detection fails criteria
    # OR if there are no detections (though count filter handles that).

    # For ingestion of singular jaguars, we want high quality.

    # Filter 1: Low Confidence (already handled by SAM3 threshold usually, but doing explicit check)
    # Filter 2: Area too small or too large

    # filter(F("confidence") < min) works on the list 'detections'

    # We match samples that HAVE detections BUT they satisfy bad conditions
    view = dataset.exists(segmentation_field).match(
        F(f"{segmentation_field}.detections")
        .filter(
            (F("confidence") < min_confidence)
            | ((F("bounding_box")[2] * F("bounding_box")[3]) < min_area_rel)
            | ((F("bounding_box")[2] * F("bounding_box")[3]) > max_area_rel)
        )
        .length()
        > 0
    )

    count = len(view)
    if count > 0:
        view.tag_samples(tag)
        logger.info("Tagged %d samples with '%s' (quality issues)", count, tag)

    return count


def cleanup_dataset(dataset: fo.Dataset, tags: list[str]) -> int:
    """Deletes samples with specified tags."""
    view = dataset.match_tags(tags)
    count = len(view)
    if count > 0:
        logger.info("Deleting %d samples with tags: %s", count, tags)
        dataset.delete_samples(view)
    return count


def run_processing(
    dataset_name: str = JID_MASTER_DATASET,
    segmentation_field: str = "sam3_segmentations",
    prompt: str = "jaguar",
    sam3_threshold: float = 0.5,
    sam3_mask_threshold: float = 0.5,
    min_area_rel: float = 0.01,
    max_area_rel: float = 1.0,
    inspect_app: bool = False,
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> fo.Dataset | None:
    """Orchestrates segmentation and filtering.

    1. Runs SAM3 segmentation.
    2. Filters samples with != 1 detection.
    3. Filters samples with poor segmentation quality.
    4. Optionally launches App for inspection.
    5. Deletes filtered samples.
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)
    logger_instance.info("Starting segmentation processing pipeline for: %s", dataset_name)

    validate_resources(dataset_name)

    if dry_run:
        logger_instance.info("DRY RUN: Would segment and filter dataset '%s'", dataset_name)
        return None

    # 1. Run SAM3
    logger_instance.info("Running SAM3...")
    dataset = run_sam3(
        dataset_name=dataset_name,
        prompt=prompt,
        threshold=sam3_threshold,
        mask_threshold=sam3_mask_threshold,
        output_field=segmentation_field,
        verbose=verbose,
    )
    if dataset is None:
        raise RuntimeError("SAM3 processing returned None")

    # 2. Filter by count (Strictly 1 jaguar)
    logger_instance.info("Filtering by detection count...")
    tag_count = "filter_count"
    n_count = filter_by_count(dataset, segmentation_field, expected_count=1, tag=tag_count)

    # 3. Filter by quality
    logger_instance.info("Filtering by quality...")
    tag_quality = "filter_quality"
    n_quality = filter_unexpected_segmentations(
        dataset,
        segmentation_field,
        min_confidence=sam3_threshold,  # Consistency
        min_area_rel=min_area_rel,
        max_area_rel=max_area_rel,
        tag=tag_quality,
    )

    # 4. Optional Manual Inspection
    filter_tags = [tag_count, tag_quality]
    filtered_view = dataset.match_tags(filter_tags)

    if inspect_app and len(filtered_view) > 0:
        logger_instance.info("Launching FiftyOne App for manual inspection of %d filtered samples...", len(filtered_view))
        logger_instance.info("Press Enter in terminal to continue to deletion...")
        fo.launch_app(view=filtered_view)
        input("Press Enter to continue...")
        # session.close() # Optional

    # 5. Cleanup
    logger_instance.info("Cleaning up filtered samples...")
    total_deleted = cleanup_dataset(dataset, filter_tags)

    if summary_location:
        summary = {
            "dataset": dataset_name,
            "sam3_field": segmentation_field,
            "filtered_count_mismatch": n_count,
            "filtered_quality_issues": n_quality,
            "total_deleted": total_deleted,
            "final_sample_count": len(dataset),
        }
        write_summary(summary, summary_location)

    logger_instance.info("Segmentation pipeline complete.")
    return dataset


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="Run segmentation ingestion pipeline")
    parser.add_argument("--dataset", type=str, default=JID_MASTER_DATASET)
    parser.add_argument("--field", type=str, default="sam3_segmentations")
    parser.add_argument("--prompt", type=str, default="jaguar")

    parser.add_argument("--sam3-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-mask-threshold", type=float, default=0.5)

    parser.add_argument("--min-area", type=float, default=0.01)
    parser.add_argument("--max-area", type=float, default=1.0)

    parser.add_argument("--inspect", action="store_true", help="Wait for manual inspection in App")
    parser.add_argument("--summary-location", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    try:
        run_processing(
            dataset_name=args.dataset,
            segmentation_field=args.field,
            prompt=args.prompt,
            sam3_threshold=args.sam3_threshold,
            sam3_mask_threshold=args.sam3_mask_threshold,
            min_area_rel=args.min_area,
            max_area_rel=args.max_area,
            inspect_app=args.inspect,
            summary_location=args.summary_location,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error("Segmentation pipeline failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
