"""Export module.

Exports dataset to data directory and to huggingface.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import fiftyone as fo
from fiftyone import ViewField as F

from jaguars.common.config import DEFAULT_GROUP_SLICE, JID_MASTER_DATASET
from jaguars.common.logging_utils import setup_logger

MODULE_NAME = "ingestion.export.export"
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


def _get_image_view(dataset: fo.Dataset) -> fo.DatasetView:
    if dataset.group_field is None:
        return dataset
    return dataset.select_group_slices(DEFAULT_GROUP_SLICE)


def _get_variant_view(
    dataset: fo.Dataset,
    variant: str,
    segmentation_field: str,
    dedup_field: str,
) -> fo.DatasetView:
    """Get dataset view for a specific variant.

    Supported variants:
    - master: Full dataset with all samples
    - segmented_deduplicated: Segmented samples, duplicates removed
    - segmented: Segmented samples, including duplicates
    - not_segmented_deduplicated: Non-segmented samples, duplicates removed
    - not_segmented: Non-segmented samples, including duplicates
    - full: Alias for master (backwards compatibility)
    - deduplicated: All samples with duplicates removed (backwards compatibility)
    - segmented_deduplicated: Same as above (backwards compatibility)
    """
    if variant in ("full", "master"):
        return dataset

    view = _get_image_view(dataset)

    # Apply segmentation filter
    if variant == "segmented_deduplicated" or variant == "segmented":
        if segmentation_field in view.get_field_schema():
            view = view.exists(segmentation_field)
        else:
            logger.warning("Segmentation field '%s' not found; returning empty view", segmentation_field)
            return view.limit(0)
    elif variant == "not_segmented_deduplicated" or variant == "not_segmented":
        if segmentation_field in view.get_field_schema():
            view = view.match((F(segmentation_field) == None) | (~F(segmentation_field).exists()))
        else:
            # If segmentation field doesn't exist, all samples are "not segmented"
            pass
    elif variant == "deduplicated":
        # Just filter duplicates, no segmentation filter
        pass

    # Apply deduplication filter
    if "deduplicated" in variant:
        if dedup_field in view.get_field_schema():
            view = view.match(F(dedup_field) != True)
        else:
            logger.warning("Dedup field '%s' not found; returning unfiltered view", dedup_field)

    return view


def _export_variant_to_disk(view: fo.DatasetView, export_dir: Path, overwrite: bool = True, include_fields: list[str] | None = None) -> None:
    """Export variant to disk.

    Args:
        view: Dataset view to export
        export_dir: Directory to export to
        overwrite: Whether to overwrite existing data
        include_fields: List of field names to include (None = all fields)
    """
    export_dir.mkdir(parents=True, exist_ok=True)

    # Ensure jaguar_id is always included if it exists
    if include_fields is None:
        # Export all fields
        view.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.FiftyOneDataset,
            overwrite=overwrite,
        )
    else:
        # Ensure jaguar_id and ground_truth are included
        fields_to_export = list(include_fields)
        if "jaguar_id" in view.get_field_schema() and "jaguar_id" not in fields_to_export:
            fields_to_export.append("jaguar_id")
        if "ground_truth" in view.get_field_schema() and "ground_truth" not in fields_to_export:
            fields_to_export.append("ground_truth")

        view.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.FiftyOneDataset,
            overwrite=overwrite,
            include_fields=fields_to_export,
        )


def _export_variant_to_fiftyone(
    view: fo.DatasetView,
    dataset_name: str,
    overwrite: bool = True,
) -> None:
    if dataset_name in fo.list_datasets():
        if overwrite:
            fo.delete_dataset(dataset_name)
        else:
            raise ValueError(f"Dataset '{dataset_name}' already exists")
    view.clone(name=dataset_name)


def _export_variant_to_huggingface(
    view: fo.DatasetView,
    repo_name: str,
) -> None:
    from fiftyone.utils.huggingface import push_to_hub
    from huggingface_hub import login

    login()
    push_to_hub(
        view,
        repo_name=repo_name,
        dataset_type=fo.types.FiftyOneDataset,
        private=True,
    )


def run_processing(
    dataset_name: str = JID_MASTER_DATASET,
    variants: Iterable[str] | None = None,
    export_targets: Iterable[str] | None = None,
    huggingface_repo: str | None = None,
    export_base_dir: Path | None = None,
    segmentation_field: str = "sam3_segmentations",
    dedup_field: str = "is_duplicate",
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> fo.Dataset | None:
    """Core logic for exporting dataset variants.

    Args:
        dataset_name: Name of the FiftyOne dataset
        variants: Dataset variants to export. Options:
            - master: Full dataset with all samples
            - segmented_deduplicated: Segmented samples, duplicates removed
            - segmented: Segmented samples, including duplicates
            - not_segmented_deduplicated: Non-segmented samples, duplicates removed
            - not_segmented: Non-segmented samples, including duplicates
            - deduplicated: All samples with duplicates removed
            - full: Alias for master (backwards compatibility)
        export_targets: Export targets: disk, huggingface, fiftyone
        huggingface_repo: Base Huggingface repository name (without username)
        export_base_dir: Base directory for disk exports
        segmentation_field: Field name for segmentation detections
        dedup_field: Field name marking duplicates
        summary_location: Path to write summary JSON
        dry_run: If True, only log what would be done
        verbose: Enable verbose logging

    Returns:
        Exported FiftyOne dataset or None if dry_run
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)
    logger_instance.info("Starting export for dataset: %s", dataset_name)

    validate_resources(dataset_name)

    if dry_run:
        logger_instance.info("DRY RUN: Would export dataset '%s'", dataset_name)
        return None

    dataset = fo.load_dataset(dataset_name)

    if variants is None:
        # Default: export all 5 variations
        variants = [
            "master",
            "segmented_deduplicated",
            "segmented",
            "not_segmented_deduplicated",
            "not_segmented",
        ]

    if export_targets is None:
        export_targets = ["disk"]

    export_base_dir = export_base_dir or Path("data/intermediate/v1") / dataset_name

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "variants": {},
        "export_targets": list(export_targets),
    }

    for variant in variants:
        view = _get_variant_view(dataset, variant, segmentation_field, dedup_field)
        variant_name = f"{dataset_name}_{variant}"
        summary["variants"][variant] = {"sample_count": len(view)}

        if dry_run:
            logger_instance.info("DRY RUN: Would export variant '%s' with %d samples", variant, len(view))
            continue

        logger_instance.info("Exporting variant '%s' (%d samples)", variant, len(view))

        if "disk" in export_targets:
            export_dir = export_base_dir / variant
            logger_instance.info("  -> Exporting to disk at %s", export_dir)
            _export_variant_to_disk(view, export_dir)

        if "fiftyone" in export_targets:
            logger_instance.info("  -> Exporting as FiftyOne dataset '%s'", variant_name)
            _export_variant_to_fiftyone(view, variant_name)

        if "huggingface" in export_targets:
            if not huggingface_repo:
                raise ValueError("huggingface_repo must be provided when exporting to huggingface")
            repo_suffix = huggingface_repo if variant == "master" else f"{huggingface_repo}-{variant}"
            logger_instance.info("  -> Exporting to Huggingface repo '%s'", repo_suffix)
            _export_variant_to_huggingface(view, repo_suffix)

    if summary_location:
        write_summary(summary, summary_location)
        logger_instance.info("Summary written to %s", summary_location)

    logger_instance.info("Export completed successfully.")
    return dataset


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="Export a FiftyOne dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=JID_MASTER_DATASET,
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variants to export: master, segmented_deduplicated, segmented, not_segmented_deduplicated, not_segmented, deduplicated, full",
    )
    parser.add_argument(
        "--export-targets",
        nargs="+",
        default=None,
        help="Export targets: disk, huggingface, fiftyone",
    )
    parser.add_argument(
        "--huggingface-repo",
        type=str,
        default=None,
        help="Huggingface repository base name (without username)",
    )
    parser.add_argument(
        "--export-base-dir",
        type=Path,
        default=None,
        help="Base directory for disk exports",
    )
    parser.add_argument(
        "--segmentation-field",
        type=str,
        default="sam3_segmentations",
        help="Field name for segmentation detections",
    )
    parser.add_argument(
        "--dedup-field",
        type=str,
        default="is_duplicate",
        help="Field name for duplicate flag",
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
            variants=args.variants,
            export_targets=args.export_targets,
            huggingface_repo=args.huggingface_repo,
            export_base_dir=args.export_base_dir,
            segmentation_field=args.segmentation_field,
            dedup_field=args.dedup_field,
            summary_location=args.summary_location,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error("Error during export: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
