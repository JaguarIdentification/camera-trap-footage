"""Dataset deduplication for jaguar ingestion.

Detects near-duplicate images using either perceptual hashing or
FiftyOne model embeddings. Duplicates are flagged (not deleted).

Key behaviors:
- Operates on full images (no segmentation/masks).
- Flags duplicates via fields + tag on samples.
- Supports grouped datasets (operates on image slice only).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from jaguars.common.config import DEFAULT_GROUP_SLICE, JID_MASTER_DATASET
from jaguars.common.fiftyone_utils import get_or_create_dataset
from jaguars.common.logging_utils import setup_logger

MODULE_NAME = "ingestion.processing.deduplicate"
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


def _iter_image_samples(dataset: fo.Dataset) -> fo.DatasetView:
    if dataset.group_field is None:
        return dataset
    return dataset.select_group_slices(DEFAULT_GROUP_SLICE)


def _compute_embeddings(
    dataset: fo.Dataset,
    model_name: str,
    embeddings_field: str,
    batch_size: int = 32,
) -> None:
    """Compute embeddings if they don't exist.

    Uses batching to avoid memory issues with large datasets.
    """
    if embeddings_field not in dataset.get_field_schema():
        logger.info("Computing embeddings with model '%s' (batch_size=%d)", model_name, batch_size)
        model = foz.load_zoo_model(model_name)
        dataset.compute_embeddings(
            model,
            embeddings_field=embeddings_field,
            batch_size=batch_size,  # Process in batches to reduce memory
        )
    else:
        logger.info("Embeddings field '%s' already exists", embeddings_field)


def _unmark_all_duplicates(
    dataset: fo.Dataset,
    is_duplicate_field: str,
    duplicate_of_field: str,
    duplicate_similarity_field: str,
    duplicate_tag: str,
) -> None:
    """Clear all duplicate markings before reprocessing."""
    logger.info("Clearing previous duplicate markings...")

    # Clear fields
    if is_duplicate_field in dataset.get_field_schema():
        dataset.set_values(is_duplicate_field, [False] * len(dataset))
    if duplicate_of_field in dataset.get_field_schema():
        dataset.set_values(duplicate_of_field, [None] * len(dataset))
    if duplicate_similarity_field in dataset.get_field_schema():
        dataset.set_values(duplicate_similarity_field, [None] * len(dataset))

    # Remove duplicate tag
    for sample in dataset.match_tags(duplicate_tag):
        sample.tags = [tag for tag in sample.tags if tag != duplicate_tag]
        sample.save()

    logger.info("Previous duplicate markings cleared")


def run_processing(
    dataset_name: str = JID_MASTER_DATASET,
    similarity_threshold: float = 0.25,
    model_name: str = "hf-hub:BVRA/MegaDescriptor-L-384",
    embeddings_field: str | None = None,
    is_duplicate_field: str = "is_duplicate",
    duplicate_of_field: str = "duplicate_of",
    duplicate_similarity_field: str = "duplicate_similarity",
    duplicate_tag: str = "duplicate",
    compute_similarity_index: bool = False,
    detect_leaky_splits: bool = False,
    split_field: str = "split",
    brain_key_prefix: str = "dedup",
    inspect_app: bool = False,
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    batch_size: int = 16,  # Reduced default for memory safety
) -> fo.Dataset | None:
    """Deduplicate dataset images using FiftyOne brain methods.

    Args:
        dataset_name: FiftyOne dataset name
        similarity_threshold: Similarity threshold in [0, 1] for near duplicates
        model_name: Model zoo name for embeddings
        embeddings_field: Optional embeddings field override
        is_duplicate_field: Field name to mark duplicates
        duplicate_of_field: Field storing canonical sample id
        duplicate_similarity_field: Field storing similarity score
        duplicate_tag: Tag applied to duplicates
        compute_similarity_index: Compute similarity index for interactive search
        detect_leaky_splits: Detect similar samples across splits
        split_field: Field name containing split assignments
        brain_key_prefix: Prefix for brain computation keys
        inspect_app: Launch FiftyOne App to manually inspect duplicates before finalizing
        summary_location: Where to write JSON summary
        dry_run: Only log actions without modifications
        verbose: Enable detailed logging
        batch_size: Batch size for embedding computation (reduce if OOM)
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)
    logger_instance.info("Starting deduplication for dataset: %s", dataset_name)

    validate_resources(dataset_name)

    if dry_run:
        logger_instance.info("DRY RUN: Would deduplicate dataset '%s'", dataset_name)
        return None

    dataset = get_or_create_dataset(dataset_name)
    view = _iter_image_samples(dataset)

    # Unmark all previous duplicates when restarting
    _unmark_all_duplicates(
        dataset,
        is_duplicate_field,
        duplicate_of_field,
        duplicate_similarity_field,
        duplicate_tag,
    )

    # Prepare fields
    if is_duplicate_field not in dataset.get_field_schema():
        dataset.add_sample_field(is_duplicate_field, fo.BooleanField)
    if duplicate_of_field not in dataset.get_field_schema():
        dataset.add_sample_field(duplicate_of_field, fo.StringField)
    if duplicate_similarity_field not in dataset.get_field_schema():
        dataset.add_sample_field(duplicate_similarity_field, fo.FloatField)

    # Compute embeddings first (for full images, not patches)
    # Default to a standard field name for full image embeddings
    if embeddings_field is None:
        # Check if embeddings_full_image already exists
        if "embeddings_full_image" in dataset.get_field_schema():
            embeddings_field = "embeddings_full_image"
        else:
            # Sanitize model name for field
            sanitized_model = model_name.split(":")[-1].replace("/", "_").replace("-", "_")
            embeddings_field = f"embeddings_{sanitized_model}"

    logger_instance.info("Using embeddings field: %s", embeddings_field)

    # Compute embeddings if they don't exist
    if embeddings_field not in dataset.get_field_schema():
        logger_instance.info("Computing embeddings with model '%s'", model_name)
        _compute_embeddings(view, model_name=model_name, embeddings_field=embeddings_field, batch_size=batch_size)
    else:
        logger_instance.info("Embeddings field '%s' already exists, using existing embeddings", embeddings_field)

    duplicate_pairs: list[tuple[str, str, float]] = []
    brain_results = {}

    if len(view) == 0:
        logger_instance.warning("No image samples to deduplicate")
    else:
        # Use FiftyOne brain to find near duplicates
        logger_instance.info("Running FiftyOne brain near duplicate detection (threshold=%.3f)", similarity_threshold)
        try:
            # Get embeddings and compute similarity manually
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            sample_ids = view.values("id")
            embeddings_list = view.values(embeddings_field)

            if not embeddings_list or len(embeddings_list) == 0:
                logger_instance.warning("No embeddings found")
                brain_results["near_duplicates_found"] = 0
            else:
                # Convert to numpy array, handling None values
                embeddings_array = np.array([e for e in embeddings_list if e is not None])

                if len(embeddings_array) > 0:
                    # Compute similarity matrix
                    sim_matrix = cosine_similarity(embeddings_array)

                    # Find duplicate pairs
                    duplicates_marked = 0
                    marked_as_dup = set()

                    for i in range(len(sample_ids)):
                        if sample_ids[i] in marked_as_dup:
                            continue
                        for j in range(i + 1, len(sample_ids)):
                            if sample_ids[j] in marked_as_dup:
                                continue
                            if sim_matrix[i, j] >= similarity_threshold:
                                # Mark j as duplicate of i
                                dup_sample = dataset[sample_ids[j]]
                                dup_sample[is_duplicate_field] = True
                                dup_sample[duplicate_of_field] = sample_ids[i]
                                dup_sample[duplicate_similarity_field] = float(sim_matrix[i, j])
                                dup_sample.tags = list(set(dup_sample.tags + [duplicate_tag]))
                                dup_sample.save()
                                marked_as_dup.add(sample_ids[j])
                                duplicates_marked += 1
                                logger_instance.debug(
                                    "Marked %s as duplicate of %s (similarity=%.4f)", sample_ids[j], sample_ids[i], sim_matrix[i, j]
                                )

                    if duplicates_marked > 0:
                        logger_instance.info("Found %d duplicate pairs", duplicates_marked)
                    else:
                        logger_instance.info("No near duplicates found")
                    brain_results["near_duplicates_found"] = duplicates_marked
                else:
                    brain_results["near_duplicates_found"] = 0

        except Exception as exc:
            logger_instance.warning("Failed to compute near duplicates: %s", exc)
            brain_results["near_duplicates_found"] = None

        # Compute similarity index for similarity search
        if compute_similarity_index:
            try:
                similarity_key = f"{brain_key_prefix}_similarity"
                logger_instance.info("Computing similarity index with key '%s'", similarity_key)
                fob.compute_similarity(
                    view,
                    embeddings=embeddings_field,
                    brain_key=similarity_key,
                    backend="sklearn",
                )
                brain_results["similarity_index"] = similarity_key
                logger_instance.info("✓ Similarity index computed")
            except Exception as exc:
                logger_instance.warning("Failed to compute similarity index: %s", exc)
                brain_results["similarity_index"] = None

        # Detect leaky splits
        if detect_leaky_splits and split_field in dataset.get_field_schema():
            try:
                logger_instance.info("Detecting leaky splits with threshold=%.3f", similarity_threshold)
                leaky_info = fob.compute_leaky_splits(
                    dataset,
                    splits=split_field,
                    embeddings=embeddings_field,
                    threshold=similarity_threshold,
                )
                brain_results["leaky_splits"] = str(leaky_info)
                logger_instance.info("✓ Leaky splits computed: %s", leaky_info)
            except Exception as exc:
                logger_instance.warning("Failed to compute leaky splits: %s", exc)
                brain_results["leaky_splits"] = None

    # Optional manual inspection before finalizing
    duplicates_count = brain_results.get("near_duplicates_found", 0) or 0
    if inspect_app and duplicates_count > 0:
        duplicates_view = dataset.match(F(is_duplicate_field) == True)
        logger_instance.info("=" * 70)
        logger_instance.info("MANUAL INSPECTION")
        logger_instance.info("=" * 70)
        logger_instance.info("Found %d duplicate samples", len(duplicates_view))
        logger_instance.info("Launching FiftyOne App for manual review...")
        logger_instance.info("Review the duplicates and close the app when done.")
        session = fo.launch_app(duplicates_view)
        input("\nPress ENTER to continue and finalize deduplication...")
        session.close()
        logger_instance.info("Manual inspection complete")

    summary = {
        "dataset": dataset_name,
        "method": "fiftyone_brain",
        "model": model_name,
        "embeddings_field": embeddings_field,
        "similarity_threshold": similarity_threshold,
        "samples_total": len(view),
        "duplicates_found": brain_results.get("near_duplicates_found", 0),
        "duplicate_tag": duplicate_tag,
        "is_duplicate_field": is_duplicate_field,
        "duplicate_of_field": duplicate_of_field,
        "brain_results": brain_results,
    }

    if summary_location:
        write_summary(summary, summary_location)
        logger_instance.info("Summary written to %s", summary_location)

    logger_instance.info("Deduplication complete: %d duplicates flagged", duplicates_count)
    dataset.save()
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate images using FiftyOne brain")
    parser.add_argument("--dataset", type=str, default=JID_MASTER_DATASET)
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--model-name", type=str, default="hf-hub:BVRA/MegaDescriptor-L-384")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding computation")
    parser.add_argument("--compute-similarity-index", action="store_true", help="Compute similarity index for similarity search")
    parser.add_argument("--detect-leaky-splits", action="store_true", help="Detect similar samples across splits")
    parser.add_argument("--split-field", type=str, default="split", help="Field containing split assignments")
    parser.add_argument("--inspect-app", action="store_true", help="Launch FiftyOne App to manually inspect duplicates")
    parser.add_argument("--summary-location", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    try:
        run_processing(
            dataset_name=args.dataset,
            similarity_threshold=args.similarity_threshold,
            model_name=args.model_name,
            batch_size=args.batch_size,
            compute_similarity_index=args.compute_similarity_index,
            detect_leaky_splits=args.detect_leaky_splits,
            split_field=args.split_field,
            inspect_app=args.inspect_app,
            summary_location=args.summary_location,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as exc:
        logger.error("Error during deduplication: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
