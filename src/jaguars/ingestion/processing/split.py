r"""Dataset splitting module for FiftyOne.

Splits a FiftyOne dataset into train/validation/test sets based on jaguar identities.
Adds split info to sample fields (e.g. "closed_set_split", "open_set_split").

Strategies:
1. Closed-set: All IDs in train.
2. Open-set: Some IDs excluded from train.

Usage (CLI):
    python src/jaguars/ingestion/processing/split.py --add-closed-set --verbose

With arguments:
    python src/jaguars/ingestion/processing/split.py \
        --dataset JID_Master_Dataset \
        --add-closed-set \
        --add-open-set \
        --verbose
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fiftyone as fo
import numpy as np
from fiftyone import ViewField as F
from tqdm import tqdm

from jaguars.common.config import DEFAULT_GROUP_SLICE, JID_MASTER_DATASET
from jaguars.common.fiftyone_utils import get_or_create_dataset
from jaguars.common.logging_utils import setup_logger

MODULE_NAME = "ingestion.processing.split"
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


def get_id_field(dataset: fo.Dataset) -> str:
    """Determine the field name/path for jaguar ID."""
    # check for 'jaguar_id' field
    if "jaguar_id" in dataset.get_field_schema():
        field = dataset.get_field("jaguar_id")
        if isinstance(field, fo.EmbeddedDocumentField):
            # Assumed to be Classification
            return "jaguar_id.label"
        return "jaguar_id"

    # check for ground_truth
    if "ground_truth" in dataset.get_field_schema():
        return "ground_truth.label"

    raise ValueError("Could not find 'jaguar_id' or 'ground_truth' field in dataset.")


def closed_set_split(
    dataset: fo.Dataset,
    id_field: str,
    field_name: str = "closed_set_split",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Perform closed-set split on dataset view. Field must already exist."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    jaguar_ids = dataset.distinct(id_field)

    train_count = 0
    val_count = 0
    test_count = 0

    updates = []  # List of (sample_id, split_label)

    for jid in tqdm(jaguar_ids, desc="Closed-set split"):
        if jid is None:
            continue

        view = dataset.match(F(id_field) == jid)
        sample_ids = view.values("id")
        n_images = len(sample_ids)

        if n_images == 0:
            continue

        current_splits = []
        if n_images == 1:
            current_splits = ["train"]
        else:
            ratios = np.array([train_ratio, val_ratio, test_ratio])
            ratios = ratios / ratios.sum()

            n_remaining = n_images - 1
            n_train, n_val, n_test = np.random.multinomial(n_remaining, ratios)

            split_labels = ["train"] * (n_train + 1) + ["val"] * n_val + ["test"] * n_test
            rng.shuffle(split_labels)
            current_splits = split_labels

        for sid, label in zip(sample_ids, current_splits, strict=False):
            updates.append((sid, label))
            if label == "train":
                train_count += 1
            elif label == "val":
                val_count += 1
            elif label == "test":
                test_count += 1

    train_ids = [sid for sid, lbl in updates if lbl == "train"]
    val_ids = [sid for sid, lbl in updates if lbl == "val"]
    test_ids = [sid for sid, lbl in updates if lbl == "test"]

    if train_ids:
        dataset.select(train_ids).set_values(field_name, ["train"] * len(train_ids))
    if val_ids:
        dataset.select(val_ids).set_values(field_name, ["val"] * len(val_ids))
    if test_ids:
        dataset.select(test_ids).set_values(field_name, ["test"] * len(test_ids))

    return {"train_samples": train_count, "val_samples": val_count, "test_samples": test_count, "total_ids": len(jaguar_ids)}


def open_set_split(
    dataset: fo.Dataset,
    id_field: str,
    field_name: str = "open_set_split",
    train_only_ids_ratio: float = 0.6,
    shared_ids_ratio: float = 0.2,
    val_test_ids_ratio: float = 0.2,
    shared_train_ratio: float = 0.8,
    shared_val_ratio: float = 0.1,
    shared_test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Perform open-set split on dataset view. Field must already exist."""
    if not np.isclose(train_only_ids_ratio + shared_ids_ratio + val_test_ids_ratio, 1.0):
        raise ValueError("ID split ratios must sum to 1.0")
    if not np.isclose(shared_train_ratio + shared_val_ratio + shared_test_ratio, 1.0):
        raise ValueError("Shared sample ratios must sum to 1.0")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    jaguar_ids = dataset.distinct(id_field)
    jaguar_ids = [j for j in jaguar_ids if j is not None]
    n_ids = len(jaguar_ids)

    rng.shuffle(jaguar_ids)

    n_train_only = int(np.floor(n_ids * train_only_ids_ratio))
    n_shared = int(np.floor(n_ids * shared_ids_ratio))

    train_only_set = set(jaguar_ids[:n_train_only])
    shared_set = set(jaguar_ids[n_train_only : n_train_only + n_shared])
    val_test_set = set(jaguar_ids[n_train_only + n_shared :])

    train_sids = []
    val_sids = []
    test_sids = []

    for jid in tqdm(jaguar_ids, desc="Open-set split"):
        view = dataset.match(F(id_field) == jid)
        sample_ids = view.values("id")
        n_images = len(sample_ids)
        if n_images == 0:
            continue

        splits = []
        if jid in train_only_set:
            splits = ["train"] * n_images
        elif jid in shared_set:
            ratios = np.array([shared_train_ratio, shared_val_ratio, shared_test_ratio])
            ratios = ratios / ratios.sum()
            n_t, n_v, n_te = np.random.multinomial(n_images, ratios)
            splits = ["train"] * n_t + ["val"] * n_v + ["test"] * n_te
            rng.shuffle(splits)
        elif jid in val_test_set:
            denom = shared_val_ratio + shared_test_ratio
            val_prob = 0.5 if denom == 0 else shared_val_ratio / denom

            n_v = np.random.binomial(n_images, val_prob)
            n_te = n_images - n_v
            splits = ["val"] * n_v + ["test"] * n_te
            rng.shuffle(splits)

        for sid, lbl in zip(sample_ids, splits, strict=False):
            if lbl == "train":
                train_sids.append(sid)
            elif lbl == "val":
                val_sids.append(sid)
            elif lbl == "test":
                test_sids.append(sid)

    if train_sids:
        dataset.select(train_sids).set_values(field_name, ["train"] * len(train_sids))
    if val_sids:
        dataset.select(val_sids).set_values(field_name, ["val"] * len(val_sids))
    if test_sids:
        dataset.select(test_sids).set_values(field_name, ["test"] * len(test_sids))

    return {
        "train_samples": len(train_sids),
        "val_samples": len(val_sids),
        "test_samples": len(test_sids),
        "train_only_ids": len(train_only_set),
        "shared_ids": len(shared_set),
        "val_test_ids": len(val_test_set),
    }


def _tag_samples_by_field(dataset: fo.Dataset, field_name: str) -> None:
    """Tag samples with train/val/test tags based on split field value."""
    if field_name not in dataset.get_field_schema():
        logger.error("Cannot tag samples: field %s not found", field_name)
        return

    # Get sample IDs for each split
    train_view = dataset.match(F(field_name) == "train")
    val_view = dataset.match(F(field_name) == "val")
    test_view = dataset.match(F(field_name) == "test")

    # Tag samples
    if len(train_view) > 0:
        train_view.tag_samples("train")
        logger.info("Tagged %d samples with 'train'", len(train_view))

    if len(val_view) > 0:
        val_view.tag_samples("val")
        logger.info("Tagged %d samples with 'val'", len(val_view))

    if len(test_view) > 0:
        test_view.tag_samples("test")
        logger.info("Tagged %d samples with 'test'", len(test_view))


def verify_integrity(dataset: fo.Dataset, field_name: str, split_type: str) -> None:
    logger.info("Verifying %s integrity...", split_type)
    if field_name not in dataset.get_field_schema():
        logger.error("Field %s missing!", field_name)
        return

    counts = dataset.count_values(field_name)
    logger.info("Counts for %s: %s", field_name, counts)


def run_processing(
    dataset_name: str = JID_MASTER_DATASET,
    add_closed_set: bool = False,
    add_open_set: bool = False,
    tag_by: str | None = None,
    # Closed set
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    # Open set
    train_only_ids_ratio: float = 0.6,
    shared_ids_ratio: float = 0.2,
    val_test_ids_ratio: float = 0.2,
    shared_train_ratio: float = 0.8,
    shared_val_ratio: float = 0.1,
    shared_test_ratio: float = 0.1,
    seed: int = 42,
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> fo.Dataset | None:
    """Core logic for splitting."""
    global logger
    logger = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)
    logger.info("Starting split processing for dataset: %s", dataset_name)

    validate_resources(dataset_name)

    # Validate tag_by parameter
    if tag_by is not None and tag_by not in ["closed", "open"]:
        raise ValueError(f"tag_by must be 'closed', 'open', or None, got '{tag_by}'")

    if dry_run:
        logger.info("DRY RUN: Would split dataset '%s'", dataset_name)
        return None

    full_dataset = get_or_create_dataset(dataset_name)

    # Filter to only image group slice if dataset has groups
    if full_dataset.group_field is not None:
        logger.info("Filtering dataset to '%s' group slice only", DEFAULT_GROUP_SLICE)
        dataset = full_dataset.select_group_slices(DEFAULT_GROUP_SLICE)
        logger.info("Working with %d samples from '%s' slice", len(dataset), DEFAULT_GROUP_SLICE)
    else:
        dataset = full_dataset
        logger.info("Dataset has no group field, working with all %d samples", len(dataset))

    # Detect ID field
    try:
        id_field = get_id_field(dataset)
        logger.info("Using ID field: %s", id_field)
    except Exception as e:
        logger.error("Cannot proceed without ID field: %s", e)
        return dataset

    report = {"dataset_name": dataset_name, "seed": seed}

    if add_closed_set:
        logger.info("Applying Closed-Set Split...")
        # Prepare field on full dataset
        if "closed_set_split" in full_dataset.get_field_schema():
            full_dataset.clear_sample_field("closed_set_split")
        else:
            full_dataset.add_sample_field("closed_set_split", fo.StringField)

        res = closed_set_split(dataset=dataset, id_field=id_field, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        report["closed_set"] = res
        verify_integrity(dataset, "closed_set_split", "Closed-Set")

    if add_open_set:
        logger.info("Applying Open-Set Split...")
        # Prepare field on full dataset
        if "open_set_split" in full_dataset.get_field_schema():
            full_dataset.clear_sample_field("open_set_split")
        else:
            full_dataset.add_sample_field("open_set_split", fo.StringField)

        res = open_set_split(
            dataset=dataset,
            id_field=id_field,
            train_only_ids_ratio=train_only_ids_ratio,
            shared_ids_ratio=shared_ids_ratio,
            val_test_ids_ratio=val_test_ids_ratio,
            shared_train_ratio=shared_train_ratio,
            shared_val_ratio=shared_val_ratio,
            shared_test_ratio=shared_test_ratio,
            seed=seed,
        )
        report["open_set"] = res
        verify_integrity(dataset, "open_set_split", "Open-Set")

    # also add it to "split" field
    full_dataset.add_sample_field("split", fo.StringField)
    main_split_field = "open_set_split" if tag_by == "open" or not add_closed_set else "closed_set_split"

    for split_label in ["train", "val", "test"]:
        sample_ids = dataset.match(F(main_split_field) == split_label).values("id")
        if sample_ids:
            full_dataset.select(sample_ids).set_values("split", [split_label] * len(sample_ids))

    # Tag samples by split if requested
    if tag_by == "closed":
        if not add_closed_set:
            logger.warning("tag_by='closed' specified but --add-closed-set not set. No tagging performed.")
        else:
            logger.info("Tagging samples by closed-set split...")
            _tag_samples_by_field(dataset, "closed_set_split")
    elif tag_by == "open":
        if not add_open_set:
            logger.warning("tag_by='open' specified but --add-open-set not set. No tagging performed.")
        else:
            logger.info("Tagging samples by open-set split...")
            _tag_samples_by_field(dataset, "open_set_split")

    full_dataset.save()

    if summary_location:
        write_summary(report, summary_location)
        logger.info("Report written to %s", summary_location)

    return full_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Split FiftyOne dataset into train/val/test.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default=JID_MASTER_DATASET)
    parser.add_argument("--add-closed-set", action="store_true")
    parser.add_argument("--add-open-set", action="store_true")
    parser.add_argument("--tag-by", type=str, choices=["closed", "open"], default=None, help="Tag samples by this split type (train/val/test tags)")

    # Closed set
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    # Open set
    parser.add_argument("--train-only-ids-ratio", type=float, default=0.6)
    parser.add_argument("--shared-ids-ratio", type=float, default=0.2)
    parser.add_argument("--val-test-ids-ratio", type=float, default=0.2)
    parser.add_argument("--shared-train-ratio", type=float, default=0.8)
    parser.add_argument("--shared-val-ratio", type=float, default=0.1)
    parser.add_argument("--shared-test-ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-location", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.add_closed_set and not args.add_open_set:
        parser.error("Must specify at least one of --add-closed-set or --add-open-set")

    try:
        run_processing(
            dataset_name=args.dataset,
            add_closed_set=args.add_closed_set,
            add_open_set=args.add_open_set,
            tag_by=args.tag_by,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            train_only_ids_ratio=args.train_only_ids_ratio,
            shared_ids_ratio=args.shared_ids_ratio,
            val_test_ids_ratio=args.val_test_ids_ratio,
            shared_train_ratio=args.shared_train_ratio,
            shared_val_ratio=args.shared_val_ratio,
            shared_test_ratio=args.shared_test_ratio,
            seed=args.seed,
            summary_location=args.summary_location,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error("Error during split: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
