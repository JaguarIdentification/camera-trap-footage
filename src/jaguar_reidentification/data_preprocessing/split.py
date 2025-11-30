"""Split jaguar identification dataset into train/validation/test sets.

This script adds split assignment columns to a CSV dataset, indicating whether each
image belongs to train, validation, or test sets. Supports two splitting strategies:

1. Closed-set split: All jaguar IDs appear in the training set. Images are randomly
   distributed across train/val/test splits, ensuring each individual has at least
   one sample in training. Single-image individuals go entirely to training.

2. Open-set split: Some jaguar IDs are exclusive to training, while others may appear
   across train/val/test sets. This simulates a more realistic scenario but is not
   a true open-set (test IDs are still seen during training for some individuals).

The script reads a CSV file, adds split column(s), and writes the result back to a file.
You can add both closed and open set splits simultaneously by specifying both split types.

Input CSV columns (required):
- JAGUAR ID: Identifier for the individual jaguar
- FILE PATH: Path to the image file

Output CSV columns added:
- closed_set_split: Split assignment (train/val/test) for closed-set strategy (if requested)
- open_set_split: Split assignment (train/val/test) for open-set strategy (if requested)

Arguments:
    --add_closed_set: Add closed-set split column (default: False)
    --add_open_set: Add open-set split column (default: False)
    --output_csv: Output CSV path (defaults to input_csv if not specified)
    
    Closed-set split parameters:
        --train_ratio: Proportion of images for training (default: 0.8)
        --val_ratio: Proportion of images for validation (default: 0.1)
        --test_ratio: Proportion of images for test (default: 0.1)
        
        Behavior: All jaguar IDs appear in training. For each ID with multiple images,
        images are randomly distributed across train/val/test while ensuring at least
        one image per ID goes to training. Single-image IDs go entirely to training.
    
    Open-set split parameters:
        ID-level ratios (how IDs are grouped):
        --train_only_ids_ratio: Proportion of IDs exclusive to training (default: 0.6)
        --shared_ids_ratio: Proportion of IDs that appear across all splits (default: 0.2)
        --val_test_ids_ratio: Proportion of IDs exclusive to val/test (default: 0.2)
        
        Sample-level ratios (how images are distributed):
        --shared_train_ratio: For shared IDs, proportion of images for training (default: 0.8)
        --shared_val_ratio: For shared/val_test IDs, proportion for validation (default: 0.1)
        --shared_test_ratio: For shared/val_test IDs, proportion for test (default: 0.1)
        
        Behavior: IDs are first grouped into train-only (all images → training), 
        shared (images split across train/val/test), and val_test-only (images split
        between val/test). This creates a more realistic scenario where some individuals
        never appear during training.

Run as a module:
    # Add closed-set split column only
    python -m src.jaguar_reidentification.data_preprocessing.split \
        --input_csv=data/intermediate/v1/preprocessed_labels.csv \
        --add_closed_set \
        --seed=42

    # Add open-set split column only
    python -m src.jaguar_reidentification.data_preprocessing.split \
        --input_csv=data/intermediate/v1/preprocessed_labels.csv \
        --add_open_set \
        --train_only_ids_ratio=0.6 \
        --shared_ids_ratio=0.2 \
        --val_test_ids_ratio=0.2 \
        --seed=42

    # Add both split columns with custom output path (can specify all ratio parameters)
    python -m src.jaguar_reidentification.data_preprocessing.split \
        --input_csv=data/intermediate/v1/preprocessed_labels.csv \
        --output_csv=data/intermediate/v1/labels_with_splits.csv \
        --add_closed_set \
        --add_open_set \
        --seed=42 \
        --generate_report
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.jaguar_reidentification.utils.utils import json_safe


def write_report(report: dict, report_path: Path) -> None:
    """Write processing report to JSON file."""
    logging.info("Writing split report to %s", report_path)
    report = json_safe(report)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)


def add_split_column(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Add a split assignment column to the dataframe.

    Args:
        df: Original dataframe
        train_df: Subset assigned to training
        val_df: Subset assigned to validation
        test_df: Subset assigned to test
        column_name: Name of the column to add

    Returns:
        Updated dataframe with split column, report dict
    """
    # Create split column initialized with None
    df[column_name] = None

    # Assign split labels based on index
    df.loc[train_df.index, column_name] = "train"
    df.loc[val_df.index, column_name] = "val"
    df.loc[test_df.index, column_name] = "test"

    report = {
        "column_name": column_name,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "train_ids": int(train_df["JAGUAR ID"].nunique()),
        "val_ids": int(val_df["JAGUAR ID"].nunique()),
        "test_ids": int(test_df["JAGUAR ID"].nunique()),
    }

    logging.info(
        "Added '%s' column: train=%d, val=%d, test=%d",
        column_name,
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return df, report


def closed_set_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Perform closed-set split: all IDs appear in training, images randomly distributed.

    Strategy:
    - For each jaguar ID with multiple images, randomly split images across train/val/test
    - Ensure at least one image per ID goes to training
    - IDs with only one image go entirely to training

    Args:
        df: Input dataframe with JAGUAR ID and image metadata
        train_ratio: Proportion of images for training (0-1)
        val_ratio: Proportion of images for validation (0-1)
        test_ratio: Proportion of images for test (0-1)
        seed: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df, report_dict
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_rows = []
    val_rows = []
    test_rows = []

    single_image_ids = 0
    multi_image_ids = 0

    # Group by JAGUAR ID and split images
    for _, group in df.groupby("JAGUAR ID"):
        group_indices = group.index.tolist()
        n_images = len(group_indices)

        if n_images == 1:
            # Single image: goes to training
            train_rows.extend(group_indices)
            single_image_ids += 1
        else:
            # Multiple images: split according to ratios, ensure at least one in training
            multi_image_ids += 1

            ratios = np.array([train_ratio, val_ratio, test_ratio])
            ratios = ratios / ratios.sum()

            # We reserve a single image to ensure training set has at least one image
            n_remaining = n_images - 1

            # fraction of train reserved in terms of total images
            reserved_fraction = 1 / n_images

            # adjust ratios for the remaining images
            adjusted_ratios = ratios - np.array([reserved_fraction, 0, 0])
            adjusted_ratios = np.clip(adjusted_ratios, 0, None)  # ensure no negative
            adjusted_ratios = adjusted_ratios / adjusted_ratios.sum()  # normalize to sum=1

            n_train_target, n_val_target, n_test_target = np.random.multinomial(n_remaining, adjusted_ratios)
            assert n_train_target + n_val_target + n_test_target == n_remaining
            n_train_target += 1  # add back the reserved 1

            # Shuffle indices
            shuffled = rng.permutation(group_indices).tolist()

            # Assign images
            train_rows.extend(shuffled[:n_train_target])
            val_rows.extend(shuffled[n_train_target : n_train_target + n_val_target])
            test_rows.extend(shuffled[n_train_target + n_val_target :])

    train_df = df.loc[train_rows]
    val_df = df.loc[val_rows]
    test_df = df.loc[test_rows]

    report = {
        "split_type": "closed",
        "total_ids": int(df["JAGUAR ID"].nunique()),
        "single_image_ids": single_image_ids,
        "multi_image_ids": multi_image_ids,
        "train_ids": int(train_df["JAGUAR ID"].nunique()),
        "val_ids": int(val_df["JAGUAR ID"].nunique()),
        "test_ids": int(test_df["JAGUAR ID"].nunique()),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_ratio_actual": len(train_df) / len(df),
        "val_ratio_actual": len(val_df) / len(df),
        "test_ratio_actual": len(test_df) / len(df),
        "train_ratio_target": train_ratio,
        "val_ratio_target": val_ratio,
        "test_ratio_target": test_ratio,
        "seed": seed,
    }

    logging.info(
        "Closed-set split complete: %d IDs total, %d with single image, %d with multiple images",
        report["total_ids"],
        single_image_ids,
        multi_image_ids,
    )
    logging.info(
        "Sample distribution: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_df),
        100 * report["train_ratio_actual"],
        len(val_df),
        100 * report["val_ratio_actual"],
        len(test_df),
        100 * report["test_ratio_actual"],
    )

    return train_df, val_df, test_df, report


def open_set_split(
    df: pd.DataFrame,
    train_only_ids_ratio: float = 0.6,
    shared_ids_ratio: float = 0.2,
    val_test_ids_ratio: float = 0.2,
    shared_train_ratio: float = 0.8,
    shared_val_ratio: float = 0.1,
    shared_test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Perform open-set split: some IDs exclusive to train, others appear across splits.

    Strategy:
    1. Split jaguar IDs into three groups:
       - train_only_ids: IDs that only appear in training
       - shared_ids: IDs that can appear in train/val/test
       - test_only_ids: IDs that only appear in val/test
    2. For train_only_ids: all images go to training
    3. For shared_ids: split images across train/val/test according to ratios
    4. For test_only_ids: split images between val and test

    Args:
        df: Input dataframe with JAGUAR ID and image metadata
        train_only_ids_ratio: Proportion of IDs exclusive to training (0-1)
        shared_ids_ratio: Proportion of IDs that can appear in all splits (0-1)
        val_test_ids_ratio: Proportion of IDs exclusive to val/test (0-1)
        shared_train_ratio: For shared IDs, proportion of images for training (0-1)
        shared_val_ratio: For shared and val_test IDs, proportion of images for validation (0-1)
        shared_test_ratio: For shared and val_test IDs, proportion of images for test (0-1)
        seed: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df, report_dict
    """
    if not np.isclose(train_only_ids_ratio + shared_ids_ratio + val_test_ids_ratio, 1.0):
        raise ValueError(f"ID split ratios must sum to 1.0, got {train_only_ids_ratio + shared_ids_ratio + val_test_ids_ratio}")

    if not np.isclose(shared_train_ratio + shared_val_ratio + shared_test_ratio, 1.0):
        raise ValueError(f"Sample split ratios must sum to 1.0, got {shared_train_ratio + shared_val_ratio + shared_test_ratio}")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Get unique jaguar IDs
    all_ids = df["JAGUAR ID"].unique()
    n_ids = len(all_ids)

    # Shuffle and split IDs into three groups
    shuffled_ids = rng.permutation(all_ids)

    n_train_only = int(np.floor(n_ids * train_only_ids_ratio))
    n_shared = int(np.floor(n_ids * shared_ids_ratio))
    n_val_test_only = n_ids - n_train_only - n_shared

    train_only_ids = set(shuffled_ids[:n_train_only])
    shared_ids = set(shuffled_ids[n_train_only : n_train_only + n_shared])
    val_test_only_ids = set(shuffled_ids[n_train_only + n_shared :])

    train_rows = []
    val_rows = []
    test_rows = []

    # Process each jaguar ID according to its group
    for jaguar_id, group in df.groupby("JAGUAR ID"):
        group_indices = group.index.tolist()
        n_images = len(group_indices)

        # Shuffle images for this ID
        shuffled = rng.permutation(group_indices).tolist()

        if jaguar_id in train_only_ids:
            # All images go to training
            train_rows.extend(shuffled)

        elif jaguar_id in shared_ids:
            # Split images across train/val/test according to shared ratios
            ratios = np.array([shared_train_ratio, shared_val_ratio, shared_test_ratio])
            ratios = ratios / ratios.sum()

            n_train, n_val, n_test = np.random.multinomial(n_images, ratios)
            assert n_train + n_val + n_test == n_images

            train_rows.extend(shuffled[:n_train])
            val_rows.extend(shuffled[n_train : n_train + n_val])
            test_rows.extend(shuffled[n_train + n_val :])
        elif jaguar_id in val_test_only_ids:
            # Split images between val and test (no training samples)
            # Use shared_val_ratio / (shared_val_ratio + shared_test_ratio) for validation proportion
            val_test_sum = shared_val_ratio + shared_test_ratio
            val_prob = shared_val_ratio / val_test_sum if val_test_sum > 0 else 0.5

            n_val = np.random.binomial(n_images, val_prob)
            assert n_val <= n_images

            val_rows.extend(shuffled[:n_val])
            test_rows.extend(shuffled[n_val:])

    train_df = df.loc[train_rows]
    val_df = df.loc[val_rows]
    test_df = df.loc[test_rows]

    # Calculate ID overlap statistics
    train_ids_set = set(train_df["JAGUAR ID"].unique())
    val_ids_set = set(val_df["JAGUAR ID"].unique())
    test_ids_set = set(test_df["JAGUAR ID"].unique())

    ids_in_all_splits = train_ids_set & val_ids_set & test_ids_set
    ids_only_in_train = train_ids_set - val_ids_set - test_ids_set
    ids_only_in_val_test = (val_ids_set | test_ids_set) - train_ids_set

    report = {
        "split_type": "open",
        "total_ids": int(df["JAGUAR ID"].nunique()),
        "train_only_ids_target": n_train_only,
        "shared_ids_target": n_shared,
        "val_test_only_ids_target": n_val_test_only,
        "train_only_ids_actual": len(ids_only_in_train),
        "shared_ids_actual": len(ids_in_all_splits),
        "val_test_only_ids_actual": len(ids_only_in_val_test),
        "train_ids": int(train_df["JAGUAR ID"].nunique()),
        "val_ids": int(val_df["JAGUAR ID"].nunique()),
        "test_ids": int(test_df["JAGUAR ID"].nunique()),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_ratio_actual": len(train_df) / len(df),
        "val_ratio_actual": len(val_df) / len(df),
        "test_ratio_actual": len(test_df) / len(df),
        "train_only_ids_ratio_target": train_only_ids_ratio,
        "shared_ids_ratio_target": shared_ids_ratio,
        "val_test_ids_ratio_target": val_test_ids_ratio,
        "shared_train_ratio_target": shared_train_ratio,
        "shared_val_ratio_target": shared_val_ratio,
        "shared_test_ratio_target": shared_test_ratio,
        "seed": seed,
    }

    logging.info(
        "Open-set split complete: %d IDs total, %d train-only, %d shared, %d test-only",
        report["total_ids"],
        len(ids_only_in_train),
        len(ids_in_all_splits),
        len(ids_only_in_val_test),
    )
    logging.info(
        "Sample distribution: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_df),
        100 * report["train_ratio_actual"],
        len(val_df),
        100 * report["val_ratio_actual"],
        len(test_df),
        100 * report["test_ratio_actual"],
    )

    return train_df, val_df, test_df, report


def check_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Verify required columns exist in the dataframe."""
    required_columns = ["JAGUAR ID", "FILE PATH"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing_columns)}")

    return df


def load_df(input_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Load and validate input CSV file."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    logging.info("Loading dataset from %s", input_csv)

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        logging.error("Failed to load CSV file: %s", e)
        raise

    df = check_required_columns(df)

    # Filter to only images (no videos)
    if "FILE TYPE" in df.columns:
        original_len = len(df)
        df = df[df["FILE TYPE"] == "IMAGE"].reset_index(drop=True)
        n_filtered = original_len - len(df)
        if n_filtered > 0:
            logging.info("Filtered out %d non-image rows", n_filtered)

    report = {
        "input_path": input_csv.as_posix(),
        "total_samples": len(df),
        "total_ids": int(df["JAGUAR ID"].nunique()),
        "columns": df.columns.tolist(),
    }

    logging.info("Loaded %d image samples with %d unique jaguar IDs", report["total_samples"], report["total_ids"])

    return df, report


def run(
    input_csv: Path,
    output_csv: Path = None,
    add_closed_set: bool = False,
    add_open_set: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_only_ids_ratio: float = 0.6,
    shared_ids_ratio: float = 0.2,
    val_test_ids_ratio: float = 0.2,
    shared_train_ratio: float = 0.8,
    shared_val_ratio: float = 0.1,
    shared_test_ratio: float = 0.1,
    seed: int = 42,
    generate_report: bool = False,
) -> None:
    """Main processing function.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (defaults to input_csv)
        add_closed_set: Add closed-set split column
        add_open_set: Add open-set split column
        train_ratio: Closed-set train ratio
        val_ratio: Closed-set val ratio
        test_ratio: Closed-set test ratio
        train_only_ids_ratio: Open-set train-only IDs ratio
        shared_ids_ratio: Open-set shared IDs ratio
        val_test_ids_ratio: Open-set val/test-only IDs ratio
        shared_train_ratio: Open-set shared train ratio
        shared_val_ratio: Open-set shared val ratio
        shared_test_ratio: Open-set shared test ratio
        seed: Random seed
        generate_report: Whether to generate JSON report
    """
    if not add_closed_set and not add_open_set:
        raise ValueError("Must specify at least one of --add_closed_set or --add_open_set")

    if output_csv is None:
        output_csv = input_csv

    report = {}
    df, report["input"] = load_df(input_csv)

    # Add closed-set split column if requested
    if add_closed_set:
        logging.info("Generating closed-set split...")
        train_df, val_df, test_df, report["closed_set_split"] = closed_set_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        df, report["closed_set_output"] = add_split_column(df, train_df, val_df, test_df, "closed_set_split")

    # Add open-set split column if requested
    if add_open_set:
        logging.info("Generating open-set split...")
        train_df, val_df, test_df, report["open_set_split"] = open_set_split(
            df,
            train_only_ids_ratio=train_only_ids_ratio,
            shared_ids_ratio=shared_ids_ratio,
            val_test_ids_ratio=val_test_ids_ratio,
            shared_train_ratio=shared_train_ratio,
            shared_val_ratio=shared_val_ratio,
            shared_test_ratio=shared_test_ratio,
            seed=seed,
        )
        df, report["open_set_output"] = add_split_column(df, train_df, val_df, test_df, "open_set_split")

    # Write output CSV
    logging.info("Writing output to %s", output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    report["output"] = {
        "output_path": output_csv.as_posix(),
        "total_rows": len(df),
        "columns_added": [],
    }
    if add_closed_set:
        report["output"]["columns_added"].append("closed_set_split")
    if add_open_set:
        report["output"]["columns_added"].append("open_set_split")

    if generate_report:
        report_path = output_csv.parent / f"{output_csv.stem}_split_report.json"
        write_report(report, report_path)


def main():
    parser = argparse.ArgumentParser(description="Add train/val/test split columns to jaguar identification dataset.")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file with image metadata.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (defaults to input_csv).",
    )
    parser.add_argument(
        "--add_closed_set",
        action="store_true",
        help="Add closed-set split column to the dataset.",
    )
    parser.add_argument(
        "--add_open_set",
        action="store_true",
        help="Add open-set split column to the dataset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="[Closed-set only] Proportion of samples for training (0-1).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="[Closed-set only] Proportion of samples for validation (0-1).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="[Closed-set only] Proportion of samples for test (0-1).",
    )
    parser.add_argument(
        "--train_only_ids_ratio",
        type=float,
        default=0.6,
        help="[Open-set only] Proportion of IDs that only appear in training (0-1).",
    )
    parser.add_argument(
        "--shared_ids_ratio",
        type=float,
        default=0.2,
        help="[Open-set only] Proportion of IDs that can appear across train/val/test (0-1).",
    )
    parser.add_argument(
        "--val_test_ids_ratio",
        type=float,
        default=0.2,
        help="[Open-set only] Proportion of IDs that only appear in val/test (0-1).",
    )
    parser.add_argument(
        "--shared_train_ratio",
        type=float,
        default=0.8,
        help="[Open-set only] For shared IDs, proportion of images for training (0-1).",
    )
    parser.add_argument(
        "--shared_val_ratio",
        type=float,
        default=0.1,
        help="[Open-set only] For shared and val_test IDs, proportion of images for validation (0-1).",
    )
    parser.add_argument(
        "--shared_test_ratio",
        type=float,
        default=0.1,
        help="[Open-set only] For shared and val_test IDs, proportion of images for test (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate a JSON report with split statistics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    run(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        add_closed_set=args.add_closed_set,
        add_open_set=args.add_open_set,
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
        generate_report=args.generate_report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
