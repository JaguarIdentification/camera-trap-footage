"""Deduplicate jaguar identification dataset using FiftyOne and image embeddings.

This script detects and handles duplicate/similar images in the dataset using
FiftyOne's brain methods and deep learning embeddings from the Model Zoo.

Important: Deduplication only compares images from the same JAGUAR ID. This ensures we
remove duplicate images of the same individual, not visually similar images of different jaguars.

FiftyOne Brain Metrics Computed:
- Uniqueness: How unique each sample is with respect to the rest
- Representativeness: How representative each sample is of nearby samples
- Similarity Index: Enables query/sort by similarity
- Visualizations: PCA, UMAP, t-SNE 2D projections
- Near Duplicates: Detects potential duplicates above threshold
- Exact Duplicates: Detects exact media duplicates
- Leaky Splits: Detects similar images across different splits

Embedding Models (from FiftyOne Model Zoo):
- resnet50-imagenet-torch: Fast baseline, 2048-D embeddings
- resnet101-imagenet-torch: Better for fine-grained distinctions (default)
- clip-vit-base32-torch: Vision-language model, 512-D embeddings

The script supports two modes:
1. Full mode: Compute embeddings using FiftyOne, detect duplicates, compute brain metrics
2. Inspect mode: Skip computation and load existing FiftyOne dataset for review

Input CSV columns (required):
- FILE PATH: Path to the image file
- JAGUAR ID: Identifier for the individual jaguar
- closed_set_split or open_set_split: Train/val/test assignment

Output:
- Updated CSV with 'is_duplicate' and 'duplicate_of' columns
- FiftyOne persistent dataset with all brain metrics for manual inspection
- JSON report with deduplication statistics

Run as a module:
    # Full deduplication with closed-set split using ResNet101
    python -m src.jaguar_reidentification.data_preprocessing.deduplicate \
        --input_csv=data/intermediate/v1/preprocessed_labels.csv \
        --output_csv=data/intermediate/v1/deduplicated_labels.csv \
        --split_column=closed_set_split \
        --model_name=resnet101 \
        --generate_report

    # Use CLIP embeddings
    python -m src.jaguar_reidentification.data_preprocessing.deduplicate \
        --input_csv=data/intermediate/v1/preprocessed_labels.csv \
        --output_csv=data/intermediate/v1/deduplicated_labels.csv \
        --split_column=open_set_split \
        --model_name=clip \
        --generate_report

    # Inspect mode: Load existing FiftyOne dataset for manual review
    python -m src.jaguar_reidentification.data_preprocessing.deduplicate \
        --input_csv=data/intermediate/v1/preprocessed_labels.csv \
        --split_column=closed_set_split \
        --inspect_only
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import numpy as np
import pandas as pd
from jaguar_reidentification.jaguar_reidentification.utils.utils import json_safe
from tqdm import tqdm


def write_report(report: dict, report_path: Path) -> None:
    """Write processing report to JSON file."""
    logging.info("Writing deduplication report to %s", report_path)
    report = json_safe(report)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)


def write_df(df: pd.DataFrame, output_csv: Path) -> dict:
    """Write deduplicated dataframe to CSV."""
    logging.info("Writing deduplicated labels to %s", output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    report = {
        "row_count": len(df),
        "columns": df.columns.tolist(),
        "duplicates_marked": int(df.get("is_duplicate", pd.Series([False])).sum()),
        "unique_images": int((~df.get("is_duplicate", pd.Series([False]))).sum()),
    }

    logging.info("Saved %d rows (%d duplicates, %d unique)", report["row_count"], report["duplicates_marked"], report["unique_images"])
    return report


def load_fiftyone_model(model_name: str) -> fo.Model:
    """Load a model from FiftyOne Model Zoo.

    Args:
        model_name: Model to use ('resnet50', 'resnet101', or 'clip')

    Returns:
        FiftyOne model and embedding field name
    """
    logging.info("Loading model '%s' from FiftyOne Model Zoo", model_name)

    if model_name == "resnet50":
        model = foz.load_zoo_model("resnet50-imagenet-torch")
    elif model_name == "resnet101":
        model = foz.load_zoo_model("resnet101-imagenet-torch")
    elif model_name == "clip":
        model = foz.load_zoo_model("clip-vit-base32-torch")
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'resnet50', 'resnet101', or 'clip'")

    return model


def compute_embeddings_fiftyone(
    dataset: fo.Dataset,
    model_name: str = "resnet101",
) -> tuple[fo.Dataset, dict[str, Any]]:
    """Compute embeddings using FiftyOne's built-in method.

    Args:
        dataset: FiftyOne dataset with samples
        model_name: Model to use ('resnet50', 'resnet101', or 'clip')

    Returns:
        Dataset with embeddings computed, report dict
    """
    embeddings_field = f"{model_name}_embeddings"

    # Check if embeddings already exist for all samples
    try:
        # Check if field exists and has values
        if embeddings_field in dataset.get_field_schema():
            # Check if any sample has embeddings
            sample_with_embeddings = dataset.first()
            if sample_with_embeddings and sample_with_embeddings[embeddings_field] is not None:
                # Count how many samples have embeddings
                num_with_embeddings = len(dataset.exists(embeddings_field))
                if num_with_embeddings == len(dataset):
                    logging.info("All samples already have embeddings in field '%s', skipping computation", embeddings_field)
                    report = {
                        "embeddings_computed": 0,
                        "embeddings_cached": num_with_embeddings,
                        "embeddings_field": embeddings_field,
                        "model_name": model_name,
                    }
                    return dataset, report
                else:
                    logging.info("Only %d/%d samples have embeddings, recomputing all", num_with_embeddings, len(dataset))
    except Exception as e:
        logging.debug("Error checking existing embeddings: %s", e)

    # Load model from FiftyOne Model Zoo
    model = load_fiftyone_model(model_name)

    # Compute embeddings using FiftyOne's built-in method
    logging.info("Computing embeddings for %d samples using model '%s'", len(dataset), model_name)
    dataset.compute_embeddings(
        model,
        embeddings_field=embeddings_field,
    )

    report = {
        "embeddings_computed": len(dataset),
        "embeddings_cached": 0,
        "embeddings_field": embeddings_field,
        "model_name": model_name,
    }

    logging.info("Computed %d embeddings in field '%s'", len(dataset), embeddings_field)
    return dataset, report


def get_embeddings_from_dataset(dataset: fo.Dataset, embeddings_field: str) -> dict:
    """Extract embeddings from FiftyOne dataset.

    Args:
        dataset: FiftyOne dataset
        embeddings_field: Name of the embeddings field

    Returns:
        Dictionary mapping sample IDs to embedding arrays
    """
    embeddings_dict = {}
    for sample in dataset:
        emb = sample[embeddings_field]
        if emb is not None:
            embeddings_dict[sample.id] = np.array(emb)
    return embeddings_dict


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between embeddings.

    Args:
        embeddings: Array of shape (N, D) where N is number of images, D is embedding dimension

    Returns:
        Similarity matrix of shape (N, N) with values in [0, 1]
    """
    # Normalize embeddings to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms

    # Compute cosine similarity as dot product of normalized vectors
    similarity = np.dot(normalized, normalized.T)

    # Clip to [0, 1] range (cosine similarity can be negative)
    similarity = np.clip(similarity, 0, 1)

    return similarity


def detect_duplicates_within_split(
    dataset: fo.Dataset,
    embeddings_field: str,
    split_value: str,
    threshold: float,
) -> tuple[fo.Dataset, dict]:
    """Detect duplicate images within a single split (train/val/test).

    Only compares images from the same JAGUAR ID to avoid marking visually similar
    images of different individuals as duplicates.

    Args:
        dataset: FiftyOne dataset with embeddings
        embeddings_field: Name of the embeddings field
        split_value: Split to process ('train', 'val', or 'test')
        threshold: Similarity threshold for marking duplicates (0-1)

    Returns:
        Updated dataset with is_duplicate and duplicate_of fields, report dict
    """
    # Filter to split
    split_view = dataset.match_tags(split_value) if split_value else dataset

    if len(split_view) == 0:
        logging.warning("No samples in split '%s'", split_value)
        return dataset, {"split": split_value, "duplicates_found": 0, "threshold": threshold}

    logging.info("Detecting duplicates in split '%s' with threshold=%.3f (%d images)", split_value, threshold, len(split_view))

    duplicates_found = 0
    duplicate_pairs = []

    # Get embeddings for this split
    embeddings_dict = get_embeddings_from_dataset(split_view, embeddings_field)

    # Group by JAGUAR ID
    jaguar_groups = {}
    for sample in split_view:
        jaguar_id = sample.jaguar_id
        if jaguar_id not in jaguar_groups:
            jaguar_groups[jaguar_id] = []
        jaguar_groups[jaguar_id].append(sample)

    # Detect duplicates within each jaguar
    for jaguar_id, samples in jaguar_groups.items():
        if len(samples) <= 1:
            continue

        # Get embeddings for this jaguar's samples
        sample_ids = [s.id for s in samples]
        embeddings = np.stack([embeddings_dict[sid] for sid in sample_ids])

        # Compute similarity matrix
        similarity = compute_similarity_matrix(embeddings)
        np.fill_diagonal(similarity, 0)

        for i in range(len(similarity)):
            max_sim_idx = np.argmax(similarity[i])
            max_sim = similarity[i, max_sim_idx]

            if max_sim >= threshold and max_sim_idx < i:
                # Mark as duplicate
                sample = samples[i]
                duplicate_of = samples[max_sim_idx]

                sample["is_duplicate"] = True
                sample["duplicate_of"] = duplicate_of.filepath
                sample["duplicate_similarity"] = float(max_sim)
                sample.save()

                duplicates_found += 1
                duplicate_pairs.append(
                    {
                        "jaguar_id": str(jaguar_id),
                        "file1": sample.filepath,
                        "file2": duplicate_of.filepath,
                        "similarity": float(max_sim),
                    }
                )

    report = {
        "split": split_value,
        "total_images": len(split_view),
        "unique_jaguar_ids": len(jaguar_groups),
        "duplicates_found": duplicates_found,
        "unique_images": len(split_view) - duplicates_found,
        "threshold": threshold,
        "duplicate_pairs": duplicate_pairs[:100],
    }

    logging.info("Found %d duplicates in split '%s' across %d jaguar IDs", duplicates_found, split_value, len(jaguar_groups))

    return dataset, report


def detect_leaky_splits(
    dataset: fo.Dataset,
    embeddings_field: str,
    threshold: float = 0.95,
) -> tuple[fo.Dataset, dict]:
    """Detect similar images across different splits using FiftyOne brain.

    Args:
        dataset: FiftyOne dataset with embeddings
        embeddings_field: Name of the embeddings field
        threshold: Similarity threshold for detecting leakage (0-1)

    Returns:
        Updated dataset, report dict
    """
    logging.info("Detecting leaky samples across splits using FiftyOne brain (threshold=%.3f)", threshold)

    try:
        # Use FiftyOne brain's compute_leaky_splits
        leaky_results = fob.compute_leaky_splits(
            dataset,
            splits="split",
            embeddings=embeddings_field,
            threshold=threshold,
        )

        report = {
            "leaky_samples_found": "computed",
            "threshold": threshold,
            "results": str(leaky_results),
        }

        logging.info("Leaky splits computation complete")

    except Exception as e:
        logging.warning("Failed to compute leaky splits: %s", e)
        report = {
            "leaky_samples_found": 0,
            "threshold": threshold,
            "error": str(e),
        }

    return dataset, report


def create_fiftyone_dataset(
    df: pd.DataFrame,
    data_root: Path,
    split_column: str,
    model_name: str = "resnet101",
    dataset_name: str = "jaguar_deduplication",
    persistent: bool = True,
    force_recreate: bool = False,
) -> tuple[fo.Dataset, dict]:
    """Create FiftyOne dataset with comprehensive brain computations.

    Args:
        df: Dataframe with sample metadata
        data_root: Root directory to resolve relative file paths
        split_column: Column name containing split assignments
        model_name: Name of embedding model used
        dataset_name: Name for the FiftyOne dataset
        persistent: Whether to persist the dataset for later inspection
        force_recreate: If True, delete and recreate dataset even if it exists

    Returns:
        FiftyOne dataset, report dict
    """
    logging.info("Creating FiftyOne dataset '%s'", dataset_name)

    # Check if dataset already exists
    if dataset_name in fo.list_datasets():
        if force_recreate:
            logging.info("Force recreate enabled, deleting existing dataset '%s'", dataset_name)
            fo.delete_dataset(dataset_name)
            dataset = None
        else:
            logging.info("Loading existing dataset '%s'", dataset_name)
            dataset = fo.load_dataset(dataset_name)

        # Check if dataset has same number of samples
        if len(dataset) == len(df):
            logging.info("Dataset already exists with %d samples, reusing it", len(dataset))
        else:
            logging.info("Dataset exists but has different number of samples (%d vs %d), recreating", len(dataset), len(df))
            fo.delete_dataset(dataset_name)
            dataset = None
    else:
        dataset = None

    # Create new dataset if needed
    if dataset is None:
        dataset = fo.Dataset(name=dataset_name, persistent=persistent)

        # Add samples
        samples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding samples to FiftyOne"):
            filepath = str(data_root / "files" / row["FILE PATH"])

            sample = fo.Sample(filepath=filepath)

            # Add metadata
            sample["jaguar_id"] = str(row.get("JAGUAR ID", ""))
            sample["split"] = str(row.get(split_column, ""))

            samples.append(sample)

        dataset.add_samples(samples)
        logging.info("Created new dataset with %d samples", len(dataset))

    # Compute embeddings using FiftyOne
    embeddings_field = f"{model_name}_embeddings"
    dataset, embeddings_report = compute_embeddings_fiftyone(dataset, model_name)

    brain_results = {}

    # 1. Compute uniqueness
    logging.info("Computing uniqueness scores...")
    try:
        fob.compute_uniqueness(
            dataset,
            embeddings=embeddings_field,
            uniqueness_field="uniqueness",
        )
        brain_results["uniqueness"] = True
        logging.info("✓ Uniqueness computed")
    except Exception as e:
        logging.warning("Failed to compute uniqueness: %s", e)
        brain_results["uniqueness"] = False

    # 2. Compute representativeness
    logging.info("Computing representativeness scores...")
    try:
        fob.compute_representativeness(
            dataset,
            embeddings=embeddings_field,
            representativeness_field="representativeness",
        )
        brain_results["representativeness"] = True
        logging.info("✓ Representativeness computed")
    except Exception as e:
        logging.warning("Failed to compute representativeness: %s", e)
        brain_results["representativeness"] = False

    # 3. Compute similarity index
    logging.info("Computing similarity index...")
    try:
        # Check if brain key already exists
        brain_keys = dataset.list_brain_runs()
        similarity_key = f"{model_name}_similarity"

        if similarity_key in brain_keys:
            logging.info("Similarity index '%s' already exists, skipping computation", similarity_key)
            brain_results["similarity"] = True
        else:
            fob.compute_similarity(
                dataset,
                embeddings=embeddings_field,
                brain_key=similarity_key,
                backend="sklearn",
            )
            brain_results["similarity"] = True
            logging.info("✓ Similarity index computed")
    except Exception as e:
        logging.warning("Failed to compute similarity: %s", e)
        brain_results["similarity"] = False

    # 4. Compute visualizations
    logging.info("Computing dimensionality reduction visualizations...")

    brain_keys = dataset.list_brain_runs()

    # PCA
    pca_key = f"{model_name}_pca"
    try:
        if pca_key in brain_keys:
            logging.info("PCA visualization '%s' already exists, skipping computation", pca_key)
            brain_results["pca"] = True
        else:
            fob.compute_visualization(
                dataset,
                embeddings=embeddings_field,
                brain_key=pca_key,
                method="pca",
                num_dims=2,
            )
            brain_results["pca"] = True
            logging.info("✓ PCA visualization computed")
    except Exception as e:
        logging.warning("Failed to compute PCA: %s", e)
        brain_results["pca"] = False

    # UMAP
    umap_key = f"{model_name}_umap"
    try:
        if umap_key in brain_keys:
            logging.info("UMAP visualization '%s' already exists, skipping computation", umap_key)
            brain_results["umap"] = True
        else:
            fob.compute_visualization(
                dataset,
                embeddings=embeddings_field,
                brain_key=umap_key,
                method="umap",
                num_dims=2,
            )
            brain_results["umap"] = True
            logging.info("✓ UMAP visualization computed")
    except Exception as e:
        error_msg = str(e)
        if "umap-learn" in error_msg:
            logging.warning("UMAP not available. Install with: pip install umap-learn")
        else:
            logging.warning("Failed to compute UMAP: %s", e)
        brain_results["umap"] = False

    # t-SNE
    tsne_key = f"{model_name}_tsne"
    try:
        if tsne_key in brain_keys:
            logging.info("t-SNE visualization '%s' already exists, skipping computation", tsne_key)
            brain_results["tsne"] = True
        else:
            fob.compute_visualization(
                dataset,
                embeddings=embeddings_field,
                brain_key=tsne_key,
                method="tsne",
                num_dims=2,
            )
            brain_results["tsne"] = True
            logging.info("✓ t-SNE visualization computed")
    except Exception as e:
        logging.warning("Failed to compute t-SNE: %s", e)
        brain_results["tsne"] = False

    # 5. Compute near duplicates
    logging.info("Computing near duplicates...")
    try:
        dup_results = fob.compute_near_duplicates(
            dataset,
            embeddings=embeddings_field,
            thresh=0.95,
        )
        brain_results["near_duplicates"] = True
        brain_results["near_duplicates_count"] = len(dup_results)
        logging.info("✓ Near duplicates computed: %d found", len(dup_results))
    except Exception as e:
        logging.warning("Failed to compute near duplicates: %s", e)
        brain_results["near_duplicates"] = False

    # 6. Compute exact duplicates
    logging.info("Computing exact duplicates...")
    try:
        exact_dup_results = fob.compute_exact_duplicates(
            dataset,
        )
        brain_results["exact_duplicates"] = True
        brain_results["exact_duplicates_count"] = len(exact_dup_results)
        logging.info("✓ Exact duplicates computed: %d found", len(exact_dup_results))
    except Exception as e:
        logging.warning("Failed to compute exact duplicates: %s", e)
        brain_results["exact_duplicates"] = False

    # 7. Compute leaky splits
    logging.info("Computing leaky splits using FiftyOne...")
    try:
        leaky_results = fob.compute_leaky_splits(
            dataset,
            splits="split",
            embeddings=embeddings_field,
        )
        brain_results["leaky_splits"] = True
        brain_results["leaky_splits_info"] = str(leaky_results)
        logging.info("✓ Leaky splits computed")
    except Exception as e:
        logging.warning("Failed to compute leaky splits: %s", e)
        brain_results["leaky_splits"] = False

    report = {
        "dataset_name": dataset_name,
        "num_samples": len(dataset),
        "persistent": persistent,
        "embedding_model": model_name,
        "embeddings_field": embeddings_field,
        "brain_results": brain_results,
    }

    # Count successful brain computations (True values only)
    successful_metrics = sum(1 for v in brain_results.values() if v is True)
    logging.info("Created FiftyOne dataset with %d samples and %d brain metrics", len(dataset), successful_metrics)

    return dataset, report


def launch_fiftyone_app(dataset: fo.Dataset, model_name: str = "resnet50") -> None:
    """Launch FiftyOne app for manual inspection with all brain features.

    Args:
        dataset: FiftyOne dataset to inspect
        model_name: Name of embedding model used
    """
    logging.info("")
    logging.info("=" * 80)
    logging.info("Launching FiftyOne app for dataset: %s", dataset.name)
    logging.info("=" * 80)
    logging.info("")
    logging.info("Available brain metrics and visualizations:")
    logging.info("")
    logging.info("FIELDS:")
    logging.info("  • uniqueness - How unique each image is (0-1)")
    logging.info("  • representativeness - How representative of nearby samples (0-1)")
    logging.info("  • jaguar_id - Individual jaguar identifier")
    logging.info("  • split - train/val/test assignment")
    logging.info("  • is_duplicate - Marked as duplicate")
    logging.info("  • is_leaky - Found in multiple splits")
    logging.info("")
    logging.info("SIMILARITY SEARCH:")
    logging.info("  1. Select an image")
    logging.info("  2. Click 'Sort by similarity' to find similar images")
    logging.info("  Brain key: %s_similarity", model_name)
    logging.info("")
    logging.info("EMBEDDINGS VISUALIZATION (click 'Embeddings' tab):")
    logging.info("  • %s_pca - PCA 2D projection", model_name)
    logging.info("  • %s_umap - UMAP 2D projection", model_name)
    logging.info("  • %s_tsne - t-SNE 2D projection", model_name)
    logging.info("  Color by: jaguar_id, split, uniqueness, representativeness")
    logging.info("")
    logging.info("DUPLICATE DETECTION:")
    logging.info("  • Near duplicates marked by FiftyOne")
    logging.info("  • Exact duplicates detected")
    logging.info("  • Leaky splits computed across train/val/test")
    logging.info("")
    logging.info("FILTERS (try these):")
    logging.info("  • is_duplicate == True")
    logging.info("  • is_leaky == True")
    logging.info("  • uniqueness < 0.5")
    logging.info("  • representativeness > 0.8")
    logging.info("  • jaguar_id == 'specific_id'")
    logging.info("")
    logging.info("=" * 80)
    logging.info("")

    session = fo.launch_app(dataset)

    # Keep session open
    logging.info("FiftyOne app launched. Press Ctrl+C to exit.")
    session.wait()


def check_required_columns(df: pd.DataFrame, split_column: str) -> pd.DataFrame:
    """Verify required columns exist in the dataframe."""
    required_columns = ["FILE PATH", "JAGUAR ID", split_column]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing_columns)}")

    return df


def load_df(input_csv: Path, split_column: str) -> tuple[pd.DataFrame, dict]:
    """Load and validate input CSV file."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    logging.info("Loading dataset from %s", input_csv)

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        logging.error("Failed to load CSV file: %s", e)
        raise

    df = check_required_columns(df, split_column)

    # Filter to only images
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
        "split_column": split_column,
        "splits": df[split_column].value_counts().to_dict(),
    }

    logging.info("Loaded %d image samples with %d unique jaguar IDs", report["total_samples"], report["total_ids"])
    logging.info("Split distribution: %s", report["splits"])

    return df, report


def run(
    input_csv: Path,
    output_csv: Path | None = None,
    split_column: str = "closed_set_split",
    train_threshold: float = 0.90,
    val_test_threshold: float = 0.95,
    leakage_threshold: float = 0.95,
    data_root: Path | None = None,
    inspect_only: bool = False,
    dataset_name: str = "jaguar_deduplication",
    model_name: str = "resnet101",
    force_recreate: bool = False,
    generate_report: bool = False,
) -> None:
    """Main processing function using FiftyOne workflow.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (defaults to input_csv with _deduplicated suffix)
        split_column: Column containing train/val/test assignments
        train_threshold: Similarity threshold for marking duplicates in training (0-1)
        val_test_threshold: Similarity threshold for marking duplicates in val/test (0-1)
        leakage_threshold: Similarity threshold for detecting cross-split leakage (0-1)
        data_root: Root directory to resolve relative file paths
        inspect_only: Skip computation and load existing FiftyOne dataset for inspection
        dataset_name: Name for the FiftyOne dataset
        model_name: Embedding model to use ('resnet50', 'resnet101', or 'clip')
        force_recreate: If True, delete and recreate dataset even if it exists
        generate_report: Whether to generate JSON report
    """
    if output_csv is None:
        output_csv = input_csv.parent / f"{input_csv.stem}_deduplicated.csv"

    if data_root is None:
        data_root = input_csv.parent

    # Inspect-only mode: load existing FiftyOne dataset
    if inspect_only:
        logging.info("Inspect-only mode: Loading existing FiftyOne dataset '%s'", dataset_name)

        if dataset_name not in fo.list_datasets():
            raise ValueError(f"FiftyOne dataset '{dataset_name}' not found. Run without --inspect_only first.")

        dataset = fo.load_dataset(dataset_name)
        launch_fiftyone_app(dataset, model_name)
        return

    # Full deduplication mode using FiftyOne
    report = {}

    # Load data
    df, report["input"] = load_df(input_csv, split_column)

    # Create FiftyOne dataset and compute all brain metrics
    # This will compute embeddings and all brain metrics in one go
    dataset, report["fiftyone"] = create_fiftyone_dataset(
        df, data_root, split_column, model_name, dataset_name, persistent=True, force_recreate=force_recreate
    )

    # Get embeddings field
    embeddings_field = f"{model_name}_embeddings"

    # Detect duplicates within each split using our custom logic
    # Note: FiftyOne's near_duplicates already ran in create_fiftyone_dataset
    # but we also run custom per-split duplicate detection
    dataset, report["train_dedup"] = detect_duplicates_within_split(dataset, embeddings_field, "train", train_threshold)
    dataset, report["val_dedup"] = detect_duplicates_within_split(dataset, embeddings_field, "val", val_test_threshold)
    dataset, report["test_dedup"] = detect_duplicates_within_split(dataset, embeddings_field, "test", val_test_threshold)

    # Leaky splits already computed in create_fiftyone_dataset via FiftyOne brain
    dataset, report["leakage"] = detect_leaky_splits(dataset, embeddings_field, leakage_threshold)

    # Export results back to CSV if needed
    # Extract duplicate info from dataset
    duplicate_data = []
    files_root = (data_root / "files").resolve()
    for sample in dataset:
        # FiftyOne samples don't have .get() method, use has_field() instead
        is_duplicate = sample["is_duplicate"] if sample.has_field("is_duplicate") else False
        duplicate_of = sample["duplicate_of"] if sample.has_field("duplicate_of") else None
        duplicate_similarity = sample["duplicate_similarity"] if sample.has_field("duplicate_similarity") else 0.0

        duplicate_data.append(
            {
                "FILE PATH": Path(sample.filepath).relative_to(files_root).as_posix(),
                "is_duplicate": is_duplicate,
                "duplicate_of": duplicate_of,
                "duplicate_similarity": duplicate_similarity,
            }
        )

    # Merge with original dataframe
    dup_df = pd.DataFrame(duplicate_data)
    df = df.merge(dup_df, on="FILE PATH", how="left")

    # Write output
    report["output"] = write_df(df, output_csv)

    if generate_report:
        report_path = output_csv.parent / f"{output_csv.stem}_report.json"
        write_report(report, report_path)

    # Launch FiftyOne app
    logging.info("\n" + "=" * 80)
    logging.info("Deduplication complete!")
    logging.info("Review results in FiftyOne app...")
    logging.info("=" * 80 + "\n")
    launch_fiftyone_app(dataset, model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate jaguar identification dataset using image embeddings.")
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
        help="Path to output CSV file (defaults to input_csv with _deduplicated suffix).",
    )
    parser.add_argument(
        "--split_column",
        type=str,
        default="closed_set_split",
        help="Column containing train/val/test split assignments (default: closed_set_split).",
    )
    parser.add_argument(
        "--train_threshold",
        type=float,
        default=0.90,
        help="Similarity threshold for marking duplicates in training set (0-1, default: 0.90).",
    )
    parser.add_argument(
        "--val_test_threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for marking duplicates in val/test sets (0-1, default: 0.95).",
    )
    parser.add_argument(
        "--leakage_threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for detecting cross-split leakage (0-1, default: 0.95).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory to resolve relative file paths (defaults to input_csv parent).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet101",
        choices=["resnet50", "resnet101", "clip"],
        help="Embedding model to use (default: resnet101). resnet101 recommended for fine-grained jaguar distinctions.",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Skip computation and load existing FiftyOne dataset for manual inspection.",
    )
    parser.add_argument(
        "--force_recreate",
        action="store_true",
        help="Force delete and recreate the dataset even if it exists.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jaguar_deduplication",
        help="Name for the FiftyOne dataset (default: jaguar_deduplication).",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate a JSON report with deduplication statistics.",
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
        split_column=args.split_column,
        train_threshold=args.train_threshold,
        val_test_threshold=args.val_test_threshold,
        leakage_threshold=args.leakage_threshold,
        data_root=Path(args.data_root) if args.data_root else None,
        inspect_only=args.inspect_only,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        force_recreate=args.force_recreate,
        generate_report=args.generate_report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
