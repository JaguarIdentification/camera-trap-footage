"""Dataset loading utilities for jaguar re-identification.

Supports loading from:
- Disk (image folders)
- HuggingFace datasets
- FiftyOne datasets
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from jaguars.reidentification.config import DatasetConfig

logger = logging.getLogger(__name__)


class DatasetMetadata:
    """Container for dataset metadata."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[str],
        embeddings: np.ndarray | None = None,
        split: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.embeddings = embeddings
        self.split = split if split is not None else ["train"] * len(image_paths)
        self.metadata = metadata if metadata is not None else {}

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)

    def get_split(self, split_name: str) -> "DatasetMetadata":
        """Get subset for specific split.

        Args:
            split_name: Name of split (train/val/test)

        Returns:
            New DatasetMetadata with filtered data
        """
        mask = np.array([s == split_name for s in self.split])
        indices = np.where(mask)[0]

        new_paths = [self.image_paths[i] for i in indices]
        new_labels = [self.labels[i] for i in indices]
        new_embeddings = self.embeddings[indices] if self.embeddings is not None else None
        new_split = [self.split[i] for i in indices]

        result = DatasetMetadata(
            image_paths=new_paths, labels=new_labels, embeddings=new_embeddings, split=new_split, metadata=self.metadata
        )
        # Use same label encoder
        result.label_encoder = self.label_encoder
        result.labels_encoded = self.label_encoder.transform(new_labels)
        result.num_classes = self.num_classes

        return result

    def __len__(self) -> int:
        return len(self.image_paths)


def load_from_disk(config: DatasetConfig) -> DatasetMetadata:
    """Load dataset from disk.

    Expected structure:
        data_dir/
            train/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    ...
            val/
            test/

    Args:
        config: Dataset configuration

    Returns:
        DatasetMetadata object
    """
    if config.data_dir is None:
        raise ValueError("data_dir must be specified for disk-based datasets")

    data_dir = Path(config.data_dir)
    image_paths = []
    labels = []
    splits = []

    for split in [config.train_split, config.val_split, config.test_split]:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        # Iterate through class folders
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            # Get all images in class folder
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_paths.append(str(img_path))
                    labels.append(class_name)
                    splits.append(split)

    logger.info(f"Loaded {len(image_paths)} images from disk")
    logger.info(f"  Train: {sum(s == config.train_split for s in splits)}")
    logger.info(f"  Val: {sum(s == config.val_split for s in splits)}")
    logger.info(f"  Test: {sum(s == config.test_split for s in splits)}")

    return DatasetMetadata(image_paths=image_paths, labels=labels, split=splits)


def load_from_huggingface(config: DatasetConfig) -> DatasetMetadata:
    """Load dataset from HuggingFace.

    Args:
        config: Dataset configuration

    Returns:
        DatasetMetadata object
    """
    if config.hf_repo is None:
        raise ValueError("hf_repo must be specified for HuggingFace datasets")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required for HuggingFace loading. Install with: pip install datasets")

    logger.info(f"Loading dataset from HuggingFace: {config.hf_repo}")

    dataset = load_dataset(config.hf_repo, revision=config.hf_revision)

    image_paths = []
    labels = []
    splits = []

    # Process each split
    for split_name in dataset.keys():
        split_data = dataset[split_name]

        for sample in split_data:
            # Extract image path or download image
            if config.image_field in sample:
                image_paths.append(sample[config.image_field])
            else:
                raise ValueError(f"Image field '{config.image_field}' not found in dataset")

            # Extract label
            if config.label_field in sample:
                labels.append(str(sample[config.label_field]))
            else:
                raise ValueError(f"Label field '{config.label_field}' not found in dataset")

            splits.append(split_name)

    logger.info(f"Loaded {len(image_paths)} images from HuggingFace")

    return DatasetMetadata(image_paths=image_paths, labels=labels, split=splits)


def load_from_fiftyone(config: DatasetConfig) -> DatasetMetadata:
    """Load dataset from FiftyOne.

    Args:
        config: Dataset configuration

    Returns:
        DatasetMetadata object
    """
    try:
        import fiftyone as fo
    except ImportError:
        raise ImportError("fiftyone library required. Install with: pip install fiftyone")

    logger.info(f"Loading dataset from FiftyOne: {config.fo_dataset_name}")

    # Load dataset
    if not fo.dataset_exists(config.fo_dataset_name):
        raise ValueError(f"FiftyOne dataset '{config.fo_dataset_name}' does not exist")

    dataset = fo.load_dataset(config.fo_dataset_name)

    image_paths = []
    labels = []
    embeddings_list: list[np.ndarray | None] = []
    splits = []

    # Check if we need to load from patches or samples
    use_patches = config.fo_patches_field is not None

    for sample in dataset:
        if use_patches:
            # Load from patches (e.g., segmentations)
            if not hasattr(sample, config.fo_patches_field):
                continue

            patches = getattr(sample, config.fo_patches_field)
            if patches is None or patches.detections is None:
                continue

            for detection in patches.detections:
                # Get label
                if hasattr(detection, config.fo_label_field):
                    label = getattr(detection, config.fo_label_field)
                    if label is None:
                        continue
                    labels.append(str(label))
                else:
                    continue

                # Get image path (from parent sample)
                image_paths.append(sample.filepath)

                # Get embedding if available
                if hasattr(detection, config.fo_embeddings_field):
                    emb = getattr(detection, config.fo_embeddings_field)
                    if emb is not None:
                        embeddings_list.append(np.array(emb))
                    else:
                        embeddings_list.append(None)
                else:
                    embeddings_list.append(None)

                # Get split
                if hasattr(sample, config.fo_split_field):
                    split = getattr(sample, config.fo_split_field)
                    splits.append(split if split is not None else "train")
                else:
                    splits.append("train")
        else:
            # Load from samples
            if not hasattr(sample, config.fo_label_field):
                continue

            label = getattr(sample, config.fo_label_field)
            if label is None:
                continue

            labels.append(str(label))
            image_paths.append(sample.filepath)

            # Get embedding
            if hasattr(sample, config.fo_embeddings_field):
                emb = getattr(sample, config.fo_embeddings_field)
                if emb is not None:
                    embeddings_list.append(np.array(emb))
                else:
                    embeddings_list.append(None)
            else:
                embeddings_list.append(None)

            # Get split
            if hasattr(sample, config.fo_split_field):
                split = getattr(sample, config.fo_split_field)
                splits.append(split if split is not None else "train")
            else:
                splits.append("train")

    # Convert embeddings to array if all are present
    embeddings = None
    if all(e is not None for e in embeddings_list):
        embeddings = np.array(embeddings_list)
        logger.info(f"Loaded embeddings with shape {embeddings.shape}")
    elif any(e is not None for e in embeddings_list):
        logger.warning("Some embeddings missing, will recompute all")

    logger.info(f"Loaded {len(image_paths)} samples from FiftyOne")
    logger.info(f"  Unique labels: {len(set(labels))}")
    logger.info(f"  Splits: {dict(pd.Series(splits).value_counts())}")

    return DatasetMetadata(image_paths=image_paths, labels=labels, embeddings=embeddings, split=splits)


def load_dataset(config: DatasetConfig) -> DatasetMetadata:
    """Load dataset based on configuration.

    Args:
        config: Dataset configuration

    Returns:
        DatasetMetadata object
    """
    if config.source == "disk":
        return load_from_disk(config)
    elif config.source == "huggingface":
        return load_from_huggingface(config)
    elif config.source == "fiftyone":
        return load_from_fiftyone(config)
    else:
        raise ValueError(f"Unsupported dataset source: {config.source}")


def save_metadata(metadata: DatasetMetadata, output_path: Path) -> None:
    """Save dataset metadata to JSON.

    Args:
        metadata: Dataset metadata
        output_path: Path to save metadata
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "num_samples": len(metadata),
        "num_classes": metadata.num_classes,
        "classes": metadata.label_encoder.classes_.tolist(),
        "split_counts": dict(pd.Series(metadata.split).value_counts()),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved metadata to {output_path}")
