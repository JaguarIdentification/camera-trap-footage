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


# Mapping from backbone names to FiftyOne embedding field names
BACKBONE_TO_FIELD_MAP = {
    # MegaDescriptor models
    "hf-hub:BVRA/MegaDescriptor-L-384": "embeddings_BVRA_MegaDescriptor_L_384",
    "BVRA/MegaDescriptor_L_384": "embeddings_BVRA_MegaDescriptor_L_384",
    "hf-hub:BVRA/MegaDescriptor-B-224": "embeddings_BVRA_MegaDescriptor_B_224",
    # DINOv2 models
    "vit_large_patch14_dinov2.lvd142m": "embeddings_DINOv2_Large",
    "vit_base_patch14_dinov2.lvd142m": "embeddings_DINOv2_Base",
    "vit_small_patch14_dinov2.lvd142m": "embeddings_DINOv2_Small",
    # CNN models
    "resnet50": "embeddings_ResNet50",
    "convnext_base": "embeddings_ConvNeXt_Base",
    "convnextv2_base.fcmae_ft_in22k_in1k": "embeddings_ConvNeXtV2_Base",
    "efficientnet_b3": "embeddings_EfficientNet_B3",
}


def get_embedding_field_for_backbone(backbone_name: str) -> str | None:
    """Get FiftyOne embedding field name for a backbone.

    Args:
        backbone_name: Name of the backbone model

    Returns:
        Embedding field name, or None if no cached embeddings exist
    """
    return BACKBONE_TO_FIELD_MAP.get(backbone_name)


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

        result = DatasetMetadata(image_paths=new_paths, labels=new_labels, embeddings=new_embeddings, split=new_split, metadata=self.metadata)
        # Use same label encoder
        result.label_encoder = self.label_encoder
        result.labels_encoded = self.label_encoder.transform(new_labels)

        # Recompute num_classes
        result.num_classes = len(set(new_labels))

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

    Automatically detects cached embeddings for the configured backbone if available.
    If config.fo_embeddings_field is None, will try to find cached embeddings based
    on the backbone name using get_embedding_field_for_backbone().

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

    # Auto-detect embedding field from backbone name if not specified
    embedding_field = config.fo_embeddings_field
    if embedding_field is None and hasattr(config, "backbone") and hasattr(config.backbone, "name"):
        detected_field = get_embedding_field_for_backbone(config.backbone.name)
        if detected_field:
            logger.info(f"Auto-detected embedding field for backbone '{config.backbone.name}': {detected_field}")
            embedding_field = detected_field
        else:
            logger.info(f"No cached embeddings found for backbone '{config.backbone.name}', will compute on-the-fly")

    # Store the detected field back in config for downstream use
    if embedding_field:
        config.fo_embeddings_field = embedding_field

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
                # Get label - try detection first, fall back to sample
                label = None
                if hasattr(detection, config.fo_label_field):
                    label = getattr(detection, config.fo_label_field)

                # Fallback to sample-level label if not on detection
                if label is None and hasattr(sample, config.fo_label_field):
                    label = getattr(sample, config.fo_label_field)

                if label is None:
                    continue

                # Extract label string from Classification object if needed
                if hasattr(label, "label"):
                    label = label.label

                labels.append(str(label))

                # Get image path (from parent sample)
                image_paths.append(sample.filepath)

                # Get embedding if available
                if config.fo_embeddings_field is not None and hasattr(detection, config.fo_embeddings_field):
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
            label = None
            if hasattr(sample, config.fo_label_field):
                label = getattr(sample, config.fo_label_field)

            if label is None:
                continue

            # Extract label string from Classification object if needed
            if hasattr(label, "label"):
                label = label.label

            labels.append(str(label))
            image_paths.append(sample.filepath)

            # Get embedding
            if config.fo_embeddings_field is not None and hasattr(sample, config.fo_embeddings_field):
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

    # Convert numpy types to native Python types for JSON serialization
    split_counts = dict(pd.Series(metadata.split).value_counts())
    split_counts = {k: int(v) for k, v in split_counts.items()}

    data = {
        "num_samples": len(metadata),
        "num_classes": int(metadata.num_classes),
        "classes": metadata.label_encoder.classes_.tolist(),
        "split_counts": split_counts,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved metadata to {output_path}")
