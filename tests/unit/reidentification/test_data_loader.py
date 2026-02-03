"""Tests for dataset loading utilities."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jaguars.reidentification.config import DatasetConfig
from jaguars.reidentification.data_loader import (
    DatasetMetadata,
    load_dataset,
    load_from_disk,
    save_metadata,
)


@pytest.fixture
def sample_disk_dataset() -> Iterator[Path]:
    """Create a temporary disk-based dataset."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir)

    # Create directory structure
    for split in ["train", "val", "test"]:
        for class_name in ["class_a", "class_b"]:
            class_dir = data_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Create sample images
            for i in range(2):
                img = Image.new("RGB", (100, 100), color=(i * 100, i * 100, i * 100))
                img_path = class_dir / f"img_{i}.jpg"
                img.save(img_path)

    yield data_dir

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


def test_dataset_metadata_init() -> None:
    """Test DatasetMetadata initialization."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    labels = ["cat", "dog", "cat"]

    metadata = DatasetMetadata(image_paths=image_paths, labels=labels)

    assert len(metadata) == 3
    assert metadata.num_classes == 2
    assert list(metadata.label_encoder.classes_) == ["cat", "dog"]
    assert len(metadata.labels_encoded) == 3


def test_dataset_metadata_with_embeddings() -> None:
    """Test DatasetMetadata with embeddings."""
    image_paths = ["img1.jpg", "img2.jpg"]
    labels = ["class_a", "class_a"]
    embeddings = np.random.randn(2, 128)

    metadata = DatasetMetadata(image_paths=image_paths, labels=labels, embeddings=embeddings)

    assert metadata.embeddings is not None
    assert metadata.embeddings.shape == (2, 128)


def test_dataset_metadata_with_splits() -> None:
    """Test DatasetMetadata with split information."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    labels = ["a", "b", "a"]
    splits = ["train", "train", "val"]

    metadata = DatasetMetadata(image_paths=image_paths, labels=labels, split=splits)

    assert metadata.split == splits


def test_dataset_metadata_get_split() -> None:
    """Test getting a specific split from metadata."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    labels = ["a", "b", "a", "b"]
    splits = ["train", "train", "val", "val"]
    embeddings = np.random.randn(4, 64)

    metadata = DatasetMetadata(image_paths=image_paths, labels=labels, embeddings=embeddings, split=splits)

    train_metadata = metadata.get_split("train")

    assert len(train_metadata) == 2
    assert train_metadata.image_paths == ["img1.jpg", "img2.jpg"]
    assert train_metadata.labels == ["a", "b"]
    assert train_metadata.embeddings is not None
    assert train_metadata.embeddings.shape == (2, 64)
    assert train_metadata.num_classes == metadata.num_classes  # Should share label encoder


def test_load_from_disk(sample_disk_dataset: Path) -> None:
    """Test loading dataset from disk."""
    config = DatasetConfig(
        source="disk",
        data_dir=sample_disk_dataset,
        train_split="train",
        val_split="val",
        test_split="test",
    )

    metadata = load_from_disk(config)

    assert len(metadata) == 12  # 2 classes * 2 images * 3 splits
    assert metadata.num_classes == 2
    assert set(metadata.split) == {"train", "val", "test"}


def test_load_from_disk_missing_dir() -> None:
    """Test loading from non-existent directory."""
    config = DatasetConfig(source="disk", data_dir=Path("/nonexistent/path"))

    # Should handle missing splits gracefully
    metadata = load_from_disk(config)
    assert len(metadata) == 0


def test_load_from_disk_no_data_dir() -> None:
    """Test that missing data_dir raises error."""
    config = DatasetConfig(source="disk", data_dir=None)

    with pytest.raises(ValueError, match="data_dir must be specified"):
        load_from_disk(config)


def test_load_dataset_disk(sample_disk_dataset: Path) -> None:
    """Test load_dataset with disk source."""
    config = DatasetConfig(source="disk", data_dir=sample_disk_dataset)

    metadata = load_dataset(config)

    assert isinstance(metadata, DatasetMetadata)
    assert len(metadata) > 0


def test_load_dataset_unsupported_source() -> None:
    """Test load_dataset with unsupported source."""
    config = DatasetConfig(source="unsupported")  # type: ignore

    with pytest.raises(ValueError, match="Unsupported dataset source"):
        load_dataset(config)


def test_save_metadata() -> None:
    """Test saving metadata to JSON."""
    temp_dir = tempfile.mkdtemp()
    output_path = Path(temp_dir) / "metadata.json"

    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    labels = ["cat", "dog", "cat"]
    metadata = DatasetMetadata(image_paths=image_paths, labels=labels)

    save_metadata(metadata, output_path)

    assert output_path.exists()

    # Read and verify
    import json

    with open(output_path) as f:
        data = json.load(f)

    assert data["num_samples"] == 3
    assert data["num_classes"] == 2
    assert set(data["classes"]) == {"cat", "dog"}

    # Cleanup
    output_path.unlink()
    Path(temp_dir).rmdir()


def test_dataset_metadata_label_encoding() -> None:
    """Test that labels are properly encoded."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    labels = ["jaguar_a", "jaguar_b", "jaguar_a"]

    metadata = DatasetMetadata(image_paths=image_paths, labels=labels)

    # Check encoding
    assert metadata.labels_encoded[0] == metadata.labels_encoded[2]  # Same label
    assert metadata.labels_encoded[0] != metadata.labels_encoded[1]  # Different label

    # Check inverse transform
    decoded = metadata.label_encoder.inverse_transform(metadata.labels_encoded)
    assert list(decoded) == labels


def test_dataset_metadata_empty() -> None:
    """Test DatasetMetadata with empty data."""
    metadata = DatasetMetadata(image_paths=[], labels=[])

    assert len(metadata) == 0
    assert metadata.num_classes == 0
    assert len(metadata.labels_encoded) == 0
