"""Tests for evaluation metrics."""

import pytest
import numpy as np
from numpy.typing import NDArray

from jaguars.reidentification.config import ModelConfig
from jaguars.reidentification.model import ArcFaceModel
from jaguars.reidentification.evaluation.evaluation import (
    compute_validation_map,
    compute_cmc,
)
from sklearn.preprocessing import LabelEncoder


@pytest.fixture
def sample_model() -> ArcFaceModel:
    """Create a simple model for testing."""
    config = ModelConfig(embedding_dim=64, hidden_dim=128)
    model = ArcFaceModel(input_dim=128, num_classes=5, config=config)
    model.eval()
    return model


@pytest.fixture
def sample_embeddings() -> tuple[NDArray[np.float32], NDArray[np.int_], LabelEncoder]:
    """Create sample embeddings and labels."""
    np.random.seed(42)

    # 20 samples, 5 classes (4 samples per class)
    num_samples = 20
    embedding_dim = 128
    num_classes = 5

    embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    labels = np.array([i % num_classes for i in range(num_samples)])

    # Make embeddings from same class more similar
    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_embeddings = embeddings[class_mask]
        # Add a class-specific bias
        embeddings[class_mask] = class_embeddings + np.random.randn(1, embedding_dim) * 0.1

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    return embeddings, labels, label_encoder


def test_compute_validation_map(sample_model: ArcFaceModel, sample_embeddings: tuple[NDArray[np.float32], NDArray[np.int_], LabelEncoder]) -> None:
    """Test computing mean Average Precision."""
    embeddings, labels, label_encoder = sample_embeddings

    map_score = compute_validation_map(
        sample_model, embeddings, labels, label_encoder, device="cpu"
    )

    assert isinstance(map_score, float)
    assert 0.0 <= map_score <= 1.0
    assert not np.isnan(map_score)


def test_compute_validation_map_perfect_separation() -> None:
    """Test mAP with perfectly separated classes."""
    # Create perfectly separated embeddings
    num_classes = 3
    samples_per_class = 4
    embedding_dim = 64

    embeddings_list = []
    labels_list = []

    for class_id in range(num_classes):
        # Each class has embeddings in a different region
        class_center = np.zeros(embedding_dim)
        class_center[class_id] = 10.0  # Widely separated

        for _ in range(samples_per_class):
            emb = class_center + np.random.randn(embedding_dim) * 0.1
            embeddings_list.append(emb)
            labels_list.append(class_id)

    embeddings = np.array(embeddings_list, dtype=np.float32)
    labels = np.array(labels_list)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    config = ModelConfig(embedding_dim=64, hidden_dim=128)
    model = ArcFaceModel(input_dim=embedding_dim, num_classes=num_classes, config=config)
    model.eval()

    map_score = compute_validation_map(model, embeddings, labels, label_encoder, device="cpu")

    # Should be very high for well-separated classes
    assert map_score > 0.5


def test_compute_cmc_basic() -> None:
    """Test CMC computation with simple embeddings."""
    # Create simple embeddings where classes are separated
    embeddings = np.array(
        [
            [1, 0, 0],
            [1, 0.1, 0],  # Same class as first
            [0, 1, 0],
            [0, 1, 0.1],  # Same class as third
            [0, 0, 1],
            [0, 0.1, 1],  # Same class as fifth
        ],
        dtype=np.float32,
    )

    labels = np.array([0, 0, 1, 1, 2, 2])

    cmc_scores = compute_cmc(embeddings, labels, top_k=[1, 2, 3])

    assert isinstance(cmc_scores, dict)
    assert all(k in cmc_scores for k in [1, 2, 3])
    assert all(0.0 <= v <= 1.0 for v in cmc_scores.values())

    # CMC@k should be non-decreasing
    assert cmc_scores[1] <= cmc_scores[2] <= cmc_scores[3]


def test_compute_cmc_perfect_retrieval() -> None:
    """Test CMC with perfect retrieval."""
    # Each query's nearest neighbor is from the same class
    embeddings = np.array(
        [
            [1.0, 0, 0],
            [0.9, 0, 0],  # Very close to first
            [0, 1.0, 0],
            [0, 0.9, 0],  # Very close to third
        ],
        dtype=np.float32,
    )

    labels = np.array([0, 0, 1, 1])

    cmc_scores = compute_cmc(embeddings, labels, top_k=[1, 2])

    # Should have perfect CMC@1
    assert cmc_scores[1] == 1.0
    assert cmc_scores[2] == 1.0


def test_compute_cmc_no_matches() -> None:
    """Test CMC when no matches exist."""
    # Each sample is unique
    embeddings = np.eye(5, dtype=np.float32)
    labels = np.arange(5)

    cmc_scores = compute_cmc(embeddings, labels, top_k=[1, 3])

    # No matches possible (each sample is unique class)
    assert cmc_scores[1] == 0.0
    # For k=3, still no matches since each class has only 1 sample (the query itself)
    assert cmc_scores[3] == 0.0


def test_compute_cmc_custom_k() -> None:
    """Test CMC with custom top-k values."""
    embeddings = np.random.randn(10, 32).astype(np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    top_k = [1, 3, 5, 10]
    cmc_scores = compute_cmc(embeddings, labels, top_k=top_k)

    assert set(cmc_scores.keys()) == set(top_k)


def test_validation_map_single_class() -> None:
    """Test mAP with only one class."""
    embeddings = np.random.randn(5, 64).astype(np.float32)
    labels = np.zeros(5, dtype=int)  # All same class

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    config = ModelConfig()
    model = ArcFaceModel(input_dim=64, num_classes=1, config=config)
    model.eval()

    # This should handle the edge case
    map_score = compute_validation_map(model, embeddings, labels, label_encoder, device="cpu")

    assert isinstance(map_score, float)
    assert not np.isnan(map_score)


def test_validation_map_consistency() -> None:
    """Test that mAP is consistent across runs."""
    np.random.seed(123)
    embeddings = np.random.randn(10, 64).astype(np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    config = ModelConfig()
    model = ArcFaceModel(input_dim=64, num_classes=5, config=config)
    model.eval()

    # Compute mAP twice
    map1 = compute_validation_map(model, embeddings, labels, label_encoder, device="cpu")
    map2 = compute_validation_map(model, embeddings, labels, label_encoder, device="cpu")

    # Should be identical (deterministic in eval mode)
    assert map1 == map2
