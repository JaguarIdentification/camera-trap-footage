"""Tests for dataset module."""

import numpy as np
import torch

from jaguars.reidentification.dataset import EmbeddingDataset


def test_embedding_dataset_init() -> None:
    """Test EmbeddingDataset initialization."""
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    labels = [0, 1]

    dataset = EmbeddingDataset(embeddings, labels)

    assert len(dataset) == 2
    assert dataset.embeddings.shape == (2, 3)
    assert dataset.labels.shape == (2,)


def test_embedding_dataset_getitem() -> None:
    """Test getting items from dataset."""
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    labels = [0, 1, 0]

    dataset = EmbeddingDataset(embeddings, labels)

    emb, label = dataset[0]
    assert torch.allclose(emb, torch.FloatTensor([1.0, 2.0, 3.0]))
    assert label == 0

    emb, label = dataset[1]
    assert torch.allclose(emb, torch.FloatTensor([4.0, 5.0, 6.0]))
    assert label == 1

    emb, label = dataset[2]
    assert torch.allclose(emb, torch.FloatTensor([7.0, 8.0, 9.0]))
    assert label == 0


def test_embedding_dataset_len() -> None:
    """Test dataset length."""
    embeddings = [[1.0] * 128 for _ in range(50)]
    labels = list(range(50))

    dataset = EmbeddingDataset(embeddings, labels)

    assert len(dataset) == 50


def test_embedding_dataset_types() -> None:
    """Test that dataset returns correct tensor types."""
    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    labels = [0, 1]

    dataset = EmbeddingDataset(embeddings, labels)

    emb, label = dataset[0]
    assert isinstance(emb, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert emb.dtype == torch.float32
    assert label.dtype == torch.int64


def test_embedding_dataset_with_numpy() -> None:
    """Test dataset with numpy arrays as input."""
    embeddings_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    labels_np = np.array([0, 1])

    # Convert to list for initialization
    dataset = EmbeddingDataset(embeddings_np.tolist(), labels_np.tolist())

    assert len(dataset) == 2
    emb, label = dataset[0]
    assert torch.allclose(emb, torch.FloatTensor([1.0, 2.0, 3.0]))


def test_embedding_dataset_indexing() -> None:
    """Test various indexing operations."""
    embeddings = [[float(i)] * 10 for i in range(20)]
    labels = [i % 5 for i in range(20)]

    dataset = EmbeddingDataset(embeddings, labels)

    # Test first and last items
    emb_first, label_first = dataset[0]
    assert torch.allclose(emb_first, torch.zeros(10))

    emb_last, label_last = dataset[19]
    assert torch.allclose(emb_last, torch.full((10,), 19.0))

    # Test negative indexing
    emb_neg, label_neg = dataset[-1]
    assert torch.allclose(emb_neg, emb_last)


def test_embedding_dataset_empty() -> None:
    """Test dataset with empty data."""
    dataset = EmbeddingDataset([], [])

    assert len(dataset) == 0
    assert dataset.embeddings.shape == (0,)
    assert dataset.labels.shape == (0,)
