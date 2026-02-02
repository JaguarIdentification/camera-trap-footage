"""Tests for re-identification model components."""

import pytest
import torch

from jaguars.reidentification.config import ModelConfig
from jaguars.reidentification.model import (
    EmbeddingProjection,
    ArcFaceLayer,
    ArcFaceModel,
    build_model,
)


@pytest.fixture
def model_config() -> ModelConfig:
    """Create a model configuration for testing."""
    return ModelConfig(
        hidden_dim=128,
        embedding_dim=64,
        dropout=0.2,
        arcface_margin=0.5,
        arcface_scale=32.0,
    )


def test_embedding_projection() -> None:
    """Test EmbeddingProjection module."""
    input_dim = 512
    hidden_dim = 256
    output_dim = 128

    model = EmbeddingProjection(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=0.3
    )

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    output = model(x)

    assert output.shape == (batch_size, output_dim)
    assert not torch.isnan(output).any()


def test_arcface_layer() -> None:
    """Test ArcFaceLayer."""
    embedding_dim = 128
    num_classes = 10
    batch_size = 4

    layer = ArcFaceLayer(
        embedding_dim=embedding_dim, num_classes=num_classes, margin=0.5, scale=32.0
    )

    # Create dummy data
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    logits = layer(embeddings, labels)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()


def test_arcface_layer_normalization() -> None:
    """Test that ArcFace normalizes embeddings."""
    embedding_dim = 64
    num_classes = 5
    batch_size = 2

    layer = ArcFaceLayer(embedding_dim=embedding_dim, num_classes=num_classes)

    # Embeddings with different magnitudes
    embeddings = torch.randn(batch_size, embedding_dim) * 10
    labels = torch.tensor([0, 1])

    logits = layer(embeddings, labels)

    # Should still work (normalized internally)
    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()


def test_arcface_model(model_config: ModelConfig) -> None:
    """Test complete ArcFaceModel."""
    input_dim = 512
    num_classes = 20
    batch_size = 8

    model = ArcFaceModel(input_dim=input_dim, num_classes=num_classes, config=model_config)

    # Test forward pass (training mode)
    x = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    logits, embeddings = model(x, labels)

    assert logits.shape == (batch_size, num_classes)
    assert embeddings.shape == (batch_size, model_config.embedding_dim)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(embeddings).any()


def test_arcface_model_get_embeddings(model_config: ModelConfig) -> None:
    """Test getting normalized embeddings."""
    input_dim = 512
    num_classes = 10
    batch_size = 4

    model = ArcFaceModel(input_dim=input_dim, num_classes=num_classes, config=model_config)

    x = torch.randn(batch_size, input_dim)
    embeddings = model.get_embeddings(x)

    assert embeddings.shape == (batch_size, model_config.embedding_dim)

    # Check that embeddings are normalized (L2 norm should be ~1)
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_build_model(model_config: ModelConfig) -> None:
    """Test model factory function."""
    input_dim = 1536
    num_classes = 50

    model = build_model(input_dim=input_dim, num_classes=num_classes, config=model_config)

    assert isinstance(model, ArcFaceModel)
    assert model.embedding_net is not None
    assert model.arcface is not None

    # Test that model can process data
    x = torch.randn(2, input_dim)
    labels = torch.tensor([0, 1])

    logits, _ = model(x, labels)
    assert logits.shape == (2, num_classes)


def test_model_training_mode() -> None:
    """Test model in training mode."""
    config = ModelConfig()
    model = ArcFaceModel(input_dim=512, num_classes=10, config=config)

    model.train()
    assert model.training

    # Dropout should be active in training mode
    x = torch.randn(4, 512)
    labels = torch.tensor([0, 1, 2, 3])

    logits1, _ = model(x, labels)
    logits2, _ = model(x, labels)

    # Outputs might differ due to dropout
    # (This is non-deterministic, so we just check they're valid)
    assert not torch.isnan(logits1).any()
    assert not torch.isnan(logits2).any()


def test_model_eval_mode() -> None:
    """Test model in evaluation mode."""
    config = ModelConfig()
    model = ArcFaceModel(input_dim=512, num_classes=10, config=config)

    model.eval()
    assert not model.training

    # Outputs should be deterministic in eval mode
    x = torch.randn(4, 512)

    with torch.no_grad():
        emb1 = model.get_embeddings(x)
        emb2 = model.get_embeddings(x)

    assert torch.allclose(emb1, emb2)


def test_model_gradient_flow() -> None:
    """Test that gradients flow through the model."""
    config = ModelConfig()
    model = ArcFaceModel(input_dim=512, num_classes=10, config=config)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Forward pass
    x = torch.randn(4, 512)
    labels = torch.tensor([0, 1, 2, 3])

    logits, _ = model(x, labels)
    loss = criterion(logits, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check that gradients exist
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
