"""Tests for backbone feature extractors."""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
from collections.abc import Iterator

from jaguars.reidentification.config import BackboneConfig
from jaguars.reidentification.backbone import (
    BackboneInterface,
    MegaDescriptorBackbone,
    get_backbone,
)


@pytest.fixture
def backbone_config() -> BackboneConfig:
    """Create backbone configuration for testing."""
    # Use a much smaller vision transformer for testing to avoid OOM
    # vit_tiny_patch16_224 is much smaller than the default models
    return BackboneConfig(
        name="vit_tiny_patch16_224",
        pretrained=False,  # Use False for faster testing
        input_size=224,  # Standard ViT input size
        batch_size=2,
        embedding_dim=192,  # vit_tiny output dimension
    )


@pytest.fixture
def sample_images() -> Iterator[list[str]]:
    """Create temporary sample images for testing."""
    temp_dir = tempfile.mkdtemp()
    image_paths = []

    for i in range(3):
        img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
        img_path = Path(temp_dir) / f"test_image_{i}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))

    yield image_paths

    # Cleanup
    for path in image_paths:
        Path(path).unlink()
    Path(temp_dir).rmdir()


def test_megadescriptor_backbone_init(backbone_config: BackboneConfig) -> None:
    """Test MegaDescriptorBackbone initialization."""
    backbone = MegaDescriptorBackbone(backbone_config)

    assert backbone is not None
    assert isinstance(backbone, BackboneInterface)
    # Embedding dim is auto-detected from the model, so just verify it's positive
    assert backbone.config.embedding_dim > 0


def test_megadescriptor_forward(backbone_config: BackboneConfig) -> None:
    """Test forward pass through backbone."""
    backbone = MegaDescriptorBackbone(backbone_config)
    backbone.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        output = backbone(x)

    assert output.shape == (batch_size, backbone_config.embedding_dim)
    assert not torch.isnan(output).any()


def test_megadescriptor_preprocessing(backbone_config: BackboneConfig) -> None:
    """Test preprocessing pipeline."""
    backbone = MegaDescriptorBackbone(backbone_config)
    preprocess = backbone.get_preprocess()

    # Create a sample image
    img = Image.new("RGB", (512, 512), color=(100, 100, 100))

    # Apply preprocessing
    tensor = preprocess(img)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_backbone_get_embedding_dim(backbone_config: BackboneConfig) -> None:
    """Test getting embedding dimension."""
    backbone = MegaDescriptorBackbone(backbone_config)
    dim = backbone.get_embedding_dim()

    assert dim == backbone_config.embedding_dim


@pytest.mark.slow
def test_extract_embeddings(backbone_config: BackboneConfig, sample_images: list[str]) -> None:
    """Test extracting embeddings from image paths."""
    backbone = MegaDescriptorBackbone(backbone_config)
    backbone.eval()

    embeddings = backbone.extract_embeddings(sample_images, device="cpu", desc="Test")

    assert embeddings.shape == (len(sample_images), backbone_config.embedding_dim)
    assert not np.isnan(embeddings).any()


def test_get_backbone_factory(backbone_config: BackboneConfig) -> None:
    """Test backbone factory function."""
    backbone = get_backbone(backbone_config, device="cpu")

    assert isinstance(backbone, MegaDescriptorBackbone)
    assert backbone.config.name == backbone_config.name


def test_get_backbone_unsupported_type() -> None:
    """Test factory with unsupported backbone type."""
    config = BackboneConfig(name="unsupported_backbone")

    # Should raise RuntimeError from timm when model doesn't exist
    with pytest.raises(RuntimeError, match="Unknown model"):
        get_backbone(config)


def test_backbone_consistency(backbone_config: BackboneConfig) -> None:
    """Test that backbone produces consistent outputs in eval mode."""
    backbone = MegaDescriptorBackbone(backbone_config)
    backbone.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output1 = backbone(x)
        output2 = backbone(x)

    # Outputs should be identical in eval mode
    assert torch.allclose(output1, output2)


def test_backbone_batch_sizes(backbone_config: BackboneConfig) -> None:
    """Test backbone with different batch sizes."""
    backbone = MegaDescriptorBackbone(backbone_config)
    backbone.eval()

    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = backbone(x)

        assert output.shape == (batch_size, backbone_config.embedding_dim)


def test_backbone_device_cpu(backbone_config: BackboneConfig) -> None:
    """Test backbone on CPU."""
    backbone = get_backbone(backbone_config, device="cpu")

    assert next(backbone.parameters()).device.type == "cpu"

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = backbone(x)

    assert output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backbone_device_cuda(backbone_config: BackboneConfig) -> None:
    """Test backbone on CUDA."""
    backbone = get_backbone(backbone_config, device="cuda")

    assert next(backbone.parameters()).device.type == "cuda"

    x = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        output = backbone(x)

    assert output.device.type == "cuda"
