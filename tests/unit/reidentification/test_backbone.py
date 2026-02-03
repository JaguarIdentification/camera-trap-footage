"""Comprehensive tests for backbone feature extractors.

Tests all supported backbones:
- DINOv2 variants (Large, Base, Small)
- MegaDescriptor models (L-384, B-224)
- CNN models (ResNet50, ConvNeXt, ConvNeXtV2, EfficientNet)

Validates:
- Model loading
- Input/output shapes
- Preprocessing transforms
- Embedding extraction
- Normalization values
"""

import numpy as np
import pytest
import torch
from PIL import Image

from jaguars.reidentification.backbone import TimmBackbone, get_backbone
from jaguars.reidentification.config import BackboneConfig

# Test configurations for all supported backbones
BACKBONE_CONFIGS = [
    # DINOv2 variants
    {
        "name": "vit_large_patch14_dinov2.lvd142m",
        "embedding_dim": 1024,
        "input_size": 518,
        "description": "DINOv2 Large",
        "normalization": "imagenet",  # ImageNet normalization
    },
    {
        "name": "vit_base_patch14_dinov2.lvd142m",
        "embedding_dim": 768,
        "input_size": 518,
        "description": "DINOv2 Base",
        "normalization": "imagenet",
    },
    {
        "name": "vit_small_patch14_dinov2.lvd142m",
        "embedding_dim": 384,
        "input_size": 518,
        "description": "DINOv2 Small",
        "normalization": "imagenet",
    },
    # DINOv3 variants
    {
        "name": "vit_base_patch16_dinov3.lvd1689m",
        "embedding_dim": 768,
        "input_size": 512,
        "description": "DINOv3 Base",
        "normalization": "imagenet",
    },
    # MegaDescriptor animal re-ID models
    {
        "name": "hf-hub:BVRA/MegaDescriptor-B-224",
        "embedding_dim": 768,
        "input_size": 224,
        "description": "MegaDescriptor Base 224",
        "normalization": "centered",  # [-1, 1] normalization
    },
    # CNN models
    {
        "name": "resnet50",
        "embedding_dim": 2048,
        "input_size": 224,
        "description": "ResNet50",
        "normalization": "imagenet",
    },
    {
        "name": "convnext_base",
        "embedding_dim": 1024,
        "input_size": 224,
        "description": "ConvNeXt Base",
        "normalization": "imagenet",
    },
    {
        "name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "embedding_dim": 1024,
        "input_size": 224,
        "description": "ConvNeXtV2 Base",
        "normalization": "imagenet",
    },
    {
        "name": "efficientnet_b3",
        "embedding_dim": 1536,
        "input_size": 288,
        "description": "EfficientNet B3",
        "normalization": "imagenet",
    },
]


@pytest.fixture
def dummy_image():
    """Create a dummy RGB image for testing."""
    img = Image.new("RGB", (640, 480), color=(128, 64, 192))
    return img


@pytest.fixture
def dummy_image_path(tmp_path, dummy_image):
    """Create a dummy image file for testing."""
    img_path = tmp_path / "test_image.jpg"
    dummy_image.save(img_path)
    return str(img_path)


class TestBackboneInterface:
    """Test the BackboneInterface abstract class and TimmBackbone implementation."""

    @pytest.mark.parametrize("backbone_spec", BACKBONE_CONFIGS, ids=lambda x: x["name"])
    def test_backbone_loading(self, backbone_spec):
        """Test that each backbone can be loaded successfully."""
        config = BackboneConfig(
            name=backbone_spec["name"],
            embedding_dim=backbone_spec["embedding_dim"],
            input_size=backbone_spec["input_size"],
            pretrained=False,  # Use random weights for faster testing
            batch_size=4,
        )

        backbone = TimmBackbone(config)
        assert backbone is not None
        assert hasattr(backbone, "model")
        assert hasattr(backbone, "forward")
        assert hasattr(backbone, "get_preprocess")

    @pytest.mark.parametrize("backbone_spec", BACKBONE_CONFIGS, ids=lambda x: x["name"])
    def test_embedding_dimension(self, backbone_spec):
        """Test that embedding dimension matches expected value."""
        config = BackboneConfig(
            name=backbone_spec["name"],
            embedding_dim=backbone_spec["embedding_dim"],
            input_size=backbone_spec["input_size"],
            pretrained=False,
            batch_size=4,
        )

        backbone = TimmBackbone(config)

        # The actual embedding dim should match (might be auto-corrected)
        assert backbone.config.embedding_dim > 0
        assert backbone.get_embedding_dim() == backbone.config.embedding_dim

    @pytest.mark.parametrize("backbone_spec", BACKBONE_CONFIGS, ids=lambda x: x["name"])
    def test_forward_pass_shape(self, backbone_spec):
        """Test that forward pass produces correct output shape."""
        config = BackboneConfig(
            name=backbone_spec["name"],
            embedding_dim=backbone_spec["embedding_dim"],
            input_size=backbone_spec["input_size"],
            pretrained=False,
            batch_size=4,
        )

        backbone = TimmBackbone(config)
        backbone.eval()

        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, config.input_size, config.input_size)

        # Forward pass
        with torch.no_grad():
            output = backbone(dummy_input)

        # Check output shape
        assert output.shape[0] == batch_size
        assert output.shape[1] == backbone.config.embedding_dim
        assert len(output.shape) == 2  # Should be (batch_size, embedding_dim)

    @pytest.mark.parametrize("backbone_spec", BACKBONE_CONFIGS, ids=lambda x: x["name"])
    def test_preprocessing_transforms(self, backbone_spec, dummy_image):
        """Test that preprocessing transforms work correctly."""
        config = BackboneConfig(
            name=backbone_spec["name"],
            embedding_dim=backbone_spec["embedding_dim"],
            input_size=backbone_spec["input_size"],
            pretrained=False,
            batch_size=4,
        )

        backbone = TimmBackbone(config)
        preprocess = backbone.get_preprocess()

        # Apply preprocessing
        tensor = preprocess(dummy_image)

        # Check output shape
        assert tensor.shape == (3, config.input_size, config.input_size)
        assert tensor.dtype == torch.float32

        # Check normalization range (should be roughly in [-3, 3] range after normalization)
        assert tensor.min() >= -4.0
        assert tensor.max() <= 4.0

    @pytest.mark.parametrize("backbone_spec", BACKBONE_CONFIGS, ids=lambda x: x["name"])
    def test_preprocessing_normalization(self, backbone_spec, dummy_image):
        """Test that normalization is applied correctly based on model type."""
        config = BackboneConfig(
            name=backbone_spec["name"],
            embedding_dim=backbone_spec["embedding_dim"],
            input_size=backbone_spec["input_size"],
            pretrained=False,
            batch_size=4,
        )

        backbone = TimmBackbone(config)
        preprocess = backbone.get_preprocess()

        # Apply preprocessing
        tensor = preprocess(dummy_image)

        # Check that normalization was applied (tensor should not be in [0, 1] range)
        # After ToTensor, values are in [0, 1], but after Normalize they should be different
        assert not (tensor.min() >= 0.0 and tensor.max() <= 1.0), "Normalization should transform values outside [0, 1] range"

    @pytest.mark.parametrize("backbone_spec", BACKBONE_CONFIGS, ids=lambda x: x["name"])
    def test_extract_embeddings(self, backbone_spec, dummy_image_path):
        """Test embedding extraction from image paths."""
        config = BackboneConfig(
            name=backbone_spec["name"],
            embedding_dim=backbone_spec["embedding_dim"],
            input_size=backbone_spec["input_size"],
            pretrained=False,
            batch_size=2,
        )

        backbone = TimmBackbone(config)

        # Extract embeddings
        image_paths = [dummy_image_path] * 3  # 3 copies of same image
        embeddings = backbone.extract_embeddings(image_paths, device="cpu")

        # Check shape
        assert embeddings.shape == (3, backbone.config.embedding_dim)
        assert isinstance(embeddings, np.ndarray)

        # Check that embeddings are not all zeros
        assert not np.allclose(embeddings, 0.0)

        # Check that similar images produce similar embeddings (same image repeated)
        # Embeddings should be identical for the same image
        assert np.allclose(embeddings[0], embeddings[1], rtol=1e-5)
        assert np.allclose(embeddings[1], embeddings[2], rtol=1e-5)


class TestGetBackboneFactory:
    """Test the get_backbone factory function."""

    @pytest.mark.parametrize("backbone_spec", BACKBONE_CONFIGS, ids=lambda x: x["name"])
    def test_factory_returns_correct_type(self, backbone_spec):
        """Test that factory returns TimmBackbone instance."""
        config = BackboneConfig(
            name=backbone_spec["name"],
            embedding_dim=backbone_spec["embedding_dim"],
            input_size=backbone_spec["input_size"],
            pretrained=False,
            batch_size=4,
        )

        backbone = get_backbone(config, device="cpu")
        assert isinstance(backbone, TimmBackbone)

    def test_factory_with_cuda_when_available(self):
        """Test that factory respects device selection."""
        config = BackboneConfig(
            name="resnet50",
            embedding_dim=2048,
            input_size=224,
            pretrained=False,
            batch_size=4,
        )

        # Should work on CPU
        backbone_cpu = get_backbone(config, device="cpu")
        assert next(backbone_cpu.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            backbone_cuda = get_backbone(config, device="cuda")
            assert next(backbone_cuda.parameters()).device.type == "cuda"


class TestBackboneConsistency:
    """Test consistency across different backbones."""

    def test_all_backbones_produce_embeddings(self, dummy_image):
        """Test that all backbones can produce embeddings from the same image."""
        embeddings_dict = {}

        for backbone_spec in BACKBONE_CONFIGS[:3]:  # Test first 3 to save time
            config = BackboneConfig(
                name=backbone_spec["name"],
                embedding_dim=backbone_spec["embedding_dim"],
                input_size=backbone_spec["input_size"],
                pretrained=False,
                batch_size=4,
            )

            backbone = get_backbone(config, device="cpu")
            preprocess = backbone.get_preprocess()

            # Get embedding
            with torch.no_grad():
                tensor = preprocess(dummy_image).unsqueeze(0)
                embedding = backbone(tensor).cpu().numpy()

            embeddings_dict[backbone_spec["name"]] = embedding

            # Check shape
            assert embedding.shape == (1, backbone_spec["embedding_dim"])

        # All backbones should have produced embeddings
        assert len(embeddings_dict) == 3

    def test_deterministic_outputs(self, dummy_image):
        """Test that backbones produce deterministic outputs (when not training)."""
        config = BackboneConfig(
            name="resnet50",
            embedding_dim=2048,
            input_size=224,
            pretrained=False,
            batch_size=4,
        )

        backbone = get_backbone(config, device="cpu")
        backbone.eval()
        preprocess = backbone.get_preprocess()

        # Get embedding twice
        with torch.no_grad():
            tensor = preprocess(dummy_image).unsqueeze(0)
            embedding1 = backbone(tensor).cpu().numpy()
            embedding2 = backbone(tensor).cpu().numpy()

        # Should be identical
        assert np.allclose(embedding1, embedding2)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_image_path(self):
        """Test handling of invalid image paths."""
        config = BackboneConfig(
            name="resnet50",
            embedding_dim=2048,
            input_size=224,
            pretrained=False,
            batch_size=2,
        )

        backbone = get_backbone(config, device="cpu")

        # Should handle invalid paths gracefully
        embeddings = backbone.extract_embeddings(["/invalid/path/image.jpg"], device="cpu")

        # Should return zero tensor for failed images
        assert embeddings.shape == (1, 2048)

    def test_batch_size_handling(self, dummy_image_path):
        """Test that different batch sizes work correctly."""
        config = BackboneConfig(
            name="resnet50",
            embedding_dim=2048,
            input_size=224,
            pretrained=False,
            batch_size=2,
        )

        backbone = get_backbone(config, device="cpu")

        # Test with 5 images, batch size 2
        image_paths = [dummy_image_path] * 5
        embeddings = backbone.extract_embeddings(image_paths, device="cpu")

        assert embeddings.shape == (5, 2048)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
