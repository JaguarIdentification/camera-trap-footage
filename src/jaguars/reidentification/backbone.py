"""Backbone feature extractors for jaguar re-identification.

This module provides abstract interfaces and concrete implementations of
feature extraction backbones (e.g., MegaDescriptor, ResNet, etc.).
"""

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from jaguars.reidentification.config import BackboneConfig


class BackboneInterface(ABC, nn.Module):
    """Abstract base class for feature extraction backbones."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Feature embeddings (batch_size, embedding_dim)
        """
        pass

    @abstractmethod
    def get_preprocess(self) -> transforms.Compose:
        """Get preprocessing pipeline for input images.

        Returns:
            Composed image transformations
        """
        pass

    @torch.no_grad()
    def extract_embeddings(self, image_paths: list[str], device: str | None = None, desc: str = "Extracting embeddings") -> np.ndarray:
        """Extract embeddings for a list of image paths.

        Args:
            image_paths: List of paths to images
            device: Device to run on (cuda/cpu)
            desc: Description for progress bar

        Returns:
            Embeddings array (num_images, embedding_dim)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.eval()
        embeddings = []
        preprocess = self.get_preprocess()

        for i in tqdm(range(0, len(image_paths), self.config.batch_size), desc=desc):
            batch_paths = image_paths[i : i + self.config.batch_size]

            # Load and preprocess batch
            batch_tensors = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = preprocess(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # Use zero tensor as fallback
                    batch_tensors.append(torch.zeros(3, self.config.input_size, self.config.input_size))

            # Stack and move to device
            batch_tensor = torch.stack(batch_tensors).to(device)

            # Get embeddings
            batch_emb = self(batch_tensor).cpu().numpy()
            embeddings.append(batch_emb)

        return np.vstack(embeddings)

    def get_embedding_dim(self) -> int:
        """Get output embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.config.embedding_dim


class MegaDescriptorBackbone(BackboneInterface):
    """MegaDescriptor (DINOv2) backbone for feature extraction."""

    def __init__(self, config: BackboneConfig):
        super().__init__(config)

        # Load pretrained model
        print(f"Loading {config.name} model...")
        self.model = timm.create_model(config.name, pretrained=config.pretrained)
        self.model.eval()

        # Verify embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config.input_size, config.input_size)
            dummy_output = self.model(dummy_input)
            actual_dim = dummy_output.shape[1]

            if actual_dim != config.embedding_dim:
                print(f"Warning: Expected embedding_dim={config.embedding_dim}, got {actual_dim}")
                config.embedding_dim = actual_dim

        print("Model loaded successfully")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Embedding dimension: {config.embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using MegaDescriptor.

        Args:
            x: Input tensor (batch_size, 3, height, width)

        Returns:
            Feature embeddings (batch_size, embedding_dim)
        """
        return cast(torch.Tensor, self.model(x))

    def get_preprocess(self) -> transforms.Compose:
        """Get preprocessing for MegaDescriptor (ImageNet normalization).

        Returns:
            Composed transformations
        """
        return transforms.Compose(
            [
                transforms.Resize((self.config.input_size, self.config.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def get_backbone(config: BackboneConfig, device: str | None = None) -> BackboneInterface:
    """Factory function to get backbone by name.

    Args:
        config: Backbone configuration
        device: Device to load model on

    Returns:
        Initialized backbone model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine backbone type from name
    # Support ViT, DINOv2, ResNet, and other timm models via MegaDescriptorBackbone
    if "dinov2" in config.name.lower() or "vit" in config.name.lower() or "resnet" in config.name.lower() or "efficientnet" in config.name.lower():
        backbone = MegaDescriptorBackbone(config)
    else:
        # Try to use MegaDescriptorBackbone as fallback for any timm model
        backbone = MegaDescriptorBackbone(config)

    backbone.to(device)
    return backbone
