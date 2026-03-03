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


class TimmBackbone(BackboneInterface):
    """Generic backbone using timm library (supports DINOv2, MegaDescriptor, ResNet, ConvNeXt, etc.)."""

    def __init__(self, config: BackboneConfig):
        super().__init__(config)

        # Load pretrained model
        print(f"Loading {config.name} model...")
        self.model = timm.create_model(config.name, pretrained=config.pretrained, num_classes=0)
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
        """Extract features using timm backbone.

        Args:
            x: Input tensor (batch_size, 3, height, width)

        Returns:
            Feature embeddings (batch_size, embedding_dim)
        """
        return cast(torch.Tensor, self.model(x))

    def get_preprocess(self) -> transforms.Compose:
        """Get preprocessing transforms based on model architecture.

        Different models use different normalization:
        - DINOv2: ImageNet normalization
        - MegaDescriptor: [-1, 1] normalization (mean=0.5, std=0.5)
        - ConvNeXt/ResNet/EfficientNet: ImageNet normalization

        Returns:
            Composed transformations
        """
        name_lower = self.config.name.lower()

        # MegaDescriptor models use [-1, 1] normalization
        if "megadescriptor" in name_lower or "bvra" in name_lower:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        # All other models use ImageNet normalization
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        return transforms.Compose(
            [
                transforms.Resize((self.config.input_size, self.config.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class MiewIDBackbone(BackboneInterface):
    """Backbone loader for MiewID models via HuggingFace transformers.

    MiewID model cards document loading via:
        AutoModel.from_pretrained("conservationxlabs/miewid-msv*", trust_remote_code=True)
    """

    def __init__(self, config: BackboneConfig):
        super().__init__(config)

        print(f"Loading {config.name} model via transformers AutoModel...")
        try:
            from transformers import AutoModel, PreTrainedModel
        except ImportError as exc:
            raise ImportError("transformers is required for MiewID backbones. Install with: pip install transformers") from exc

        # Compatibility patch for newer transformers versions where
        # remote-code models may not define this attribute expected by
        # mark_tied_weights_as_initialized()
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
        elif not isinstance(PreTrainedModel.all_tied_weights_keys, dict):
            PreTrainedModel.all_tied_weights_keys = {}

        self.model = AutoModel.from_pretrained(config.name, trust_remote_code=True)
        self.model.eval()

        # Verify embedding dimension (best-effort)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config.input_size, config.input_size)
            dummy_output = self.forward(dummy_input)
            if dummy_output.ndim != 2:
                raise ValueError(f"Unexpected MiewID output shape: {tuple(dummy_output.shape)}")
            actual_dim = int(dummy_output.shape[1])

            if actual_dim != config.embedding_dim:
                print(f"Warning: Expected embedding_dim={config.embedding_dim}, got {actual_dim}")
                config.embedding_dim = actual_dim

        print("Model loaded successfully")
        print("  Backend: transformers.AutoModel (trust_remote_code=True)")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Embedding dimension: {config.embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)

        # Common output patterns from HF models / custom remote code
        if isinstance(out, torch.Tensor):
            emb = out
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            emb = out.last_hidden_state.mean(dim=1)
        elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            emb = out[0]
        else:
            raise ValueError(f"Unsupported output type from MiewID model: {type(out)}")

        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        elif emb.ndim > 2:
            emb = emb.view(emb.shape[0], -1)

        return cast(torch.Tensor, emb)

    def get_preprocess(self) -> transforms.Compose:
        # MiewID model card uses ImageNet normalization with 440x440 input
        return transforms.Compose(
            [
                transforms.Resize((self.config.input_size, self.config.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def get_backbone(config: BackboneConfig, device: str | None = None) -> BackboneInterface:
    """Factory function to get backbone by name.

    All models are loaded via timm, which supports:
    - DINOv2/DINOv3 (Vision Transformers)
    - MegaDescriptor (Animal re-ID models)
    - ResNet, ConvNeXt, EfficientNet (CNNs)
    - Any other timm model

    Args:
        config: Backbone configuration
        device: Device to load model on

    Returns:
        Initialized backbone model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # MiewID models are loaded via transformers AutoModel with remote code
    if config.name.startswith("conservationxlabs/miewid-"):
        backbone = MiewIDBackbone(config)
    else:
        backbone = TimmBackbone(config)

    backbone.to(device)
    return backbone
