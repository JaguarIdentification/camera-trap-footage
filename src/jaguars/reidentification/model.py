"""Model components for jaguar re-identification.

This module provides:
- Embedding projection networks
- Loss functions (ArcFace, Triplet, etc.)
- Complete re-identification models
"""

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from jaguars.reidentification.config import ModelConfig


class EmbeddingProjection(nn.Module):
    """Projects MegaDescriptor embeddings to a lower-dimensional space.

    Architecture: input_dim -> hidden_dim -> output_dim
    """

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, output_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(x))


class ArcFaceLayer(nn.Module):
    """ArcFace (Additive Angular Margin Loss) layer.

    The loss is computed as:
        L = -log(exp(s * cos(theta_y + m)) / (exp(s * cos(theta_y + m)) + sum(exp(s * cos(theta_j)))))

    where:
        - theta_y is the angle between embedding and ground truth class center
        - m is the angular margin (default 0.5 radians, about 28.6 degrees)
        - s is the feature scale (default 64)
    """

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Learnable weight matrix (class prototypes on the hypersphere)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trigonometric values for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold for numerical stability
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: (batch_size, embedding_dim) - will be normalized
            labels: (batch_size,) - ground truth class indices

        Returns:
            logits: (batch_size, num_classes) - ArcFace logits for cross-entropy loss
        """
        # Normalize embeddings and weights to unit length
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: cos(theta)
        cosine = F.linear(embeddings, weight_norm)
        cosine = cosine.clamp(-1.0, 1.0)

        # Compute sin(theta) from cos(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Compute cos(theta + m) using angle addition formula
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply threshold to handle theta + m >= pi
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale logits
        output = output * self.scale

        return output


class ArcFaceModel(nn.Module):
    """Complete model: Embedding Projection + ArcFace.

    This is a convenience wrapper that combines the embedding projection
    and ArcFace loss layer for training.
    """

    def __init__(self, input_dim: int, num_classes: int, config: ModelConfig) -> None:
        """Initialize ArcFace model.

        Args:
            input_dim: Input feature dimension (from backbone)
            num_classes: Number of identity classes
            config: Model configuration
        """
        super().__init__()
        self.embedding_net = EmbeddingProjection(
            input_dim=input_dim, hidden_dim=config.hidden_dim, output_dim=config.embedding_dim, dropout=config.dropout
        )
        self.arcface = ArcFaceLayer(
            embedding_dim=config.embedding_dim, num_classes=num_classes, margin=config.arcface_margin, scale=config.arcface_scale
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training (requires labels for ArcFace).

        Args:
            x: Input features (batch_size, input_dim)
            labels: Ground truth labels (batch_size,)

        Returns:
            Tuple of (logits, embeddings)
                - logits: ArcFace logits (batch_size, num_classes)
                - embeddings: Normalized embeddings (batch_size, embedding_dim)
        """
        embeddings = self.embedding_net(x)
        logits = self.arcface(embeddings, labels)
        return logits, embeddings

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized embeddings for inference.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Normalized embeddings (batch_size, embedding_dim)
        """
        embeddings = self.embedding_net(x)
        return F.normalize(embeddings, p=2, dim=1)


def build_model(input_dim: int, num_classes: int, config: ModelConfig) -> ArcFaceModel:
    """Factory function to build re-identification model.

    Args:
        input_dim: Input feature dimension from backbone
        num_classes: Number of identity classes
        config: Model configuration

    Returns:
        Initialized model
    """
    model = ArcFaceModel(input_dim=input_dim, num_classes=num_classes, config=config)
    print("Model initialized:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  ArcFace margin: {config.arcface_margin}")
    print(f"  ArcFace scale: {config.arcface_scale}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model
