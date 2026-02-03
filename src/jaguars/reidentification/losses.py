"""Loss functions for jaguar re-identification.

This module provides various loss functions for metric learning and classification:
- Cross Entropy variants (standard, label smoothing, focal)
- ArcFace (Additive Angular Margin)
- SubCenterArcFace (with multiple sub-centers)
- TripletLoss (with online hard/semi-hard mining)
- Combined losses for multi-task learning

References:
- ArcFace: https://arxiv.org/abs/1801.07698
- SubCenterArcFace: https://arxiv.org/abs/2003.09150
- FocalLoss: https://arxiv.org/abs/1708.02002
- TripletLoss: https://arxiv.org/abs/1503.03832
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Classification Losses
# =============================================================================


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification.

    Addresses class imbalance by down-weighting well-classified examples.
    Uses logits as input and applies softmax internally.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
               Default 2.0 works well in practice.
        alpha: Class balancing weight. If None, no class balancing.
        reduction: How to reduce the loss ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (batch_size, num_classes) raw logits
            targets: (batch_size,) ground truth class indices

        Returns:
            Focal loss value
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.long().view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            loss = self.alpha * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LogSoftmaxNLLLoss(nn.Module):
    """NLL loss with internal log-softmax for logits input.

    Equivalent to CrossEntropyLoss but split into two operations.
    Useful when you need access to intermediate log-probabilities.
    """

    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        super().__init__()
        self.nll = nn.NLLLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute NLL loss on log-softmax of logits.

        Args:
            logits: (batch_size, num_classes) raw logits
            targets: (batch_size,) ground truth class indices

        Returns:
            NLL loss value
        """
        return self.nll(F.log_softmax(logits, dim=1), targets)


# =============================================================================
# Angular Margin Losses
# =============================================================================


class ArcFaceLoss(nn.Module):
    """ArcFace (Additive Angular Margin) loss.

    Adds an angular margin penalty to the target logit, enforcing
    tighter intra-class clustering on the hypersphere.

    The loss modifies the cosine similarity as:
        cos(theta + m) for the target class

    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of identity classes
        scale: Feature scaling factor (default 64.0)
        margin: Angular margin in radians (default 0.5 ≈ 28.6°)
        easy_margin: Use easy margin strategy for stability
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Learnable weight matrix (class prototypes on the hypersphere)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trigonometric values for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold for numerical stability
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace logits.

        Args:
            embeddings: (batch_size, embedding_dim) - will be normalized
            labels: (batch_size,) ground truth class indices

        Returns:
            logits: (batch_size, num_classes) scaled logits for cross-entropy
        """
        # Normalize embeddings and weights to unit length
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: cos(theta)
        cosine = F.linear(embeddings, weight_norm)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Compute sin(theta) from cos(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Compute cos(theta + m) using angle addition formula
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Apply threshold to handle theta + m >= pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale logits
        output = output * self.scale

        return output

    def get_cosine_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get cosine similarity to all class centers (for inference).

        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            cosine: (batch_size, num_classes) cosine similarities
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(embeddings, weight_norm)


class SubCenterArcFaceLoss(nn.Module):
    """Sub-Center ArcFace loss.

    Uses multiple sub-centers per class to handle intra-class variation.
    The dominant sub-center drives the decision boundary while others
    capture outliers and variations.

    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of identity classes
        scale: Feature scaling factor
        margin: Angular margin in radians
        num_subcenters: Number of sub-centers per class (default 2)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
        num_subcenters: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.num_subcenters = num_subcenters

        # Weight shape: (num_classes * num_subcenters, embedding_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * num_subcenters, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trigonometric values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Compute SubCenter ArcFace logits.

        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,) ground truth (None for inference)

        Returns:
            logits: (batch_size, num_classes) scaled logits
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity with all sub-centers
        cosine_all = F.linear(embeddings, weight_norm)

        # Reshape to (batch, num_classes, num_subcenters) and take max
        cosine = cosine_all.view(-1, self.num_classes, self.num_subcenters)
        cosine = cosine.max(dim=2)[0]  # Take dominant sub-center
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cosine * self.scale

        # Apply angular margin
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        return output * self.scale


# =============================================================================
# Metric Learning Losses
# =============================================================================


class TripletLoss(nn.Module):
    """Triplet loss with online mining strategies.

    Supports different mining strategies:
    - 'all': Use all valid triplets
    - 'hard': Use hardest positive and hardest negative
    - 'semi-hard': Use semi-hard negatives (within margin)

    L = max(d(a, p) - d(a, n) + margin, 0)

    Args:
        margin: Margin for triplet loss
        mining: Mining strategy ('all', 'hard', 'semi-hard')
        distance: Distance metric ('euclidean', 'cosine')
    """

    def __init__(
        self,
        margin: float = 0.3,
        mining: Literal["all", "hard", "semi-hard"] = "hard",
        distance: Literal["euclidean", "cosine"] = "euclidean",
    ) -> None:
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.distance = distance

    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix.

        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            distances: (batch_size, batch_size) distance matrix
        """
        if self.distance == "cosine":
            embeddings = F.normalize(embeddings, p=2, dim=1)
            # Cosine distance = 1 - cosine_similarity
            return 1 - torch.mm(embeddings, embeddings.t())
        else:
            # Euclidean distance
            dot_product = torch.mm(embeddings, embeddings.t())
            squared_norm = torch.diag(dot_product)
            distances = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1)
            # Numerical stability
            distances = torch.clamp(distances, min=0.0)
            return torch.sqrt(distances + 1e-8)

    def _get_triplet_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Get mask for valid triplets (a, p, n) where a != p != n and y_a == y_p != y_n.

        Args:
            labels: (batch_size,)

        Returns:
            mask: (batch_size, batch_size, batch_size) boolean mask
        """
        batch_size = labels.size(0)
        device = labels.device

        # Indices i, j, k must be distinct
        indices_equal = torch.eye(batch_size, dtype=torch.bool, device=device)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)
        distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

        # y[i] == y[j] and y[i] != y[k]
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = labels_equal.unsqueeze(2)
        i_not_equal_k = (~labels_equal).unsqueeze(1)
        valid_labels = i_equal_j & i_not_equal_k

        return distinct_indices & valid_labels

    def _mine_all_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mine all valid triplets and compute loss.

        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,)

        Returns:
            loss: Scalar triplet loss
        """
        distances = self._pairwise_distance(embeddings)
        mask = self._get_triplet_mask(labels)

        # Compute triplet loss for all valid triplets
        anchor_positive_dist = distances.unsqueeze(2)
        anchor_negative_dist = distances.unsqueeze(1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Apply mask and ReLU
        triplet_loss = triplet_loss * mask.float()
        triplet_loss = F.relu(triplet_loss)

        # Average over valid triplets
        num_positive_triplets = (triplet_loss > 1e-16).float().sum()
        if num_positive_triplets > 0:
            return triplet_loss.sum() / num_positive_triplets
        return torch.tensor(0.0, device=embeddings.device)

    def _mine_hard_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mine hardest triplets: hardest positive and hardest negative for each anchor.

        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,)

        Returns:
            loss: Scalar triplet loss
        """
        distances = self._pairwise_distance(embeddings)
        batch_size = labels.size(0)

        # Mask for positive pairs (same class, different sample)
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(batch_size, dtype=torch.bool, device=labels.device)

        # Mask for negative pairs (different class)
        negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

        # For each anchor, get the hardest positive (largest distance)
        # Set non-positive pairs to 0 so they don't affect max
        masked_positives = distances * positive_mask.float()
        hardest_positive_dist, _ = masked_positives.max(dim=1)

        # For each anchor, get the hardest negative (smallest distance)
        # Set non-negative pairs to inf so they don't affect min
        max_dist = distances.max() + 1
        masked_negatives = distances + (~negative_mask).float() * max_dist
        hardest_negative_dist, _ = masked_negatives.min(dim=1)

        # Compute triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        # Only count anchors that have at least one positive
        valid_anchors = positive_mask.any(dim=1)
        if valid_anchors.sum() > 0:
            return triplet_loss[valid_anchors].mean()
        return torch.tensor(0.0, device=embeddings.device)

    def _mine_semi_hard_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mine semi-hard triplets: negatives harder than positives but within margin.

        Semi-hard: d(a, p) < d(a, n) < d(a, p) + margin

        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,)

        Returns:
            loss: Scalar triplet loss
        """
        distances = self._pairwise_distance(embeddings)
        batch_size = labels.size(0)

        # Masks
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

        # For each anchor-positive pair, find semi-hard negatives
        losses = []
        for i in range(batch_size):
            # Get positive distances for anchor i
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = negative_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            for p_idx in pos_indices:
                d_ap = distances[i, p_idx]

                # Find semi-hard negatives
                d_ans = distances[i, neg_indices]
                semi_hard_mask = (d_ans > d_ap) & (d_ans < d_ap + self.margin)

                if semi_hard_mask.any():
                    # Use the hardest semi-hard negative
                    semi_hard_dists = d_ans[semi_hard_mask]
                    d_an = semi_hard_dists.min()
                    loss = F.relu(d_ap - d_an + self.margin)
                    losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        # Fall back to hard mining if no semi-hard triplets found
        return self._mine_hard_triplets(embeddings, labels)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss.

        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,) class labels

        Returns:
            loss: Scalar triplet loss value
        """
        if self.mining == "all":
            return self._mine_all_triplets(embeddings, labels)
        elif self.mining == "hard":
            return self._mine_hard_triplets(embeddings, labels)
        elif self.mining == "semi-hard":
            return self._mine_semi_hard_triplets(embeddings, labels)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")


class CombinedLoss(nn.Module):
    """Combined loss for multi-task learning.

    Combines classification loss (e.g., ArcFace + CrossEntropy) with
    metric learning loss (e.g., TripletLoss).

    Args:
        classification_loss: Loss for classification head
        metric_loss: Loss for embeddings (optional)
        metric_weight: Weight for metric loss (default 0.5)
    """

    def __init__(
        self,
        classification_loss: nn.Module,
        metric_loss: nn.Module | None = None,
        metric_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.classification_loss = classification_loss
        self.metric_loss = metric_loss
        self.metric_weight = metric_weight

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            logits: (batch_size, num_classes) classification logits
            embeddings: (batch_size, embedding_dim) normalized embeddings
            labels: (batch_size,) ground truth labels

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        cls_loss = self.classification_loss(logits, labels)
        loss_dict = {"cls_loss": cls_loss.item()}

        if self.metric_loss is not None:
            met_loss = self.metric_loss(embeddings, labels)
            loss_dict["metric_loss"] = met_loss.item()
            total_loss = cls_loss + self.metric_weight * met_loss
            loss_dict["total_loss"] = total_loss.item()
        else:
            total_loss = cls_loss
            loss_dict["total_loss"] = cls_loss.item()

        return total_loss, loss_dict


# =============================================================================
# Loss Factory
# =============================================================================


def build_loss(
    loss_name: str,
    num_classes: int | None = None,
    embedding_dim: int | None = None,
    **kwargs,
) -> nn.Module:
    """Factory function to build loss functions.

    Args:
        loss_name: Name of the loss function
        num_classes: Number of classes (required for ArcFace variants)
        embedding_dim: Embedding dimension (required for ArcFace variants)
        **kwargs: Additional arguments for specific losses

    Supported loss names:
        - 'cross_entropy': Standard CrossEntropyLoss
        - 'label_smoothing': CrossEntropyLoss with label smoothing
        - 'focal': FocalLoss
        - 'nll': LogSoftmaxNLLLoss
        - 'arcface': ArcFaceLoss
        - 'subcenter_arcface': SubCenterArcFaceLoss
        - 'triplet': TripletLoss

    Returns:
        Initialized loss module
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif loss_name == "label_smoothing":
        smoothing = kwargs.get("label_smoothing", 0.1)
        return nn.CrossEntropyLoss(label_smoothing=smoothing)

    elif loss_name == "focal":
        gamma = kwargs.get("focal_gamma", 2.0)
        alpha = kwargs.get("focal_alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)

    elif loss_name == "nll":
        return LogSoftmaxNLLLoss()

    elif loss_name == "multi_margin":
        margin = kwargs.get("multi_margin_margin", 1.0)
        p = kwargs.get("multi_margin_p", 1)
        return nn.MultiMarginLoss(margin=margin, p=p)

    elif loss_name == "arcface":
        if num_classes is None or embedding_dim is None:
            raise ValueError("ArcFace requires num_classes and embedding_dim")
        scale = kwargs.get("arcface_scale", 64.0)
        margin = kwargs.get("arcface_margin", 0.5)
        easy_margin = kwargs.get("arcface_easy_margin", False)
        return ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin,
        )

    elif loss_name == "subcenter_arcface":
        if num_classes is None or embedding_dim is None:
            raise ValueError("SubCenterArcFace requires num_classes and embedding_dim")
        scale = kwargs.get("arcface_scale", 30.0)
        margin = kwargs.get("arcface_margin", 0.5)
        num_subcenters = kwargs.get("num_subcenters", 2)
        return SubCenterArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
            num_subcenters=num_subcenters,
        )

    elif loss_name == "triplet":
        margin = kwargs.get("triplet_margin", 0.3)
        mining = kwargs.get("triplet_mining", "hard")
        distance = kwargs.get("triplet_distance", "euclidean")
        return TripletLoss(margin=margin, mining=mining, distance=distance)

    else:
        raise ValueError(f"Unknown loss: {loss_name}")
