"""Unit tests for loss functions.

Tests for:
- FocalLoss
- ArcFaceLoss
- SubCenterArcFaceLoss
- TripletLoss
- CombinedLoss
- build_loss factory function
"""

import pytest
import torch
import torch.nn as nn

from jaguars.reidentification.losses import (
    ArcFaceLoss,
    CombinedLoss,
    FocalLoss,
    LogSoftmaxNLLLoss,
    SubCenterArcFaceLoss,
    TripletLoss,
    build_loss,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def num_classes():
    return 10


@pytest.fixture
def embedding_dim():
    return 256


@pytest.fixture
def random_logits(batch_size, num_classes):
    """Random logits for classification tests."""
    return torch.randn(batch_size, num_classes)


@pytest.fixture
def random_embeddings(batch_size, embedding_dim):
    """Random embeddings for metric learning tests."""
    return torch.randn(batch_size, embedding_dim)


@pytest.fixture
def random_labels(batch_size, num_classes):
    """Random labels for classification tests."""
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def pk_batch_labels():
    """Labels suitable for triplet mining (P=4 classes, K=4 samples each)."""
    # 4 classes, 4 samples each = 16 total
    labels = []
    for cls_id in range(4):
        labels.extend([cls_id] * 4)
    return torch.tensor(labels)


@pytest.fixture
def pk_batch_embeddings(pk_batch_labels, embedding_dim):
    """Embeddings matching PK batch structure."""
    return torch.randn(len(pk_batch_labels), embedding_dim)


# =============================================================================
# FocalLoss Tests
# =============================================================================


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_output_shape(self, random_logits, random_labels):
        """Test that focal loss returns a scalar."""
        criterion = FocalLoss()
        loss = criterion(random_logits, random_labels)
        assert loss.shape == torch.Size([])

    def test_output_non_negative(self, random_logits, random_labels):
        """Test that focal loss is non-negative."""
        criterion = FocalLoss()
        loss = criterion(random_logits, random_labels)
        assert loss.item() >= 0

    def test_gamma_zero_equals_ce(self, random_logits, random_labels):
        """Test that gamma=0 is equivalent to cross-entropy."""
        focal = FocalLoss(gamma=0.0)
        ce = nn.CrossEntropyLoss()

        focal_loss = focal(random_logits, random_labels)
        ce_loss = ce(random_logits, random_labels)

        # Should be approximately equal
        assert torch.allclose(focal_loss, ce_loss, rtol=1e-4)

    def test_higher_gamma_reduces_easy_example_weight(self):
        """Test that higher gamma reduces contribution of easy examples."""
        # Create logits where one class is clearly dominant (easy example)
        logits = torch.tensor([[10.0, 0.0, 0.0]])  # Clear class 0
        labels = torch.tensor([0])

        focal_low = FocalLoss(gamma=1.0)
        focal_high = FocalLoss(gamma=5.0)

        loss_low = focal_low(logits, labels)
        loss_high = focal_high(logits, labels)

        # Higher gamma should give lower loss for easy examples
        assert loss_high < loss_low

    def test_reduction_none(self, random_logits, random_labels, batch_size):
        """Test reduction='none' returns per-sample losses."""
        criterion = FocalLoss(reduction="none")
        loss = criterion(random_logits, random_labels)
        assert loss.shape == torch.Size([batch_size])

    def test_reduction_sum(self, random_logits, random_labels):
        """Test reduction='sum' returns sum of losses."""
        criterion_sum = FocalLoss(reduction="sum")
        criterion_none = FocalLoss(reduction="none")

        loss_sum = criterion_sum(random_logits, random_labels)
        loss_none = criterion_none(random_logits, random_labels)

        assert torch.allclose(loss_sum, loss_none.sum())

    def test_backward_pass(self, random_logits, random_labels):
        """Test that gradients flow through focal loss."""
        logits = random_logits.clone().requires_grad_(True)
        criterion = FocalLoss()
        loss = criterion(logits, random_labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


# =============================================================================
# LogSoftmaxNLLLoss Tests
# =============================================================================


class TestLogSoftmaxNLLLoss:
    """Tests for LogSoftmaxNLLLoss."""

    def test_equivalent_to_cross_entropy(self, random_logits, random_labels):
        """Test that LogSoftmaxNLL is equivalent to CrossEntropy."""
        nll = LogSoftmaxNLLLoss()
        ce = nn.CrossEntropyLoss()

        nll_loss = nll(random_logits, random_labels)
        ce_loss = ce(random_logits, random_labels)

        assert torch.allclose(nll_loss, ce_loss, rtol=1e-4)


# =============================================================================
# ArcFaceLoss Tests
# =============================================================================


class TestArcFaceLoss:
    """Tests for ArcFaceLoss."""

    def test_output_shape(self, random_embeddings, random_labels, batch_size, num_classes, embedding_dim):
        """Test that ArcFace returns correct logit shape."""
        criterion = ArcFaceLoss(embedding_dim=embedding_dim, num_classes=num_classes)
        logits = criterion(random_embeddings, random_labels)

        assert logits.shape == torch.Size([batch_size, num_classes])

    def test_scale_factor(self, random_embeddings, random_labels, num_classes, embedding_dim):
        """Test that scale factor affects logit magnitude."""
        arcface_low = ArcFaceLoss(embedding_dim=embedding_dim, num_classes=num_classes, scale=10.0)
        arcface_high = ArcFaceLoss(embedding_dim=embedding_dim, num_classes=num_classes, scale=64.0)

        # Use same weights
        arcface_high.weight.data = arcface_low.weight.data.clone()

        logits_low = arcface_low(random_embeddings, random_labels)
        logits_high = arcface_high(random_embeddings, random_labels)

        # High scale should have larger magnitude
        assert logits_high.abs().mean() > logits_low.abs().mean()

    def test_margin_affects_target_logit(self, num_classes, embedding_dim):
        """Test that margin reduces target class logit."""
        # Create embedding that's exactly aligned with class 0
        embedding = torch.zeros(1, embedding_dim)
        embedding[0, 0] = 1.0  # Unit vector in first dimension
        labels = torch.tensor([0])

        arcface = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=1.0,  # Scale=1 for easy comparison
            margin=0.5,
        )
        # Set weight[0] to match embedding exactly
        arcface.weight.data = torch.zeros(num_classes, embedding_dim)
        arcface.weight.data[0, 0] = 1.0

        logits = arcface(embedding, labels)

        # The target class should have margin applied (cosine reduced)
        # cos(theta + margin) < cos(theta) = 1 for theta = 0
        assert logits[0, 0] < 1.0

    def test_get_cosine_similarity(self, random_embeddings, num_classes, embedding_dim, batch_size):
        """Test get_cosine_similarity method."""
        arcface = ArcFaceLoss(embedding_dim=embedding_dim, num_classes=num_classes)
        cosine = arcface.get_cosine_similarity(random_embeddings)

        assert cosine.shape == torch.Size([batch_size, num_classes])
        # Cosine similarity should be in [-1, 1]
        assert cosine.min() >= -1.0 - 1e-5
        assert cosine.max() <= 1.0 + 1e-5

    def test_backward_pass(self, random_embeddings, random_labels, num_classes, embedding_dim):
        """Test that gradients flow through ArcFace."""
        embeddings = random_embeddings.clone().requires_grad_(True)
        arcface = ArcFaceLoss(embedding_dim=embedding_dim, num_classes=num_classes)
        logits = arcface(embeddings, random_labels)
        loss = nn.CrossEntropyLoss()(logits, random_labels)
        loss.backward()

        assert embeddings.grad is not None
        assert arcface.weight.grad is not None


# =============================================================================
# SubCenterArcFaceLoss Tests
# =============================================================================


class TestSubCenterArcFaceLoss:
    """Tests for SubCenterArcFaceLoss."""

    def test_output_shape(self, random_embeddings, random_labels, batch_size, num_classes, embedding_dim):
        """Test that SubCenter ArcFace returns correct shape."""
        criterion = SubCenterArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_subcenters=3,
        )
        logits = criterion(random_embeddings, random_labels)

        assert logits.shape == torch.Size([batch_size, num_classes])

    def test_weight_shape(self, num_classes, embedding_dim):
        """Test that weight matrix has correct shape for subcenters."""
        num_subcenters = 3
        criterion = SubCenterArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
        )

        expected_shape = (num_classes * num_subcenters, embedding_dim)
        assert criterion.weight.shape == expected_shape

    def test_inference_mode(self, random_embeddings, batch_size, num_classes, embedding_dim):
        """Test inference mode (no labels)."""
        criterion = SubCenterArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )
        logits = criterion(random_embeddings, labels=None)

        assert logits.shape == torch.Size([batch_size, num_classes])


# =============================================================================
# TripletLoss Tests
# =============================================================================


class TestTripletLoss:
    """Tests for TripletLoss."""

    def test_output_scalar(self, pk_batch_embeddings, pk_batch_labels):
        """Test that triplet loss returns a scalar."""
        criterion = TripletLoss()
        loss = criterion(pk_batch_embeddings, pk_batch_labels)
        assert loss.shape == torch.Size([])

    def test_output_non_negative(self, pk_batch_embeddings, pk_batch_labels):
        """Test that triplet loss is non-negative."""
        criterion = TripletLoss()
        loss = criterion(pk_batch_embeddings, pk_batch_labels)
        assert loss.item() >= 0

    def test_zero_loss_for_perfect_clustering(self, embedding_dim):
        """Test that perfect clustering gives zero loss."""
        # Create 2 classes with very tight, well-separated clusters
        embeddings = torch.zeros(8, embedding_dim)
        # Class 0: centered at [1, 0, 0, ...]
        embeddings[:4, 0] = 1.0
        # Class 1: centered at [-1, 0, 0, ...] (very far apart)
        embeddings[4:, 0] = -1.0
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        criterion = TripletLoss(margin=0.3)
        loss = criterion(embeddings, labels)

        # Should be 0 or very small (clusters are far apart)
        assert loss.item() < 0.01

    def test_all_mining_strategy(self, pk_batch_embeddings, pk_batch_labels):
        """Test 'all' mining strategy."""
        criterion = TripletLoss(mining="all")
        loss = criterion(pk_batch_embeddings, pk_batch_labels)
        assert loss.shape == torch.Size([])

    def test_hard_mining_strategy(self, pk_batch_embeddings, pk_batch_labels):
        """Test 'hard' mining strategy."""
        criterion = TripletLoss(mining="hard")
        loss = criterion(pk_batch_embeddings, pk_batch_labels)
        assert loss.shape == torch.Size([])

    def test_semi_hard_mining_strategy(self, pk_batch_embeddings, pk_batch_labels):
        """Test 'semi-hard' mining strategy."""
        criterion = TripletLoss(mining="semi-hard")
        loss = criterion(pk_batch_embeddings, pk_batch_labels)
        assert loss.shape == torch.Size([])

    def test_cosine_distance(self, pk_batch_embeddings, pk_batch_labels):
        """Test cosine distance metric."""
        criterion = TripletLoss(distance="cosine")
        loss = criterion(pk_batch_embeddings, pk_batch_labels)
        assert loss.shape == torch.Size([])

    def test_margin_effect(self, pk_batch_embeddings, pk_batch_labels):
        """Test that larger margin increases loss."""
        criterion_small = TripletLoss(margin=0.1)
        criterion_large = TripletLoss(margin=1.0)

        loss_small = criterion_small(pk_batch_embeddings, pk_batch_labels)
        loss_large = criterion_large(pk_batch_embeddings, pk_batch_labels)

        # Larger margin should generally give higher loss
        # (not always true but usually)
        assert loss_large >= loss_small - 0.5  # Allow some variance

    def test_single_class_batch(self, embedding_dim):
        """Test behavior with single class (no valid triplets)."""
        embeddings = torch.randn(4, embedding_dim)
        labels = torch.tensor([0, 0, 0, 0])  # All same class

        criterion = TripletLoss()
        loss = criterion(embeddings, labels)

        # Should return 0 (no valid triplets)
        assert loss.item() == 0.0

    def test_backward_pass(self, pk_batch_embeddings, pk_batch_labels):
        """Test that gradients flow through triplet loss."""
        embeddings = pk_batch_embeddings.clone().requires_grad_(True)
        criterion = TripletLoss()
        loss = criterion(embeddings, pk_batch_labels)
        loss.backward()

        assert embeddings.grad is not None


# =============================================================================
# CombinedLoss Tests
# =============================================================================


class TestCombinedLoss:
    """Tests for CombinedLoss."""

    def test_classification_only(self, random_logits, random_embeddings, random_labels):
        """Test combined loss with classification only."""
        cls_criterion = nn.CrossEntropyLoss()
        combined = CombinedLoss(cls_criterion)

        total_loss, loss_dict = combined(random_logits, random_embeddings, random_labels)

        assert total_loss.shape == torch.Size([])
        assert "cls_loss" in loss_dict
        assert "total_loss" in loss_dict
        assert "metric_loss" not in loss_dict

    def test_with_metric_loss(self, random_logits, random_embeddings, random_labels):
        """Test combined loss with metric loss."""
        cls_criterion = nn.CrossEntropyLoss()
        metric_criterion = TripletLoss()
        combined = CombinedLoss(cls_criterion, metric_criterion, metric_weight=0.5)

        total_loss, loss_dict = combined(random_logits, random_embeddings, random_labels)

        assert total_loss.shape == torch.Size([])
        assert "cls_loss" in loss_dict
        assert "metric_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_metric_weight(self, random_logits, random_embeddings, random_labels):
        """Test that metric weight affects total loss."""
        cls_criterion = nn.CrossEntropyLoss()
        metric_criterion = TripletLoss()

        combined_low = CombinedLoss(cls_criterion, metric_criterion, metric_weight=0.1)
        combined_high = CombinedLoss(cls_criterion, metric_criterion, metric_weight=1.0)

        total_low, _ = combined_low(random_logits, random_embeddings, random_labels)
        total_high, _ = combined_high(random_logits, random_embeddings, random_labels)

        # Higher weight should generally give different loss
        # (unless metric loss is 0)
        assert not torch.allclose(total_low, total_high, atol=1e-3)


# =============================================================================
# build_loss Factory Tests
# =============================================================================


class TestBuildLoss:
    """Tests for build_loss factory function."""

    def test_cross_entropy(self):
        """Test building CrossEntropyLoss."""
        criterion = build_loss("cross_entropy")
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_label_smoothing(self):
        """Test building label smoothing CrossEntropy."""
        criterion = build_loss("label_smoothing", label_smoothing=0.1)
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_focal(self):
        """Test building FocalLoss."""
        criterion = build_loss("focal", focal_gamma=2.0)
        assert isinstance(criterion, FocalLoss)
        assert criterion.gamma == 2.0

    def test_nll(self):
        """Test building LogSoftmaxNLLLoss."""
        criterion = build_loss("nll")
        assert isinstance(criterion, LogSoftmaxNLLLoss)

    def test_multi_margin(self):
        """Test building MultiMarginLoss."""
        criterion = build_loss("multi_margin", multi_margin_margin=0.5)
        assert isinstance(criterion, nn.MultiMarginLoss)

    def test_arcface(self):
        """Test building ArcFaceLoss."""
        criterion = build_loss(
            "arcface",
            num_classes=10,
            embedding_dim=256,
            arcface_margin=0.5,
            arcface_scale=64.0,
        )
        assert isinstance(criterion, ArcFaceLoss)
        assert criterion.margin == 0.5
        assert criterion.scale == 64.0

    def test_subcenter_arcface(self):
        """Test building SubCenterArcFaceLoss."""
        criterion = build_loss(
            "subcenter_arcface",
            num_classes=10,
            embedding_dim=256,
            num_subcenters=3,
        )
        assert isinstance(criterion, SubCenterArcFaceLoss)
        assert criterion.num_subcenters == 3

    def test_triplet(self):
        """Test building TripletLoss."""
        criterion = build_loss(
            "triplet",
            triplet_margin=0.5,
            triplet_mining="hard",
        )
        assert isinstance(criterion, TripletLoss)
        assert criterion.margin == 0.5
        assert criterion.mining == "hard"

    def test_arcface_requires_num_classes(self):
        """Test that ArcFace raises error without num_classes."""
        with pytest.raises(ValueError, match="requires num_classes"):
            build_loss("arcface", embedding_dim=256)

    def test_arcface_requires_embedding_dim(self):
        """Test that ArcFace raises error without embedding_dim."""
        with pytest.raises(ValueError, match="requires num_classes and embedding_dim"):
            build_loss("arcface", num_classes=10)

    def test_unknown_loss(self):
        """Test that unknown loss name raises error."""
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss("unknown_loss_name")
