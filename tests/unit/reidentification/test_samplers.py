"""Unit tests for sampling strategies."""

from collections import Counter

import numpy as np
import pytest

from jaguars.reidentification.samplers import ClassBalancedSampler, PKSampler

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def balanced_labels():
    """Labels with equal class distribution (10 classes, 20 samples each)."""
    labels = []
    for cls_id in range(10):
        labels.extend([cls_id] * 20)
    return np.array(labels)


@pytest.fixture
def imbalanced_labels():
    """Labels with imbalanced class distribution."""
    labels = []
    # Class 0: 100 samples, Class 1: 50, Class 2: 25, etc.
    for cls_id, count in enumerate([100, 50, 25, 10, 5]):
        labels.extend([cls_id] * count)
    return np.array(labels)


@pytest.fixture
def few_classes_labels():
    """Labels with only 3 classes, 10 samples each."""
    labels = []
    for cls_id in range(3):
        labels.extend([cls_id] * 10)
    return np.array(labels)


# =============================================================================
# PKSampler Tests
# =============================================================================


class TestPKSampler:
    """Tests for PKSampler."""

    def test_batch_has_p_classes(self, balanced_labels):
        """Test that each batch contains exactly P classes."""
        p, k = 4, 4
        sampler = PKSampler(balanced_labels, p=p, k=k)

        for batch_indices in sampler:
            batch_labels = balanced_labels[batch_indices]
            unique_classes = np.unique(batch_labels)
            assert len(unique_classes) == p

    def test_batch_has_k_samples_per_class(self, balanced_labels):
        """Test that each batch has K samples per class."""
        p, k = 4, 4
        sampler = PKSampler(balanced_labels, p=p, k=k)

        for batch_indices in sampler:
            batch_labels = balanced_labels[batch_indices]
            class_counts = Counter(batch_labels)
            for count in class_counts.values():
                assert count == k

    def test_batch_size(self, balanced_labels):
        """Test that batch size is P * K."""
        p, k = 5, 3
        sampler = PKSampler(balanced_labels, p=p, k=k)

        for batch_indices in sampler:
            assert len(batch_indices) == p * k

    def test_raises_if_not_enough_classes(self, few_classes_labels):
        """Test that error is raised if fewer than P valid classes."""
        with pytest.raises(ValueError, match="Not enough classes"):
            PKSampler(few_classes_labels, p=5, k=4)

    def test_filters_classes_with_few_samples(self, imbalanced_labels):
        """Test that classes with fewer than K samples are filtered."""
        # k=20 means classes with <20 samples are excluded
        # Original: 100, 50, 25, 10, 5 -> only classes 0, 1, 2 have >= 20 samples
        sampler = PKSampler(imbalanced_labels, p=2, k=20)

        # Only classes 0, 1, 2 have >= 20 samples
        assert len(sampler.valid_classes) == 3

    def test_len_returns_num_batches(self, balanced_labels):
        """Test that __len__ returns correct number of batches."""
        p, k = 4, 4
        sampler = PKSampler(balanced_labels, p=p, k=k)

        # 10 classes, P=4, drop_last=True -> 10 // 4 = 2 batches
        expected_batches = 10 // p
        assert len(sampler) == expected_batches

    def test_len_with_drop_last_false(self, balanced_labels):
        """Test __len__ with drop_last=False."""
        p = 3
        sampler = PKSampler(balanced_labels, p=p, k=4, drop_last=False)

        # 10 classes, P=3, drop_last=False -> ceil(10/3) = 4 batches
        expected_batches = (10 + p - 1) // p
        assert len(sampler) == expected_batches

    def test_all_samples_covered_eventually(self, balanced_labels):
        """Test that all classes appear across multiple epochs."""
        p, k = 4, 4
        sampler = PKSampler(balanced_labels, p=p, k=k)

        # Collect classes across multiple epochs
        seen_classes = set()
        for _ in range(10):  # Multiple epochs
            for batch_indices in sampler:
                batch_labels = balanced_labels[batch_indices]
                seen_classes.update(batch_labels.tolist())

        # All 10 classes should eventually appear
        assert len(seen_classes) == 10

    def test_with_replacement_for_small_classes(self):
        """Test that sampling works with replacement for small classes."""
        # Create labels where some classes have fewer than k samples
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])  # Class 1 has only 2 samples

        # k=3 but class 1 only has 2 samples - should still work with replacement
        sampler = PKSampler(labels, p=2, k=3)

        # Should not raise an error
        batch = list(sampler)[0]
        assert len(batch) == 2 * 3


# =============================================================================
# ClassBalancedSampler Tests
# =============================================================================


class TestClassBalancedSampler:
    """Tests for ClassBalancedSampler."""

    def test_output_length(self, balanced_labels):
        """Test that sampler returns correct number of samples."""
        num_samples = 50
        sampler = ClassBalancedSampler(balanced_labels, num_samples=num_samples)
        indices = list(sampler)

        assert len(indices) == num_samples

    def test_default_num_samples(self, balanced_labels):
        """Test that default num_samples equals dataset size."""
        sampler = ClassBalancedSampler(balanced_labels)
        assert len(sampler) == len(balanced_labels)

    def test_balances_imbalanced_labels(self, imbalanced_labels):
        """Test that sampling balances class distribution."""
        num_samples = 1000
        sampler = ClassBalancedSampler(imbalanced_labels, num_samples=num_samples)

        # Sample and count
        indices = list(sampler)
        sampled_labels = imbalanced_labels[indices]
        class_counts = Counter(sampled_labels)

        # Should be more balanced than original
        # Original: 100, 50, 25, 10, 5
        # After balancing, smaller classes should be oversampled
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())

        # The ratio should be much smaller than 100/5 = 20
        ratio = max_count / min_count
        assert ratio < 5  # Loose bound, but much better than 20

    def test_with_replacement(self, balanced_labels):
        """Test that sampling with replacement can exceed dataset size."""
        num_samples = len(balanced_labels) * 2
        sampler = ClassBalancedSampler(balanced_labels, num_samples=num_samples, replacement=True)

        indices = list(sampler)
        assert len(indices) == num_samples

    def test_valid_indices(self, balanced_labels):
        """Test that all indices are valid."""
        sampler = ClassBalancedSampler(balanced_labels)
        indices = list(sampler)

        for idx in indices:
            assert 0 <= idx < len(balanced_labels)
