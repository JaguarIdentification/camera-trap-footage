"""Sampling strategies for metric learning.

This module provides samplers for creating balanced batches
suitable for metric learning losses like TripletLoss.

References:
- In Defense of the Triplet Loss for Person Re-ID: https://arxiv.org/abs/1703.07737
"""

from collections import defaultdict
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class PKSampler(Sampler[int]):
    """P-K Sampler for balanced batch construction.

    Creates batches with exactly P classes and K samples per class.
    This ensures every batch has valid triplets for metric learning.

    For each batch:
    - Randomly select P classes
    - Randomly select K samples from each class
    - Total batch size = P * K

    Args:
        labels: Array of class labels for all samples
        p: Number of classes per batch
        k: Number of samples per class
        drop_last: Whether to drop the last incomplete batch

    Example:
        >>> sampler = PKSampler(labels, p=8, k=4)  # 32 samples per batch
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        labels: np.ndarray | list[int],
        p: int = 8,
        k: int = 4,
        drop_last: bool = True,
    ) -> None:
        super().__init__(None)  # type: ignore

        self.labels = np.asarray(labels)
        self.p = p
        self.k = k
        self.drop_last = drop_last

        # Build index mapping: class_id -> list of sample indices
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[int(label)].append(idx)

        # Filter classes with at least k samples
        self.valid_classes = [cls for cls, indices in self.class_to_indices.items() if len(indices) >= k]

        if len(self.valid_classes) < p:
            raise ValueError(f"Not enough classes with at least {k} samples. " f"Found {len(self.valid_classes)}, need {p}.")

        # Calculate number of batches
        self.num_batches = len(self.valid_classes) // p
        if not drop_last and len(self.valid_classes) % p != 0:
            self.num_batches += 1

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices."""
        # Shuffle classes at the start of each epoch
        classes = np.array(self.valid_classes)
        np.random.shuffle(classes)

        # Generate batches
        for batch_idx in range(self.num_batches):
            batch_indices: list[int] = []
            start = batch_idx * self.p
            end = min(start + self.p, len(classes))

            for cls in classes[start:end]:
                # Get all indices for this class
                class_indices = np.array(self.class_to_indices[int(cls)])

                # Sample k indices (with replacement if necessary)
                if len(class_indices) >= self.k:
                    selected = np.random.choice(class_indices, size=self.k, replace=False)
                else:
                    # With replacement for classes with fewer than k samples
                    selected = np.random.choice(class_indices, size=self.k, replace=True)

                batch_indices.extend(selected.tolist())

            yield batch_indices

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches


class ClassBalancedSampler(Sampler[int]):
    """Class-balanced sampler using inverse class frequency weights.

    Useful for imbalanced datasets. Oversamples minority classes
    and undersamples majority classes.

    Args:
        labels: Array of class labels for all samples
        num_samples: Total number of samples to draw per epoch
                     (default: len(labels))
        replacement: Whether to sample with replacement
    """

    def __init__(
        self,
        labels: np.ndarray | list[int],
        num_samples: int | None = None,
        replacement: bool = True,
    ) -> None:
        super().__init__(None)  # type: ignore

        labels = np.asarray(labels)
        self.num_samples = num_samples if num_samples else len(labels)
        self.replacement = replacement

        # Compute class weights (inverse frequency)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts

        # Assign weight to each sample based on its class
        self.weights = torch.from_numpy(class_weights[labels]).float()

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices."""
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples
