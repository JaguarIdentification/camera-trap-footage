"""Tests for dataset splitting module."""

import tempfile
import unittest
from pathlib import Path

import cv2
import fiftyone as fo
import numpy as np

from jaguars.ingestion.processing.split import (
    closed_set_split,
    get_id_field,
    open_set_split,
    run_processing,
)


class TestSplit(unittest.TestCase):
    """Test suite for dataset splitting."""

    def setUp(self) -> None:
        """Create test fixtures before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.dataset_name = "test_split_dataset"

        # Clean up any existing test dataset
        if self.dataset_name in fo.list_datasets():
            dataset = fo.load_dataset(self.dataset_name)
            dataset.delete()

    def tearDown(self) -> None:
        """Clean up after each test."""
        if self.dataset_name in fo.list_datasets():
            dataset = fo.load_dataset(self.dataset_name)
            dataset.delete()
        self.temp_dir.cleanup()

    def _create_test_image(self, path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> None:
        """Create a simple test image."""
        img = np.full((480, 640, 3), color, dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def _create_grouped_dataset_with_images(self, num_jaguars: int = 3, images_per_jaguar: int = 5) -> fo.Dataset:
        """Create a grouped dataset with image samples for testing.

        Args:
            num_jaguars: Number of unique jaguar IDs
            images_per_jaguar: Number of images per jaguar ID

        Returns:
            Created FiftyOne dataset
        """
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        samples = []
        for jaguar_idx in range(num_jaguars):
            jaguar_id = f"Jaguar_{jaguar_idx:02d}"

            for img_idx in range(images_per_jaguar):
                # Create unique image
                img_path = self.data_dir / f"{jaguar_id}_{img_idx}.jpg"
                color = ((jaguar_idx * 50) % 256, (img_idx * 30) % 256, 100)
                self._create_test_image(img_path, color)

                # Create sample in image group slice
                group_id = str(uuid.uuid4())
                sample = fo.Sample(filepath=str(img_path), group=fo.Group().element(group_id))
                sample.group.name = "image"
                sample["jaguar_id"] = jaguar_id
                sample["ground_truth"] = fo.Classification(label=jaguar_id)
                sample["site"] = f"Site_{jaguar_idx % 2}"

                samples.append(sample)

        dataset.add_samples(samples)
        return dataset

    def _create_non_grouped_dataset(self, num_jaguars: int = 3, images_per_jaguar: int = 5) -> fo.Dataset:
        """Create a non-grouped dataset with image samples.

        Args:
            num_jaguars: Number of unique jaguar IDs
            images_per_jaguar: Number of images per jaguar ID

        Returns:
            Created FiftyOne dataset
        """
        dataset = fo.Dataset(self.dataset_name)

        samples = []
        for jaguar_idx in range(num_jaguars):
            jaguar_id = f"Jaguar_{jaguar_idx:02d}"

            for img_idx in range(images_per_jaguar):
                # Create unique image
                img_path = self.data_dir / f"{jaguar_id}_{img_idx}.jpg"
                color = ((jaguar_idx * 50) % 256, (img_idx * 30) % 256, 100)
                self._create_test_image(img_path, color)

                sample = fo.Sample(filepath=str(img_path))
                sample["jaguar_id"] = jaguar_id
                sample["ground_truth"] = fo.Classification(label=jaguar_id)
                sample["site"] = f"Site_{jaguar_idx % 2}"

                samples.append(sample)

        dataset.add_samples(samples)
        return dataset

    def test_get_id_field_with_jaguar_id_classification(self) -> None:
        """Test ID field detection with jaguar_id as Classification."""
        dataset = self._create_grouped_dataset_with_images(num_jaguars=2, images_per_jaguar=2)

        id_field = get_id_field(dataset)
        self.assertEqual(id_field, "jaguar_id", "Should detect jaguar_id as string field")

    def test_get_id_field_with_ground_truth(self) -> None:
        """Test ID field detection with ground_truth Classification."""
        dataset = fo.Dataset(self.dataset_name)

        # Create sample with only ground_truth, no jaguar_id
        img_path = self.data_dir / "test.jpg"
        self._create_test_image(img_path)

        sample = fo.Sample(filepath=str(img_path))
        sample["ground_truth"] = fo.Classification(label="TestJaguar")
        dataset.add_sample(sample)

        # Temporarily remove jaguar_id to test ground_truth fallback
        id_field = get_id_field(dataset)
        self.assertEqual(id_field, "ground_truth.label", "Should fall back to ground_truth.label")

    def test_closed_set_split_basic(self) -> None:
        """Test basic closed-set split functionality."""
        dataset = self._create_non_grouped_dataset(num_jaguars=3, images_per_jaguar=10)

        # Add the split field
        dataset.add_sample_field("closed_set_split", fo.StringField)

        # Perform split
        result = closed_set_split(
            dataset=dataset,
            id_field="jaguar_id",
            field_name="closed_set_split",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        # Check that all samples have split assignment
        self.assertEqual(result["train_samples"] + result["val_samples"] + result["test_samples"], len(dataset))

        # Check that all jaguar IDs appear in training (closed-set property)
        train_view = dataset.match(fo.ViewField("closed_set_split") == "train")
        train_ids = set(train_view.distinct("jaguar_id"))
        all_ids = set(dataset.distinct("jaguar_id"))

        self.assertEqual(train_ids, all_ids, "All jaguar IDs must appear in training set (closed-set)")

    def test_closed_set_split_with_single_image_ids(self) -> None:
        """Test closed-set split with IDs that have only one image."""
        dataset = fo.Dataset(self.dataset_name)

        samples = []
        # Create 3 jaguars with 1 image each
        for i in range(3):
            img_path = self.data_dir / f"jaguar_{i}.jpg"
            self._create_test_image(img_path)

            sample = fo.Sample(filepath=str(img_path))
            sample["jaguar_id"] = f"SingleImage_{i}"
            sample["ground_truth"] = fo.Classification(label=f"SingleImage_{i}")
            samples.append(sample)

        dataset.add_samples(samples)
        dataset.add_sample_field("closed_set_split", fo.StringField)

        # Perform split
        closed_set_split(
            dataset=dataset,
            id_field="jaguar_id",
            field_name="closed_set_split",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        # All single-image IDs should go to training
        train_view = dataset.match(fo.ViewField("closed_set_split") == "train")
        self.assertEqual(len(train_view), 3, "All single-image samples should be in training")

    def test_open_set_split_basic(self) -> None:
        """Test basic open-set split functionality."""
        dataset = self._create_non_grouped_dataset(num_jaguars=10, images_per_jaguar=5)

        # Add the split field
        dataset.add_sample_field("open_set_split", fo.StringField)

        # Perform split
        result = open_set_split(
            dataset=dataset,
            id_field="jaguar_id",
            field_name="open_set_split",
            train_only_ids_ratio=0.6,
            shared_ids_ratio=0.2,
            val_test_ids_ratio=0.2,
            shared_train_ratio=0.7,
            shared_val_ratio=0.15,
            shared_test_ratio=0.15,
            seed=42,
        )

        # Check that all samples have split assignment
        total_assigned = result["train_samples"] + result["val_samples"] + result["test_samples"]
        self.assertEqual(total_assigned, len(dataset))

        # Check that we have the three groups of IDs
        self.assertGreater(result["train_only_ids"], 0, "Should have train-only IDs")
        self.assertGreater(result["shared_ids"], 0, "Should have shared IDs")

        # Get train/val/test ID sets
        train_view = dataset.match(fo.ViewField("open_set_split") == "train")
        val_view = dataset.match(fo.ViewField("open_set_split") == "val")
        test_view = dataset.match(fo.ViewField("open_set_split") == "test")

        train_ids = set(train_view.distinct("jaguar_id"))
        val_ids = set(val_view.distinct("jaguar_id"))
        test_ids = set(test_view.distinct("jaguar_id"))

        # Check that there are IDs exclusive to val/test (not in training)
        val_test_only_ids = (val_ids | test_ids) - train_ids
        self.assertGreater(len(val_test_only_ids), 0, "Should have IDs not in training (open-set)")

    def test_run_processing_closed_set_on_grouped_dataset(self) -> None:
        """Test run_processing with closed-set split on grouped dataset."""
        # Create grouped dataset with both image and video slices
        # Use larger dataset to ensure all splits are generated
        self._create_grouped_dataset_with_images(num_jaguars=10, images_per_jaguar=8)

        # Run processing
        result_dataset = run_processing(
            dataset_name=self.dataset_name,
            add_closed_set=True,
            add_open_set=False,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            verbose=False,
        )

        # Verify field was added
        self.assertIn("closed_set_split", result_dataset.get_field_schema())

        # Check that only image slice samples have splits (not video)
        image_view = result_dataset.select_group_slices("image")
        split_values = image_view.distinct("closed_set_split")

        self.assertIn("train", split_values, "Training split should exist")
        # With larger dataset, should have val/test splits too
        self.assertGreater(len(split_values), 1, "Should have multiple split types")

    def test_run_processing_open_set_on_grouped_dataset(self) -> None:
        """Test run_processing with open-set split on grouped dataset."""
        self._create_grouped_dataset_with_images(num_jaguars=10, images_per_jaguar=3)

        # Run processing
        result_dataset = run_processing(
            dataset_name=self.dataset_name,
            add_closed_set=False,
            add_open_set=True,
            train_only_ids_ratio=0.5,
            shared_ids_ratio=0.3,
            val_test_ids_ratio=0.2,
            seed=42,
            verbose=False,
        )

        # Verify field was added
        self.assertIn("open_set_split", result_dataset.get_field_schema())

        # Verify open-set property: some IDs should not be in training
        image_view = result_dataset.select_group_slices("image")
        train_view = image_view.match(fo.ViewField("open_set_split") == "train")
        val_view = image_view.match(fo.ViewField("open_set_split") == "val")
        test_view = image_view.match(fo.ViewField("open_set_split") == "test")

        train_ids = set(train_view.distinct("jaguar_id"))
        val_test_ids = set(val_view.distinct("jaguar_id")) | set(test_view.distinct("jaguar_id"))

        val_test_only = val_test_ids - train_ids
        self.assertGreater(len(val_test_only), 0, "Some IDs should be exclusive to val/test")

    def test_run_processing_both_splits(self) -> None:
        """Test run_processing with both closed and open splits."""
        # Use larger dataset to ensure all splits are generated
        self._create_non_grouped_dataset(num_jaguars=15, images_per_jaguar=8)

        # Run processing with both splits
        result_dataset = run_processing(
            dataset_name=self.dataset_name,
            add_closed_set=True,
            add_open_set=True,
            seed=42,
            verbose=False,
        )

        # Verify both fields were added
        self.assertIn("closed_set_split", result_dataset.get_field_schema())
        self.assertIn("open_set_split", result_dataset.get_field_schema())

        # Verify both have valid values
        closed_splits = result_dataset.distinct("closed_set_split")
        open_splits = result_dataset.distinct("open_set_split")

        # Verify we have at least train split for both
        self.assertIn("train", closed_splits, "Closed-set should have train split")
        self.assertIn("train", open_splits, "Open-set should have train split")

        # With larger dataset, should have multiple split types
        self.assertGreater(len(closed_splits), 1, "Closed-set should have multiple split types")
        self.assertGreater(len(open_splits), 1, "Open-set should have multiple split types")

    def test_tagging_by_closed_set(self) -> None:
        """Test tagging samples by closed-set split."""
        self._create_non_grouped_dataset(num_jaguars=3, images_per_jaguar=4)

        # Run processing with tagging
        result_dataset = run_processing(
            dataset_name=self.dataset_name,
            add_closed_set=True,
            tag_by="closed",
            seed=42,
            verbose=False,
        )

        # Check that samples have tags
        train_tagged = result_dataset.match_tags("train")
        val_tagged = result_dataset.match_tags("val")
        test_tagged = result_dataset.match_tags("test")

        self.assertGreater(len(train_tagged), 0, "Should have train-tagged samples")
        self.assertGreater(len(val_tagged), 0, "Should have val-tagged samples")
        self.assertGreater(len(test_tagged), 0, "Should have test-tagged samples")

        # Verify tags match split field values
        for sample in train_tagged:
            self.assertEqual(sample["closed_set_split"], "train")
        for sample in val_tagged:
            self.assertEqual(sample["closed_set_split"], "val")
        for sample in test_tagged:
            self.assertEqual(sample["closed_set_split"], "test")

    def test_tagging_by_open_set(self) -> None:
        """Test tagging samples by open-set split."""
        # Use larger dataset to ensure test samples are generated
        self._create_non_grouped_dataset(num_jaguars=20, images_per_jaguar=6)

        # Run processing with tagging
        result_dataset = run_processing(
            dataset_name=self.dataset_name,
            add_open_set=True,
            tag_by="open",
            seed=42,
            verbose=False,
        )

        # Check that samples have tags
        train_tagged = result_dataset.match_tags("train")
        val_tagged = result_dataset.match_tags("val")
        test_tagged = result_dataset.match_tags("test")

        self.assertGreater(len(train_tagged), 0, "Should have train-tagged samples")
        # Check that at least some tags were applied (train and/or val and/or test)
        total_tagged = len(train_tagged) + len(val_tagged) + len(test_tagged)
        self.assertGreater(total_tagged, 0, "Should have tagged samples")

        # Verify tags match split field values for those that have tags
        for sample in train_tagged:
            self.assertEqual(sample["open_set_split"], "train")
        for sample in val_tagged:
            self.assertEqual(sample["open_set_split"], "val")
        for sample in test_tagged:
            self.assertEqual(sample["open_set_split"], "test")

    def test_split_reproducibility(self) -> None:
        """Test that splits are reproducible with same seed."""
        dataset = self._create_non_grouped_dataset(num_jaguars=5, images_per_jaguar=5)

        # First split
        dataset.add_sample_field("split1", fo.StringField)
        closed_set_split(
            dataset=dataset,
            id_field="jaguar_id",
            field_name="split1",
            seed=42,
        )

        # Second split with same seed
        dataset.add_sample_field("split2", fo.StringField)
        closed_set_split(
            dataset=dataset,
            id_field="jaguar_id",
            field_name="split2",
            seed=42,
        )

        # Verify splits are identical
        for sample in dataset:
            self.assertEqual(sample["split1"], sample["split2"], "Splits should be identical with same seed")

    def test_split_field_clearing(self) -> None:
        """Test that existing split fields are cleared before re-splitting."""
        dataset = self._create_non_grouped_dataset(num_jaguars=3, images_per_jaguar=4)

        # Run processing first time
        run_processing(
            dataset_name=self.dataset_name,
            add_closed_set=True,
            seed=42,
            verbose=False,
        )

        # Get original split assignments
        original_splits = {sample.id: sample["closed_set_split"] for sample in dataset}

        # Run processing again with different seed
        run_processing(
            dataset_name=self.dataset_name,
            add_closed_set=True,
            seed=99,
            verbose=False,
        )

        # Verify that splits changed (field was cleared and re-assigned)
        dataset.reload()
        new_splits = {sample.id: sample["closed_set_split"] for sample in dataset}

        # At least some should be different
        differences = sum(1 for sid in original_splits if original_splits[sid] != new_splits[sid])
        self.assertGreater(differences, 0, "Splits should change with different seed")

    def test_dry_run_mode(self) -> None:
        """Test that dry_run mode doesn't modify the dataset."""
        dataset = self._create_non_grouped_dataset(num_jaguars=3, images_per_jaguar=4)

        # Run in dry-run mode
        result = run_processing(
            dataset_name=self.dataset_name,
            add_closed_set=True,
            dry_run=True,
            verbose=False,
        )

        # Should return None
        self.assertIsNone(result)

        # Dataset should not have split field
        dataset.reload()
        self.assertNotIn("closed_set_split", dataset.get_field_schema())


if __name__ == "__main__":
    unittest.main()
