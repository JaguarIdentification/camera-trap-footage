"""Tests for metadata computation module."""

import tempfile
import unittest
from pathlib import Path

import cv2
import fiftyone as fo
import numpy as np

from jaguars.ingestion.processing.compute_metadata import run_processing


class TestComputeMetadata(unittest.TestCase):
    """Test suite for metadata computation."""

    def setUp(self) -> None:
        """Create test fixtures before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.dataset_name = "test_metadata_dataset"

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

    def _create_test_image(self, path: Path, width: int = 640, height: int = 480) -> None:
        """Create a simple test image."""
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def _create_test_video(self, path: Path, duration_seconds: float = 1.0, fps: int = 10) -> None:
        """Create a simple test video."""
        width, height = 640, 480
        num_frames = int(duration_seconds * fps)

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))  # type: ignore[attr-defined]

        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()

    def test_compute_metadata_for_images(self) -> None:
        """Test computing metadata for image samples."""
        # Create test images with different sizes
        img1_path = self.data_dir / "image1.jpg"
        img2_path = self.data_dir / "image2.jpg"
        self._create_test_image(img1_path, width=640, height=480)
        self._create_test_image(img2_path, width=1280, height=720)

        # Create dataset with images
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        group1_id = str(uuid.uuid4())
        sample1 = fo.Sample(filepath=str(img1_path), group=fo.Group().element(group1_id))
        sample1.group.name = "image"

        group2_id = str(uuid.uuid4())
        sample2 = fo.Sample(filepath=str(img2_path), group=fo.Group().element(group2_id))
        sample2.group.name = "image"

        dataset.add_samples([sample1, sample2])

        # Initially metadata should not exist
        for sample in dataset:
            self.assertIsNone(sample.metadata)

        # Compute metadata
        dataset = run_processing(
            dataset_name=self.dataset_name,
            verbose=True,
        )

        # Check metadata was added
        self.assertIsNotNone(dataset)
        image_view = dataset.select_group_slices("image")
        self.assertEqual(len(image_view), 2)

        # Verify metadata for first image
        sample1 = image_view.first()
        self.assertIsNotNone(sample1.metadata)
        self.assertEqual(sample1.metadata.width, 640)
        self.assertEqual(sample1.metadata.height, 480)
        self.assertEqual(sample1.metadata.num_channels, 3)

    def test_compute_metadata_for_videos(self) -> None:
        """Test computing metadata for video samples."""
        # Create test video
        video_path = self.data_dir / "video.mp4"
        self._create_test_video(video_path, duration_seconds=2.0, fps=10)

        # Create dataset with video
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        group_id = str(uuid.uuid4())
        video_sample = fo.Sample(filepath=str(video_path), group=fo.Group().element(group_id))
        video_sample.group.name = "video"
        dataset.add_sample(video_sample)

        # Compute metadata
        dataset = run_processing(
            dataset_name=self.dataset_name,
            verbose=True,
        )

        # Check metadata was added
        self.assertIsNotNone(dataset)
        video_view = dataset.select_group_slices("video")
        video_sample = video_view.first()

        self.assertIsNotNone(video_sample.metadata)
        self.assertEqual(video_sample.metadata.frame_width, 640)
        self.assertEqual(video_sample.metadata.frame_height, 480)
        self.assertAlmostEqual(video_sample.metadata.frame_rate, 10.0, places=1)
        self.assertGreater(video_sample.metadata.total_frame_count, 0)

    def test_compute_metadata_mixed_dataset(self) -> None:
        """Test computing metadata for dataset with both images and videos."""
        # Create test files
        img_path = self.data_dir / "image.jpg"
        video_path = self.data_dir / "video.mp4"
        self._create_test_image(img_path)
        self._create_test_video(video_path, duration_seconds=1.0, fps=10)

        # Create dataset with both
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        # Add image
        group1_id = str(uuid.uuid4())
        img_sample = fo.Sample(filepath=str(img_path), group=fo.Group().element(group1_id))
        img_sample.group.name = "image"

        # Add video
        group2_id = str(uuid.uuid4())
        video_sample = fo.Sample(filepath=str(video_path), group=fo.Group().element(group2_id))
        video_sample.group.name = "video"

        dataset.add_samples([img_sample, video_sample])

        # Compute metadata
        dataset = run_processing(
            dataset_name=self.dataset_name,
            verbose=True,
        )

        # Check both have metadata
        image_view = dataset.select_group_slices("image")
        video_view = dataset.select_group_slices("video")

        img_sample = image_view.first()
        video_sample = video_view.first()

        self.assertIsNotNone(img_sample.metadata)
        self.assertIsNotNone(video_sample.metadata)

    def test_overwrite_existing_metadata(self) -> None:
        """Test overwriting existing metadata."""
        # Create test image
        img_path = self.data_dir / "image.jpg"
        self._create_test_image(img_path, width=800, height=600)

        # Create dataset
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        group_id = str(uuid.uuid4())
        sample = fo.Sample(filepath=str(img_path), group=fo.Group().element(group_id))
        sample.group.name = "image"
        dataset.add_sample(sample)

        # Compute metadata first time
        dataset = run_processing(
            dataset_name=self.dataset_name,
            overwrite=False,
            verbose=True,
        )

        # Store original metadata
        sample = dataset.select_group_slices("image").first()
        original_metadata = sample.metadata

        # Compute again with overwrite=True
        dataset = run_processing(
            dataset_name=self.dataset_name,
            overwrite=True,
            verbose=True,
        )

        # Metadata should still exist
        sample = dataset.select_group_slices("image").first()
        self.assertIsNotNone(sample.metadata)
        self.assertEqual(sample.metadata.width, original_metadata.width)

    def test_dry_run_mode(self) -> None:
        """Test that dry run mode doesn't modify the dataset."""
        # Create test image
        img_path = self.data_dir / "test_image.jpg"
        self._create_test_image(img_path)

        # Create dataset
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        group_id = str(uuid.uuid4())
        sample = fo.Sample(filepath=str(img_path), group=fo.Group().element(group_id))
        sample.group.name = "image"
        dataset.add_sample(sample)

        # Run in dry-run mode
        result = run_processing(
            dataset_name=self.dataset_name,
            dry_run=True,
            verbose=True,
        )

        # Check nothing was modified
        self.assertIsNone(result)
        dataset = fo.load_dataset(self.dataset_name)
        sample = dataset.first()
        self.assertIsNone(sample.metadata)

    def test_empty_dataset(self) -> None:
        """Test handling of empty dataset."""
        # Create empty dataset
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        # Compute metadata (should not fail)
        dataset = run_processing(
            dataset_name=self.dataset_name,
            verbose=True,
        )

        # Dataset should still exist but be empty
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 0)

    def test_invalid_dataset_name(self) -> None:
        """Test that invalid dataset name raises error."""
        with self.assertRaises(ValueError):
            run_processing(
                dataset_name="nonexistent_dataset",
                verbose=True,
            )

    def test_summary_generation(self) -> None:
        """Test that summary file is generated when requested."""
        # Create test image
        img_path = self.data_dir / "image.jpg"
        self._create_test_image(img_path)

        # Create dataset
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        group_id = str(uuid.uuid4())
        sample = fo.Sample(filepath=str(img_path), group=fo.Group().element(group_id))
        sample.group.name = "image"
        dataset.add_sample(sample)

        # Define summary path
        summary_path = self.data_dir / "summary.json"

        # Compute metadata with summary
        dataset = run_processing(
            dataset_name=self.dataset_name,
            summary_location=summary_path,
            verbose=True,
        )

        # Check summary file exists
        self.assertTrue(summary_path.exists())

        # Load and verify summary content
        import json

        with open(summary_path) as f:
            summary = json.load(f)

        self.assertIn("dataset", summary)
        self.assertEqual(summary["dataset"], self.dataset_name)
        self.assertIn("total_samples", summary)
        self.assertEqual(summary["total_samples"], 1)


if __name__ == "__main__":
    unittest.main()
