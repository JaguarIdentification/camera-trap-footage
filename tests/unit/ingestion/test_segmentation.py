"""Tests for segmentation processing module."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fiftyone as fo

from jaguars.ingestion.processing.segmentation import cleanup_dataset, filter_by_count, filter_unexpected_segmentations, run_processing


class TestSegmentationProcessing(unittest.TestCase):
    """Test suite for segmentation processing."""

    def setUp(self) -> None:
        """Create test dataset."""
        self.dataset_name = "test_segmentation_processing"

        # Clean up any existing test dataset
        if self.dataset_name in fo.list_datasets():
            dataset = fo.load_dataset(self.dataset_name)
            dataset.delete()

        self.dataset = fo.Dataset(self.dataset_name)
        self.segmentation_field = "sam3_segmentations"

    def tearDown(self) -> None:
        """Clean up."""
        if self.dataset_name in fo.list_datasets():
            dataset = fo.load_dataset(self.dataset_name)
            dataset.delete()

    def test_filter_by_count(self) -> None:
        """Test filtering by detection count."""
        # Add 3 samples: 0 dets, 1 det, 2 dets
        samples = []

        # 0 detections (None field)
        s0 = fo.Sample(filepath="image0.jpg")
        samples.append(s0)

        # 1 detection
        s1 = fo.Sample(filepath="image1.jpg")
        s1[self.segmentation_field] = fo.Detections(detections=[fo.Detection(label="jaguar", confidence=0.9)])
        samples.append(s1)

        # 2 detections
        s2 = fo.Sample(filepath="image2.jpg")
        s2[self.segmentation_field] = fo.Detections(
            detections=[fo.Detection(label="jaguar", confidence=0.8), fo.Detection(label="jaguar", confidence=0.7)]
        )
        samples.append(s2)

        self.dataset.add_samples(samples)

        # Filter (expecting 1)
        count = filter_by_count(self.dataset, self.segmentation_field, expected_count=1)

        # Should tag 2 samples (s0 and s2)
        self.assertEqual(count, 2)

        # Check tags
        tagged_view = self.dataset.match_tags("filter_count")
        self.assertEqual(len(tagged_view), 2)

        filepaths = [Path(fp).name for fp in tagged_view.values("filepath")]
        self.assertIn("image0.jpg", filepaths)
        self.assertIn("image2.jpg", filepaths)
        self.assertNotIn("image1.jpg", filepaths)

    def test_filter_unexpected_segmentations(self) -> None:
        """Test filtering by quality (confidence/area)."""
        samples = []

        # Good sample
        s_good = fo.Sample(filepath="image_good.jpg")
        s_good[self.segmentation_field] = fo.Detections(
            detections=[fo.Detection(label="jaguar", confidence=0.9, bounding_box=[0.2, 0.2, 0.5, 0.5])]  # Area = 0.25 (Good)
        )
        samples.append(s_good)

        # Bad confidence
        s_bad_conf = fo.Sample(filepath="image_bad_conf.jpg")
        s_bad_conf[self.segmentation_field] = fo.Detections(
            detections=[fo.Detection(label="jaguar", confidence=0.4, bounding_box=[0.2, 0.2, 0.5, 0.5])]  # < 0.5
        )
        samples.append(s_bad_conf)

        # Bad area (too small)
        s_small = fo.Sample(filepath="image_small.jpg")
        s_small[self.segmentation_field] = fo.Detections(
            detections=[fo.Detection(label="jaguar", confidence=0.9, bounding_box=[0.1, 0.1, 0.05, 0.05])]  # Area = 0.0025 < 0.01
        )
        samples.append(s_small)

        # Bad area (too large)
        s_large = fo.Sample(filepath="image_large.jpg")
        s_large[self.segmentation_field] = fo.Detections(
            detections=[
                fo.Detection(
                    label="jaguar",
                    confidence=0.9,
                    bounding_box=[0.0, 0.0, 1.0, 1.0],  # Area = 1.0 >= 1.0 (Wait, max default is 1.0, let's test strict >)
                )
            ]
        )
        # 1.0 is default max, so let's set max to 0.9 for test
        samples.append(s_large)

        self.dataset.add_samples(samples)

        count = filter_unexpected_segmentations(self.dataset, self.segmentation_field, min_confidence=0.5, min_area_rel=0.01, max_area_rel=0.9)

        # Should tag 3 samples (bad_conf, small, large)
        self.assertEqual(count, 3)

        tagged_view = self.dataset.match_tags("filter_quality")
        filepaths = [Path(fp).name for fp in tagged_view.values("filepath")]
        self.assertIn("image_bad_conf.jpg", filepaths)
        self.assertIn("image_small.jpg", filepaths)
        self.assertIn("image_large.jpg", filepaths)
        self.assertNotIn("image_good.jpg", filepaths)

    def test_cleanup_dataset(self) -> None:
        """Test deleting tagged samples."""
        s1 = fo.Sample(filepath="1.jpg", tags=["delete_me"])
        s2 = fo.Sample(filepath="2.jpg", tags=["keep_me"])
        self.dataset.add_samples([s1, s2])

        deleted_count = cleanup_dataset(self.dataset, ["delete_me"])

        self.assertEqual(deleted_count, 1)
        self.assertEqual(len(self.dataset), 1)
        # Fiftyone stores absolute paths by default or relative. In test env they might be cast.
        # Just check it ends with 2.jpg or compare basename
        self.assertTrue(self.dataset.first().filepath.endswith("2.jpg"))

    @patch("jaguars.ingestion.processing.segmentation.run_sam3")
    def test_run_processing(self, mock_sam3: MagicMock) -> None:
        """Test full pipeline orchestration."""
        # Setup mock to return self.dataset
        mock_sam3.return_value = self.dataset

        # Add samples to dataset that mock SAM3 would "produce"
        # 1. Good sample
        s1 = fo.Sample(filepath="good.jpg")
        s1[self.segmentation_field] = fo.Detections(detections=[fo.Detection(label="jaguar", confidence=0.9, bounding_box=[0, 0, 0.5, 0.5])])

        # 2. Bad sample (0 detections)
        s2 = fo.Sample(filepath="bad.jpg")

        self.dataset.add_samples([s1, s2])

        result_ds = run_processing(dataset_name=self.dataset_name, segmentation_field=self.segmentation_field, dry_run=False)

        # Check SAM3 was called
        mock_sam3.assert_called_once()

        # Check that bad sample was deleted (because run_processing calls cleanup)
        self.assertEqual(len(result_ds), 1)
        self.assertTrue(result_ds.first().filepath.endswith("good.jpg"))


if __name__ == "__main__":
    unittest.main()
