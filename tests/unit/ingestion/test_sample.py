"""Tests for video frame sampling module."""

import tempfile
import unittest
from pathlib import Path

import cv2
import fiftyone as fo
import numpy as np

from jaguars.ingestion.processing.sample import sample_video_frames


class TestSampler(unittest.TestCase):
    """Test suite for video frame sampling."""

    def setUp(self) -> None:
        """Create test fixtures before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.dataset_name = "test_sampler_dataset"

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

    def _create_test_video(self, path: Path, duration_seconds: float = 2.0, fps: int = 10) -> None:
        """Create a simple test video with changing content."""
        width, height = 640, 480
        num_frames = int(duration_seconds * fps)

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))  # type: ignore[attr-defined]

        for i in range(num_frames):
            # Create frame with changing content (frame number displayed)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add some variation to each frame
            color = (i * 10 % 256, (i * 20) % 256, (i * 30) % 256)
            cv2.rectangle(frame, (50, 50), (width - 50, height - 50), color, -1)
            cv2.putText(
                frame,
                f"Frame {i}",
                (width // 2 - 50, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            writer.write(frame)

        writer.release()

    def test_sample_frames_from_single_video(self) -> None:
        """Test sampling frames from a single video in a grouped dataset."""
        # Create test video
        video_path = self.data_dir / "test_video.mp4"
        self._create_test_video(video_path, duration_seconds=2.0, fps=10)

        # Create grouped dataset with one video
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        group_id = str(uuid.uuid4())

        video_sample = fo.Sample(filepath=str(video_path), group=fo.Group().element(group_id))
        video_sample.group.name = "video"
        video_sample["jaguar_id"] = "TestJaguar"
        video_sample["ground_truth"] = fo.Classification(label="TestJaguar")
        video_sample["site"] = "TestSite"
        dataset.add_sample(video_sample)

        # Sample frames: 2 fps for first 1 second, then 0.5 fps for rest
        dataset = sample_video_frames(
            dataset_name=self.dataset_name,
            early_fps=2.0,
            early_seconds=1.0,
            late_fps=0.5,
            output_format="jpg",
        )

        # Check that frames were added
        video_view = dataset.select_group_slices("video")
        image_view = dataset.select_group_slices("image")

        self.assertEqual(len(video_view), 1, "Should have 1 video sample")
        self.assertGreater(len(image_view), 0, "Should have sampled frames")

        # Get the actual video sample to check its group ID
        video_sample_loaded = video_view.first()

        # Check that frames are in the same group as the video
        for frame_sample in image_view:
            self.assertEqual(frame_sample.group.id, video_sample_loaded.group.id, "Frame should be in same group as video")
            self.assertEqual(frame_sample.group.name, "image", "Frame slice should be 'image'")
            self.assertEqual(frame_sample["source_type"], "video_frame")
            self.assertEqual(frame_sample["source"], video_sample_loaded.id)

            # Check metadata was copied
            self.assertEqual(frame_sample["jaguar_id"], "TestJaguar")
            self.assertEqual(frame_sample["ground_truth"].label, "TestJaguar")
            self.assertEqual(frame_sample["site"], "TestSite")

    def test_sample_early_phase_only(self) -> None:
        """Test sampling only the early phase frames."""
        # Create test video
        video_path = self.data_dir / "test_video.mp4"
        self._create_test_video(video_path, duration_seconds=2.0, fps=10)

        # Create dataset with video
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        group_id = str(uuid.uuid4())

        video_sample = fo.Sample(filepath=str(video_path), group=fo.Group().element(group_id))
        video_sample.group.name = "video"
        dataset.add_sample(video_sample)

        # Sample only early phase: 2 fps for first 0.5 seconds, then 0 fps (no late frames)
        dataset = sample_video_frames(
            dataset_name=self.dataset_name,
            early_fps=2.0,
            early_seconds=0.5,
            late_fps=0,  # No late phase sampling
            output_format="jpg",
        )

        # Check that frames were added
        image_view = dataset.select_group_slices("image")
        self.assertGreater(len(image_view), 0, "Should have sampled early phase frames")
        # With fps=10 and early_fps=2 for 0.5s, we should get ~1 frame (frame 0 at 0.0s)

    def test_sample_multiple_videos_in_different_groups(self) -> None:
        """Test sampling frames from multiple videos, each in their own group."""
        # Create two test videos
        video1_path = self.data_dir / "video1.mp4"
        video2_path = self.data_dir / "video2.mp4"
        self._create_test_video(video1_path, duration_seconds=1.0, fps=10)
        self._create_test_video(video2_path, duration_seconds=1.0, fps=10)

        # Create dataset with two videos in separate groups
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        import uuid

        group1_id = str(uuid.uuid4())
        video1_sample = fo.Sample(filepath=str(video1_path), group=fo.Group().element(group1_id))
        video1_sample.group.name = "video"
        video1_sample["jaguar_id"] = "Jaguar1"
        video1_sample["ground_truth"] = fo.Classification(label="Jaguar1")

        group2_id = str(uuid.uuid4())
        video2_sample = fo.Sample(filepath=str(video2_path), group=fo.Group().element(group2_id))
        video2_sample.group.name = "video"
        video2_sample["jaguar_id"] = "Jaguar2"
        video2_sample["ground_truth"] = fo.Classification(label="Jaguar2")

        dataset.add_samples([video1_sample, video2_sample])

        # Sample frames: 2 fps for first 0.3s, then 0.5 fps for rest
        dataset = sample_video_frames(
            dataset_name=self.dataset_name,
            early_fps=2.0,
            early_seconds=0.3,
            late_fps=0.5,
            output_format="jpg",
        )

        # Check that both videos have frames
        video_view = dataset.select_group_slices("video")
        image_view = dataset.select_group_slices("image")

        self.assertEqual(len(video_view), 2, "Should have 2 video samples")
        self.assertGreater(len(image_view), 0, "Should have sampled frames")

        # Get the actual group IDs from the loaded video samples
        video_group_ids = {v.group.id for v in video_view}

        # Build mapping of group ID to jaguar ID
        group_to_jaguar = {}
        for v in video_view:
            group_to_jaguar[v.group.id] = v["jaguar_id"]

        # Check that frames are properly grouped and have correct metadata
        for frame_sample in image_view:
            self.assertIn(frame_sample.group.id, video_group_ids)
            # Check that jaguar_id was copied correctly
            expected_jaguar_id = group_to_jaguar[frame_sample.group.id]
            self.assertEqual(frame_sample["jaguar_id"], expected_jaguar_id)
            # Check that ground_truth classification was copied
            self.assertEqual(frame_sample["ground_truth"].label, expected_jaguar_id)

    def test_no_videos_in_dataset(self) -> None:
        """Test that sampling handles datasets with no videos gracefully."""
        # Create dataset with only an image (no videos)
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_group_field("group", default="image")

        # Create a dummy image
        image_path = self.data_dir / "test_image.jpg"
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), img)

        import uuid

        group_id = str(uuid.uuid4())

        image_sample = fo.Sample(filepath=str(image_path), group=fo.Group().element(group_id))
        image_sample.group.name = "image"
        dataset.add_sample(image_sample)

        initial_count = len(dataset)

        # Try to sample frames (should not fail, just no frames added)
        dataset = sample_video_frames(
            dataset_name=self.dataset_name,
            early_fps=1.0,
            early_seconds=5.0,
            late_fps=1.0 / 6.0,
            output_format="jpg",
        )

        # Dataset should be unchanged
        self.assertEqual(len(dataset), initial_count, "Dataset should be unchanged when no videos present")


if __name__ == "__main__":
    unittest.main()
