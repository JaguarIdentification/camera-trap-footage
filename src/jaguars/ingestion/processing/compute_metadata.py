r"""Video frame sampling module.

Samples frames from videos in a FiftyOne grouped dataset and adds sampled frames
to the same group as their source video, in the "image" slice.

Sampling strategy (configurable):
- Phase 1 (early_fps): Sample at high fps for the first early_seconds of video
- Phase 2 (late_fps): Sample at lower fps for the remainder of video

Example: 1 fps for first 5s, then 1 frame every 6s (fps=1/6) for the rest.

Usage (via CLI):
    python src/jaguars/ingestion/processing/sample.py --verbose

    # Or adjust parameters as needed
    python src/jaguars/ingestion/processing/sample.py \\
        --dataset JID_Master_Dataset \\
        --early-fps 2.0 \\
        --early-seconds 10.0 \\
        --late-fps 0.5 \\
        --output-format jpg \\
        --verbose
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import fiftyone as fo
from tqdm import tqdm

from jaguars.common.config import DEFAULT_GROUP_SLICE, JID_MASTER_DATASET
from jaguars.common.fiftyone_utils import get_or_create_dataset
from jaguars.common.logging_utils import setup_logger

MODULE_NAME = "ingestion.processing.sample"
logger = setup_logger(MODULE_NAME)


def validate_resources(dataset_name: str) -> None:
    """Checks if dataset exists and has videos."""
    if dataset_name not in fo.list_datasets():
        raise ValueError(f"Dataset '{dataset_name}' does not exist")


def run_processing(
    dataset_name: str = JID_MASTER_DATASET,
    dry_run: bool = False,
    verbose: bool = False,
) -> fo.Dataset | None:
    """Core Logic for computing metadata.

    Computes metadata for videos and images in the dataset.

    Args:
        dataset_name: Name of the FiftyOne dataset
        dry_run: If True, only validate resources without processing
        verbose: Enable detailed logging

    Returns:
        Updated FiftyOne dataset with computed metadata or None if dry_run is True
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)
    logger_instance.info("Starting metadata computation for dataset: %s", dataset_name)

    validate_resources(dataset_name)

    if dry_run:
        logger_instance.info("DRY RUN: Would compute metadata for dataset '%s'", dataset_name)
        return None

    # Load dataset and compute metadata
    dataset = fo.load_dataset(dataset_name)
    dataset.compute_metadata()

    logger_instance.info("Video frame sampling completed successfully.")
    return dataset


def sample_video_frames(
    dataset_name: str = JID_MASTER_DATASET,
    early_fps: float = 1.0,
    early_seconds: float = 5.0,
    late_fps: float = 1.0 / 6.0,
    output_format: str = "jpg",
) -> fo.Dataset:
    """Sample frames from videos in a FiftyOne grouped dataset.

    Uses a two-phase sampling strategy:
    - Phase 1: Sample at early_fps for the first early_seconds of each video
    - Phase 2: Sample at late_fps for the remainder of each video

    This function:
    1. Finds all video samples in the dataset (slice="video")
    2. Extracts frames from each video using OpenCV
    3. Adds sampled frames as image samples to the same group as the source video
    4. Preserves metadata links between videos and their frames

    Args:
        dataset_name: Name of the FiftyOne grouped dataset
        early_fps: Frames per second to sample in first early_seconds (default: 1.0)
        early_seconds: Duration in seconds to sample at early_fps (default: 5.0)
        late_fps: Frames per second to sample for remainder (default: ~0.167 = 1/6 fps)
        output_format: Image format (jpg, png, etc.)

    Returns:
        Updated FiftyOne dataset with sampled frames added
    """
    dataset = get_or_create_dataset(dataset_name)

    # Ensure dataset has group field
    if dataset.group_field is None:
        dataset.add_group_field("group", default=DEFAULT_GROUP_SLICE)
        logger.info("Added group field to dataset")

    # Get all video samples - check if video slice exists first
    try:
        video_view = dataset.select_group_slices("video")
        num_videos = len(video_view)
    except ValueError:
        # No video slice exists
        logger.warning("No video slice found in dataset")
        return dataset

    if num_videos == 0:
        logger.warning("No video samples found in dataset")
        return dataset

    logger.info("Found %d videos to sample from", num_videos)

    # Get media directory for storing frames
    base_dir = Path(fo.config.default_dataset_dir) / dataset_name / "media" / "frames"
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Sampling strategy: %s fps for first %s seconds, then %s fps for remainder", early_fps, early_seconds, late_fps)

    # Now process each video and sample frames
    new_frame_samples = []
    total_frames = 0

    for video_sample in tqdm(video_view, desc="Sampling videos", unit="video"):
        video_filepath = Path(video_sample.filepath)
        video_name = video_filepath.stem

        # Open video and extract frame information
        cap = cv2.VideoCapture(str(video_filepath))
        if not cap.isOpened():
            logger.warning("Could not open video: %s", video_filepath)
            continue

        fps_actual = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps_actual <= 0 or frame_count <= 0:
            logger.warning("Invalid video metadata for %s (fps=%s, frames=%s)", video_name, fps_actual, frame_count)
            cap.release()
            continue

        # Calculate phase boundary
        early_frame_count = int(fps_actual * early_seconds)

        # Create output directory for this video
        # Append sample ID to ensure uniqueness in case of duplicate filenames
        frames_dir = base_dir / f"{video_name}_{video_sample.id}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Determine which frames to extract
        frames_to_extract = set()

        # Phase 1: Early frames at early_fps
        if early_fps > 0:
            early_interval = max(1, int(fps_actual / early_fps))
            for frame_idx in range(0, min(early_frame_count, frame_count), early_interval):
                frames_to_extract.add(frame_idx)

        # Phase 2: Late frames at late_fps
        if late_fps > 0:
            late_interval = max(1, int(fps_actual / late_fps))
            for frame_idx in range(early_frame_count, frame_count, late_interval):
                frames_to_extract.add(frame_idx)

        if not frames_to_extract:
            logger.warning("No frames to extract for video %s", video_name)
            cap.release()
            continue

        logger.debug(
            "Extracting %d frames from video %s (fps=%s, total_frames=%s, early_frames=%s)",
            len(frames_to_extract),
            video_name,
            fps_actual,
            frame_count,
            early_frame_count,
        )

        # Extract and save frames
        frame_number = 0
        current_frame_idx = 0
        extracted_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx in frames_to_extract:
                # Save frame to disk
                frame_path = frames_dir / f"{frame_number:06d}.{output_format}"
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append((frame_number, frame_path))
                frame_number += 1

            current_frame_idx += 1

        cap.release()

        logger.debug("Extracted %d frames for video %s", len(extracted_frames), video_name)

        # Create image samples for each extracted frame
        for frame_num, frame_path in tqdm(extracted_frames, desc=f"Adding frames for {video_name}", unit="frame"):
            # Create new sample in the same group as the source video
            frame_sample = fo.Sample(filepath=str(frame_path))
            frame_sample["group"] = fo.Group(id=video_sample["group"].id, name=DEFAULT_GROUP_SLICE)

            # Add metadata linking back to source video
            frame_sample["source_type"] = "video_frame"
            frame_sample["source"] = video_sample.id
            frame_sample["source_filepath"] = str(video_filepath)
            frame_sample["frame_number"] = frame_num

            # Copy relevant metadata from video to frame
            if "jaguar_id" in video_sample:
                frame_sample["jaguar_id"] = video_sample["jaguar_id"]
            if "ground_truth" in video_sample:
                frame_sample["ground_truth"] = video_sample["ground_truth"]
            if "site" in video_sample:
                frame_sample["site"] = video_sample["site"]
            if "cam" in video_sample:
                frame_sample["cam"] = video_sample["cam"]
            if "date" in video_sample:
                frame_sample["date"] = video_sample["date"]
            if "time" in video_sample:
                frame_sample["time"] = video_sample["time"]
            if "datetime" in video_sample:
                frame_sample["datetime"] = video_sample["datetime"]
            if "location" in video_sample:
                frame_sample["location"] = video_sample["location"]
            if "sighting_id" in video_sample:
                frame_sample["sighting_id"] = video_sample["sighting_id"]

            new_frame_samples.append(frame_sample)
            total_frames += 1

    # Add all new frame samples to the dataset
    if new_frame_samples:
        dataset.add_samples(new_frame_samples)
        dataset.save()
        logger.info("Added %d sampled frames from %d videos", total_frames, num_videos)
    else:
        logger.warning("No frames were sampled from any videos")

    # Log final summary
    summary = {
        "videos_processed": num_videos,
        "frames_added": total_frames,
        "total_image_samples": len(dataset.select_group_slices(DEFAULT_GROUP_SLICE)),
        "total_video_samples": len(dataset.select_group_slices("video")),
        "total_samples": len(dataset),
    }
    logger.info("Sampling Summary: %s", json.dumps(summary, indent=2))

    return dataset


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="Sample frames from videos in a FiftyOne grouped dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=JID_MASTER_DATASET,
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    try:
        run_processing(
            dataset_name=args.dataset,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error("Error during compute metadata: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
