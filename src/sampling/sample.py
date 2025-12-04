import os
import sys
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Optional

# /Users/mehdisaurus/Documents/1Drittes/CV/jaguar project/camera-trap-footage/src/fiftyone/sample.py
"""
Sample screenshots from all videos under data/intermediate/v1/files.

Uses `fiftyone.utils.video.sample_videos` to extract frames from videos.
Output is written to `data/intermediate/v1/screenshots/<video_stem>/*.jpg`
"""

try:
    import fiftyone as fo
    import fiftyone.utils.video as fouv
except Exception:
    fo = None
    fouv = None

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MPG", ".mpg", ".MP4"}


def find_videos(root: Path):
    print(f"Searching for videos in: {root}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


@dataclass
class Config:
    input: Path
    output: Path
    fps: float
    fmt: str


def parse_args(argv: Optional[list] = None) -> Config:
    parser = argparse.ArgumentParser(description="Sample screenshots from videos using fiftyone.utils.video")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/intermediate/v1/files"),
        help="Directory containing input videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/intermediate/v1/screenshots"),
        help="Directory to write screenshots",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to sample (per video). Use 0 to sample one frame (first) only.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        dest="format",
        help="Image format for screenshots (jpg, png, ...)",
    )

    args = parser.parse_args(argv)
    return Config(input=args.input, output=args.output, fps=args.fps, fmt=args.format)


def sample_videos_with_fiftyone(videos: list, output_dir: Path, fps: float, fmt: str = "jpg"):
    """Sample videos using FiftyOne's sample_videos function"""
    if fo is None or fouv is None:
        raise RuntimeError("FiftyOne is not available")
    
    if not hasattr(fouv, "sample_videos"):
        raise RuntimeError("fiftyone.utils.video.sample_videos is not available")
    
    # Create a temporary FiftyOne dataset from the video files
    samples = []
    for video_path in videos:
        sample = fo.Sample(filepath=str(video_path))
        samples.append(sample)
    
    # Create a temporary dataset
    dataset = fo.Dataset("temp_video_sampling")
    dataset.add_samples(samples)
    
    try:
        # Use sample_videos to extract frames
        frames_patt = f"%06d.{fmt}"
        
        # Handle fps=0 case (extract only first frame)
        if fps == 0:
            sample_fps = None
            frames = 1
        else:
            sample_fps = fps
            frames = None
        
        fouv.sample_videos(
            dataset,
            frames_patt=frames_patt,
            fps=sample_fps,
            frames=frames,
            output_dir=str(output_dir),
            skip_failures=True,
            verbose=True
        )
        
    finally:
        # Clean up the temporary dataset
        dataset.delete()


def main():
    config = parse_args()

    if not config.input.exists():
        print(f"Input directory does not exist: {config.input}", file=sys.stderr)
        sys.exit(1)

    videos = list(find_videos(config.input))
    if not videos:
        print("No video files found.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(videos)} videos. Sampling to {config.output} at {config.fps} fps.")

    # Ensure output directory exists
    config.output.mkdir(parents=True, exist_ok=True)

    try:
        sample_videos_with_fiftyone(videos, config.output, config.fps, config.fmt)
        print("Done.")
    except Exception as e:
        print(f"Video sampling failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()