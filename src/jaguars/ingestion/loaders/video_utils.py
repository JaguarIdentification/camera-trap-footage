"""Utility functions for video format conversion used by loaders.

Provides helpers to convert `.avi` videos to `.mp4` using `ffmpeg`, with
Windows-friendly defaults. Mirrors the conversion logic in
`orientation/preprocessing/preprocess_videos.py` for reuse across ingestion
pipelines without requiring pandas.

Requirements:
- `ffmpeg` must be installed and available on PATH.

Example:
    from pathlib import Path
    from jaguars.ingestion.loaders.video_utils import convert_avi_to_mp4, batch_convert_avi

    src = Path("C:/data/raw/video.avi")
    dst = Path("C:/data/processed/video.mp4")
    ok = convert_avi_to_mp4(src, dst)

    # Batch from a directory
    avi_files = list(Path("C:/data/raw").rglob("*.avi"))
    report = batch_convert_avi(avi_files, delete_original=True)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


def _find_existing_mp4_for_stem(src: Path) -> Path | None:
    """Return an existing MP4 path matching ``src`` stem, regardless of case."""
    preferred = src.with_suffix(".MP4")
    if preferred.exists():
        return preferred

    lower = src.with_suffix(".mp4")
    if lower.exists():
        return lower

    for candidate in src.parent.glob(f"{src.stem}.*"):
        if candidate.is_file() and candidate.suffix.lower() == ".mp4":
            return candidate

    return None


def convert_avi_to_mp4(
    src: Path,
    dst: Path,
    overwrite: bool = True,
    codec: str = "libx264",
) -> bool:
    """Convert a single AVI file to MP4 using ffmpeg.

    - Preserves audio if present
    - Uses yuv420p pixel format for broad compatibility (Windows-friendly)
    - Adds `+faststart` to enable progressive playback

    Args:
        src: Source AVI file path
        dst: Destination MP4 file path
        overwrite: Whether to overwrite the destination file if it exists
        codec: Video codec to use (default: libx264)

    Returns:
        True if conversion succeeded and produced a non-empty file; False otherwise.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logging.error("ffmpeg not found on PATH")
        return False

    # -n = no overwrite, -y = overwrite
    overwrite_flag = "-y" if overwrite else "-n"
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        overwrite_flag,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        # Map first video stream, include audio if present
        "-map",
        "0:v:0",
        "-map",
        "0:a:?",
        # Video encoding settings
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        # Audio settings (if audio exists)
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        # Make file playable before full download
        "-movflags",
        "+faststart",
        str(dst),
    ]

    try:
        subprocess.run(cmd, check=True)
        # Basic sanity check: file exists and non-zero size
        if not dst.exists() or dst.stat().st_size == 0:
            logging.error("ffmpeg produced empty file for %s", dst)
            return False
        logging.info("Converted: %s -> %s", src, dst)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("ffmpeg failed for %s: %s", src, e)
        return False
    except Exception as e:
        logging.exception("Unexpected error converting %s: %s", src, e)
        return False


def batch_convert_avi(
    files: list[Path],
    *,
    delete_original: bool = False,
    overwrite: bool = False,
    codec: str = "libx264",
) -> dict[str, int]:
    """Convert a list of AVI files to MP4.

    Destination uses the same path with `.MP4` extension (uppercase to match
    existing convention). Optionally deletes original AVI files after successful
    conversion.

    Args:
        files: List of AVI file paths
        delete_original: Delete source AVI after successful conversion
        overwrite: Whether to overwrite existing destination files
        codec: Video codec to use (default: libx264)

    Returns:
        Report dictionary with counts: total_files, converted_files,
        failed_conversions, skipped_files
    """
    report = {
        "total_files": 0,
        "converted_files": 0,
        "failed_conversions": 0,
        "skipped_files": 0,
    }

    for src in tqdm(files, desc="Converting AVI to MP4"):
        if src.suffix.lower() != ".avi":
            continue

        report["total_files"] += 1
        existing_mp4 = _find_existing_mp4_for_stem(src)
        dst = src.with_suffix(".MP4")

        if existing_mp4 is not None and not overwrite:
            logging.info("Skipping conversion, output exists: %s", existing_mp4)
            report["skipped_files"] += 1
            continue

        if existing_mp4 is not None and overwrite:
            dst = existing_mp4

        ok = convert_avi_to_mp4(src, dst, overwrite=overwrite, codec=codec)
        if ok:
            report["converted_files"] += 1
            if delete_original:
                try:
                    logging.info("Deleting original AVI file: %s", src)
                    src.unlink()
                except Exception as e:
                    logging.warning("Failed to delete original AVI file %s: %s", src, e)
        else:
            report["failed_conversions"] += 1

    return report
