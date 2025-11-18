"""Copy and preprocess video & image files for jaguar re-identification dataset.

Takes a CSV file with columns including:
- RAW DATA PATH: path to the root directory of the raw data
- FILE PATH: path to the file relative to RAW DATA PATH
- FILE EXTENSION: file extension of the file (e.g., .avi, .mp4, .jpg)
- FILE NAME: name of the file
- CAMERA TRAP SITE: identifier for the camera trap site
- CAM: camera number at the site

Further include:
- LATITUDE
- LONGITUDE
- CAMERA ID
- JAGUAR ID
- LOCATION
- CAMERA MODEL
- DATE
- TIME
- TEMP C
- DATETIME
- ORIGINAL FILE NAME
- SIGHTING ID
- RAW DATA PATH
- FILE TYPE
- ORIGINAL CAM
- ORIGINAL SITE

The script performs the following preprocessing steps:
1. Flattens the directory structure by copying files to a single 'files' directory
    with standardized filenames: <CAMERA TRAP SITE>_CAM<CAM>_<original filename>
2. Converts .avi files to .mp4 format using ffmpeg.
3. Updates the CSV file to reflect new file paths and extensions.

The outputted CSV file contains the following columns:
- DATASET PATH: path to the root directory of the processed dataset
- FILE PATH: path to the file relative to DATASET PATH
- ORIGINAL FILE EXTENSION: original file extension before conversion
- RAW FILE PATH: original file path relative to RAW DATA PATH

The FILE EXTENSION column is renamed to ORIGINAL FILE EXTENSION in the output CSV.
"""

import argparse
import json
import logging
import pandas as pd
from pathlib import Path
import shutil
import subprocess

from src.jaguar_reidentification.utils.utils import json_safe


def write_report(report: dict, report_path: Path) -> None:
    logging.info("Writing processing report to %s", report_path)
    report = json_safe(report)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)


def write_df(df: pd.DataFrame, output_csv: Path) -> None:
    logging.info("Writing adjusted labels to %s", output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def convert_avi_to_mp4(src: Path, dst: Path, overwrite: bool = True, codec: str = "libx264") -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logging.error("ffmpeg not found on PATH")
        return False

    # -n = no overwrite, -y = overwrite
    overwrite_flag = "-y" if overwrite else "-n"
    # Ensure destination directory exists
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
        # Video encoding settings (Windows-friendly)
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        # Uncomment to control quality/size further
        # "-crf",
        # "23",
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


def convert_all_avi_to_mp4(df: pd.DataFrame, dst: Path) -> tuple[pd.DataFrame, dict]:
    report = {
        "total_files": 0,
        "converted_files": 0,
        "failed_conversions": 0,
        "skipped_files": 0,
    }

    for idx, row in df.iterrows():
        if row["FILE EXTENSION"].lower() != ".avi":
            continue

        report["total_files"] += 1
        src_path = dst / row["FILE PATH"]
        dst_path = src_path.with_suffix(".MP4")

        if dst_path.exists():
            logging.info("Skipping conversion, output exists: %s", dst_path)
            df.loc[[idx], "FILE PATH"] = dst_path.relative_to(dst).as_posix()
            report["skipped_files"] += 1
            continue

        success = convert_avi_to_mp4(src_path, dst_path)
        if success:
            df.loc[[idx], "FILE PATH"] = dst_path.relative_to(dst).as_posix()
            report["converted_files"] += 1
        else:
            report["failed_conversions"] += 1
        
        try: 
            logging.info("Deleting original AVI file: %s", src_path)
            src_path.unlink()
        except Exception as e:
            logging.warning("Failed to delete original AVI file %s: %s", src_path, e)

    return df, report


def flatten_files(df: pd.DataFrame, dst: Path) -> pd.DataFrame:
    df["DATASET PATH"] = dst.as_posix()
    for idx, row in df.iterrows():
        src_path = Path(row["RAW DATA PATH"]) / row["FILE PATH"]

        old_file_name = row["FILE NAME"]
        new_file_name = (row["CAMERA TRAP SITE"] + "_CAM_" + row["CAM"] + "_" + old_file_name).replace(" ", "_")
        dst_path = dst / "files" / new_file_name

        df.loc[[idx], "RAW FILE PATH"] = row["FILE PATH"]
        df.loc[[idx], "FILE PATH"] = dst_path.relative_to(dst).as_posix()

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not dst_path.exists():
            logging.info("Copying %s -> %s", src_path, dst_path)
            shutil.copy2(src_path, dst_path)
        else:
            logging.debug("File already exists, skipping copy: %s", dst_path)

    return df


def check_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "RAW DATA PATH",
        "FILE PATH",
        "FILE EXTENSION",
        "FILE NAME",
        "CAMERA TRAP SITE",
        "CAM",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing_columns)}")

    return df


def load_df(input_csv: Path) -> pd.DataFrame:
    if input_csv.exists():
        logging.info("Loading raw labels from %s", input_csv)
    else:
        raise FileNotFoundError(f"Input CSV file does not exist: {input_csv}")

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        logging.error("Retrying after failed to load CSV file: %s", e)
        df = pd.read_csv(input_csv, sep=",")
        logging.info("Success: Loaded CSV file with ',' separator")

    logging.info("Loaded %d rows with columns: %s", len(df), ", ".join(df.columns))

    return check_required_columns(df)


def run(
    input_csv: Path,
    output_csv: Path,
    generate_report: bool = False,
) -> None:
    df = load_df(input_csv)

    df = flatten_files(df, output_csv.parent)
    df, report = convert_all_avi_to_mp4(df, output_csv.parent)

    if "Files Name" in df.columns:
        df.drop(columns=["Files Name"], inplace=True)
    if "FILE EXTENSION" in df.columns:
        df.rename(columns={"FILE EXTENSION": "ORIGINAL FILE EXTENSION"}, inplace=True)

    write_df(df, output_csv)

    if generate_report:
        write_report(report, output_csv.parent / "video_preprocessing_report.json")

def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess jaguar identification labels.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="labels.csv",
        help="Path to the raw labels CSV file. Or name of the CSV file in input_dir.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the cleaned labels CSV file. Also the root directory for processed files.",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Should generate a processing report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    run(
        Path(args.input_csv),
        Path(args.output_csv),
        args.generate_report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
