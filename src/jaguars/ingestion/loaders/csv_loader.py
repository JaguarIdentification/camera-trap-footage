"""CSV ingestion loader module.

Usage (via CLI):
    # Default run command:
    python src/jaguars/ingestion/loaders/csv_loader.py --input-dir "data/raw/17_11_2025" --verbose
    
    # With all options:
    python src/jaguars/ingestion/loaders/csv_loader.py \
        --input-dir "data/raw/17_11_2025" \
        --input-csv "labels.csv" \
        --dataset "JID_Master_Dataset" \
        --output-cleaned-csv "data/intermediate/v1/cleaned_labels.csv" \
        --generate-report \
        --verbose
"""

import argparse
import difflib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import fiftyone as fo
import pandas as pd

from jaguars.common.config import DEFAULT_GROUP_SLICE, GROUP_FIELD_NAME, JID_MASTER_DATASET, VIDEO_GROUP_SLICE
from jaguars.common.fiftyone_utils import get_or_create_dataset
from jaguars.common.logging_utils import setup_logger
from jaguars.ingestion.loaders.video_utils import batch_convert_avi

MODULE_NAME = "ingestion.loaders.csv_loader"
logger = setup_logger(MODULE_NAME)


def validate_resources(input_dir: Path, input_csv: Path) -> None:
    """Checks if inputs exist."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    csv_path = _as_path(input_dir, input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")


def write_summary(summary_data: dict[str, Any], summary_path: Path) -> None:
    """Generates summary report."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4, default=str)


def run_processing(
    dataset_name: str,
    input_dir: Path,
    input_csv: Path = Path("labels.csv"),
    output_cleaned_csv: Path | None = None,
    generate_report: bool = False,
    auto_match_missing: bool = True,
    match_threshold: float = 0.95,
    suggest_threshold: float = 0.80,
    summary_location: str | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> fo.Dataset | None:
    """Core Logic for CSV ingestion. Wrapper around ingest_csv_labels to match the common module pattern."""
    logger = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)
    logger.info("Starting processing for dataset: %s", dataset_name)

    validate_resources(input_dir, input_csv)

    if dry_run:
        logger.info("Dry run enabled - no changes made.")
        return None

    # Call the actual implementation
    ds = ingest_csv_labels(
        input_dir=input_dir,
        input_csv=input_csv,
        dataset_name=dataset_name,
        output_cleaned_csv=output_cleaned_csv,
        generate_report=generate_report,
        auto_match_missing=auto_match_missing,
        match_threshold=match_threshold,
        suggest_threshold=suggest_threshold,
    )

    if summary_location:
        summary_data = {
            "status": "success",
            "dataset_name": dataset_name,
            "input_dir": str(input_dir),
            "input_csv": str(input_csv),
            "processed": True,
        }
        write_summary(summary_data, Path(summary_location))

    logger.info("Processing completed successfully.")
    return ds


def _as_path(input_dir: Path, maybe_rel: Path) -> Path:
    if maybe_rel.exists():
        return maybe_rel
    return input_dir / maybe_rel


def _json_safe(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    return str(obj)


def write_report(report: dict[str, Any], report_path: Path) -> None:
    """Write processing report to JSON file."""
    logger.info("Writing processing report to %s", report_path)
    report = _json_safe(report)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)


def find_videos_without_labels(df: pd.DataFrame, input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find video files in input_dir that do not have a corresponding entry in df."""
    labeled_files = set()
    if len(df) > 0 and "FILE PATH" in df.columns:
        labeled_files = set(df["FILE PATH"].dropna().astype(str).tolist())

    all_video_files = set()

    for ext in [".MP4", ".mp4", ".mov", ".MOV", ".avi", ".AVI"]:
        video_files = list(input_dir.rglob(f"*{ext}"))
        all_video_files.update([str(p.relative_to(input_dir).as_posix()) for p in video_files])

    unlabeled_files = all_video_files - labeled_files
    logger.info("Found %d video files without labels", len(unlabeled_files))

    report = {
        "num_videos_without_labels": len(unlabeled_files),
        "videos_without_labels": sorted(list(unlabeled_files)),
    }

    return df, report


def filter_rows_with_invalid_paths(df: pd.DataFrame, input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter out rows where the FILE PATH is empty or does not exist on disk."""
    non_empty_path_mask = df["FILE PATH"].apply(lambda p: isinstance(p, str) and p.strip() != "")
    num_empty = (~non_empty_path_mask).sum()
    logger.info("Filtering out %d rows with empty FILE PATHs", num_empty)

    filtered_df = df[non_empty_path_mask].reset_index(drop=True)

    exists_mask = filtered_df["FILE PATH"].apply(lambda p: (input_dir / p).exists())
    num_not_exist = (~exists_mask).sum()
    logger.info("Filtering out %d rows where FILE PATH not found", num_not_exist)

    report = {
        "num_empty_file_paths": int(num_empty),
        "num_file_path_not_found": int(num_not_exist),
        "file_paths_not_found": sorted(filtered_df.loc[~exists_mask, "FILE PATH"].tolist()),
    }

    filtered_df = filtered_df[exists_mask].reset_index(drop=True)
    return filtered_df, report


def convert_avi_entries(df: pd.DataFrame, input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert AVI files referenced in the dataframe to MP4 and update paths.

    - Finds rows with FILE PATH ending in .avi/.AVI or FILE EXTENSION == .avi/.AVI
    - Converts corresponding files on disk to .MP4 using ffmpeg via `batch_convert_avi`
    - Updates the dataframe's FILE PATH to the new .MP4 relative path when conversion succeeded
    - Updates FILE EXTENSION to '.MP4' for converted rows

    Returns the possibly updated dataframe and a conversion report dict.
    """
    if "FILE PATH" not in df.columns:
        return df, {"total_files": 0, "converted_files": 0, "failed_conversions": 0, "skipped_files": 0}

    # Identify AVI files by extension or path suffix
    def _is_avi_row(row: pd.Series) -> bool:
        ext = str(row.get("FILE EXTENSION", "")).lower()
        path = str(row.get("FILE PATH", ""))
        return ext == ".avi" or path.lower().endswith(".avi")

    avi_rows = df[df.apply(_is_avi_row, axis=1)].copy()
    avi_abs_paths = []
    for _, row in avi_rows.iterrows():
        rel = row.get("FILE PATH")
        if isinstance(rel, str) and rel.strip():
            p = (input_dir / rel).resolve()
            if p.exists():
                avi_abs_paths.append(p)

    if not avi_abs_paths:
        return df, {"total_files": 0, "converted_files": 0, "failed_conversions": 0, "skipped_files": 0}

    # Perform batch conversion (do not delete originals by default)
    report = batch_convert_avi(avi_abs_paths, delete_original=False, overwrite=True)

    # Update dataframe paths for successfully converted files
    converted_set = set()
    for src in avi_abs_paths:
        dst = src.with_suffix(".MP4")
        # Consider as converted if destination exists (even if reported as skipped due to pre-existence)
        if dst.exists():
            converted_set.add(src)

    if converted_set:
        for idx, row in df.iterrows():
            rel = row.get("FILE PATH")
            if not isinstance(rel, str) or not rel.strip():
                continue
            abs_path = (input_dir / rel).resolve()
            if abs_path in converted_set:
                new_rel = Path(rel).with_suffix(".MP4").as_posix()
                df.at[idx, "FILE PATH"] = new_rel
                df.at[idx, "FILE EXTENSION"] = ".MP4"

    return df, report


def _norm_filename(name: str) -> str:
    """Normalize filename for fuzzy comparison: lowercase, collapse spaces, drop copy suffix patterns."""
    s = name.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _best_match(name: str, candidates: list[str]) -> tuple[str | None, float]:
    """Find best fuzzy match for filename among candidates."""
    if not candidates:
        return None, 0.0
    target = _norm_filename(name)
    best = None
    best_score = 0.0
    for c in candidates:
        filename = Path(c).name
        norm_file_name = _norm_filename(filename)
        score = difflib.SequenceMatcher(None, target, norm_file_name).ratio()
        if score > best_score or filename == target:
            best_score = score
            best = c
    return best, best_score


def _index_existing_files(input_dir: Path) -> dict[str, Any]:
    """Build indexes of existing media files under input_dir/sites.

    Returns a dict with:
    - all_paths: list of posix relative paths
    - by_site: site_name -> list of rel paths
    - by_name: filename(lower) -> list of rel paths
    """
    video_exts = {".mp4", ".mov", ".avi", ".mkv"}
    base = input_dir
    all_files: list[str] = []
    by_site: dict[str, list[str]] = {}
    by_name: dict[str, list[str]] = {}
    sites_root = base / "sites"
    if not sites_root.exists():
        return {"all_paths": all_files, "by_site": by_site, "by_name": by_name}
    for p in sites_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in video_exts:
            continue
        rel = p.relative_to(base).as_posix()
        all_files.append(rel)
        parts = Path(rel).parts
        site = parts[1] if len(parts) >= 2 else ""
        by_site.setdefault(site, []).append(rel)
        by_name.setdefault(Path(rel).name.lower(), []).append(rel)
    return {"all_paths": all_files, "by_site": by_site, "by_name": by_name}


def auto_match_missing_paths(
    df: pd.DataFrame,
    input_dir: Path,
    accept_threshold: float = 0.95,
    suggest_threshold: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Automatically find spelling mistakes in file paths and suggest corrections.

    For rows whose FILE PATH does not exist on disk, try to locate the correct file by fuzzy-matching the filename
    within the same site first, then globally.
    - If best score >= accept_threshold: update FILE PATH in-place.
    - Else if best score >= suggest_threshold: add to 'suggestions' (not applied).
    """
    if "FILE PATH" not in df.columns:
        return df, {}

    idx = _index_existing_files(input_dir)
    by_site: dict[str, list[str]] = idx["by_site"]
    by_name: dict[str, list[str]] = idx["by_name"]
    all_paths: list[str] = idx["all_paths"]

    def _exists(rel: str) -> bool:
        try:
            return (input_dir / rel).exists()
        except Exception:
            return False

    missing_mask = df["FILE PATH"].apply(lambda p: isinstance(p, str) and p.strip() != "" and not _exists(p))
    missing = df.loc[missing_mask, "FILE PATH"].tolist()

    applied = []
    suggestions = []
    unresolved = []

    for rel in missing:
        try:
            parts = Path(rel).parts
            site = parts[1] if len(parts) >= 2 and parts[0] == "sites" else ""
            filename = Path(rel).name

            # Try exact name match first
            if filename.lower() in by_name:
                candidates_exact = by_name[filename.lower()]
                if len(candidates_exact) == 1:
                    best: str | None = candidates_exact[0]
                    best_score = 1.0
                else:
                    best, best_score = _best_match(filename, candidates_exact)
            # Then try site-specific fuzzy match
            elif site and site in by_site:
                best, best_score = _best_match(filename, by_site[site])
            # Finally try global fuzzy match
            else:
                best, best_score = _best_match(filename, all_paths)

            if best and best_score >= accept_threshold:
                applied.append({"from": rel, "_to_": best, "score": best_score})
            elif best and best_score >= suggest_threshold:
                suggestions.append({"from": rel, "_to_": best, "score": best_score})
            else:
                unresolved.append(rel)
        except Exception:
            unresolved.append(rel)

    # Apply accepted matches
    if applied:
        m = {a["from"]: a["_to_"] for a in applied}
        for idx_row, row in df.loc[missing_mask].iterrows():
            old_path = row["FILE PATH"]
            if old_path in m:
                df.at[idx_row, "FILE PATH"] = m[old_path]

    report = {
        "checked_missing": len(missing),
        "applied": sorted(applied, key=lambda x: (x["score"], x["from"]), reverse=True),
        "num_applied": len(applied),
        "suggestions": sorted(suggestions, key=lambda x: (x["score"], x["from"]), reverse=True),
        "num_suggestions": len(suggestions),
        "unresolved": unresolved,
        "num_unresolved": len(unresolved),
        "accept_threshold": accept_threshold,
        "suggest_threshold": suggest_threshold,
    }
    return df, report


def _get_file_type(file_name: str) -> str:
    """Determine file type from extension."""
    ext = Path(file_name).suffix.lower()
    video_exts = {".mp4", ".mov", ".avi"}
    image_exts = {".jpg", ".jpeg", ".png"}
    if ext in video_exts:
        return "VIDEO"
    elif ext in image_exts:
        return "IMAGE"
    else:
        return "UNKNOWN"


def _get_file_path(row: pd.Series, file_type: str, input_dir: Path) -> str | None:
    """Construct expected file path from row metadata."""
    site = row["CAMERA TRAP SITE"]
    if pd.isna(site) or not str(site).strip():
        return None
    site_path: Path = Path("sites") / str(site).strip()
    if file_type == "VIDEO":
        cam = row["CAM"]
        if pd.isna(cam):
            return None
        dir_path: Path | None = site_path / ("CAM " + str(cam))
    else:  # IMAGE or unknown
        photos_dir_names = ["ID PHOTOS", "PHOTOS ID"]
        dir_path = None
        for pdn in photos_dir_names:
            candidate = input_dir / site_path / pdn
            if candidate.exists():
                dir_path = site_path / pdn
                break
        else:
            # If none exist, default to first option
            dir_path = site_path / photos_dir_names[0]

    orig_filename = row["ORIGINAL FILE NAME"]
    if pd.isna(orig_filename):
        return None
    file_path = (dir_path / str(orig_filename)).as_posix() if dir_path is not None else None
    return file_path


def add_file_metadata(df: pd.DataFrame, input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add FILE PATH, FILE TYPE and FILE EXTENSION columns based on the file name extension."""
    file_paths = []
    file_type_list = []
    file_extension_list = []

    for _, row in df.iterrows():
        file_name = str(row["ORIGINAL FILE NAME"]).strip()
        ext = Path(file_name).suffix.lower()
        file_type = _get_file_type(file_name)
        file_path = _get_file_path(row, file_type, input_dir)

        file_extension_list.append(ext)
        file_type_list.append(file_type)
        file_paths.append(file_path)

    df["RAW DATA PATH"] = input_dir.as_posix()
    df["FILE PATH"] = file_paths
    df["FILE TYPE"] = file_type_list
    df["FILE EXTENSION"] = file_extension_list

    df["FILE NAME"] = df["ORIGINAL FILE NAME"]
    df["ORIGINAL CAM"] = df["CAM"]
    df["ORIGINAL SITE"] = df["CAMERA TRAP SITE"]

    report = {
        "num_files_processed": len(df),
        "file_types_found": df["FILE TYPE"].value_counts().to_dict(),
        "file_extensions_found": df["FILE EXTENSION"].value_counts().to_dict(),
    }

    logger.info("Added file metadata: %s", report)
    return df, report


def check_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check for duplicate entries in (JAGUAR ID, CAMERA TRAP SITE, CAM, ORIGINAL FILE NAME)."""
    duplicate_mask = df.duplicated(subset=["JAGUAR ID", "CAMERA TRAP SITE", "CAM", "ORIGINAL FILE NAME"], keep="first")
    num_duplicates = duplicate_mask.sum()
    if num_duplicates > 0:
        logger.info("Found %d duplicate entries based on (JAGUAR ID, CAMERA TRAP SITE, CAM, ORIGINAL FILE NAME)", num_duplicates)
        logger.info("Duplicate entries:\n%s", df[duplicate_mask])
    report = {
        "num_duplicates": int(num_duplicates),
        "duplicate_rows": df[duplicate_mask].to_dict(orient="records"),
    }

    filtered_df = df[~duplicate_mask].reset_index(drop=True)
    return filtered_df, report


def split_multifile_sightings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split rows with multiple file names in 'Files Name' column into separate rows.

    Assumes that multiple file names are separated by commas or ' and '.
    Assigns a common SIGHTING ID to the split rows.
    """
    new_rows = []
    sighting_id = 1
    num_rows_with_multifile_sightings = 0
    for _, row in df.iterrows():
        file_names = str(row["Files Name"]).replace(" and ", ",").split(",")
        file_names = [fn.strip() for fn in file_names if fn.strip()]

        if len(file_names) > 1:
            num_rows_with_multifile_sightings += 1

        for fn in file_names:
            new_row = row.copy()
            new_row["Files Name"] = fn
            new_row["ORIGINAL FILE NAME"] = fn
            new_row["SIGHTING ID"] = f"S{sighting_id}"
            new_rows.append(new_row)
        sighting_id += 1

    new_df = pd.DataFrame(new_rows).reset_index(drop=True)
    report = {
        "num_rows_with_multifile_sightings": num_rows_with_multifile_sightings,
        "num_multifile_sightings": len(new_df) - len(df),
        "total_rows_after_split": len(new_df),
    }

    logger.info("Split %d rows with multifile sightings into %d total rows", num_rows_with_multifile_sightings, len(new_df))
    return new_df, report


def clean_date_formats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize DATE column to ISO format YYYY-MM-DD where possible.

    Insert DATETIME column combining DATE and TIME if both exist.
    """
    report: dict[str, Any] = {"unparsed_dates": []}
    if "DATE" not in df.columns:
        logger.info("No DATE column found; skipping date cleaning")
        report["dates_parsed"] = 0
        report["total_rows"] = len(df)
        return df, report

    # Try to parse DATE column with pandas; coerce errors to NaT
    parsed = df["DATE"].apply(lambda x: pd.to_datetime(str(x), errors="coerce"))
    num_parsed = int(parsed.notna().sum())

    report["unparsed_dates"] = df.loc[parsed.isna(), "DATE"].tolist()

    # Replace with ISO formatted dates where parsing succeeded
    df["DATE"] = parsed.dt.strftime("%Y-%m-%d")
    df["DATE"] = df["DATE"].where(df["DATE"].notna(), "")

    # If TIME column exists, create DATETIME column
    if "TIME" in df.columns:
        parsed_time = df["TIME"].apply(lambda x: pd.to_datetime(str(x), errors="coerce", format="%H:%M:%S"))
        df["DATETIME"] = pd.NaT
        valid_datetime_mask = parsed.notna() & parsed_time.notna()
        df.loc[valid_datetime_mask, "DATETIME"] = (
            parsed.loc[valid_datetime_mask].dt.strftime("%Y-%m-%d") + " " + parsed_time.loc[valid_datetime_mask].dt.strftime("%H:%M:%S")
        )  # noqa: E501

    report["dates_parsed"] = num_parsed
    report["total_rows"] = len(df)
    logger.info("Parsed %d/%d DATE entries", num_parsed, len(df))
    return df, report


def fill_missing_jaguar_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing JAGUAR ID entries with new unique IDs starting from U1, U2, ..."""
    missing_id_mask = df["JAGUAR ID"].isnull() | (df["JAGUAR ID"].astype(str).str.lower() == "?")
    num_missing = missing_id_mask.sum()
    logger.info("Filling %d missing JAGUAR IDs", num_missing)
    max_id_num = 0
    if num_missing > 0:
        # Find max existing U number
        for jid in df.loc[~missing_id_mask, "JAGUAR ID"].astype(str):
            match = re.match(r"^U(\d+).*$", jid, re.IGNORECASE)
            if match:
                try:
                    num = int(match.group(1))
                    if num > max_id_num:
                        max_id_num = num
                except ValueError:
                    continue

        # Assign new IDs
        for idx_row in df[missing_id_mask].index:
            max_id_num += 1
            df.at[idx_row, "JAGUAR ID"] = f"U{max_id_num}"

    report = {
        "starting_unknown_id": "U" + str(max_id_num + 1) if num_missing > 0 else "U1",
        "num_filled": int(num_missing),
    }

    logger.info("Assigned %d new JAGUAR IDs", report["num_filled"])
    return df, report


def check_required_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check for required columns and clean up the dataframe."""
    report = {}

    required_columns = [
        "CAMERA TRAP SITE",
        "JAGUAR ID",
        "CAM",
        "Files Name",
    ]
    optional_columns = [
        "LATITUDE",
        "LONGITUDE",
        "CAMERA ID",
        "LOCATION",
        "CAMERA MODEL",
        "DATE",
        "TIME",
        "TEMP C",
        "NOTES/ ERRORS",
    ]

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    optional_columns_missing = [col for col in optional_columns if col not in df.columns]
    if optional_columns_missing:
        logger.info("Missing optional columns: %s", optional_columns_missing)
    else:
        logger.info("All optional columns present")

    # Standardize all text columns
    for col in required_columns + optional_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)

    # Rename NOTES/ ERRORS to NOTES
    if "NOTES/ ERRORS" in df.columns:
        df["NOTES"] = df["NOTES/ ERRORS"]
        df = df.drop(columns=["NOTES/ ERRORS"])

    # Remove all other columns not in required or optional
    allowed_columns = set(required_columns + optional_columns + ["NOTES"])
    extra_columns = [col for col in df.columns if col not in allowed_columns]
    if extra_columns:
        df = df.drop(columns=extra_columns)

    # Filter out rows with all required fields empty
    all_empty_mask = df[required_columns].isnull().all(axis=1)
    num_all_empty = all_empty_mask.sum()
    if num_all_empty > 0:
        df = df[~all_empty_mask].reset_index(drop=True)

    report["num_rows_filtered_all_empty_required_fields"] = int(num_all_empty)
    return df, report


def load_df(input_dir: Path, input_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the input CSV file."""
    csv_path = input_csv if input_csv.exists() else input_dir / input_csv

    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    except Exception as e:
        logger.error("Failed to load CSV: %s", e)
        raise

    columns = df.columns.tolist()
    report = {
        "row_count": len(df),
        "columns": columns,
    }
    logger.info("Loaded %d rows with columns: %s", report["row_count"], ", ".join(columns))

    df, result = check_required_columns(df)
    report.update(result)

    return df, report


def _run_cleaning_pipeline(
    input_dir: Path,
    input_csv: Path,
    *,
    auto_match_missing: bool,
    match_threshold: float,
    suggest_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full CSV cleaning pipeline."""
    report: dict[str, Any] = {}

    df, report["input"] = load_df(input_dir, input_csv)
    df, report["filled_jaguar_ids"] = fill_missing_jaguar_ids(df)
    df, report["cleaned_dates"] = clean_date_formats(df)
    df, report["split_multifile_sightings"] = split_multifile_sightings(df)
    df, report["checked_duplicates"] = check_duplicates(df)
    df, report["added_file_metadata"] = add_file_metadata(df, input_dir)
    if auto_match_missing:
        df, report["auto_matched_paths"] = auto_match_missing_paths(
            df, input_dir, accept_threshold=match_threshold, suggest_threshold=suggest_threshold
        )
    df, report["filtered_invalid_paths"] = filter_rows_with_invalid_paths(df, input_dir)
    # Convert any AVI files to MP4 and update paths before ingest
    df, report["converted_avi_to_mp4"] = convert_avi_entries(df, input_dir)
    df, report["videos_without_labels"] = find_videos_without_labels(df, input_dir)

    return df, report


def ingest_csv_labels(
    *,
    input_dir: Path,
    input_csv: Path = Path("labels.csv"),
    dataset_name: str = JID_MASTER_DATASET,
    output_cleaned_csv: Path | None = None,
    generate_report: bool = False,
    auto_match_missing: bool = True,
    match_threshold: float = 0.95,
    suggest_threshold: float = 0.80,
) -> fo.Dataset:
    """Cleans labels and ingests into FiftyOne grouped dataset.

    This function follows the same cleaning pipeline.
    Creates a grouped dataset with 'image' and 'video' slices. Each CSV row gets its own group.

    Returns:
        fo.Dataset: The grouped dataset containing image and video slices.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    csv_path = _as_path(input_dir, Path(input_csv))
    if not csv_path.exists():
        raise FileNotFoundError(f"labels CSV not found: {csv_path}")

    logger.info("Cleaning labels using CSV cleaning pipeline: %s", csv_path)
    cleaned_df, report = _run_cleaning_pipeline(
        input_dir,
        Path(input_csv),
        auto_match_missing=auto_match_missing,
        match_threshold=match_threshold,
        suggest_threshold=suggest_threshold,
    )

    if output_cleaned_csv is not None:
        output_cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_cleaned_csv, index=False)
        logger.info("Writing cleaned labels to %s", output_cleaned_csv)
        if generate_report:
            write_report(report, output_cleaned_csv.parent / "cleaning_report.json")

    # Create or load grouped dataset with image and video slices
    dataset = get_or_create_dataset(dataset_name)

    # Set up group field if not already configured
    if dataset.group_field is None:
        dataset.add_group_field(GROUP_FIELD_NAME, default=DEFAULT_GROUP_SLICE)
        logger.info("Configured grouped dataset with 'image' and 'video' slices")

    # Get existing filepaths from all slices to avoid duplicates
    if len(dataset) > 0:
        # For grouped datasets, we need to get filepaths from all slices
        all_filepaths = set()
        for slice_name in [DEFAULT_GROUP_SLICE, VIDEO_GROUP_SLICE]:
            try:
                slice_view = dataset.select_group_slices(slice_name)
                all_filepaths.update(slice_view.values("filepath"))
            except:  # noqa: E722
                pass
        existing_filepaths = all_filepaths
    else:
        existing_filepaths = set()

    new_samples: list[fo.Sample] = []
    new_image_count = 0
    new_video_count = 0

    for _, row in cleaned_df.iterrows():
        rel_path = row.get("FILE PATH")
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue

        abs_path = (input_dir / rel_path).resolve()
        if not abs_path.exists():
            continue

        if str(abs_path) in existing_filepaths:
            continue

        file_type = str(row.get("FILE TYPE") or "UNKNOWN").upper()
        slice_name = VIDEO_GROUP_SLICE if file_type == "VIDEO" else DEFAULT_GROUP_SLICE

        # Each CSV row gets its own group ID (UUID-based to ensure uniqueness)
        group_id = str(uuid.uuid4())

        sample = fo.Sample(filepath=str(abs_path), group=fo.Group().element(group_id))
        sample.group.name = slice_name

        sample["source_type"] = "csv"
        sample["source"] = "csv_17_11_2025"
        sample["csv_source"] = str(csv_path)

        # Canonical downstream fields
        if pd.notna(row.get("JAGUAR ID")):
            sample["jaguar_id"] = str(row.get("JAGUAR ID"))
            sample["ground_truth"] = fo.Classification(label=str(row.get("JAGUAR ID")))
        if pd.notna(row.get("CAMERA TRAP SITE")):
            sample["site"] = str(row.get("CAMERA TRAP SITE"))
        if pd.notna(row.get("CAM")):
            sample["cam"] = str(row.get("CAM"))
        if pd.notna(row.get("SIGHTING ID")):
            sample["sighting_id"] = str(row.get("SIGHTING ID"))
        if pd.notna(row.get("DATE")):
            sample["date"] = str(row.get("DATE"))
        if pd.notna(row.get("TIME")):
            sample["time"] = str(row.get("TIME"))
        if pd.notna(row.get("DATETIME")) and str(row.get("DATETIME")).strip() != "":
            sample["datetime"] = str(row.get("DATETIME"))
        if pd.notna(row.get("LOCATION")):
            sample["location"] = str(row.get("LOCATION"))
        if pd.notna(row.get("NOTES")):
            sample["notes"] = str(row.get("NOTES"))

        # Preserve useful legacy fields for traceability
        sample["raw_data_path"] = str(input_dir)
        sample["file_path"] = str(rel_path)
        sample["file_type"] = str(file_type)
        sample["file_extension"] = str(row.get("FILE EXTENSION")) if pd.notna(row.get("FILE EXTENSION")) else None
        sample["original_file_name"] = str(row.get("ORIGINAL FILE NAME")) if pd.notna(row.get("ORIGINAL FILE NAME")) else None

        new_samples.append(sample)
        existing_filepaths.add(str(abs_path))

        if slice_name == VIDEO_GROUP_SLICE:
            new_video_count += 1
        else:
            new_image_count += 1

    if new_samples:
        dataset.add_samples(new_samples)
        dataset.save()
        logger.info("Added %d samples to grouped dataset (%d images, %d videos)", len(new_samples), new_image_count, new_video_count)
    else:
        logger.info("No new samples to add")

    # Get slice counts
    image_slice = dataset.select_group_slices(DEFAULT_GROUP_SLICE)
    video_slice = dataset.select_group_slices(VIDEO_GROUP_SLICE)

    # Log ingestion summary
    summary = {
        "dataset_name": dataset_name,
        "is_grouped": True,
        "slices": [DEFAULT_GROUP_SLICE, VIDEO_GROUP_SLICE],
        "csv_source": str(csv_path),
        "new_image_samples_added": new_image_count,
        "new_video_samples_added": new_video_count,
        "total_image_samples_in_dataset": len(image_slice),
        "total_video_samples_in_dataset": len(video_slice),
        "total_samples_in_dataset": len(image_slice) + len(video_slice),  # Sum both slices for grouped datasets
        "cleaned_rows_total": len(cleaned_df),
    }
    logger.info("CSV Ingestion Summary: %s", json.dumps(summary, indent=2))

    return dataset


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="CSV Ingestion Loader")
    parser.add_argument("--dataset", default=JID_MASTER_DATASET, type=str, help="Name of the FiftyOne dataset.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Path to input directory containing raw data.")
    parser.add_argument("--input-csv", type=Path, default=Path("labels.csv"), help="Relative path to labels CSV.")
    parser.add_argument("--output-cleaned-csv", type=Path, help="Path to write cleaned CSV output.")
    parser.add_argument("--generate-report", action="store_true", help="Generate JSON report.")
    parser.add_argument("--auto-match-missing", action="store_true", default=True, help="Auto-match missing file paths.")
    parser.add_argument("--match-threshold", type=float, default=0.95, help="Threshold for auto-matching.")
    parser.add_argument("--suggest-threshold", type=float, default=0.80, help="Threshold for suggesting matches.")
    parser.add_argument("--summary-location", type=str, help="Location to save summary report.")
    parser.add_argument("--dry-run", action="store_true", help="If set, no changes will be made.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    args = parser.parse_args()

    run_processing(
        dataset_name=args.dataset,
        input_dir=args.input_dir,
        input_csv=args.input_csv,
        output_cleaned_csv=args.output_cleaned_csv,
        generate_report=args.generate_report,
        auto_match_missing=args.auto_match_missing,
        match_threshold=args.match_threshold,
        suggest_threshold=args.suggest_threshold,
        summary_location=args.summary_location,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
