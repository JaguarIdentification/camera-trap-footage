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

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS


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
        for p in video_files:
            rel = str(p.relative_to(input_dir).as_posix())
            if _is_jaguar_candidate_in_new_tree(rel):
                all_video_files.add(rel)

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
    """Convert AVI files once and update dataframe paths/extensions to MP4.

    - Converts all AVI files found recursively in ``input_dir`` to ``.MP4``
    - Skips conversion when output ``.MP4`` already exists
    - Updates dataframe ``FILE PATH``/``FILE EXTENSION`` for AVI rows when MP4 is available

    Returns the possibly updated dataframe and a conversion report dict.
    """
    avi_files = [p.resolve() for p in input_dir.rglob("*.avi") if p.is_file()]
    avi_files += [p.resolve() for p in input_dir.rglob("*.AVI") if p.is_file()]
    unique_avi_files = list(dict.fromkeys(avi_files))

    if unique_avi_files:
        logger.info("Converting %d AVI files recursively under %s", len(unique_avi_files), input_dir)
        report = batch_convert_avi(unique_avi_files, delete_original=False, overwrite=False)
    else:
        report = {"total_files": 0, "converted_files": 0, "failed_conversions": 0, "skipped_files": 0}

    if "FILE PATH" not in df.columns:
        report["updated_dataframe_rows"] = 0
        return df, report

    # Identify AVI files by extension or path suffix
    def _is_avi_row(row: pd.Series) -> bool:
        ext = str(row.get("FILE EXTENSION", "")).lower()
        path = str(row.get("FILE PATH", ""))
        return ext == ".avi" or path.lower().endswith(".avi")

    def _find_existing_mp4_for_path(src: Path) -> Path | None:
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

    updated_rows = 0
    for idx, row in df[df.apply(_is_avi_row, axis=1)].iterrows():
        rel = row.get("FILE PATH")
        if not isinstance(rel, str) or not rel.strip():
            continue

        abs_path = (input_dir / rel).resolve()
        existing_mp4 = _find_existing_mp4_for_path(abs_path)
        if existing_mp4 is not None:
            df.at[idx, "FILE PATH"] = existing_mp4.relative_to(input_dir).as_posix()
            df.at[idx, "FILE EXTENSION"] = existing_mp4.suffix
            updated_rows += 1

    report["updated_dataframe_rows"] = updated_rows
    return df, report


def _norm_filename(name: str) -> str:
    """Normalize filename for fuzzy comparison: lowercase, collapse spaces, drop copy suffix patterns."""
    s = name.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_site_value(site_value: Any) -> str | None:
    if pd.isna(site_value):
        return None
    site_raw = _normalize_whitespace(str(site_value)).upper()
    match = re.search(r"SITE\s*0*(\d+)", site_raw)
    if match:
        return f"SITE {int(match.group(1))}"
    return site_raw if site_raw else None


def _normalize_cam_value(cam_value: Any) -> str | None:
    if pd.isna(cam_value):
        return None
    cam_raw = _normalize_whitespace(str(cam_value)).upper()
    compact = re.sub(r"[^A-Z0-9]", "", cam_raw)
    if compact in {"A", "CAMA"}:
        return "CAM A"
    if compact in {"B", "CAMB"}:
        return "CAM B"
    if compact.startswith("CAMA"):
        return "CAM A"
    if compact.startswith("CAMB"):
        return "CAM B"
    return cam_raw if cam_raw else None


def _extract_site_cam_from_path(rel: str) -> tuple[str | None, str | None]:
    parts = Path(rel).parts
    site: str | None = None
    cam: str | None = None
    for part in parts:
        if site is None:
            site = _normalize_site_value(part)
        if cam is None:
            candidate_cam = _normalize_cam_value(part)
            if candidate_cam in {"CAM A", "CAM B"}:
                cam = candidate_cam
    return site, cam


def _is_new_filtered_tree(parts: tuple[str, ...]) -> bool:
    return any("jaguars filtered" in p.lower() for p in parts)


def _is_meaningful_post_cam_dir(part: str) -> bool:
    token = _normalize_whitespace(part).lower()
    if token == "":
        return False
    if token == "100media":
        return False
    if token.isdigit():
        return False
    if token.startswith("conf_"):
        return False
    return True


def _is_jaguar_or_photo_dir(part: str) -> bool:
    token = _normalize_whitespace(part).lower()
    return "jaguar" in token or token in {"data photos", "id photos", "photos id"}


def _is_jaguar_candidate_in_new_tree(rel: str) -> bool:
    parts = Path(rel).parts
    if not _is_new_filtered_tree(parts):
        return True

    directory_parts = [p for p in parts[:-1] if _is_meaningful_post_cam_dir(p)]
    return any(_is_jaguar_or_photo_dir(p) for p in directory_parts)


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
    base = input_dir
    all_files: list[str] = []
    by_site: dict[str, list[str]] = {}
    by_site_cam: dict[tuple[str, str], list[str]] = {}
    by_name: dict[str, list[str]] = {}
    by_stem: dict[str, list[str]] = {}

    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in MEDIA_EXTENSIONS:
            continue

        rel = p.relative_to(base).as_posix()
        if not _is_jaguar_candidate_in_new_tree(rel):
            continue

        all_files.append(rel)
        site, cam = _extract_site_cam_from_path(rel)
        if site:
            by_site.setdefault(site, []).append(rel)
            if cam:
                by_site_cam.setdefault((site, cam), []).append(rel)
        by_name.setdefault(Path(rel).name.lower(), []).append(rel)
        by_stem.setdefault(_norm_filename(Path(rel).stem), []).append(rel)

    return {
        "all_paths": all_files,
        "by_site": by_site,
        "by_site_cam": by_site_cam,
        "by_name": by_name,
        "by_stem": by_stem,
    }


def _score_candidate_path(candidate_rel: str, row: pd.Series, file_type: str) -> tuple[int, int]:
    candidate_site, candidate_cam = _extract_site_cam_from_path(candidate_rel)
    row_site = _normalize_site_value(row.get("CAMERA TRAP SITE"))
    row_cam = _normalize_cam_value(row.get("CAM"))

    score = 0
    if row_site and candidate_site == row_site:
        score += 4
    if row_cam and candidate_cam == row_cam:
        score += 3

    lower = candidate_rel.lower()
    if "jaguar" in lower:
        score += 1
    if file_type == "IMAGE" and any(tag in lower for tag in ["data photos", "id photos", "photos id"]):
        score += 1

    # tie-breaker: shallower path is preferred
    depth_penalty = len(Path(candidate_rel).parts)
    return score, -depth_penalty


def _select_best_candidate_path(candidates: list[str], row: pd.Series, file_type: str) -> str | None:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda rel: _score_candidate_path(rel, row, file_type), reverse=True)
    return ranked[0]


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
    by_site_cam: dict[tuple[str, str], list[str]] = idx["by_site_cam"]
    by_name: dict[str, list[str]] = idx["by_name"]
    by_stem: dict[str, list[str]] = idx["by_stem"]
    all_paths: list[str] = idx["all_paths"]

    def _exists(rel: str) -> bool:
        try:
            return (input_dir / rel).exists()
        except Exception:
            return False

    missing_mask = df["FILE PATH"].apply(lambda p: isinstance(p, str) and p.strip() != "" and not _exists(p))
    applied = []
    suggestions = []
    unresolved = []

    for idx_row, row in df.loc[missing_mask].iterrows():
        rel = str(row.get("FILE PATH"))
        try:
            filename = Path(rel).name
            if filename.strip() == "":
                maybe_file = row.get("ORIGINAL FILE NAME") or row.get("Files Name")
                filename = str(maybe_file) if pd.notna(maybe_file) else ""

            site = _normalize_site_value(row.get("CAMERA TRAP SITE"))
            cam = _normalize_cam_value(row.get("CAM"))
            file_type = _get_file_type(filename)

            best: str | None = None
            best_score = 0.0

            # Prefer exact filename matches first, ranked by site/cam/path score
            exact_candidates = by_name.get(filename.lower(), [])
            if exact_candidates:
                exact_best = _select_best_candidate_path(exact_candidates, row, file_type)
                if exact_best is not None:
                    best = exact_best
                    best_score = 1.0

            # Then try exact stem matches (handles missing/wrong extensions)
            stem = _norm_filename(Path(filename).stem)
            if best is None and stem in by_stem:
                stem_best = _select_best_candidate_path(by_stem[stem], row, file_type)
                if stem_best is not None:
                    best = stem_best
                    stem_name = _norm_filename(Path(stem_best).stem)
                    best_score = 0.99 if stem_name == stem else 0.0

            candidate_sets: list[list[str]] = []
            if site and cam and (site, cam) in by_site_cam:
                candidate_sets.append(by_site_cam[(site, cam)])
            if site and site in by_site:
                candidate_sets.append(by_site[site])
            candidate_sets.append(all_paths)

            for candidates in candidate_sets:
                cand_best, cand_score = _best_match(filename, candidates)
                if cand_best is not None and cand_score > best_score:
                    best = cand_best
                    best_score = cand_score

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
        "checked_missing": int(missing_mask.sum()),
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
    if ext in VIDEO_EXTENSIONS:
        return "VIDEO"
    elif ext in IMAGE_EXTENSIONS:
        return "IMAGE"
    else:
        return "UNKNOWN"


def _get_file_path(row: pd.Series, file_type: str, input_dir: Path, file_index: dict[str, Any] | None = None) -> str | None:
    """Construct expected file path from row metadata."""
    orig_filename = row["ORIGINAL FILE NAME"]
    if pd.isna(orig_filename):
        return None

    normalized_file_name = _normalize_whitespace(str(orig_filename))
    if not normalized_file_name:
        return None

    if file_index is not None:
        by_name: dict[str, list[str]] = file_index.get("by_name", {})
        by_stem: dict[str, list[str]] = file_index.get("by_stem", {})

        exact_candidates = by_name.get(normalized_file_name.lower(), [])
        if exact_candidates:
            best = _select_best_candidate_path(exact_candidates, row, file_type)
            if best is not None:
                return best

        file_stem = _norm_filename(Path(normalized_file_name).stem)
        if file_stem and file_stem in by_stem:
            best = _select_best_candidate_path(by_stem[file_stem], row, file_type)
            if best is not None:
                return best

    site = _normalize_site_value(row.get("CAMERA TRAP SITE"))
    if not site:
        return None
    site_path: Path = Path("sites") / site
    if file_type == "VIDEO":
        cam = _normalize_cam_value(row.get("CAM"))
        if cam is None:
            # Keep a non-empty fallback to allow downstream auto-match
            dir_path = site_path
        else:
            dir_path = site_path / cam
    else:  # IMAGE or unknown
        photos_dir_names = ["ID PHOTOS", "PHOTOS ID", "DATA PHOTOS"]
        dir_path = None
        for pdn in photos_dir_names:
            candidate = input_dir / site_path / pdn
            if candidate.exists():
                dir_path = site_path / pdn
                break
        else:
            # If none exist, default to first option
            dir_path = site_path / photos_dir_names[0]

    file_path = (dir_path / normalized_file_name).as_posix() if dir_path is not None else None
    return file_path


def add_file_metadata(df: pd.DataFrame, input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add FILE PATH, FILE TYPE and FILE EXTENSION columns based on the file name extension."""
    file_index = _index_existing_files(input_dir)
    file_paths = []
    file_type_list = []
    file_extension_list = []

    for _, row in df.iterrows():
        file_name = str(row["ORIGINAL FILE NAME"]).strip()
        ext = Path(file_name).suffix.lower()
        file_type = _get_file_type(file_name)
        file_path = _get_file_path(row, file_type, input_dir, file_index=file_index)

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

    parse_attempts = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": None, "engine": "python", "encoding": "utf-8"},
    ]

    last_error: Exception | None = None
    df: pd.DataFrame | None = None
    for parse_kwargs in parse_attempts:
        try:
            candidate = pd.read_csv(csv_path, **parse_kwargs)
            if len(candidate.columns) > 1:
                df = candidate
                break
            df = candidate
        except Exception as e:
            last_error = e

    if df is None:
        logger.error("Failed to load CSV: %s", last_error)
        raise RuntimeError(f"Failed to load CSV at {csv_path}") from last_error

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
    df, report["converted_avi_to_mp4"] = convert_avi_entries(df, input_dir)
    if auto_match_missing:
        df, report["auto_matched_paths"] = auto_match_missing_paths(
            df, input_dir, accept_threshold=match_threshold, suggest_threshold=suggest_threshold
        )
    df, report["filtered_invalid_paths"] = filter_rows_with_invalid_paths(df, input_dir)
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

    input_csv_path = Path(input_csv)
    csv_path = _as_path(input_dir, input_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"labels CSV not found: {csv_path}")

    logger.info("Cleaning labels using CSV cleaning pipeline: %s", csv_path)
    cleaned_df, report = _run_cleaning_pipeline(
        input_dir,
        csv_path,
        auto_match_missing=auto_match_missing,
        match_threshold=match_threshold,
        suggest_threshold=suggest_threshold,
    )

    report_path = csv_path.parent
    if output_cleaned_csv is not None:
        report_path = output_cleaned_csv.parent
        output_cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_cleaned_csv, index=False)
        logger.info("Writing cleaned labels to %s", output_cleaned_csv)

    if generate_report:
        report["output_cleaned_csv"] = str(output_cleaned_csv)
        write_report(report, report_path / "cleaning_report.json")

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

        file_suffix = abs_path.suffix.lower()
        if file_suffix in VIDEO_EXTENSIONS:
            file_type = "VIDEO"
            slice_name = VIDEO_GROUP_SLICE
        elif file_suffix in IMAGE_EXTENSIONS:
            file_type = "IMAGE"
            slice_name = DEFAULT_GROUP_SLICE
        else:
            fallback_type = str(row.get("FILE TYPE") or "UNKNOWN").upper()
            if fallback_type == "VIDEO":
                file_type = "VIDEO"
                slice_name = VIDEO_GROUP_SLICE
            elif fallback_type == "IMAGE":
                file_type = "IMAGE"
                slice_name = DEFAULT_GROUP_SLICE
            else:
                logger.warning("Skipping unsupported media file: %s", abs_path)
                continue

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
        sample["file_type"] = file_type
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
