"""Clean and preprocess jaguar identification labels. The labels are a .csv file which contains the metadata per jaguar sighting.

This script processes the raw labels file, cleans the data and outputs a cleaned .csv file for further processing. This includes:
- Some JAGUAR ID columns are empty. These are assigned a new ID.
- Cleaning different DATE formats.
- Checking for duplicates in JAGUAR ID, SITE, CAM and Files Name.
- Most sightings map to a single video. Some jaguar sightings map to several images or videos around the same time.
  These are grouped together under a common SIGHTING ID. Each row should only have one file name (video or image).
- Add FILE PATH, FILE TYPE and FILE EXTENSION columns based on the file name extension.
- Filtering entries with file names that do not exist in the image data.
- Finding files that do not have an entry in the labels.

A report is generated, logged and optionally saved containing processing information about each step.

Run as a module:
    python -m src.jaguar_reidentification.data_preprocessing.clean_labels --input_dir=data/raw/ --input_csv=labels.csv --output_csv=data/processed/cleaned_labels.csv --generate_report
"""  # noqa: E501

import argparse
import difflib
import json
import logging
import pandas as pd
from pathlib import Path
import re

from src.jaguar_reidentification.utils.utils import json_safe


def write_report(report: dict, report_path: Path) -> None:
    logging.info("Writing processing report to %s", report_path)
    report = json_safe(report)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)


def write_df(df: pd.DataFrame, output_csv: Path) -> dict:
    logging.info("Writing cleaned labels to %s", output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    columns = df.columns.tolist()
    report = {
        "row_count": len(df),
        "columns": columns,
        "unique_jaguar_ids": df["JAGUAR ID"].nunique(),
        "file_types": df["FILE TYPE"].value_counts().to_dict(),
        "file_extensions": df["FILE EXTENSION"].value_counts().to_dict(),
    }
    logging.info("Saved %d cleaned rows with columns: %s", report["row_count"], ", ".join(columns))
    return report


def find_videos_without_labels(df: pd.DataFrame, input_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Find video files in input_dir that do not have a corresponding entry in df."""
    labeled_files = set(df["FILE PATH"].dropna().astype(str).tolist())
    all_video_files = set()

    for ext in [".MP4", ".mp4", ".mov", ".MOV", ".avi", ".AVI"]:
        video_files = list(input_dir.rglob(f"*{ext}"))
        all_video_files.update([str(p.relative_to(input_dir).as_posix()) for p in video_files])

    unlabeled_files = all_video_files - labeled_files
    logging.info("Found %d video files without labels", len(unlabeled_files))

    report = {
        "num_videos_without_labels": len(unlabeled_files),
        "videos_without_labels": sorted(list(unlabeled_files)),
    }

    return df, report


def filter_rows_with_invalid_paths(df: pd.DataFrame, input_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Filter out rows where the the FILE PATH is empty or does not exist on disk."""
    non_empty_path_mask = df["FILE PATH"].apply(lambda p: isinstance(p, str) and p.strip() != "")
    num_empty = (~non_empty_path_mask).sum()
    logging.info("Filtering out %d rows with empty FILE PATHs", num_empty)

    filtered_df = df[non_empty_path_mask].reset_index(drop=True)

    exists_mask = filtered_df["FILE PATH"].apply(lambda p: (input_dir / p).exists())
    num_not_exist = (~exists_mask).sum()
    logging.info("Filtering out %d rows where FILE PATH not found", num_not_exist)

    report = {
        "num_empty_file_paths": int(num_empty),
        "num_file_path_not_found": int(num_not_exist),
        "file_paths_not_found": sorted(filtered_df.loc[~exists_mask, "FILE PATH"].tolist()),
    }

    return filtered_df, report


def _norm_filename(name: str) -> str:
    # normalize filename for fuzzy compare: lowercase, collapse spaces, drop " copy" suffix patterns
    s = name.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _best_match(name: str, candidates: list[str]) -> tuple[str | None, float]:
    if not candidates:
        return None, 0.0
    target = _norm_filename(name)
    best = None
    best_score = 0.0
    for c in candidates:
        name = Path(c).name
        norm_file_name = _norm_filename(name)
        score = difflib.SequenceMatcher(None, target, norm_file_name).ratio()
        if score > best_score or name == target:
            best_score, best = score, c
    return best, best_score


def _index_existing_files(input_dir: Path) -> dict:
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
        rel = p.relative_to(base).as_posix()  # e.g., sites/SITE X/CAM A/DSCF1234.MP4
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
) -> tuple[pd.DataFrame, dict]:
    """Automatically find spelling mistakes in file paths and suggest corrections.
    
    For rows whose FILE PATH does not exist on disk, try to locate the correct file by fuzzy-matching the filename
    within the same site first, then globally.
    - If best score >= accept_threshold: update FILE PATH in-place.
    - Else if best score >= suggest_threshold: add to 'suggestions' (not applied).
    """
    if "FILE PATH" not in df.columns:
        return df, {"skipped": "no FILE PATH column"}

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
            # First check if it exists in the other directories (i.e. cameras) of the same site
            parts = Path(rel).parts
            filename = Path(rel).name
            site = parts[1] if len(parts) >= 2 and parts[0].lower() == "sites" else None

            # 1) Try within same site
            site_candidates = by_site.get(site, []) if site else []
            cand, score = _best_match(filename, site_candidates)
            
            # 2) find by exact filename match (case-insensitive) in other sites
            if not cand or score < accept_threshold:
                name_candidates = by_name.get(filename.lower(), [])
                name_cand, name_score = _best_match(filename, name_candidates) if name_candidates else (cand, score)
                if name_cand and name_score > score:
                    cand, score = name_cand, name_score
            
            # 3) Fallback: try fuzzy across all files
            if not cand:
                cand2, score2 = _best_match(filename, all_paths)
                if (cand2 and score2 > score):
                    cand, score = cand2, score2

            if cand and score >= accept_threshold:
                applied.append({"from": rel, "_to_": cand, "score": float(round(score, 4))})
            elif cand and score >= suggest_threshold:
                suggestions.append({"from": rel, "cand": cand, "score": float(round(score, 4))})
            else:
                unresolved.append(rel)
        except Exception:
            unresolved.append(rel)

    # Apply accepted matches
    if applied:
        m = {a["from"]: a["_to_"] for a in applied}
        df.loc[missing_mask, "FILE PATH"] = df.loc[missing_mask, "FILE PATH"].map(lambda x: m.get(x, x))

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
    ext = Path(file_name).suffix.lower()
    video_exts = {".mp4", ".mov", ".avi"}
    image_exts = {".jpg", ".jpeg", ".png"}
    if ext in video_exts:
        return "VIDEO"
    elif ext in image_exts:
        return "IMAGE"
    else:
        return "UNKNOWN"


def _get_file_path(row: pd.Series, file_type: str, input_dir: Path) -> str:
    site_path: Path = Path("sites") / row["CAMERA TRAP SITE"]
    if file_type == "VIDEO":
        dir_path: Path = site_path / ("CAM " + str(row["CAM"]))
    else:  # IMAGE or unknown
        photos_dir_names = ["ID PHOTOS", "PHOTOS ID"]
        for pdn in photos_dir_names:
            potential_path = input_dir / site_path / pdn
            if potential_path.exists():
                dir_path = site_path / pdn
                break
        else:
            # we assign a default path and check if the file exists in the next step
            dir_path = site_path / "DATA PHOTOS"

    file_path = (dir_path / row["ORIGINAL FILE NAME"]).as_posix() if dir_path is not None else None
    
    return file_path


def add_file_metadata(df: pd.DataFrame, input_dir: Path) -> tuple[pd.DataFrame, dict]:
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

    report = {
        "num_files_processed": len(df),
        "file_types_found": df["FILE TYPE"].value_counts().to_dict(),
        "file_extensions_found": df["FILE EXTENSION"].value_counts().to_dict(),
    }

    logging.info("Added file metadata: %s", report)
    return df, report


def check_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Check for duplicate entries in (JAGUAR ID, CAMERA TRAP SITE, CAM, ORIGINAL FILE NAME)."""
    duplicate_mask = df.duplicated(subset=["JAGUAR ID", "CAMERA TRAP SITE", "CAM", "ORIGINAL FILE NAME"], keep="first")
    num_duplicates = duplicate_mask.sum()
    if num_duplicates > 0:
        logging.info("Found %d duplicate entries based on (JAGUAR ID, CAMERA TRAP SITE, CAM, ORIGINAL FILE NAME)", num_duplicates)
        logging.info("Duplicate entries:\n%s", df[duplicate_mask])
    report = {
        "num_duplicates": int(num_duplicates),
        "duplicate_rows": df[duplicate_mask].to_dict(orient="records"),
    }

    filtered_df = df[~duplicate_mask].reset_index(drop=True)
    return filtered_df, report


def split_multifile_sightings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Split rows with multiple file names in 'Files Name' column into separate rows.

    Assumes that multiple file names are separated by commas or ' and '.
    Assigns a common SIGHTING ID to the split rows.
    """
    new_rows = []
    sighting_id = 1
    num_rows_with_multifile_sightings = 0
    for i, row in df.iterrows():
        file_names = str(row["Files Name"]).replace(" and ", ",").split(",")
        file_names = [fn.strip() for fn in file_names if fn.strip()]

        if len(file_names) > 1:
            logging.info("Splitting row %d with multiple files into %d rows: %s", i, len(file_names), row["Files Name"])
            num_rows_with_multifile_sightings += 1

        for fn in file_names:
            new_row = row.copy()
            new_row["ORIGINAL FILE NAME"] = fn
            new_row["SIGHTING ID"] = str(sighting_id).zfill(5)
            new_rows.append(new_row)
        sighting_id += 1

    new_df = pd.DataFrame(new_rows).reset_index(drop=True)
    report = {
        "num_rows_with_multifile_sightings": num_rows_with_multifile_sightings,
        "num_multifile_sightings": len(new_df) - len(df),
        "total_rows_after_split": len(new_df),
    }

    logging.info("Split %d rows with multifile sightings into %d total rows", num_rows_with_multifile_sightings, len(new_df))

    return new_df, report


def clean_date_formats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The inputted data may contain both "09/15/2024" and "15.09.2024" formats etc.

    Normalize DATE column to ISO format YYYY-MM-DD where possible; if no DATE column
    exists or parsing fails just return input and record counts.
    
    Insert DATETIME column combining DATE and TIME if both exist.
    """
    report: dict = {
        "unparsed_dates": [],
    }
    if "DATE" not in df.columns:
        logging.info("No DATE column found; skipping date cleaning")
        report["dates_parsed"] = 0
        report["total_rows"] = len(df)
        return df, report

    # Try to parse DATE column with pandas; coerce errors to NaT
    parsed = df["DATE"].apply(lambda x: pd.to_datetime(str(x), errors="coerce"))
    num_parsed = int(parsed.notna().sum())

    report["unparsed_dates"] = df.loc[parsed.isna(), "DATE"].tolist()

    # Replace with ISO formatted dates where parsing succeeded, leave others as empty strings
    df["DATE"] = parsed.dt.strftime("%Y-%m-%d")
    df["DATE"] = df["DATE"].where(df["DATE"].notna(), "")

    # If TIME column exists, create DATETIME column
    if "TIME" in df.columns:
        parsed_time = df["TIME"].apply(lambda x: pd.to_datetime(str(x), errors="coerce", format="%H:%M:%S"))
        df["DATETIME"] = pd.NaT
        valid_datetime_mask = parsed.notna() & parsed_time.notna()
        df.loc[valid_datetime_mask, "DATETIME"] = (
            parsed[valid_datetime_mask].dt.normalize() + 
            parsed_time[valid_datetime_mask].dt.time.apply(lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second))
        )
        df["DATETIME"] = df["DATETIME"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["DATETIME"] = df["DATETIME"].where(df["DATETIME"].notna(), "")

    report["dates_parsed"] = num_parsed
    report["total_rows"] = len(df)
    logging.info("Parsed %d/%d DATE entries", num_parsed, len(df))
    return df, report


def fill_missing_jaguar_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill missing JAGUAR ID entries with new unique IDs starting from U1, U2, ...

    Since some U IDs may already exist, we find the max existing U number and continue from there.

    Note: This assumes that if different rows have missing JAGUAR IDs, they also correspond to different jaguars.
    If the labeler sees an individual twice they should assign their own Unknown ID consistently instead of leaving it empty.
    """
    missing_id_mask = (
        df["JAGUAR ID"].isnull() | 
        (df["JAGUAR ID"].astype(str).str.lower() == "?")
    )
    num_missing = missing_id_mask.sum()
    logging.info("Filling %d missing JAGUAR IDs", num_missing)
    max_id_num = 0
    if num_missing > 0:
        # find max existing ID number starting with U
        for jag_id in df.loc[~missing_id_mask, "JAGUAR ID"].astype(str):
            if jag_id.startswith("U"):
                try:
                    num = int(jag_id[1:])
                    if num > max_id_num:
                        max_id_num = num
                except ValueError:
                    continue

        # Assign new IDs to missing entries
        new_ids = [f"U{max_id_num + i + 1}" for i in range(num_missing)]
        df.loc[missing_id_mask, "JAGUAR ID"] = new_ids

    report = {
        "starting_unknown_id": "U" + str(max_id_num + 1),
        "num_filled": num_missing,
    }

    logging.info("Assigned %d new JAGUAR IDs starting from %s", report["num_filled"], report["starting_unknown_id"])

    return df, report


def check_required_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}

    required_columns = [
        "CAMERA TRAP SITE",
        "JAGUAR ID",
        "CAM",
        "JAGUAR ID",
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
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing_columns)}")

    optional_columns_missing = [col for col in optional_columns if col not in df.columns]
    if optional_columns_missing:
        logging.warning("Input CSV is missing optional columns: %s", ", ".join(optional_columns_missing))
        report["missing_optional_columns"] = optional_columns_missing
    else:
        report["missing_optional_columns"] = []

    # Standardize all text columns
    for col in required_columns + optional_columns:
        if col in df.columns:
            s = df[col]
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                s = s.apply(lambda v: v.strip() if isinstance(v, str) else v)
                s = s.replace("", pd.NA)
                df[col] = s

    # Rename NOTES/ ERRORS to NOTES
    if "NOTES/ ERRORS" in df.columns:
        df = df.rename(columns={"NOTES/ ERRORS": "NOTES"})
        df["NOTES"] = df["NOTES"].fillna("")

    # Remove all other columns not in required or optional
    allowed_columns = set(required_columns + optional_columns)
    extra_columns = [col for col in df.columns if col not in allowed_columns]
    if extra_columns:
        logging.info("Removing extra columns not in required/optional list: %s", ", ".join(extra_columns))
        df = df.drop(columns=extra_columns)

    # Filter out rows with all required fields empty
    all_empty_mask = df[required_columns].isnull().all(axis=1)
    num_all_empty = all_empty_mask.sum()
    if num_all_empty > 0:
        logging.info("Filtering out %d rows with all required fields empty", num_all_empty)
        df = df[~all_empty_mask].reset_index(drop=True)

    report["num_rows_filtered_all_empty_required_fields"] = num_all_empty

    return df, report


def load_df(input_dir: Path, input_csv: Path) -> tuple[pd.DataFrame, dict]:
    if input_csv.exists():
        logging.info("Loading raw labels from %s", input_csv)
        input_path = input_csv
    else:
        input_path = input_dir / input_csv
        logging.info("Loading raw labels from %s", input_path)

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error("Retrying after failed to load CSV file: %s", e)
        df = pd.read_csv(input_path, sep=";")
        logging.info("Success: Loaded CSV file with ';' separator")

    columns = df.columns.tolist()
    report = {
        "row_count": len(df),
        "columns": columns,
    }
    logging.info("Loaded %d rows with columns: %s", report["row_count"], ", ".join(columns))

    df, result = check_required_columns(df)
    report.update(result)

    return df, report


def run(
    input_dir: Path,
    output_csv: Path,
    input_csv: Path = Path("labels.csv"),
    generate_report: bool = False,
    auto_match_missing: bool = True,
    match_threshold: float = 0.95,
    suggest_threshold: float = 0.80,
):
    report = {}
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
    df, report["videos_without_labels"] = find_videos_without_labels(df, input_dir)
    report["output"] = write_df(df, output_csv)
    if generate_report:
        write_report(report, output_csv.parent / "cleaning_report.json")


def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess jaguar identification labels.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the raw labels CSV file.",
    )
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
        help="Path to save the cleaned labels CSV file.",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Should generate a processing report.",
    )
    parser.add_argument(
        "--no_auto_match",
        action="store_true",
        help="Disable fuzzy auto-matching of missing FILE PATHs.",
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.95,
        help="Acceptance threshold for auto-matching (0..1).",
    )
    parser.add_argument(
        "--suggest_threshold",
        type=float,
        default=0.80,
        help="Suggestion threshold (0..1). Below this no suggestion is recorded.",
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
        Path(args.input_dir),
        Path(args.output_csv),
        input_csv=Path(args.input_csv),
        generate_report=args.generate_report,
        auto_match_missing=not args.no_auto_match,
        match_threshold=float(args.match_threshold),
        suggest_threshold=float(args.suggest_threshold),
    )


if __name__ == "__main__":
    raise SystemExit(main())
