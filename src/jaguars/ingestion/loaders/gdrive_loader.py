"""Download private Google Drive assets with a service account.

Supported:
- Recursive folder download from Google Drive
- Google Sheets export to CSV

Examples:
    python -m jaguars.ingestion.loaders.gdrive_loader \
        --service-account-file gdrive-download-hpc-289de9a5ebe0.json \
        --folder-id "<FOLDER_ID>" \
        --output-dir data/raw/25_02_2026

    python -m jaguars.ingestion.loaders.gdrive_loader \
        --service-account-file gdrive-download-hpc-289de9a5ebe0.json \
        --spreadsheet-url "https://docs.google.com/spreadsheets/d/<SHEET_ID>" \
        --output-dir data/raw/25_02_2026
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

from jaguars.common.logging_utils import setup_logger

logger = setup_logger("ingestion.loaders.gdrive_loader")

_FOLDER_PATTERN = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
_SHEET_PATTERN = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")
_FOLDER_MIME = "application/vnd.google-apps.folder"


def _extract_folder_id(folder_url: str) -> str:
    match = _FOLDER_PATTERN.search(folder_url)
    if match is None:
        raise ValueError(f"Could not parse Google Drive folder id from URL: {folder_url}")
    return match.group(1)


def _extract_spreadsheet_id(spreadsheet_url: str) -> str:
    match = _SHEET_PATTERN.search(spreadsheet_url)
    if match is None:
        raise ValueError(f"Could not parse spreadsheet id from URL: {spreadsheet_url}")
    return match.group(1)


def _resolve_folder_id(*, folder_url: str | None, folder_id: str | None) -> str:
    if folder_id:
        return folder_id
    if folder_url:
        return _extract_folder_id(folder_url)
    raise ValueError("Provide either --folder-url or --folder-id")


def _resolve_spreadsheet_id(*, spreadsheet_url: str | None, spreadsheet_id: str | None) -> str:
    if spreadsheet_id:
        return spreadsheet_id
    if spreadsheet_url:
        return _extract_spreadsheet_id(spreadsheet_url)
    raise ValueError("Provide either --spreadsheet-url or --spreadsheet-id")


def _build_drive_service(service_account_file: Path):
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    if not service_account_file.exists():
        raise FileNotFoundError(f"Service account file not found: {service_account_file}")

    creds = service_account.Credentials.from_service_account_file(service_account_file.as_posix(), scopes=scopes)
    return build("drive", "v3", credentials=creds)


def _list_children(drive_service, folder_id: str) -> list[dict[str, str]]:
    query = f"'{folder_id}' in parents and trashed = false"
    fields = "nextPageToken, files(id, name, mimeType)"
    page_token = None
    items: list[dict[str, str]] = []

    while True:
        response = (
            drive_service.files()
            .list(q=query, fields=fields, pageToken=page_token, supportsAllDrives=True, includeItemsFromAllDrives=True)
            .execute()
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return items


def _download_file(drive_service, file_id: str, output_path: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)

    with open(output_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _download_folder_recursive(drive_service, folder_id: str, output_dir: Path, downloaded: list[str]) -> None:
    for item in _list_children(drive_service, folder_id):
        item_id = item["id"]
        item_name = item["name"]
        item_mime = item.get("mimeType", "")
        target_path = output_dir / item_name

        if item_mime == _FOLDER_MIME:
            target_path.mkdir(parents=True, exist_ok=True)
            _download_folder_recursive(drive_service, item_id, target_path, downloaded)
            continue

        if item_mime.startswith("application/vnd.google-apps"):
            logger.warning("Skipping Google-native file type (%s): %s", item_mime, item_name)
            continue

        logger.info("Downloading file: %s", target_path)
        _download_file(drive_service, item_id, target_path)
        downloaded.append(target_path.as_posix())


def download_gdrive_folder(
    *,
    service_account_file: Path,
    folder_url: str | None = None,
    folder_id: str | None = None,
    output_dir: Path = Path("data/raw"),
) -> list[str]:
    """Download a private Google Drive folder recursively using service account auth."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_folder_id = _resolve_folder_id(folder_url=folder_url, folder_id=folder_id)

    logger.info("Downloading private Google Drive folder id: %s", resolved_folder_id)
    logger.info("Output directory: %s", output_dir)
    logger.info("Using service account file: %s", service_account_file)

    drive_service = _build_drive_service(service_account_file)
    downloaded_files: list[str] = []
    _download_folder_recursive(drive_service, resolved_folder_id, output_dir, downloaded_files)

    if not downloaded_files:
        raise RuntimeError("No files were downloaded from folder. Verify service-account access and folder contents.")

    logger.info("Downloaded %d files from folder", len(downloaded_files))
    return downloaded_files


def _default_spreadsheet_output(output_dir: Path) -> Path:
    labels_path = output_dir / "labels.csv"
    if labels_path.exists():
        return output_dir / "labels_from_spreadsheet.csv"
    return output_dir / "spreadsheet.csv"


def download_spreadsheet_csv(
    *,
    service_account_file: Path,
    spreadsheet_url: str | None = None,
    spreadsheet_id: str | None = None,
    output_dir: Path = Path("data/raw"),
    output_file: Path | None = None,
) -> Path:
    """Export a Google Spreadsheet to CSV using service account auth.

    The CSV is written into output_dir by default, i.e. next to labels in that folder.
    """
    spreadsheet_id_resolved = _resolve_spreadsheet_id(spreadsheet_url=spreadsheet_url, spreadsheet_id=spreadsheet_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_file or _default_spreadsheet_output(output_dir)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    logger.info("Exporting spreadsheet id: %s", spreadsheet_id_resolved)
    logger.info("Saving CSV to: %s", output_path)
    logger.info("Using service account file: %s", service_account_file)

    drive_service = _build_drive_service(service_account_file)
    request = drive_service.files().export_media(fileId=spreadsheet_id_resolved, mimeType="text/csv")

    from googleapiclient.http import MediaIoBaseDownload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    logger.info("Spreadsheet export finished: %s", output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download private Google Drive folders and sheets with a service account")
    parser.add_argument("--service-account-file", type=Path, required=True, help="Path to service account JSON key file")

    parser.add_argument("--folder-url", type=str, default=None, help="Google Drive folder URL")
    parser.add_argument("--folder-id", type=str, default=None, help="Google Drive folder id")

    parser.add_argument("--spreadsheet-url", type=str, default=None, help="Google Spreadsheet URL")
    parser.add_argument("--spreadsheet-id", type=str, default=None, help="Google Spreadsheet id")

    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Destination directory")
    parser.add_argument(
        "--spreadsheet-output-file",
        type=Path,
        default=None,
        help="Optional output CSV path for spreadsheet export; default is inside output-dir",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    did_anything = False

    if args.folder_url or args.folder_id:
        files = download_gdrive_folder(
            service_account_file=args.service_account_file,
            folder_url=args.folder_url,
            folder_id=args.folder_id,
            output_dir=args.output_dir,
        )
        logger.info("Folder download complete. First file: %s", files[0])
        did_anything = True

    if args.spreadsheet_url or args.spreadsheet_id:
        spreadsheet_path = download_spreadsheet_csv(
            service_account_file=args.service_account_file,
            spreadsheet_url=args.spreadsheet_url,
            spreadsheet_id=args.spreadsheet_id,
            output_dir=args.output_dir,
            output_file=args.spreadsheet_output_file,
        )
        logger.info("Spreadsheet download complete: %s", spreadsheet_path)
        did_anything = True

    if not did_anything:
        raise ValueError(
            "Nothing to do. Provide at least one of: (--folder-url | --folder-id) or (--spreadsheet-url | --spreadsheet-id)."
        )


if __name__ == "__main__":
    main()
