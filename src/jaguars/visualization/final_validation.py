import hashlib
import math
import os
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image, UnidentifiedImageError

from jaguars.visualization.final_lineage import Enrichment
from jaguars.visualization.final_records import FrozenAnnotation, TerminalRecord

HASH_CHUNK_SIZE = 1024 * 1024
APPROVED_STORAGE_ROOT = Path("/Volumes/CameraTrapPython/fiftyone")
DuplicateKey = TypeVar("DuplicateKey", bound=Hashable)


class IntegrityError(ValueError):
    """Raised when final snapshot source integrity validation fails."""


class StorageSafetyError(ValueError):
    """Raised when required mounts or generated-state paths are unsafe."""


@dataclass(frozen=True)
class MediaIntegrity:
    sha256: str
    size_bytes: int
    width: int
    height: int


@dataclass(frozen=True)
class ValidatedRecord:
    terminal: TerminalRecord
    enrichment: Enrichment
    integrity: MediaIntegrity


def _detection_entries(
    annotation: FrozenAnnotation,
    field: str,
    context: str,
    errors: list[str],
) -> tuple[tuple[int, Mapping[str, Any]], ...]:
    detections = annotation.get("detections")
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)) or not detections:
        errors.append(f"{context}: {field}.detections must be a nonempty sequence")
        return ()

    entries: list[tuple[int, Mapping[str, Any]]] = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            errors.append(f"{context}: {field}.detections[{index}] must be an object")
        else:
            entries.append((index, detection))
    return tuple(entries)


def _validate_bbox(value: object) -> str | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return "must be a four-value sequence [x, y, width, height]"
    if any(isinstance(component, bool) or not isinstance(component, (int, float)) or not math.isfinite(component) for component in value):
        return "must contain four finite numeric values"

    x, y, width, height = value
    if x < 0 or y < 0 or width < 0 or height < 0:
        return "must contain nonnegative normalized values"
    if x + width > 1 or y + height > 1:
        return "must remain within normalized image bounds"
    return None


def _is_nonempty_serialized_mask(mask: object) -> bool:
    if isinstance(mask, str):
        return bool(mask.strip())
    if isinstance(mask, (Mapping, Sequence)) and not isinstance(mask, bytes):
        return bool(mask)
    return False


def validate_annotations(terminal: TerminalRecord) -> None:
    """Validate terminal body boxes and serialized body masks."""
    context = terminal.relative_filepath
    errors: list[str] = []
    bboxes = _detection_entries(
        terminal.bboxes_body,
        "bboxes_body",
        context,
        errors,
    )
    segmentations = _detection_entries(
        terminal.segmentations_body,
        "segmentations_body",
        context,
        errors,
    )

    for index, detection in bboxes:
        message = _validate_bbox(detection.get("bounding_box"))
        if message is not None:
            errors.append(f"{context}: bboxes_body.detections[{index}].bounding_box {message}")
    for index, detection in segmentations:
        if "bounding_box" in detection:
            message = _validate_bbox(detection["bounding_box"])
            if message is not None:
                errors.append(f"{context}: segmentations_body.detections[{index}].bounding_box " f"{message}")
        if not _is_nonempty_serialized_mask(detection.get("mask")):
            errors.append(f"{context}: segmentations_body.detections[{index}].mask " "must be nonempty serialized data")

    if errors:
        raise IntegrityError("; ".join(errors))


def validate_record(
    terminal: TerminalRecord,
    enrichment: Enrichment,
) -> ValidatedRecord:
    """Validate and aggregate one terminal record."""
    validate_annotations(terminal)
    return ValidatedRecord(
        terminal=terminal,
        enrichment=enrichment,
        integrity=validate_media(terminal.filepath),
    )


def _duplicate_groups(
    records: Sequence[ValidatedRecord],
    key: Callable[[ValidatedRecord], DuplicateKey],
) -> list[tuple[DuplicateKey, list[ValidatedRecord]]]:
    grouped: dict[DuplicateKey, list[ValidatedRecord]] = {}
    for record in records:
        grouped.setdefault(key(record), []).append(record)
    return [(value, matches) for value, matches in grouped.items() if len(matches) > 1]


def _record_names(records: Sequence[ValidatedRecord]) -> str:
    return ", ".join(record.terminal.relative_filepath for record in records)


def validate_unique_records(records: Sequence[ValidatedRecord]) -> None:
    """Reject canonical filepath and content-hash duplicates."""
    duplicate_paths = _duplicate_groups(
        records,
        lambda record: record.terminal.filepath.resolve(),
    )
    duplicate_hashes = _duplicate_groups(
        records,
        lambda record: record.integrity.sha256,
    )
    errors = [f"duplicate canonical path {path}: {_record_names(matches)}" for path, matches in duplicate_paths]
    errors.extend(f"duplicate SHA-256 {digest}: {_record_names(matches)}" for digest, matches in duplicate_hashes)
    if errors:
        raise IntegrityError("; ".join(errors))


def validate_expected_count(actual_count: int, expected_count: int) -> None:
    """Require the validated batch to contain the expected sample count."""
    if actual_count != expected_count:
        raise IntegrityError(f"expected {expected_count} records, found {actual_count}")


def validate_records(
    records: Sequence[tuple[TerminalRecord, Enrichment]],
    *,
    expected_count: int | None = None,
) -> list[ValidatedRecord]:
    """Validate a complete terminal/enrichment batch."""
    errors: list[str] = []
    validated: list[ValidatedRecord] = []
    if expected_count is not None:
        try:
            validate_expected_count(len(records), expected_count)
        except IntegrityError as exc:
            errors.append(str(exc))

    for terminal, enrichment in records:
        try:
            validated.append(validate_record(terminal, enrichment))
        except IntegrityError as exc:
            errors.append(str(exc))

    try:
        validate_unique_records(validated)
    except IntegrityError as exc:
        errors.append(str(exc))

    if errors:
        raise IntegrityError("; ".join(errors))
    return validated


def validate_mounts(
    paths: Sequence[Path],
    is_mount: Callable[[Path], bool] = os.path.ismount,
) -> None:
    """Require every configured external volume root to be a mountpoint."""
    errors: list[str] = []
    for path in paths:
        try:
            mounted = is_mount(path)
        except (OSError, ValueError) as exc:
            errors.append(f"could not inspect required mount {path}: {exc}")
            continue
        if not mounted:
            errors.append(f"required path is not a mounted filesystem: {path}")
    if errors:
        raise StorageSafetyError("; ".join(errors))


def validate_storage_paths(
    paths: Path | Sequence[Path],
    approved_root: Path = APPROVED_STORAGE_ROOT,
) -> None:
    """Require generated-state paths to resolve strictly below one root."""
    candidates = (paths,) if isinstance(paths, Path) else tuple(paths)
    errors: list[str] = []
    try:
        resolved_root = approved_root.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise StorageSafetyError(f"could not resolve approved storage root {approved_root}: {exc}") from exc

    for path in candidates:
        try:
            resolved_path = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"could not resolve generated-state path {path}: {exc}")
            continue
        if resolved_path == resolved_root:
            errors.append(f"generated-state path must be a strict descendant of " f"{resolved_root}, not the root itself: {path}")
        elif not resolved_path.is_relative_to(resolved_root):
            errors.append(f"generated-state path resolves outside approved storage root " f"{resolved_root}: {path} -> {resolved_path}")

    if errors:
        raise StorageSafetyError("; ".join(errors))


def validate_media(path: Path) -> MediaIntegrity:
    """Validate and fingerprint one final image."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise IntegrityError(f"media is missing or unreadable: {path}: {exc}") from exc
    if not path.is_file():
        raise IntegrityError(f"media is not a file: {path}")

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size

        digest = hashlib.sha256()
        with path.open("rb") as media:
            for chunk in iter(lambda: media.read(HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise IntegrityError(f"media is unreadable: {path}: {exc}") from exc

    return MediaIntegrity(
        sha256=digest.hexdigest(),
        size_bytes=stat.st_size,
        width=width,
        height=height,
    )
