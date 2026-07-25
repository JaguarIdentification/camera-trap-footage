import hashlib
import io
import math
import os
import zlib
from base64 import b64decode
from binascii import Error as Base64Error
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError
import numpy as np
from numpy.typing import NDArray

from jaguars.visualization.final_lineage import Enrichment
from jaguars.visualization.final_records import FrozenAnnotation, TerminalRecord

HASH_CHUNK_SIZE = 1024 * 1024
APPROVED_STORAGE_ROOT = Path("/Volumes/CameraTrapPython/fiftyone")
DuplicateKey = TypeVar("DuplicateKey", bound=Hashable)
DuplicateItem = TypeVar("DuplicateItem")


class IntegrityError(ValueError):
    """Raised when final snapshot source integrity validation fails."""


class StorageSafetyError(ValueError):
    """Raised when required mounts or generated-state paths are unsafe."""


@dataclass(frozen=True)
class ValidationDiagnostics:
    annotation_failed: int = 0
    enrichment_failed: int = 0
    media_failed: int = 0
    duplicate_path_groups: int = 0
    duplicate_path_pairs: int = 0
    duplicate_hash_groups: int = 0
    duplicate_hash_pairs: int = 0
    unique_paths: int | None = None
    unique_sha256: int | None = None
    errors: tuple[str, ...] = ()


class BatchValidationError(IntegrityError):
    """Raised with partial records and categorized batch diagnostics."""

    def __init__(
        self,
        message: str,
        records: Sequence["ValidatedRecord"],
        diagnostics: ValidationDiagnostics,
    ) -> None:
        super().__init__(message)
        self.records = tuple(records)
        self.diagnostics = diagnostics


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

    @property
    def resolved_jaguar_id(self) -> str | None:
        if self.terminal.jaguar_id is not None:
            return self.terminal.jaguar_id
        if self.enrichment.status != "matched":
            return None
        enriched_identity = self.enrichment.fields.get("jaguar_id")
        if not isinstance(enriched_identity, str) or not enriched_identity.strip():
            return None
        return enriched_identity.strip()


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


def _decoded_mask(mask: object) -> NDArray[np.generic]:
    if isinstance(mask, Mapping):
        binary = mask.get("$binary")
        if not isinstance(binary, Mapping):
            raise ValueError("mapping must contain a $binary object")
        encoded = binary.get("base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("$binary.base64 must be a nonempty string")
        serialized = b64decode(encoded, validate=True)
        with io.BytesIO(zlib.decompress(serialized)) as stream:
            decoded = np.load(stream, allow_pickle=False)
    elif isinstance(mask, str):
        serialized = b64decode(mask, validate=True)
        with io.BytesIO(zlib.decompress(serialized)) as stream:
            decoded = np.load(stream, allow_pickle=False)
    elif isinstance(mask, Sequence) and not isinstance(mask, (str, bytes)):
        decoded = np.asarray(mask)
    else:
        raise ValueError("mask must be serialized binary data or an array")

    if not isinstance(decoded, np.ndarray) or decoded.ndim != 2:
        raise ValueError("mask must decode to a two-dimensional array")
    if decoded.dtype.kind not in ("b", "i", "u"):
        raise ValueError("mask must decode to boolean or integer values")
    if decoded.size == 0 or not np.logical_or(decoded == 0, decoded == 1).all():
        raise ValueError("mask must decode to nonempty binary values")
    return decoded


def validate_annotations(terminal: TerminalRecord) -> None:
    """Validate terminal body boxes and serialized body masks."""
    context = terminal.relative_filepath
    errors: list[str] = []
    missing_bboxes = terminal.bboxes_body is None
    missing_segmentations = terminal.segmentations_body is None
    if missing_bboxes or missing_segmentations:
        if not terminal.review_required:
            raise IntegrityError(f"{context}: missing body annotations are only allowed when review_required is true")
        if not missing_bboxes or not missing_segmentations:
            errors.append(f"{context}: review samples must omit both bboxes_body and segmentations_body")
        if not isinstance(terminal.review_reason, str) or not terminal.review_reason.strip():
            errors.append(f"{context}: review_reason must be nonempty when review_required")
        if terminal.review_status != "pending":
            errors.append(f"{context}: review_status must be pending when review_required")
        if "needs_annotation_review" not in terminal.tags:
            errors.append(f"{context}: review tags must contain needs_annotation_review")
        if errors:
            raise IntegrityError("; ".join(errors))
        return
    if terminal.review_required:
        raise IntegrityError(f"{context}: review samples must omit bboxes_body and segmentations_body")
    bboxes_annotation = terminal.bboxes_body
    segmentations_annotation = terminal.segmentations_body
    if bboxes_annotation is None or segmentations_annotation is None:
        raise AssertionError("annotation presence was checked above")

    bboxes = _detection_entries(
        bboxes_annotation,
        "bboxes_body",
        context,
        errors,
    )
    segmentations = _detection_entries(
        segmentations_annotation,
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
        try:
            _decoded_mask(detection.get("mask"))
        except (Base64Error, OSError, ValueError, zlib.error) as error:
            errors.append(f"{context}: segmentations_body.detections[{index}].mask could not decode: {error}")

    if errors:
        raise IntegrityError("; ".join(errors))


def validate_record(
    terminal: TerminalRecord,
    enrichment: Enrichment,
) -> ValidatedRecord:
    """Validate and aggregate one terminal record."""
    validate_annotations(terminal)
    validate_enrichment(enrichment, terminal.relative_filepath)
    return ValidatedRecord(
        terminal=terminal,
        enrichment=enrichment,
        integrity=validate_media(terminal.filepath),
    )


_STRING_ENRICHMENT_FIELDS = (
    "jaguar_id",
    "sighting_id",
    "site",
    "location",
    "camera_id",
    "camera_side",
    "camera_model",
    "capture_date",
    "capture_time",
    "capture_datetime",
    "original_filename",
    "source_media_path",
    "source_type",
)
_SPLIT_FIELDS = ("closed_set_split", "open_set_split")
_APPROVED_SPLITS = frozenset(("train", "val", "test"))
_COORDINATE_FIELDS = ("latitude", "longitude")


def validate_enrichment(
    enrichment: Enrichment,
    context: str,
) -> None:
    """Validate approved enrichment values without importing FiftyOne."""
    errors: list[str] = []
    for field_name in _STRING_ENRICHMENT_FIELDS:
        value = enrichment.fields.get(field_name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{context}: {field_name} must be a nonempty string when populated")
    for field_name in _SPLIT_FIELDS:
        value = enrichment.fields.get(field_name)
        if value is not None and (not isinstance(value, str) or value not in _APPROVED_SPLITS):
            errors.append(f"{context}: {field_name} must be one of train, val, test when populated")
    for field_name in _COORDINATE_FIELDS:
        value = enrichment.fields.get(field_name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            errors.append(f"{context}: {field_name} must be a finite numeric value when populated")
    if errors:
        raise IntegrityError("; ".join(errors))


def _duplicate_groups(
    items: Sequence[DuplicateItem],
    key: Callable[[DuplicateItem], DuplicateKey],
) -> list[tuple[DuplicateKey, list[DuplicateItem]]]:
    grouped: dict[DuplicateKey, list[DuplicateItem]] = {}
    for item in items:
        grouped.setdefault(key(item), []).append(item)
    return [(value, matches) for value, matches in grouped.items() if len(matches) > 1]


def _record_names(records: Sequence[TerminalRecord]) -> str:
    return ", ".join(record.relative_filepath for record in records)


def _duplicate_path_errors(records: Sequence[TerminalRecord]) -> list[str]:
    grouped: dict[Path, list[TerminalRecord]] = {}
    errors: list[str] = []
    for record in records:
        try:
            canonical_path = record.filepath.resolve(strict=record.filepath.is_symlink())
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{record.relative_filepath}: could not resolve canonical path {record.filepath}: {exc}")
        else:
            grouped.setdefault(canonical_path, []).append(record)

    errors.extend(f"duplicate canonical path {path}: {_record_names(matches)}" for path, matches in grouped.items() if len(matches) > 1)
    return errors


def _path_diagnostics(
    records: Sequence[TerminalRecord],
) -> tuple[int, int, int]:
    grouped: dict[Path, int] = {}
    for record in records:
        try:
            canonical_path = record.filepath.resolve(strict=record.filepath.is_symlink())
        except (OSError, RuntimeError, ValueError):
            continue
        grouped[canonical_path] = grouped.get(canonical_path, 0) + 1
    duplicate_sizes = [size for size in grouped.values() if size > 1]
    return (
        len(grouped),
        len(duplicate_sizes),
        sum(size * (size - 1) // 2 for size in duplicate_sizes),
    )


def _duplicate_hash_errors(records: Sequence[ValidatedRecord]) -> list[str]:
    duplicate_hashes = _duplicate_groups(
        records,
        lambda record: record.integrity.sha256,
    )
    return [f"duplicate SHA-256 {digest}: {_record_names([record.terminal for record in matches])}" for digest, matches in duplicate_hashes]


def validate_unique_records(records: Sequence[ValidatedRecord]) -> None:
    """Reject canonical filepath and content-hash duplicates."""
    errors = _duplicate_path_errors([record.terminal for record in records])
    errors.extend(_duplicate_hash_errors(records))
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
    annotation_failed = 0
    enrichment_failed = 0
    media_failed = 0
    if expected_count is not None:
        try:
            validate_expected_count(len(records), expected_count)
        except IntegrityError as exc:
            errors.append(str(exc))

    errors.extend(_duplicate_path_errors([terminal for terminal, _ in records]))
    for terminal, enrichment in records:
        try:
            validate_annotations(terminal)
        except IntegrityError as exc:
            annotation_failed += 1
            errors.append(str(exc))
        try:
            validate_enrichment(enrichment, terminal.relative_filepath)
        except IntegrityError as exc:
            enrichment_failed += 1
            errors.append(str(exc))
        try:
            integrity = validate_media(terminal.filepath)
        except IntegrityError as exc:
            media_failed += 1
            errors.append(str(exc))
        else:
            validated.append(
                ValidatedRecord(
                    terminal=terminal,
                    enrichment=enrichment,
                    integrity=integrity,
                )
            )

    duplicate_hashes = _duplicate_groups(
        validated,
        lambda record: record.integrity.sha256,
    )
    errors.extend(f"duplicate SHA-256 {digest}: {_record_names([record.terminal for record in matches])}" for digest, matches in duplicate_hashes)

    if errors:
        unique_paths, duplicate_path_groups, duplicate_path_pairs = _path_diagnostics([terminal for terminal, _ in records])
        raise BatchValidationError(
            "; ".join(errors),
            validated,
            ValidationDiagnostics(
                annotation_failed=annotation_failed,
                enrichment_failed=enrichment_failed,
                media_failed=media_failed,
                duplicate_path_groups=duplicate_path_groups,
                duplicate_path_pairs=duplicate_path_pairs,
                duplicate_hash_groups=len(duplicate_hashes),
                duplicate_hash_pairs=sum(len(matches) * (len(matches) - 1) // 2 for _, matches in duplicate_hashes),
                unique_paths=unique_paths,
                unique_sha256=len({record.integrity.sha256 for record in validated}),
                errors=tuple(errors),
            ),
        )
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
    except (OSError, ValueError) as exc:
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
    except (DecompressionBombError, OSError, ValueError, UnidentifiedImageError) as exc:
        raise IntegrityError(f"media is unreadable: {path}: {exc}") from exc

    return MediaIntegrity(
        sha256=digest.hexdigest(),
        size_bytes=stat.st_size,
        width=width,
        height=height,
    )
