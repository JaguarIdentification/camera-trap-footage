from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, cast
from uuid import uuid4

import fiftyone as fo
import fiftyone.core.fields as fof
import fiftyone.core.odm as foo
import numpy as np
from fiftyone import ViewField as F

from jaguars.visualization.final_lineage import Scalar
from jaguars.visualization.final_records import FrozenAnnotation, annotation_to_dict
from jaguars.visualization.final_validation import ValidatedRecord

BATCH_SIZE = 100
OWNERSHIP_INFO_KEY = "jaguars_snapshot_ownership_token"

APPROVED_ENRICHMENT_FIELDS = (
    "closed_set_split",
    "open_set_split",
    "sighting_id",
    "site",
    "location",
    "camera_id",
    "camera_side",
    "camera_model",
    "latitude",
    "longitude",
    "capture_date",
    "capture_time",
    "capture_datetime",
    "original_filename",
    "source_media_path",
    "source_type",
)

APPROVED_SAMPLE_FIELDS = (
    "jaguar_id",
    "ground_truth",
    "bboxes_body",
    "segmentations_body",
    "lineage_status",
    "lineage_match_method",
    *APPROVED_ENRICHMENT_FIELDS,
    "sha256",
    "size_bytes",
    "width",
    "height",
)

SAVED_VIEW_NAMES = (
    "All final samples",
    "Lineage issues",
    "Closed-set train",
    "Closed-set val",
    "Closed-set test",
    "Open-set train",
    "Open-set val",
    "Open-set test",
)

_STRING_ENRICHMENT_FIELDS = (
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
_FLOAT_ENRICHMENT_FIELDS = ("latitude", "longitude")
_REQUIRED_VALUE_FIELDS = (
    "bboxes_body",
    "segmentations_body",
    "lineage_status",
    "sha256",
    "size_bytes",
    "width",
    "height",
)


class SnapshotError(ValueError):
    """Base class for controlled snapshot construction failures."""


class SnapshotCollisionError(SnapshotError):
    """Raised when an atomic snapshot name is not available."""


class SnapshotValidationError(SnapshotError):
    """Raised when a temporary snapshot fails pre-publication validation."""


class SnapshotReplacementError(SnapshotError):
    """Raised when transactional replacement cannot safely complete."""


class PublishedSnapshotCleanupError(SnapshotReplacementError):
    """Raised when publication succeeded but retirement of the old backup failed."""

    def __init__(self, message: str, dataset: fo.Dataset) -> None:
        super().__init__(message)
        self.published_dataset = dataset


@dataclass(frozen=True)
class _OwnedDataset:
    dataset_id: Any
    token: str


class _SnapshotPhase(Enum):
    PREFLIGHT = auto()
    STAGED = auto()
    PROMOTION_ATTEMPTED = auto()
    PUBLISHED = auto()


@dataclass
class _TransactionState:
    phase: _SnapshotPhase = _SnapshotPhase.PREFLIGHT


@dataclass(frozen=True)
class _PublishedReplacement:
    original: fo.Dataset
    original_id: Any
    dataset_name: str
    backup_name: str
    ownership: _OwnedDataset


def _validate_decoded_mask(mask: object, context: str) -> None:
    if not isinstance(mask, np.ndarray):
        raise SnapshotValidationError(f"{context} must decode to a numpy array")
    if mask.ndim != 2:
        raise SnapshotValidationError(f"{context} must decode to a two-dimensional array")
    if mask.dtype.kind not in ("b", "i", "u"):
        raise SnapshotValidationError(f"{context} must decode to a boolean or integer dtype")
    if mask.size == 0 or not np.logical_or(mask == 0, mask == 1).all():
        raise SnapshotValidationError(f"{context} must contain nonempty binary values")


def _detections_from_export(
    annotation: FrozenAnnotation,
    field_name: str,
) -> fo.Detections:
    payload = annotation_to_dict(annotation)
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise SnapshotValidationError(f"{field_name}.detections must be a list")

    reconstructed: list[fo.Detection] = []
    for index, detection in enumerate(detections):
        mask_context = f"{field_name}.detections[{index}].mask"
        try:
            reconstructed_detection = fo.Detection.from_dict(detection)
        except Exception as exc:
            if "mask" in detection:
                raise SnapshotValidationError(f"{mask_context} could not decode") from exc
            raise SnapshotValidationError(f"{field_name}.detections[{index}] could not be reconstructed") from exc
        if "mask" in detection:
            _validate_decoded_mask(reconstructed_detection.mask, mask_context)
        reconstructed.append(reconstructed_detection)

    return fo.Detections(detections=reconstructed)


def _optional_string(value: Scalar) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Scalar) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SnapshotValidationError("boolean enrichment values cannot be stored as coordinates")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"invalid coordinate enrichment value: {value!r}") from exc


def _optional_split(field_name: str, value: Scalar) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SnapshotValidationError(f"{field_name} must be a string when populated")
    return value


def validated_record_to_sample(record: ValidatedRecord) -> fo.Sample:
    """Map one immutable validated record into the approved sample schema."""
    enrichment: dict[str, object] = {field: _optional_string(record.enrichment.fields.get(field)) for field in _STRING_ENRICHMENT_FIELDS}
    enrichment.update({field: _optional_split(field, record.enrichment.fields.get(field)) for field in _SPLIT_FIELDS})
    enrichment.update({field: _optional_float(record.enrichment.fields.get(field)) for field in _FLOAT_ENRICHMENT_FIELDS})
    terminal = record.terminal
    integrity = record.integrity
    resolved_identity = record.resolved_jaguar_id
    return fo.Sample(
        filepath=str(terminal.filepath),
        jaguar_id=resolved_identity,
        ground_truth=None if resolved_identity is None else fo.Classification(label=resolved_identity),
        bboxes_body=_detections_from_export(terminal.bboxes_body, "bboxes_body"),
        segmentations_body=_detections_from_export(
            terminal.segmentations_body,
            "segmentations_body",
        ),
        lineage_status=record.enrichment.status,
        lineage_match_method=record.enrichment.match_method,
        **enrichment,
        sha256=integrity.sha256,
        size_bytes=integrity.size_bytes,
        width=integrity.width,
        height=integrity.height,
    )


def _declare_schema(dataset: fo.Dataset) -> None:
    embedded_fields = {
        "ground_truth": fo.Classification,
        "bboxes_body": fo.Detections,
        "segmentations_body": fo.Detections,
    }
    string_fields = (
        "jaguar_id",
        "lineage_status",
        "lineage_match_method",
        *_SPLIT_FIELDS,
        *_STRING_ENRICHMENT_FIELDS,
        "sha256",
    )
    for field_name in string_fields:
        dataset.add_sample_field(field_name, fof.StringField)
    for field_name, document_type in embedded_fields.items():
        dataset.add_sample_field(
            field_name,
            fof.EmbeddedDocumentField,
            embedded_doc_type=document_type,
        )
    for field_name in _FLOAT_ENRICHMENT_FIELDS:
        dataset.add_sample_field(field_name, fof.FloatField)
    for field_name in ("size_bytes", "width", "height"):
        dataset.add_sample_field(field_name, fof.IntField)


def _save_views(dataset: fo.Dataset) -> None:
    dataset.save_view("All final samples", dataset.view())
    dataset.save_view(
        "Lineage issues",
        dataset.match(F("lineage_status") != "matched"),
    )
    for split in ("train", "val", "test"):
        dataset.save_view(
            f"Closed-set {split}",
            dataset.match(F("closed_set_split") == split),
        )
    for split in ("train", "val", "test"):
        dataset.save_view(
            f"Open-set {split}",
            dataset.match(F("open_set_split") == split),
        )


def _expected_identity(record: ValidatedRecord) -> tuple[object, ...]:
    resolved_identity = record.resolved_jaguar_id
    return (
        str(record.terminal.filepath),
        resolved_identity,
        resolved_identity,
        record.integrity.sha256,
        record.integrity.size_bytes,
        record.integrity.width,
        record.integrity.height,
    )


def _actual_identity(sample: fo.Sample) -> tuple[object, ...]:
    ground_truth = sample.ground_truth
    ground_truth_label = ground_truth.label if isinstance(ground_truth, fo.Classification) else None
    return (
        sample.filepath,
        sample.jaguar_id,
        ground_truth_label,
        sample.sha256,
        sample.size_bytes,
        sample.width,
        sample.height,
    )


def _validate_required_fields(dataset: fo.Dataset) -> None:
    schema = dataset.get_field_schema()
    missing_schema = sorted(set(APPROVED_SAMPLE_FIELDS) - set(schema))
    if missing_schema:
        raise SnapshotValidationError(f"required fields missing from schema: {', '.join(missing_schema)}")

    for sample in dataset.iter_samples(progress=False):
        missing_values = [field for field in _REQUIRED_VALUE_FIELDS if sample.get_field(field) is None]
        if missing_values:
            raise SnapshotValidationError(f"required fields missing values for {sample.filepath}: {', '.join(missing_values)}")


def _validate_split_values(dataset: fo.Dataset) -> None:
    for sample in dataset.iter_samples(progress=False):
        for field_name in _SPLIT_FIELDS:
            value = sample.get_field(field_name)
            if value is not None and value not in _APPROVED_SPLITS:
                raise SnapshotValidationError(f"{field_name} must be one of train, val, test when populated; found {value!r}")


def _validate_snapshot(
    dataset: fo.Dataset,
    records: Sequence[ValidatedRecord],
) -> None:
    expected_count = len(records)
    actual_count = len(dataset)
    if actual_count != expected_count:
        raise SnapshotValidationError(f"expected {expected_count} samples, found {actual_count}")

    expected_identities = sorted((_expected_identity(record) for record in records), key=repr)
    actual_identities = sorted((_actual_identity(sample) for sample in dataset.iter_samples(progress=False)), key=repr)
    if actual_identities != expected_identities:
        raise SnapshotValidationError("snapshot identity agreement failed")

    _validate_required_fields(dataset)
    _validate_split_values(dataset)
    actual_views = tuple(dataset.list_saved_views())
    if actual_views != SAVED_VIEW_NAMES:
        raise SnapshotValidationError(f"saved views do not match approved views: expected {SAVED_VIEW_NAMES!r}, found {actual_views!r}")


def _insert_bounded(
    dataset: fo.Dataset,
    records: Sequence[ValidatedRecord],
) -> None:
    for start in range(0, len(records), BATCH_SIZE):
        batch = [validated_record_to_sample(record) for record in records[start : start + BATCH_SIZE]]
        dataset.add_samples(
            batch,
            expand_schema=False,
            dynamic=False,
            validate=True,
            batcher=False,
            progress=False,
        )


def _refuse_collisions(dataset_name: str, temporary_name: str) -> None:
    if dataset_name == temporary_name:
        raise SnapshotCollisionError("final and temporary dataset names must be distinct")
    if fo.dataset_exists(dataset_name):
        raise SnapshotCollisionError(f"final dataset already exists: {dataset_name}")
    if fo.dataset_exists(temporary_name):
        raise SnapshotCollisionError(f"temporary dataset already exists: {temporary_name}")


def _build_dataset_name(temporary_name: str) -> str:
    return f"{temporary_name}--build-{uuid4().hex}"


def _build_backup_name(temporary_name: str) -> str:
    return f"{temporary_name}--backup-{uuid4().hex}"


def _rename_dataset(dataset: fo.Dataset, name: str) -> None:
    dataset.name = name


def _database_document_by_name(name: str) -> dict[str, Any] | None:
    return cast(
        dict[str, Any] | None,
        foo.get_db_conn().datasets.find_one({"name": name}),
    )


def _database_document_by_id(dataset_id: Any) -> dict[str, Any] | None:
    return cast(
        dict[str, Any] | None,
        foo.get_db_conn().datasets.find_one({"_id": dataset_id}),
    )


def _query_document_by_id_with_retry(dataset_id: Any) -> dict[str, Any] | None:
    try:
        return _database_document_by_id(dataset_id)
    except BaseException:
        return _database_document_by_id(dataset_id)


def _query_document_by_name_with_retry(name: str) -> dict[str, Any] | None:
    try:
        return _database_document_by_name(name)
    except BaseException:
        return _database_document_by_name(name)


def _document_name(document: dict[str, Any] | None) -> str | None:
    if document is None:
        return None
    name = document.get("name")
    return name if isinstance(name, str) else None


def _document_has_ownership(
    document: dict[str, Any] | None,
    ownership: _OwnedDataset,
) -> bool:
    if document is None or document.get("_id") != ownership.dataset_id:
        return False
    info = document.get("info")
    return isinstance(info, dict) and info.get(OWNERSHIP_INFO_KEY) == ownership.token


def _owned_document(ownership: _OwnedDataset) -> dict[str, Any] | None:
    document = _database_document_by_id(ownership.dataset_id)
    return document if _document_has_ownership(document, ownership) else None


def _delete_owned_dataset(
    dataset: fo.Dataset,
    ownership: _OwnedDataset,
) -> bool:
    if _owned_document(ownership) is None:
        return False

    dataset._doc.reload()
    if dataset._doc.id != ownership.dataset_id:
        raise SnapshotReplacementError("refusing to delete staging dataset whose database identity changed")
    if dataset.info.get(OWNERSHIP_INFO_KEY) != ownership.token:
        raise SnapshotReplacementError("refusing to delete staging dataset without persisted ownership proof")

    dataset.delete()
    if _database_document_by_id(ownership.dataset_id) is not None:
        raise SnapshotReplacementError("owned staging dataset remained after cleanup")
    return True


def _create_owned_staging(
    name: str,
    ownership_token: str,
) -> tuple[fo.Dataset, _OwnedDataset]:
    try:
        dataset = fo.Dataset(name, persistent=True)
    except BaseException as construction_error:
        persisted = _database_document_by_name(name)
        if persisted is not None:
            raise SnapshotReplacementError(
                f"staging construction failed after metadata was persisted at {name}; " "ownership could not be proven, so the artifact was retained"
            ) from construction_error
        if isinstance(construction_error, ValueError) and "not available" in str(construction_error):
            raise SnapshotCollisionError(f"generated build dataset became occupied: {name}") from construction_error
        raise

    ownership = _OwnedDataset(dataset_id=dataset._doc.id, token=ownership_token)
    dataset.info[OWNERSHIP_INFO_KEY] = ownership_token
    try:
        dataset.save()
    except BaseException as ownership_error:
        if _owned_document(ownership) is not None:
            _delete_owned_dataset(dataset, ownership)
            raise
        raise SnapshotReplacementError(
            f"staging ownership metadata failed at {name}; " "ownership could not be proven, so the artifact was retained"
        ) from ownership_error

    if _owned_document(ownership) is None:
        raise SnapshotReplacementError(f"staging ownership metadata was not persisted at {name}; " "the unproven artifact was retained")
    return dataset, ownership


def _refuse_replacement_collisions(
    dataset_name: str,
    temporary_name: str,
    expected_original_id: Any,
) -> None:
    if dataset_name == temporary_name:
        raise SnapshotCollisionError("final and temporary dataset names must be distinct")
    original_document = _database_document_by_name(dataset_name)
    if original_document is None:
        raise SnapshotReplacementError(f"confirmed final dataset disappeared before staging: {dataset_name}")
    if original_document.get("_id") != expected_original_id:
        raise SnapshotReplacementError(f"confirmed final dataset changed before staging: {dataset_name}")
    if fo.dataset_exists(temporary_name):
        raise SnapshotCollisionError(f"temporary dataset already exists: {temporary_name}")


def _restore_original(
    original: fo.Dataset,
    original_id: Any,
    dataset_name: str,
    backup_name: str,
    promotion_error: BaseException,
) -> None:
    original_document = _query_document_by_id_with_retry(original_id)
    original_name = _document_name(original_document)
    final_document = _query_document_by_name_with_retry(dataset_name)

    if original_name == dataset_name and final_document is not None and final_document.get("_id") == original_id:
        return
    if original_name != backup_name:
        raise SnapshotReplacementError(f"replacement failed and the old dataset location is unknown: {original_name!r}") from promotion_error
    if final_document is not None:
        raise SnapshotReplacementError(
            f"foreign dataset occupies final name {dataset_name}; " f"old dataset remains safely recoverable at {backup_name}"
        ) from promotion_error

    original._doc.reload()
    try:
        _rename_dataset(original, dataset_name)
    except BaseException as rollback_error:
        restored_document = _query_document_by_id_with_retry(original_id)
        restored_final = _query_document_by_name_with_retry(dataset_name)
        if _document_name(restored_document) == dataset_name and restored_final is not None and restored_final.get("_id") == original_id:
            return
        raise SnapshotReplacementError(
            f"replacement promotion failed and rollback failed; " f"old dataset remains at {_document_name(restored_document)!r}: {rollback_error}"
        ) from promotion_error

    restored_document = _query_document_by_id_with_retry(original_id)
    restored_final = _query_document_by_name_with_retry(dataset_name)
    if _document_name(restored_document) != dataset_name or restored_final is None or restored_final.get("_id") != original_id:
        raise SnapshotReplacementError(
            f"rollback did not restore the old dataset; " f"it remains at {_document_name(restored_document)!r}"
        ) from promotion_error


def _recover_backup_rename_failure(
    original: fo.Dataset,
    original_id: Any,
    dataset_name: str,
    backup_name: str,
    rename_error: BaseException,
) -> None:
    original_document = _database_document_by_id(original_id)
    original_name = _document_name(original_document)
    final_document = _database_document_by_name(dataset_name)
    if original_name == dataset_name and final_document is not None and final_document.get("_id") == original_id:
        return
    if original_name == backup_name:
        _restore_original(
            original,
            original_id,
            dataset_name,
            backup_name,
            rename_error,
        )
        return
    raise SnapshotReplacementError(f"old dataset rename failed with unexpected database location {original_name!r}") from rename_error


def _recover_promotion_failure(
    staged: fo.Dataset,
    ownership: _OwnedDataset,
    original: fo.Dataset,
    original_id: Any,
    dataset_name: str,
    backup_name: str,
    promotion_error: BaseException,
) -> None:
    try:
        final_document = _query_document_by_name_with_retry(dataset_name)
    except BaseException as query_error:
        raise SnapshotReplacementError(
            f"promotion recovery could not query final state; old dataset remains "
            f"recoverable at {backup_name} and owned replacement state was retained"
        ) from query_error

    if final_document is not None:
        if _document_has_ownership(final_document, ownership):
            _delete_owned_for_recovery(
                staged,
                ownership,
                backup_name,
                promotion_error,
            )
        else:
            _delete_owned_for_recovery(
                staged,
                ownership,
                backup_name,
                promotion_error,
            )
            raise SnapshotReplacementError(
                f"foreign dataset occupies final name {dataset_name}; " f"old dataset remains safely recoverable at {backup_name}"
            ) from promotion_error
    else:
        _delete_owned_for_recovery(
            staged,
            ownership,
            backup_name,
            promotion_error,
        )

    _restore_original(
        original,
        original_id,
        dataset_name,
        backup_name,
        promotion_error,
    )


def _delete_owned_for_recovery(
    staged: fo.Dataset,
    ownership: _OwnedDataset,
    backup_name: str,
    promotion_error: BaseException,
) -> None:
    try:
        _delete_owned_dataset(staged, ownership)
        return
    except BaseException:
        if _owned_document(ownership) is None:
            return

    try:
        _delete_owned_dataset(staged, ownership)
    except BaseException as retry_error:
        remaining = _owned_document(ownership)
        if remaining is None:
            return
        raise SnapshotReplacementError(
            f"owned replacement recovery artifact remains at "
            f"{_document_name(remaining)!r}; old dataset remains recoverable "
            f"at {backup_name}: {retry_error}"
        ) from promotion_error


def _query_published_document_with_retry(
    dataset_name: str,
) -> dict[str, Any] | None:
    return _query_document_by_name_with_retry(dataset_name)


def _promote_replacement(
    staged: fo.Dataset,
    ownership: _OwnedDataset,
    dataset_name: str,
    temporary_name: str,
    expected_original_id: Any,
    transaction: _TransactionState,
) -> _PublishedReplacement:
    original_document = _database_document_by_name(dataset_name)
    if original_document is None:
        raise SnapshotReplacementError(f"confirmed final dataset disappeared before promotion: {dataset_name}")
    if original_document.get("_id") != expected_original_id:
        raise SnapshotReplacementError(f"confirmed final dataset changed before promotion: {dataset_name}")
    original_id = expected_original_id
    original = fo.load_dataset(dataset_name, reload=True)
    if original._doc.id != expected_original_id:
        raise SnapshotReplacementError(f"confirmed final dataset changed while loading for promotion: {dataset_name}")
    backup_name = _build_backup_name(temporary_name)
    if _database_document_by_name(backup_name) is not None:
        raise SnapshotCollisionError(f"replacement backup dataset already exists: {backup_name}")

    try:
        _rename_dataset(original, backup_name)
    except BaseException as first_rename_error:
        _recover_backup_rename_failure(
            original,
            original_id,
            dataset_name,
            backup_name,
            first_rename_error,
        )
        raise

    transaction.phase = _SnapshotPhase.PROMOTION_ATTEMPTED
    try:
        renamed_original = _query_document_by_id_with_retry(original_id)
    except BaseException as query_error:
        _recover_promotion_failure(
            staged,
            ownership,
            original,
            original_id,
            dataset_name,
            backup_name,
            query_error,
        )
        raise
    if _document_name(renamed_original) != backup_name:
        state_error = SnapshotReplacementError("old dataset rename did not persist")
        _recover_backup_rename_failure(
            original,
            original_id,
            dataset_name,
            backup_name,
            state_error,
        )
        raise state_error

    try:
        _rename_dataset(staged, dataset_name)
    except BaseException as promotion_error:
        _recover_promotion_failure(
            staged,
            ownership,
            original,
            original_id,
            dataset_name,
            backup_name,
            promotion_error,
        )
        raise

    try:
        promoted_document = _query_published_document_with_retry(dataset_name)
    except BaseException as verification_error:
        _recover_promotion_failure(
            staged,
            ownership,
            original,
            original_id,
            dataset_name,
            backup_name,
            verification_error,
        )
        raise
    if not _document_has_ownership(promoted_document, ownership):
        state_error = SnapshotReplacementError("replacement promotion did not persist owned staging identity")
        _recover_promotion_failure(
            staged,
            ownership,
            original,
            original_id,
            dataset_name,
            backup_name,
            state_error,
        )
        raise state_error

    return _PublishedReplacement(
        original=original,
        original_id=original_id,
        dataset_name=dataset_name,
        backup_name=backup_name,
        ownership=ownership,
    )


def _delete_replaced_backup(replacement: _PublishedReplacement) -> None:
    original_document = _database_document_by_id(replacement.original_id)
    if _document_name(original_document) != replacement.backup_name:
        raise SnapshotReplacementError(
            f"replacement remains published, but old backup location is "
            f"{_document_name(original_document)!r} instead of {replacement.backup_name!r}"
        )

    try:
        replacement.original._doc.reload()
        replacement.original.delete()
    except BaseException as cleanup_error:
        published_document = _database_document_by_name(replacement.dataset_name)
        if not _document_has_ownership(published_document, replacement.ownership):
            raise SnapshotReplacementError(
                "old backup cleanup failed and the verified replacement publication changed unexpectedly"
            ) from cleanup_error

        remaining_backup = _database_document_by_id(replacement.original_id)
        remaining_name = _document_name(remaining_backup)
        if remaining_backup is None:
            raise SnapshotReplacementError(
                f"replacement remains published; old backup was already removed " f"despite cleanup error: {cleanup_error}"
            ) from cleanup_error
        if remaining_name == replacement.backup_name:
            raise SnapshotReplacementError(
                f"replacement remains published; old backup remains recoverable " f"at {replacement.backup_name}: {cleanup_error}"
            ) from cleanup_error
        raise SnapshotReplacementError(
            f"replacement remains published; old dataset remains at unexpected " f"location {remaining_name!r}: {cleanup_error}"
        ) from cleanup_error

    remaining_backup = _database_document_by_id(replacement.original_id)
    if remaining_backup is not None:
        raise SnapshotReplacementError(
            f"replacement remains published; old backup cleanup returned without " f"deleting {_document_name(remaining_backup)!r}"
        )


def create_snapshot(
    records: Sequence[ValidatedRecord],
    dataset_name: str,
    temporary_name: str,
    *,
    replace_existing: bool = False,
    expected_original_id: object | None = None,
) -> fo.Dataset:
    """Atomically publish a persistent, validated FiftyOne snapshot."""
    dataset: fo.Dataset | None = None
    ownership: _OwnedDataset | None = None
    transaction = _TransactionState()
    try:
        if replace_existing:
            if expected_original_id is None:
                raise SnapshotReplacementError("replacement requires the confirmed original dataset ID")
            _refuse_replacement_collisions(
                dataset_name,
                temporary_name,
                expected_original_id,
            )
        else:
            if expected_original_id is not None:
                raise SnapshotReplacementError("ordinary creation cannot specify an original dataset ID")
            _refuse_collisions(dataset_name, temporary_name)
        owned_build_name = _build_dataset_name(temporary_name)
        if _database_document_by_name(owned_build_name) is not None:
            raise SnapshotCollisionError(f"generated build dataset already exists: {owned_build_name}")
        dataset, ownership = _create_owned_staging(
            owned_build_name,
            uuid4().hex,
        )
        transaction.phase = _SnapshotPhase.STAGED
        _declare_schema(dataset)
        _insert_bounded(dataset, records)
        _save_views(dataset)
        _validate_snapshot(dataset, records)
        if replace_existing:
            replacement = _promote_replacement(
                dataset,
                ownership,
                dataset_name,
                temporary_name,
                expected_original_id,
                transaction,
            )
            transaction.phase = _SnapshotPhase.PUBLISHED
            try:
                _delete_replaced_backup(replacement)
            except BaseException as cleanup_error:
                raise PublishedSnapshotCleanupError(
                    str(cleanup_error),
                    dataset,
                ) from cleanup_error
            dataset._doc.reload()
            return dataset
        _rename_dataset(dataset, dataset_name)
        published_document = _database_document_by_name(dataset_name)
        if not _document_has_ownership(published_document, ownership):
            raise SnapshotReplacementError("snapshot publication did not persist owned staging identity")
        transaction.phase = _SnapshotPhase.PUBLISHED
        dataset._doc.reload()
        return dataset
    except BaseException as operation_error:
        if (
            transaction.phase
            in (
                _SnapshotPhase.PREFLIGHT,
                _SnapshotPhase.STAGED,
            )
            and dataset is not None
            and ownership is not None
        ):
            try:
                _delete_owned_dataset(dataset, ownership)
            except BaseException as cleanup_error:
                raise SnapshotReplacementError(f"owned staging cleanup failed: {cleanup_error}") from operation_error
        raise
