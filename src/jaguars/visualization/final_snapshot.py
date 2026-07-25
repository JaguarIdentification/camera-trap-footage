from collections.abc import Sequence

import fiftyone as fo
import fiftyone.core.fields as fof
import numpy as np
from fiftyone import ViewField as F

from jaguars.visualization.final_lineage import Scalar
from jaguars.visualization.final_records import FrozenAnnotation, annotation_to_dict
from jaguars.visualization.final_validation import ValidatedRecord

BATCH_SIZE = 100

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
    "jaguar_id",
    "ground_truth",
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
    return fo.Sample(
        filepath=str(terminal.filepath),
        jaguar_id=terminal.jaguar_id,
        ground_truth=fo.Classification(label=terminal.jaguar_id),
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
    return (
        str(record.terminal.filepath),
        record.terminal.jaguar_id,
        record.terminal.jaguar_id,
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


def create_snapshot(
    records: Sequence[ValidatedRecord],
    dataset_name: str,
    temporary_name: str,
) -> fo.Dataset:
    """Atomically publish a persistent, validated FiftyOne snapshot."""
    temporary_name_was_absent = False
    dataset: fo.Dataset | None = None
    try:
        _refuse_collisions(dataset_name, temporary_name)
        temporary_name_was_absent = True
        dataset = fo.Dataset(temporary_name, persistent=True)
        _declare_schema(dataset)
        _insert_bounded(dataset, records)
        _save_views(dataset)
        _validate_snapshot(dataset, records)
        dataset.name = dataset_name
        return dataset
    except Exception:
        if dataset is not None and not dataset.deleted:
            dataset.delete()
        elif temporary_name_was_absent and fo.dataset_exists(temporary_name):
            fo.delete_dataset(temporary_name)
        raise
