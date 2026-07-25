from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import fiftyone as fo
import fiftyone.core.fields as fof
import fiftyone.core.utils as fou
import numpy as np
import pytest

import jaguars.visualization.final_snapshot as final_snapshot
from jaguars.visualization.final_lineage import Enrichment
from jaguars.visualization.final_records import FrozenAnnotation, TerminalRecord
from jaguars.visualization.final_snapshot import (
    APPROVED_SAMPLE_FIELDS,
    SAVED_VIEW_NAMES,
    SnapshotCollisionError,
    SnapshotValidationError,
    create_snapshot,
    validated_record_to_sample,
)
from jaguars.visualization.final_validation import MediaIntegrity, ValidatedRecord


def _validated_record(
    tmp_path: Path,
    *,
    index: int = 0,
    lineage_status: str = "matched",
    match_method: str | None = "source_id",
    closed_set_split: str = "train",
    open_set_split: str = "val",
) -> ValidatedRecord:
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    annotation_mask = fou.serialize_numpy_array(mask, ascii=True)
    filepath = tmp_path / f"immutable-{index}.jpg"
    filepath.write_bytes(f"source media {index}".encode())
    terminal = TerminalRecord(
        source_id=f"source-{index}",
        filepath=filepath,
        relative_filepath=f"data/immutable-{index}.jpg",
        jaguar_id=f"J{index:02d}",
        bboxes_body=cast(
            FrozenAnnotation,
            {
                "detections": [
                    {
                        "label": "jaguar",
                        "bounding_box": [0.1, 0.2, 0.3, 0.4],
                        "confidence": 0.91,
                    }
                ]
            },
        ),
        segmentations_body=cast(
            FrozenAnnotation,
            {
                "detections": [
                    {
                        "label": "jaguar",
                        "bounding_box": [0.1, 0.2, 0.3, 0.4],
                        "mask": annotation_mask,
                    }
                ]
            },
        ),
    )
    enrichment = Enrichment(
        status=cast(Any, lineage_status),
        match_method=cast(Any, match_method),
        fields=MappingProxyType(
            {
                "closed_set_split": closed_set_split,
                "open_set_split": open_set_split,
                "sighting_id": f"S{index}",
                "site": "Site A",
                "location": "North",
                "camera_id": "CAM-7",
                "camera_side": "left",
                "camera_model": "Model X",
                "latitude": -3.1,
                "longitude": -60.2,
                "capture_date": "2024-01-02",
                "capture_time": "03:04:05",
                "capture_datetime": "2024-01-02 03:04:05",
                "original_filename": f"IMG_{index}.JPG",
                "source_media_path": f"/immutable/raw/IMG_{index}.JPG",
                "source_type": "csv",
                "not_approved": "must never enter the snapshot",
            }
        ),
    )
    return ValidatedRecord(
        terminal=terminal,
        enrichment=enrichment,
        integrity=MediaIntegrity(
            sha256=f"{index:064x}",
            size_bytes=100 + index,
            width=1920,
            height=1080,
        ),
    )


def _six_split_records(tmp_path: Path) -> list[ValidatedRecord]:
    combinations = [
        ("train", "train"),
        ("val", "val"),
        ("test", "test"),
        ("train", "val"),
        ("val", "test"),
        ("test", "train"),
    ]
    return [
        _validated_record(
            tmp_path,
            index=index,
            lineage_status="ambiguous" if index == 1 else "matched",
            match_method=None if index == 1 else "source_id",
            closed_set_split=closed_split,
            open_set_split=open_split,
        )
        for index, (closed_split, open_split) in enumerate(combinations)
    ]


def test_validated_record_maps_only_approved_schema_and_reconstructs_labels(
    tmp_path: Path,
) -> None:
    record = _validated_record(tmp_path)

    sample = validated_record_to_sample(record)

    assert sample.filepath == str(record.terminal.filepath)
    assert sample.jaguar_id == "J00"
    assert isinstance(sample.ground_truth, fo.Classification)
    assert sample.ground_truth.label == "J00"
    assert isinstance(sample.bboxes_body, fo.Detections)
    assert sample.bboxes_body.detections[0].bounding_box == [0.1, 0.2, 0.3, 0.4]
    assert isinstance(sample.segmentations_body, fo.Detections)
    np.testing.assert_array_equal(
        sample.segmentations_body.detections[0].mask,
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
    )
    assert sample.lineage_status == "matched"
    assert sample.lineage_match_method == "source_id"
    assert sample.latitude == -3.1
    assert sample.source_type == "csv"
    assert sample.sha256 == "0" * 64
    assert sample.size_bytes == 100
    assert sample.width == 1920
    assert sample.height == 1080
    assert "not_approved" not in sample.field_names
    assert set(sample.field_names) - {
        "id",
        "filepath",
        "tags",
        "metadata",
        "created_at",
        "last_modified_at",
    } == set(APPROVED_SAMPLE_FIELDS)


def test_snapshot_declares_schema_inserts_persists_and_saves_eight_views(
    tmp_path: Path,
    dataset_names: tuple[str, str],
) -> None:
    final_name, temporary_name = dataset_names
    records = _six_split_records(tmp_path)

    dataset = create_snapshot(records, final_name, temporary_name)

    assert dataset.name == final_name
    assert dataset.persistent is True
    assert fo.dataset_exists(final_name)
    assert not fo.dataset_exists(temporary_name)
    assert len(dataset) == len(records)

    schema = dataset.get_field_schema()
    expected_types = {
        "jaguar_id": fof.StringField,
        "ground_truth": fof.EmbeddedDocumentField,
        "bboxes_body": fof.EmbeddedDocumentField,
        "segmentations_body": fof.EmbeddedDocumentField,
        "lineage_status": fof.StringField,
        "lineage_match_method": fof.StringField,
        "closed_set_split": fof.StringField,
        "open_set_split": fof.StringField,
        "sighting_id": fof.StringField,
        "site": fof.StringField,
        "location": fof.StringField,
        "camera_id": fof.StringField,
        "camera_side": fof.StringField,
        "camera_model": fof.StringField,
        "latitude": fof.FloatField,
        "longitude": fof.FloatField,
        "capture_date": fof.StringField,
        "capture_time": fof.StringField,
        "capture_datetime": fof.StringField,
        "original_filename": fof.StringField,
        "source_media_path": fof.StringField,
        "source_type": fof.StringField,
        "sha256": fof.StringField,
        "size_bytes": fof.IntField,
        "width": fof.IntField,
        "height": fof.IntField,
    }
    assert set(expected_types) == set(APPROVED_SAMPLE_FIELDS)
    for field_name, field_type in expected_types.items():
        assert isinstance(schema[field_name], field_type)
    assert schema["ground_truth"].document_type is fo.Classification
    assert schema["bboxes_body"].document_type is fo.Detections
    assert schema["segmentations_body"].document_type is fo.Detections

    assert dataset.list_saved_views() == list(SAVED_VIEW_NAMES)
    assert len(dataset.load_saved_view("All final samples")) == 6
    assert len(dataset.load_saved_view("Lineage issues")) == 1
    assert len(dataset.load_saved_view("Closed-set train")) == 2
    assert len(dataset.load_saved_view("Closed-set val")) == 2
    assert len(dataset.load_saved_view("Closed-set test")) == 2
    assert len(dataset.load_saved_view("Open-set train")) == 2
    assert len(dataset.load_saved_view("Open-set val")) == 2
    assert len(dataset.load_saved_view("Open-set test")) == 2

    loaded = fo.load_dataset(final_name)
    persisted = loaded[str(dataset.first().id)]
    assert persisted.ground_truth.label == persisted.jaguar_id
    np.testing.assert_array_equal(
        persisted.segmentations_body.detections[0].mask,
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
    )


def test_count_mismatch_removes_both_atomic_names(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    records = _six_split_records(tmp_path)
    original_add_samples = fo.Dataset.add_samples

    def drop_last_sample(dataset: fo.Dataset, samples: list[fo.Sample], **kwargs: Any) -> list[str]:
        return cast(list[str], original_add_samples(dataset, samples[:-1], **kwargs))

    monkeypatch.setattr(fo.Dataset, "add_samples", drop_last_sample)

    with pytest.raises(SnapshotValidationError, match="expected 6 samples, found 5"):
        create_snapshot(records, final_name, temporary_name)

    assert not fo.dataset_exists(final_name)
    assert not fo.dataset_exists(temporary_name)


def test_snapshot_inserts_in_bounded_batches(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    records = [_validated_record(tmp_path, index=index) for index in range(205)]
    original_add_samples = fo.Dataset.add_samples
    batch_sizes: list[int] = []

    def capture_batch_size(dataset: fo.Dataset, samples: list[fo.Sample], **kwargs: Any) -> list[str]:
        batch_sizes.append(len(samples))
        return cast(list[str], original_add_samples(dataset, samples, **kwargs))

    monkeypatch.setattr(fo.Dataset, "add_samples", capture_batch_size)

    create_snapshot(records, final_name, temporary_name)

    assert batch_sizes == [100, 100, 5]


def test_identity_mismatch_removes_both_atomic_names(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    record = _validated_record(tmp_path)
    original_mapper = final_snapshot.validated_record_to_sample

    def mismatched_mapper(validated: ValidatedRecord) -> fo.Sample:
        sample = original_mapper(validated)
        sample.jaguar_id = "WRONG"
        return sample

    monkeypatch.setattr(final_snapshot, "validated_record_to_sample", mismatched_mapper)

    with pytest.raises(SnapshotValidationError, match="identity agreement"):
        create_snapshot([record], final_name, temporary_name)

    assert not fo.dataset_exists(final_name)
    assert not fo.dataset_exists(temporary_name)


def test_validation_failure_after_insert_removes_owned_temporary_dataset(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names

    def fail_validation(dataset: fo.Dataset, records: list[ValidatedRecord]) -> None:
        raise SnapshotValidationError("forced validation failure")

    monkeypatch.setattr(final_snapshot, "_validate_snapshot", fail_validation)

    with pytest.raises(SnapshotValidationError, match="forced validation failure"):
        create_snapshot([_validated_record(tmp_path)], final_name, temporary_name)

    assert not fo.dataset_exists(final_name)
    assert not fo.dataset_exists(temporary_name)


def test_existing_final_dataset_is_untouched(
    tmp_path: Path,
    dataset_names: tuple[str, str],
) -> None:
    final_name, temporary_name = dataset_names
    existing = fo.Dataset(final_name, persistent=True)
    existing.info["owner"] = "someone-else"
    existing.save()

    with pytest.raises(SnapshotCollisionError, match="final dataset already exists"):
        create_snapshot([_validated_record(tmp_path)], final_name, temporary_name)

    reloaded = fo.load_dataset(final_name)
    assert reloaded.info["owner"] == "someone-else"
    assert not fo.dataset_exists(temporary_name)


def test_temporary_name_collision_is_never_deleted_or_renamed(
    tmp_path: Path,
    dataset_names: tuple[str, str],
) -> None:
    final_name, temporary_name = dataset_names
    existing = fo.Dataset(temporary_name, persistent=True)
    existing.info["owner"] = "someone-else"
    existing.save()

    with pytest.raises(SnapshotCollisionError, match="temporary dataset already exists"):
        create_snapshot([_validated_record(tmp_path)], final_name, temporary_name)

    reloaded = fo.load_dataset(temporary_name)
    assert reloaded.info["owner"] == "someone-else"
    assert not fo.dataset_exists(final_name)


def test_final_and_temporary_names_must_be_distinct(
    tmp_path: Path,
    dataset_names: tuple[str, str],
) -> None:
    final_name, _ = dataset_names

    with pytest.raises(SnapshotCollisionError, match="must be distinct"):
        create_snapshot([_validated_record(tmp_path)], final_name, final_name)

    assert not fo.dataset_exists(final_name)
