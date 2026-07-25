import hashlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import fiftyone as fo
import fiftyone.core.dataset as fod
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
    SnapshotError,
    SnapshotReplacementError,
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


def _with_enrichment_fields(record: ValidatedRecord, fields: dict[str, object]) -> ValidatedRecord:
    return replace(
        record,
        enrichment=replace(
            record.enrichment,
            fields=MappingProxyType(cast(dict[str, Any], fields)),
        ),
    )


def _with_segmentation_mask(record: ValidatedRecord, mask: object) -> ValidatedRecord:
    terminal = replace(
        record.terminal,
        segmentations_body=cast(
            FrozenAnnotation,
            {
                "detections": [
                    {
                        "label": "jaguar",
                        "bounding_box": [0.1, 0.2, 0.3, 0.4],
                        "mask": mask,
                    }
                ]
            },
        ),
    )
    return replace(record, terminal=terminal)


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


def test_sample_identity_is_optional_and_uses_only_resolved_lineage(
    tmp_path: Path,
) -> None:
    record = _validated_record(tmp_path)
    missing_terminal = replace(record.terminal, jaguar_id=None)
    unresolved = replace(record, terminal=missing_terminal)
    resolved = _with_enrichment_fields(
        unresolved,
        {**record.enrichment.fields, "jaguar_id": "F11"},
    )

    unresolved_sample = validated_record_to_sample(unresolved)
    resolved_sample = validated_record_to_sample(resolved)

    assert unresolved_sample.jaguar_id is None
    assert unresolved_sample.ground_truth is None
    assert resolved_sample.jaguar_id == "F11"
    assert isinstance(resolved_sample.ground_truth, fo.Classification)
    assert resolved_sample.ground_truth.label == "F11"


def test_snapshot_preserves_sample_with_unresolved_identity(
    tmp_path: Path,
    dataset_names: tuple[str, str],
) -> None:
    final_name, temporary_name = dataset_names
    record = _validated_record(tmp_path)
    unresolved = replace(record, terminal=replace(record.terminal, jaguar_id=None))

    dataset = create_snapshot([unresolved], final_name, temporary_name)

    sample = dataset.first()
    assert sample.jaguar_id is None
    assert sample.ground_truth is None
    assert len(dataset) == 1


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


@pytest.mark.parametrize("explicit_none", [False, True])
def test_absent_split_values_are_allowed_and_excluded_from_split_views(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    explicit_none: bool,
) -> None:
    final_name, temporary_name = dataset_names
    record = _validated_record(tmp_path)
    fields = dict(record.enrichment.fields)
    if explicit_none:
        fields["closed_set_split"] = None
        fields["open_set_split"] = None
    else:
        fields.pop("closed_set_split")
        fields.pop("open_set_split")

    dataset = create_snapshot(
        [_with_enrichment_fields(record, fields)],
        final_name,
        temporary_name,
    )

    sample = dataset.first()
    assert sample.closed_set_split is None
    assert sample.open_set_split is None
    for view_name in SAVED_VIEW_NAMES[2:]:
        assert len(dataset.load_saved_view(view_name)) == 0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("closed_set_split", "development"),
        ("open_set_split", 7),
    ],
)
def test_populated_split_values_must_be_approved_strings(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    field_name: str,
    invalid_value: object,
) -> None:
    final_name, temporary_name = dataset_names
    record = _validated_record(tmp_path)
    fields = dict(record.enrichment.fields)
    fields[field_name] = invalid_value

    with pytest.raises(SnapshotValidationError, match=field_name):
        create_snapshot(
            [_with_enrichment_fields(record, fields)],
            final_name,
            temporary_name,
        )

    assert not fo.dataset_exists(final_name)
    assert not fo.dataset_exists(temporary_name)


@pytest.mark.parametrize(
    ("mask_payload", "expected_message"),
    [
        ("not-a-serialized-mask", "could not decode"),
        (
            fou.serialize_numpy_array(np.ones((2, 2, 1), dtype=np.uint8), ascii=True),
            "two-dimensional",
        ),
        (
            fou.serialize_numpy_array(np.ones((2, 2), dtype=np.float32), ascii=True),
            "boolean or integer dtype",
        ),
        (
            fou.serialize_numpy_array(np.array([[0, 2], [1, 0]], dtype=np.uint8), ascii=True),
            "binary values",
        ),
    ],
)
def test_invalid_serialized_masks_fail_cleanly_before_publication(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    mask_payload: str,
    expected_message: str,
) -> None:
    final_name, temporary_name = dataset_names
    record = _with_segmentation_mask(_validated_record(tmp_path), mask_payload)

    with pytest.raises(
        SnapshotError,
        match=rf"segmentations_body.*mask.*{expected_message}",
    ):
        create_snapshot([record], final_name, temporary_name)

    assert not fo.dataset_exists(final_name)
    assert not fo.dataset_exists(temporary_name)


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


def test_constructor_failure_after_metadata_insert_removes_temporary_dataset(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names

    def fail_index_creation(sample_collection_name: str, frame_collection_name: str | None) -> None:
        raise RuntimeError("forced index creation failure")

    monkeypatch.setattr(fod, "_create_indexes", fail_index_creation)

    with pytest.raises(RuntimeError, match="forced index creation failure"):
        create_snapshot([_validated_record(tmp_path)], final_name, temporary_name)

    assert not fo.dataset_exists(final_name)
    assert not fo.dataset_exists(temporary_name)


def test_racing_temporary_base_claimant_survives_owned_constructor_failure(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    owned_build_name = f"{temporary_name}--build-deterministic-owner-token"
    original_refuse_collisions = final_snapshot._refuse_collisions
    original_create_indexes = fod._create_indexes
    index_calls = 0

    def claim_base_after_preflight(dataset_name: str, temporary_base: str) -> None:
        original_refuse_collisions(dataset_name, temporary_base)
        claimant = fo.Dataset(temporary_base, persistent=True)
        claimant.info["owner"] = "racing-caller"
        claimant.save()

    def fail_owned_index_creation(sample_collection_name: str, frame_collection_name: str | None) -> None:
        nonlocal index_calls
        index_calls += 1
        if index_calls == 1:
            original_create_indexes(sample_collection_name, frame_collection_name)
            return
        raise RuntimeError("forced owned build index failure")

    monkeypatch.setattr(final_snapshot, "_refuse_collisions", claim_base_after_preflight)
    monkeypatch.setattr(
        final_snapshot,
        "_build_dataset_name",
        lambda temporary_base: owned_build_name,
        raising=False,
    )
    monkeypatch.setattr(fod, "_create_indexes", fail_owned_index_creation)

    with pytest.raises(RuntimeError, match="forced owned build index failure"):
        create_snapshot([_validated_record(tmp_path)], final_name, temporary_name)

    claimant = fo.load_dataset(temporary_name)
    assert claimant.info["owner"] == "racing-caller"
    assert not fo.dataset_exists(owned_build_name)
    assert not fo.dataset_exists(final_name)


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


def test_validation_failure_after_insert_removes_only_owned_temporary_dataset(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    record = _validated_record(tmp_path)
    source_path = record.terminal.filepath
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    def fail_validation(dataset: fo.Dataset, records: list[ValidatedRecord]) -> None:
        raise SnapshotValidationError("forced validation failure")

    monkeypatch.setattr(final_snapshot, "_validate_snapshot", fail_validation)

    with pytest.raises(SnapshotValidationError, match="forced validation failure"):
        create_snapshot([record], final_name, temporary_name)

    assert not fo.dataset_exists(final_name)
    assert not fo.dataset_exists(temporary_name)
    assert source_path.is_file()
    assert source_path.read_bytes() == source_bytes
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_sha256


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


def _existing_snapshot(
    tmp_path: Path,
    dataset_name: str,
) -> tuple[fo.Dataset, Path, bytes]:
    media_path = tmp_path / "old-source.jpg"
    media_bytes = b"old immutable source media"
    media_path.write_bytes(media_bytes)
    dataset = fo.Dataset(dataset_name, persistent=True)
    dataset.info["generation"] = "old"
    dataset.add_sample(fo.Sample(filepath=str(media_path), generation="old"))
    dataset.save()
    return dataset, media_path, media_bytes


def test_transactional_replacement_build_failure_preserves_old_final(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    _existing_snapshot(tmp_path, final_name)

    def fail_validation(dataset: fo.Dataset, records: list[ValidatedRecord]) -> None:
        raise SnapshotValidationError("replacement validation failed")

    monkeypatch.setattr(final_snapshot, "_validate_snapshot", fail_validation)

    with pytest.raises(SnapshotValidationError, match="replacement validation failed"):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    assert fo.load_dataset(final_name).info["generation"] == "old"
    assert not any(name.startswith(f"{temporary_name}--") for name in fo.list_datasets())


def test_transactional_replacement_first_rename_failure_preserves_old_final(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    _existing_snapshot(tmp_path, final_name)

    def fail_first_rename(dataset: fo.Dataset, name: str) -> None:
        raise RuntimeError("first rename failed")

    monkeypatch.setattr(final_snapshot, "_rename_dataset", fail_first_rename)

    with pytest.raises(RuntimeError, match="first rename failed"):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    assert fo.load_dataset(final_name).info["generation"] == "old"
    assert not any(name.startswith(f"{temporary_name}--") for name in fo.list_datasets())


def test_transactional_replacement_second_rename_failure_rolls_back_old_final(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    _existing_snapshot(tmp_path, final_name)
    original_rename = final_snapshot._rename_dataset
    calls = 0

    def fail_promotion(dataset: fo.Dataset, name: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second rename failed")
        original_rename(dataset, name)

    monkeypatch.setattr(final_snapshot, "_rename_dataset", fail_promotion)

    with pytest.raises(RuntimeError, match="second rename failed"):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    assert calls == 3
    assert fo.load_dataset(final_name).info["generation"] == "old"
    assert not any(name.startswith(f"{temporary_name}--") for name in fo.list_datasets())


def test_transactional_replacement_rollback_failure_keeps_owned_backup(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    _existing_snapshot(tmp_path, final_name)
    original_rename = final_snapshot._rename_dataset
    calls = 0

    def fail_promotion_and_rollback(dataset: fo.Dataset, name: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_rename(dataset, name)
            return
        raise RuntimeError(f"rename {calls} failed")

    monkeypatch.setattr(
        final_snapshot,
        "_rename_dataset",
        fail_promotion_and_rollback,
    )

    with pytest.raises(SnapshotReplacementError, match="rollback failed"):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    assert not fo.dataset_exists(final_name)
    backup_names = [name for name in fo.list_datasets() if name.startswith(f"{temporary_name}--backup-")]
    assert len(backup_names) == 1
    assert fo.load_dataset(backup_names[0]).info["generation"] == "old"
    assert not any("--build-" in name for name in fo.list_datasets())


def test_transactional_replacement_swaps_records_without_deleting_media_or_foreign_datasets(
    tmp_path: Path,
    dataset_names: tuple[str, str],
) -> None:
    final_name, temporary_name = dataset_names
    _, old_media_path, old_media_bytes = _existing_snapshot(tmp_path, final_name)
    foreign = fo.Dataset(temporary_name, persistent=True)
    foreign.info["owner"] = "foreign"
    foreign.save()

    with pytest.raises(SnapshotCollisionError, match="temporary dataset already exists"):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    assert fo.load_dataset(final_name).info["generation"] == "old"
    assert fo.load_dataset(temporary_name).info["owner"] == "foreign"
    foreign.delete()

    replacement = create_snapshot(
        [_validated_record(tmp_path)],
        final_name,
        temporary_name,
        replace_existing=True,
    )

    assert replacement.name == final_name
    assert len(replacement) == 1
    assert replacement.first().jaguar_id == "J00"
    assert old_media_path.read_bytes() == old_media_bytes
    assert not any(name.startswith(f"{temporary_name}--") for name in fo.list_datasets())
