import hashlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import fiftyone as fo
import fiftyone.core.dataset as fod
import fiftyone.core.fields as fof
import fiftyone.core.odm as foo
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
    create_snapshot as _create_snapshot,
    validated_record_to_sample,
)
from jaguars.visualization.final_validation import MediaIntegrity, ValidatedRecord


def _database_document(name: str) -> dict[str, Any] | None:
    return cast(
        dict[str, Any] | None,
        foo.get_db_conn().datasets.find_one({"name": name}),
    )


def _database_names_with_prefix(prefix: str) -> list[str]:
    return sorted(
        document["name"]
        for document in foo.get_db_conn().datasets.find(
            {"name": {"$regex": f"^{prefix}"}},
            {"name": 1},
        )
    )


def create_snapshot(
    records: list[ValidatedRecord],
    dataset_name: str,
    temporary_name: str,
    *,
    replace_existing: bool = False,
    expected_original_id: object | None = None,
) -> fo.Dataset:
    if replace_existing and expected_original_id is None:
        original = _database_document(dataset_name)
        assert original is not None
        expected_original_id = original["_id"]
    return _create_snapshot(
        records,
        dataset_name,
        temporary_name,
        replace_existing=replace_existing,
        expected_original_id=expected_original_id,
    )


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


def test_constructor_failure_after_metadata_insert_retains_unproven_artifact(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    owned_build_name = f"{temporary_name}--build-constructor-failure"

    def fail_index_creation(sample_collection_name: str, frame_collection_name: str | None) -> None:
        raise RuntimeError("forced index creation failure")

    monkeypatch.setattr(
        final_snapshot,
        "_build_dataset_name",
        lambda temporary_base: owned_build_name,
    )
    monkeypatch.setattr(fod, "_create_indexes", fail_index_creation)

    with pytest.raises(
        SnapshotReplacementError,
        match="ownership could not be proven.*retained",
    ):
        create_snapshot([_validated_record(tmp_path)], final_name, temporary_name)

    assert not fo.dataset_exists(final_name)
    assert not fo.dataset_exists(temporary_name)
    unproven = _database_document(owned_build_name)
    assert unproven is not None
    assert final_snapshot.OWNERSHIP_INFO_KEY not in unproven.get("info", {})
    fo.delete_dataset(owned_build_name)


def test_constructor_value_error_after_metadata_insert_is_not_misclassified_as_collision(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    owned_build_name = f"{temporary_name}--build-constructor-value-error"

    def fail_index_creation(
        sample_collection_name: str,
        frame_collection_name: str | None,
    ) -> None:
        raise ValueError(f"Dataset name '{owned_build_name}' is not available")

    monkeypatch.setattr(
        final_snapshot,
        "_build_dataset_name",
        lambda temporary_base: owned_build_name,
    )
    monkeypatch.setattr(fod, "_create_indexes", fail_index_creation)

    with pytest.raises(
        SnapshotReplacementError,
        match="ownership could not be proven.*retained",
    ):
        create_snapshot([_validated_record(tmp_path)], final_name, temporary_name)

    unproven = _database_document(owned_build_name)
    assert unproven is not None
    assert final_snapshot.OWNERSHIP_INFO_KEY not in unproven.get("info", {})
    fo.delete_dataset(owned_build_name)


def test_generated_build_name_collision_is_never_deleted_or_modified(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    occupied_build_name = f"{temporary_name}--build-deterministic-collision"
    foreign = fo.Dataset(occupied_build_name, persistent=True)
    foreign.info["owner"] = "foreign-generated-name"
    foreign.save()

    monkeypatch.setattr(
        final_snapshot,
        "_build_dataset_name",
        lambda temporary_base: occupied_build_name,
    )

    with pytest.raises(SnapshotCollisionError, match="generated build dataset already exists"):
        create_snapshot([_validated_record(tmp_path)], final_name, temporary_name)

    assert fo.load_dataset(occupied_build_name).info["owner"] == "foreign-generated-name"
    assert not fo.dataset_exists(final_name)
    foreign.delete()


def test_constructor_name_race_never_deletes_unproven_claimant(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    raced_build_name = f"{temporary_name}--build-constructor-race"
    real_dataset_constructor = fo.Dataset
    constructor_calls = 0

    def racing_constructor(name: str, *args: Any, **kwargs: Any) -> fo.Dataset:
        nonlocal constructor_calls
        constructor_calls += 1
        if constructor_calls == 1:
            foreign = real_dataset_constructor(name, persistent=True)
            foreign.info["owner"] = "foreign-racing-constructor"
            foreign.save()
        return real_dataset_constructor(name, *args, **kwargs)

    monkeypatch.setattr(
        final_snapshot,
        "_build_dataset_name",
        lambda temporary_base: raced_build_name,
    )
    monkeypatch.setattr(final_snapshot.fo, "Dataset", racing_constructor)

    with pytest.raises(
        SnapshotReplacementError,
        match="ownership could not be proven.*retained",
    ):
        create_snapshot([_validated_record(tmp_path)], final_name, temporary_name)

    assert fo.load_dataset(raced_build_name).info["owner"] == "foreign-racing-constructor"
    assert not fo.dataset_exists(final_name)
    fo.delete_dataset(raced_build_name)


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


def test_replacement_aborts_if_confirmed_final_is_replaced_during_staging(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    displaced_name = f"{temporary_name}--foreign-displaced-original"
    real_validate = final_snapshot._validate_snapshot

    def replace_confirmed_final(
        dataset: fo.Dataset,
        records: list[ValidatedRecord],
    ) -> None:
        real_validate(dataset, records)
        original.name = displaced_name
        foreign = fo.Dataset(final_name, persistent=True)
        foreign.info["owner"] = "foreign-final-racer"
        foreign.save()

    monkeypatch.setattr(
        final_snapshot,
        "_validate_snapshot",
        replace_confirmed_final,
    )

    with pytest.raises(
        SnapshotReplacementError,
        match="confirmed final dataset changed before promotion",
    ):
        _create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
            expected_original_id=original_id,
        )

    assert fo.load_dataset(final_name).info["owner"] == "foreign-final-racer"
    assert fo.load_dataset(displaced_name).info["generation"] == "old"
    assert _database_names_with_prefix(f"{temporary_name}--build-") == []

    fo.delete_dataset(final_name)
    fo.delete_dataset(displaced_name)


def test_replacement_aborts_if_confirmed_final_disappears_during_staging(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_validate = final_snapshot._validate_snapshot

    def delete_confirmed_final(
        dataset: fo.Dataset,
        records: list[ValidatedRecord],
    ) -> None:
        real_validate(dataset, records)
        original.delete()

    monkeypatch.setattr(
        final_snapshot,
        "_validate_snapshot",
        delete_confirmed_final,
    )

    with pytest.raises(
        SnapshotReplacementError,
        match="confirmed final dataset disappeared before promotion",
    ):
        _create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
            expected_original_id=original_id,
        )

    assert not fo.dataset_exists(final_name)
    assert _database_names_with_prefix(f"{temporary_name}--") == []


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


def test_transactional_replacement_first_rename_failure_after_database_save_restores_old_final(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_rename = final_snapshot._rename_dataset
    calls = 0

    def save_then_fail_with_stale_name(dataset: fo.Dataset, name: str) -> None:
        nonlocal calls
        calls += 1
        old_name = dataset.name
        old_slug = dataset.slug
        real_rename(dataset, name)
        if calls == 1:
            dataset._doc.name = old_name
            dataset._doc.slug = old_slug
            raise RuntimeError("first rename failed after database save")

    monkeypatch.setattr(
        final_snapshot,
        "_rename_dataset",
        save_then_fail_with_stale_name,
    )

    with pytest.raises(RuntimeError, match="first rename failed after database save"):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    final_document = _database_document(final_name)
    assert final_document is not None
    assert final_document["_id"] == original_id
    assert final_document["info"]["generation"] == "old"
    assert _database_names_with_prefix(f"{temporary_name}--") == []


def test_persisted_old_rename_with_transient_recovery_queries_restores_final(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_rename = final_snapshot._rename_dataset
    real_query_by_id = final_snapshot._database_document_by_id
    query_failures = 0
    rename_calls = 0

    def persist_then_raise(dataset: fo.Dataset, name: str) -> None:
        nonlocal rename_calls
        rename_calls += 1
        real_rename(dataset, name)
        if rename_calls == 1:
            raise RuntimeError("old rename persisted then raised")

    def transient_recovery_query(dataset_id: object) -> dict[str, object] | None:
        nonlocal query_failures
        if dataset_id == original_id and query_failures < 2:
            query_failures += 1
            raise RuntimeError("transient recovery query")
        return real_query_by_id(dataset_id)

    monkeypatch.setattr(final_snapshot, "_rename_dataset", persist_then_raise)
    monkeypatch.setattr(
        final_snapshot,
        "_database_document_by_id",
        transient_recovery_query,
    )

    with pytest.raises(RuntimeError, match="old rename persisted then raised"):
        _create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
            expected_original_id=original_id,
        )

    final_document = _database_document(final_name)
    assert final_document is not None
    assert final_document["_id"] == original_id
    assert final_document["info"]["generation"] == "old"
    assert _database_names_with_prefix(f"{temporary_name}--") == []

    fo.delete_dataset(final_name)


def test_transactional_replacement_promotion_failure_after_database_save_restores_old_final(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_rename = final_snapshot._rename_dataset
    calls = 0

    def save_then_fail_with_stale_name(dataset: fo.Dataset, name: str) -> None:
        nonlocal calls
        calls += 1
        old_name = dataset.name
        old_slug = dataset.slug
        real_rename(dataset, name)
        if calls == 2:
            dataset._doc.name = old_name
            dataset._doc.slug = old_slug
            raise RuntimeError("promotion failed after database save")

    monkeypatch.setattr(
        final_snapshot,
        "_rename_dataset",
        save_then_fail_with_stale_name,
    )

    with pytest.raises(RuntimeError, match="promotion failed after database save"):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    final_document = _database_document(final_name)
    assert final_document is not None
    assert final_document["_id"] == original_id
    assert final_document["info"]["generation"] == "old"
    assert _database_names_with_prefix(f"{temporary_name}--") == []


def test_transient_publication_query_failure_recovers_without_generic_cleanup(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_query = final_snapshot._database_document_by_name
    failed_once = False

    def fail_once_after_owned_publication(name: str) -> dict[str, Any] | None:
        nonlocal failed_once
        document = real_query(name)
        info = None if document is None else document.get("info")
        if name == final_name and isinstance(info, dict) and final_snapshot.OWNERSHIP_INFO_KEY in info and not failed_once:
            failed_once = True
            raise RuntimeError("transient publication query failure")
        return document

    monkeypatch.setattr(
        final_snapshot,
        "_database_document_by_name",
        fail_once_after_owned_publication,
    )

    replacement = _create_snapshot(
        [_validated_record(tmp_path)],
        final_name,
        temporary_name,
        replace_existing=True,
        expected_original_id=original_id,
    )

    assert failed_once is True
    assert len(replacement) == 1
    assert replacement.first().jaguar_id == "J00"
    assert _database_names_with_prefix(f"{temporary_name}--") == []


@pytest.mark.parametrize("delete_timing", ["before", "after"])
def test_promotion_recovery_handles_owned_new_delete_exception_without_empty_final(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    delete_timing: str,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_rename = final_snapshot._rename_dataset
    real_delete = fo.Dataset.delete
    rename_calls = 0
    failed_delete_once = False

    def persist_promotion_then_fail(dataset: fo.Dataset, name: str) -> None:
        nonlocal rename_calls
        rename_calls += 1
        real_rename(dataset, name)
        if rename_calls == 2:
            raise RuntimeError("promotion raised after database save")

    def fail_owned_new_delete_once(dataset: fo.Dataset) -> None:
        nonlocal failed_delete_once
        if dataset._doc.id != original_id and not failed_delete_once:
            failed_delete_once = True
            if delete_timing == "after":
                real_delete(dataset)
            raise RuntimeError(f"owned new delete failed {delete_timing} effect")
        real_delete(dataset)

    monkeypatch.setattr(
        final_snapshot,
        "_rename_dataset",
        persist_promotion_then_fail,
    )
    monkeypatch.setattr(fo.Dataset, "delete", fail_owned_new_delete_once)

    with pytest.raises(RuntimeError, match="promotion raised after database save"):
        _create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
            expected_original_id=original_id,
        )

    restored = fo.load_dataset(final_name, reload=True)
    assert restored._doc.id == original_id
    assert restored.info["generation"] == "old"
    assert failed_delete_once is True
    assert _database_names_with_prefix(f"{temporary_name}--") == []


def test_transactional_replacement_never_deletes_foreign_final_claimed_during_promotion(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_rename = final_snapshot._rename_dataset
    calls = 0

    def foreign_claim_then_fail(dataset: fo.Dataset, name: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            foreign = fo.Dataset(final_name, persistent=True)
            foreign.info["owner"] = "foreign-promotion-racer"
            foreign.save()
            raise RuntimeError("promotion lost name race")
        real_rename(dataset, name)

    monkeypatch.setattr(
        final_snapshot,
        "_rename_dataset",
        foreign_claim_then_fail,
    )

    with pytest.raises(
        SnapshotReplacementError,
        match="foreign dataset occupies final name.*old dataset remains",
    ):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    foreign_document = _database_document(final_name)
    assert foreign_document is not None
    assert foreign_document["info"]["owner"] == "foreign-promotion-racer"
    backup_names = _database_names_with_prefix(f"{temporary_name}--backup-")
    assert len(backup_names) == 1
    backup_document = _database_document(backup_names[0])
    assert backup_document is not None
    assert backup_document["_id"] == original_id
    assert backup_document["info"]["generation"] == "old"
    assert _database_names_with_prefix(f"{temporary_name}--build-") == []

    fo.delete_dataset(final_name)
    fo.delete_dataset(backup_names[0])


def test_replacement_rechecks_loaded_original_identity_before_rename(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    displaced_name = f"{temporary_name}--pinned-original"
    real_load_dataset = fo.load_dataset
    raced = False

    def race_during_load(name: str, *args: object, **kwargs: object) -> fo.Dataset:
        nonlocal raced
        if name == final_name and kwargs.get("reload") is True and not raced:
            raced = True
            original.name = displaced_name
            foreign = fo.Dataset(final_name, persistent=True)
            foreign.info["owner"] = "foreign-load-racer"
            foreign.save()
        return real_load_dataset(name, *args, **kwargs)

    monkeypatch.setattr(final_snapshot.fo, "load_dataset", race_during_load)

    with pytest.raises(
        SnapshotReplacementError,
        match="changed while loading for promotion",
    ):
        _create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
            expected_original_id=original_id,
        )

    assert real_load_dataset(final_name).info["owner"] == "foreign-load-racer"
    assert real_load_dataset(displaced_name).info["generation"] == "old"
    assert _database_names_with_prefix(f"{temporary_name}--build-") == []

    fo.delete_dataset(final_name)
    fo.delete_dataset(displaced_name)


def test_query_failure_immediately_after_old_rename_restores_final(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_query_by_id = final_snapshot._database_document_by_id
    failures = 0

    def fail_post_rename_query(dataset_id: object) -> dict[str, object] | None:
        nonlocal failures
        document = real_query_by_id(dataset_id)
        if (
            dataset_id == original_id
            and document is not None
            and str(document.get("name", "")).startswith(f"{temporary_name}--backup-")
            and failures < 2
        ):
            failures += 1
            raise RuntimeError("post-rename query unavailable")
        return document

    monkeypatch.setattr(
        final_snapshot,
        "_database_document_by_id",
        fail_post_rename_query,
    )

    replacement = _create_snapshot(
        [_validated_record(tmp_path)],
        final_name,
        temporary_name,
        replace_existing=True,
        expected_original_id=original_id,
    )

    published = fo.load_dataset(final_name)
    assert published._doc.id == replacement._doc.id
    assert published._doc.id != original_id
    assert _database_names_with_prefix(f"{temporary_name}--") == []

    fo.delete_dataset(final_name)


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


def test_backup_delete_failure_before_effect_retains_valid_new_final_and_old_backup(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_delete = fo.Dataset.delete

    def fail_old_backup_delete(dataset: fo.Dataset) -> None:
        if dataset._doc.id == original_id:
            raise RuntimeError("old backup delete failed before effect")
        real_delete(dataset)

    monkeypatch.setattr(fo.Dataset, "delete", fail_old_backup_delete)

    with pytest.raises(
        SnapshotReplacementError,
        match="replacement remains published.*old backup remains recoverable",
    ):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    new_final = fo.load_dataset(final_name, reload=True)
    assert new_final._doc.id != original_id
    assert len(new_final) == 1
    assert new_final.first().jaguar_id == "J00"
    assert isinstance(
        new_final.info[final_snapshot.OWNERSHIP_INFO_KEY],
        str,
    )

    backup_names = _database_names_with_prefix(f"{temporary_name}--backup-")
    assert len(backup_names) == 1
    backup_document = _database_document(backup_names[0])
    assert backup_document is not None
    assert backup_document["_id"] == original_id
    assert backup_document["info"]["generation"] == "old"
    assert _database_names_with_prefix(f"{temporary_name}--build-") == []

    real_delete(new_final)
    real_delete(fo.load_dataset(backup_names[0], reload=True))


def test_backup_delete_failure_after_effect_retains_valid_new_final_without_old_backup(
    tmp_path: Path,
    dataset_names: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name, temporary_name = dataset_names
    original, _, _ = _existing_snapshot(tmp_path, final_name)
    original_id = original._doc.id
    real_delete = fo.Dataset.delete

    def delete_old_backup_then_fail(dataset: fo.Dataset) -> None:
        real_delete(dataset)
        if dataset._doc.id == original_id:
            raise RuntimeError("old backup delete failed after effect")

    monkeypatch.setattr(fo.Dataset, "delete", delete_old_backup_then_fail)

    with pytest.raises(
        SnapshotReplacementError,
        match="replacement remains published.*old backup was already removed",
    ):
        create_snapshot(
            [_validated_record(tmp_path)],
            final_name,
            temporary_name,
            replace_existing=True,
        )

    new_final = fo.load_dataset(final_name, reload=True)
    assert new_final._doc.id != original_id
    assert len(new_final) == 1
    assert new_final.first().jaguar_id == "J00"
    assert isinstance(
        new_final.info[final_snapshot.OWNERSHIP_INFO_KEY],
        str,
    )
    assert _database_document(temporary_name) is None
    assert _database_names_with_prefix(f"{temporary_name}--") == []

    real_delete(new_final)


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
