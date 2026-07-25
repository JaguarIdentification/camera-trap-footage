import json
from collections.abc import Mapping, MutableMapping, MutableSequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from jaguars.visualization.final_records import (
    FrozenAnnotation,
    FrozenJsonValue,
    TerminalExportError,
    TerminalRecord,
    annotation_to_dict,
    load_terminal_records,
)


def _load_payload(export_dir: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((export_dir / "samples.json").read_text(encoding="utf-8")),
    )


def _write_payload(export_dir: Path, payload: dict[str, Any]) -> None:
    (export_dir / "samples.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_terminal_records_resolves_paths_ids_and_annotations(
    terminal_export: Path,
) -> None:
    records = load_terminal_records(terminal_export)

    assert [record.relative_filepath for record in records] == [
        "data/a.jpg",
        "data/b.jpg",
    ]
    assert [record.filepath for record in records] == [
        (terminal_export / "data/a.jpg").resolve(),
        (terminal_export / "data/b.jpg").resolve(),
    ]
    assert [record.source_id for record in records] == ["source-a", "source-b"]
    assert records[0].jaguar_id == "F11"
    bboxes = cast(
        tuple[Mapping[str, FrozenJsonValue], ...],
        records[0].bboxes_body["detections"],
    )
    segmentations = cast(
        tuple[Mapping[str, FrozenJsonValue], ...],
        records[0].segmentations_body["detections"],
    )
    assert bboxes[0]["label"] == "jaguar"
    assert segmentations[0]["mask"]


def test_terminal_records_are_frozen_and_drop_transient_fields(
    terminal_export: Path,
) -> None:
    records = load_terminal_records(terminal_export)
    record = records[0]

    with pytest.raises(FrozenInstanceError):
        record.jaguar_id = "changed"  # type: ignore[misc]
    assert not hasattr(record, "_dataset_id")
    assert not hasattr(record, "_rand")
    assert record.tags == ()
    assert records[1].tags == ()
    assert not hasattr(record, "created_at")


def test_terminal_record_annotations_are_deeply_immutable(
    terminal_export: Path,
) -> None:
    record = load_terminal_records(terminal_export)[0]
    detections = cast(
        tuple[Mapping[str, FrozenJsonValue], ...],
        record.bboxes_body["detections"],
    )
    mutable_detection = cast(
        MutableMapping[str, FrozenJsonValue],
        detections[0],
    )
    bounding_box = cast(tuple[FrozenJsonValue, ...], detections[0]["bounding_box"])
    mutable_bounding_box = cast(MutableSequence[FrozenJsonValue], bounding_box)

    assert isinstance(detections, tuple)
    assert isinstance(bounding_box, tuple)
    with pytest.raises(TypeError):
        mutable_detection["label"] = "leopard"
    with pytest.raises(TypeError):
        mutable_bounding_box[0] = 0.5


def test_direct_terminal_record_construction_freezes_annotation_aliases(
    tmp_path: Path,
) -> None:
    annotations: dict[str, Any] = {
        "detections": [
            {
                "label": "jaguar",
                "bounding_box": [0.1, 0.2, 0.3, 0.4],
            }
        ]
    }
    record = TerminalRecord(
        source_id="source",
        filepath=tmp_path / "image.jpg",
        relative_filepath="data/image.jpg",
        jaguar_id="F11",
        bboxes_body=cast(FrozenAnnotation, annotations),
        segmentations_body=cast(FrozenAnnotation, annotations),
    )

    annotations["detections"][0]["label"] = "leopard"
    annotations["detections"].append({"label": "another"})

    detections = cast(
        tuple[Mapping[str, FrozenJsonValue], ...],
        record.bboxes_body["detections"],
    )
    assert len(detections) == 1
    assert detections[0]["label"] == "jaguar"
    with pytest.raises(TypeError):
        cast(MutableMapping[str, FrozenJsonValue], detections[0])["label"] = "puma"


def test_annotation_to_dict_returns_independent_mutable_data(
    terminal_export: Path,
) -> None:
    record = load_terminal_records(terminal_export)[0]

    mutable = annotation_to_dict(record.bboxes_body)
    mutable["detections"][0]["label"] = "leopard"

    detections = cast(
        tuple[Mapping[str, FrozenJsonValue], ...],
        record.bboxes_body["detections"],
    )
    assert detections[0]["label"] == "jaguar"


@pytest.mark.parametrize("missing_kind", ["absent", "null"])
def test_load_terminal_records_preserves_missing_identity_as_none(
    terminal_export: Path,
    missing_kind: str,
) -> None:
    payload = _load_payload(terminal_export)
    sample = payload["samples"][0]
    if missing_kind == "absent":
        del sample["jaguar_id"]
    else:
        sample["jaguar_id"] = None
    _write_payload(terminal_export, payload)

    records = load_terminal_records(terminal_export)

    assert records[1].jaguar_id is None


@pytest.mark.parametrize("jaguar_id", ["", "   ", 7, [], {}])
def test_load_terminal_records_rejects_invalid_populated_identity(
    terminal_export: Path,
    jaguar_id: object,
) -> None:
    payload = _load_payload(terminal_export)
    payload["samples"][0]["jaguar_id"] = jaguar_id
    _write_payload(terminal_export, payload)

    with pytest.raises(TerminalExportError, match="jaguar_id"):
        load_terminal_records(terminal_export)


@pytest.mark.parametrize(
    "unsafe_filepath",
    ["../outside.jpg", "data/../../outside.jpg", "/tmp/outside.jpg"],
)
def test_load_terminal_records_rejects_paths_outside_export(
    terminal_export: Path,
    unsafe_filepath: str,
) -> None:
    payload = _load_payload(terminal_export)
    payload["samples"][0]["filepath"] = unsafe_filepath
    _write_payload(terminal_export, payload)

    with pytest.raises(TerminalExportError, match="below export directory"):
        load_terminal_records(terminal_export)


def test_load_terminal_records_rejects_export_root_as_media(
    terminal_export: Path,
) -> None:
    payload = _load_payload(terminal_export)
    payload["samples"][0]["filepath"] = "."
    _write_payload(terminal_export, payload)

    with pytest.raises(TerminalExportError, match="below export directory"):
        load_terminal_records(terminal_export)


def test_load_terminal_records_normalizes_embedded_nul_path_error(
    terminal_export: Path,
) -> None:
    payload = _load_payload(terminal_export)
    payload["samples"][0]["filepath"] = "data/\0.jpg"
    _write_payload(terminal_export, payload)

    with pytest.raises(TerminalExportError, match="filepath"):
        load_terminal_records(terminal_export)


def test_load_terminal_records_normalizes_invalid_utf8_error(
    terminal_export: Path,
) -> None:
    (terminal_export / "samples.json").write_bytes(b"\xff")

    with pytest.raises(TerminalExportError, match="could not read"):
        load_terminal_records(terminal_export)


@pytest.mark.parametrize("field", ["bboxes_body", "segmentations_body"])
@pytest.mark.parametrize(
    "invalid_value",
    [None, [], {}, {"detections": "not-a-list"}, {"detections": []}],
)
def test_load_terminal_records_rejects_malformed_or_empty_annotations(
    terminal_export: Path,
    field: str,
    invalid_value: object,
) -> None:
    payload = _load_payload(terminal_export)
    sample = payload["samples"][0]
    if invalid_value is None:
        del sample[field]
    else:
        sample[field] = invalid_value
    _write_payload(terminal_export, payload)

    with pytest.raises(TerminalExportError, match=field):
        load_terminal_records(terminal_export)


def test_load_terminal_records_accepts_media_only_pending_review_sample(
    terminal_export: Path,
) -> None:
    payload = _load_payload(terminal_export)
    sample = payload["samples"][0]
    del sample["bboxes_body"]
    del sample["segmentations_body"]
    sample["tags"] = ["needs_annotation_review"]
    sample["review_required"] = True
    sample["review_reason"] = "zero-area mask"
    sample["review_status"] = "pending"
    _write_payload(terminal_export, payload)

    record = load_terminal_records(terminal_export)[1]

    assert record.bboxes_body is None
    assert record.segmentations_body is None
    assert record.review_required is True
    assert record.review_reason == "zero-area mask"
    assert record.review_status == "pending"
    assert record.tags == ("needs_annotation_review",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("review_reason", ""),
        ("review_status", "complete"),
        ("tags", []),
    ],
)
def test_load_terminal_records_rejects_incomplete_review_contract(
    terminal_export: Path,
    field: str,
    value: object,
) -> None:
    payload = _load_payload(terminal_export)
    sample = payload["samples"][0]
    del sample["bboxes_body"]
    del sample["segmentations_body"]
    sample.update(
        {
            "tags": ["needs_annotation_review"],
            "review_required": True,
            "review_reason": "zero-area mask",
            "review_status": "pending",
        }
    )
    sample[field] = value
    _write_payload(terminal_export, payload)

    with pytest.raises(TerminalExportError, match="review"):
        load_terminal_records(terminal_export)
