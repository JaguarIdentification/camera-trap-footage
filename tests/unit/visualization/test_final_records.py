import json
from collections.abc import Mapping, MutableMapping, MutableSequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from jaguars.visualization.final_records import (
    FrozenJsonValue,
    TerminalExportError,
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
    record = load_terminal_records(terminal_export)[0]

    with pytest.raises(FrozenInstanceError):
        record.jaguar_id = "changed"  # type: ignore[misc]
    assert not hasattr(record, "_dataset_id")
    assert not hasattr(record, "_rand")
    assert not hasattr(record, "tags")
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


@pytest.mark.parametrize("jaguar_id", [None, "", "   "])
def test_load_terminal_records_rejects_missing_or_empty_identity(
    terminal_export: Path,
    jaguar_id: object,
) -> None:
    payload = _load_payload(terminal_export)
    sample = payload["samples"][0]
    if jaguar_id is None:
        del sample["jaguar_id"]
    else:
        sample["jaguar_id"] = jaguar_id
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
