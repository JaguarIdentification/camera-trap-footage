import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from jaguars.visualization.final_records import (
    TerminalExportError,
    load_terminal_records,
)


def _load_payload(export_dir: Path) -> dict[str, object]:
    return json.loads((export_dir / "samples.json").read_text(encoding="utf-8"))


def _write_payload(export_dir: Path, payload: dict[str, object]) -> None:
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
    assert records[0].bboxes_body["detections"][0]["label"] == "jaguar"
    assert records[0].segmentations_body["detections"][0]["mask"]


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
