import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TerminalExportError(ValueError):
    """Raised when a terminal FiftyOne export is malformed or unsafe."""


@dataclass(frozen=True)
class TerminalRecord:
    source_id: str
    filepath: Path
    relative_filepath: str
    jaguar_id: str
    bboxes_body: dict[str, Any]
    segmentations_body: dict[str, Any]


def _source_id(sample: dict[str, Any]) -> str:
    raw_id = sample.get("_id")
    if not isinstance(raw_id, dict):
        raise TerminalExportError("sample _id must contain a MongoDB $oid")

    oid = raw_id.get("$oid")
    if not isinstance(oid, str) or not oid.strip():
        raise TerminalExportError("sample _id must contain a nonempty MongoDB $oid")
    return oid


def _annotation_container(
    sample: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    container = sample.get(field)
    if not isinstance(container, dict):
        raise TerminalExportError(f"{field} must be an annotation object")

    detections = container.get("detections")
    if not isinstance(detections, list) or not detections:
        raise TerminalExportError(f"{field}.detections must be a nonempty list")
    if not all(isinstance(detection, dict) for detection in detections):
        raise TerminalExportError(f"{field}.detections entries must be objects")
    return container


def _media_path(export_dir: Path, raw_filepath: object) -> tuple[Path, str]:
    if not isinstance(raw_filepath, str) or not raw_filepath.strip():
        raise TerminalExportError("filepath must be a nonempty string")

    export_root = export_dir.resolve()
    supplied_path = Path(raw_filepath)
    candidate = supplied_path if supplied_path.is_absolute() else export_root / supplied_path
    resolved_path = candidate.resolve()

    if resolved_path == export_root or not resolved_path.is_relative_to(export_root):
        raise TerminalExportError(
            f"filepath must resolve below export directory: {raw_filepath}"
        )

    relative_filepath = resolved_path.relative_to(export_root).as_posix()
    return resolved_path, relative_filepath


def _parse_terminal_sample(
    export_dir: Path,
    sample: object,
) -> TerminalRecord:
    if not isinstance(sample, dict):
        raise TerminalExportError("each samples.json entry must be an object")

    jaguar_id = sample.get("jaguar_id")
    if not isinstance(jaguar_id, str) or not jaguar_id.strip():
        raise TerminalExportError("jaguar_id must be a nonempty string")

    filepath, relative_filepath = _media_path(export_dir, sample.get("filepath"))
    return TerminalRecord(
        source_id=_source_id(sample),
        filepath=filepath,
        relative_filepath=relative_filepath,
        jaguar_id=jaguar_id.strip(),
        bboxes_body=_annotation_container(sample, "bboxes_body"),
        segmentations_body=_annotation_container(sample, "segmentations_body"),
    )


def load_terminal_records(export_dir: Path) -> list[TerminalRecord]:
    """Load terminal samples from a FiftyOne JSON export."""
    samples_path = export_dir / "samples.json"
    try:
        payload = json.loads(samples_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalExportError(f"could not read {samples_path}: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise TerminalExportError("samples.json must contain a samples list")

    records = [
        _parse_terminal_sample(export_dir, sample)
        for sample in payload["samples"]
    ]
    return sorted(records, key=lambda record: record.relative_filepath)
