import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

JsonScalar = str | int | float | bool | None
FrozenJsonValue = JsonScalar | Mapping[str, "FrozenJsonValue"] | tuple["FrozenJsonValue", ...]
FrozenAnnotation = Mapping[str, FrozenJsonValue]


class TerminalExportError(ValueError):
    """Raised when a terminal FiftyOne export is malformed or unsafe."""


@dataclass(frozen=True)
class TerminalRecord:
    source_id: str
    filepath: Path
    relative_filepath: str
    jaguar_id: str | None
    bboxes_body: FrozenAnnotation
    segmentations_body: FrozenAnnotation

    def __post_init__(self) -> None:
        """Detach annotations from caller-owned mutable containers."""
        object.__setattr__(self, "bboxes_body", _freeze_annotation(self.bboxes_body))
        object.__setattr__(
            self,
            "segmentations_body",
            _freeze_annotation(self.segmentations_body),
        )


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
) -> Mapping[str, Any]:
    container = sample.get(field)
    if not isinstance(container, dict):
        raise TerminalExportError(f"{field} must be an annotation object")

    detections = container.get("detections")
    if not isinstance(detections, list) or not detections:
        raise TerminalExportError(f"{field}.detections must be a nonempty list")
    if not all(isinstance(detection, dict) for detection in detections):
        raise TerminalExportError(f"{field}.detections entries must be objects")
    return container


def _freeze_json(value: Any) -> FrozenJsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(nested) for nested in value)
    raise TerminalExportError(f"annotation contains unsupported value type: {type(value).__name__}")


def _freeze_annotation(annotation: Mapping[str, Any]) -> FrozenAnnotation:
    frozen = _freeze_json(annotation)
    if not isinstance(frozen, Mapping):
        raise TerminalExportError("annotation must be an object")
    return frozen


def _thaw_json(value: FrozenJsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value


def annotation_to_dict(annotation: FrozenAnnotation) -> dict[str, Any]:
    """Return an independent, mutable dictionary for a frozen annotation."""
    return {key: _thaw_json(value) for key, value in annotation.items()}


def _media_path(export_dir: Path, raw_filepath: object) -> tuple[Path, str]:
    if not isinstance(raw_filepath, str) or not raw_filepath.strip():
        raise TerminalExportError("filepath must be a nonempty string")

    try:
        export_root = export_dir.resolve()
        supplied_path = Path(raw_filepath)
        candidate = supplied_path if supplied_path.is_absolute() else export_root / supplied_path
        resolved_path = candidate.resolve()
    except (OSError, ValueError) as exc:
        raise TerminalExportError(f"invalid filepath {raw_filepath!r}: {exc}") from exc

    if resolved_path == export_root or not resolved_path.is_relative_to(export_root):
        raise TerminalExportError(f"filepath must resolve below export directory: {raw_filepath}")

    relative_filepath = resolved_path.relative_to(export_root).as_posix()
    return resolved_path, relative_filepath


def _parse_terminal_sample(
    export_dir: Path,
    sample: object,
) -> TerminalRecord:
    if not isinstance(sample, dict):
        raise TerminalExportError("each samples.json entry must be an object")

    raw_jaguar_id = sample.get("jaguar_id")
    if raw_jaguar_id is None:
        jaguar_id = None
    elif not isinstance(raw_jaguar_id, str) or not raw_jaguar_id.strip():
        raise TerminalExportError("jaguar_id must be null or a nonempty string")
    else:
        jaguar_id = raw_jaguar_id.strip()

    filepath, relative_filepath = _media_path(export_dir, sample.get("filepath"))
    return TerminalRecord(
        source_id=_source_id(sample),
        filepath=filepath,
        relative_filepath=relative_filepath,
        jaguar_id=jaguar_id,
        bboxes_body=_annotation_container(sample, "bboxes_body"),
        segmentations_body=_annotation_container(sample, "segmentations_body"),
    )


def load_terminal_records(export_dir: Path) -> list[TerminalRecord]:
    """Load terminal samples from a FiftyOne JSON export."""
    samples_path = export_dir / "samples.json"
    try:
        payload = json.loads(samples_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TerminalExportError(f"could not read {samples_path}: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise TerminalExportError("samples.json must contain a samples list")

    records = [_parse_terminal_sample(export_dir, sample) for sample in payload["samples"]]
    return sorted(records, key=lambda record: record.relative_filepath)
