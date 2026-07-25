import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import posixpath
from pathlib import Path
import re
from types import MappingProxyType
from typing import Literal

from jaguars.visualization.final_records import TerminalRecord

Scalar = str | int | float | bool | None
LineageStatus = Literal["matched", "ambiguous", "missing"]
MatchMethod = Literal[
    "source_id",
    "normalized_source_filepath",
    "export_relative_filepath",
    "unique_filename",
]
CandidateType = Literal["export", "manifest", "provided"]


@dataclass(frozen=True)
class LineageCandidate:
    candidate_type: CandidateType = "provided"
    source_id: str | None = None
    source_ids: tuple[str, ...] = ()
    normalized_source_filepath: str | None = None
    export_relative_filepath: str | None = None
    sighting_id: Scalar = None
    jaguar_id: Scalar = None
    closed_set_split: Scalar = None
    open_set_split: Scalar = None
    site: Scalar = None
    location: Scalar = None
    camera_id: Scalar = None
    camera_side: Scalar = None
    camera_model: Scalar = None
    latitude: Scalar = None
    longitude: Scalar = None
    capture_date: Scalar = None
    capture_time: Scalar = None
    capture_datetime: Scalar = None
    original_filename: str | None = None
    source_media_path: Scalar = None
    source_type: Scalar = None

    @property
    def fields(self) -> Mapping[str, Scalar]:
        """Return populated enrichment fields."""
        return MappingProxyType(
            {
                key: value
                for key, value in {
                    "closed_set_split": self.closed_set_split,
                    "open_set_split": self.open_set_split,
                    "sighting_id": self.sighting_id,
                    "site": self.site,
                    "location": self.location,
                    "camera_id": self.camera_id,
                    "camera_side": self.camera_side,
                    "camera_model": self.camera_model,
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "capture_date": self.capture_date,
                    "capture_time": self.capture_time,
                    "capture_datetime": self.capture_datetime,
                    "original_filename": self.original_filename,
                    "source_media_path": self.source_media_path,
                    "source_type": self.source_type,
                }.items()
                if value is not None
            }
        )


@dataclass(frozen=True)
class Enrichment:
    status: LineageStatus
    match_method: MatchMethod | None
    fields: Mapping[str, Scalar]


class LineageIndex:
    def __init__(self, candidates: Iterable[LineageCandidate]) -> None:
        candidate_list = list(candidates)
        self._by_source_id: dict[str, list[LineageCandidate]] = {}
        self._by_normalized_source_filepath: dict[str, list[LineageCandidate]] = {}
        self._by_export_relative_filepath: dict[str, list[LineageCandidate]] = {}
        self._by_filename: dict[str, list[LineageCandidate]] = {}
        self._manifests_by_identity_filename: dict[
            tuple[Scalar, str],
            list[LineageCandidate],
        ] = {}
        for candidate in candidate_list:
            source_ids = ((candidate.source_id,) if candidate.source_id is not None else ()) + candidate.source_ids
            for source_id in dict.fromkeys(source_ids):
                self._by_source_id.setdefault(source_id, []).append(candidate)
            if candidate.normalized_source_filepath is not None:
                source_filepath = _normalize_path(candidate.normalized_source_filepath)
                self._by_normalized_source_filepath.setdefault(source_filepath, []).append(candidate)
            if candidate.export_relative_filepath is not None:
                relative_filepath = _normalize_path(candidate.export_relative_filepath)
                self._by_export_relative_filepath.setdefault(relative_filepath, []).append(candidate)
            candidate_filenames = [
                filename
                for filename in (
                    candidate.original_filename,
                    (
                        posixpath.basename(_normalize_path(candidate.export_relative_filepath))
                        if candidate.export_relative_filepath is not None
                        else None
                    ),
                    (
                        posixpath.basename(_normalize_path(candidate.normalized_source_filepath))
                        if candidate.normalized_source_filepath is not None
                        else None
                    ),
                )
                if filename is not None
            ]
            for filename in dict.fromkeys(candidate_filenames):
                self._by_filename.setdefault(filename, []).append(candidate)
            if candidate.original_filename is not None and candidate.candidate_type == "manifest":
                identity_filename = (
                    candidate.jaguar_id,
                    candidate.original_filename,
                )
                self._manifests_by_identity_filename.setdefault(
                    identity_filename,
                    [],
                ).append(candidate)

    @classmethod
    def from_candidates(cls, candidates: Iterable[LineageCandidate]) -> "LineageIndex":
        """Build exact lineage indexes without discarding duplicate keys."""
        return cls(candidates)

    def enrich(self, terminal: TerminalRecord) -> Enrichment:
        """Enrich one terminal record using exact precedence-ordered matches."""
        lookups: tuple[
            tuple[MatchMethod, list[LineageCandidate]],
            ...,
        ] = (
            ("source_id", self._by_source_id.get(terminal.source_id, [])),
            (
                "normalized_source_filepath",
                self._by_normalized_source_filepath.get(
                    _normalize_path(terminal.filepath.as_posix()),
                    [],
                ),
            ),
            (
                "export_relative_filepath",
                self._by_export_relative_filepath.get(
                    _normalize_path(terminal.relative_filepath),
                    [],
                ),
            ),
            ("unique_filename", self._by_filename.get(terminal.filepath.name, [])),
        )
        for method, matches in lookups:
            matched_candidate = self._one_logical_candidate(matches)
            if matched_candidate is not None:
                return Enrichment(
                    status="matched",
                    match_method=method,
                    fields=matched_candidate.fields,
                )
            if matches:
                return Enrichment(status="ambiguous", match_method=None, fields=MappingProxyType({}))
        return Enrichment(status="missing", match_method=None, fields=MappingProxyType({}))

    def _one_logical_candidate(
        self,
        candidates: list[LineageCandidate],
    ) -> LineageCandidate | None:
        if len(candidates) == 1:
            candidate = candidates[0]
            if candidate.candidate_type != "export":
                return candidate
            manifest_matches = self._manifests_by_identity_filename.get(
                (candidate.jaguar_id, candidate.original_filename or ""),
                [],
            )
            if len(manifest_matches) == 1:
                merged = _merge_export_manifest(candidate, manifest_matches[0])
                if merged is not None:
                    return merged
            return candidate

        exports = [candidate for candidate in candidates if candidate.candidate_type == "export"]
        manifests = [candidate for candidate in candidates if candidate.candidate_type == "manifest"]
        if len(exports) == 1 and len(manifests) == 1 and len(candidates) == 2:
            return _merge_export_manifest(exports[0], manifests[0])
        return None


def _normalize_path(value: str) -> str:
    return posixpath.normpath(value.strip().replace("\\", "/"))


def _is_absolute_path(value: str) -> bool:
    normalized = _normalize_path(value)
    return posixpath.isabs(normalized) or bool(re.match(r"^[A-Za-z]:/", normalized))


def _compose_source_path(
    *,
    direct_source_path: str | None,
    file_path: str | None,
    dataset_path: str | None,
    raw_data_path: str | None,
) -> str | None:
    if direct_source_path is not None:
        path = _normalize_path(direct_source_path)
        base_path = raw_data_path or dataset_path
    elif file_path is not None:
        path = _normalize_path(file_path)
        base_path = dataset_path or raw_data_path
    else:
        return None

    if _is_absolute_path(path) or base_path is None:
        return path
    return _normalize_path(posixpath.join(_normalize_path(base_path), path))


_ENRICHMENT_FIELD_NAMES = (
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
    "source_media_path",
    "source_type",
)


def _merge_export_manifest(
    export: LineageCandidate,
    manifest: LineageCandidate,
) -> LineageCandidate | None:
    if (
        export.jaguar_id is None
        or export.jaguar_id != manifest.jaguar_id
        or export.original_filename is None
        or export.original_filename != manifest.original_filename
    ):
        return None
    merged_fields = {
        field: (getattr(export, field) if getattr(export, field) is not None else getattr(manifest, field)) for field in _ENRICHMENT_FIELD_NAMES
    }
    return LineageCandidate(
        candidate_type="export",
        source_id=export.source_id,
        source_ids=export.source_ids,
        normalized_source_filepath=export.normalized_source_filepath,
        export_relative_filepath=export.export_relative_filepath,
        jaguar_id=export.jaguar_id,
        original_filename=export.original_filename,
        **merged_fields,
    )


_HEADER_ALIASES: Mapping[str, tuple[str, ...]] = {
    "source_id": ("SOURCE_ID", "SAMPLE_ID", "ID"),
    "export_relative_filepath": ("EXPORT_RELATIVE_FILEPATH",),
    "jaguar_id": ("JAGUAR_ID",),
    "closed_set_split": ("CLOSED_SET_SPLIT", "CLOSED_SPLIT"),
    "open_set_split": ("OPEN_SET_SPLIT", "OPEN_SPLIT"),
    "sighting_id": ("SIGHTING_ID",),
    "site": ("SITE", "CAMERA_TRAP_SITE", "ORIGINAL_SITE"),
    "location": ("LOCATION",),
    "camera_id": ("CAMERA_ID",),
    "camera_side": ("CAMERA_SIDE", "CAM", "ORIGINAL_CAM"),
    "camera_model": ("CAMERA_MODEL",),
    "latitude": ("LATITUDE", "LAT"),
    "longitude": ("LONGITUDE", "LON", "LNG"),
    "capture_date": ("CAPTURE_DATE", "DATE"),
    "capture_time": ("CAPTURE_TIME", "TIME"),
    "capture_datetime": ("CAPTURE_DATETIME", "DATETIME"),
    "original_filename": (
        "ORIGINAL_FILENAME",
        "ORIGINAL_FILE_NAME",
        "FILES_NAME",
        "FILE_NAME",
        "FILENAME",
    ),
    "source_media_path": (
        "SOURCE_MEDIA_PATH",
        "SOURCE_FILEPATH",
        "RAW_FILE_PATH",
    ),
    "file_path": ("FILE_PATH",),
    "dataset_path": ("DATASET_PATH",),
    "raw_data_path": ("RAW_DATA_PATH",),
    "source_type": ("SOURCE_TYPE",),
}


def _normalize_header(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")


def _first_value(row: Mapping[str, str | None], field: str) -> str | None:
    for alias in _HEADER_ALIASES[field]:
        value = row.get(alias)
        if value is not None and value.strip():
            return value.strip()
    return None


def _coordinate(row: Mapping[str, str | None], field: str) -> Scalar:
    value = _first_value(row, field)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def load_manifest_candidates(manifest_path: Path) -> tuple[LineageCandidate, ...]:
    """Load one split manifest into normalized lineage candidates."""
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        rows = [{_normalize_header(header): value for header, value in row.items() if header is not None} for row in reader]

    default_source_type = "pptx" if manifest_path.name.startswith("pptx_") else "csv"
    candidates = []
    for row in rows:
        source_media_path = _compose_source_path(
            direct_source_path=_first_value(row, "source_media_path"),
            file_path=_first_value(row, "file_path"),
            dataset_path=_first_value(row, "dataset_path"),
            raw_data_path=_first_value(row, "raw_data_path"),
        )
        original_filename = _first_value(row, "original_filename")
        if original_filename is None and source_media_path is not None:
            original_filename = Path(_normalize_path(source_media_path)).name
        candidates.append(
            LineageCandidate(
                candidate_type="manifest",
                source_id=_first_value(row, "source_id"),
                normalized_source_filepath=source_media_path,
                export_relative_filepath=_first_value(
                    row,
                    "export_relative_filepath",
                ),
                jaguar_id=_first_value(row, "jaguar_id"),
                closed_set_split=_first_value(row, "closed_set_split"),
                open_set_split=_first_value(row, "open_set_split"),
                sighting_id=_first_value(row, "sighting_id"),
                site=_first_value(row, "site"),
                location=_first_value(row, "location"),
                camera_id=_first_value(row, "camera_id"),
                camera_side=_first_value(row, "camera_side"),
                camera_model=_first_value(row, "camera_model"),
                latitude=_coordinate(row, "latitude"),
                longitude=_coordinate(row, "longitude"),
                capture_date=_first_value(row, "capture_date"),
                capture_time=_first_value(row, "capture_time"),
                capture_datetime=_first_value(row, "capture_datetime"),
                original_filename=original_filename,
                source_media_path=source_media_path,
                source_type=_first_value(row, "source_type") or default_source_type,
            )
        )
    return tuple(candidates)


def _json_scalar(
    sample: Mapping[str, object],
    field: str,
) -> Scalar:
    for alias in _HEADER_ALIASES[field]:
        value = sample.get(alias)
        if value is None or not isinstance(value, (str, int, float, bool)):
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            return stripped
        return value
    return None


def _json_string(
    sample: Mapping[str, object],
    field: str,
) -> str | None:
    value = _json_scalar(sample, field)
    return value if isinstance(value, str) else None


def _json_source_ids(sample: Mapping[str, object]) -> tuple[str, ...]:
    source_ids = []
    for field in ("ID", "SOURCE_ID", "SAMPLE_ID", "SOURCE"):
        value = sample.get(field)
        if isinstance(value, Mapping):
            value = value.get("$oid")
        if isinstance(value, str) and value.strip():
            source_ids.append(value.strip())
    return tuple(dict.fromkeys(source_ids))


def load_export_candidates(export_dir: Path) -> tuple[LineageCandidate, ...]:
    """Load normalized candidates from a FiftyOne JSON export."""
    payload = json.loads((export_dir / "samples.json").read_text(encoding="utf-8"))
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("samples.json must contain a samples list")

    candidates = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            raise ValueError("each samples.json entry must be an object")
        sample = {_normalize_header(str(key)): value for key, value in raw_sample.items()}
        raw_filepath = sample.get("FILEPATH")
        filepath = _normalize_path(raw_filepath) if isinstance(raw_filepath, str) and raw_filepath.strip() else None
        if filepath is not None and posixpath.isabs(filepath):
            export_root = _normalize_path(export_dir.resolve().as_posix())
            prefix = f"{export_root}/"
            export_relative_filepath = filepath[len(prefix) :] if filepath.startswith(prefix) else None
        else:
            export_relative_filepath = filepath

        source_media_path = _compose_source_path(
            direct_source_path=_json_string(sample, "source_media_path"),
            file_path=_json_string(sample, "file_path"),
            dataset_path=_json_string(sample, "dataset_path"),
            raw_data_path=_json_string(sample, "raw_data_path"),
        )
        if source_media_path is None and filepath is not None and posixpath.isabs(filepath):
            source_media_path = filepath
        normalized_source_filepath = source_media_path
        original_filename = _json_string(sample, "original_filename")
        if original_filename is None and filepath is not None:
            original_filename = Path(filepath).name

        source_ids = _json_source_ids(sample)
        candidates.append(
            LineageCandidate(
                candidate_type="export",
                source_id=source_ids[0] if source_ids else None,
                source_ids=source_ids[1:],
                normalized_source_filepath=normalized_source_filepath,
                export_relative_filepath=export_relative_filepath,
                jaguar_id=_json_scalar(sample, "jaguar_id"),
                closed_set_split=_json_scalar(sample, "closed_set_split"),
                open_set_split=_json_scalar(sample, "open_set_split"),
                sighting_id=_json_scalar(sample, "sighting_id"),
                site=_json_scalar(sample, "site"),
                location=_json_scalar(sample, "location"),
                camera_id=_json_scalar(sample, "camera_id"),
                camera_side=_json_scalar(sample, "camera_side"),
                camera_model=_json_scalar(sample, "camera_model"),
                latitude=_json_scalar(sample, "latitude"),
                longitude=_json_scalar(sample, "longitude"),
                capture_date=_json_scalar(sample, "capture_date"),
                capture_time=_json_scalar(sample, "capture_time"),
                capture_datetime=_json_scalar(sample, "capture_datetime"),
                original_filename=original_filename,
                source_media_path=source_media_path,
                source_type=_json_scalar(sample, "source_type"),
            )
        )
    return tuple(candidates)


def load_lineage_candidates(intermediate_dir: Path) -> tuple[LineageCandidate, ...]:
    """Load the approved upstream exports and terminal split manifests."""
    fiftyone_root = intermediate_dir / "fo_jaguars"
    export_dirs = (
        fiftyone_root / "exports/segmented_deduplicated",
        fiftyone_root / "exports/segmented",
        fiftyone_root / "exports/deduplicated",
        fiftyone_root / "ingested",
    )
    manifest_paths = (
        intermediate_dir / "labels_with_splits.csv",
        intermediate_dir / "pptx_extracted_labels_with_splits.csv",
    )
    export_candidates = (candidate for export_dir in export_dirs for candidate in load_export_candidates(export_dir))
    manifest_candidates = (candidate for manifest_path in manifest_paths for candidate in load_manifest_candidates(manifest_path))
    return tuple((*export_candidates, *manifest_candidates))
