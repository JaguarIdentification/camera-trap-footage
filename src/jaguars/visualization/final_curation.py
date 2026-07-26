"""Deterministic construction of the final curated terminal export."""

import argparse
import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from jaguars.visualization.final_lineage import (
    Enrichment,
    LineageIndex,
    load_lineage_candidates_from_paths,
)
from jaguars.visualization.final_records import (
    FrozenAnnotation,
    FrozenJsonValue,
    TerminalRecord,
    TerminalExportError,
    annotation_to_dict,
    load_terminal_records,
)
from jaguars.visualization.final_validation import (
    IntegrityError,
    validate_annotations,
    validate_media,
    validate_records,
)

DEFAULT_SOURCE_EXPORT_DIR = Path("data/intermediate/v1/fo_jaguars/labeled_segmented_jaguars_primitive")
DEFAULT_TARGET_MOUNT_ROOT = Path("/Volumes/CameraTrapPython")
DEFAULT_CURATED_EXPORT_ROOT = DEFAULT_TARGET_MOUNT_ROOT / "fiftyone/exports"
DEFAULT_TARGET_EXPORT_DIR = DEFAULT_CURATED_EXPORT_ROOT / "JaguarCameraTrap_Final_Curated_v1"
DEFAULT_INTERMEDIATE_DIR = Path("data/intermediate/v1")
DEFAULT_UPSTREAM_EXPORT_DIRS = (
    DEFAULT_INTERMEDIATE_DIR / "fo_jaguars/exports/segmented_deduplicated",
    DEFAULT_INTERMEDIATE_DIR / "fo_jaguars/exports/segmented",
    DEFAULT_INTERMEDIATE_DIR / "fo_jaguars/exports/deduplicated",
    DEFAULT_INTERMEDIATE_DIR / "fo_jaguars/ingested",
)
DEFAULT_MANIFEST_PATHS = (
    DEFAULT_INTERMEDIATE_DIR / "labels_with_splits.csv",
    DEFAULT_INTERMEDIATE_DIR / "pptx_extracted_labels_with_splits.csv",
)
POLICY_VERSION = "final-curated-v1"
NEEDS_REVIEW_TAG = "needs_annotation_review"
_LINUX_AT_FDCWD = -100
_ATTR_BIT_MAP_COUNT = 5
_ATTR_VOL_CAPABILITIES = 0x00020000
_ATTR_VOL_INFO = 0x80000000
_RENAME_EXCL = 0x00000004
_RENAME_NOREPLACE = 1
_VOL_CAPABILITIES_INTERFACES = 1
_VOL_CAP_INT_RENAME_EXCL = 0x00080000


@dataclass(frozen=True)
class CurationPolicy:
    version: str
    false_positive_paths: frozenset[str]
    review_cases: Mapping[str, str]
    clipped_bboxes: Mapping[str, tuple[float, float, float, float]]
    expected_source_count: int
    expected_curated_count: int
    expected_populated_identities: int
    expected_null_identities: int
    expected_distinct_identities: int


class _AttributeList(ctypes.Structure):
    _fields_ = [
        ("bitmapcount", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


class _VolumeCapabilities(ctypes.Structure):
    _fields_ = [
        ("capabilities", ctypes.c_uint32 * 4),
        ("valid", ctypes.c_uint32 * 4),
    ]


class _VolumeCapabilitiesBuffer(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("capabilities", _VolumeCapabilities),
    ]


DEFAULT_POLICY = CurationPolicy(
    version=POLICY_VERSION,
    false_positive_paths=frozenset(
        {
            "data/000005-11.jpg",
            "data/000010-8.jpg",
            "data/000010-9.jpg",
            "data/000015-9.jpg",
            "data/000030-25.jpg",
            "data/000030-6.jpg",
        }
    ),
    review_cases=MappingProxyType(
        {
            "data/000001-143.jpg": "zero-area mask",
            "data/000002-144.jpg": "zero-area mask",
            "data/000010-18.jpg": "zero-area mask",
            "data/000005-126.jpg": "zero-area mask; species uncertain",
        }
    ),
    clipped_bboxes=MappingProxyType(
        {
            "data/000004-120.jpg": (
                0.0005632716049382716,
                0.0,
                0.8508101851851853,
                0.9984678819444443,
            ),
            "data/000005-61.jpg": (
                0.8909876543209877,
                0.0,
                0.10871141975308632,
                0.9980946180555556,
            ),
            "data/000025-40.jpg": (
                0.0008101851851851853,
                0.0,
                0.9981172839506173,
                0.998927951388889,
            ),
        }
    ),
    expected_source_count=1367,
    expected_curated_count=1322,
    expected_populated_identities=1108,
    expected_null_identities=214,
    expected_distinct_identities=59,
)


class CurationError(RuntimeError):
    """Base class for controlled terminal-export curation failures."""


class CurationConflictError(CurationError):
    """Raised when an exact-content group has conflicting final semantics."""


@dataclass(frozen=True)
class CurationDrop:
    relative_filepath: str
    reason: str
    representative_filepath: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class CurationHashGroup:
    sha256: str
    member_paths: tuple[str, ...]
    representative_path: str


@dataclass(frozen=True)
class CuratedSample:
    terminal: TerminalRecord
    source_sample: Mapping[str, Any]
    sha256: str
    review_reason: str | None = None
    clipped_bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CurationPlan:
    requested_source_dir: Path
    source_dir: Path
    target_dir: Path
    resolved_target_dir: Path
    policy_version: str
    source_samples_sha256: str
    source_count: int
    curated_count: int
    unique_hashes: int
    populated_identities: int
    null_identities: int
    distinct_identities: int
    kept_paths: tuple[str, ...]
    dropped: tuple[CurationDrop, ...]
    hash_groups: tuple[CurationHashGroup, ...]
    clipped_paths: tuple[str, ...]
    review_paths: tuple[str, ...]
    selected: tuple[CuratedSample, ...]


@dataclass(frozen=True)
class ExpectedTargetState:
    exists: bool
    canonical_path: Path | None = None
    device: int | None = None
    inode: int | None = None
    report_sha256: str | None = None
    report_source_export: str | None = None
    report_source_samples_sha256: str | None = None


@dataclass(frozen=True)
class _DirectoryIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class _EntryIdentity:
    device: int
    inode: int
    file_type: int


@dataclass(frozen=True)
class _ValidatedTargetStorage:
    mount_path: Path
    target_path: Path
    mount_identity: _DirectoryIdentity
    is_mount_fn: Callable[[Path], bool]


@dataclass(frozen=True)
class _PinnedTargetParent:
    logical_path: Path
    descriptor: int
    identity: _DirectoryIdentity


@dataclass(frozen=True)
class _OwnedDirectory:
    name: str
    descriptor: int
    identity: _DirectoryIdentity


def _without_generated_ids(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _without_generated_ids(nested) for key, nested in value.items() if key != "_id"}
    if isinstance(value, (list, tuple)):
        return tuple(_without_generated_ids(nested) for nested in value)
    return value


def normalize_semantic_annotations(
    record: TerminalRecord,
) -> tuple[FrozenJsonValue, FrozenJsonValue]:
    """Return annotation semantics without generated detection identifiers."""
    bboxes, segmentations = _required_annotations(record)
    return (
        _without_generated_ids(annotation_to_dict(bboxes)),
        _without_generated_ids(annotation_to_dict(segmentations)),
    )


def _required_annotations(
    record: TerminalRecord,
) -> tuple[FrozenAnnotation, FrozenAnnotation]:
    if record.bboxes_body is None or record.segmentations_body is None:
        raise CurationConflictError(f"{record.relative_filepath}: duplicate selection requires body annotations")
    return record.bboxes_body, record.segmentations_body


def _annotation_is_valid(record: TerminalRecord) -> bool:
    try:
        validate_annotations(record)
    except IntegrityError:
        return False
    return True


def _has_compatible_match(
    record: TerminalRecord,
    enrichment: Enrichment | None,
) -> bool:
    if enrichment is None or enrichment.status != "matched":
        return False
    enriched_identity = enrichment.fields.get("jaguar_id")
    return enriched_identity is None or record.jaguar_id is None or enriched_identity == record.jaguar_id


def choose_representative(
    records: Sequence[TerminalRecord],
    *,
    enrichments: Mapping[str, Enrichment] | None = None,
    require_semantic_agreement: bool = True,
) -> TerminalRecord:
    """Choose one representative from an exact-content duplicate group."""
    if not records:
        raise CurationConflictError("cannot choose a representative from an empty group")

    identities = {record.jaguar_id for record in records if record.jaguar_id is not None}
    if len(identities) > 1:
        raise CurationConflictError(f"duplicate group contains conflicting populated identities: {sorted(identities)!r}")

    if require_semantic_agreement:
        semantics = {
            json.dumps(
                normalize_semantic_annotations(record),
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        }
        if len(semantics) > 1:
            names = sorted(record.relative_filepath for record in records)
            raise CurationConflictError(f"duplicate group contains conflicting semantic annotations: {names!r}")

    enrichment_by_path = enrichments or {}

    def rank(record: TerminalRecord) -> tuple[int, int, int, str]:
        return (
            -int(_annotation_is_valid(record)),
            -int(record.jaguar_id is not None),
            -int(
                _has_compatible_match(
                    record,
                    enrichment_by_path.get(record.relative_filepath),
                )
            ),
            record.relative_filepath,
        )

    return min(records, key=rank)


def _read_source_samples(source_dir: Path) -> tuple[dict[str, Any], str]:
    samples_path = source_dir / "samples.json"
    try:
        payload_bytes = samples_path.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CurationError(f"could not read source samples {samples_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise CurationError("source samples.json must contain a samples list")
    if not all(isinstance(sample, dict) for sample in payload["samples"]):
        raise CurationError("source samples.json entries must be objects")
    return payload, hashlib.sha256(payload_bytes).hexdigest()


def _clipped_terminal(
    terminal: TerminalRecord,
    clipped_bbox: tuple[float, float, float, float],
) -> TerminalRecord:
    bboxes_annotation, _ = _required_annotations(terminal)
    bboxes = annotation_to_dict(bboxes_annotation)
    detections = bboxes.get("detections")
    if not isinstance(detections, list) or len(detections) != 1:
        raise CurationConflictError(f"{terminal.relative_filepath}: audited bbox clip requires exactly one body detection")
    detection = detections[0]
    if not isinstance(detection, dict):
        raise CurationConflictError(f"{terminal.relative_filepath}: audited bbox detection must be an object")
    detection["bounding_box"] = list(clipped_bbox)
    return replace(
        terminal,
        bboxes_body=bboxes,
    )


def _validate_policy_paths(
    records_by_path: Mapping[str, TerminalRecord],
    policy: CurationPolicy,
) -> None:
    configured = set(policy.false_positive_paths) | set(policy.review_cases) | set(policy.clipped_bboxes)
    missing = sorted(configured - set(records_by_path))
    if missing:
        raise CurationConflictError(f"curation policy paths are absent from source export: {missing!r}")


def _enforce_expected_counts(
    *,
    policy: CurationPolicy,
    source_count: int,
    selected: Sequence[CuratedSample],
) -> None:
    identities = [sample.terminal.jaguar_id for sample in selected]
    populated = sum(identity is not None for identity in identities)
    null = len(identities) - populated
    distinct = len({identity for identity in identities if identity is not None})
    expected = {
        "source samples": (policy.expected_source_count, source_count),
        "curated samples": (policy.expected_curated_count, len(selected)),
        "populated identities": (policy.expected_populated_identities, populated),
        "null identities": (policy.expected_null_identities, null),
        "distinct identities": (policy.expected_distinct_identities, distinct),
    }
    errors = [f"expected {wanted} {name}, found {actual}" for name, (wanted, actual) in expected.items() if wanted != actual]
    if errors:
        raise CurationConflictError("; ".join(errors))


def _enforce_policy_actions(
    *,
    policy: CurationPolicy,
    selected: Sequence[CuratedSample],
    drops: Sequence[CurationDrop],
) -> None:
    actual_false_positives = {drop.relative_filepath for drop in drops if drop.reason == "confirmed_false_positive"}
    actual_reviews = {sample.terminal.relative_filepath for sample in selected if sample.review_reason is not None}
    actual_clips = {sample.terminal.relative_filepath for sample in selected if sample.clipped_bbox is not None}
    expected_to_actual = (
        ("false-positive exclusions", set(policy.false_positive_paths), actual_false_positives),
        ("pending reviews", set(policy.review_cases), actual_reviews),
        ("bbox clips", set(policy.clipped_bboxes), actual_clips),
    )
    errors = [f"{name} missing {sorted(expected - actual)!r}" for name, expected, actual in expected_to_actual if expected != actual]
    if errors:
        raise CurationConflictError("configured curation policy actions were not applied exactly: " + "; ".join(errors))


def build_curation_plan(
    source_dir: Path = DEFAULT_SOURCE_EXPORT_DIR,
    target_dir: Path = DEFAULT_TARGET_EXPORT_DIR,
    *,
    policy: CurationPolicy = DEFAULT_POLICY,
    enrichments: Mapping[str, Enrichment] | None = None,
) -> CurationPlan:
    """Compute and validate all curation decisions without writing output."""
    requested_source = source_dir.expanduser().absolute()
    source_root = requested_source.resolve(strict=False)
    requested_target = target_dir.expanduser().absolute()
    resolved_target = requested_target.resolve(strict=False)
    if source_root == resolved_target:
        raise CurationError("source and target export directories must be distinct")

    payload, source_samples_sha256 = _read_source_samples(source_root)
    terminals = load_terminal_records(source_root)
    records_by_path = {terminal.relative_filepath: terminal for terminal in terminals}
    if len(records_by_path) != len(terminals):
        raise CurationConflictError("source export contains duplicate relative filepaths")
    _validate_policy_paths(records_by_path, policy)

    raw_samples = {str(sample["filepath"]): sample for sample in payload["samples"] if isinstance(sample.get("filepath"), str)}
    if set(raw_samples) != set(records_by_path):
        raise CurationConflictError("source samples could not be mapped one-to-one by relative filepath")

    hashes = {terminal.relative_filepath: validate_media(terminal.filepath).sha256 for terminal in terminals}
    grouped: dict[str, list[TerminalRecord]] = {}
    for terminal in terminals:
        grouped.setdefault(hashes[terminal.relative_filepath], []).append(terminal)

    deduplicated: list[TerminalRecord] = []
    drops: list[CurationDrop] = []
    hash_groups: list[CurationHashGroup] = []
    for digest, group in sorted(grouped.items()):
        representative = choose_representative(
            group,
            enrichments=enrichments,
        )
        deduplicated.append(representative)
        if len(group) > 1:
            member_paths = tuple(sorted(record.relative_filepath for record in group))
            hash_groups.append(
                CurationHashGroup(
                    sha256=digest,
                    member_paths=member_paths,
                    representative_path=representative.relative_filepath,
                )
            )
            drops.extend(
                CurationDrop(
                    relative_filepath=record.relative_filepath,
                    reason="exact_content_duplicate",
                    representative_filepath=representative.relative_filepath,
                    sha256=digest,
                )
                for record in group
                if record is not representative
            )

    selected: list[CuratedSample] = []
    selected_digests: set[str] = set()
    for terminal in deduplicated:
        path = terminal.relative_filepath
        digest = hashes[path]
        if path in policy.false_positive_paths:
            drops.append(
                CurationDrop(
                    relative_filepath=path,
                    reason="confirmed_false_positive",
                    sha256=digest,
                )
            )
            continue

        review_reason = policy.review_cases.get(path)
        clipped_bbox = policy.clipped_bboxes.get(path)
        curated_terminal = terminal
        if clipped_bbox is not None:
            curated_terminal = _clipped_terminal(terminal, clipped_bbox)
        if review_reason is None:
            validate_annotations(curated_terminal)
        elif not review_reason.strip():
            raise CurationConflictError(f"{path}: review reason must be nonempty")
        if digest in selected_digests:
            raise CurationConflictError(f"curated selection retained duplicate SHA-256 {digest}")
        selected_digests.add(digest)
        selected.append(
            CuratedSample(
                terminal=curated_terminal,
                source_sample=MappingProxyType(raw_samples[path]),
                sha256=digest,
                review_reason=review_reason,
                clipped_bbox=clipped_bbox,
            )
        )

    selected.sort(key=lambda sample: sample.terminal.relative_filepath)
    _enforce_policy_actions(
        policy=policy,
        selected=selected,
        drops=drops,
    )
    _enforce_expected_counts(
        policy=policy,
        source_count=len(terminals),
        selected=selected,
    )
    identities = [sample.terminal.jaguar_id for sample in selected]
    populated = sum(identity is not None for identity in identities)
    return CurationPlan(
        requested_source_dir=requested_source,
        source_dir=source_root,
        target_dir=requested_target,
        resolved_target_dir=resolved_target,
        policy_version=policy.version,
        source_samples_sha256=source_samples_sha256,
        source_count=len(terminals),
        curated_count=len(selected),
        unique_hashes=len(selected_digests),
        populated_identities=populated,
        null_identities=len(selected) - populated,
        distinct_identities=len({identity for identity in identities if identity is not None}),
        kept_paths=tuple(sample.terminal.relative_filepath for sample in selected),
        dropped=tuple(sorted(drops, key=lambda drop: (drop.relative_filepath, drop.reason))),
        hash_groups=tuple(sorted(hash_groups, key=lambda group: group.sha256)),
        clipped_paths=tuple(sample.terminal.relative_filepath for sample in selected if sample.clipped_bbox is not None),
        review_paths=tuple(sample.terminal.relative_filepath for sample in selected if sample.review_reason is not None),
        selected=tuple(selected),
    )


def _curated_sample_payload(
    sample: CuratedSample,
    policy_version: str,
) -> dict[str, object]:
    terminal = sample.terminal
    stable_id = hashlib.sha256(f"{policy_version}\0{terminal.relative_filepath}".encode()).hexdigest()[:24]
    payload: dict[str, object] = {
        "_id": {"$oid": stable_id},
        "filepath": str(terminal.filepath),
        "tags": [],
        "jaguar_id": terminal.jaguar_id,
    }
    if sample.review_reason is not None:
        payload.update(
            {
                "tags": [NEEDS_REVIEW_TAG],
                "review_required": True,
                "review_reason": sample.review_reason,
                "review_status": "pending",
            }
        )
    else:
        bboxes, segmentations = _required_annotations(terminal)
        payload["bboxes_body"] = _without_generated_ids(annotation_to_dict(bboxes))
        payload["segmentations_body"] = _without_generated_ids(annotation_to_dict(segmentations))
    return payload


def _metadata_payload(plan: CurationPlan) -> dict[str, object]:
    source_metadata_path = plan.source_dir / "metadata.json"
    try:
        source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CurationError(f"could not read source metadata {source_metadata_path}: {exc}") from exc
    if not isinstance(source, dict):
        raise CurationError("source metadata.json must contain an object")
    version = source.get("version")
    return {
        "name": plan.target_dir.name,
        "slug": plan.target_dir.name.replace("_", "-").casefold(),
        "version": version if isinstance(version, str) else None,
        "persistent": False,
        "media_type": "image",
        "tags": ["final_curated"],
        "info": {
            "curation_policy_version": plan.policy_version,
            "media_storage": "canonical_source_references",
            "allowed_media_root": str(plan.source_dir / "data"),
            "source_export": str(plan.source_dir),
            "source_samples_sha256": plan.source_samples_sha256,
            "sample_count": plan.curated_count,
        },
    }


def _drop_payload(drop: CurationDrop) -> dict[str, object]:
    return {
        "relative_filepath": drop.relative_filepath,
        "reason": drop.reason,
        "representative_filepath": drop.representative_filepath,
        "sha256": drop.sha256,
    }


def _report_payload(plan: CurationPlan) -> dict[str, object]:
    return {
        "policy_version": plan.policy_version,
        "media_storage": "canonical_source_references",
        "allowed_media_root": str(plan.source_dir / "data"),
        "source_export": str(plan.source_dir),
        "target_export": str(plan.target_dir),
        "source_samples_sha256": plan.source_samples_sha256,
        "counts": {
            "source": plan.source_count,
            "curated": plan.curated_count,
            "dropped": len(plan.dropped),
            "duplicate_groups": len(plan.hash_groups),
            "unique_hashes": plan.unique_hashes,
            "populated_identities": plan.populated_identities,
            "null_identities": plan.null_identities,
            "distinct_identities": plan.distinct_identities,
            "review_required": len(plan.review_paths),
            "bbox_clipped": len(plan.clipped_paths),
        },
        "kept_paths": list(plan.kept_paths),
        "dropped": [_drop_payload(drop) for drop in plan.dropped],
        "hash_groups": [
            {
                "sha256": group.sha256,
                "member_paths": list(group.member_paths),
                "representative_path": group.representative_path,
            }
            for group in plan.hash_groups
        ],
        "clips": [
            {
                "relative_filepath": sample.terminal.relative_filepath,
                "bounding_box": list(sample.clipped_bbox) if sample.clipped_bbox is not None else None,
            }
            for sample in plan.selected
            if sample.clipped_bbox is not None
        ],
        "reviews": [
            {
                "relative_filepath": sample.terminal.relative_filepath,
                "reason": sample.review_reason,
                "action": "stripped_invalid_annotations",
            }
            for sample in plan.selected
            if sample.review_reason is not None
        ],
        "media_sha256": {sample.terminal.relative_filepath: sample.sha256 for sample in plan.selected},
    }


def _write_json_at(
    directory_fd: int,
    name: str,
    payload: object,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as stream:
            descriptor = -1
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)


def _build_export_at(
    plan: CurationPlan,
    directory_fd: int,
) -> None:
    samples_payload = {"samples": [_curated_sample_payload(sample, plan.policy_version) for sample in plan.selected]}
    _write_json_at(
        directory_fd,
        "samples.json",
        samples_payload,
    )
    _write_json_at(
        directory_fd,
        "metadata.json",
        _metadata_payload(plan),
    )
    _write_json_at(
        directory_fd,
        "curation_report.json",
        _report_payload(plan),
    )


def _validate_staged_export(
    plan: CurationPlan,
    staging_dir: Path,
) -> None:
    try:
        terminals = load_terminal_records(
            staging_dir,
            allowed_media_root=plan.source_dir / "data",
        )
        validated = validate_records(
            [
                (
                    terminal,
                    Enrichment(
                        status="missing",
                        match_method=None,
                        fields=MappingProxyType({}),
                    ),
                )
                for terminal in terminals
            ],
            expected_count=plan.curated_count,
        )
    except (TerminalExportError, IntegrityError) as exc:
        raise CurationError(f"staged export failed validation: {exc}") from exc

    expected_entries = {
        "curation_report.json",
        "metadata.json",
        "samples.json",
    }
    actual_entries = {path.name for path in staging_dir.iterdir()}
    if actual_entries != expected_entries:
        raise CurationError(
            "staged export failed validation: metadata-only export contains " f"unexpected entries {sorted(actual_entries - expected_entries)!r}"
        )
    actual_paths = tuple(record.terminal.relative_filepath for record in validated)
    if actual_paths != plan.kept_paths:
        raise CurationError("staged export failed validation: retained paths differ from " "the approved curation plan")
    expected_hashes = {sample.terminal.relative_filepath: sample.sha256 for sample in plan.selected}
    actual_hashes = {record.terminal.relative_filepath: record.integrity.sha256 for record in validated}
    if actual_hashes != expected_hashes:
        raise CurationError("staged export failed validation: media hashes differ from " "the approved curation plan")
    _, current_source_hash = _read_source_samples(plan.source_dir)
    if current_source_hash != plan.source_samples_sha256:
        raise CurationError("staged export failed validation: source samples.json changed " "during materialization")


def _target_report_identity(
    target: Path,
) -> tuple[str | None, str | None, str | None]:
    report_path = target / "curation_report.json"
    try:
        report_stat = report_path.lstat()
    except FileNotFoundError:
        return None, None, None
    except OSError as exc:
        raise CurationError(f"could not inspect existing target report {report_path}: {exc}") from exc
    if stat.S_ISLNK(report_stat.st_mode):
        raise CurationError(f"existing target report may not be a symbolic link: {report_path}")
    if not stat.S_ISREG(report_stat.st_mode):
        raise CurationError(f"existing target report must be a regular file: {report_path}")
    try:
        payload_bytes = report_path.read_bytes()
    except OSError as exc:
        raise CurationError(f"could not read existing target report {report_path}: {exc}") from exc
    report_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeError, json.JSONDecodeError):
        return report_sha256, None, None
    if not isinstance(payload, dict):
        return report_sha256, None, None
    source_export = payload.get("source_export")
    source_samples_sha256 = payload.get("source_samples_sha256")
    return (
        report_sha256,
        source_export if isinstance(source_export, str) else None,
        (source_samples_sha256 if isinstance(source_samples_sha256, str) else None),
    )


def _capture_target_state(target: Path) -> ExpectedTargetState:
    try:
        initial_stat = target.lstat()
    except FileNotFoundError:
        return ExpectedTargetState(exists=False)
    except OSError as exc:
        raise CurationError(f"could not inspect target export {target}: {exc}") from exc
    if stat.S_ISLNK(initial_stat.st_mode):
        raise CurationError(f"target export may not be a symbolic link: {target}")
    if not stat.S_ISDIR(initial_stat.st_mode):
        raise CurationError(f"target export must be a directory: {target}")
    try:
        canonical_path = target.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CurationError(f"could not resolve existing target export {target}: {exc}") from exc
    report_sha256, source_export, source_samples_sha256 = _target_report_identity(target)
    try:
        final_stat = target.lstat()
    except OSError as exc:
        raise CurationError(f"target changed during state capture: {target}: {exc}") from exc
    initial_identity = (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_mode,
    )
    final_identity = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_mode,
    )
    if initial_identity != final_identity:
        raise CurationError(f"target changed during state capture: {target}")
    return ExpectedTargetState(
        exists=True,
        canonical_path=canonical_path,
        device=final_stat.st_dev,
        inode=final_stat.st_ino,
        report_sha256=report_sha256,
        report_source_export=source_export,
        report_source_samples_sha256=source_samples_sha256,
    )


def _require_unchanged_target(
    target: Path,
    expected: ExpectedTargetState,
) -> None:
    try:
        actual = _capture_target_state(target)
    except CurationError as exc:
        raise CurationError(f"target changed during materialization: {target}: {exc}") from exc
    if actual != expected:
        raise CurationError("target changed during materialization; concurrent target was " f"preserved: {target}")


def _same_pinned_identity(
    expected: ExpectedTargetState,
    actual: ExpectedTargetState,
) -> bool:
    return (
        expected.exists
        and actual.exists
        and expected.device == actual.device
        and expected.inode == actual.inode
        and expected.report_sha256 == actual.report_sha256
        and expected.report_source_export == actual.report_source_export
        and expected.report_source_samples_sha256 == actual.report_source_samples_sha256
    )


def _capture_target_state_at(
    parent: _PinnedTargetParent,
    target_name: str,
    canonical_path: Path,
) -> ExpectedTargetState:
    state = _capture_target_state(
        _descriptor_path(parent.descriptor) / target_name,
    )
    if not state.exists:
        return state
    return replace(
        state,
        canonical_path=canonical_path,
    )


def _require_unchanged_target_at(
    parent: _PinnedTargetParent,
    target_name: str,
    canonical_path: Path,
    expected: ExpectedTargetState,
) -> None:
    try:
        actual = _capture_target_state_at(
            parent,
            target_name,
            canonical_path,
        )
    except CurationError as exc:
        raise CurationError(f"target changed during materialization: {canonical_path}: {exc}") from exc
    if actual != expected:
        raise CurationError("target changed during materialization; concurrent target was " f"preserved: {canonical_path}")


def _has_atomic_rename_capability(
    capabilities: int,
    valid: int,
) -> bool:
    return bool(capabilities & valid & _VOL_CAP_INT_RENAME_EXCL)


def supports_atomic_directory_noreplace(path: Path) -> bool:
    """Return whether the path's volume supports atomic exclusive rename."""
    if sys.platform == "darwin":
        try:
            resolved = path.expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return False
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            getattrlist = libc.getattrlist
        except AttributeError:
            return False
        getattrlist.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(_AttributeList),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_ulong,
        ]
        getattrlist.restype = ctypes.c_int
        attributes = _AttributeList(
            _ATTR_BIT_MAP_COUNT,
            0,
            0,
            _ATTR_VOL_INFO | _ATTR_VOL_CAPABILITIES,
            0,
            0,
            0,
        )
        buffer = _VolumeCapabilitiesBuffer()
        result = getattrlist(
            os.fsencode(resolved),
            ctypes.byref(attributes),
            ctypes.byref(buffer),
            ctypes.sizeof(buffer),
            0,
        )
        if result != 0:
            return False
        capabilities = buffer.capabilities.capabilities[_VOL_CAPABILITIES_INTERFACES]
        valid = buffer.capabilities.valid[_VOL_CAPABILITIES_INTERFACES]
        return _has_atomic_rename_capability(capabilities, valid)
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            _ = libc.renameat2
        except AttributeError:
            return False
        return True
    return False


def _path_device(path: Path) -> int:
    return path.stat().st_dev


def _deepest_existing_ancestor(
    path: Path,
    *,
    boundary: Path,
) -> Path:
    candidate = path
    while True:
        try:
            candidate.stat()
        except FileNotFoundError:
            if candidate == boundary:
                raise CurationError(f"curated target mount disappeared during validation: {boundary}") from None
            parent = candidate.parent
            if parent == candidate or not candidate.is_relative_to(boundary):
                raise CurationError(f"curated target escaped its mounted filesystem during validation: {path}") from None
            candidate = parent
            continue
        except OSError as exc:
            raise CurationError(f"could not inspect curated target storage at {candidate}: {exc}") from exc
        return candidate


def _validate_target_storage(
    target: Path,
    *,
    mount_root: Path = DEFAULT_TARGET_MOUNT_ROOT,
    approved_root: Path = DEFAULT_CURATED_EXPORT_ROOT,
    is_mount_fn: Callable[[Path], bool] = os.path.ismount,
    capability_probe: Callable[[Path], bool] = supports_atomic_directory_noreplace,
    device_fn: Callable[[Path], int] = _path_device,
) -> _ValidatedTargetStorage:
    if not target.expanduser().is_absolute():
        raise CurationError(f"curated target must be an absolute path: {target}")
    if not is_mount_fn(mount_root):
        raise CurationError(f"curated target volume must be an actual mounted filesystem: {mount_root}")
    try:
        resolved_mount = mount_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CurationError(f"could not resolve curated target mount {mount_root}: {exc}") from exc
    resolved_approved_root = approved_root.expanduser().resolve(strict=False)
    resolved_target = target.expanduser().resolve(strict=False)
    if resolved_approved_root == resolved_mount or not resolved_approved_root.is_relative_to(resolved_mount):
        raise CurationError(f"approved curated export root must be a strict descendant of {resolved_mount}: {resolved_approved_root}")
    if resolved_target == resolved_approved_root or not resolved_target.is_relative_to(resolved_approved_root):
        raise CurationError(f"curated target must be a strict descendant of {resolved_approved_root}: {resolved_target}")
    mount_identity = _capture_directory_identity(resolved_mount)
    if mount_identity is None:
        raise CurationError(f"curated target mount is not a real directory: {resolved_mount}")
    probe_path = _deepest_existing_ancestor(
        resolved_target,
        boundary=resolved_mount,
    )
    try:
        mount_device = device_fn(resolved_mount)
        target_device = device_fn(probe_path)
    except OSError as exc:
        raise CurationError(f"could not inspect curated target filesystem: {exc}") from exc
    if target_device != mount_device:
        raise CurationError("curated target must remain on the same filesystem as " f"{resolved_mount}; nested mounts are not allowed: {probe_path}")
    if not capability_probe(probe_path):
        raise CurationError(f"curated target volume lacks atomic no-clobber directory rename support: {probe_path}")
    if _capture_directory_identity(resolved_mount) != mount_identity or not is_mount_fn(resolved_mount):
        raise CurationError(f"curated target mount changed during validation: {resolved_mount}")
    return _ValidatedTargetStorage(
        mount_path=resolved_mount,
        target_path=resolved_target,
        mount_identity=mount_identity,
        is_mount_fn=is_mount_fn,
    )


def _preserve_unexpected_backup(
    target: Path,
    backup: Path,
) -> Path:
    target_occupied = target.is_symlink() or target.exists()
    if not target_occupied:
        try:
            _rename_directory_noreplace(backup, target)
        except (CurationError, OSError):
            pass
        else:
            return target

    recovery = target.parent / f".{target.name}.recovery-{uuid4().hex}"
    try:
        _rename_directory_noreplace(backup, recovery)
    except (CurationError, OSError) as exc:
        raise CurationError("could not restore or move the preserved target; " f"recover it manually from {backup}: {exc}") from exc
    return recovery


def _capture_directory_identity(path: Path) -> _DirectoryIdentity | None:
    try:
        path_stat = path.lstat()
    except OSError:
        return None
    if not stat.S_ISDIR(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        return None
    return _DirectoryIdentity(
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
    )


def _identity_from_stat(path_stat: os.stat_result) -> _DirectoryIdentity | None:
    if not stat.S_ISDIR(path_stat.st_mode):
        return None
    return _DirectoryIdentity(
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
    )


def _entry_identity_from_stat(path_stat: os.stat_result) -> _EntryIdentity:
    return _EntryIdentity(
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
        file_type=stat.S_IFMT(path_stat.st_mode),
    )


def _capture_entry_identity_at(
    parent_fd: int,
    name: str,
) -> _EntryIdentity | None:
    try:
        path_stat = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError:
        return None
    return _entry_identity_from_stat(path_stat)


def _capture_directory_identity_at(
    parent_fd: int,
    name: str,
) -> _DirectoryIdentity | None:
    try:
        path_stat = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError:
        return None
    return _identity_from_stat(path_stat)


def _retire_owned_directory_at(
    parent_fd: int,
    name: str,
    owned_fd: int,
    expected: _DirectoryIdentity,
) -> bool:
    try:
        descriptor_identity = _identity_from_stat(os.fstat(owned_fd))
    except OSError:
        return False
    if descriptor_identity != expected:
        return False
    if _capture_directory_identity_at(parent_fd, name) != expected:
        return False
    if ".building-" in name:
        base = name.split(".building-", 1)[0]
        retired_name = f"{base}.retired-staging-{uuid4().hex}"
    elif ".backup-" in name:
        base = name.split(".backup-", 1)[0]
        retired_name = f"{base}.retired-backup-{uuid4().hex}"
    else:
        retired_name = f".retired-directory-{uuid4().hex}"
    return _retire_owned_entry_at(
        parent_fd,
        name,
        _EntryIdentity(
            device=expected.device,
            inode=expected.inode,
            file_type=stat.S_IFDIR,
        ),
        retired_name,
    )


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _descriptor_path(descriptor: int) -> Path:
    if sys.platform == "darwin":
        import fcntl

        try:
            raw_path = fcntl.fcntl(
                descriptor,
                50,  # F_GETPATH
                b"\0" * 1024,
            )
        except OSError as exc:
            raise CurationError(f"could not resolve pinned directory descriptor: {exc}") from exc
        return Path(raw_path.split(b"\0", 1)[0].decode())
    proc_path = Path("/proc/self/fd") / str(descriptor)
    if proc_path.exists():
        return proc_path
    return Path("/dev/fd") / str(descriptor)


def _open_directory_at(
    parent_fd: int,
    name: str,
) -> int:
    return os.open(
        name,
        _directory_open_flags(),
        dir_fd=parent_fd,
    )


@contextmanager
def _open_pinned_target_parent(
    storage: _ValidatedTargetStorage,
) -> Iterator[_PinnedTargetParent]:
    root_fd: int | None = None
    current_fd: int | None = None
    try:
        try:
            root_fd = os.open(
                storage.mount_path,
                _directory_open_flags(),
            )
        except OSError as exc:
            raise CurationError(f"could not pin curated target mount {storage.mount_path}: {exc}") from exc
        current_fd = root_fd
        try:
            opened_root_identity = _identity_from_stat(os.fstat(root_fd))
        except OSError as exc:
            raise CurationError(f"could not inspect pinned curated target mount {storage.mount_path}: {exc}") from exc
        path_root_identity = _capture_directory_identity(storage.mount_path)
        if (
            opened_root_identity != storage.mount_identity
            or path_root_identity != storage.mount_identity
            or not storage.is_mount_fn(storage.mount_path)
        ):
            raise CurationError(f"curated target mount changed after validation: {storage.mount_path}")

        try:
            relative_parent = storage.target_path.parent.relative_to(storage.mount_path)
        except ValueError as exc:
            raise CurationError(f"approved target parent escaped the mounted volume: {storage.target_path.parent}") from exc
        for component in relative_parent.parts:
            child_fd: int | None = None
            try:
                try:
                    child_fd = _open_directory_at(current_fd, component)
                except FileNotFoundError:
                    with suppress(FileExistsError):
                        os.mkdir(
                            component,
                            mode=0o755,
                            dir_fd=current_fd,
                        )
                    child_fd = _open_directory_at(current_fd, component)
            except OSError as exc:
                raise CurationError(
                    "approved target parent changed or contains an unsafe " f"component {component!r}: {storage.target_path.parent}: {exc}"
                ) from exc
            try:
                child_identity = _identity_from_stat(os.fstat(child_fd))
            except OSError as exc:
                os.close(child_fd)
                raise CurationError(f"could not inspect approved target parent {storage.target_path.parent}: {exc}") from exc
            if child_identity is None or child_identity.device != storage.mount_identity.device:
                os.close(child_fd)
                raise CurationError("approved target parent must remain on the validated " f"filesystem: {storage.target_path.parent}")
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd

        assert current_fd is not None
        final_identity = _identity_from_stat(os.fstat(current_fd))
        if final_identity is None:
            raise CurationError(f"approved target parent is not a directory: {storage.target_path.parent}")
        yield _PinnedTargetParent(
            logical_path=storage.target_path.parent,
            descriptor=current_fd,
            identity=final_identity,
        )
    finally:
        if current_fd is not None and current_fd != root_fd:
            with suppress(OSError):
                os.close(current_fd)
        if root_fd is not None:
            with suppress(OSError):
                os.close(root_fd)


def _create_owned_staging(
    parent: _PinnedTargetParent,
    target_name: str,
) -> _OwnedDirectory:
    for _ in range(100):
        name = f".{target_name}.building-{uuid4().hex}"
        try:
            os.mkdir(
                name,
                mode=0o700,
                dir_fd=parent.descriptor,
            )
        except FileExistsError:
            continue
        except OSError as exc:
            raise CurationError(f"could not create curated export staging directory: {exc}") from exc
        expected = _capture_directory_identity_at(parent.descriptor, name)
        if expected is None:
            raise CurationError(f"could not pin owned staging directory: {parent.logical_path / name}")
        descriptor: int | None = None
        try:
            descriptor = _open_directory_at(parent.descriptor, name)
            opened_identity = _identity_from_stat(os.fstat(descriptor))
        except OSError as exc:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            raise CurationError(f"could not pin owned staging directory: {parent.logical_path / name}: {exc}") from exc
        if opened_identity != expected:
            os.close(descriptor)
            raise CurationError(f"owned staging directory changed during creation: {parent.logical_path / name}")
        return _OwnedDirectory(
            name=name,
            descriptor=descriptor,
            identity=expected,
        )
    raise CurationError(f"could not allocate unique staging directory below {parent.logical_path}")


def _require_pinned_parent_path(
    parent: _PinnedTargetParent,
) -> None:
    if _capture_directory_identity(parent.logical_path) != parent.identity:
        raise CurationError(f"approved target parent changed during materialization: {parent.logical_path}")


def _require_published_staging_identity(
    parent: _PinnedTargetParent,
    target_name: str,
    staging: _OwnedDirectory,
) -> None:
    published_identity = _capture_directory_identity_at(
        parent.descriptor,
        target_name,
    )
    if published_identity == staging.identity:
        return
    try:
        owned_location = _descriptor_path(staging.descriptor)
    except CurationError:
        owned_location = Path(f"<open directory descriptor {staging.descriptor}>")
    raise CurationError(
        "published directory did not match the owned staging identity; "
        f"the unexpected target was preserved at {parent.logical_path / target_name} "
        f"and the owned staging directory remains at {owned_location}"
    )


def _rename_directory_noreplace_at(
    parent_fd: int,
    source_name: str,
    target_name: str,
) -> None:
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameatx_np = libc.renameatx_np
        except AttributeError as exc:
            raise CurationError("atomic descriptor-relative directory publication is unavailable on macOS") from exc
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            parent_fd,
            os.fsencode(source_name),
            parent_fd,
            os.fsencode(target_name),
            _RENAME_EXCL,
        )
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise CurationError("atomic descriptor-relative directory publication is unavailable on Linux") from exc
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_fd,
            os.fsencode(source_name),
            parent_fd,
            os.fsencode(target_name),
            _RENAME_NOREPLACE,
        )
    else:
        raise CurationError(f"descriptor-relative publication is unsupported on platform {sys.platform!r}")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            target_name,
        )


def _entry_exists_at(
    parent_fd: int,
    name: str,
) -> bool:
    try:
        os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _restore_unexpected_retired_entry_at(
    parent_fd: int,
    original_name: str,
    retired_name: str,
) -> None:
    if not _entry_exists_at(parent_fd, original_name):
        try:
            _rename_directory_noreplace_at(
                parent_fd,
                retired_name,
                original_name,
            )
        except (CurationError, OSError):
            pass
        else:
            return
    recovery_name = f".{Path(original_name).name}.recovery-{uuid4().hex}"
    with suppress(CurationError, OSError):
        _rename_directory_noreplace_at(
            parent_fd,
            retired_name,
            recovery_name,
        )


def _retire_owned_entry_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    retired_name: str,
) -> bool:
    if _capture_entry_identity_at(parent_fd, name) != expected:
        return False
    try:
        _rename_directory_noreplace_at(
            parent_fd,
            name,
            retired_name,
        )
    except (CurationError, OSError):
        return False
    if _capture_entry_identity_at(parent_fd, retired_name) == expected:
        return True
    _restore_unexpected_retired_entry_at(
        parent_fd,
        name,
        retired_name,
    )
    return False


def _preserve_unexpected_backup_at(
    parent: _PinnedTargetParent,
    target_name: str,
    backup_name: str,
) -> Path:
    if not _entry_exists_at(parent.descriptor, target_name):
        try:
            _rename_directory_noreplace_at(
                parent.descriptor,
                backup_name,
                target_name,
            )
        except (CurationError, OSError):
            pass
        else:
            return parent.logical_path / target_name

    recovery_name = f".{target_name}.recovery-{uuid4().hex}"
    try:
        _rename_directory_noreplace_at(
            parent.descriptor,
            backup_name,
            recovery_name,
        )
    except (CurationError, OSError) as exc:
        raise CurationError(
            "could not restore or move the preserved target; recover it " f"manually from {parent.logical_path / backup_name}: {exc}"
        ) from exc
    return parent.logical_path / recovery_name


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically rename a directory while refusing to replace any target."""
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            os.fsencode(source),
            os.fsencode(target),
            _RENAME_EXCL,
        )
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise CurationError("atomic no-clobber directory publication is unavailable on " "this Linux system") from exc
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _LINUX_AT_FDCWD,
            os.fsencode(source),
            _LINUX_AT_FDCWD,
            os.fsencode(target),
            _RENAME_NOREPLACE,
        )
    elif os.name == "nt":
        os.rename(source, target)
        return
    else:
        raise CurationError("atomic no-clobber directory publication is unsupported on " f"platform {sys.platform!r}")

    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(target),
        )


@contextmanager
def _exclusive_target_lock(
    target: Path,
    *,
    parent_fd: int | None = None,
) -> Iterator[None]:
    lock_path = target.parent / f".{target.name}.lock"
    lock_name = lock_path.name
    descriptor: int | None = None
    opened_parent_fd: int | None = None
    owned_identity: _EntryIdentity | None = None
    try:
        if parent_fd is None:
            try:
                opened_parent_fd = os.open(
                    target.parent,
                    _directory_open_flags(),
                )
            except OSError as exc:
                raise CurationError(f"could not pin curated export lock parent {target.parent}: {exc}") from exc
            parent_fd = opened_parent_fd
        try:
            descriptor = os.open(
                lock_name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise CurationError(f"curated export lock already exists: {lock_path}") from exc
        except OSError as exc:
            raise CurationError(f"could not acquire curated export lock {lock_path}: {exc}") from exc
        try:
            lock_stat = os.fstat(descriptor)
            owned_identity = _entry_identity_from_stat(lock_stat)
            token = f"{os.getpid()}:{uuid4().hex}\n".encode()
            os.write(descriptor, token)
            os.fsync(descriptor)
        except OSError as exc:
            raise CurationError(f"could not initialize curated export lock {lock_path}: {exc}") from exc
        yield
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_failed = False
        if parent_fd is not None and descriptor is not None and owned_identity is not None:
            retired_name = f".{target.name}.released-lock-{uuid4().hex}"
            cleanup_failed = not _retire_owned_entry_at(
                parent_fd,
                lock_name,
                owned_identity,
                retired_name,
            )
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if opened_parent_fd is not None:
            with suppress(OSError):
                os.close(opened_parent_fd)
        if cleanup_failed and not active_exception:
            raise CurationError("could not safely release curated export lock without " f"touching a foreign replacement: {lock_path}")


def materialize_curated_export(
    plan: CurationPlan,
    *,
    overwrite: bool = False,
    confirmed: bool = False,
    expected_target_state: ExpectedTargetState | None = None,
) -> Path:
    """Atomically materialize metadata that references original source media."""
    target = plan.target_dir
    if target.is_symlink():
        raise CurationError(f"target export may not be a symbolic link: {target}")
    current_resolved_target = target.resolve(strict=False)
    if current_resolved_target != plan.resolved_target_dir:
        raise CurationError("target export resolution changed after planning: " f"{target} -> {current_resolved_target}")
    if target == plan.requested_source_dir or target.is_relative_to(plan.requested_source_dir):
        raise CurationError("target export must not be inside the source export")
    if plan.requested_source_dir.is_relative_to(target):
        raise CurationError("target export must not contain the source export")
    if current_resolved_target == plan.source_dir or current_resolved_target.is_relative_to(plan.source_dir):
        raise CurationError("target export must not be inside the source export")
    if plan.source_dir.is_relative_to(current_resolved_target):
        raise CurationError("target export must not contain the source export")

    storage = _validate_target_storage(target)
    if storage.target_path != plan.resolved_target_dir:
        raise CurationError("target export resolution changed during storage validation: " f"{target} -> {storage.target_path}")
    with _open_pinned_target_parent(storage) as parent:
        if expected_target_state is None:
            initial_target_state = _capture_target_state_at(
                parent,
                target.name,
                plan.resolved_target_dir,
            )
        else:
            _require_unchanged_target_at(
                parent,
                target.name,
                plan.resolved_target_dir,
                expected_target_state,
            )
            initial_target_state = expected_target_state
        if initial_target_state.exists and initial_target_state.canonical_path != plan.resolved_target_dir:
            raise CurationError("target export resolution changed after planning: " f"{target} -> {initial_target_state.canonical_path}")

        with _exclusive_target_lock(
            target,
            parent_fd=parent.descriptor,
        ):
            _require_unchanged_target_at(
                parent,
                target.name,
                plan.resolved_target_dir,
                initial_target_state,
            )
            preflight = initial_target_state
            if preflight.exists:
                if not overwrite:
                    raise CurationError(f"target export already exists: {target}")
                if not confirmed:
                    raise CurationError("overwrite requires explicit confirmation")

            staging = _create_owned_staging(
                parent,
                target.name,
            )
            staging_rename_returned = False
            try:
                _require_pinned_parent_path(parent)
                _build_export_at(
                    plan,
                    staging.descriptor,
                )
                _validate_staged_export(
                    plan,
                    _descriptor_path(staging.descriptor),
                )
                _require_pinned_parent_path(parent)
                with suppress(OSError):
                    os.fsync(staging.descriptor)
                if not preflight.exists:
                    _require_unchanged_target_at(
                        parent,
                        target.name,
                        plan.resolved_target_dir,
                        preflight,
                    )
                    try:
                        _rename_directory_noreplace_at(
                            parent.descriptor,
                            staging.name,
                            target.name,
                        )
                        staging_rename_returned = True
                    except OSError as exc:
                        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                            raise CurationError(
                                "could not publish curated export because a " f"concurrent target appeared and was preserved: {target}"
                            ) from exc
                        raise CurationError(f"could not publish curated export to {target}: {exc}") from exc
                    _require_published_staging_identity(
                        parent,
                        target.name,
                        staging,
                    )
                    _require_pinned_parent_path(parent)
                    return target

                backup_name = f".{target.name}.backup-{uuid4().hex}"
                backup = parent.logical_path / backup_name
                _require_unchanged_target_at(
                    parent,
                    target.name,
                    plan.resolved_target_dir,
                    preflight,
                )
                try:
                    _rename_directory_noreplace_at(
                        parent.descriptor,
                        target.name,
                        backup_name,
                    )
                except (CurationError, OSError) as exc:
                    raise CurationError("could not preserve existing target before " f"publication: {target}: {exc}") from exc
                try:
                    backup_state = _capture_target_state_at(
                        parent,
                        backup_name,
                        plan.resolved_target_dir,
                    )
                except CurationError as exc:
                    preserved_at = _preserve_unexpected_backup_at(
                        parent,
                        target.name,
                        backup_name,
                    )
                    raise CurationError(
                        "backup identity could not be verified against the "
                        "confirmed target; publication aborted and the "
                        f"unexpected entry was preserved at {preserved_at}"
                    ) from exc
                if not _same_pinned_identity(preflight, backup_state):
                    preserved_at = _preserve_unexpected_backup_at(
                        parent,
                        target.name,
                        backup_name,
                    )
                    raise CurationError(
                        "backup identity did not match the confirmed target; "
                        "publication aborted and the unexpected directory was "
                        f"preserved at {preserved_at}"
                    )
                try:
                    post_backup_target = _capture_target_state_at(
                        parent,
                        target.name,
                        plan.resolved_target_dir,
                    )
                except CurationError as exc:
                    preserved_at = _preserve_unexpected_backup_at(
                        parent,
                        target.name,
                        backup_name,
                    )
                    raise CurationError(
                        "target changed after the confirmed target was backed "
                        "up; publication aborted and the confirmed backup was "
                        f"preserved at {preserved_at}"
                    ) from exc
                if post_backup_target.exists:
                    preserved_at = _preserve_unexpected_backup_at(
                        parent,
                        target.name,
                        backup_name,
                    )
                    raise CurationError(
                        "target changed after the confirmed target was backed "
                        "up; publication aborted and the confirmed backup was "
                        f"preserved at {preserved_at}"
                    )
                try:
                    _rename_directory_noreplace_at(
                        parent.descriptor,
                        staging.name,
                        target.name,
                    )
                    staging_rename_returned = True
                except BaseException as exc:
                    preserved_at = _preserve_unexpected_backup_at(
                        parent,
                        target.name,
                        backup_name,
                    )
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    if preserved_at == target:
                        raise CurationError("could not publish curated export; previous target was restored") from exc
                    raise CurationError(
                        "could not publish curated export because a "
                        "concurrent target appeared; the concurrent target "
                        "was preserved and the previous target is recoverable "
                        f"at {preserved_at}"
                    ) from exc
                _require_published_staging_identity(
                    parent,
                    target.name,
                    staging,
                )
                _require_pinned_parent_path(parent)
                backup_before_delete = _capture_target_state_at(
                    parent,
                    backup_name,
                    plan.resolved_target_dir,
                )
                if not _same_pinned_identity(preflight, backup_before_delete):
                    raise CurationError(
                        "curated export was published but backup identity " "changed; the unexpected backup was retained at " f"{backup}"
                    )
                backup_fd: int | None = None
                try:
                    backup_fd = _open_directory_at(
                        parent.descriptor,
                        backup_name,
                    )
                    backup_identity = _identity_from_stat(os.fstat(backup_fd))
                    if preflight.device is None or preflight.inode is None:
                        raise CurationError("curated export was published but the confirmed " "target identity was incomplete")
                    expected_backup_identity = _DirectoryIdentity(
                        device=preflight.device,
                        inode=preflight.inode,
                    )
                    if backup_identity != expected_backup_identity or not _retire_owned_directory_at(
                        parent.descriptor,
                        backup_name,
                        backup_fd,
                        expected_backup_identity,
                    ):
                        raise CurationError("curated export was published but the old backup " f"could not be safely removed: {backup}")
                except OSError as exc:
                    raise CurationError("curated export was published but old backup remains " f"at {backup}: {exc}") from exc
                finally:
                    if backup_fd is not None:
                        with suppress(OSError):
                            os.close(backup_fd)
                return target
            finally:
                if not staging_rename_returned:
                    _retire_owned_directory_at(
                        parent.descriptor,
                        staging.name,
                        staging.descriptor,
                        staging.identity,
                    )
                with suppress(OSError):
                    os.close(staging.descriptor)


class _ArgumentParser(argparse.ArgumentParser):
    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: Any = None,
    ) -> Any:
        parsed = super().parse_args(args, namespace)
        if parsed.yes and not parsed.overwrite:
            self.error("--yes requires --overwrite")
        if parsed.overwrite and not parsed.create:
            self.error("--overwrite requires --create")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the curation report without writing",
    )
    mode.add_argument(
        "--create",
        action="store_true",
        help="Atomically materialize the curated export",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_EXPORT_DIR,
        help="Original terminal FiftyOne export",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_EXPORT_DIR,
        help="Curated terminal export",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the exact target export after confirmation",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm overwrite noninteractively",
    )
    return parser


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _default_enrichments(source_dir: Path) -> Mapping[str, Enrichment] | None:
    if source_dir.resolve(strict=False) != DEFAULT_SOURCE_EXPORT_DIR.resolve(strict=False):
        return None
    candidates = load_lineage_candidates_from_paths(
        DEFAULT_UPSTREAM_EXPORT_DIRS,
        DEFAULT_MANIFEST_PATHS,
    )
    index = LineageIndex.from_candidates(candidates)
    return MappingProxyType({record.relative_filepath: index.enrich(record) for record in load_terminal_records(source_dir)})


def run(
    args: argparse.Namespace,
    *,
    policy: CurationPolicy = DEFAULT_POLICY,
    input_fn: Callable[[str], str] = input,
    isatty_fn: Callable[[], bool] | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run a read-only audit or guarded materialization."""
    enrichments = _default_enrichments(args.source)
    plan = build_curation_plan(
        args.source,
        args.target,
        policy=policy,
        enrichments=enrichments,
    )
    if not args.create:
        output_fn(json.dumps(_report_payload(plan), indent=2, sort_keys=True))
        return 0

    expected_target_state = _capture_target_state(plan.target_dir)
    confirmed = args.yes
    if expected_target_state.exists and args.overwrite and not confirmed:
        isatty = sys.stdin.isatty if isatty_fn is None else isatty_fn
        if not isatty():
            raise CurationError("noninteractive overwrite requires --yes")
        prompt = f"Type the exact target path '{plan.target_dir}' " "to replace the curated export: "
        try:
            confirmed = input_fn(prompt) == str(plan.target_dir)
        except EOFError:
            confirmed = False
        if not confirmed:
            raise CurationError(f"overwrite declined for curated export: {plan.target_dir}")

    materialize_curated_export(
        plan,
        overwrite=args.overwrite,
        confirmed=confirmed,
        expected_target_state=expected_target_state,
    )
    output_fn(json.dumps(_report_payload(plan), indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except CurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
