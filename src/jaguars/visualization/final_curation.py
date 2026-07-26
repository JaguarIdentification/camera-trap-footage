"""Deterministic construction of the final curated terminal export."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
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
DEFAULT_TARGET_EXPORT_DIR = Path("data/intermediate/v1/fo_jaguars/labeled_segmented_jaguars_final_curated_v1")
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


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _build_export(
    plan: CurationPlan,
    staging_dir: Path,
) -> None:
    samples_payload = {"samples": [_curated_sample_payload(sample, plan.policy_version) for sample in plan.selected]}
    _write_json(staging_dir / "samples.json", samples_payload)
    _write_json(staging_dir / "metadata.json", _metadata_payload(plan))
    _write_json(staging_dir / "curation_report.json", _report_payload(plan))


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


def _preserve_unexpected_backup(
    target: Path,
    backup: Path,
) -> Path:
    target_occupied = target.is_symlink() or target.exists()
    if not target_occupied:
        try:
            os.replace(backup, target)
        except OSError:
            pass
        else:
            return target

    recovery = target.parent / f".{target.name}.recovery-{uuid4().hex}"
    try:
        os.replace(backup, recovery)
    except OSError as exc:
        raise CurationError(
            "backup identity did not match the confirmed target and could not " f"be restored or moved; preserve it manually at {backup}: {exc}"
        ) from exc
    return recovery


@contextmanager
def _exclusive_target_lock(target: Path) -> Iterator[None]:
    lock_path = target.parent / f".{target.name}.lock"
    descriptor: int | None = None
    owned_identity: tuple[int, int] | None = None
    try:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise CurationError(f"curated export lock already exists: {lock_path}") from exc
        except OSError as exc:
            raise CurationError(f"could not acquire curated export lock {lock_path}: {exc}") from exc
        try:
            lock_stat = os.fstat(descriptor)
            owned_identity = (lock_stat.st_dev, lock_stat.st_ino)
            token = f"{os.getpid()}:{uuid4().hex}\n".encode()
            os.write(descriptor, token)
            os.fsync(descriptor)
        except OSError as exc:
            raise CurationError(f"could not initialize curated export lock {lock_path}: {exc}") from exc
        yield
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_error: OSError | None = None
        if descriptor is not None and owned_identity is not None:
            try:
                current_stat = lock_path.lstat()
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_error = exc
            else:
                current_identity = (
                    current_stat.st_dev,
                    current_stat.st_ino,
                )
                if stat.S_ISREG(current_stat.st_mode) and current_identity == owned_identity:
                    try:
                        lock_path.unlink()
                    except OSError as exc:
                        cleanup_error = exc
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if cleanup_error is not None and not active_exception:
            raise CurationError(f"could not release curated export lock {lock_path}: " f"{cleanup_error}") from cleanup_error


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

    if expected_target_state is None:
        initial_target_state = _capture_target_state(target)
    else:
        _require_unchanged_target(target, expected_target_state)
        initial_target_state = expected_target_state
    if initial_target_state.exists and initial_target_state.canonical_path != plan.resolved_target_dir:
        raise CurationError("target export resolution changed after planning: " f"{target} -> {initial_target_state.canonical_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_target_lock(target):
        _require_unchanged_target(target, initial_target_state)
        preflight = initial_target_state
        if preflight.exists:
            if not overwrite:
                raise CurationError(f"target export already exists: {target}")
            if not confirmed:
                raise CurationError("overwrite requires explicit confirmation")
        elif target.resolve(strict=False) != plan.resolved_target_dir:
            raise CurationError("target export resolution changed after planning: " f"{target} -> {target.resolve(strict=False)}")

        staging = Path(
            tempfile.mkdtemp(
                dir=target.parent,
                prefix=f".{target.name}.building-",
            )
        )
        try:
            _build_export(plan, staging)
            _validate_staged_export(plan, staging)
            if not preflight.exists:
                _require_unchanged_target(target, preflight)
                try:
                    os.replace(staging, target)
                except OSError as exc:
                    raise CurationError(f"could not publish curated export to {target}: {exc}") from exc
                return target

            backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
            _require_unchanged_target(target, preflight)
            try:
                os.replace(target, backup)
            except OSError as exc:
                raise CurationError(f"could not preserve existing target before publication: " f"{target}: {exc}") from exc
            try:
                backup_state = _capture_target_state(backup)
            except CurationError as exc:
                preserved_at = _preserve_unexpected_backup(target, backup)
                raise CurationError(
                    "backup identity could not be verified against the "
                    "confirmed target; publication aborted and the unexpected "
                    f"entry was preserved at {preserved_at}"
                ) from exc
            if not _same_pinned_identity(preflight, backup_state):
                preserved_at = _preserve_unexpected_backup(target, backup)
                raise CurationError(
                    "backup identity did not match the confirmed target; "
                    "publication aborted and the unexpected directory was "
                    f"preserved at {preserved_at}"
                )
            try:
                post_backup_target = _capture_target_state(target)
            except CurationError as exc:
                preserved_at = _preserve_unexpected_backup(target, backup)
                raise CurationError(
                    "target changed after the confirmed target was backed up; "
                    "publication aborted and the confirmed backup was "
                    f"preserved at {preserved_at}"
                ) from exc
            if post_backup_target.exists:
                preserved_at = _preserve_unexpected_backup(target, backup)
                raise CurationError(
                    "target changed after the confirmed target was backed up; "
                    "publication aborted and the confirmed backup was "
                    f"preserved at {preserved_at}"
                )
            try:
                os.replace(staging, target)
            except BaseException as exc:
                if target.is_symlink() or target.exists():
                    preserved_at = _preserve_unexpected_backup(target, backup)
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    raise CurationError(
                        "could not publish curated export because a concurrent "
                        "target appeared; the concurrent target was preserved "
                        "and the previous target is recoverable at "
                        f"{preserved_at}"
                    ) from exc
                try:
                    if backup.exists():
                        os.replace(backup, target)
                except OSError as restore_error:
                    raise CurationError(
                        "could not publish curated export or restore the " f"previous target; recoverable backup remains at {backup}"
                    ) from restore_error
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise CurationError("could not publish curated export; previous target was " "restored") from exc
            backup_before_delete = _capture_target_state(backup)
            if not _same_pinned_identity(preflight, backup_before_delete):
                raise CurationError("curated export was published but backup identity changed; " f"the unexpected backup was retained at {backup}")
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                raise CurationError("curated export was published but old backup remains at " f"{backup}: {exc}") from exc
            return target
        finally:
            if staging.exists():
                with suppress(OSError):
                    shutil.rmtree(staging)


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
