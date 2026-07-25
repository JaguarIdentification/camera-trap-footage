"""Guarded CLI for the final curated FiftyOne dataset."""

import argparse
import importlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, overload

from jaguars.visualization.final_lineage import LineageIndex, load_lineage_candidates
from jaguars.visualization.final_records import TerminalExportError, load_terminal_records
from jaguars.visualization.final_validation import (
    IntegrityError,
    StorageSafetyError,
    ValidatedRecord,
    validate_mounts,
    validate_records,
    validate_storage_paths,
)

DEFAULT_DATASET_NAME = "JaguarCameraTrap_Final_Curated_v1"
DEFAULT_INTERMEDIATE_DIR = Path("data/intermediate/v1")
DEFAULT_TERMINAL_EXPORT_DIR = DEFAULT_INTERMEDIATE_DIR / "fo_jaguars/labeled_segmented_jaguars_primitive"
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
DEFAULT_STATE_ROOT = Path("/Volumes/CameraTrapPython/fiftyone")
DEFAULT_DATABASE_DIR = DEFAULT_STATE_ROOT / "var/lib/mongo"
DEFAULT_REPORT_DIR = DEFAULT_STATE_ROOT / DEFAULT_DATASET_NAME
DEFAULT_DATASET_DIR = DEFAULT_STATE_ROOT / "datasets"
DEFAULT_CONFIG_PATH = DEFAULT_STATE_ROOT / "config.json"
DEFAULT_MODEL_ZOO_DIR = DEFAULT_STATE_ROOT / "models"
DEFAULT_PLUGINS_DIR = DEFAULT_STATE_ROOT / "plugins"
DEFAULT_MOUNT_ROOTS = (Path("/Volumes/Extreme SSD"), Path("/Volumes/CameraTrapPython"))
DEFAULT_ADDRESS = "localhost"
DEFAULT_PORT = 5151
EXPECTED_SAMPLE_COUNT = 1367
_REPORT_ENRICHMENT_FIELDS = (
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
    "original_filename",
    "source_media_path",
    "source_type",
)

_Namespace = TypeVar("_Namespace")


class FinalDatasetError(RuntimeError):
    """Base class for controlled final-dataset workflow failures."""


class DatasetExistsError(FinalDatasetError):
    """Raised when ordinary creation would replace an existing snapshot."""


class OverwriteDeclinedError(FinalDatasetError):
    """Raised when the exact overwrite confirmation was not supplied."""


@dataclass(frozen=True)
class RuntimePaths:
    intermediate_dir: Path
    terminal_export_dir: Path
    upstream_export_dirs: tuple[Path, ...]
    manifest_paths: tuple[Path, ...]
    state_root: Path
    database_dir: Path
    report_dir: Path
    dataset_dir: Path
    config_path: Path
    model_zoo_dir: Path
    plugins_dir: Path
    mount_roots: tuple[Path, ...]


@dataclass(frozen=True)
class Audit:
    records: tuple[ValidatedRecord, ...]
    terminal_count: int
    lineage_candidate_count: int


@dataclass(frozen=True)
class SnapshotSummary:
    constructed_count: int
    field_population: Mapping[str, int]
    saved_views: tuple[str, ...]


@dataclass(frozen=True)
class RunReport:
    started_at: str
    finished_at: str
    dataset_name: str
    mode: str
    status: str
    paths: Mapping[str, object]
    counts: Mapping[str, int | None]
    hash_validation: Mapping[str, object]
    media_validation: Mapping[str, object]
    lineage: Mapping[str, object]
    field_population: Mapping[str, int]
    views: Mapping[str, object]
    failure: Mapping[str, str] | None

    @classmethod
    def from_audit(
        cls,
        audit: Audit,
        *,
        paths: RuntimePaths | None = None,
        dataset_name: str = DEFAULT_DATASET_NAME,
        mode: str = "audit",
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> "RunReport":
        started = started_at or datetime.now(timezone.utc)
        finished = finished_at or started
        audit_status = "not_requested" if mode == "launch-only" else "passed"
        lineage_status = "not_requested" if mode == "launch-only" else "complete"
        status_counts = Counter(record.enrichment.status for record in audit.records)
        method_counts = Counter(record.enrichment.match_method for record in audit.records if record.enrichment.match_method is not None)
        hashes = {record.integrity.sha256 for record in audit.records}
        path_values = {} if paths is None else _report_paths(paths)
        return cls(
            started_at=_utc_text(started),
            finished_at=_utc_text(finished),
            dataset_name=dataset_name,
            mode=mode,
            status="completed",
            paths=MappingProxyType(path_values),
            counts=MappingProxyType(
                {
                    "terminal": audit.terminal_count,
                    "lineage_candidates": audit.lineage_candidate_count,
                    "validated": len(audit.records),
                    "constructed": None,
                }
            ),
            hash_validation=MappingProxyType(
                {
                    "status": audit_status,
                    "unique_sha256": len(hashes),
                    "validated": len(audit.records),
                }
            ),
            media_validation=MappingProxyType(
                {
                    "status": audit_status,
                    "readable": len(audit.records),
                    "unique_paths": len({record.terminal.filepath for record in audit.records}),
                }
            ),
            lineage=MappingProxyType(
                {
                    "status": lineage_status,
                    "statuses": {status: status_counts[status] for status in ("matched", "ambiguous", "missing")},
                    "methods": {
                        method: method_counts[method]
                        for method in (
                            "source_id",
                            "normalized_source_filepath",
                            "export_relative_filepath",
                            "unique_filename",
                        )
                    },
                }
            ),
            field_population=MappingProxyType(_audit_field_population(audit.records)),
            views=MappingProxyType({"status": "not_requested", "names": []}),
            failure=None,
        )

    def completed(
        self,
        summary: SnapshotSummary | None = None,
        *,
        finished_at: datetime | None = None,
    ) -> "RunReport":
        counts = dict(self.counts)
        fields = dict(self.field_population)
        views: dict[str, object] = dict(self.views)
        if summary is not None:
            counts["constructed"] = summary.constructed_count
            fields = dict(summary.field_population)
            views = {"status": "created", "names": list(summary.saved_views)}
        return replace(
            self,
            finished_at=_utc_text(finished_at or datetime.now(timezone.utc)),
            status="completed",
            counts=MappingProxyType(counts),
            field_population=MappingProxyType(fields),
            views=MappingProxyType(views),
            failure=None,
        )

    def failed(
        self,
        error: BaseException,
        *,
        finished_at: datetime | None = None,
    ) -> "RunReport":
        return replace(
            self,
            finished_at=_utc_text(finished_at or datetime.now(timezone.utc)),
            status="failed",
            failure=MappingProxyType({"type": type(error).__name__, "message": str(error)}),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "dataset_name": self.dataset_name,
            "mode": self.mode,
            "status": self.status,
            "paths": _plain(self.paths),
            "counts": _plain(self.counts),
            "hash_validation": _plain(self.hash_validation),
            "media_validation": _plain(self.media_validation),
            "lineage": _plain(self.lineage),
            "field_population": _plain(self.field_population),
            "views": _plain(self.views),
            "failure": None if self.failure is None else _plain(self.failure),
        }


@dataclass(frozen=True)
class Services:
    validate_runtime: Callable[[RuntimePaths], RuntimePaths]
    audit: Callable[[RuntimePaths], Audit]
    dataset_exists: Callable[[str], bool]
    load_dataset: Callable[[str], Any]
    delete_dataset: Callable[[str], None]
    create_snapshot: Callable[[Sequence[ValidatedRecord], str, str], Any]
    verify_snapshot: Callable[[Any, Sequence[ValidatedRecord]], SnapshotSummary]
    launch: Callable[[Any, str, int], None]
    write_report: Callable[[RunReport, Path], Path]
    now: Callable[[], datetime]
    input_fn: Callable[[str], str]
    output_fn: Callable[[str], None]


class _ArgumentParser(argparse.ArgumentParser):
    @overload
    def parse_args(
        self,
        args: Iterable[str] | None = ...,
        namespace: None = ...,
    ) -> argparse.Namespace: ...

    @overload
    def parse_args(self, args: Iterable[str] | None, namespace: _Namespace) -> _Namespace: ...

    @overload
    def parse_args(self, *, namespace: _Namespace) -> _Namespace: ...

    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: Any = None,
    ) -> Any:
        parsed = super().parse_args(args, namespace)
        if parsed.yes and not parsed.overwrite:
            self.error("--yes requires --overwrite")
        if parsed.overwrite and (parsed.dry_run or parsed.launch_only):
            self.error("--overwrite cannot be used with --dry-run or --launch-only")
        return parsed


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Audit and report without opening FiftyOne")
    modes.add_argument("--create-only", action="store_true", help="Create and verify without launching the App")
    modes.add_argument("--launch-only", action="store_true", help="Launch the existing snapshot without auditing media")
    parser.add_argument("--overwrite", action="store_true", help="Replace the exact persistent dataset record")
    parser.add_argument("--yes", action="store_true", help="Confirm an overwrite noninteractively")
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="FiftyOne App bind address")
    parser.add_argument("--port", default=DEFAULT_PORT, type=_port, help="FiftyOne App port")
    parser.set_defaults(dataset_name=DEFAULT_DATASET_NAME)
    return parser


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def configure_fiftyone_environment(paths: RuntimePaths) -> None:
    os.environ["FIFTYONE_CONFIG_PATH"] = str(paths.config_path)
    os.environ["FIFTYONE_DATABASE_DIR"] = str(paths.database_dir)
    os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(paths.dataset_dir)
    os.environ["FIFTYONE_DEFAULT_DATASET_DIR"] = str(paths.dataset_dir)
    os.environ["FIFTYONE_MODEL_ZOO_DIR"] = str(paths.model_zoo_dir)
    os.environ["FIFTYONE_PLUGINS_DIR"] = str(paths.plugins_dir)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(nested) for nested in value]
    return value


def _report_paths(paths: RuntimePaths) -> dict[str, object]:
    return {
        "intermediate_dir": str(paths.intermediate_dir.resolve(strict=False)),
        "terminal_export_dir": str(paths.terminal_export_dir.resolve(strict=False)),
        "upstream_export_dirs": [str(path.resolve(strict=False)) for path in paths.upstream_export_dirs],
        "manifest_paths": [str(path.resolve(strict=False)) for path in paths.manifest_paths],
        "state_root": str(paths.state_root.resolve(strict=False)),
        "database_dir": str(paths.database_dir.resolve(strict=False)),
        "report_dir": str(paths.report_dir.resolve(strict=False)),
        "dataset_dir": str(paths.dataset_dir.resolve(strict=False)),
        "config_path": str(paths.config_path.resolve(strict=False)),
        "model_zoo_dir": str(paths.model_zoo_dir.resolve(strict=False)),
        "plugins_dir": str(paths.plugins_dir.resolve(strict=False)),
    }


def _audit_field_population(records: Sequence[ValidatedRecord]) -> dict[str, int]:
    fields = {
        "jaguar_id": len(records),
        "ground_truth": len(records),
        "bboxes_body": len(records),
        "segmentations_body": len(records),
        "lineage_status": len(records),
        "lineage_match_method": sum(record.enrichment.match_method is not None for record in records),
        "sha256": len(records),
        "size_bytes": len(records),
        "width": len(records),
        "height": len(records),
    }
    fields.update(
        {field_name: sum(record.enrichment.fields.get(field_name) is not None for record in records) for field_name in _REPORT_ENRICHMENT_FIELDS}
    )
    return dict(sorted(fields.items()))


def build_validated_records(
    paths: RuntimePaths,
    *,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> Audit:
    terminals = load_terminal_records(paths.terminal_export_dir)
    candidates = load_lineage_candidates(paths.intermediate_dir)
    index = LineageIndex.from_candidates(candidates)
    records = validate_records(
        [(terminal, index.enrich(terminal)) for terminal in terminals],
        expected_count=expected_count,
    )
    return Audit(
        records=tuple(records),
        terminal_count=len(terminals),
        lineage_candidate_count=len(candidates),
    )


def _slug(value: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", value)
    slug = re.sub(r"[^a-z0-9]+", "-", separated.casefold()).strip("-")
    return slug or "dataset"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise


def write_report(report: RunReport, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.fromisoformat(report.started_at.replace("Z", "+00:00"))
    report_path = report_dir / f"{started_at.strftime('%Y%m%dT%H%M%SZ')}_{_slug(report.dataset_name)}.json"
    payload = report.to_dict()
    _atomic_json(report_path, payload)
    _atomic_json(report_dir / "latest.json", payload)
    return report_path


def _fiftyone_launch_app(dataset: Any, *, address: str, port: int) -> Any:
    fiftyone = importlib.import_module("fiftyone")
    return fiftyone.launch_app(dataset, address=address, port=port)


def launch_and_wait(
    dataset: Any,
    address: str,
    port: int,
    *,
    launch_app: Callable[..., Any] = _fiftyone_launch_app,
) -> None:
    session = launch_app(dataset, address=address, port=port)
    try:
        with suppress(KeyboardInterrupt):
            session.wait()
    finally:
        session.close()


def default_runtime_paths() -> RuntimePaths:
    return RuntimePaths(
        intermediate_dir=DEFAULT_INTERMEDIATE_DIR,
        terminal_export_dir=DEFAULT_TERMINAL_EXPORT_DIR,
        upstream_export_dirs=DEFAULT_UPSTREAM_EXPORT_DIRS,
        manifest_paths=DEFAULT_MANIFEST_PATHS,
        state_root=DEFAULT_STATE_ROOT,
        database_dir=DEFAULT_DATABASE_DIR,
        report_dir=DEFAULT_REPORT_DIR,
        dataset_dir=DEFAULT_DATASET_DIR,
        config_path=DEFAULT_CONFIG_PATH,
        model_zoo_dir=DEFAULT_MODEL_ZOO_DIR,
        plugins_dir=DEFAULT_PLUGINS_DIR,
        mount_roots=DEFAULT_MOUNT_ROOTS,
    )


def validate_runtime_paths(
    paths: RuntimePaths,
    *,
    approved_state_root: Path = DEFAULT_STATE_ROOT,
    is_mount: Callable[[Path], bool] = os.path.ismount,
) -> RuntimePaths:
    validate_mounts(paths.mount_roots, is_mount=is_mount)
    resolved_state_root = paths.state_root.resolve(strict=False)
    resolved_approved_root = approved_state_root.resolve(strict=False)
    if resolved_state_root != resolved_approved_root:
        raise StorageSafetyError(
            f"state root must resolve to approved external storage: "
            f"{paths.state_root} -> {resolved_state_root}; expected {resolved_approved_root}"
        )
    validate_storage_paths(
        (
            paths.database_dir,
            paths.report_dir,
            paths.dataset_dir,
            paths.config_path,
            paths.model_zoo_dir,
            paths.plugins_dir,
        ),
        resolved_approved_root,
    )
    return RuntimePaths(
        intermediate_dir=paths.intermediate_dir.resolve(strict=False),
        terminal_export_dir=paths.terminal_export_dir.resolve(strict=False),
        upstream_export_dirs=tuple(path.resolve(strict=False) for path in paths.upstream_export_dirs),
        manifest_paths=tuple(path.resolve(strict=False) for path in paths.manifest_paths),
        state_root=paths.state_root.resolve(strict=False),
        database_dir=paths.database_dir.resolve(strict=False),
        report_dir=paths.report_dir.resolve(strict=False),
        dataset_dir=paths.dataset_dir.resolve(strict=False),
        config_path=paths.config_path.resolve(strict=False),
        model_zoo_dir=paths.model_zoo_dir.resolve(strict=False),
        plugins_dir=paths.plugins_dir.resolve(strict=False),
        mount_roots=tuple(path.resolve(strict=False) for path in paths.mount_roots),
    )


def _fiftyone() -> Any:
    return importlib.import_module("fiftyone")


def _dataset_exists(dataset_name: str) -> bool:
    return bool(_fiftyone().dataset_exists(dataset_name))


def _load_dataset(dataset_name: str) -> Any:
    return _fiftyone().load_dataset(dataset_name)


def _delete_dataset(dataset_name: str) -> None:
    _fiftyone().delete_dataset(dataset_name)


def _create_snapshot(
    records: Sequence[ValidatedRecord],
    dataset_name: str,
    temporary_name: str,
) -> Any:
    snapshot = importlib.import_module("jaguars.visualization.final_snapshot")
    return snapshot.create_snapshot(records, dataset_name, temporary_name)


def _verify_snapshot(
    dataset: Any,
    records: Sequence[ValidatedRecord],
) -> SnapshotSummary:
    snapshot = importlib.import_module("jaguars.visualization.final_snapshot")
    snapshot._validate_snapshot(dataset, records)
    field_population = {field_name: int(dataset.count(field_name)) for field_name in snapshot.APPROVED_SAMPLE_FIELDS}
    return SnapshotSummary(
        constructed_count=len(dataset),
        field_population=MappingProxyType(field_population),
        saved_views=tuple(dataset.list_saved_views()),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


DEFAULT_SERVICES = Services(
    validate_runtime=validate_runtime_paths,
    audit=build_validated_records,
    dataset_exists=_dataset_exists,
    load_dataset=_load_dataset,
    delete_dataset=_delete_dataset,
    create_snapshot=_create_snapshot,
    verify_snapshot=_verify_snapshot,
    launch=launch_and_wait,
    write_report=write_report,
    now=_utc_now,
    input_fn=lambda prompt: input(prompt),
    output_fn=print,
)


def confirm_overwrite(
    dataset_name: str,
    existing_count: int,
    proposed_count: int,
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> bool:
    output_fn(f"Existing dataset samples: {existing_count}")
    output_fn(f"Proposed dataset samples: {proposed_count}")
    try:
        prompt = f"Type the exact dataset name '{dataset_name}' to replace its dataset record: "
        response = input(prompt) if input_fn is None else input_fn(prompt)
    except EOFError:
        return False
    return response == dataset_name


def _mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry-run"
    if args.create_only:
        return "create-only"
    if args.launch_only:
        return "launch-only"
    return "create-and-launch"


def run(
    args: argparse.Namespace,
    services: Services | None = None,
    *,
    paths: RuntimePaths | None = None,
) -> int:
    active_services = services or DEFAULT_SERVICES
    configured_paths = paths or default_runtime_paths()
    started_at = active_services.now()
    runtime_paths = configured_paths
    audit = Audit(records=(), terminal_count=0, lineage_candidate_count=0)
    report: RunReport | None = None
    runtime_validated = False
    try:
        runtime_paths = active_services.validate_runtime(configured_paths)
        runtime_validated = True
        configure_fiftyone_environment(runtime_paths)
        if args.launch_only:
            dataset = active_services.load_dataset(args.dataset_name)
            report = RunReport.from_audit(
                audit,
                paths=runtime_paths,
                dataset_name=args.dataset_name,
                mode=_mode(args),
                started_at=started_at,
                finished_at=active_services.now(),
            )
            active_services.write_report(report.completed(finished_at=active_services.now()), runtime_paths.report_dir)
            active_services.launch(dataset, args.address, args.port)
            return 0

        audit = active_services.audit(runtime_paths)
        report = RunReport.from_audit(
            audit,
            paths=runtime_paths,
            dataset_name=args.dataset_name,
            mode=_mode(args),
            started_at=started_at,
            finished_at=active_services.now(),
        )
        if args.dry_run:
            active_services.write_report(report.completed(finished_at=active_services.now()), runtime_paths.report_dir)
            return 0

        exists = active_services.dataset_exists(args.dataset_name)
        if exists and not args.overwrite:
            raise DatasetExistsError(f"dataset already exists: {args.dataset_name}")
        if exists:
            existing_dataset = active_services.load_dataset(args.dataset_name)
            existing_count = len(existing_dataset)
            if args.yes:
                active_services.output_fn(f"Existing dataset samples: {existing_count}")
                active_services.output_fn(f"Proposed dataset samples: {len(audit.records)}")
                confirmed = True
            else:
                confirmed = confirm_overwrite(
                    args.dataset_name,
                    existing_count,
                    len(audit.records),
                    input_fn=active_services.input_fn,
                    output_fn=active_services.output_fn,
                )
            if not confirmed:
                raise OverwriteDeclinedError(f"overwrite declined for dataset: {args.dataset_name}")
            active_services.delete_dataset(args.dataset_name)

        dataset = active_services.create_snapshot(
            audit.records,
            args.dataset_name,
            f"{args.dataset_name}__building",
        )
        summary = active_services.verify_snapshot(dataset, audit.records)
        completed = report.completed(summary, finished_at=active_services.now())
        active_services.write_report(completed, runtime_paths.report_dir)
        if not args.create_only:
            active_services.launch(dataset, args.address, args.port)
        return 0
    except Exception as error:
        failure_report = report or RunReport.from_audit(
            audit,
            paths=runtime_paths,
            dataset_name=args.dataset_name,
            mode=_mode(args),
            started_at=started_at,
            finished_at=active_services.now(),
        )
        if runtime_validated:
            with suppress(Exception):
                active_services.write_report(
                    failure_report.failed(error, finished_at=active_services.now()),
                    runtime_paths.report_dir,
                )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (FinalDatasetError, IntegrityError, StorageSafetyError, TerminalExportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
