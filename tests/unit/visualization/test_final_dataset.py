import json
import os
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from types import MappingProxyType
from typing import Any, cast

import pytest

from jaguars.visualization.final_dataset import (
    Audit,
    AuditError,
    AuditValidation,
    DEFAULT_ADDRESS,
    DEFAULT_DATABASE_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_DATASET_NAME,
    DEFAULT_MANIFEST_PATHS,
    DEFAULT_PORT,
    DEFAULT_REPORT_DIR,
    DEFAULT_TERMINAL_EXPORT_DIR,
    DEFAULT_UPSTREAM_EXPORT_DIRS,
    DatasetExistsError,
    OverwriteDeclinedError,
    RunReport,
    RuntimePaths,
    Services,
    SnapshotSummary,
    build_validated_records,
    configure_fiftyone_environment,
    confirm_overwrite,
    default_runtime_paths,
    launch_and_wait,
    parse_args,
    run,
    validate_imported_fiftyone_configuration,
    write_report,
)
from jaguars.visualization.final_lineage import Enrichment
from jaguars.visualization.final_records import TerminalRecord
from jaguars.visualization.final_validation import (
    IntegrityError,
    MediaIntegrity,
    StorageSafetyError,
    ValidatedRecord,
)


@pytest.fixture(autouse=True)
def _clear_inherited_fiftyone_database_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "FIFTYONE_DATABASE_URI",
        "FIFTYONE_PRIVATE_DATABASE_PORT",
        "FIFTYONE_DATABASE_NAME",
    ):
        monkeypatch.delenv(key, raising=False)


def test_cli_and_runtime_defaults_are_the_approved_final_snapshot_values() -> None:
    args = parse_args([])
    paths = default_runtime_paths()

    assert (args.dataset_name, args.address, args.port) == (
        "JaguarCameraTrap_Final_Curated_v1",
        DEFAULT_ADDRESS,
        DEFAULT_PORT,
    )
    assert paths.terminal_export_dir == DEFAULT_TERMINAL_EXPORT_DIR
    assert paths.upstream_export_dirs == DEFAULT_UPSTREAM_EXPORT_DIRS
    assert paths.manifest_paths == DEFAULT_MANIFEST_PATHS
    assert paths.database_dir == DEFAULT_DATABASE_DIR
    assert paths.report_dir == DEFAULT_REPORT_DIR
    assert paths.dataset_dir == DEFAULT_DATASET_DIR


@pytest.mark.parametrize(
    "argv",
    [
        ["--dry-run", "--launch-only"],
        ["--create-only", "--launch-only"],
        ["--dry-run", "--overwrite"],
        ["--launch-only", "--overwrite"],
        ["--yes"],
    ],
)
def test_cli_rejects_incompatible_modes(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        parse_args(argv)

    assert caught.value.code == 2


@pytest.mark.parametrize("answer", ["wrong-name", "", "yes", "JaguarCameraTrap_Final_Curated_V1"])
def test_overwrite_confirmation_requires_the_exact_dataset_name(answer: str) -> None:
    assert (
        confirm_overwrite(
            "JaguarCameraTrap_Final_Curated_v1",
            existing_count=1200,
            proposed_count=1367,
            input_fn=lambda _prompt: answer,
        )
        is False
    )


def test_overwrite_confirmation_accepts_the_exact_dataset_name() -> None:
    assert (
        confirm_overwrite(
            "JaguarCameraTrap_Final_Curated_v1",
            existing_count=1200,
            proposed_count=1367,
            input_fn=lambda _prompt: "JaguarCameraTrap_Final_Curated_v1",
        )
        is True
    )


def test_default_confirmation_reads_builtins_input_dynamically(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: DEFAULT_DATASET_NAME)

    assert confirm_overwrite(DEFAULT_DATASET_NAME, 1200, 1367) is True


def test_fiftyone_paths_are_configured_before_any_lazy_import(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    paths = RuntimePaths(
        intermediate_dir=tmp_path / "data/intermediate/v1",
        terminal_export_dir=tmp_path / "data/intermediate/v1/fo_jaguars/terminal",
        upstream_export_dirs=(),
        manifest_paths=(),
        state_root=state_root,
        database_dir=state_root / "var/lib/mongo",
        report_dir=state_root / DEFAULT_DATASET_NAME,
        dataset_dir=state_root / "datasets",
        config_path=state_root / "config.json",
        model_zoo_dir=state_root / "models",
        plugins_dir=state_root / "plugins",
        mount_roots=(),
    )

    configure_fiftyone_environment(paths)

    assert os.environ["FIFTYONE_CONFIG_PATH"] == str(paths.config_path)
    assert os.environ["FIFTYONE_DATABASE_DIR"] == str(paths.database_dir)
    assert os.environ["FIFTYONE_DATASET_ZOO_DIR"] == str(paths.dataset_dir)
    assert os.environ["FIFTYONE_DEFAULT_DATASET_DIR"] == str(paths.dataset_dir)
    assert os.environ["FIFTYONE_MODEL_ZOO_DIR"] == str(paths.model_zoo_dir)
    assert os.environ["FIFTYONE_PLUGINS_DIR"] == str(paths.plugins_dir)


@pytest.mark.parametrize(
    "unsafe_key",
    [
        "FIFTYONE_DATABASE_URI",
        "FIFTYONE_PRIVATE_DATABASE_PORT",
        "FIFTYONE_DATABASE_NAME",
    ],
)
def test_fiftyone_configuration_rejects_inherited_external_connection_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_key: str,
) -> None:
    paths = _runtime_paths(tmp_path)
    monkeypatch.setenv(unsafe_key, "mongodb://foreign.example:27017" if unsafe_key.endswith("URI") else "27018")

    with pytest.raises(StorageSafetyError, match=unsafe_key):
        configure_fiftyone_environment(paths)


@pytest.mark.parametrize(
    ("config_payload", "expected_message"),
    [
        ({"database_uri": "mongodb://foreign.example:27017"}, "database_uri"),
        ({"database_dir": "/tmp/foreign-fiftyone"}, "database_dir"),
        ({"database_name": "foreign"}, "database_name"),
    ],
)
def test_fiftyone_configuration_rejects_external_connection_state_in_config_file(
    tmp_path: Path,
    config_payload: dict[str, str],
    expected_message: str,
) -> None:
    paths = _runtime_paths(tmp_path)
    paths.config_path.parent.mkdir(parents=True)
    paths.config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    with pytest.raises(StorageSafetyError, match=expected_message):
        configure_fiftyone_environment(paths)


@pytest.mark.parametrize(
    ("database_uri", "private_port", "database_dir", "database_name", "expected_message"),
    [
        ("mongodb://foreign.example:27017", None, None, "fiftyone", "database_uri"),
        (None, "27018", None, "fiftyone", "FIFTYONE_PRIVATE_DATABASE_PORT"),
        (None, None, "/tmp/foreign-fiftyone", "fiftyone", "database_dir"),
        (None, None, None, "foreign", "database_name"),
    ],
)
def test_post_import_fiftyone_configuration_must_match_approved_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    database_uri: str | None,
    private_port: str | None,
    database_dir: str | None,
    database_name: str,
    expected_message: str,
) -> None:
    paths = _runtime_paths(tmp_path)
    monkeypatch.delenv("FIFTYONE_PRIVATE_DATABASE_PORT", raising=False)
    if private_port is not None:
        monkeypatch.setenv("FIFTYONE_PRIVATE_DATABASE_PORT", private_port)
    fake_fiftyone = SimpleNamespace(
        config=SimpleNamespace(
            database_uri=database_uri,
            database_dir=database_dir or str(paths.database_dir),
            database_name=database_name,
        )
    )

    with pytest.raises(StorageSafetyError, match=expected_message):
        validate_imported_fiftyone_configuration(fake_fiftyone, paths.database_dir)


def _runtime_paths(tmp_path: Path) -> RuntimePaths:
    state_root = tmp_path / "state"
    return RuntimePaths(
        intermediate_dir=tmp_path / "data/intermediate/v1",
        terminal_export_dir=tmp_path / "data/intermediate/v1/fo_jaguars/terminal",
        upstream_export_dirs=(),
        manifest_paths=(),
        state_root=state_root,
        database_dir=state_root / "var/lib/mongo",
        report_dir=state_root / DEFAULT_DATASET_NAME,
        dataset_dir=state_root / "datasets",
        config_path=state_root / "config.json",
        model_zoo_dir=state_root / "models",
        plugins_dir=state_root / "plugins",
        mount_roots=(),
    )


def _record(tmp_path: Path) -> ValidatedRecord:
    terminal = TerminalRecord(
        source_id="source-a",
        filepath=tmp_path / "a.jpg",
        relative_filepath="data/a.jpg",
        jaguar_id="F11",
        bboxes_body={"detections": [{"bounding_box": [0.1, 0.2, 0.3, 0.4]}]},
        segmentations_body={
            "detections": [
                {
                    "bounding_box": [0.1, 0.2, 0.3, 0.4],
                    "mask": [[0, 1], [1, 0]],
                }
            ]
        },
    )
    return ValidatedRecord(
        terminal=terminal,
        enrichment=Enrichment(
            status="matched",
            match_method="source_id",
            fields=MappingProxyType({"closed_set_split": "train"}),
        ),
        integrity=MediaIntegrity("a" * 64, 100, 8, 6),
    )


def _services(
    *,
    events: list[str],
    reports: list[RunReport],
    audit: Audit,
    existing: bool = False,
    input_answer: str = "",
) -> Services:
    dataset = cast(Any, [object()] * 1200)

    def forbidden(label: str) -> Any:
        raise AssertionError(f"{label} must not be called")

    return Services(
        validate_runtime=lambda paths: events.append("validate") or paths,
        audit=lambda paths: events.append("audit") or audit,
        dataset_exists=lambda name: events.append("exists") or existing,
        load_dataset=lambda name: events.append("load") or dataset,
        dataset_id=lambda snapshot: events.append("pin-id") or "original-dataset-id",
        create_snapshot=lambda records, name, temporary, replace_existing, expected_original_id: events.append(
            f"create:{'replace' if replace_existing else 'new'}:{expected_original_id}"
        )
        or dataset,
        verify_snapshot=lambda snapshot, records: events.append("verify")
        or SnapshotSummary(
            constructed_count=len(records),
            field_population={"jaguar_id": len(records)},
            saved_views=("All final samples",),
        ),
        launch=lambda snapshot, address, port: events.append("launch"),
        write_report=lambda report, report_dir: reports.append(report) or report_dir / "run.json",
        now=lambda: datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
        input_fn=lambda prompt: input_answer,
        isatty=lambda: True,
        output_fn=lambda message: events.append(f"print:{message}"),
    )


def test_dry_run_audits_and_reports_without_fiftyone_services(tmp_path: Path) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(events=events, reports=reports, audit=audit)
    services = Services(
        **{
            **services.__dict__,
            "dataset_exists": lambda name: pytest.fail("dry-run connected to FiftyOne"),
            "load_dataset": lambda name: pytest.fail("dry-run connected to FiftyOne"),
            "create_snapshot": lambda records, name, temporary, replace_existing, expected_original_id: pytest.fail("dry-run connected to FiftyOne"),
            "verify_snapshot": lambda snapshot, records: pytest.fail("dry-run connected to FiftyOne"),
            "launch": lambda snapshot, address, port: pytest.fail("dry-run launched the App"),
        }
    )

    assert run(parse_args(["--dry-run"]), services=services, paths=_runtime_paths(tmp_path)) == 0

    assert events == ["validate", "audit"]
    assert reports[0].status == "completed"
    assert reports[0].counts["terminal"] == 1
    assert reports[0].counts["validated"] == 1


def test_launch_only_validates_storage_then_loads_without_media_audit(tmp_path: Path) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(events=events, reports=reports, audit=audit)
    services = Services(
        **{
            **services.__dict__,
            "audit": lambda paths: pytest.fail("launch-only audited media"),
            "dataset_exists": lambda name: pytest.fail("launch-only queried before loading"),
        }
    )

    assert run(parse_args(["--launch-only"]), services=services, paths=_runtime_paths(tmp_path)) == 0

    assert events == ["validate", "load", "launch"]
    assert reports[0].mode == "launch-only"


def test_ordinary_creation_refuses_an_existing_dataset_after_audit(tmp_path: Path) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(events=events, reports=reports, audit=audit, existing=True)

    with pytest.raises(DatasetExistsError, match=DEFAULT_DATASET_NAME):
        run(parse_args(["--create-only"]), services=services, paths=_runtime_paths(tmp_path))

    assert events == ["validate", "audit", "exists"]
    assert reports[0].status == "failed"
    assert reports[0].failure == {
        "type": "DatasetExistsError",
        "message": f"dataset already exists: {DEFAULT_DATASET_NAME}",
    }


def test_default_mode_audits_creates_reports_then_launches(tmp_path: Path) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(events=events, reports=reports, audit=audit)
    services = Services(
        **{
            **services.__dict__,
            "write_report": lambda report, report_dir: events.append("report") or reports.append(report) or report_dir / "run.json",
        }
    )

    assert run(parse_args([]), services=services, paths=_runtime_paths(tmp_path)) == 0

    assert events == [
        "validate",
        "audit",
        "exists",
        "create:new:None",
        "verify",
        "launch",
        "report",
    ]


def test_overwrite_decline_never_deletes_a_dataset_or_media(tmp_path: Path) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    media_path = _record(tmp_path).terminal.filepath
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(
        events=events,
        reports=reports,
        audit=audit,
        existing=True,
        input_answer="no",
    )

    with pytest.raises(OverwriteDeclinedError):
        run(parse_args(["--create-only", "--overwrite"]), services=services, paths=_runtime_paths(tmp_path))

    assert events[:4] == ["validate", "audit", "exists", "load"]
    assert not any(event.startswith("create:") for event in events)
    assert not hasattr(Path, "delete")
    assert media_path == tmp_path / "a.jpg"


def test_overwrite_audits_and_confirms_before_transactional_replacement(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(
        events=events,
        reports=reports,
        audit=audit,
        existing=True,
        input_answer=DEFAULT_DATASET_NAME,
    )

    assert (
        run(
            parse_args(["--create-only", "--overwrite"]),
            services=services,
            paths=_runtime_paths(tmp_path),
        )
        == 0
    )

    significant = [
        event
        for event in events
        if event
        in {
            "validate",
            "audit",
            "exists",
            "load",
            "pin-id",
            "create:replace:original-dataset-id",
            "verify",
        }
    ]
    assert significant == [
        "validate",
        "audit",
        "exists",
        "load",
        "pin-id",
        "create:replace:original-dataset-id",
        "verify",
    ]
    assert reports[0].counts["constructed"] == 1


def test_yes_overwrite_is_noninteractive_but_still_prints_counts(tmp_path: Path) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(events=events, reports=reports, audit=audit, existing=True)
    services = Services(
        **{
            **services.__dict__,
            "input_fn": lambda prompt: pytest.fail("--yes prompted for confirmation"),
        }
    )

    assert (
        run(
            parse_args(["--create-only", "--overwrite", "--yes"]),
            services=services,
            paths=_runtime_paths(tmp_path),
        )
        == 0
    )

    assert "print:Existing dataset samples: 1200" in events
    assert "print:Proposed dataset samples: 1" in events
    assert "create:replace:original-dataset-id" in events


def test_non_tty_overwrite_requires_yes_even_if_input_would_match(tmp_path: Path) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(
        events=events,
        reports=reports,
        audit=audit,
        existing=True,
        input_answer=DEFAULT_DATASET_NAME,
    )
    services = Services(**{**services.__dict__, "isatty": lambda: False})

    with pytest.raises(OverwriteDeclinedError):
        run(
            parse_args(["--create-only", "--overwrite"]),
            services=services,
            paths=_runtime_paths(tmp_path),
        )

    assert not any(event.startswith("create:") for event in events)


def test_build_validated_records_honors_exact_configured_lineage_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaguars.visualization import final_dataset

    paths = _runtime_paths(tmp_path)
    validated = _record(tmp_path)
    terminal = validated.terminal
    candidate = object()
    enrichment = validated.enrichment
    events: list[object] = []

    class Index:
        def enrich(self, record: TerminalRecord) -> Enrichment:
            assert record is terminal
            return enrichment

    monkeypatch.setattr(
        final_dataset,
        "load_terminal_records",
        lambda export_dir: events.append(("terminal", export_dir)) or [terminal],
    )
    monkeypatch.setattr(
        final_dataset,
        "load_lineage_candidates_from_paths",
        lambda export_dirs, manifest_paths: events.append(("lineage", tuple(export_dirs), tuple(manifest_paths))) or (candidate,),
    )
    monkeypatch.setattr(
        final_dataset.LineageIndex,
        "from_candidates",
        lambda candidates: events.append(("index", tuple(candidates))) or Index(),
    )

    def validate(
        pairs: Sequence[tuple[TerminalRecord, Enrichment]],
        *,
        expected_count: int | None = None,
    ) -> list[ValidatedRecord]:
        events.append(("validate-records", tuple(pairs), expected_count))
        return [validated]

    monkeypatch.setattr(final_dataset, "validate_records", validate)

    audit = build_validated_records(
        paths,
        expected_count=1,
        expected_terminal_identity_populated=1,
        expected_terminal_identity_null=0,
    )

    assert audit == Audit(
        records=(validated,),
        terminal_count=1,
        lineage_candidate_count=1,
        terminal_identity_populated=1,
        terminal_identity_null=0,
        resolved_identity_populated=1,
        resolved_identity_null=0,
        validation=AuditValidation(
            unique_paths=1,
            unique_sha256=1,
        ),
        lineage_status_counts=MappingProxyType({"matched": 1}),
        lineage_method_counts=MappingProxyType({"source_id": 1}),
    )
    assert events == [
        ("terminal", paths.terminal_export_dir),
        ("lineage", paths.upstream_export_dirs, paths.manifest_paths),
        ("index", (candidate,)),
        ("validate-records", ((terminal, enrichment),), 1),
    ]


def test_build_validated_records_enforces_frozen_terminal_identity_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaguars.visualization import final_dataset

    paths = _runtime_paths(tmp_path)
    terminal = _record(tmp_path).terminal
    monkeypatch.setattr(
        final_dataset,
        "load_terminal_records",
        lambda export_dir: [terminal],
    )
    monkeypatch.setattr(
        final_dataset,
        "load_lineage_candidates_from_paths",
        lambda export_dirs, manifest_paths: (),
    )

    with pytest.raises(
        IntegrityError,
        match="expected 0 populated terminal identities, found 1",
    ):
        build_validated_records(
            paths,
            expected_count=1,
            expected_terminal_identity_populated=0,
            expected_terminal_identity_null=1,
        )


def test_reports_serialize_success_and_failure_and_atomically_publish_latest(tmp_path: Path) -> None:
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    report = RunReport.from_audit(
        audit,
        paths=_runtime_paths(tmp_path),
        started_at=datetime(2026, 7, 25, 12, 34, 56, tzinfo=timezone.utc),
    ).completed(
        SnapshotSummary(1, {"jaguar_id": 1}, ("All final samples",)),
        finished_at=datetime(2026, 7, 25, 12, 35, tzinfo=timezone.utc),
    )

    report_path = write_report(report, tmp_path / "reports")

    serialized = json.loads(report_path.read_text(encoding="utf-8"))
    latest = json.loads((tmp_path / "reports/latest.json").read_text(encoding="utf-8"))
    assert serialized == latest == report.to_dict()
    assert report_path.name == "20260725T123456Z_jaguar-camera-trap-final-curated-v1.json"
    assert not list((tmp_path / "reports").glob("*.tmp"))

    failure = report.failed(ValueError("injected failure"))
    failure_payload = json.loads(json.dumps(failure.to_dict()))
    assert failure_payload["status"] == "failed"
    assert failure_payload["failure"] == {
        "type": "ValueError",
        "message": "injected failure",
    }


def test_report_identity_population_counts_only_resolved_values(tmp_path: Path) -> None:
    record = _record(tmp_path)
    missing_terminal = replace(record.terminal, jaguar_id=None)
    unresolved = replace(
        record,
        terminal=missing_terminal,
        enrichment=replace(
            record.enrichment,
            status="missing",
            match_method=None,
            fields=MappingProxyType({}),
        ),
    )
    resolved = replace(
        record,
        terminal=missing_terminal,
        enrichment=replace(
            record.enrichment,
            fields=MappingProxyType({"jaguar_id": "F11"}),
        ),
    )

    report = RunReport.from_audit(
        Audit(
            records=(unresolved, resolved),
            terminal_count=2,
            lineage_candidate_count=1,
        )
    )

    assert report.field_population["jaguar_id"] == 1
    assert report.field_population["ground_truth"] == 1


def test_report_separates_frozen_terminal_and_resolved_identity_counts(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    unresolved = replace(
        record,
        terminal=replace(record.terminal, jaguar_id=None),
        enrichment=replace(
            record.enrichment,
            status="missing",
            match_method=None,
            fields=MappingProxyType({}),
        ),
    )
    resolved = replace(
        unresolved,
        enrichment=replace(
            record.enrichment,
            fields=MappingProxyType({"jaguar_id": "F11"}),
        ),
    )

    report = RunReport.from_audit(
        Audit(
            records=(unresolved, resolved),
            terminal_count=2,
            lineage_candidate_count=1,
            terminal_identity_populated=0,
            terminal_identity_null=2,
            resolved_identity_populated=1,
            resolved_identity_null=1,
        )
    )

    assert report.counts["terminal_identity_populated"] == 0
    assert report.counts["terminal_identity_null"] == 2
    assert report.counts["resolved_identity_populated"] == 1
    assert report.counts["resolved_identity_null"] == 1


def test_media_report_passes_when_all_media_was_read_despite_other_failures(
    tmp_path: Path,
) -> None:
    report = RunReport.from_audit(
        Audit(
            records=(_record(tmp_path),),
            terminal_count=1,
            lineage_candidate_count=0,
            validation=AuditValidation(
                annotation_failed=1,
                duplicate_hash_groups=1,
                duplicate_hash_pairs=1,
                unique_paths=1,
                unique_sha256=1,
            ),
            validation_state="failed",
        )
    )

    assert report.media_validation["status"] == "passed"
    assert report.hash_validation["status"] == "failed"


def test_audit_failure_report_retains_partial_counts_and_failed_validation_state(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    record = _record(tmp_path)
    partial = Audit(
        records=(record,),
        terminal_count=2,
        lineage_candidate_count=6,
        terminal_identity_populated=1,
        terminal_identity_null=1,
        resolved_identity_populated=1,
        resolved_identity_null=1,
        validation=AuditValidation(
            annotation_failed=1,
            media_failed=1,
            duplicate_hash_groups=1,
            duplicate_hash_pairs=1,
            unique_paths=1,
            unique_sha256=1,
            errors=("malformed bbox", "missing media", "duplicate hash"),
        ),
        phase="validation",
        last_successful_phase="lineage_load",
        lineage_state="complete",
        validation_state="failed",
    )
    services = _services(
        events=events,
        reports=reports,
        audit=partial,
    )
    services = Services(
        **{
            **services.__dict__,
            "audit": lambda paths: (_ for _ in ()).throw(AuditError("real audit failed", partial)),
        }
    )

    with pytest.raises(AuditError, match="real audit failed"):
        run(
            parse_args(["--dry-run"]),
            services=services,
            paths=_runtime_paths(tmp_path),
        )

    failure = reports[-1]
    assert failure.status == "failed"
    assert failure.counts["terminal"] == 2
    assert failure.counts["validated"] == 1
    assert failure.counts["annotation_failed"] == 1
    assert failure.counts["media_failed"] == 1
    assert failure.hash_validation == {
        "status": "not_completed",
        "unique_sha256": 1,
        "validated": 1,
        "duplicate_groups": 1,
        "duplicate_pairs": 1,
    }
    assert failure.media_validation["status"] == "failed"
    assert failure.phase == "validation"
    assert failure.last_successful_phase == "lineage_load"
    assert failure.lineage["status"] == "complete"


def test_terminal_parse_failure_reports_no_started_lineage_or_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaguars.visualization import final_dataset

    monkeypatch.setattr(
        final_dataset,
        "load_terminal_records",
        lambda export_dir: (_ for _ in ()).throw(ValueError("broken terminal export")),
    )

    with pytest.raises(AuditError) as caught:
        build_validated_records(_runtime_paths(tmp_path))

    report = RunReport.from_audit(caught.value.audit).failed(caught.value)
    assert report.phase == "terminal_parse"
    assert report.last_successful_phase == "startup"
    assert report.lineage["status"] == "not_started"
    assert report.hash_validation["status"] == "not_completed"
    assert report.media_validation["status"] == "not_completed"


def test_lineage_loader_failure_reports_parsed_terminal_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaguars.visualization import final_dataset

    terminal = _record(tmp_path).terminal
    monkeypatch.setattr(final_dataset, "load_terminal_records", lambda export_dir: [terminal])
    monkeypatch.setattr(
        final_dataset,
        "load_lineage_candidates_from_paths",
        lambda export_dirs, manifest_paths: (_ for _ in ()).throw(ValueError("broken lineage export")),
    )

    with pytest.raises(AuditError) as caught:
        build_validated_records(_runtime_paths(tmp_path))

    report = RunReport.from_audit(caught.value.audit).failed(caught.value)
    assert report.phase == "lineage_load"
    assert report.last_successful_phase == "terminal_parse"
    assert report.counts["terminal"] == 1
    assert report.counts["terminal_identity_populated"] == 1
    assert report.lineage["status"] == "failed"
    assert report.hash_validation["status"] == "not_completed"
    assert report.media_validation["status"] == "not_completed"


def test_snapshot_failure_report_marks_snapshot_creation_phase(tmp_path: Path) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(events=events, reports=reports, audit=audit)
    services = Services(
        **{
            **services.__dict__,
            "create_snapshot": lambda *args: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
        }
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        run(parse_args(["--create-only"]), services=services, paths=_runtime_paths(tmp_path))

    failure = reports[-1]
    assert failure.phase == "snapshot_creation"
    assert failure.last_successful_phase == "validation"
    assert failure.views["status"] == "not_requested"


def test_published_cleanup_failure_report_retains_snapshot_summary(tmp_path: Path) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(records=(_record(tmp_path),), terminal_count=1, lineage_candidate_count=6)
    services = _services(events=events, reports=reports, audit=audit, existing=True)
    published = [object()]

    class CleanupFailure(RuntimeError):
        published_dataset = published

    services = Services(
        **{
            **services.__dict__,
            "create_snapshot": lambda *args: (_ for _ in ()).throw(CleanupFailure("old backup remains")),
        }
    )

    with pytest.raises(CleanupFailure):
        run(
            parse_args(["--create-only", "--overwrite", "--yes"]),
            services=services,
            paths=_runtime_paths(tmp_path),
        )

    failure = reports[-1]
    assert failure.phase == "cleanup_failed"
    assert failure.last_successful_phase == "snapshot_created"
    assert failure.counts["constructed"] == 1
    assert failure.views == {
        "status": "created",
        "names": ["All final samples"],
    }


def test_launch_failure_report_retains_verified_snapshot_summary(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    reports: list[RunReport] = []
    audit = Audit(
        records=(_record(tmp_path),),
        terminal_count=1,
        lineage_candidate_count=6,
    )
    services = _services(events=events, reports=reports, audit=audit)
    services = Services(
        **{
            **services.__dict__,
            "launch": lambda snapshot, address, port: (_ for _ in ()).throw(RuntimeError("App launch failed")),
        }
    )

    with pytest.raises(RuntimeError, match="App launch failed"):
        run(parse_args([]), services=services, paths=_runtime_paths(tmp_path))

    failure = reports[-1]
    assert failure.status == "failed"
    assert failure.counts["constructed"] == 1
    assert failure.views == {
        "status": "created",
        "names": ["All final samples"],
    }
    assert failure.phase == "app_launch"
    assert failure.last_successful_phase == "snapshot_created"


def test_app_session_closes_after_wait_and_keyboard_interrupt() -> None:
    events: list[str] = []

    class Session:
        def wait(self) -> None:
            events.append("wait")
            raise KeyboardInterrupt

        def close(self) -> None:
            events.append("close")

    def launch_app(dataset: object, *, address: str, port: int) -> Session:
        assert address == "localhost"
        assert port == 5151
        events.append("launch")
        return Session()

    launch_and_wait(object(), "localhost", 5151, launch_app=launch_app)

    assert events == ["launch", "wait", "close"]
