import os
import posixpath
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from jaguars.visualization import final_dataset
from jaguars.visualization.final_dataset import (
    AuditError,
    EXPECTED_SAMPLE_COUNT,
    build_validated_records,
    default_runtime_paths,
)
from jaguars.visualization.final_lineage import LineageCandidate
from jaguars.visualization.final_validation import validate_expected_count

REAL_INTERMEDIATE_DIR = Path(
    os.environ.get(
        "JAGUARS_REAL_INTERMEDIATE_DIR",
        "/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data/intermediate/v1",
    )
)
REAL_TERMINAL_EXPORT_DIR = REAL_INTERMEDIATE_DIR / "fo_jaguars/labeled_segmented_jaguars_primitive"

pytestmark = pytest.mark.skipif(
    not (REAL_TERMINAL_EXPORT_DIR / "samples.json").is_file(),
    reason="real curated terminal export is not mounted",
)


def _candidate_filenames(candidate: LineageCandidate) -> set[str]:
    return {
        filename
        for filename in (
            candidate.original_filename,
            (posixpath.basename(candidate.export_relative_filepath.replace("\\", "/")) if candidate.export_relative_filepath is not None else None),
            (
                posixpath.basename(candidate.normalized_source_filepath.replace("\\", "/"))
                if candidate.normalized_source_filepath is not None
                else None
            ),
        )
        if filename is not None
    }


def test_real_terminal_pipeline_preserves_all_records_through_expected_count_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = replace(
        default_runtime_paths(),
        intermediate_dir=REAL_INTERMEDIATE_DIR,
        terminal_export_dir=REAL_TERMINAL_EXPORT_DIR,
        upstream_export_dirs=(
            REAL_INTERMEDIATE_DIR / "fo_jaguars/exports/segmented_deduplicated",
            REAL_INTERMEDIATE_DIR / "fo_jaguars/exports/segmented",
            REAL_INTERMEDIATE_DIR / "fo_jaguars/exports/deduplicated",
            REAL_INTERMEDIATE_DIR / "fo_jaguars/ingested",
        ),
        manifest_paths=(
            REAL_INTERMEDIATE_DIR / "labels_with_splits.csv",
            REAL_INTERMEDIATE_DIR / "pptx_extracted_labels_with_splits.csv",
        ),
    )
    captured: dict[str, Any] = {}
    real_lineage_loader = final_dataset.load_lineage_candidates_from_paths

    def capture_candidates(
        export_dirs: tuple[Path, ...],
        manifest_paths: tuple[Path, ...],
    ) -> tuple[LineageCandidate, ...]:
        candidates = real_lineage_loader(export_dirs, manifest_paths)
        captured["candidates"] = candidates
        return candidates

    def count_only_validation(
        pairs: list[tuple[Any, Any]],
        *,
        expected_count: int | None = None,
    ) -> list[Any]:
        assert expected_count is not None
        validate_expected_count(len(pairs), expected_count)
        captured["pairs"] = pairs
        captured["expected_count"] = expected_count
        return []

    monkeypatch.setattr(
        final_dataset,
        "load_lineage_candidates_from_paths",
        capture_candidates,
    )
    monkeypatch.setattr(final_dataset, "validate_records", count_only_validation)

    audit = build_validated_records(paths)

    pairs = captured["pairs"]
    candidates = captured["candidates"]
    terminal_records = [terminal for terminal, _ in pairs]
    enrichments = [enrichment for _, enrichment in pairs]
    upstream_filenames = {filename for candidate in candidates for filename in _candidate_filenames(candidate)}
    resolved_identities = [
        terminal.jaguar_id
        or (enrichment.fields.get("jaguar_id") if enrichment.status == "matched" and isinstance(enrichment.fields.get("jaguar_id"), str) else None)
        for terminal, enrichment in pairs
    ]

    assert captured["expected_count"] == EXPECTED_SAMPLE_COUNT == 1367
    assert audit.terminal_count == len(pairs) == 1367
    assert sum(terminal.jaguar_id is not None for terminal in terminal_records) == 1120
    assert sum(terminal.jaguar_id is None for terminal in terminal_records) == 247
    assert sum(terminal.jaguar_id is None and terminal.filepath.name in upstream_filenames for terminal in terminal_records) == 105
    assert Counter(enrichment.status for enrichment in enrichments) == {
        "ambiguous": 703,
        "missing": 664,
    }
    assert sum(identity is not None for identity in resolved_identities) == 1120
    assert sum(identity is None for identity in resolved_identities) == 247


def test_real_audit_reports_known_strict_bbox_and_duplicate_hash_failures() -> None:
    paths = replace(
        default_runtime_paths(),
        intermediate_dir=REAL_INTERMEDIATE_DIR,
        terminal_export_dir=REAL_TERMINAL_EXPORT_DIR,
        upstream_export_dirs=(
            REAL_INTERMEDIATE_DIR / "fo_jaguars/exports/segmented_deduplicated",
            REAL_INTERMEDIATE_DIR / "fo_jaguars/exports/segmented",
            REAL_INTERMEDIATE_DIR / "fo_jaguars/exports/deduplicated",
            REAL_INTERMEDIATE_DIR / "fo_jaguars/ingested",
        ),
        manifest_paths=(
            REAL_INTERMEDIATE_DIR / "labels_with_splits.csv",
            REAL_INTERMEDIATE_DIR / "pptx_extracted_labels_with_splits.csv",
        ),
    )

    with pytest.raises(AuditError) as caught:
        build_validated_records(paths)

    audit = caught.value.audit
    assert audit.terminal_count == 1367
    assert audit.terminal_identity_populated == 1120
    assert audit.terminal_identity_null == 247
    assert audit.validation.annotation_failed == 15
    assert audit.validation.enrichment_failed == 0
    assert audit.validation.media_failed == 0
    assert audit.validation.duplicate_path_groups == 0
    assert audit.validation.unique_paths == 1367
    assert audit.validation.duplicate_hash_groups == 39
    assert audit.validation.duplicate_hash_pairs == 39
    assert audit.validation.unique_sha256 == 1328
