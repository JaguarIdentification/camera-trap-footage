import hashlib
import os
from collections import Counter
from pathlib import Path

import pytest

from jaguars.visualization.final_curation import (
    DEFAULT_POLICY,
    build_curation_plan,
)
from jaguars.visualization.final_dataset import EXPECTED_SAMPLE_COUNT
from jaguars.visualization.final_lineage import (
    LineageIndex,
    load_lineage_candidates_from_paths,
)
from jaguars.visualization.final_records import load_terminal_records
from jaguars.visualization.final_validation import validate_records

REAL_INTERMEDIATE_DIR = Path(
    os.environ.get(
        "JAGUARS_REAL_INTERMEDIATE_DIR",
        "/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data/intermediate/v1",
    )
)
REAL_SOURCE_EXPORT_DIR = REAL_INTERMEDIATE_DIR / "fo_jaguars/labeled_segmented_jaguars_primitive"
REAL_CURATED_EXPORT_DIR = REAL_INTERMEDIATE_DIR / "fo_jaguars/labeled_segmented_jaguars_final_curated_v1"
REAL_UPSTREAM_EXPORT_DIRS = (
    REAL_INTERMEDIATE_DIR / "fo_jaguars/exports/segmented_deduplicated",
    REAL_INTERMEDIATE_DIR / "fo_jaguars/exports/segmented",
    REAL_INTERMEDIATE_DIR / "fo_jaguars/exports/deduplicated",
    REAL_INTERMEDIATE_DIR / "fo_jaguars/ingested",
)
REAL_MANIFEST_PATHS = (
    REAL_INTERMEDIATE_DIR / "labels_with_splits.csv",
    REAL_INTERMEDIATE_DIR / "pptx_extracted_labels_with_splits.csv",
)

pytestmark = pytest.mark.skipif(
    not (REAL_SOURCE_EXPORT_DIR / "samples.json").is_file(),
    reason="real terminal source export is not mounted",
)


def test_real_curation_dry_run_accepts_approved_1322_sample_target() -> None:
    source_samples = REAL_SOURCE_EXPORT_DIR / "samples.json"
    before = hashlib.sha256(source_samples.read_bytes()).hexdigest()
    target_existed = REAL_CURATED_EXPORT_DIR.exists()

    plan = build_curation_plan(
        REAL_SOURCE_EXPORT_DIR,
        REAL_CURATED_EXPORT_DIR,
    )

    assert hashlib.sha256(source_samples.read_bytes()).hexdigest() == before
    assert REAL_CURATED_EXPORT_DIR.exists() is target_existed
    assert plan.source_count == 1367
    assert plan.curated_count == EXPECTED_SAMPLE_COUNT == 1322
    assert plan.unique_hashes == 1322
    assert plan.populated_identities == 1108
    assert plan.null_identities == 214
    assert plan.distinct_identities == 59
    assert len(plan.hash_groups) == 39
    assert Counter(drop.reason for drop in plan.dropped) == {
        "exact_content_duplicate": 39,
        "confirmed_false_positive": 6,
    }
    assert plan.review_paths == tuple(sorted(DEFAULT_POLICY.review_cases))
    assert plan.clipped_paths == tuple(sorted(DEFAULT_POLICY.clipped_bboxes))


@pytest.mark.skipif(
    not (REAL_CURATED_EXPORT_DIR / "samples.json").is_file(),
    reason="curated export has not been materialized yet",
)
def test_real_curated_export_is_parser_and_validator_green() -> None:
    terminals = load_terminal_records(REAL_CURATED_EXPORT_DIR)
    candidates = load_lineage_candidates_from_paths(
        REAL_UPSTREAM_EXPORT_DIRS,
        REAL_MANIFEST_PATHS,
    )
    index = LineageIndex.from_candidates(candidates)

    records = validate_records(
        [(terminal, index.enrich(terminal)) for terminal in terminals],
        expected_count=EXPECTED_SAMPLE_COUNT,
    )

    assert len(records) == 1322
    assert len({record.integrity.sha256 for record in records}) == 1322
    assert sum(record.terminal.review_required for record in records) == 4
    assert sum(record.terminal.bboxes_body is not None for record in records) == 1318
    assert sum(record.terminal.segmentations_body is not None for record in records) == 1318
    assert sum(record.terminal.jaguar_id is not None for record in records) == 1108
