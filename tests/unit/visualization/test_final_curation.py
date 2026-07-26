import json
import os
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from PIL import Image
import pytest

import jaguars.visualization.final_curation as final_curation
from jaguars.visualization.final_curation import (
    DEFAULT_SOURCE_EXPORT_DIR,
    DEFAULT_TARGET_EXPORT_DIR,
    CurationError,
    CurationPolicy,
    CurationConflictError,
    build_curation_plan,
    choose_representative,
    materialize_curated_export,
    normalize_semantic_annotations,
    parse_args,
    run,
)
from jaguars.visualization.final_lineage import Enrichment
from jaguars.visualization.final_records import (
    FrozenAnnotation,
    TerminalRecord,
    load_terminal_records,
)
from jaguars.visualization.final_validation import validate_records


@pytest.fixture(autouse=True)
def _allow_isolated_target_storage(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Keep synthetic materialization tests independent of production mounts."""
    if request.node.name.startswith("test_target_storage_"):
        return

    def validate_isolated_target(target: Path) -> object:
        resolved_target = target.resolve(strict=False)
        mount_path = resolved_target.parent
        while not mount_path.exists():
            mount_path = mount_path.parent
        mount_identity = final_curation._capture_directory_identity(mount_path)
        assert mount_identity is not None
        return final_curation._ValidatedTargetStorage(
            mount_path=mount_path,
            target_path=resolved_target,
            mount_identity=mount_identity,
            is_mount_fn=lambda _path: True,
        )

    monkeypatch.setattr(
        final_curation,
        "_validate_target_storage",
        validate_isolated_target,
    )


def _annotation(
    *,
    detection_id: str,
    bbox: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
    mask: tuple[tuple[int, ...], ...] | None = None,
) -> FrozenAnnotation:
    detection: dict[str, Any] = {
        "_id": {"$oid": detection_id},
        "_cls": "Detection",
        "label": "jaguar",
        "bounding_box": bbox,
    }
    if mask is not None:
        detection["mask"] = mask
    return cast(
        FrozenAnnotation,
        {"_cls": "Detections", "detections": [detection]},
    )


def _terminal(
    tmp_path: Path,
    name: str,
    *,
    jaguar_id: str | None = "F11",
    detection_id: str | None = None,
    valid: bool = True,
) -> TerminalRecord:
    identifier = detection_id or name
    bbox = (0.1, 0.2, 0.3, 0.4) if valid else (-0.1, 0.2, 0.3, 0.4)
    return TerminalRecord(
        source_id=f"id-{name}",
        filepath=tmp_path / name,
        relative_filepath=f"data/{name}",
        jaguar_id=jaguar_id,
        bboxes_body=_annotation(detection_id=identifier, bbox=bbox),
        segmentations_body=_annotation(
            detection_id=identifier,
            bbox=bbox,
            mask=((0, 1), (1, 0)),
        ),
    )


def _matched(identity: str | None = "F11") -> Enrichment:
    fields = {} if identity is None else {"jaguar_id": identity}
    return Enrichment(
        status="matched",
        match_method="source_id",
        fields=MappingProxyType(fields),
    )


def test_semantic_annotation_normalization_ignores_generated_detection_ids(
    tmp_path: Path,
) -> None:
    first = _terminal(tmp_path, "a.jpg", detection_id="generated-a")
    second = _terminal(tmp_path, "b.jpg", detection_id="generated-b")

    assert normalize_semantic_annotations(first) == normalize_semantic_annotations(second)


def test_representative_prefers_valid_then_identity_then_compatible_enrichment(
    tmp_path: Path,
) -> None:
    invalid = _terminal(tmp_path, "a.jpg", valid=False)
    valid_null = _terminal(tmp_path, "b.jpg", jaguar_id=None, valid=True)
    valid_identity_missing = _terminal(tmp_path, "c.jpg", valid=True)
    valid_identity_matched = _terminal(tmp_path, "d.jpg", valid=True)
    enrichments = {
        valid_identity_matched.relative_filepath: _matched(),
    }

    chosen = choose_representative(
        [invalid, valid_null, valid_identity_missing, valid_identity_matched],
        enrichments=enrichments,
        require_semantic_agreement=False,
    )

    assert chosen.relative_filepath == "data/d.jpg"


def test_representative_uses_lexicographic_path_as_final_tiebreaker(
    tmp_path: Path,
) -> None:
    later = _terminal(tmp_path, "z.jpg", detection_id="generated-z")
    earlier = _terminal(tmp_path, "a.jpg", detection_id="generated-a")

    chosen = choose_representative([later, earlier])

    assert chosen.relative_filepath == "data/a.jpg"


def test_representative_semantics_ignore_mapping_insertion_order(
    tmp_path: Path,
) -> None:
    first = _terminal(tmp_path, "a.jpg", detection_id="generated-a")
    reversed_detection = {
        "bounding_box": (0.1, 0.2, 0.3, 0.4),
        "label": "jaguar",
        "_cls": "Detection",
        "_id": {"$oid": "generated-b"},
    }
    second = TerminalRecord(
        source_id="id-b",
        filepath=tmp_path / "b.jpg",
        relative_filepath="data/b.jpg",
        jaguar_id="F11",
        bboxes_body=cast(
            FrozenAnnotation,
            {"detections": [reversed_detection], "_cls": "Detections"},
        ),
        segmentations_body=cast(
            FrozenAnnotation,
            {
                "detections": [
                    {
                        **reversed_detection,
                        "mask": ((0, 1), (1, 0)),
                    }
                ],
                "_cls": "Detections",
            },
        ),
    )

    chosen = choose_representative([second, first])

    assert chosen.relative_filepath == "data/a.jpg"


def test_representative_rejects_conflicting_populated_identities(
    tmp_path: Path,
) -> None:
    first = _terminal(tmp_path, "a.jpg", jaguar_id="F11")
    second = _terminal(tmp_path, "b.jpg", jaguar_id="M03")

    with pytest.raises(CurationConflictError, match="identities"):
        choose_representative([first, second])


def test_representative_rejects_semantically_different_annotations(
    tmp_path: Path,
) -> None:
    first = _terminal(tmp_path, "a.jpg")
    second = TerminalRecord(
        source_id="id-b",
        filepath=tmp_path / "b.jpg",
        relative_filepath="data/b.jpg",
        jaguar_id="F11",
        bboxes_body=_annotation(
            detection_id="generated-b",
            bbox=(0.2, 0.2, 0.3, 0.4),
        ),
        segmentations_body=_annotation(
            detection_id="generated-b",
            bbox=(0.2, 0.2, 0.3, 0.4),
            mask=((0, 1), (1, 0)),
        ),
    )

    with pytest.raises(CurationConflictError, match="semantic annotations"):
        choose_representative([first, second])


def _sample(
    name: str,
    *,
    source_id: str,
    jaguar_id: str | None,
    detection_id: str,
    bbox: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
    segmentation_bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, object]:
    return {
        "_id": {"$oid": source_id},
        "_dataset_id": {"$oid": "stale-dataset"},
        "_rand": 0.25,
        "filepath": f"data/{name}",
        "jaguar_id": jaguar_id,
        "bboxes_body": _annotation(
            detection_id=detection_id,
            bbox=bbox,
        ),
        "segmentations_body": _annotation(
            detection_id=detection_id,
            bbox=segmentation_bbox or bbox,
            mask=((0, 1), (1, 0)),
        ),
    }


def _write_source_export(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    media = source / "data"
    media.mkdir(parents=True)
    Image.new("RGB", (8, 6), (10, 20, 30)).save(media / "a.png")
    (media / "b.png").write_bytes((media / "a.png").read_bytes())
    Image.new("RGB", (8, 6), (30, 20, 10)).save(media / "false.png")
    Image.new("RGB", (8, 6), (40, 50, 60)).save(media / "review.png")
    Image.new("RGB", (8, 6), (70, 80, 90)).save(media / "clip.png")
    samples = [
        _sample(
            "a.png",
            source_id="a" * 24,
            jaguar_id="F11",
            detection_id="generated-a",
        ),
        _sample(
            "b.png",
            source_id="b" * 24,
            jaguar_id="F11",
            detection_id="generated-b",
        ),
        _sample(
            "false.png",
            source_id="c" * 24,
            jaguar_id=None,
            detection_id="generated-c",
        ),
        _sample(
            "review.png",
            source_id="d" * 24,
            jaguar_id="F14",
            detection_id="generated-d",
            bbox=(-0.01, 0.2, 0.3, 0.4),
        ),
        _sample(
            "clip.png",
            source_id="e" * 24,
            jaguar_id=None,
            detection_id="generated-e",
            bbox=(0.1, -0.1, 0.3, 0.7),
            segmentation_bbox=(0.1, 0.0, 0.3, 0.6),
        ),
    ]
    (source / "samples.json").write_text(
        json.dumps({"samples": samples}),
        encoding="utf-8",
    )
    (source / "metadata.json").write_text(
        json.dumps({"name": "source", "version": "1.11.0", "_id": {"$oid": "f" * 24}}),
        encoding="utf-8",
    )
    return source


def _fixture_policy() -> CurationPolicy:
    return CurationPolicy(
        version="test-v1",
        false_positive_paths=frozenset({"data/false.png"}),
        review_cases=MappingProxyType({"data/review.png": "zero-area mask"}),
        clipped_bboxes=MappingProxyType({"data/clip.png": (0.1, 0.0, 0.3, 0.6)}),
        expected_source_count=5,
        expected_curated_count=3,
        expected_populated_identities=2,
        expected_null_identities=1,
        expected_distinct_identities=2,
    )


def test_dry_run_plan_deduplicates_excludes_reviews_clips_and_validates_counts(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)

    plan = build_curation_plan(
        source,
        tmp_path / "target",
        policy=_fixture_policy(),
    )

    assert plan.source_count == 5
    assert plan.curated_count == 3
    assert plan.unique_hashes == 3
    assert plan.populated_identities == 2
    assert plan.null_identities == 1
    assert plan.distinct_identities == 2
    assert plan.kept_paths == (
        "data/a.png",
        "data/clip.png",
        "data/review.png",
    )
    assert {(drop.relative_filepath, drop.reason) for drop in plan.dropped} == {
        ("data/b.png", "exact_content_duplicate"),
        ("data/false.png", "confirmed_false_positive"),
    }
    assert plan.clipped_paths == ("data/clip.png",)
    assert plan.review_paths == ("data/review.png",)
    assert len(plan.hash_groups) == 1
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize(
    "policy",
    [
        replace(
            _fixture_policy(),
            false_positive_paths=frozenset(
                {
                    "data/b.png",
                    "data/false.png",
                }
            ),
        ),
        replace(
            _fixture_policy(),
            review_cases=MappingProxyType(
                {
                    "data/b.png": "duplicate review case",
                    "data/review.png": "zero-area mask",
                }
            ),
        ),
        replace(
            _fixture_policy(),
            clipped_bboxes=MappingProxyType(
                {
                    "data/b.png": (0.1, 0.2, 0.3, 0.4),
                    "data/clip.png": (0.1, 0.0, 0.3, 0.6),
                }
            ),
        ),
    ],
)
def test_plan_rejects_configured_policy_actions_lost_during_deduplication(
    tmp_path: Path,
    policy: CurationPolicy,
) -> None:
    source = _write_source_export(tmp_path)

    with pytest.raises(CurationConflictError, match="policy actions"):
        build_curation_plan(
            source,
            tmp_path / "target",
            policy=policy,
        )


def test_materialization_references_original_media_and_writes_metadata_only(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    source_samples_before = (source / "samples.json").read_bytes()
    source_media_before = {path.name: path.read_bytes() for path in (source / "data").iterdir()}
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    result = materialize_curated_export(plan)

    assert result == target.resolve()
    assert (source / "samples.json").read_bytes() == source_samples_before
    assert {path.name: path.read_bytes() for path in (source / "data").iterdir()} == source_media_before
    assert {path.name for path in target.iterdir()} == {
        "curation_report.json",
        "metadata.json",
        "samples.json",
    }
    assert not (target / "data").exists()
    released_locks = list(tmp_path.glob(".target.released-lock-*"))
    assert len(released_locks) == 1
    assert released_locks[0].is_file()

    payload = json.loads((target / "samples.json").read_text(encoding="utf-8"))
    samples = {Path(sample["filepath"]).name: sample for sample in payload["samples"]}
    assert {Path(sample["filepath"]) for sample in samples.values()} == {sample.terminal.filepath for sample in plan.selected}
    assert all(Path(sample["filepath"]).is_relative_to((source / "data").resolve()) for sample in samples.values())
    assert all("_dataset_id" not in sample and "_rand" not in sample for sample in samples.values())
    assert all(len(sample["_id"]["$oid"]) == 24 for sample in samples.values())

    review = samples["review.png"]
    assert "bboxes_body" not in review
    assert "segmentations_body" not in review
    assert review["review_required"] is True
    assert review["review_reason"] == "zero-area mask"
    assert review["review_status"] == "pending"
    assert review["tags"] == ["needs_annotation_review"]

    clipped = samples["clip.png"]
    assert clipped["bboxes_body"]["detections"][0]["bounding_box"] == [
        0.1,
        0.0,
        0.3,
        0.6,
    ]
    assert "_id" not in clipped["bboxes_body"]["detections"][0]
    assert "_id" not in clipped["segmentations_body"]["detections"][0]

    report = json.loads((target / "curation_report.json").read_text(encoding="utf-8"))
    assert report["policy_version"] == "test-v1"
    assert report["media_storage"] == "canonical_source_references"
    assert report["allowed_media_root"] == str((source / "data").resolve())
    assert report["source_samples_sha256"] == plan.source_samples_sha256
    assert report["counts"] == {
        "source": 5,
        "curated": 3,
        "dropped": 2,
        "duplicate_groups": 1,
        "unique_hashes": 3,
        "populated_identities": 2,
        "null_identities": 1,
        "distinct_identities": 2,
        "review_required": 1,
        "bbox_clipped": 1,
    }
    assert report["reviews"] == [
        {
            "relative_filepath": "data/review.png",
            "reason": "zero-area mask",
            "action": "stripped_invalid_annotations",
        }
    ]

    parsed = load_terminal_records(
        target,
        allowed_media_root=source / "data",
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
            for terminal in parsed
        ],
        expected_count=3,
    )
    assert len(validated) == 3
    assert sum(record.terminal.review_required for record in validated) == 1


def test_materialization_validates_staging_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_payload = final_curation._curated_sample_payload

    def malformed_payload(
        sample: final_curation.CuratedSample,
        policy_version: str,
    ) -> dict[str, object]:
        payload = real_payload(sample, policy_version)
        if sample.review_reason is None:
            payload.pop("bboxes_body", None)
        return payload

    monkeypatch.setattr(
        final_curation,
        "_curated_sample_payload",
        malformed_payload,
    )

    with pytest.raises(CurationError, match="staged export"):
        materialize_curated_export(plan)

    assert not target.exists()
    assert not list(tmp_path.glob(".target.building-*"))
    retired_staging = list(tmp_path.glob(".target.retired-staging-*"))
    assert len(retired_staging) == 1
    assert {path.name for path in retired_staging[0].iterdir()} == {
        "curation_report.json",
        "metadata.json",
        "samples.json",
    }


def test_materialization_rejects_media_drift_after_planning(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    Image.new("RGB", (8, 6), (100, 110, 120)).save(source / "data/a.png")

    with pytest.raises(CurationError, match="media hashes differ"):
        materialize_curated_export(plan)

    assert not target.exists()
    assert not list(tmp_path.glob(".target.building-*"))


@pytest.mark.parametrize("overwrite", [False, True])
def test_materialization_preserves_target_created_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_validate = final_curation._validate_staged_export

    def validate_then_create_concurrent_target(
        curation_plan: final_curation.CurationPlan,
        staging_dir: Path,
    ) -> None:
        real_validate(curation_plan, staging_dir)
        target.mkdir()
        (target / "foreign.txt").write_text("must survive", encoding="utf-8")

    monkeypatch.setattr(
        final_curation,
        "_validate_staged_export",
        validate_then_create_concurrent_target,
    )

    with pytest.raises(CurationError, match="target changed"):
        materialize_curated_export(
            plan,
            overwrite=overwrite,
        )

    assert (target / "foreign.txt").read_text(encoding="utf-8") == "must survive"
    assert not (target / "curation_report.json").exists()
    assert not list(tmp_path.glob(".target.building-*"))
    assert not list(tmp_path.glob(".target.backup-*"))
    assert not (tmp_path / ".target.lock").exists()


def test_materialization_preserves_replacement_of_confirmed_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    original = target / "original.txt"
    original.write_text("confirmed target", encoding="utf-8")
    displaced = tmp_path / "confirmed-target-displaced"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_validate = final_curation._validate_staged_export

    def validate_then_replace_confirmed_target(
        curation_plan: final_curation.CurationPlan,
        staging_dir: Path,
    ) -> None:
        real_validate(curation_plan, staging_dir)
        final_curation.os.replace(target, displaced)
        target.mkdir()
        (target / "foreign.txt").write_text("must survive", encoding="utf-8")

    monkeypatch.setattr(
        final_curation,
        "_validate_staged_export",
        validate_then_replace_confirmed_target,
    )

    with pytest.raises(CurationError, match="target changed"):
        materialize_curated_export(
            plan,
            overwrite=True,
            confirmed=True,
        )

    assert (target / "foreign.txt").read_text(encoding="utf-8") == "must survive"
    assert (displaced / "original.txt").read_text(encoding="utf-8") == "confirmed target"
    assert not list(tmp_path.glob(".target.building-*"))
    assert not list(tmp_path.glob(".target.backup-*"))
    assert not (tmp_path / ".target.lock").exists()


def test_materialization_verifies_backup_identity_after_final_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    (target / "original.txt").write_text("confirmed target", encoding="utf-8")
    displaced = tmp_path / "confirmed-target-displaced"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_rename_at = final_curation._rename_directory_noreplace_at
    real_replace = final_curation.os.replace
    injected = False

    def replace_target_between_check_and_rename(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal injected
        if not injected and source_name == target.name and target_name.startswith(".target.backup-"):
            injected = True
            real_replace(target, displaced)
            target.mkdir()
            (target / "foreign.txt").write_text("must survive", encoding="utf-8")
            real_replace(target, target.parent / target_name)
            return
        real_rename_at(parent_fd, source_name, target_name)

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        replace_target_between_check_and_rename,
    )

    with pytest.raises(CurationError, match="backup identity"):
        materialize_curated_export(
            plan,
            overwrite=True,
            confirmed=True,
        )

    assert injected is True
    assert (target / "foreign.txt").read_text(encoding="utf-8") == "must survive"
    assert (displaced / "original.txt").read_text(encoding="utf-8") == "confirmed target"
    assert not list(tmp_path.glob(".target.backup-*"))
    assert not list(tmp_path.glob(".target.recovery-*"))
    assert not list(tmp_path.glob(".target.building-*"))


def test_materialization_retains_unexpected_backup_when_restore_target_collides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    (target / "original.txt").write_text("confirmed target", encoding="utf-8")
    displaced = tmp_path / "confirmed-target-displaced"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_rename_at = final_curation._rename_directory_noreplace_at
    real_replace = final_curation.os.replace
    injected = False

    def replace_target_and_create_restore_collision(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal injected
        if not injected and source_name == target.name and target_name.startswith(".target.backup-"):
            injected = True
            real_replace(target, displaced)
            target.mkdir()
            (target / "foreign.txt").write_text("recover me", encoding="utf-8")
            real_replace(target, target.parent / target_name)
            target.mkdir()
            (target / "collision.txt").write_text("do not replace", encoding="utf-8")
            return
        real_rename_at(parent_fd, source_name, target_name)

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        replace_target_and_create_restore_collision,
    )

    with pytest.raises(CurationError, match="backup identity"):
        materialize_curated_export(
            plan,
            overwrite=True,
            confirmed=True,
        )

    recoveries = list(tmp_path.glob(".target.recovery-*"))
    assert injected is True
    assert (target / "collision.txt").read_text(encoding="utf-8") == "do not replace"
    assert len(recoveries) == 1
    assert (recoveries[0] / "foreign.txt").read_text(encoding="utf-8") == "recover me"
    assert (displaced / "original.txt").read_text(encoding="utf-8") == "confirmed target"
    assert not list(tmp_path.glob(".target.backup-*"))
    assert not list(tmp_path.glob(".target.building-*"))


def test_materialization_pins_existing_curation_report_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    initial_plan = build_curation_plan(source, target, policy=_fixture_policy())
    materialize_curated_export(initial_plan)
    replacement_plan = build_curation_plan(
        source,
        target,
        policy=_fixture_policy(),
    )
    real_validate = final_curation._validate_staged_export

    def validate_then_change_report(
        curation_plan: final_curation.CurationPlan,
        staging_dir: Path,
    ) -> None:
        real_validate(curation_plan, staging_dir)
        report_path = target / "curation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["source_samples_sha256"] = "foreign-source"
        report_path.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(
        final_curation,
        "_validate_staged_export",
        validate_then_change_report,
    )

    with pytest.raises(CurationError, match="target changed"):
        materialize_curated_export(
            replacement_plan,
            overwrite=True,
            confirmed=True,
        )

    report = json.loads((target / "curation_report.json").read_text(encoding="utf-8"))
    assert report["source_samples_sha256"] == "foreign-source"
    assert not list(tmp_path.glob(".target.building-*"))
    assert not list(tmp_path.glob(".target.backup-*"))
    assert not (tmp_path / ".target.lock").exists()


def test_materialization_refuses_foreign_sibling_lock_without_removing_it(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    lock = tmp_path / ".target.lock"
    lock.write_text("foreign owner", encoding="utf-8")
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    with pytest.raises(CurationError, match="lock"):
        materialize_curated_export(plan)

    assert lock.read_text(encoding="utf-8") == "foreign owner"
    assert not target.exists()
    assert not list(tmp_path.glob(".target.building-*"))


def test_materialization_cleanup_never_removes_replacement_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    lock = tmp_path / ".target.lock"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_validate = final_curation._validate_staged_export

    def validate_then_replace_lock(
        curation_plan: final_curation.CurationPlan,
        staging_dir: Path,
    ) -> None:
        real_validate(curation_plan, staging_dir)
        lock.unlink()
        lock.write_text("foreign replacement", encoding="utf-8")
        raise CurationError("injected validation failure")

    monkeypatch.setattr(
        final_curation,
        "_validate_staged_export",
        validate_then_replace_lock,
    )

    with pytest.raises(CurationError, match="injected"):
        materialize_curated_export(plan)

    assert lock.read_text(encoding="utf-8") == "foreign replacement"
    assert not target.exists()
    assert not list(tmp_path.glob(".target.building-*"))


@pytest.mark.skipif(
    final_curation.sys.platform != "darwin",
    reason="exercises macOS renamex_np(RENAME_EXCL)",
)
def test_macos_publication_atomically_preserves_empty_claimant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_rename_noreplace = final_curation._rename_directory_noreplace_at
    claimant_inode: int | None = None

    def claim_then_publish(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal claimant_inode
        if target_name == target.name:
            target.mkdir()
            claimant_inode = target.stat().st_ino
        real_rename_noreplace(parent_fd, source_name, target_name)

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        claim_then_publish,
    )

    with pytest.raises(CurationError, match="concurrent target.*preserved"):
        materialize_curated_export(plan)

    assert claimant_inode is not None
    assert target.stat().st_ino == claimant_inode
    assert list(target.iterdir()) == []
    assert not (target / "curation_report.json").exists()
    assert not list(tmp_path.glob(".target.building-*"))
    assert not (tmp_path / ".target.lock").exists()


def test_target_storage_rejects_unmounted_state_volume(tmp_path: Path) -> None:
    mount_root = tmp_path / "CameraTrapPython"
    approved_root = mount_root / "fiftyone/exports"
    target = approved_root / "curated"

    with pytest.raises(CurationError, match="mounted filesystem"):
        final_curation._validate_target_storage(
            target,
            mount_root=mount_root,
            approved_root=approved_root,
            is_mount_fn=lambda _path: False,
            capability_probe=lambda _path: True,
        )


def test_target_storage_requires_absolute_target(tmp_path: Path) -> None:
    mount_root = tmp_path / "CameraTrapPython"
    mount_root.mkdir()
    approved_root = mount_root / "fiftyone/exports"

    with pytest.raises(CurationError, match="absolute path"):
        final_curation._validate_target_storage(
            Path("relative-curated-target"),
            mount_root=mount_root,
            approved_root=approved_root,
            is_mount_fn=lambda _path: True,
            capability_probe=lambda _path: True,
        )


@pytest.mark.parametrize(
    ("capabilities", "valid", "expected"),
    [
        (0x00080000, 0x00080000, True),
        (0x00000000, 0x00080000, False),
        (0x00080000, 0x00000000, False),
    ],
)
def test_target_storage_requires_valid_exclusive_rename_capability_bit(
    capabilities: int,
    valid: int,
    expected: bool,
) -> None:
    assert final_curation._has_atomic_rename_capability(capabilities, valid) is expected


@pytest.mark.parametrize("target_kind", ["root", "outside"])
def test_target_storage_requires_strict_approved_root_descendant(
    tmp_path: Path,
    target_kind: str,
) -> None:
    mount_root = tmp_path / "CameraTrapPython"
    mount_root.mkdir()
    approved_root = mount_root / "fiftyone/exports"
    target = approved_root if target_kind == "root" else mount_root / "foreign/curated"

    with pytest.raises(CurationError, match="strict descendant"):
        final_curation._validate_target_storage(
            target,
            mount_root=mount_root,
            approved_root=approved_root,
            is_mount_fn=lambda _path: True,
            capability_probe=lambda _path: True,
        )


def test_target_storage_rejects_volume_without_atomic_noreplace(
    tmp_path: Path,
) -> None:
    mount_root = tmp_path / "CameraTrapPython"
    mount_root.mkdir()
    approved_root = mount_root / "fiftyone/exports"
    target = approved_root / "curated"

    with pytest.raises(CurationError, match="atomic no-clobber"):
        final_curation._validate_target_storage(
            target,
            mount_root=mount_root,
            approved_root=approved_root,
            is_mount_fn=lambda _path: True,
            capability_probe=lambda _path: False,
        )


def test_target_storage_accepts_capable_approved_volume(tmp_path: Path) -> None:
    mount_root = tmp_path / "CameraTrapPython"
    approved_root = mount_root / "fiftyone/exports"
    target = approved_root / "curated"
    approved_root.mkdir(parents=True)

    final_curation._validate_target_storage(
        target,
        mount_root=mount_root,
        approved_root=approved_root,
        is_mount_fn=lambda _path: True,
        capability_probe=lambda _path: True,
    )


def test_target_storage_probes_deepest_existing_target_ancestor(
    tmp_path: Path,
) -> None:
    mount_root = tmp_path / "CameraTrapPython"
    approved_root = mount_root / "fiftyone/exports"
    deepest_existing = approved_root / "nested"
    deepest_existing.mkdir(parents=True)
    target = deepest_existing / "missing/curated"
    probed: list[Path] = []

    final_curation._validate_target_storage(
        target,
        mount_root=mount_root,
        approved_root=approved_root,
        is_mount_fn=lambda _path: True,
        capability_probe=lambda path: not probed.append(path),
    )

    assert probed == [deepest_existing.resolve()]


def test_target_storage_rejects_nested_target_filesystem(
    tmp_path: Path,
) -> None:
    mount_root = tmp_path / "CameraTrapPython"
    approved_root = mount_root / "fiftyone/exports"
    nested_mount = approved_root / "nested"
    nested_mount.mkdir(parents=True)
    target = nested_mount / "curated"

    with pytest.raises(CurationError, match="same filesystem"):
        final_curation._validate_target_storage(
            target,
            mount_root=mount_root,
            approved_root=approved_root,
            is_mount_fn=lambda _path: True,
            capability_probe=lambda _path: True,
            device_fn=lambda path: 2 if path == nested_mount.resolve() else 1,
        )


def test_target_storage_rejects_mount_replaced_during_capability_probe(
    tmp_path: Path,
) -> None:
    mount_root = tmp_path / "CameraTrapPython"
    mount_root.mkdir()
    approved_root = mount_root / "fiftyone/exports"
    target = approved_root / "curated"
    displaced = tmp_path / "displaced-mount"

    def replace_mount(_path: Path) -> bool:
        mount_root.rename(displaced)
        mount_root.mkdir()
        return True

    with pytest.raises(CurationError, match="changed during validation"):
        final_curation._validate_target_storage(
            target,
            mount_root=mount_root,
            approved_root=approved_root,
            is_mount_fn=lambda _path: True,
            capability_probe=replace_mount,
        )


def test_target_storage_ancestor_swap_cannot_redirect_first_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    mount_root = tmp_path / "CameraTrapPython"
    approved_root = mount_root / "fiftyone/exports"
    approved_root.mkdir(parents=True)
    target = approved_root / "curated"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    displaced = tmp_path / "displaced-fiftyone"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_validate = final_curation._validate_target_storage

    def validate_then_swap_ancestor(_target: Path) -> object:
        validation = real_validate(
            target,
            mount_root=mount_root,
            approved_root=approved_root,
            is_mount_fn=lambda _path: True,
            capability_probe=lambda _path: True,
        )
        (mount_root / "fiftyone").rename(displaced)
        (mount_root / "fiftyone").symlink_to(
            outside,
            target_is_directory=True,
        )
        return validation

    monkeypatch.setattr(
        final_curation,
        "_validate_target_storage",
        validate_then_swap_ancestor,
    )

    with pytest.raises(CurationError, match="approved target parent"):
        materialize_curated_export(plan)

    assert not (outside / "exports").exists()
    assert not list(outside.glob("**/.curated.lock"))
    assert not list(outside.glob("**/.curated.building-*"))


def test_target_storage_ancestor_swap_during_build_cannot_redirect_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    mount_root = tmp_path / "CameraTrapPython"
    approved_root = mount_root / "fiftyone/exports"
    approved_root.mkdir(parents=True)
    target = approved_root / "curated"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    displaced = tmp_path / "displaced-fiftyone"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_validate = final_curation._validate_target_storage
    real_build_at = final_curation._build_export_at

    def validate_storage(_target: Path) -> object:
        return real_validate(
            target,
            mount_root=mount_root,
            approved_root=approved_root,
            is_mount_fn=lambda _path: True,
            capability_probe=lambda _path: True,
        )

    def swap_ancestor_then_build(
        curation_plan: final_curation.CurationPlan,
        descriptor: int,
    ) -> None:
        staging_name = final_curation._descriptor_path(descriptor).name
        (mount_root / "fiftyone").rename(displaced)
        (mount_root / "fiftyone").symlink_to(
            outside,
            target_is_directory=True,
        )
        foreign_staging = outside / "exports" / staging_name
        foreign_staging.mkdir(parents=True)
        real_build_at(curation_plan, descriptor)

    monkeypatch.setattr(
        final_curation,
        "_validate_target_storage",
        validate_storage,
    )
    monkeypatch.setattr(
        final_curation,
        "_build_export_at",
        swap_ancestor_then_build,
    )

    with pytest.raises(CurationError, match="approved target parent changed"):
        materialize_curated_export(plan)

    assert not list(outside.glob("**/*.json"))
    assert not (displaced / "exports/curated").exists()


def test_target_storage_parent_relocation_during_publication_is_not_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    mount_root = tmp_path / "CameraTrapPython"
    approved_root = mount_root / "fiftyone/exports"
    approved_root.mkdir(parents=True)
    target = approved_root / "curated"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    displaced = tmp_path / "displaced-exports"
    real_validate = final_curation._validate_target_storage
    real_rename_at = final_curation._rename_directory_noreplace_at

    def validate_storage(_target: Path) -> object:
        return real_validate(
            target,
            mount_root=mount_root,
            approved_root=approved_root,
            is_mount_fn=lambda _path: True,
            capability_probe=lambda _path: True,
        )

    def publish_then_relocate_parent(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        real_rename_at(parent_fd, source_name, target_name)
        if source_name.startswith(".curated.building-") and target_name == target.name:
            approved_root.rename(displaced)
            approved_root.mkdir(parents=True)

    monkeypatch.setattr(
        final_curation,
        "_validate_target_storage",
        validate_storage,
    )
    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        publish_then_relocate_parent,
    )

    with pytest.raises(CurationError, match="approved target parent changed"):
        materialize_curated_export(plan)

    assert not target.exists()
    assert (displaced / "curated/curation_report.json").is_file()


def test_target_storage_guard_runs_before_parent_or_staging_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "missing-parent/target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    def reject_storage(_target: Path) -> None:
        raise CurationError("injected storage rejection")

    monkeypatch.setattr(
        final_curation,
        "_validate_target_storage",
        reject_storage,
    )

    with pytest.raises(CurationError, match="storage rejection"):
        materialize_curated_export(plan)

    assert not target.parent.exists()
    assert not list(tmp_path.glob("**/.target.building-*"))
    assert not list(tmp_path.glob("**/.target.lock"))


def test_atomic_publication_fails_closed_on_unsupported_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    monkeypatch.setattr(final_curation.sys, "platform", "unsupported")

    with pytest.raises(CurationError, match="unsupported"):
        final_curation._rename_directory_noreplace(source, target)

    assert source.is_dir()
    assert not target.exists()


def test_materialization_requires_explicit_confirmed_overwrite_and_replaces_atomically(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "existing.txt"
    sentinel.write_text("keep until replacement is ready", encoding="utf-8")
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    with pytest.raises(CurationError, match="already exists"):
        materialize_curated_export(plan)
    assert sentinel.read_text(encoding="utf-8") == "keep until replacement is ready"

    with pytest.raises(CurationError, match="confirmation"):
        materialize_curated_export(plan, overwrite=True)
    assert sentinel.read_text(encoding="utf-8") == "keep until replacement is ready"

    materialize_curated_export(plan, overwrite=True, confirmed=True)

    assert not sentinel.exists()
    assert (target / "curation_report.json").is_file()
    assert not list(tmp_path.glob(".target.backup-*"))
    assert not (tmp_path / ".target.lock").exists()


def test_overwrite_promotion_failure_restores_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "existing.txt"
    sentinel.write_text("recover me", encoding="utf-8")
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_rename_noreplace = final_curation._rename_directory_noreplace_at

    def fail_promotion(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        if source_name.startswith(".target.building-") and target_name == target.name:
            raise OSError("injected promotion failure")
        real_rename_noreplace(parent_fd, source_name, target_name)

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        fail_promotion,
    )

    with pytest.raises(CurationError, match="publish"):
        materialize_curated_export(plan, overwrite=True, confirmed=True)

    assert sentinel.read_text(encoding="utf-8") == "recover me"
    assert not list(tmp_path.glob(".target.building-*"))
    assert not list(tmp_path.glob(".target.backup-*"))


def test_overwrite_promotion_collision_preserves_claimant_and_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "existing.txt"
    sentinel.write_text("recover me", encoding="utf-8")
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_rename_noreplace = final_curation._rename_directory_noreplace_at
    claimant_inode: int | None = None

    def collide_at_promotion(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal claimant_inode
        if source_name.startswith(".target.building-") and target_name == target.name:
            target.mkdir()
            claimant_inode = target.stat().st_ino
        real_rename_noreplace(parent_fd, source_name, target_name)

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        collide_at_promotion,
    )

    with pytest.raises(
        CurationError,
        match=r"concurrent target.*preserved.*\.target\.recovery-",
    ):
        materialize_curated_export(plan, overwrite=True, confirmed=True)

    assert claimant_inode is not None
    assert target.stat().st_ino == claimant_inode
    assert list(target.iterdir()) == []
    recoveries = list(tmp_path.glob(".target.recovery-*"))
    assert len(recoveries) == 1
    assert (recoveries[0] / "existing.txt").read_text(encoding="utf-8") == ("recover me")
    assert not list(tmp_path.glob(".target.building-*"))
    assert not list(tmp_path.glob(".target.backup-*"))
    assert not (tmp_path / ".target.lock").exists()


def test_owned_staging_cleanup_preserves_foreign_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_rename_noreplace = final_curation._rename_directory_noreplace_at
    foreign_inode: int | None = None
    foreign_path: Path | None = None

    def publish_then_replace_staging(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal foreign_inode, foreign_path
        real_rename_noreplace(parent_fd, source_name, target_name)
        if target_name == target.name:
            source_path = target.parent / source_name
            source_path.mkdir()
            (source_path / "foreign.txt").write_text(
                "must survive",
                encoding="utf-8",
            )
            foreign_path = source_path
            foreign_inode = source_path.stat().st_ino

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        publish_then_replace_staging,
    )

    materialize_curated_export(plan)

    assert foreign_path is not None
    assert foreign_inode is not None
    assert foreign_path.stat().st_ino == foreign_inode
    assert (foreign_path / "foreign.txt").read_text(encoding="utf-8") == ("must survive")
    assert (target / "curation_report.json").is_file()


def test_publication_rejects_foreign_replacement_of_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_rename_at = final_curation._rename_directory_noreplace_at
    displaced = tmp_path / "owned-staging-displaced"
    foreign_inode: int | None = None

    def replace_staging_then_publish(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal foreign_inode
        if source_name.startswith(".target.building-") and target_name == target.name:
            staging_path = target.parent / source_name
            staging_path.rename(displaced)
            staging_path.mkdir()
            (staging_path / "foreign.txt").write_text(
                "must survive",
                encoding="utf-8",
            )
            foreign_inode = staging_path.stat().st_ino
        real_rename_at(parent_fd, source_name, target_name)

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        replace_staging_then_publish,
    )

    with pytest.raises(CurationError, match="owned staging"):
        materialize_curated_export(plan)

    assert foreign_inode is not None
    assert target.stat().st_ino == foreign_inode
    assert (target / "foreign.txt").read_text(encoding="utf-8") == ("must survive")
    assert (displaced / "curation_report.json").is_file()


def test_publication_mismatch_never_cleans_owned_export_moved_back_to_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())
    real_rename_at = final_curation._rename_directory_noreplace_at
    returned_staging: Path | None = None

    def publish_move_owned_back_and_claim_target(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal returned_staging
        real_rename_at(parent_fd, source_name, target_name)
        if source_name.startswith(".target.building-") and target_name == target.name:
            returned_staging = target.parent / source_name
            target.rename(returned_staging)
            target.mkdir()
            (target / "foreign.txt").write_text(
                "must survive",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        publish_move_owned_back_and_claim_target,
    )

    with pytest.raises(CurationError, match="owned staging"):
        materialize_curated_export(plan)

    assert returned_staging is not None
    assert (returned_staging / "curation_report.json").is_file()
    assert (target / "foreign.txt").read_text(encoding="utf-8") == ("must survive")


def test_owned_directory_retirement_preserves_preexisting_foreign_replacement(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".target.building-owned"
    staging.mkdir()
    (staging / "samples.json").write_text("owned", encoding="utf-8")
    expected = final_curation._capture_directory_identity(staging)
    assert expected is not None
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    staging_fd = os.open(staging, os.O_RDONLY)
    displaced = tmp_path / "owned-displaced"
    staging.rename(displaced)
    staging.mkdir()
    (staging / "foreign.txt").write_text(
        "must survive",
        encoding="utf-8",
    )
    foreign_inode = staging.stat().st_ino
    try:
        removed = final_curation._retire_owned_directory_at(
            parent_fd,
            staging.name,
            staging_fd,
            expected,
        )
    finally:
        os.close(staging_fd)
        os.close(parent_fd)

    assert removed is False
    assert staging.stat().st_ino == foreign_inode
    assert (staging / "foreign.txt").read_text(encoding="utf-8") == ("must survive")
    assert displaced.is_dir()
    assert (displaced / "samples.json").read_text(encoding="utf-8") == "owned"


def test_owned_directory_retirement_restores_last_moment_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".target.building-owned"
    staging.mkdir()
    (staging / "samples.json").write_text("owned", encoding="utf-8")
    expected = final_curation._capture_directory_identity(staging)
    assert expected is not None
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    staging_fd = os.open(staging, os.O_RDONLY)
    displaced = tmp_path / "owned-displaced-at-retirement"
    foreign_inode: int | None = None
    real_rename_at = final_curation._rename_directory_noreplace_at

    def swap_at_retirement(
        directory_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal foreign_inode
        if source_name == staging.name and ".retired-staging-" in target_name:
            staging.rename(displaced)
            staging.mkdir()
            foreign_inode = staging.stat().st_ino
        real_rename_at(directory_fd, source_name, target_name)

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        swap_at_retirement,
    )
    try:
        removed = final_curation._retire_owned_directory_at(
            parent_fd,
            staging.name,
            staging_fd,
            expected,
        )
    finally:
        os.close(staging_fd)
        os.close(parent_fd)

    assert removed is False
    assert foreign_inode is not None
    assert staging.stat().st_ino == foreign_inode
    assert displaced.is_dir()
    assert (displaced / "samples.json").read_text(encoding="utf-8") == "owned"


def test_lock_retirement_restores_last_moment_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    lock = tmp_path / ".target.lock"
    displaced = tmp_path / "owned-lock-displaced"
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    real_rename_at = final_curation._rename_directory_noreplace_at
    foreign_inode: int | None = None

    def swap_at_retirement(
        directory_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal foreign_inode
        if source_name == lock.name and ".released-lock-" in target_name:
            lock.rename(displaced)
            lock.write_text(
                "foreign replacement",
                encoding="utf-8",
            )
            foreign_inode = lock.stat().st_ino
        real_rename_at(directory_fd, source_name, target_name)

    monkeypatch.setattr(
        final_curation,
        "_rename_directory_noreplace_at",
        swap_at_retirement,
    )
    try:
        with pytest.raises(
            CurationError,
            match="safely release",
        ), final_curation._exclusive_target_lock(
            target,
            parent_fd=parent_fd,
        ):
            pass
    finally:
        os.close(parent_fd)

    assert foreign_inode is not None
    assert lock.stat().st_ino == foreign_inode
    assert lock.read_text(encoding="utf-8") == "foreign replacement"


def test_materialization_rejects_a_target_that_contains_the_source(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    source = _write_source_export(target)
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    with pytest.raises(CurationError, match="source export"):
        materialize_curated_export(
            plan,
            overwrite=True,
            confirmed=True,
        )

    assert source.is_dir()
    assert (source / "samples.json").is_file()


def test_materialization_rejects_lexical_source_descendant_through_symlink(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    redirect = source / "redirect"
    redirect.symlink_to(outside, target_is_directory=True)
    target = redirect / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    with pytest.raises(CurationError, match="inside the source export"):
        materialize_curated_export(plan)

    assert redirect.is_symlink()
    assert not target.exists()


def test_materialization_rejects_symlink_target_without_touching_referent(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    referent = tmp_path / "referent"
    referent.mkdir()
    sentinel = referent / "existing.txt"
    sentinel.write_text("must survive", encoding="utf-8")
    target = tmp_path / "target"
    target.symlink_to(referent, target_is_directory=True)
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    with pytest.raises(CurationError, match="symbolic link"):
        materialize_curated_export(
            plan,
            overwrite=True,
            confirmed=True,
        )

    assert target.is_symlink()
    assert target.resolve() == referent.resolve()
    assert sentinel.read_text(encoding="utf-8") == "must survive"


def test_curation_cli_defaults_to_read_only_audit_of_approved_paths() -> None:
    args = parse_args([])

    assert args.source == DEFAULT_SOURCE_EXPORT_DIR
    assert args.target == DEFAULT_TARGET_EXPORT_DIR == Path("/Volumes/CameraTrapPython/fiftyone/exports/" "JaguarCameraTrap_Final_Curated_v1")
    assert args.create is False
    assert args.overwrite is False
    assert args.yes is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--yes"],
        ["--overwrite"],
        ["--create", "--yes"],
    ],
)
def test_curation_cli_rejects_unsafe_overwrite_combinations(
    argv: list[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        parse_args(argv)

    assert caught.value.code == 2


def test_curation_cli_dry_run_reports_without_writing(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    output: list[str] = []

    exit_code = run(
        parse_args(["--source", str(source), "--target", str(target)]),
        policy=_fixture_policy(),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert not target.exists()
    report = json.loads(output[-1])
    assert report["counts"]["curated"] == 3
    assert report["counts"]["review_required"] == 1


def test_curation_cli_create_requires_exact_interactive_overwrite_confirmation(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    (target / "existing.txt").write_text("unchanged", encoding="utf-8")
    args = parse_args(
        [
            "--create",
            "--overwrite",
            "--source",
            str(source),
            "--target",
            str(target),
        ]
    )

    with pytest.raises(CurationError, match="declined"):
        run(
            args,
            policy=_fixture_policy(),
            input_fn=lambda _prompt: "wrong target",
            isatty_fn=lambda: True,
            output_fn=lambda _message: None,
        )
    assert (target / "existing.txt").read_text(encoding="utf-8") == "unchanged"

    exit_code = run(
        args,
        policy=_fixture_policy(),
        input_fn=lambda _prompt: str(target.resolve()),
        isatty_fn=lambda: True,
        output_fn=lambda _message: None,
    )

    assert exit_code == 0
    assert (target / "curation_report.json").is_file()


def test_curation_cli_noninteractive_overwrite_requires_yes_even_with_exact_input(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "existing.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    input_called = False

    def exact_input(_prompt: str) -> str:
        nonlocal input_called
        input_called = True
        return str(target.resolve())

    args = parse_args(
        [
            "--create",
            "--overwrite",
            "--source",
            str(source),
            "--target",
            str(target),
        ]
    )

    with pytest.raises(CurationError, match="--yes"):
        run(
            args,
            policy=_fixture_policy(),
            input_fn=exact_input,
            isatty_fn=lambda: False,
            output_fn=lambda _message: None,
        )

    assert input_called is False
    assert sentinel.read_text(encoding="utf-8") == "unchanged"

    yes_args = parse_args(
        [
            "--create",
            "--overwrite",
            "--yes",
            "--source",
            str(source),
            "--target",
            str(target),
        ]
    )
    exit_code = run(
        yes_args,
        policy=_fixture_policy(),
        input_fn=lambda _prompt: pytest.fail("noninteractive --yes must not prompt"),
        isatty_fn=lambda: False,
        output_fn=lambda _message: None,
    )

    assert exit_code == 0
    assert not sentinel.exists()
    assert (target / "curation_report.json").is_file()


def test_curation_cli_pins_target_before_interactive_confirmation(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    (target / "original.txt").write_text("confirmed target", encoding="utf-8")
    displaced = tmp_path / "confirmed-target-displaced"
    args = parse_args(
        [
            "--create",
            "--overwrite",
            "--source",
            str(source),
            "--target",
            str(target),
        ]
    )

    def replace_during_prompt(_prompt: str) -> str:
        final_curation.os.replace(target, displaced)
        target.mkdir()
        (target / "foreign.txt").write_text("must survive", encoding="utf-8")
        return str(target.resolve())

    with pytest.raises(CurationError, match="target changed"):
        run(
            args,
            policy=_fixture_policy(),
            input_fn=replace_during_prompt,
            isatty_fn=lambda: True,
            output_fn=lambda _message: None,
        )

    assert (target / "foreign.txt").read_text(encoding="utf-8") == "must survive"
    assert (displaced / "original.txt").read_text(encoding="utf-8") == "confirmed target"
    assert not list(tmp_path.glob(".target.building-*"))
    assert not list(tmp_path.glob(".target.backup-*"))
    assert not (tmp_path / ".target.lock").exists()
