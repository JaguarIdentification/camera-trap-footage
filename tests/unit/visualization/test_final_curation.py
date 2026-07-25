import errno
import json
import shutil
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


def test_materialization_hardlinks_media_rewrites_samples_and_records_sidecar(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    source_samples_before = (source / "samples.json").read_bytes()
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    result = materialize_curated_export(plan)

    assert result == target.resolve()
    assert (source / "samples.json").read_bytes() == source_samples_before
    for relative_path in plan.kept_paths:
        assert (source / relative_path).stat().st_ino == (target / relative_path).stat().st_ino

    payload = json.loads((target / "samples.json").read_text(encoding="utf-8"))
    samples = {sample["filepath"]: sample for sample in payload["samples"]}
    assert tuple(sorted(samples)) == plan.kept_paths
    assert all("_dataset_id" not in sample and "_rand" not in sample for sample in samples.values())
    assert all(len(sample["_id"]["$oid"]) == 24 for sample in samples.values())

    review = samples["data/review.png"]
    assert "bboxes_body" not in review
    assert "segmentations_body" not in review
    assert review["review_required"] is True
    assert review["review_reason"] == "zero-area mask"
    assert review["review_status"] == "pending"
    assert review["tags"] == ["needs_annotation_review"]

    clipped = samples["data/clip.png"]
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

    parsed = load_terminal_records(target)
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


def test_materialization_rejects_a_link_adapter_that_copies_media(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    with pytest.raises(CurationError, match="hardlink"):
        materialize_curated_export(plan, link_fn=shutil.copyfile)

    assert not target.exists()
    assert not list(tmp_path.glob(".target.building-*"))


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


def test_materialization_fails_without_partial_target_when_hardlinks_unsupported(
    tmp_path: Path,
) -> None:
    source = _write_source_export(tmp_path)
    target = tmp_path / "target"
    plan = build_curation_plan(source, target, policy=_fixture_policy())

    def unsupported(_source: Path, _target: Path) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    with pytest.raises(CurationError, match="hardlink"):
        materialize_curated_export(plan, link_fn=unsupported)

    assert not target.exists()
    assert not list(tmp_path.glob(".target.building-*"))
    assert len(list((source / "data").iterdir())) == 5


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
    real_replace = final_curation.os.replace
    calls = 0

    def fail_promotion(source_path: Path, target_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected promotion failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(final_curation.os, "replace", fail_promotion)

    with pytest.raises(CurationError, match="publish"):
        materialize_curated_export(plan, overwrite=True, confirmed=True)

    assert sentinel.read_text(encoding="utf-8") == "recover me"
    assert not list(tmp_path.glob(".target.building-*"))
    assert not list(tmp_path.glob(".target.backup-*"))


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


def test_curation_cli_defaults_to_read_only_audit_of_approved_paths() -> None:
    args = parse_args([])

    assert args.source == DEFAULT_SOURCE_EXPORT_DIR
    assert args.target == DEFAULT_TARGET_EXPORT_DIR
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
            output_fn=lambda _message: None,
        )
    assert (target / "existing.txt").read_text(encoding="utf-8") == "unchanged"

    exit_code = run(
        args,
        policy=_fixture_policy(),
        input_fn=lambda _prompt: str(target.resolve()),
        output_fn=lambda _message: None,
    )

    assert exit_code == 0
    assert (target / "curation_report.json").is_file()
