import hashlib
import io
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from PIL import Image
import pytest

from jaguars.visualization.final_lineage import Enrichment
from jaguars.visualization.final_records import FrozenAnnotation, TerminalRecord
from jaguars.visualization.final_validation import (
    HASH_CHUNK_SIZE,
    IntegrityError,
    MediaIntegrity,
    StorageSafetyError,
    ValidatedRecord,
    validate_annotations,
    validate_mounts,
    validate_media,
    validate_record,
    validate_records,
    validate_storage_paths,
    validate_unique_records,
)


def _annotation(detection: dict[str, Any]) -> FrozenAnnotation:
    return cast(FrozenAnnotation, {"detections": [detection]})


def _terminal(
    tmp_path: Path,
    *,
    bbox: object = (0.1, 0.2, 0.3, 0.4),
    filepath: Path | None = None,
    jaguar_id: str | None = "F11",
    mask: object = ((0, 1), (1, 0)),
    relative_filepath: str = "data/a.png",
    source_id: str = "source-a",
) -> TerminalRecord:
    return TerminalRecord(
        source_id=source_id,
        filepath=filepath or tmp_path / "a.png",
        relative_filepath=relative_filepath,
        jaguar_id=jaguar_id,
        bboxes_body=_annotation({"label": "jaguar", "bounding_box": bbox}),
        segmentations_body=_annotation(
            {
                "label": "jaguar",
                "bounding_box": bbox,
                "mask": mask,
            }
        ),
    )


def test_validate_media_returns_hash_size_and_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "jaguar.png"
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(image_path)

    integrity = validate_media(image_path)

    assert integrity.sha256 == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert integrity.size_bytes == image_path.stat().st_size
    assert (integrity.width, integrity.height) == (8, 6)


@pytest.mark.parametrize("kind", ["missing", "directory", "corrupt"])
def test_validate_media_rejects_missing_nonfile_and_unreadable_media(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "media"
    if kind == "directory":
        path.mkdir()
    elif kind == "corrupt":
        path.write_bytes(b"not an image")

    with pytest.raises(IntegrityError, match=r"media .*: .*media"):
        validate_media(path)


def test_validate_media_streams_hash_in_one_mib_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "large.bmp"
    Image.new("RGB", (800, 600), color=(20, 40, 60)).save(image_path)
    payload = image_path.read_bytes()
    requested_sizes: list[int] = []

    class TrackingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested_sizes.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: TrackingReader(payload))

    integrity = validate_media(image_path)

    assert integrity.sha256 == hashlib.sha256(payload).hexdigest()
    assert requested_sizes == [HASH_CHUNK_SIZE, HASH_CHUNK_SIZE, HASH_CHUNK_SIZE]


@pytest.mark.parametrize(
    "bbox",
    [
        (0.1, 0.2, 0.3),
        ("0.1", 0.2, 0.3, 0.4),
        (True, 0.2, 0.3, 0.4),
        (-0.1, 0.2, 0.3, 0.4),
        (0.8, 0.2, 0.3, 0.4),
        (0.1, 0.8, 0.3, 0.4),
        (float("nan"), 0.2, 0.3, 0.4),
        (float("inf"), 0.2, 0.3, 0.4),
    ],
)
def test_validate_annotations_rejects_malformed_bounding_boxes(
    tmp_path: Path,
    bbox: object,
) -> None:
    terminal = _terminal(tmp_path, bbox=bbox)

    with pytest.raises(
        IntegrityError,
        match=r"data/a\.png: bboxes_body\.detections\[0\]\.bounding_box",
    ):
        validate_annotations(terminal)


@pytest.mark.parametrize("mask", [None, "", "  ", (), [], {}])
def test_validate_annotations_requires_nonempty_serialized_masks(
    tmp_path: Path,
    mask: object,
) -> None:
    terminal = _terminal(tmp_path, mask=mask)

    with pytest.raises(
        IntegrityError,
        match=r"data/a\.png: segmentations_body\.detections\[0\]\.mask",
    ):
        validate_annotations(terminal)


def test_validate_annotations_accepts_normalized_boxes_and_mask(tmp_path: Path) -> None:
    validate_annotations(_terminal(tmp_path))


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"detections": "not-a-sequence"},
        {"detections": []},
        {"detections": [42]},
    ],
)
def test_validate_annotations_requires_detections_containers(
    tmp_path: Path,
    invalid: object,
) -> None:
    terminal = _terminal(tmp_path)
    malformed = TerminalRecord(
        source_id=terminal.source_id,
        filepath=terminal.filepath,
        relative_filepath=terminal.relative_filepath,
        jaguar_id=terminal.jaguar_id,
        bboxes_body=cast(FrozenAnnotation, invalid),
        segmentations_body=terminal.segmentations_body,
    )

    with pytest.raises(IntegrityError, match=r"data/a\.png: bboxes_body\.detections"):
        validate_annotations(malformed)


def _enrichment() -> Enrichment:
    return Enrichment(
        status="matched",
        match_method="source_id",
        fields=MappingProxyType({"sighting_id": "S1"}),
    )


def _validated(
    terminal: TerminalRecord,
    sha256: str,
) -> ValidatedRecord:
    return ValidatedRecord(
        terminal=terminal,
        enrichment=_enrichment(),
        integrity=MediaIntegrity(
            sha256=sha256,
            size_bytes=100,
            width=8,
            height=6,
        ),
    )


def test_validated_record_aggregates_immutable_pipeline_records(tmp_path: Path) -> None:
    image_path = tmp_path / "a.png"
    Image.new("RGB", (8, 6)).save(image_path)

    record = validate_record(
        _terminal(tmp_path, filepath=image_path),
        _enrichment(),
    )

    assert record.terminal.jaguar_id == "F11"
    assert record.enrichment.fields["sighting_id"] == "S1"
    assert (record.integrity.width, record.integrity.height) == (8, 6)
    with pytest.raises(FrozenInstanceError):
        record.integrity = MediaIntegrity("other", 1, 1, 1)  # type: ignore[misc]


def test_validated_record_resolves_only_verified_string_identity_from_lineage(
    tmp_path: Path,
) -> None:
    terminal = _terminal(tmp_path, jaguar_id=None)
    integrity = MediaIntegrity("a" * 64, 100, 8, 6)

    matched = ValidatedRecord(
        terminal=terminal,
        enrichment=Enrichment(
            status="matched",
            match_method="unique_filename",
            fields=MappingProxyType({"jaguar_id": "F11"}),
        ),
        integrity=integrity,
    )
    missing = ValidatedRecord(
        terminal=terminal,
        enrichment=Enrichment(
            status="missing",
            match_method=None,
            fields=MappingProxyType({"jaguar_id": "F11"}),
        ),
        integrity=integrity,
    )
    invalid = ValidatedRecord(
        terminal=terminal,
        enrichment=Enrichment(
            status="matched",
            match_method="unique_filename",
            fields=MappingProxyType({"jaguar_id": 11}),
        ),
        integrity=integrity,
    )

    assert matched.resolved_jaguar_id == "F11"
    assert missing.resolved_jaguar_id is None
    assert invalid.resolved_jaguar_id is None


def test_validated_record_preserves_terminal_identity_over_lineage(
    tmp_path: Path,
) -> None:
    record = ValidatedRecord(
        terminal=_terminal(tmp_path, jaguar_id="F11"),
        enrichment=Enrichment(
            status="matched",
            match_method="source_id",
            fields=MappingProxyType({"jaguar_id": "F11"}),
        ),
        integrity=MediaIntegrity("a" * 64, 100, 8, 6),
    )

    assert record.resolved_jaguar_id == "F11"


def test_validate_unique_records_aggregates_duplicate_paths_and_hashes(
    tmp_path: Path,
) -> None:
    first = _validated(
        _terminal(tmp_path, filepath=tmp_path / "folder/../a.png"),
        "same-hash",
    )
    second = _validated(
        _terminal(
            tmp_path,
            filepath=tmp_path / "a.png",
            relative_filepath="data/second.png",
            source_id="source-b",
        ),
        "same-hash",
    )

    with pytest.raises(IntegrityError) as caught:
        validate_unique_records([first, second])

    assert "duplicate canonical path" in str(caught.value)
    assert "duplicate SHA-256" in str(caught.value)
    assert "data/a.png" in str(caught.value)
    assert "data/second.png" in str(caught.value)


def test_validate_records_validates_a_batch_and_expected_count(tmp_path: Path) -> None:
    image_path = tmp_path / "a.png"
    Image.new("RGB", (8, 6)).save(image_path)
    pair = (_terminal(tmp_path, filepath=image_path), _enrichment())

    records = validate_records([pair], expected_count=1)

    assert len(records) == 1
    with pytest.raises(IntegrityError, match="expected 2 records, found 1"):
        validate_records([pair], expected_count=2)


def test_validate_records_aggregates_annotation_and_media_errors_per_record(
    tmp_path: Path,
) -> None:
    terminal = _terminal(
        tmp_path,
        bbox=("bad", 0.2, 0.3, 0.4),
        filepath=tmp_path / "missing.png",
        relative_filepath="data/broken.png",
    )

    with pytest.raises(IntegrityError) as caught:
        validate_records([(terminal, _enrichment())])

    message = str(caught.value)
    assert "data/broken.png: bboxes_body.detections[0].bounding_box" in message
    assert f"media is missing or unreadable: {terminal.filepath}" in message


def test_validate_records_reports_duplicates_alongside_annotation_failures(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(first_path)
    second_path.write_bytes(first_path.read_bytes())
    first = _terminal(tmp_path, filepath=first_path)
    second = _terminal(
        tmp_path,
        bbox=(float("nan"), 0.2, 0.3, 0.4),
        filepath=second_path,
        relative_filepath="data/second.png",
        source_id="source-b",
    )

    with pytest.raises(IntegrityError) as caught:
        validate_records([(first, _enrichment()), (second, _enrichment())])

    message = str(caught.value)
    assert "data/second.png: bboxes_body.detections[0].bounding_box" in message
    assert "duplicate SHA-256" in message
    assert "data/a.png, data/second.png" in message


def test_validate_records_detects_duplicate_paths_when_media_is_missing(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.png"
    first = _terminal(tmp_path, filepath=missing_path)
    second = _terminal(
        tmp_path,
        filepath=missing_path,
        relative_filepath="data/second.png",
        source_id="source-b",
    )

    with pytest.raises(IntegrityError) as caught:
        validate_records([(first, _enrichment()), (second, _enrichment())])

    message = str(caught.value)
    assert message.count("media is missing or unreadable") == 2
    assert message.count("duplicate canonical path") == 1
    assert "data/a.png, data/second.png" in message


def test_validate_records_aggregates_symlink_loop_resolution_and_media_errors(
    tmp_path: Path,
) -> None:
    loop_path = tmp_path / "loop.png"
    loop_path.symlink_to(loop_path)
    missing_path = tmp_path / "missing.png"
    loop = _terminal(
        tmp_path,
        filepath=loop_path,
        relative_filepath="data/loop.png",
    )
    missing = _terminal(
        tmp_path,
        filepath=missing_path,
        relative_filepath="data/missing.png",
        source_id="source-b",
    )

    with pytest.raises(IntegrityError) as caught:
        validate_records([(loop, _enrichment()), (missing, _enrichment())])

    message = str(caught.value)
    assert "data/loop.png: could not resolve canonical path" in message
    assert f"media is missing or unreadable: {loop_path}" in message
    assert f"media is missing or unreadable: {missing_path}" in message


def test_validate_records_aggregates_decompression_bomb_and_missing_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "large.png"
    Image.new("RGB", (8, 6)).save(image_path)
    missing_path = tmp_path / "missing.png"
    image = _terminal(
        tmp_path,
        filepath=image_path,
        relative_filepath="data/large.png",
    )
    missing = _terminal(
        tmp_path,
        filepath=missing_path,
        relative_filepath="data/missing.png",
        source_id="source-b",
    )
    original_limit = Image.MAX_IMAGE_PIXELS

    with monkeypatch.context() as scoped:
        scoped.setattr(Image, "MAX_IMAGE_PIXELS", 1)
        with pytest.raises(IntegrityError) as caught:
            validate_records([(image, _enrichment()), (missing, _enrichment())])

    message = str(caught.value)
    assert f"media is unreadable: {image_path}" in message
    assert "exceeds limit" in message
    assert f"media is missing or unreadable: {missing_path}" in message
    assert original_limit == Image.MAX_IMAGE_PIXELS


def test_validate_mounts_uses_injected_mount_predicate() -> None:
    required = [Path("/Volumes/Extreme SSD"), Path("/Volumes/CameraTrapPython")]
    inspected: list[Path] = []

    def is_mount(path: Path) -> bool:
        inspected.append(path)
        return path.name == "CameraTrapPython"

    with pytest.raises(StorageSafetyError, match=r"/Volumes/Extreme SSD"):
        validate_mounts(required, is_mount=is_mount)

    assert inspected == required


def test_validate_mounts_aggregates_all_missing_mounts() -> None:
    required = [Path("/first"), Path("/second")]

    with pytest.raises(StorageSafetyError) as caught:
        validate_mounts(required, is_mount=lambda _path: False)

    assert "/first" in str(caught.value)
    assert "/second" in str(caught.value)


def test_storage_paths_accept_strict_descendants_of_approved_root(
    tmp_path: Path,
) -> None:
    approved_root = tmp_path / "fiftyone"
    approved_root.mkdir()
    paths = (
        approved_root / "var/lib/mongo",
        approved_root / "reports",
        approved_root / "datasets",
    )

    validate_storage_paths(paths, approved_root)


@pytest.mark.parametrize("unsafe_kind", ["outside", "root-equal"])
def test_storage_paths_reject_outside_and_root_equal_paths(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    approved_root = tmp_path / "fiftyone"
    approved_root.mkdir()
    unsafe_path = tmp_path / "outside" if unsafe_kind == "outside" else approved_root

    with pytest.raises(StorageSafetyError, match="outside|strict descendant"):
        validate_storage_paths(unsafe_path, approved_root)


def test_storage_paths_reject_symlink_escape(tmp_path: Path) -> None:
    approved_root = tmp_path / "fiftyone"
    outside = tmp_path / "outside"
    approved_root.mkdir()
    outside.mkdir()
    (approved_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageSafetyError, match="outside"):
        validate_storage_paths(approved_root / "escape/database", approved_root)
