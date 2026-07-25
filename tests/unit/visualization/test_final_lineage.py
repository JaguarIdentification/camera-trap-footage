import json
from pathlib import Path
from typing import cast

from jaguars.visualization.final_lineage import (
    LineageCandidate,
    LineageIndex,
    load_export_candidates,
    load_lineage_candidates,
    load_manifest_candidates,
)
from jaguars.visualization.final_records import FrozenAnnotation, TerminalRecord


def _terminal(
    tmp_path: Path,
    relative_filepath: str,
    *,
    source_id: str = "terminal-id",
    jaguar_id: str = "F11",
) -> TerminalRecord:
    return TerminalRecord(
        source_id=source_id,
        filepath=tmp_path / relative_filepath,
        relative_filepath=relative_filepath,
        jaguar_id=jaguar_id,
        bboxes_body=cast(
            FrozenAnnotation,
            {"detections": [{"label": "jaguar"}]},
        ),
        segmentations_body=cast(
            FrozenAnnotation,
            {"detections": [{"label": "jaguar", "mask": [[1]]}]},
        ),
    )


def test_preserved_source_id_is_the_highest_precedence_match(tmp_path: Path) -> None:
    index = LineageIndex.from_candidates(
        [
            LineageCandidate(
                source_id="terminal-id",
                export_relative_filepath="data/other.jpg",
                sighting_id="by-id",
            ),
            LineageCandidate(
                source_id="other-id",
                export_relative_filepath="data/a.jpg",
                sighting_id="by-path",
            ),
        ]
    )

    enrichment = index.enrich(_terminal(tmp_path, "data/a.jpg"))

    assert enrichment.status == "matched"
    assert enrichment.match_method == "source_id"
    assert enrichment.fields["sighting_id"] == "by-id"


def test_export_relative_filepath_precedes_filename_fallback(tmp_path: Path) -> None:
    index = LineageIndex.from_candidates(
        [
            LineageCandidate(
                export_relative_filepath=r".\data\folder\..\a.jpg",
                sighting_id="relative-path",
            ),
            LineageCandidate(
                export_relative_filepath="data/elsewhere/a.jpg",
                sighting_id="filename",
            ),
        ]
    )

    enrichment = index.enrich(_terminal(tmp_path, "data/a.jpg", source_id="unmatched"))

    assert enrichment.status == "matched"
    assert enrichment.match_method == "export_relative_filepath"
    assert enrichment.fields["sighting_id"] == "relative-path"


def test_normalized_source_filepath_precedes_export_relative_path(tmp_path: Path) -> None:
    terminal = _terminal(tmp_path, "data/a.jpg", source_id="unmatched")
    equivalent_source_path = f"{tmp_path}/data/folder/../a.jpg".replace("/", "\\")
    index = LineageIndex.from_candidates(
        [
            LineageCandidate(
                normalized_source_filepath=equivalent_source_path,
                export_relative_filepath="data/other.jpg",
                sighting_id="source-path",
            ),
            LineageCandidate(
                export_relative_filepath="data/a.jpg",
                sighting_id="relative-path",
            ),
        ]
    )

    enrichment = index.enrich(terminal)

    assert enrichment.status == "matched"
    assert enrichment.match_method == "normalized_source_filepath"
    assert enrichment.fields["sighting_id"] == "source-path"


def test_exact_filename_matches_only_when_globally_unique(tmp_path: Path) -> None:
    index = LineageIndex.from_candidates(
        [
            LineageCandidate(original_filename="a.jpg", sighting_id="filename"),
            LineageCandidate(original_filename="other.jpg", sighting_id="other"),
        ]
    )

    enrichment = index.enrich(_terminal(tmp_path, "data/a.jpg", source_id="unmatched"))

    assert enrichment.status == "matched"
    assert enrichment.match_method == "unique_filename"
    assert enrichment.fields["sighting_id"] == "filename"


def test_export_path_basename_participates_in_exact_filename_fallback(
    tmp_path: Path,
) -> None:
    index = LineageIndex.from_candidates(
        [
            LineageCandidate(
                export_relative_filepath="other/a.jpg",
                original_filename="raw-video.mp4",
                sighting_id="export-filename",
            )
        ]
    )

    enrichment = index.enrich(_terminal(tmp_path, "terminal/a.jpg", source_id="unmatched"))

    assert enrichment.status == "matched"
    assert enrichment.match_method == "unique_filename"
    assert enrichment.fields["sighting_id"] == "export-filename"


def test_distinct_candidates_are_ambiguous_without_lower_precedence_fallback(
    tmp_path: Path,
) -> None:
    index = LineageIndex.from_candidates(
        [
            LineageCandidate(
                normalized_source_filepath=(tmp_path / "data/a.jpg").as_posix(),
                sighting_id="one",
            ),
            LineageCandidate(
                normalized_source_filepath=(tmp_path / "data/a.jpg").as_posix(),
                sighting_id="two",
            ),
            LineageCandidate(
                export_relative_filepath="data/a.jpg",
                sighting_id="lower-precedence",
            ),
        ]
    )

    enrichment = index.enrich(_terminal(tmp_path, "data/a.jpg", source_id="unmatched"))

    assert enrichment.status == "ambiguous"
    assert enrichment.match_method is None
    assert dict(enrichment.fields) == {}


def test_separately_loaded_value_equal_candidates_remain_ambiguous(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "labels_with_splits.csv"
    manifest.write_text(
        "ORIGINAL FILE NAME,SIGHTING ID\n" "a.jpg,same\n" "a.jpg,same\n",
        encoding="utf-8",
    )
    candidates = load_manifest_candidates(manifest)
    assert candidates[0] == candidates[1]
    index = LineageIndex.from_candidates(candidates)

    enrichment = index.enrich(_terminal(tmp_path, "data/a.jpg", source_id="unmatched"))

    assert enrichment.status == "ambiguous"
    assert enrichment.match_method is None


def test_camera_manifest_normalizes_legacy_headers_and_both_splits(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "labels_with_splits.csv"
    manifest.write_text(
        "CLOSED SET SPLIT,OPEN SET SPLIT,JAGUAR ID,SIGHTING ID,CAMERA TRAP SITE,"
        "LOCATION,CAMERA ID,CAM,CAMERA MODEL,LATITUDE,LONGITUDE,DATE,TIME,DATETIME,"
        "ORIGINAL FILE NAME,FILE PATH\n"
        "train,test,F11,S1,Site 1,North,C07,left,Model X,-3.1,-60.2,2024-01-02,"
        "03:04:05,2024-01-02 03:04:05,IMG_1.JPG,media/IMG_1.JPG\n",
        encoding="utf-8",
    )

    candidate = load_manifest_candidates(manifest)[0]

    assert candidate.closed_set_split == "train"
    assert candidate.open_set_split == "test"
    assert candidate.jaguar_id == "F11"
    assert candidate.sighting_id == "S1"
    assert candidate.site == "Site 1"
    assert candidate.location == "North"
    assert candidate.camera_id == "C07"
    assert candidate.camera_side == "left"
    assert candidate.camera_model == "Model X"
    assert candidate.latitude == -3.1
    assert candidate.longitude == -60.2
    assert candidate.capture_date == "2024-01-02"
    assert candidate.capture_time == "03:04:05"
    assert candidate.capture_datetime == "2024-01-02 03:04:05"
    assert candidate.original_filename == "IMG_1.JPG"
    assert candidate.source_media_path == "media/IMG_1.JPG"
    assert candidate.source_type == "csv"


def test_pptx_manifest_normalizes_its_alternate_headers_and_dataset_path(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "pptx_extracted_labels_with_splits.csv"
    manifest.write_text(
        "closed_split,open_split,Jaguar ID,Sighting ID,Original Site,Original Cam,"
        "Files Name,Dataset Path,File Path,Raw Data Path\n"
        "val,train,M03,PP4,Site 9,right,slide_4_img_1.png,/dataset/extracted,"
        "slide_4_img_1.png,/source/guide.pptx\n",
        encoding="utf-8",
    )

    candidate = load_manifest_candidates(manifest)[0]

    assert candidate.closed_set_split == "val"
    assert candidate.open_set_split == "train"
    assert candidate.site == "Site 9"
    assert candidate.camera_side == "right"
    assert candidate.original_filename == "slide_4_img_1.png"
    assert candidate.normalized_source_filepath == "/dataset/extracted/slide_4_img_1.png"
    assert candidate.source_media_path == "/dataset/extracted/slide_4_img_1.png"
    assert candidate.source_type == "pptx"


def test_manifest_resolves_relative_and_absolute_raw_file_paths(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "labels_with_splits.csv"
    manifest.write_text(
        "JAGUAR ID,ORIGINAL FILE NAME,RAW DATA PATH,DATASET PATH,FILE PATH,"
        "RAW FILE PATH\n"
        "F11,a.jpg,/raw/camera,/dataset/processed,derived/a.jpg,original/a.jpg\n"
        "M03,b.jpg,/raw/camera,/dataset/processed,derived/b.jpg,/archive/b.jpg\n"
        "U02,c.jpg,/raw/camera,,derived/c.jpg,\n",
        encoding="utf-8",
    )

    relative_raw, absolute_raw, raw_plus_file = load_manifest_candidates(manifest)

    assert relative_raw.normalized_source_filepath == "/raw/camera/original/a.jpg"
    assert relative_raw.source_media_path == "/raw/camera/original/a.jpg"
    assert absolute_raw.normalized_source_filepath == "/archive/b.jpg"
    assert absolute_raw.source_media_path == "/archive/b.jpg"
    assert raw_plus_file.normalized_source_filepath == "/raw/camera/derived/c.jpg"
    assert raw_plus_file.source_media_path == "/raw/camera/derived/c.jpg"


def test_export_candidates_normalize_fiftyone_ids_paths_and_approved_fields(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "segmented"
    export_dir.mkdir()
    (export_dir / "samples.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "_id": {"$oid": "preserved-id"},
                        "filepath": r"data\folder\..\a.jpg",
                        "source_filepath": r"/raw\camera\..\a.jpg",
                        "jaguar_id": "F11",
                        "closed_set_split": "train",
                        "open_set_split": "test",
                        "sighting_id": "S1",
                        "site": "Site 1",
                        "location": "North",
                        "camera_id": "C07",
                        "cam": "left",
                        "camera_model": "Model X",
                        "latitude": -3.1,
                        "longitude": -60.2,
                        "date": "2024-01-02",
                        "time": "03:04:05",
                        "datetime": "2024-01-02 03:04:05",
                        "original_file_name": "ORIGINAL.JPG",
                        "source_type": "video_frame",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidate = load_export_candidates(export_dir)[0]

    assert candidate.source_id == "preserved-id"
    assert candidate.normalized_source_filepath == "/raw/a.jpg"
    assert candidate.export_relative_filepath == "data/a.jpg"
    assert candidate.fields == {
        "closed_set_split": "train",
        "open_set_split": "test",
        "sighting_id": "S1",
        "site": "Site 1",
        "location": "North",
        "camera_id": "C07",
        "camera_side": "left",
        "camera_model": "Model X",
        "latitude": -3.1,
        "longitude": -60.2,
        "capture_date": "2024-01-02",
        "capture_time": "03:04:05",
        "capture_datetime": "2024-01-02 03:04:05",
        "original_filename": "ORIGINAL.JPG",
        "source_media_path": "/raw/a.jpg",
        "source_type": "video_frame",
    }


def test_export_preserves_sample_and_source_id_aliases_for_exact_matching(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "ingested"
    export_dir.mkdir()
    (export_dir / "samples.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "_id": {"$oid": "sample-id"},
                        "source": {"$oid": "parent-source-id"},
                        "filepath": "data/upstream.jpg",
                        "sighting_id": "S1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate = load_export_candidates(export_dir)[0]

    enrichment = LineageIndex.from_candidates([candidate]).enrich(
        _terminal(
            tmp_path,
            "data/terminal.jpg",
            source_id="parent-source-id",
        )
    )

    assert enrichment.status == "matched"
    assert enrichment.match_method == "source_id"
    assert enrichment.fields["sighting_id"] == "S1"


def test_export_resolves_actual_lowercase_source_path_fields(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "ingested"
    export_dir.mkdir()
    (export_dir / "samples.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "_id": {"$oid": "raw-plus-file"},
                        "filepath": "data/one.jpg",
                        "raw_data_path": "/raw/camera",
                        "file_path": "one.jpg",
                    },
                    {
                        "_id": {"$oid": "relative-source"},
                        "filepath": "data/two.jpg",
                        "raw_data_path": "/raw/camera",
                        "file_path": "derived/two.jpg",
                        "source_filepath": "original/two.jpg",
                    },
                    {
                        "_id": {"$oid": "absolute-raw-file"},
                        "filepath": "data/three.jpg",
                        "raw_data_path": "/raw/camera",
                        "raw_file_path": "/archive/three.jpg",
                    },
                    {
                        "_id": {"$oid": "absolute-source"},
                        "filepath": "data/four.jpg",
                        "raw_data_path": "/raw/camera",
                        "source_filepath": "/archive/four.jpg",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_export_candidates(export_dir)

    assert [candidate.normalized_source_filepath for candidate in candidates] == [
        "/raw/camera/one.jpg",
        "/raw/camera/original/two.jpg",
        "/archive/three.jpg",
        "/archive/four.jpg",
    ]
    assert [candidate.source_media_path for candidate in candidates] == [
        "/raw/camera/one.jpg",
        "/raw/camera/original/two.jpg",
        "/archive/three.jpg",
        "/archive/four.jpg",
    ]


def test_absolute_export_filepath_is_also_a_normalized_source_path(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "deduplicated"
    export_dir.mkdir()
    source_path = tmp_path / "raw/folder/../a.jpg"
    (export_dir / "samples.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "_id": {"$oid": "sample-id"},
                        "filepath": source_path.as_posix(),
                        "sighting_id": "S1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidate = load_export_candidates(export_dir)[0]

    assert candidate.normalized_source_filepath == (tmp_path / "raw/a.jpg").as_posix()


def test_windows_absolute_export_filepath_is_a_source_not_export_relative(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "deduplicated"
    export_dir.mkdir()
    (export_dir / "samples.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "_id": {"$oid": "windows-sample"},
                        "filepath": r"C:\camera\folder\..\a.jpg",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidate = load_export_candidates(export_dir)[0]

    assert candidate.normalized_source_filepath == "C:/camera/a.jpg"
    assert candidate.source_media_path == "C:/camera/a.jpg"
    assert candidate.export_relative_filepath is None


def test_load_lineage_candidates_reads_all_four_exports_and_both_manifests(
    tmp_path: Path,
) -> None:
    intermediate_dir = tmp_path / "intermediate" / "v1"
    export_root = intermediate_dir / "fo_jaguars"
    export_paths = (
        export_root / "exports/segmented_deduplicated",
        export_root / "exports/segmented",
        export_root / "exports/deduplicated",
        export_root / "ingested",
    )
    for sequence, export_path in enumerate(export_paths, start=1):
        export_path.mkdir(parents=True)
        (export_path / "samples.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "_id": {"$oid": f"export-{sequence}"},
                            "filepath": f"data/export-{sequence}.jpg",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    (intermediate_dir / "labels_with_splits.csv").write_text(
        "SAMPLE ID,ORIGINAL FILE NAME\nmanifest-camera,camera.jpg\n",
        encoding="utf-8",
    )
    (intermediate_dir / "pptx_extracted_labels_with_splits.csv").write_text(
        "SAMPLE ID,Files Name\nmanifest-pptx,pptx.jpg\n",
        encoding="utf-8",
    )

    candidates = load_lineage_candidates(intermediate_dir)

    assert [candidate.source_id for candidate in candidates] == [
        "export-1",
        "export-2",
        "export-3",
        "export-4",
        "manifest-camera",
        "manifest-pptx",
    ]


def test_unique_export_and_manifest_information_merge_when_identity_and_name_agree(
    tmp_path: Path,
) -> None:
    index = LineageIndex.from_candidates(
        [
            LineageCandidate(
                candidate_type="export",
                export_relative_filepath="data/a.jpg",
                jaguar_id="F11",
                original_filename="ORIGINAL.JPG",
                site="Site 1",
                source_type="video_frame",
            ),
            LineageCandidate(
                candidate_type="manifest",
                jaguar_id="F11",
                original_filename="ORIGINAL.JPG",
                closed_set_split="train",
                open_set_split="test",
                source_type="csv",
            ),
        ]
    )

    enrichment = index.enrich(_terminal(tmp_path, "data/a.jpg"))

    assert enrichment.status == "matched"
    assert enrichment.match_method == "export_relative_filepath"
    assert enrichment.fields["site"] == "Site 1"
    assert enrichment.fields["closed_set_split"] == "train"
    assert enrichment.fields["open_set_split"] == "test"
    assert enrichment.fields["source_type"] == "video_frame"


def test_incompatible_manifest_is_not_merged_into_an_exact_export_match(
    tmp_path: Path,
) -> None:
    index = LineageIndex.from_candidates(
        [
            LineageCandidate(
                candidate_type="export",
                export_relative_filepath="data/a.jpg",
                jaguar_id="F11",
                original_filename="ORIGINAL.JPG",
                site="Site 1",
            ),
            LineageCandidate(
                candidate_type="manifest",
                jaguar_id="M03",
                original_filename="ORIGINAL.JPG",
                closed_set_split="train",
            ),
        ]
    )

    enrichment = index.enrich(_terminal(tmp_path, "data/a.jpg"))

    assert enrichment.status == "matched"
    assert enrichment.fields["site"] == "Site 1"
    assert "closed_set_split" not in enrichment.fields


def test_duplicate_value_equal_manifest_rows_are_not_merged_into_unique_export(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "labels_with_splits.csv"
    manifest.write_text(
        "JAGUAR ID,ORIGINAL FILE NAME,CLOSED SET SPLIT\n" "F11,ORIGINAL.JPG,train\n" "F11,ORIGINAL.JPG,train\n",
        encoding="utf-8",
    )
    export = LineageCandidate(
        candidate_type="export",
        export_relative_filepath="data/a.jpg",
        jaguar_id="F11",
        original_filename="ORIGINAL.JPG",
        site="Site 1",
    )
    index = LineageIndex.from_candidates([export, *load_manifest_candidates(manifest)])

    enrichment = index.enrich(_terminal(tmp_path, "data/a.jpg"))

    assert enrichment.status == "matched"
    assert enrichment.match_method == "export_relative_filepath"
    assert enrichment.fields["site"] == "Site 1"
    assert "closed_set_split" not in enrichment.fields


def test_duplicate_exact_filename_candidates_are_ambiguous(tmp_path: Path) -> None:
    index = LineageIndex.from_candidates(
        [
            LineageCandidate(original_filename="same.jpg", sighting_id="one"),
            LineageCandidate(original_filename="same.jpg", sighting_id="two"),
        ]
    )

    enrichment = index.enrich(_terminal(tmp_path, "data/same.jpg", source_id="unmatched"))

    assert enrichment.status == "ambiguous"
    assert enrichment.match_method is None


def test_unmatched_terminal_record_reports_missing(tmp_path: Path) -> None:
    enrichment = LineageIndex.from_candidates([]).enrich(_terminal(tmp_path, "data/a.jpg"))

    assert enrichment.status == "missing"
    assert enrichment.match_method is None
    assert dict(enrichment.fields) == {}


def test_filename_matching_never_infers_numeric_prefixes(tmp_path: Path) -> None:
    index = LineageIndex.from_candidates([LineageCandidate(original_filename="a.jpg", sighting_id="S1")])

    enrichment = index.enrich(_terminal(tmp_path, "data/001_a.jpg", source_id="unmatched"))

    assert enrichment.status == "missing"


def test_enrichment_is_deterministic_across_candidate_input_order(
    tmp_path: Path,
) -> None:
    candidates = [
        LineageCandidate(
            normalized_source_filepath=(tmp_path / "data/a.jpg").as_posix(),
            sighting_id="one",
        ),
        LineageCandidate(
            normalized_source_filepath=(tmp_path / "data/a.jpg").as_posix(),
            sighting_id="two",
        ),
    ]
    terminal = _terminal(tmp_path, "data/a.jpg", source_id="unmatched")

    forward = LineageIndex.from_candidates(candidates).enrich(terminal)
    reverse = LineageIndex.from_candidates(reversed(candidates)).enrich(terminal)

    assert forward == reverse
