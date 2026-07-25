import json
from pathlib import Path

import pytest


def _detections(*, mask: bool = False) -> dict[str, object]:
    detection: dict[str, object] = {
        "label": "jaguar",
        "bounding_box": [0.1, 0.2, 0.3, 0.4],
    }
    if mask:
        detection["mask"] = [[0, 1], [1, 0]]
    return {"detections": [detection]}


@pytest.fixture
def terminal_export(tmp_path: Path) -> Path:
    export_dir = tmp_path / "terminal-export"
    data_dir = export_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "a.jpg").write_bytes(b"jpeg-a")
    (data_dir / "b.jpg").write_bytes(b"jpeg-b")

    samples = [
        {
            "_id": {"$oid": "source-b"},
            "_dataset_id": {"$oid": "dataset"},
            "_rand": 0.9,
            "filepath": "data/b.jpg",
            "jaguar_id": "M03",
            "bboxes_body": _detections(),
            "segmentations_body": _detections(mask=True),
            "tags": ["transient"],
        },
        {
            "_id": {"$oid": "source-a"},
            "_dataset_id": {"$oid": "dataset"},
            "_rand": 0.1,
            "filepath": "data/a.jpg",
            "jaguar_id": "F11",
            "bboxes_body": _detections(),
            "segmentations_body": _detections(mask=True),
            "created_at": {"$date": "2026-07-25T00:00:00Z"},
        },
    ]
    (export_dir / "samples.json").write_text(
        json.dumps({"samples": samples}),
        encoding="utf-8",
    )
    return export_dir
