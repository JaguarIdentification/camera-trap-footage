import uuid
from pathlib import Path

import fiftyone as fo
import numpy as np
from PIL import Image
from fiftyone import ViewField as F

from jaguars.ingestion.processing.deduplicate import run_processing


def _write_image(path: Path, color: int) -> None:
    array = np.full((64, 64, 3), color, dtype=np.uint8)
    img = Image.fromarray(array)
    img.save(path)


def test_deduplicate_hash_marks_duplicates(tmp_path: Path) -> None:
    dataset_name = f"test_dedup_{uuid.uuid4().hex}"
    img_a = tmp_path / "img_a.jpg"
    img_b = tmp_path / "img_b.jpg"
    img_c = tmp_path / "img_c.jpg"

    _write_image(img_a, 0)
    _write_image(img_b, 0)
    _write_image(img_c, 255)

    dataset = fo.Dataset(name=dataset_name, persistent=False)
    dataset.add_samples(
        [
            fo.Sample(filepath=str(img_a)),
            fo.Sample(filepath=str(img_b)),
            fo.Sample(filepath=str(img_c)),
        ]
    )

    run_processing(
        dataset_name=dataset_name,
        similarity_threshold=0.85,  # Lower threshold for near duplicates (identical images should be >0.95)
        model_name="resnet50-imagenet-torch",
        batch_size=8,
    )

    dataset = fo.load_dataset(dataset_name)
    dup_view = dataset.match(F("is_duplicate") == True)
    assert len(dup_view) == 1

    dup_sample = dup_view.first()
    assert "duplicate" in dup_sample.tags
    assert dup_sample["duplicate_of"] is not None
    assert dup_sample["duplicate_similarity"] >= 0.85

    fo.delete_dataset(dataset_name)