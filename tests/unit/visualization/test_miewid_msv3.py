import numpy as np
import pytest

from jaguars.visualization.miewid_msv3 import (
    EMBEDDING_DIM,
    MISSING_SEGMENTATION_FILENAMES,
    assert_matching_embedding_checksums,
    assert_remote_head,
    embedding_checksum,
    plan_existing_embedding,
    validate_embedding,
    validate_population,
    _ensure_schema,
)


def test_validate_embedding_accepts_raw_float32_vector() -> None:
    vector = np.arange(EMBEDDING_DIM, dtype=np.float32)
    validated = validate_embedding(vector)
    assert validated.dtype == np.float32
    assert validated.shape == (EMBEDDING_DIM,)


@pytest.mark.parametrize(
    "vector",
    [
        np.zeros(EMBEDDING_DIM - 1, dtype=np.float32),
        np.zeros(EMBEDDING_DIM, dtype=np.float64),
        np.full(EMBEDDING_DIM, np.nan, dtype=np.float32),
    ],
)
def test_validate_embedding_rejects_wrong_contract(vector: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_embedding(vector)


def test_embedding_checksum_is_over_canonical_float32_bytes() -> None:
    vector = np.arange(EMBEDDING_DIM, dtype=np.float32)
    assert embedding_checksum(vector) == embedding_checksum(vector.copy())


def test_existing_valid_embedding_is_skipped_only_for_same_revision() -> None:
    vector = np.ones(EMBEDDING_DIM, dtype=np.float32)
    checksum = embedding_checksum(vector)
    assert plan_existing_embedding(vector, checksum, "abc", expected_revision="abc", overwrite=False) == "skip"
    with pytest.raises(RuntimeError, match="overwrite-embeddings"):
        plan_existing_embedding(vector, checksum, "old", expected_revision="abc", overwrite=False)
    assert plan_existing_embedding(vector, checksum, "old", expected_revision="abc", overwrite=True) == "compute"


def test_population_requires_exactly_four_agreed_exceptions() -> None:
    valid = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    records = [(f"sample-{index}.jpg", valid, "complete") for index in range(1318)]
    records.extend((name, None, "missing_segmentation") for name in MISSING_SEGMENTATION_FILENAMES)
    validate_population(records)
    records[-1] = (records[-1][0], valid, "complete")
    with pytest.raises(ValueError, match="population contract"):
        validate_population(records)


def test_remote_head_guard_rejects_concurrent_update() -> None:
    with pytest.raises(RuntimeError, match="remote head changed"):
        assert_remote_head("old", "new")
    assert_remote_head("same", "same")


def test_round_trip_checksums_must_match_by_media_hash() -> None:
    assert_matching_embedding_checksums({"media": "embedding"}, {"media": "embedding"})
    with pytest.raises(ValueError, match="checksums differ"):
        assert_matching_embedding_checksums({"media": "embedding"}, {"media": "changed"})


def test_inference_schema_is_created_before_sample_writes() -> None:
    class Dataset:
        def __init__(self) -> None:
            self.added: list[str] = []

        def get_field_schema(self) -> dict[str, object]:
            return {}

        def add_sample_field(self, name: str, _field: object) -> None:
            self.added.append(name)

    dataset = Dataset()
    _ensure_schema(dataset)
    assert dataset.added == [
        "miewid_msv3_embedding",
        "miewid_msv3_embedding_status",
        "miewid_msv3_embedding_sha256",
        "miewid_msv3_model_revision",
    ]
