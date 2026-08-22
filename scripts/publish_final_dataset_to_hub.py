"""Publish and verify the private final-curated FiftyOne Hub distribution."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from jaguars.visualization.final_dataset import (
    DEFAULT_DATASET_NAME,
    configure_fiftyone_environment,
    default_runtime_paths,
    validate_runtime_paths,
)
from jaguars.visualization.miewid_msv3 import (
    CHECKSUM_FIELD,
    assert_matching_embedding_checksums,
    assert_remote_head,
    validate_dataset as validate_miewid_dataset,
)

HF_USERNAME = "andandandand"
REPO_NAME = "jaguar-camera-trap-final-curated-v1"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
PREVIEW_PATH = Path("artifacts/huggingface/final-curated-v1-preview.png")
VERIFICATION_DATASET_NAME = f"{DEFAULT_DATASET_NAME}_hub_verification"
EXPECTED_COUNT = 1322
EXPECTED_IDENTIFIED = 1108
EXPECTED_REVIEW = 4
EXPECTED_ANNOTATED = 1318
EXPECTED_REVIEW_FILENAMES = {
    "000001-143.jpg",
    "000002-144.jpg",
    "000005-126.jpg",
    "000010-18.jpg",
}
TAGS = [
    "fiftyone",
    "computer-vision",
    "image",
    "object-detection",
    "image-segmentation",
    "wildlife",
    "camera-trap",
    "jaguar",
    "animal-re-identification",
]
SHORT_DESCRIPTION = "Private final-curated FiftyOne snapshot of Jaguar ID Project camera-trap imagery."
DATASET_DESCRIPTION = (
    "Private FiftyOne distribution of 1,322 final curated jaguar camera-trap artifacts. "
    "It contains 1,108 identified and 214 unidentified samples across 59 known identities. "
    "Body bounding boxes and segmentation masks are available for 1,318 samples; four samples "
    "are retained without annotations and tagged for review. Train, validation, and test splits "
    "are not assigned, and capture time/location metadata is not linked because lineage could "
    "not be resolved reliably. Each of the 1,318 segmented foreground cutouts includes a raw "
    "2,152-dimensional float32 embedding from conservationxlabs/miewid-msv3; the four annotation-review "
    "samples explicitly have no embedding. Exact model-revision and preprocessing provenance is stored "
    "in the FiftyOne dataset metadata.\n\n"
    "Images are the property of the [Jaguar ID Project](https://www.jaguaridproject.com/). "
    "Access to this private repository does not grant permission to publish or redistribute the images."
)


def _load_local_dataset() -> Any:
    paths = validate_runtime_paths(default_runtime_paths())
    configure_fiftyone_environment(paths)

    import fiftyone as fo

    configured_dataset_dir = Path(fo.config.default_dataset_dir).resolve(strict=False)
    expected_dataset_dir = paths.dataset_dir.resolve(strict=False)
    if configured_dataset_dir != expected_dataset_dir:
        raise RuntimeError(f"FiftyOne default dataset directory must be {expected_dataset_dir}, found {configured_dataset_dir}")

    return fo.load_dataset(DEFAULT_DATASET_NAME)


def _validate_dataset(dataset: Any) -> None:
    from jaguars.visualization.final_snapshot import SAVED_VIEW_NAMES

    if len(dataset) != EXPECTED_COUNT:
        raise RuntimeError(f"expected {EXPECTED_COUNT} samples, found {len(dataset)}")
    if dataset.count("jaguar_id") != EXPECTED_IDENTIFIED:
        raise RuntimeError(f"expected {EXPECTED_IDENTIFIED} identified samples, found {dataset.count('jaguar_id')}")
    if dataset.count("bboxes_body") != EXPECTED_ANNOTATED:
        raise RuntimeError(f"expected {EXPECTED_ANNOTATED} body boxes, found {dataset.count('bboxes_body')}")
    if dataset.count("segmentations_body") != EXPECTED_ANNOTATED:
        raise RuntimeError(f"expected {EXPECTED_ANNOTATED} masks, found {dataset.count('segmentations_body')}")

    hashes = dataset.values("sha256")
    if len(hashes) != EXPECTED_COUNT or None in hashes or len(set(hashes)) != EXPECTED_COUNT:
        raise RuntimeError("expected one populated, unique sha256 value per sample")

    views = tuple(dataset.list_saved_views())
    if views != SAVED_VIEW_NAMES:
        raise RuntimeError(f"unexpected saved views: {views!r}")

    review = dataset.load_saved_view("Annotation review")
    if len(review) != EXPECTED_REVIEW:
        raise RuntimeError(f"expected {EXPECTED_REVIEW} review samples, found {len(review)}")
    review_filenames = {Path(sample.filepath).name for sample in review}
    if review_filenames != EXPECTED_REVIEW_FILENAMES:
        raise RuntimeError(f"unexpected review filenames: {sorted(review_filenames)!r}")
    for sample in review:
        if sample.tags != ["needs_annotation_review"]:
            raise RuntimeError(f"unexpected review tags for {sample.filepath}: {sample.tags!r}")
        if not sample.review_required or sample.review_status != "pending":
            raise RuntimeError(f"unexpected review state for {sample.filepath}")
        if sample.bboxes_body is not None or sample.segmentations_body is not None:
            raise RuntimeError(f"review sample was unexpectedly annotated: {sample.filepath}")
    validate_miewid_dataset(dataset)


def _validate_media_hashes(dataset: Any) -> None:
    for sample in dataset:
        digest = hashlib.sha256()
        with open(sample.filepath, "rb") as media:
            for chunk in iter(lambda: media.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != sample.sha256:
            raise RuntimeError(f"media hash mismatch: {sample.filepath}")


def _authenticated_api() -> HfApi:
    api = HfApi()
    username = api.whoami().get("name")
    if username != HF_USERNAME:
        raise RuntimeError(f"expected Hugging Face user {HF_USERNAME!r}, authenticated as {username!r}")
    return api


def preflight() -> None:
    """Validate the local snapshot and report the existing remote head."""
    dataset = _load_local_dataset()
    _validate_dataset(dataset)
    _validate_media_hashes(dataset)
    if not PREVIEW_PATH.is_file():
        raise RuntimeError(f"preview does not exist: {PREVIEW_PATH}")

    api = _authenticated_api()
    info = api.repo_info(REPO_ID, repo_type="dataset")
    if not info.private:
        raise RuntimeError(f"repository is not private: {REPO_ID}")
    print(f"preflight passed: {REPO_ID}, head={info.sha}, {len(dataset)} samples, private=True, chunk_size=250")


def upload(expected_head: str) -> None:
    """Upload the validated snapshot when the existing remote head is unchanged."""
    dataset = _load_local_dataset()
    _validate_dataset(dataset)
    _validate_media_hashes(dataset)
    if not PREVIEW_PATH.is_file():
        raise RuntimeError(f"preview does not exist: {PREVIEW_PATH}")

    api = _authenticated_api()
    before = api.repo_info(REPO_ID, repo_type="dataset")
    if not before.private:
        raise RuntimeError(f"repository is not private: {REPO_ID}")
    assert_remote_head(expected_head, before.sha)

    from fiftyone.utils.huggingface import push_to_hub

    push_to_hub(
        dataset,
        REPO_NAME,
        description=SHORT_DESCRIPTION,
        license=None,
        tags=list(TAGS),
        private=True,
        exist_ok=True,
        min_fiftyone_version="1.19.0",
        preview_path=str(PREVIEW_PATH.resolve()),
        chunk_size=250,
        dataset_description=DATASET_DESCRIPTION,
    )

    info = api.repo_info(REPO_ID, repo_type="dataset")
    if not info.private:
        raise RuntimeError(f"uploaded repository is not private: {REPO_ID}")
    if info.sha == expected_head:
        raise RuntimeError("upload did not create a new Hub commit")
    print(f"upload completed: https://huggingface.co/datasets/{REPO_ID}, head={info.sha}")


def verify() -> None:
    """Load the private Hub distribution and verify it against the source."""
    dataset = _load_local_dataset()
    expected_hashes = set(dataset.values("sha256"))
    expected_embedding_checksums = dict(zip(dataset.values("sha256"), dataset.values(CHECKSUM_FIELD), strict=True))
    api = _authenticated_api()
    info = api.repo_info(REPO_ID, repo_type="dataset")
    if not info.private:
        raise RuntimeError(f"repository is not private: {REPO_ID}")

    import fiftyone as fo
    from fiftyone.utils.huggingface import load_from_hub

    if fo.dataset_exists(VERIFICATION_DATASET_NAME):
        raise RuntimeError(f"verification dataset already exists: {VERIFICATION_DATASET_NAME}")

    verification = load_from_hub(
        REPO_ID,
        persistent=False,
        name=VERIFICATION_DATASET_NAME,
        overwrite=False,
    )
    _validate_dataset(verification)
    _validate_media_hashes(verification)
    if set(verification.values("sha256")) != expected_hashes:
        raise RuntimeError("Hub distribution hashes differ from the final curated snapshot")
    actual_embedding_checksums = dict(zip(verification.values("sha256"), verification.values(CHECKSUM_FIELD), strict=True))
    assert_matching_embedding_checksums(expected_embedding_checksums, actual_embedding_checksums)

    fo.delete_dataset(VERIFICATION_DATASET_NAME)
    print(f"round-trip verification passed: {REPO_ID}, {EXPECTED_COUNT} samples")


def main() -> None:
    """Run the selected publication action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "upload", "verify"))
    parser.add_argument("--expected-head", help="Remote dataset commit observed during preflight; required for upload")
    args = parser.parse_args()
    if args.action == "upload":
        if not args.expected_head:
            parser.error("--expected-head is required for upload")
        upload(args.expected_head)
    else:
        {"preflight": preflight, "verify": verify}[args.action]()


if __name__ == "__main__":
    main()
