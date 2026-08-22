"""Guarded MiewID MSv3 enrichment for the final curated cutout dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from jaguars.visualization.final_dataset import (
    DEFAULT_DATASET_NAME,
    configure_fiftyone_environment,
    default_runtime_paths,
    validate_runtime_paths,
)

MODEL_ID = "conservationxlabs/miewid-msv3"
EMBEDDING_FIELD = "miewid_msv3_embedding"
STATUS_FIELD = "miewid_msv3_embedding_status"
CHECKSUM_FIELD = "miewid_msv3_embedding_sha256"
REVISION_FIELD = "miewid_msv3_model_revision"
EMBEDDING_DIM = 2152
EXPECTED_EMBEDDINGS = 1318
EXPECTED_SAMPLES = 1322
MISSING_SEGMENTATION_FILENAMES = frozenset(
    {"000001-143.jpg", "000002-144.jpg", "000005-126.jpg", "000010-18.jpg"}
)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def assert_remote_head(expected: str, actual: str) -> None:
    """Abort publication when the remote changed after preflight."""
    if actual != expected:
        raise RuntimeError(f"remote head changed: expected {expected}, found {actual}")


def assert_matching_embedding_checksums(expected: dict[str, str], actual: dict[str, str]) -> None:
    """Require exact per-media embedding equality after a Hub round trip."""
    if actual != expected:
        raise ValueError("round-trip embedding checksums differ")


def validate_embedding(value: Any) -> np.ndarray:
    """Return an embedding only when it exactly satisfies the storage contract."""
    vector = np.asarray(value)
    if vector.dtype != np.float32:
        raise ValueError(f"embedding dtype must be float32, found {vector.dtype}")
    if vector.shape != (EMBEDDING_DIM,):
        raise ValueError(f"embedding shape must be ({EMBEDDING_DIM},), found {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("embedding contains non-finite values")
    return np.ascontiguousarray(vector)


def embedding_checksum(value: Any) -> str:
    """Hash the canonical raw float32 representation of an embedding."""
    return hashlib.sha256(validate_embedding(value).tobytes(order="C")).hexdigest()


def plan_existing_embedding(
    value: Any,
    checksum: str | None,
    revision: str | None,
    *,
    expected_revision: str,
    overwrite: bool,
) -> str:
    """Choose whether a populated sample is safely reusable."""
    try:
        valid = embedding_checksum(value) == checksum
    except (TypeError, ValueError):
        valid = False
    if valid and revision == expected_revision:
        return "skip"
    if value is not None and not overwrite:
        raise RuntimeError("conflicting MiewID embedding; rerun with --overwrite-embeddings")
    return "compute"


def validate_population(records: Iterable[tuple[str, Any, str | None]]) -> None:
    """Validate the complete publishable sample population."""
    rows = list(records)
    completed = 0
    missing: set[str] = set()
    for filename, value, status in rows:
        if value is None:
            if status == "missing_segmentation":
                missing.add(filename)
            continue
        validate_embedding(value)
        if status != "complete":
            raise ValueError(f"population contract failed: {filename} has vector with status {status!r}")
        completed += 1
    if len(rows) != EXPECTED_SAMPLES or completed != EXPECTED_EMBEDDINGS or missing != MISSING_SEGMENTATION_FILENAMES:
        raise ValueError(
            f"population contract failed: samples={len(rows)}, complete={completed}, missing={sorted(missing)}"
        )


def _load_dataset() -> Any:
    paths = validate_runtime_paths(default_runtime_paths())
    configure_fiftyone_environment(paths)
    import fiftyone as fo

    return fo.load_dataset(DEFAULT_DATASET_NAME)


def resolve_model_revision() -> str:
    """Resolve the model's floating ref to an immutable commit SHA."""
    from huggingface_hub import HfApi

    revision = HfApi().model_info(MODEL_ID).sha
    if not revision or len(revision) != 40:
        raise RuntimeError(f"could not resolve immutable model revision: {revision!r}")
    return revision


def create_checkpoint(dataset: Any) -> Path:
    """Export a metadata-only recovery checkpoint without copying media."""
    import fiftyone as fo

    paths = default_runtime_paths()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = paths.report_dir / "miewid_msv3_checkpoints" / stamp
    if target.exists():
        raise RuntimeError(f"checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset.export(export_dir=str(target), dataset_type=fo.types.FiftyOneDataset, export_media=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset.name,
        "dataset_id": str(dataset._doc.id),
        "sample_count": len(dataset),
        "media_copied": False,
        "saved_views": list(dataset.list_saved_views()),
    }
    (target / "miewid_checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def _preprocess() -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [transforms.Resize((440, 440)), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]
    )


def _load_model(revision: str, device: Any) -> Any:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=True)
    model.eval()
    return model.to(device)


def _forward(model: Any, tensors: list[Any], device: Any) -> np.ndarray:
    import torch

    with torch.no_grad():
        output = model(torch.stack(tensors).to(device))
    return output.detach().to("cpu", dtype=torch.float32).numpy()


def _dataset_records(dataset: Any) -> list[tuple[str, Any, str | None]]:
    return [
        (Path(sample.filepath).name, sample.get_field(EMBEDDING_FIELD), sample.get_field(STATUS_FIELD))
        for sample in dataset
    ]


def _ensure_schema(dataset: Any) -> None:
    """Create the four enrichment fields before any resumable sample writes."""
    import fiftyone as fo

    schema = dataset.get_field_schema()
    field_types = (
        (EMBEDDING_FIELD, fo.VectorField),
        (STATUS_FIELD, fo.StringField),
        (CHECKSUM_FIELD, fo.StringField),
        (REVISION_FIELD, fo.StringField),
    )
    for name, field_type in field_types:
        if name not in schema:
            dataset.add_sample_field(name, field_type)


def validate_dataset(dataset: Any) -> None:
    """Validate vectors, statuses, checksums, revisions, and top-level provenance."""
    validate_population(_dataset_records(dataset))
    revisions: set[str] = set()
    for sample in dataset:
        vector = sample.get_field(EMBEDDING_FIELD)
        if vector is None:
            continue
        if embedding_checksum(vector) != sample.get_field(CHECKSUM_FIELD):
            raise ValueError(f"embedding checksum mismatch: {sample.filepath}")
        revisions.add(sample.get_field(REVISION_FIELD))
    provenance = dataset.info.get("miewid_msv3")
    if not isinstance(provenance, dict) or provenance.get("model_id") != MODEL_ID:
        raise ValueError("missing MiewID MSv3 dataset provenance")
    if revisions != {provenance.get("revision")}:
        raise ValueError(f"mixed or invalid model revisions: {revisions}")


def run_inference(dataset: Any, revision: str, *, batch_size: int = 16, overwrite: bool = False) -> None:
    """Mutate the persistent dataset with resumable MPS-first inference."""
    import torch
    from PIL import Image

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the primary M4 inference path")
    device = torch.device("mps")
    model = _load_model(revision, device)
    preprocess = _preprocess()
    _ensure_schema(dataset)
    pending: list[Any] = []
    for sample in dataset:
        filename = Path(sample.filepath).name
        if filename in MISSING_SEGMENTATION_FILENAMES:
            sample[EMBEDDING_FIELD] = None
            sample[STATUS_FIELD] = "missing_segmentation"
            sample[CHECKSUM_FIELD] = None
            sample[REVISION_FIELD] = revision
            sample.save()
            continue
        action = plan_existing_embedding(
            sample.get_field(EMBEDDING_FIELD),
            sample.get_field(CHECKSUM_FIELD),
            sample.get_field(REVISION_FIELD),
            expected_revision=revision,
            overwrite=overwrite,
        )
        if action == "compute":
            pending.append(sample)

    cpu_model: Any | None = None
    index = 0
    adaptive = max(1, batch_size)
    while index < len(pending):
        batch = pending[index : index + adaptive]
        tensors = [preprocess(Image.open(sample.filepath).convert("RGB")) for sample in batch]
        try:
            outputs = _forward(model, tensors, device)
        except RuntimeError as exc:
            message = str(exc).lower()
            if ("out of memory" in message or "mps" in message) and adaptive > 1:
                adaptive = max(1, adaptive // 2)
                torch.mps.empty_cache()
                continue
            if "not implemented" not in message and "mps" not in message:
                raise
            if cpu_model is None:
                cpu_model = _load_model(revision, torch.device("cpu"))
            outputs = _forward(cpu_model, tensors, torch.device("cpu"))
        for sample, output in zip(batch, outputs, strict=True):
            vector = validate_embedding(np.asarray(output, dtype=np.float32))
            sample[EMBEDDING_FIELD] = vector
            sample[STATUS_FIELD] = "complete"
            sample[CHECKSUM_FIELD] = embedding_checksum(vector)
            sample[REVISION_FIELD] = revision
            sample.save()
        index += len(batch)

    dataset.info["miewid_msv3"] = {
        "model_id": MODEL_ID,
        "revision": revision,
        "input": "published foreground-only JPEG cutout",
        "resize": [440, 440],
        "mean": list(MEAN),
        "std": list(STD),
        "output_dimension": EMBEDDING_DIM,
        "dtype": "float32",
        "l2_normalized": False,
        "primary_device": "mps",
    }
    dataset.save()
    validate_dataset(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "checkpoint", "infer", "validate"))
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite-embeddings", action="store_true")
    args = parser.parse_args()
    dataset = _load_dataset()
    if args.action == "preflight":
        print(json.dumps({"dataset": dataset.name, "samples": len(dataset), "model_revision": resolve_model_revision()}, indent=2))
    elif args.action == "checkpoint":
        print(create_checkpoint(dataset))
    elif args.action == "infer":
        revision = args.revision or resolve_model_revision()
        run_inference(dataset, revision, batch_size=args.batch_size, overwrite=args.overwrite_embeddings)
        print(f"inference complete: revision={revision}")
    else:
        validate_dataset(dataset)
        print(f"validation passed: {EXPECTED_EMBEDDINGS} embeddings")


if __name__ == "__main__":
    main()
