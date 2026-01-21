"""Ingestion pipeline orchestration.

This module wires together ingestion loaders (CSV, PPTX, etc.) via function
arguments (no config classes).

It is designed to be callable as:
- Python function from notebooks
- CLI module: `python -m src.ingestion.pipeline \
    --input_dir "data/raw/17_11_2025" \
    --pptx "data/raw/17_11_2025/CAMERA TRAP ID GUIDE UPDATED by Oscar2025.pptx"`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import fiftyone as fo

from jaguars.common.config import JID_MASTER_DATASET
from jaguars.common.logging_utils import setup_logger
from jaguars.ingestion.loaders.csv_loader import ingest_csv_labels
from jaguars.ingestion.loaders.pptx_loader import ingest_pptx_slides

logger = setup_logger("ingestion.pipeline")


def run_data_loaders(
    *,
    input_dir: Path | None = None,
    labels_csv: Path | None = None,
    pptx_path: Path | None = None,
    dataset_name: str = JID_MASTER_DATASET,
    pptx_media_dir: Path | None = None,
    pptx_detections_field: str = "pptx_detections",
    auto_match_missing: bool = True,
    match_threshold: float = 0.95,
    suggest_threshold: float = 0.80,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Runs data loaders to ingest raw data into FiftyOne datasets."""
    results: dict[str, Any] = {}

    if overwrite and fo.dataset_exists(dataset_name):
        logger.info("Deleting existing dataset: %s", dataset_name)
        fo.delete_dataset(dataset_name)

    if input_dir is not None:
        logger.info("Running CSV loader: %s (labels_csv=%s, dataset=%s)", input_dir, labels_csv, dataset_name)
        csv_kwargs: dict[str, Any] = {
            "input_dir": Path(input_dir),
            "dataset_name": dataset_name,
            "auto_match_missing": auto_match_missing,
            "match_threshold": match_threshold,
            "suggest_threshold": suggest_threshold,
        }
        if labels_csv is not None:
            csv_kwargs["input_csv"] = Path(labels_csv)
        ds = ingest_csv_labels(**csv_kwargs)

        # Short assertions to ensure expected state
        assert isinstance(ds, fo.Dataset), "CSV loader did not return a FiftyOne dataset"
        assert ds.group_field is not None, "Expected grouped dataset with 'image' and 'video' slices"
        try:
            results["images"] = ds.select_group_slices("image")
            results["videos"] = ds.select_group_slices("video")
        except Exception as e:
            raise AssertionError("Expected accessible 'image' and 'video' slices") from e

        results["dataset"] = ds
    else:
        logger.warning("No labels_csv provided; skipping CSV ingestion")

    if pptx_path is not None:
        logger.info("Running PPTX loader: %s (dataset=%s)", pptx_path, dataset_name)
        ds = ingest_pptx_slides(
            pptx_path=Path(pptx_path),
            dataset_name=dataset_name,
            media_dir=pptx_media_dir,
            detections_field=pptx_detections_field,
        )

        # Short assertions
        assert isinstance(ds, fo.Dataset), "PPTX loader did not return a FiftyOne dataset"
        assert ds.name == dataset_name, "PPTX loader ingested into unexpected dataset"

        # Refresh images slice after PPTX additions
        results.setdefault("dataset", ds)
        results["images"] = results["dataset"].select_group_slices("image")
    else:
        logger.warning("No pptx_path provided; skipping PPTX ingestion")

    return results


def run_processing_pipeline() -> dict[str, Any]:
    """Runs processing steps on the ingested dataset.

    Returns:
        Dict with processing results.
    """
    results: dict[str, Any] = {}

    # 1. sample
    # 2. compute metadata
    # 3. segmentation
    # 4. add embeddings
    # 5. split
    # 6. deduplicate

    return results


def run_ingestion_pipeline(
    *,
    input_dir: Path | None = None,
    labels_csv: Path | None = None,
    pptx_path: Path | None = None,
    dataset_name: str = JID_MASTER_DATASET,
    export_dir: Path | None = Path("data/intermediate/v1/fo_jaguars/ingested"),
    pptx_media_dir: Path | None = None,
    pptx_detections_field: str = "pptx_detections",
    auto_match_missing: bool = True,
    match_threshold: float = 0.95,
    suggest_threshold: float = 0.80,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Runs ingestion from the specified sources.

    Args:
        input_dir: Raw data root (required for CSV ingestion).
        labels_csv: Raw labels CSV path (absolute or relative to input_dir).
        pptx_path: PPTX file to ingest.
        dataset_name: FiftyOne dataset for images.
        export_dir: Directory to export the ingested FiftyOne dataset.
        pptx_media_dir: Optional directory to store PPTX-derived images.
        pptx_detections_field: Field name to store PPTX crop boxes as `fo.Detections`.
        auto_match_missing: Whether to auto-match missing samples during CSV ingestion.
        match_threshold: Similarity threshold for auto-matching samples during CSV ingestion.
        suggest_threshold: Similarity threshold for suggesting matches during CSV ingestion.
        overwrite: Whether to overwrite existing datasets.

    Returns:
        Dict with keys: "images", "videos".
    """
    results: dict[str, Any] = {}

    results["loaders"] = run_data_loaders(
        input_dir=input_dir,
        labels_csv=labels_csv,
        pptx_path=pptx_path,
        dataset_name=dataset_name,
        pptx_media_dir=pptx_media_dir,
        pptx_detections_field=pptx_detections_field,
        auto_match_missing=auto_match_missing,
        match_threshold=match_threshold,
        suggest_threshold=suggest_threshold,
        overwrite=overwrite,
    )

    if "dataset" in results["loaders"]:
        results["dataset"] = results["loaders"]["dataset"]
    elif dataset_name in fo.list_datasets():
        results["dataset"] = fo.load_dataset(dataset_name)
    else:
        raise ValueError("No dataset was created during ingestion and no existing dataset found with the name: " + dataset_name)

    results["processing"] = run_processing_pipeline()

    if export_dir is not None:
        results["dataset"].export(
            export_dir=export_dir.as_posix(),
            dataset_type=fo.types.FiftyOneDataset,
            overwrite=True,
        )
    else:
        logger.warning("No export_dir provided; skipping dataset export")

    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest raw sources into FiftyOne datasets")
    p.add_argument("--input_dir", type=str, required=True, help="Raw data directory")
    p.add_argument("--labels_csv", type=str, default=None, help="Raw labels CSV path (absolute or relative to input_dir)")
    p.add_argument("--pptx", type=str, required=True, help="PPTX file path")

    p.add_argument("--dataset", type=str, default=JID_MASTER_DATASET)

    p.add_argument("--pptx_media_dir", type=str, default=None)
    p.add_argument("--export_dir", type=Path, default=Path("data/intermediate/v1/fo_jaguars/ingested"))
    p.add_argument("--pptx_detections_field", type=str, default="pptx_detections")

    p.add_argument("--no_auto_match", action="store_true")
    p.add_argument("--match_threshold", type=float, default=0.95)
    p.add_argument("--suggest_threshold", type=float, default=0.80)

    p.add_argument("--overwrite", action="store_true", default=False)

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    pptx_path = Path(args.pptx) if args.pptx else None
    input_dir = Path(args.input_dir) if args.input_dir else None
    labels_csv = Path(args.labels_csv) if args.labels_csv else None
    pptx_media_dir = Path(args.pptx_media_dir) if args.pptx_media_dir else None
    export_dir = Path(args.export_dir) if args.export_dir else None

    run_ingestion_pipeline(
        input_dir=input_dir,
        labels_csv=labels_csv,
        pptx_path=pptx_path,
        dataset_name=args.dataset,
        export_dir=export_dir,
        pptx_media_dir=pptx_media_dir,
        pptx_detections_field=args.pptx_detections_field,
        auto_match_missing=not args.no_auto_match,
        match_threshold=float(args.match_threshold),
        suggest_threshold=float(args.suggest_threshold),
        overwrite=args.overwrite,
    )

    logger.info("Ingestion pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
