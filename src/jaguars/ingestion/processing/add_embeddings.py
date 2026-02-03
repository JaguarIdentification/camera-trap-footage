"""Embedding computation module.

Computes image embeddings using a specified model (default: MegaDescriptor-L-384)
and stores them in a sample field.
"""

import argparse
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import fiftyone as fo
import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm

from jaguars.common.config import JID_MASTER_DATASET
from jaguars.common.logging_utils import setup_logger

MODULE_NAME = "ingestion.processing.add_embeddings"
logger = setup_logger(MODULE_NAME)


def validate_resources(dataset_name: str) -> None:
    """Checks if dataset exists."""
    if dataset_name not in fo.list_datasets():
        raise ValueError(f"Dataset '{dataset_name}' does not exist")


def write_summary(summary_data: dict[str, Any], summary_path: Path) -> None:
    """Generates summary report."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)


def get_device() -> torch.device:
    """Determines the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@torch.no_grad()
def compute_embeddings_batch(
    items: list[tuple[str, list[float] | None | Any, np.ndarray | None]],
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[dict[int, np.ndarray], list[str]]:
    """Compute embeddings for a list of items (filepaths + optional bboxes + optional masks).

    Args:
        items: List of (filepath, bbox, mask) tuples. bbox is [x, y, w, h] (normalized) or None.
               mask may be None, a numpy array, a PIL Image, or a path to an image.
        model: Loaded PyTorch model
        preprocess: Transform pipeline
        device: Torch device
        batch_size: Number of images to process per batch

    Returns:
        Tuple of (Dictionary mapping index in input list to embedding numpy array, List of error messages)
    """
    embeddings_map = {}
    errors = []

    # We iterate by index to handle batches
    for i in tqdm(range(0, len(items), batch_size), desc="Computing embeddings", leave=False):
        batch_items = items[i : i + batch_size]

        batch_tensors = []
        valid_indices = []

        for idx, item in enumerate(batch_items):
            # support (path, bbox) and (path, bbox, mask)
            global_idx = i + idx
            try:
                path, bbox, mask = item

                img = Image.open(path).convert("RGB")

                # Apply bbox crop first (if present)
                if bbox:
                    img_w, img_h = img.size
                    x, y, w, h = bbox
                    left = int(x * img_w)
                    top = int(y * img_h)
                    right = int((x + w) * img_w)
                    bottom = int((y + h) * img_h)
                    if right > left and bottom > top:
                        img = img.crop((left, top, right, bottom))
                    else:
                        raise ValueError(f"Invalid crop dimensions: {left}, {top}, {right}, {bottom}")

                # If a mask is provided, apply it so only masked pixels remain
                if mask is not None:
                    # Normalize mask to a PIL Image in 'L' mode
                    try:
                        if isinstance(mask, str) and Path(mask).exists():
                            mask_img = Image.open(mask).convert("L")
                        elif isinstance(mask, (list, tuple)):
                            mask_arr = np.array(mask, dtype=np.uint8)
                            mask_img = Image.fromarray(mask_arr).convert("L")
                        elif isinstance(mask, np.ndarray):
                            mask_img = Image.fromarray(mask.astype(np.uint8)).convert("L")
                        else:
                            # Some FiftyOne mask fields may already be a PIL-like object
                            mask_img = Image.fromarray(np.array(mask)).convert("L")
                    except Exception:
                        # If mask cannot be interpreted, skip masking
                        mask_img = None

                    if mask_img is not None:
                        # Resize mask to match current image size (nearest to preserve binary mask)
                        mask_resized = mask_img.resize(img.size, resample=Resampling.NEAREST)
                        # Create background image (black)
                        bg = Image.new("RGB", img.size, (0, 0, 0))
                        # Composite: keep pixels where mask != 0
                        img = Image.composite(img, bg, mask_resized)

                tensor = preprocess(img)
                batch_tensors.append(tensor)
                valid_indices.append(global_idx)
            except Exception as e:
                error_msg = f"Error loading item {global_idx} ({item}): {e}"
                errors.append(error_msg)

        if not batch_tensors:
            continue

        # Stack and move to device
        batch_tensor = torch.stack(batch_tensors).to(device)

        # Get embeddings
        batch_emb = model(batch_tensor).cpu().numpy()

        # Store with indices as keys
        for j in range(len(valid_indices)):
            embeddings_map[valid_indices[j]] = batch_emb[j].flatten()

    return embeddings_map, errors


def run_processing(
    dataset_name: str = JID_MASTER_DATASET,
    model_name: str = "hf-hub:BVRA/MegaDescriptor-L-384",
    embedding_field: str | None = None,
    patches_field: str | None = None,
    mask_field: str | None = None,
    batch_size: int = 32,
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> fo.Dataset | None:
    """Core Logic for computing embeddings.

    Args:
        dataset_name: Name of the FiftyOne dataset
        model_name: Name of the timm model to load
        embedding_field: Field name to store embeddings. Defaults to 'embeddings_{sanitized_model_name}'
        patches_field: If provided, computes embeddings for object patches in this field (e.g. 'detections').
        mask_field: If provided (and patches_field is set), uses masks from this field (e.g. 'mask').
        batch_size: Batch size for inference
        summary_location: Path to write summary JSON
        dry_run: If True, only log what would be done
        verbose: Enable detailed logging

    Returns:
        Updated FiftyOne dataset or None if dry_run
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)

    # Determine default field name if not provided
    if embedding_field is None:
        sanitized_model = model_name.split(":")[-1].replace("/", "_").replace("-", "_")
        embedding_field = f"embeddings_{sanitized_model}"

    logger_instance.info("Starting embedding computation for dataset: %s", dataset_name)
    logger_instance.info("Model: %s", model_name)
    logger_instance.info("Target Field: %s", embedding_field)
    if patches_field:
        logger_instance.info("Processing patches from field: %s", patches_field)
        if mask_field:
            logger_instance.info("Using masks from field: %s", mask_field)
    logger_instance.info("Batch Size: %d", batch_size)

    validate_resources(dataset_name)

    if dry_run:
        logger_instance.info("DRY RUN: Would compute embeddings using %s into field %s", model_name, embedding_field)
        return None

    dataset = fo.load_dataset(dataset_name)

    # Select only the 'image' slice, as we cannot compute embeddings on videos directly with this model
    try:
        view = dataset.select_group_slices("image")
        logger_instance.info("Selected 'image' slice for processing. %d samples found.", len(view))
    except (ValueError, AttributeError):
        view = dataset
        logger_instance.info("Using entire dataset for processing. %d samples.", len(view))

    items = []
    sample_det_map = []  # List of (sample_id, detection_id) for patch mapping

    if patches_field:
        logger_instance.info("Gathering detection patches from field '%s'...", patches_field)
        # Iterate over samples to collect patches robustly
        # We use iter_samples on the view (which is typically the 'image' slice)
        # Note: We do NOT convert to patches view here to allow robust write-back later

        count = 0
        for sample in view.select_fields([patches_field, "filepath"]).iter_samples(autosave=False):
            if sample[patches_field] is None:
                continue

            detections_obj = sample[patches_field]
            # Ensure it is a label list with detections
            if not hasattr(detections_obj, "detections") or not detections_obj.detections:
                continue

            for det in detections_obj.detections:
                mask = None
                if mask_field:
                    with contextlib.suppress(Exception):
                        mask = getattr(det, mask_field) if hasattr(det, mask_field) else det.get(mask_field)

                items.append((sample.filepath, det.bounding_box, mask))
                sample_det_map.append((sample.id, det.id))
                count += 1

        logger_instance.info("Collected %d patches.", count)
    else:
        if len(view) == 0:
            logger_instance.warning("No samples found to process.")
            return dataset

        logger_instance.info("Gathering inputs...")
        filepaths = view.values("filepath")
        items = [(fp, None, None) for fp in filepaths]

    if not items:
        logger_instance.warning("No items collected to process.")
        return dataset

    device = get_device()
    logger_instance.info("Using device: %s", device)

    # Load Model
    logger_instance.info("Loading model %s...", model_name)
    start_time = time.time()
    try:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        model.to(device)
    except Exception as e:
        logger_instance.error("Failed to load model %s: %s", model_name, e)
        raise

    load_time = time.time() - start_time
    logger_instance.info("Model loaded in %.2f seconds", load_time)

    # Preprocessing
    # MegaDescriptor expects 384x384 and ImageNet normalization
    preprocess = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Compute Embeddings
    logger_instance.info("Computing embeddings...")
    compute_start = time.time()
    embeddings_map, errors = compute_embeddings_batch(items, model, preprocess, device, batch_size=batch_size)
    compute_time = time.time() - compute_start

    if errors:
        logger_instance.warning("Encountered %d errors during processing", len(errors))
        for err in errors[:5]:  # Log first few errors
            logger_instance.debug(err)

    logger_instance.info("Computed %d embeddings in %.2f seconds", len(embeddings_map), compute_time)

    # Save to Dataset
    logger_instance.info("Saving embeddings to field '%s'...", embedding_field)

    samples_with_embeddings_count = 0

    if patches_field:
        # Save back to detections using collected IDs
        updates_by_sample: dict[str, dict[str, np.ndarray]] = {}

        for idx in range(len(items)):
            if idx in embeddings_map:
                s_id, d_id = sample_det_map[idx]
                if s_id not in updates_by_sample:
                    updates_by_sample[s_id] = {}
                updates_by_sample[s_id][d_id] = embeddings_map[idx]
                samples_with_embeddings_count += 1

        logger_instance.info("Updating %d samples with new detection embeddings...", len(updates_by_sample))

        for s_id, det_updates in tqdm(updates_by_sample.items(), desc="Saving results", leave=False):
            # Load sample from dataset directly
            sample = dataset[s_id]

            dirty = False
            detections_obj = sample[patches_field]
            if detections_obj and hasattr(detections_obj, "detections"):
                for det in detections_obj.detections:
                    if det.id in det_updates:
                        det[embedding_field] = det_updates[det.id]
                        dirty = True

            if dirty:
                sample.save()

    else:
        # Store embeddings in order for set_values
        ordered_embeddings: list[np.ndarray[tuple[Any, ...], np.dtype[Any]] | None] = []

        # Iterate indices matching the items list
        for idx in range(len(items)):
            if idx in embeddings_map:
                ordered_embeddings.append(embeddings_map[idx])
                samples_with_embeddings_count += 1
            else:
                ordered_embeddings.append(None)

        view.set_values(embedding_field, ordered_embeddings)
        dataset.save()

    dataset.reload()

    if summary_location:
        summary = {
            "dataset": dataset_name,
            "model": model_name,
            "embedding_field": embedding_field,
            "total_samples_in_view": len(view),
            "embeddings_computed": samples_with_embeddings_count,
            "errors": len(errors),
            "compute_time": compute_time,
            "device": str(device),
        }
        write_summary(summary, summary_location)

    logger_instance.info("Embedding computation completed successfully.")
    return dataset


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compute image embeddings for a FiftyOne dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=JID_MASTER_DATASET,
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hf-hub:BVRA/MegaDescriptor-L-384",
        help="Timm model name (default: %(default)s)",
    )
    parser.add_argument(
        "--field",
        type=str,
        default=None,
        help="Field name for embeddings (default: embeddings_{model_name})",
    )
    parser.add_argument(
        "--patches-field",
        type=str,
        default=None,
        help="If set, compute embeddings for patches in this field (e.g. 'detections').",
    )
    parser.add_argument(
        "--mask-field",
        type=str,
        default=None,
        help="If set (with --patches-field), uses masks from this field (e.g. 'mask').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-location",
        type=Path,
        default=None,
        help="Path to write summary JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    try:
        run_processing(
            dataset_name=args.dataset,
            model_name=args.model,
            embedding_field=args.field,
            patches_field=args.patches_field,
            mask_field=args.mask_field,
            batch_size=args.batch_size,
            summary_location=args.summary_location,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error("Error during embedding computation: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
