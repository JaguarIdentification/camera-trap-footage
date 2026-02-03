"""Add embeddings for multiple backbones to FiftyOne dataset.

This module extends the single-backbone embedding computation to support
multiple backbones, storing each backbone's embeddings in a separate field.

Usage in ingestion pipeline:
    from jaguars.ingestion.processing.add_multi_backbone_embeddings import run_processing

    run_processing(
        dataset_name="JID_Master_Dataset",
        backbones=[
            "hf-hub:BVRA/MegaDescriptor-L-384",
            "hf-hub:BVRA/MegaDescriptor-B-224",
            "vit_large_patch14_dinov2.lvd142m",
            # ... more backbones
        ],
        patches_field="sam3_segmentations",  # Optional: for segmented images
        mask_field="mask",  # Optional: for masked embeddings
    )
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import fiftyone as fo
import numpy as np
import torch
from tqdm import tqdm

from jaguars.common.logging_utils import setup_logger
from jaguars.reidentification.backbone import get_backbone
from jaguars.reidentification.config import BackboneConfig

MODULE_NAME = "ingestion.processing.add_multi_backbone_embeddings"
logger = setup_logger(MODULE_NAME)


# Default backbones to compute embeddings for
DEFAULT_BACKBONES = [
    # MegaDescriptor models (wildlife specialists)
    {
        "name": "hf-hub:BVRA/MegaDescriptor-L-384",
        "embedding_dim": 1536,
        "input_size": 384,
        "field_suffix": "BVRA_MegaDescriptor_L_384",
    },
    {
        "name": "hf-hub:BVRA/MegaDescriptor-B-224",
        "embedding_dim": 768,
        "input_size": 224,
        "field_suffix": "BVRA_MegaDescriptor_B_224",
    },
    # DINOv2 models
    {
        "name": "vit_large_patch14_dinov2.lvd142m",
        "embedding_dim": 1024,
        "input_size": 518,
        "field_suffix": "DINOv2_Large",
    },
    {
        "name": "vit_base_patch14_dinov2.lvd142m",
        "embedding_dim": 768,
        "input_size": 518,
        "field_suffix": "DINOv2_Base",
    },
    # CNN models
    {
        "name": "resnet50",
        "embedding_dim": 2048,
        "input_size": 224,
        "field_suffix": "ResNet50",
    },
    {
        "name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "embedding_dim": 1024,
        "input_size": 224,
        "field_suffix": "ConvNeXtV2_Base",
    },
    # DINOv3 models (latest self-supervised)
    {
        "name": "vit_large_patch16_dinov3.lvd1689m",
        "embedding_dim": 1024,
        "input_size": 512,
        "field_suffix": "DINOv3_Large",
    },
    {
        "name": "vit_base_patch16_dinov3.lvd1689m",
        "embedding_dim": 768,
        "input_size": 512,
        "field_suffix": "DINOv3_Base",
    },
]


def validate_resources(
    dataset_name: str,
    backbones: list[dict[str, Any]],
) -> None:
    """Validate that all required resources exist.

    Args:
        dataset_name: Name of FiftyOne dataset
        backbones: List of backbone configurations

    Raises:
        ValueError: If resources are missing
    """
    # Check dataset exists
    if not fo.dataset_exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' does not exist")

    # Check CUDA availability (recommended but not required)
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, will use CPU (this will be slow)")

    logger.info("Resource validation passed")


def run_processing(
    dataset_name: str,
    backbones: list[dict[str, Any]] | None = None,
    patches_field: str | None = None,
    mask_field: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute embeddings for multiple backbones and store in FiftyOne.

    Args:
        dataset_name: Name of FiftyOne dataset
        backbones: List of backbone configs (name, embedding_dim, input_size, field_suffix).
                   If None, uses DEFAULT_BACKBONES.
        patches_field: Optional field containing detection patches (e.g., "sam3_segmentations")
        mask_field: Optional mask field for masked embedding (e.g., "mask")
        batch_size: Batch size for embedding computation
        device: Device to use (cuda/cpu)
        overwrite: Whether to overwrite existing embeddings
        dry_run: Only validate, don't compute
        verbose: Verbose logging

    Returns:
        Dictionary with results for each backbone
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)

    if backbones is None:
        backbones = DEFAULT_BACKBONES

    logger_instance.info(f"Computing embeddings for {len(backbones)} backbones")
    for bb in backbones:
        logger_instance.info(f"  - {bb['name']} → embeddings_{bb['field_suffix']}")

    # Validate resources
    validate_resources(dataset_name, backbones)

    if dry_run:
        logger_instance.info("Dry run completed successfully")
        return {"status": "dry_run"}

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger_instance.info(f"Using device: {device}")

    # Load dataset
    dataset = fo.load_dataset(dataset_name)

    # Get image view (only process images, not videos)
    images_view = dataset.select_group_slices("image")

    results = {}

    # Process each backbone
    for backbone_config in backbones:
        backbone_name = backbone_config["name"]
        field_suffix = backbone_config["field_suffix"]
        embedding_field = f"embeddings_{field_suffix}"

        logger_instance.info(f"\n{'='*70}")
        logger_instance.info(f"Processing backbone: {backbone_name}")
        logger_instance.info(f"  Embedding field: {embedding_field}")
        logger_instance.info(f"{'='*70}")

        # Check if embeddings already exist
        if patches_field:
            # For patches, check if field exists on detections
            sample = images_view.first()
            if sample and hasattr(sample, patches_field):
                patches = getattr(sample, patches_field)
                if patches and patches.detections and len(patches.detections) > 0:
                    detection = patches.detections[0]
                    if hasattr(detection, embedding_field) and not overwrite:
                        logger_instance.info(f"  ✓ Embeddings already exist, skipping (use --overwrite to recompute)")
                        results[backbone_name] = {"status": "skipped", "reason": "already_exists"}
                        continue
        else:
            # For full images, check if field exists on samples
            if embedding_field in images_view.get_field_schema() and not overwrite:
                logger_instance.info(f"  ✓ Embeddings already exist, skipping (use --overwrite to recompute)")
                results[backbone_name] = {"status": "skipped", "reason": "already_exists"}
                continue

        # Create backbone
        config = BackboneConfig(
            name=backbone_name,
            embedding_dim=backbone_config["embedding_dim"],
            input_size=backbone_config["input_size"],
            pretrained=True,
            batch_size=batch_size,
        )

        logger_instance.info(f"  Loading model...")
        backbone = get_backbone(config, device=device)
        backbone.eval()

        # Compute embeddings
        if patches_field:
            logger_instance.info(f"  Computing patch embeddings from field: {patches_field}")
            
            # Collect all patches with their image paths and bboxes
            patch_items = []  # List of (filepath, bbox, mask)
            sample_det_map = []  # List of (sample_id, detection_id)
            
            logger_instance.info(f"  Gathering patches...")
            for sample in images_view:
                if not hasattr(sample, patches_field):
                    continue

                patches = getattr(sample, patches_field)
                if not patches or not patches.detections:
                    continue

                for detection in patches.detections:
                    # Get bbox and optional mask
                    bbox = detection.bounding_box if hasattr(detection, 'bounding_box') else None
                    mask_data = None
                    if mask_field and hasattr(detection, mask_field):
                        mask_data = getattr(detection, mask_field)
                    
                    patch_items.append((sample.filepath, bbox, mask_data))
                    sample_det_map.append((sample.id, detection.id))
            
            logger_instance.info(f"  Found {len(patch_items)} patches to process")
            
            if patch_items:
                # Use backbone's batch extraction
                from PIL import Image
                from PIL.Image import Resampling
                import contextlib
                
                preprocess = backbone.get_preprocess()
                embeddings_list = []
                
                # Process in batches
                for i in tqdm(range(0, len(patch_items), batch_size), desc=f"Computing {backbone_name} embeddings"):
                    batch_items = patch_items[i : i + batch_size]
                    batch_tensors = []
                    
                    for filepath, bbox, mask_data in batch_items:
                        try:
                            img = Image.open(filepath).convert("RGB")
                            
                            # Crop to bbox if present
                            if bbox:
                                img_w, img_h = img.size
                                x, y, w, h = bbox
                                left = int(x * img_w)
                                top = int(y * img_h)
                                right = int((x + w) * img_w)
                                bottom = int((y + h) * img_h)
                                if right > left and bottom > top:
                                    img = img.crop((left, top, right, bottom))
                            
                            # Apply mask if present
                            if mask_data is not None:
                                try:
                                    if isinstance(mask_data, str) and Path(mask_data).exists():
                                        mask_img = Image.open(mask_data).convert("L")
                                    elif isinstance(mask_data, (list, tuple)):
                                        mask_arr = np.array(mask_data, dtype=np.uint8)
                                        mask_img = Image.fromarray(mask_arr).convert("L")
                                    elif isinstance(mask_data, np.ndarray):
                                        mask_img = Image.fromarray(mask_data.astype(np.uint8)).convert("L")
                                    else:
                                        mask_img = Image.fromarray(np.array(mask_data)).convert("L")
                                    
                                    # Resize mask to match cropped image
                                    mask_resized = mask_img.resize(img.size, resample=Resampling.NEAREST)
                                    bg = Image.new("RGB", img.size, (0, 0, 0))
                                    img = Image.composite(img, bg, mask_resized)
                                except Exception:
                                    pass  # Skip masking if it fails
                            
                            tensor = preprocess(img)
                            batch_tensors.append(tensor)
                        except Exception as e:
                            logger_instance.warning(f"Error processing patch: {e}")
                            # Add zero tensor as fallback
                            batch_tensors.append(torch.zeros(3, config.input_size, config.input_size))
                    
                    if batch_tensors:
                        batch_tensor = torch.stack(batch_tensors).to(device)
                        with torch.no_grad():
                            batch_emb = backbone(batch_tensor).cpu().numpy()
                        embeddings_list.extend([emb for emb in batch_emb])
                
                # Store embeddings back to detections
                logger_instance.info(f"  Storing {len(embeddings_list)} embeddings...")
                updates_by_sample = {}  # sample_id -> {detection_id -> embedding}
                
                for idx, (sample_id, det_id) in enumerate(sample_det_map):
                    if idx < len(embeddings_list):
                        if sample_id not in updates_by_sample:
                            updates_by_sample[sample_id] = {}
                        updates_by_sample[sample_id][det_id] = embeddings_list[idx]
                
                # Update samples
                for sample_id, det_updates in tqdm(updates_by_sample.items(), desc="Saving", leave=False):
                    sample = dataset[sample_id]
                    patches = getattr(sample, patches_field)
                    if patches and hasattr(patches, "detections"):
                        for det in patches.detections:
                            if det.id in det_updates:
                                det[embedding_field] = det_updates[det.id].tolist()
                    sample.save()
                
                logger_instance.info(f"  ✓ Computed embeddings for {len(embeddings_list)} patches")
                results[backbone_name] = {"status": "completed", "num_patches": len(embeddings_list)}
            else:
                logger_instance.info(f"  No patches found to process")
                results[backbone_name] = {"status": "skipped", "reason": "no_patches"}

        else:
            logger_instance.info(f"  Computing full image embeddings")

            # Collect all image paths
            image_paths = [s.filepath for s in images_view]

            # Compute embeddings
            embeddings = backbone.extract_embeddings(image_paths, device=device, desc=f"Computing {backbone_name} embeddings")

            # Store embeddings
            logger_instance.info(f"  Storing embeddings in field: {embedding_field}")
            for sample, emb in zip(images_view, embeddings):
                sample[embedding_field] = emb.tolist()
                sample.save()

            logger_instance.info(f"  ✓ Computed embeddings for {len(images_view)} images")
            results[backbone_name] = {"status": "completed", "num_samples": len(images_view)}

    logger_instance.info(f"\n{'='*70}")
    logger_instance.info("Multi-backbone embedding computation completed!")
    logger_instance.info(f"{'='*70}")

    return {"status": "completed", "backbones": results}


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Add embeddings for multiple backbones to FiftyOne dataset")

    parser.add_argument("--dataset-name", type=str, required=True, help="FiftyOne dataset name")
    parser.add_argument("--patches-field", type=str, help="Field containing detection patches")
    parser.add_argument("--mask-field", type=str, help="Mask field for masked embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Validate only")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    run_processing(
        dataset_name=args.dataset_name,
        patches_field=args.patches_field,
        mask_field=args.mask_field,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
