"""Background influence experiment.

Trains two re-identification models with a MiewID backbone and compares their
performance:

1. **Segmented jaguar model** – trained on cropped, segmented jaguar patches
   (background removed, transparent).
2. **Background-only model** – trained on the full image with the jaguar
   masked out (jaguar region transparent).

Both models share the same architecture, loss function, and hyper-parameters;
only the input images differ.

Usage
-----
From a notebook::

    from jaguars.background_influence.train import run_experiment
    results = run_experiment()

From the CLI::

    python -m jaguars.background_influence.train \\
        --fo-dataset JaguarCameraTrap/jaguars_camera_trap_0226_fiftyone \\
        --num-epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import fiftyone as fo
import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

from jaguars.common.logging_utils import setup_logger
from jaguars.reidentification.backbone import get_backbone
from jaguars.reidentification.config import (
    BackboneConfig,
    ReidentificationConfig,
    get_default_config,
)
from jaguars.reidentification.data_loader import DatasetMetadata
from jaguars.reidentification.dataset import EmbeddingDataset
from jaguars.reidentification.training.train import (
    build_training_criterion,
    train_epoch,
    validate_epoch,
)
from jaguars.reidentification.evaluation.evaluation import compute_comprehensive_metrics
from jaguars.reidentification.model import build_model
from jaguars.segmentation.utils import get_segmented_bbox_image, remove_jaguar_from_image

MODULE_NAME = "background_influence.train"
logger = setup_logger(MODULE_NAME)

# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------


def _pil_to_rgb(img: PILImage.Image) -> PILImage.Image:
    """Convert any PIL image (RGBA, L, …) to a solid RGB image.

    Transparent pixels are composited onto a black background so that the
    backbone receives a regular 3-channel input.
    """
    if img.mode == "RGBA":
        background = PILImage.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])  # alpha channel as mask
        return background
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_embeddings_from_images(
    images: list[PILImage.Image],
    backbone_config: BackboneConfig,
    device: str,
    batch_size: int = 32,
    desc: str = "Extracting embeddings",
) -> np.ndarray:
    """Run a backbone on a list of PIL images and return the embeddings.

    Parameters
    ----------
    images:
        List of PIL RGB images (already pre-processed into the desired
        variant, e.g. segmented or background-only).
    backbone_config:
        Configuration for the timm backbone.
    device:
        ``"cuda"``, ``"mps"`` or ``"cpu"``.
    batch_size:
        Images per forward pass.
    desc:
        Progress-bar description.

    Returns
    -------
    np.ndarray of shape ``(len(images), embedding_dim)``.
    """
    backbone = get_backbone(backbone_config, device)
    backbone.eval()
    preprocess = backbone.get_preprocess()

    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc=desc):
            batch_imgs = images[i : i + batch_size]
            tensors = [preprocess(img) for img in batch_imgs]
            batch_tensor = torch.stack(tensors).to(device)
            emb = backbone(batch_tensor).cpu().numpy()
            all_embeddings.append(emb)

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_dataset(
    fo_dataset_name: str,
    label_field: str = "ground_truth",
    split_field: str = "split",
    patches_field: str = "sam3_segmentations",
) -> dict[str, Any]:
    """Load a FiftyOne dataset and produce two parallel lists of images.

    For every sample that has a valid segmentation mask the function creates:
    * A *segmented* image  (cropped jaguar on transparent background → RGB).
    * A *background-only* image  (jaguar removed → RGB).

    Returns
    -------
    dict with keys:
        ``"segmented_images"``  – list[PILImage]
        ``"background_images"`` – list[PILImage]
        ``"labels"``            – list[str]
        ``"splits"``            – list[str]
    """
    if not fo.dataset_exists(fo_dataset_name):
        raise ValueError(f"FiftyOne dataset '{fo_dataset_name}' does not exist")

    dataset = fo.load_dataset(fo_dataset_name)

    segmented_images: list[PILImage.Image] = []
    background_images: list[PILImage.Image] = []
    labels: list[str] = []
    splits: list[str] = []

    skipped = 0
    processed = 0

    for sample in tqdm(dataset, desc="Preparing images"):
        processed += 1

        # --- label ----------------------------------------------------------
        label_obj = getattr(sample, label_field, None)
        if label_obj is None:
            skipped += 1
        else:
            label = label_obj.label if hasattr(label_obj, "label") else str(label_obj)

            # --- split ----------------------------------------------------------
            split = getattr(sample, split_field, "train") or "train"

            # --- segmented jaguar -----------------------------------------------
            seg_img = get_segmented_bbox_image(sample)
            if seg_img is None:
                skipped += 1
            else:
                # --- background only ------------------------------------------------
                bg_img = remove_jaguar_from_image(sample)
                if bg_img is None:
                    skipped += 1
                else:
                    segmented_images.append(_pil_to_rgb(seg_img))
                    background_images.append(_pil_to_rgb(bg_img))
                    labels.append(label)
                    splits.append(split)

        if processed % 1000 == 0:
            logger.info(
                "  [%d images processed] %d skipped so far (no detections/mask)",
                processed,
                skipped,
            )

    logger.info(
        "Dataset prepared: %d samples, %d skipped, %d unique labels",
        len(labels),
        skipped,
        len(set(labels)),
    )
    return {
        "segmented_images": segmented_images,
        "background_images": background_images,
        "labels": labels,
        "splits": splits,
    }


# ---------------------------------------------------------------------------
# Single-variant training loop
# ---------------------------------------------------------------------------


def _train_variant(
    variant_name: str,
    embeddings: np.ndarray,
    metadata: DatasetMetadata,
    config: ReidentificationConfig,
    device: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train a re-identification model on *one* image variant.

    This mirrors the core of ``reidentification.training.train.run_processing``
    but receives pre-computed embeddings directly instead of loading them from a
    FiftyOne field.
    """
    logger_inst = setup_logger(
        f"{MODULE_NAME}.{variant_name}",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    logger_inst.info("=" * 70)
    logger_inst.info("Training variant: %s", variant_name)
    logger_inst.info("=" * 70)

    # Split ----------------------------------------------------------------
    train_data = metadata.get_split(config.dataset.train_split)
    val_data = metadata.get_split(config.dataset.val_split)

    train_mask = np.array([s == config.dataset.train_split for s in metadata.split])
    val_mask = np.array([s == config.dataset.val_split for s in metadata.split])

    train_emb = embeddings[train_mask]
    val_emb = embeddings[val_mask]

    logger_inst.info("  Train: %d samples", len(train_data))
    logger_inst.info("  Val:   %d samples", len(val_data))

    # DataLoaders ----------------------------------------------------------
    from torch.utils.data import DataLoader

    train_ds = EmbeddingDataset(train_emb.tolist(), train_data.labels_encoded.tolist())
    val_ds = EmbeddingDataset(val_emb.tolist(), val_data.labels_encoded.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )

    # Model ----------------------------------------------------------------
    input_dim = train_emb.shape[1]
    model = build_model(input_dim, metadata.num_classes, config.model)
    model.to(device)

    # Loss -----------------------------------------------------------------
    cls_criterion, triplet_criterion, _ = build_training_criterion(
        config, metadata.num_classes, config.model.embedding_dim
    )

    # Optimiser / scheduler ------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler: Any = None
    if config.training.scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
        )
    elif config.training.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs
        )

    # Training loop --------------------------------------------------------
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_map": [],
        "lr": [],
    }

    best_val_metric = float("-inf")
    best_epoch = 0
    patience_counter = 0
    save_dir = config.training.save_dir / variant_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.training.num_epochs):
        logger_inst.info("Epoch %d/%d", epoch + 1, config.training.num_epochs)

        # -- train --
        train_loss, train_acc, _ = train_epoch(
            model,
            train_loader,
            cls_criterion,
            optimizer,
            device,
            triplet_criterion=triplet_criterion,
            triplet_weight=config.training.triplet_weight,
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # -- validate --
        val_loss, val_acc = validate_epoch(
            model,
            val_loader,
            cls_criterion,
            device,
            triplet_criterion=triplet_criterion,
            triplet_weight=config.training.triplet_weight,
        )
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # -- comprehensive metrics --
        val_metrics = compute_comprehensive_metrics(
            model=model,
            val_embeddings=val_emb,
            val_labels=val_data.labels_encoded,
            train_labels=train_data.labels_encoded,
            all_labels=metadata.labels_encoded,
            label_encoder=metadata.label_encoder,
            device=device,
            max_cmc_rank=50,
            min_total_samples_for_filtered=9,
        )

        val_map = val_metrics["identity_balanced_map"]
        history["val_map"].append(val_map)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        logger_inst.info(
            "  loss=%.4f  acc=%.2f%%  val_loss=%.4f  val_acc=%.2f%%  mAP=%.4f  CMC@1=%.4f",
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_map,
            val_metrics["cmc@1"],
        )

        # -- scheduler --
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # -- best model --
        if val_map > best_val_metric:
            best_val_metric = val_map
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": val_map,
                    "val_loss": val_loss,
                    "config": config,
                    "variant": variant_name,
                },
                save_dir / "best_model.pt",
            )
            logger_inst.info("  ✓ Saved best model (mAP=%.4f)", val_map)
        else:
            patience_counter += 1

        # -- early stopping --
        if patience_counter >= config.training.early_stopping_patience:
            logger_inst.info("  Early stopping after %d epochs", epoch + 1)
            break

    results = {
        "variant": variant_name,
        "status": "completed",
        "best_epoch": best_epoch + 1,
        "best_val_map": float(max(history["val_map"])),
        "best_val_loss": float(min(history["val_loss"])),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_val_map": float(history["val_map"][-1]),
        "num_epochs_trained": len(history["train_loss"]),
        "history": {k: [float(v) for v in vs] for k, vs in history.items()},
    }

    # Persist summary
    summary_path = save_dir / "training_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger_inst.info("Summary written to %s", summary_path)

    return results


# ---------------------------------------------------------------------------
# Main experiment entry-point
# ---------------------------------------------------------------------------


def run_experiment(
    fo_dataset_name: str = "JaguarCameraTrap/jaguars_camera_trap_0226_fiftyone",
    config: ReidentificationConfig | None = None,
    device: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full background-influence experiment.

    Steps
    -----
    1. Load the FiftyOne dataset and build two image variants per sample.
    2. Extract backbone embeddings for each variant.
    3. Train a re-id model on the *segmented* embeddings.
    4. Train a re-id model on the *background-only* embeddings.
    5. Return a comparison dict.

    Parameters
    ----------
    fo_dataset_name:
        Name of the FiftyOne dataset to use.
    config:
        Reidentification config.  If ``None`` a sensible default is created.
    device:
        Compute device.  Auto-detected when ``None``.
    verbose:
        Extra logging.

    Returns
    -------
    dict with keys ``"segmented"``, ``"background"``, and ``"comparison"``.
    """
    logger_inst = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)

    # -- config -----------------------------------------------------------
    if config is None:
        config = get_default_config()
        # Use MiewID (MegaDescriptor-L-384) as backbone
        config.backbone.name = "hf-hub:BVRA/MegaDescriptor-L-384"
        config.backbone.input_size = 384
        config.backbone.embedding_dim = 1536
        config.training.save_dir = Path("data/models/background_influence")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    config.training.device = device

    logger_inst.info("Device: %s", device)
    logger_inst.info("Backbone: %s", config.backbone.name)

    # -- 1. Prepare images ------------------------------------------------
    logger_inst.info("Preparing image variants …")
    data = prepare_dataset(
        fo_dataset_name=fo_dataset_name,
        label_field=config.dataset.fo_label_field,
        split_field=config.dataset.fo_split_field,
        patches_field=config.dataset.fo_patches_field,
    )

    # Build a shared DatasetMetadata (label encoder must be identical for
    # both variants so that class indices are comparable).
    metadata = DatasetMetadata(
        image_paths=[""] * len(data["labels"]),  # not used
        labels=data["labels"],
        split=data["splits"],
    )

    logger_inst.info(
        "Samples: %d  |  Classes: %d  |  Train: %d  |  Val: %d",
        len(data["labels"]),
        metadata.num_classes,
        sum(1 for s in data["splits"] if s == config.dataset.train_split),
        sum(1 for s in data["splits"] if s == config.dataset.val_split),
    )

    # -- 2. Extract embeddings -------------------------------------------
    logger_inst.info("Extracting embeddings for segmented images …")
    seg_embeddings = extract_embeddings_from_images(
        data["segmented_images"],
        config.backbone,
        device,
        batch_size=config.backbone.batch_size,
        desc="Segmented embeddings",
    )

    logger_inst.info("Extracting embeddings for background-only images …")
    bg_embeddings = extract_embeddings_from_images(
        data["background_images"],
        config.backbone,
        device,
        batch_size=config.backbone.batch_size,
        desc="Background embeddings",
    )

    # -- 3. Train segmented model ----------------------------------------
    logger_inst.info("Training segmented-jaguar model …")
    seg_results = _train_variant(
        "segmented",
        seg_embeddings,
        metadata,
        config,
        device,
        verbose=verbose,
    )

    # -- 4. Train background model ---------------------------------------
    logger_inst.info("Training background-only model …")
    bg_results = _train_variant(
        "background",
        bg_embeddings,
        metadata,
        config,
        device,
        verbose=verbose,
    )

    # -- 5. Comparison ---------------------------------------------------
    comparison = {
        "segmented_best_map": seg_results["best_val_map"],
        "background_best_map": bg_results["best_val_map"],
        "map_difference": seg_results["best_val_map"] - bg_results["best_val_map"],
    }

    logger_inst.info("=" * 70)
    logger_inst.info("EXPERIMENT RESULTS")
    logger_inst.info("=" * 70)
    logger_inst.info(
        "  Segmented mAP:   %.4f  (epoch %d)",
        seg_results["best_val_map"],
        seg_results["best_epoch"],
    )
    logger_inst.info(
        "  Background mAP:  %.4f  (epoch %d)",
        bg_results["best_val_map"],
        bg_results["best_epoch"],
    )
    logger_inst.info("  Δ mAP (seg - bg): %.4f", comparison["map_difference"])

    all_results = {
        "segmented": seg_results,
        "background": bg_results,
        "comparison": comparison,
    }

    # Persist combined results
    results_path = config.training.save_dir / "experiment_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    logger_inst.info("Full results saved to %s", results_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description="Background influence experiment for jaguar re-identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fo-dataset",
        type=str,
        default="JaguarCameraTrap/jaguars_camera_trap_0226_fiftyone",
        help="FiftyOne dataset name",
    )
    parser.add_argument("--backbone", type=str, default="hf-hub:BVRA/MegaDescriptor-L-384")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("data/models/background_influence"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    config = get_default_config()
    config.backbone.name = args.backbone
    config.backbone.input_size = 384
    config.backbone.embedding_dim = 1536
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.num_epochs
    config.training.learning_rate = args.learning_rate
    config.training.save_dir = args.save_dir

    run_experiment(
        fo_dataset_name=args.fo_dataset,
        config=config,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
