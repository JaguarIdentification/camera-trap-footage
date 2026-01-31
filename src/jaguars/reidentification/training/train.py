"""Training module for jaguar re-identification.

This module follows the architecture pattern:
- validate_resources()
- write_summary()
- run_processing()
- main()
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from jaguars.common.logging_utils import setup_logger
from jaguars.reidentification.backbone import get_backbone
from jaguars.reidentification.config import ReidentificationConfig, get_default_config, load_config_from_dict
from jaguars.reidentification.data_loader import load_dataset
from jaguars.reidentification.dataset import EmbeddingDataset
from jaguars.reidentification.evaluation.evaluation import compute_validation_map
from jaguars.reidentification.model import build_model

MODULE_NAME = "reidentification.training"
logger = setup_logger(MODULE_NAME)


def validate_resources(config: ReidentificationConfig) -> None:
    """Validate that all required resources are available.

    Args:
        config: Complete configuration

    Raises:
        ValueError: If required resources are missing
    """
    # Check dataset source
    if config.dataset.source == "disk" and config.dataset.data_dir is None:
        raise ValueError("data_dir must be specified for disk-based datasets")
    elif config.dataset.source == "huggingface" and config.dataset.hf_repo is None:
        raise ValueError("hf_repo must be specified for HuggingFace datasets")
    elif config.dataset.source == "fiftyone":
        try:
            import fiftyone as fo

            if not fo.dataset_exists(config.dataset.fo_dataset_name):
                raise ValueError(f"FiftyOne dataset '{config.dataset.fo_dataset_name}' does not exist")
        except ImportError:
            raise ImportError("fiftyone library required for FiftyOne datasets")

    # Check save directory
    if not config.dry_run:
        config.training.save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Resource validation passed")


def write_summary(summary_data: dict[str, Any], summary_location: Path, to_wandb: bool = False) -> None:
    """Write training summary to file and optionally to wandb.

    Args:
        summary_data: Summary data dictionary
        summary_location: Path to save summary JSON
        to_wandb: Whether to log to wandb
    """
    summary_location.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_location, "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Summary written to {summary_location}")

    if to_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(summary_data)
        except ImportError:
            logger.warning("wandb not installed, skipping wandb logging")


def train_epoch(
    model: nn.Module, loader: DataLoader[Any], criterion: nn.Module, optimizer: Optimizer, device: str, log_wandb: bool = False
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        log_wandb: Whether to log to wandb

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm.tqdm(loader, desc="Training", leave=False)
    for batch_idx, (embeddings, labels) in enumerate(pbar):
        embeddings, labels = embeddings.to(device), labels.to(device)

        # Forward pass
        logits, _ = model(embeddings, labels)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})

        # Log to wandb
        if log_wandb:
            try:
                import wandb

                if wandb.run is not None and batch_idx % 10 == 0:
                    wandb.log({"train/batch_loss": loss.item(), "train/batch_acc": 100.0 * correct / total})
            except ImportError:
                pass

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate_epoch(model: nn.Module, loader: DataLoader[Any], criterion: nn.Module, device: str) -> tuple[float, float]:
    """Validate for one epoch.

    Args:
        model: Model to validate
        loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(loader, desc="Validation", leave=False)
        for embeddings, labels in pbar:
            embeddings, labels = embeddings.to(device), labels.to(device)

            # Forward pass
            logits, _ = model(embeddings, labels)
            loss = criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def run_processing(
    config: ReidentificationConfig | None = None,
    config_dict: dict[str, Any] | None = None,
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Core training logic.

    Args:
        config: Complete configuration object
        config_dict: Configuration as dictionary (alternative to config)
        summary_location: Path to write summary JSON
        dry_run: If True, only validate and log what would be done
        verbose: Enable detailed logging

    Returns:
        Dictionary with training results
    """
    # Setup logging
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)

    # Load configuration
    if config is None:
        if config_dict is not None:
            config = load_config_from_dict(config_dict)
        else:
            config = get_default_config()

    if dry_run:
        config.dry_run = True

    logger_instance.info("Starting re-identification training...")
    logger_instance.info(f"Dataset source: {config.dataset.source}")
    logger_instance.info(f"Backbone: {config.backbone.name}")
    logger_instance.info(f"Device: {config.training.device}")

    # Validate resources
    validate_resources(config)

    if dry_run:
        logger_instance.info("DRY RUN: Would train model with provided configuration")
        return {"status": "dry_run"}

    # Initialize wandb
    if config.wandb.enabled:
        try:
            import wandb

            wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                name=config.wandb.run_name,
                tags=config.wandb.tags,
                notes=config.wandb.notes,
                config={
                    "backbone": config.backbone.__dict__,
                    "model": config.model.__dict__,
                    "training": config.training.__dict__,
                    "dataset": config.dataset.source,
                },
            )
        except ImportError:
            logger_instance.warning("wandb not installed, skipping wandb logging")
            config.wandb.enabled = False

    # Set device
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger_instance.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load dataset
    logger_instance.info("Loading dataset...")
    dataset_metadata = load_dataset(config.dataset)

    # Split dataset
    train_data = dataset_metadata.get_split(config.dataset.train_split)
    val_data = dataset_metadata.get_split(config.dataset.val_split)

    logger_instance.info(f"Dataset loaded:")
    logger_instance.info(f"  Train: {len(train_data)} samples")
    logger_instance.info(f"  Val: {len(val_data)} samples")
    logger_instance.info(f"  Num classes: {dataset_metadata.num_classes}")

    # Extract or load embeddings
    if train_data.embeddings is None:
        logger_instance.info("Extracting embeddings with backbone...")
        backbone = get_backbone(config.backbone, device)
        backbone.eval()

        train_embeddings = backbone.extract_embeddings(train_data.image_paths, device, desc="Train embeddings")
        val_embeddings = backbone.extract_embeddings(val_data.image_paths, device, desc="Val embeddings")

        logger_instance.info(f"Embeddings extracted: {train_embeddings.shape}")
    else:
        logger_instance.info("Using pre-computed embeddings")
        train_embeddings = train_data.embeddings if train_data.embeddings is not None else np.array([])
        val_embeddings = val_data.embeddings if val_data.embeddings is not None else np.array([])

    # Create datasets and dataloaders
    train_dataset = EmbeddingDataset(train_embeddings.tolist(), train_data.labels_encoded.tolist())
    val_dataset = EmbeddingDataset(val_embeddings.tolist(), val_data.labels_encoded.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )

    logger_instance.info(f"DataLoaders created:")
    logger_instance.info(f"  Train batches: {len(train_loader)}")
    logger_instance.info(f"  Val batches: {len(val_loader)}")

    # Build model
    input_dim = train_embeddings.shape[1]
    model = build_model(input_dim, dataset_metadata.num_classes, config.model)
    model.to(device)

    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    # Setup scheduler
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | torch.optim.lr_scheduler.CosineAnnealingLR | None
    if config.training.scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.training.scheduler_factor, patience=config.training.scheduler_patience
        )
    elif config.training.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    else:
        scheduler = None

    logger_instance.info("Training components initialized")

    # Training loop
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_map": [], "lr": []}

    best_val_metric = float("-inf") if config.training.early_stopping_mode == "max" else float("inf")
    best_epoch = 0
    patience_counter = 0

    logger_instance.info(f"Starting training for {config.training.num_epochs} epochs...")
    logger_instance.info("=" * 70)

    for epoch in range(config.training.num_epochs):
        logger_instance.info(f"\nEpoch {epoch+1}/{config.training.num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config.wandb.enabled)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Compute validation mAP
        val_map = compute_validation_map(model, val_embeddings, val_data.labels_encoded, dataset_metadata.label_encoder, device)
        history["val_map"].append(val_map)

        # Get learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # Log metrics
        logger_instance.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger_instance.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger_instance.info(f"Val mAP: {val_map:.4f}")
        logger_instance.info(f"Learning Rate: {current_lr:.6f}")

        # Log to wandb
        if config.wandb.enabled:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train/loss": train_loss,
                            "train/acc": train_acc,
                            "val/loss": val_loss,
                            "val/acc": val_acc,
                            "val/map": val_map,
                            "lr": current_lr,
                        }
                    )
            except ImportError:
                pass

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Check for improvement
        current_metric = val_map if config.training.early_stopping_metric == "val_map" else val_loss
        is_better = (
            current_metric > best_val_metric
            if config.training.early_stopping_mode == "max"
            else current_metric < best_val_metric
        )

        if is_better:
            best_val_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            if config.training.save_best_only:
                checkpoint_path = config.training.save_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_map": val_map,
                        "val_loss": val_loss,
                        "config": config,
                    },
                    checkpoint_path,
                )
                logger_instance.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % config.training.checkpoint_frequency == 0:
            checkpoint_path = config.training.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": val_map,
                    "val_loss": val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            logger_instance.info(f"Saved checkpoint to {checkpoint_path}")

        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            logger_instance.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    logger_instance.info("=" * 70)
    logger_instance.info("Training completed!")
    logger_instance.info(f"Best epoch: {best_epoch+1}")
    logger_instance.info(f"Best {config.training.early_stopping_metric}: {best_val_metric:.4f}")

    # Prepare results
    results = {
        "status": "completed",
        "best_epoch": best_epoch + 1,
        "best_val_map": float(max(history["val_map"])),
        "best_val_loss": float(min(history["val_loss"])),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_val_map": float(history["val_map"][-1]),
        "num_epochs": len(history["train_loss"]),
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    # Write summary
    if summary_location is not None:
        write_summary(results, summary_location, config.wandb.enabled)

    # Close wandb
    if config.wandb.enabled:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass

    return results


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train jaguar re-identification model", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Dataset args
    parser.add_argument("--dataset-source", type=str, choices=["disk", "huggingface", "fiftyone"], default="fiftyone")
    parser.add_argument("--data-dir", type=Path, help="Data directory (for disk source)")
    parser.add_argument("--hf-repo", type=str, help="HuggingFace repo (for huggingface source)")
    parser.add_argument("--fo-dataset", type=str, default="JID_Master_Dataset", help="FiftyOne dataset name")

    # Training args
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")

    # Model args
    parser.add_argument("--backbone", type=str, default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)

    # Wandb args
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="jaguar-reidentification")
    parser.add_argument("--wandb-entity", type=str, default=None)

    # Output args
    parser.add_argument("--save-dir", type=Path, default=Path("data/models/reidentification"))
    parser.add_argument("--summary-location", type=Path, default=None)

    # Runtime args
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Build configuration
    config = get_default_config()

    # Dataset config
    config.dataset.source = args.dataset_source
    if args.data_dir:
        config.dataset.data_dir = args.data_dir
    if args.hf_repo:
        config.dataset.hf_repo = args.hf_repo
    config.dataset.fo_dataset_name = args.fo_dataset

    # Training config
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.num_epochs
    config.training.learning_rate = args.learning_rate
    config.training.device = args.device
    config.training.save_dir = args.save_dir

    # Model config
    config.backbone.name = args.backbone
    config.model.embedding_dim = args.embedding_dim
    config.model.hidden_dim = args.hidden_dim

    # Wandb config
    config.wandb.enabled = args.wandb
    config.wandb.project = args.wandb_project
    config.wandb.entity = args.wandb_entity

    # Runtime
    config.dry_run = args.dry_run
    config.verbose = args.verbose

    try:
        run_processing(config=config, summary_location=args.summary_location, dry_run=args.dry_run, verbose=args.verbose)
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
