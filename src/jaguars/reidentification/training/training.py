import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import wandb

from typing import Any
from collections.abc import Callable
import tqdm


from jaguars.reidentification.dataset import EmbeddingDataset
from jaguars.reidentification.model import ArcFaceModel, EmbeddingProjection


def train_epoch(model: ArcFaceModel, loader: DataLoader, criterion: Callable, optimizer: Optimizer, device: str):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm.tqdm(loader, desc="Training", leave=False)
    for embeddings, labels in pbar:
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

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate_epoch(model: ArcFaceModel, loader: DataLoader, criterion: Callable, device: str):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(loader, desc="Validation", leave=False)
        for embeddings, labels in pbar:
            embeddings, labels = embeddings.to(device), labels.to(device)

            logits, _ = model(embeddings, labels)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(
    model: ArcFaceModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable,
    optimizer: Optimizer,
    scheduler: Any,
    device: str,
    config: dict[str, Any],
    baseline_val_embeddings: Any,
    val_data: Any,
    label_encoder: Any,
    num_classes: int,
) -> None:
    """Training function for jaguar re-identification model."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_map": [], "lr": []}

    best_val_loss = float("inf")
    best_map = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training for {config['num_epochs']} epochs...")
    print("=" * 70)

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Compute validation mAP
        val_map = compute_validation_map(model, baseline_val_embeddings, val_data["ground_truth"].values, label_encoder)

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Store history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_map"].append(val_map)
        history["lr"].append(current_lr)

        # Log to W&B
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_map": val_map,
                "learning_rate": current_lr,
            }
        )

        # Print summary
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        print(f"  Val mAP:    {val_map:.4f} | LR: {current_lr:.2e}")

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_map = val_map
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint_path = config["checkpoint_dir"] / "arcface_best.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_map": val_map,
                    "config": config,
                    "label_encoder_classes": label_encoder.classes_.tolist(),
                    "num_classes": num_classes,
                },
                checkpoint_path,
            )

            print("  [New best model saved]")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config['patience']}")

        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        print("\n" + "=" * 70)
        print(f"Training complete!")
        print(f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val mAP: {best_map:.4f})")

        # Log best metrics as W&B summary for easy comparison across runs
        wandb.run.summary["best_val_mAP"] = best_map
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["total_epochs"] = len(history["train_loss"])


def load_backbone_model() -> EmbeddingProjection:
    """Load pre-trained backbone model for embedding projection."""
    pass


def create_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    # Create datasets
    train_dataset = EmbeddingDataset(baseline_train_embeddings, train_data["label_encoded"].values)
    val_dataset = EmbeddingDataset(baseline_val_embeddings, val_data["label_encoded"].values)

    # Create dataloaders
    # Note: pin_memory=False for MPS compatibility
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)

    print("DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {config['batch_size']}")

    return train_loader, val_loader


def run(config: Any) -> None:
    wandb.init(entity=config["wandb_entity"], project=config["wandb_project"], config=config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    model = ArcFaceModel(
        input_dim=config["megadescriptor_dim"],
        num_classes=config["num_classes"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        margin=config["arcface_margin"],
        dropout=config["dropout"],
    ).to(device)

    # Setup training components
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    print("Training components initialized:")
    print("  Loss: CrossEntropyLoss")
    print(f"  Optimizer: AdamW (lr={config['learning_rate']}, weight_decay={config['weight_decay']})")
    print("  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")

    train(criterion, optimizer, scheduler, device)
