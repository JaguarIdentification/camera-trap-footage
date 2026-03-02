"""Configuration module for jaguar re-identification.

This module defines all configuration dataclasses for training, evaluation,
model architecture, and dataset loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class BackboneConfig:
    """Configuration for backbone feature extractor."""

    name: str = "vit_large_patch14_dinov2.lvd142m"  # MegaDescriptor-L-384
    pretrained: bool = True
    input_size: int = 384
    batch_size: int = 32
    embedding_dim: int = 1536  # Output dimension from backbone


@dataclass
class ModelConfig:
    """Configuration for re-identification model architecture."""

    # Embedding projection
    hidden_dim: int = 512
    embedding_dim: int = 256
    dropout: float = 0.3

    # ArcFace loss
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    source: Literal["disk", "huggingface", "fiftyone"] = "fiftyone"

    # For disk-based datasets
    data_dir: Path | None = None
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    # For HuggingFace datasets
    hf_repo: str | None = None
    hf_revision: str | None = None

    # For FiftyOne datasets
    fo_dataset_name: str = "JID_Master_Dataset"
    fo_patches_field: str = "sam3_segmentations"
    fo_embeddings_field: str = "embeddings"
    fo_label_field: str = "ground_truth"
    fo_split_field: str = "split"

    # Common settings
    image_field: str = "filepath"
    label_field: str = "label"
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50

    # Loss function
    loss_name: Literal[
        "cross_entropy",
        "label_smoothing",
        "focal",
        "multi_margin",
        "nll",
        "arcface",
        "subcenter_arcface",
        "triplet",
        "arcface_triplet",
        "random_baseline",
        "backbone_only",
    ] = "arcface"
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha: float | None = None
    multi_margin_p: int = 1
    multi_margin_margin: float = 1.0

    # Triplet loss config
    triplet_margin: float = 0.3
    triplet_mining: Literal["all", "hard", "semi-hard"] = "hard"
    triplet_distance: Literal["euclidean", "cosine"] = "euclidean"
    triplet_weight: float = 0.5  # Weight when combined with classification loss

    # PK Sampler config (for triplet loss)
    use_pk_sampler: bool = False
    pk_p: int = 8  # Number of classes per batch
    pk_k: int = 4  # Number of samples per class

    # Scheduler
    scheduler_type: Literal["reduce_on_plateau", "cosine", "step"] = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: Literal["val_loss", "val_map"] = "val_map"
    early_stopping_mode: Literal["min", "max"] = "max"

    # Checkpointing
    save_dir: Path = Path("data/models/reidentification")
    checkpoint_frequency: int = 5
    save_best_only: bool = True

    # Device
    device: str = "cuda"  # Will auto-detect if cuda available


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    # Metrics
    compute_map: bool = True
    compute_cmc: bool = True
    cmc_top_k: list[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # Output
    output_dir: Path = Path("data/results/reidentification")
    save_embeddings: bool = True
    save_predictions: bool = True

    # FiftyOne integration
    add_to_fiftyone: bool = False
    fo_dataset_name: str | None = None
    fo_predictions_field: str = "reid_predictions"
    fo_embeddings_field: str = "reid_embeddings"


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    enabled: bool = True
    entity: str | None = None
    project: str = "camera-trap-reidentification"
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None

    # Logging
    log_frequency: int = 10  # Log every N batches
    log_gradients: bool = False
    log_model: bool = True


@dataclass
class ReidentificationConfig:
    """Complete configuration for re-identification pipeline."""

    # Sub-configs
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    baseline_mode: str | None = None  # "random_baseline" or "backbone_only" for baseline evaluation modes

    # Runtime
    seed: int = 42
    verbose: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure paths are Path objects
        if self.dataset.data_dir is not None and not isinstance(self.dataset.data_dir, Path):
            self.dataset.data_dir = Path(self.dataset.data_dir)

        if not isinstance(self.training.save_dir, Path):
            self.training.save_dir = Path(self.training.save_dir)

        if not isinstance(self.evaluation.output_dir, Path):
            self.evaluation.output_dir = Path(self.evaluation.output_dir)

        # Create directories
        self.training.save_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation.output_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> ReidentificationConfig:
    """Get default configuration for re-identification."""
    return ReidentificationConfig()


def load_config_from_dict(config_dict: dict[str, Any]) -> ReidentificationConfig:
    """Load configuration from dictionary."""

    # Recursively convert nested dicts to dataclasses
    def _nested_dataclass(cls: type, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}  # type: ignore
        kwargs = {}
        for key, value in data.items():
            if key in fieldtypes:
                field_type = fieldtypes[key]
                # Check if field type is a dataclass
                if hasattr(field_type, "__dataclass_fields__"):
                    kwargs[key] = _nested_dataclass(field_type, value)
                else:
                    kwargs[key] = value
        return cls(**kwargs)

    return _nested_dataclass(ReidentificationConfig, config_dict)  # type: ignore
