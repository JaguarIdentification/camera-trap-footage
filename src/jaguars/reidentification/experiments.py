"""Experiment configurations for jaguar re-identification.

This module defines experiment configurations for systematic evaluation
of different components: backbones, losses, optimizers, augmentations, etc.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from jaguars.reidentification.config import ReidentificationConfig, get_default_config


class BackboneSpec(TypedDict):
    """Type specification for backbone configuration."""

    name: str
    embedding_dim: int
    description: str


class LossSpec(TypedDict):
    """Type specification for loss configuration."""

    name: str
    description: str
    config_updates: dict[str, dict[str, float]]


class OptimizerSpec(TypedDict):
    """Type specification for optimizer configuration."""

    name: str
    description: str
    optimizer: str
    scheduler: str
    lr: float


class AugmentationSpec(TypedDict):
    """Type specification for augmentation configuration."""

    name: str
    description: str
    augmentations: list[str]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    description: str
    base_config: ReidentificationConfig
    variations: dict[str, Any] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=lambda: [42])
    tags: list[str] = field(default_factory=list)


def get_backbone_experiments() -> list[ExperimentConfig]:
    """Experiment 2: Backbone Comparison.
    
    Which backbone gives best mAP-efficiency tradeoff?
    Test 4-5 backbones with same loss, schedule, augmentation, embedding dimension.
    """
    base_config = get_default_config()
    base_config.wandb.enabled = True
    base_config.wandb.project = "jaguar-reid-backbones"
    base_config.training.num_epochs = 50
    
    backbones: list[BackboneSpec] = [
        BackboneSpec(
            name="vit_small_patch14_dinov2.lvd142m",
            embedding_dim=384,
            description="DINOv2 Small (22M params, fast)",
        ),
        BackboneSpec(
            name="vit_base_patch14_dinov2.lvd142m",
            embedding_dim=768,
            description="DINOv2 Base (86M params, balanced)",
        ),
        BackboneSpec(
            name="vit_large_patch14_dinov2.lvd142m",
            embedding_dim=1536,
            description="MegaDescriptor Large (304M params, accurate)",
        ),
        BackboneSpec(
            name="resnet50",
            embedding_dim=2048,
            description="ResNet50 (25M params, baseline CNN)",
        ),
        BackboneSpec(
            name="efficientnet_b0",
            embedding_dim=1280,
            description="EfficientNet-B0 (5M params, efficient)",
        ),
    ]
    
    experiments = []
    for backbone in backbones:
        config = get_default_config()
        config.wandb = base_config.wandb
        config.training = base_config.training
        config.backbone.name = backbone["name"]
        config.backbone.embedding_dim = backbone["embedding_dim"]
        
        experiments.append(
            ExperimentConfig(
                name=f"backbone_{backbone['name']}",
                description=backbone["description"],
                base_config=config,
                tags=["backbone", "comparison"],
            )
        )
    
    return experiments


def get_loss_experiments() -> list[ExperimentConfig]:
    """Experiment 3: Loss Function Comparison.
    
    Which loss best fits jaguar re-ID?
    Test: ArcFace, Cross Entropy, Triplet, Center, Contrastive, Focal, Circle, CosFace.
    """
    base_config = get_default_config()
    base_config.wandb.enabled = True
    base_config.wandb.project = "jaguar-reid-losses"
    base_config.training.num_epochs = 50
    
    losses: list[LossSpec] = [
        LossSpec(
            name="arcface",
            description="ArcFace (additive angular margin)",
            config_updates={"model": {"arcface_margin": 0.5, "arcface_scale": 64.0}},
        ),
        LossSpec(
            name="arcface_cosface_hybrid",
            description="Hybrid ArcFace + CosFace",
            config_updates={"model": {"arcface_margin": 0.3, "arcface_scale": 30.0}},
        ),
        LossSpec(
            name="large_margin_arcface",
            description="ArcFace with larger margin",
            config_updates={"model": {"arcface_margin": 0.7, "arcface_scale": 64.0}},
        ),
        LossSpec(
            name="small_margin_arcface",
            description="ArcFace with smaller margin",
            config_updates={"model": {"arcface_margin": 0.3, "arcface_scale": 64.0}},
        ),
    ]
    
    experiments = []
    for loss in losses:
        config = get_default_config()
        config.wandb = base_config.wandb
        config.training = base_config.training
        
        # Apply config updates
        config_updates = loss["config_updates"]
        for key, value in config_updates.items():
            if key == "model":
                for k, v in value.items():
                    setattr(config.model, k, v)
        
        experiments.append(
            ExperimentConfig(
                name=f"loss_{loss['name']}",
                description=loss["description"],
                base_config=config,
                tags=["loss", "comparison"],
            )
        )
    
    return experiments


def get_optimizer_experiments() -> list[ExperimentConfig]:
    """Experiment 5: Optimizer + Scheduler Study.
    
    Which optimizer + LR schedule is most stable and accurate?
    Test: AdamW, SGD × (Cosine, OneCycle, ReduceOnPlateau).
    """
    base_config = get_default_config()
    base_config.wandb.enabled = True
    base_config.wandb.project = "jaguar-reid-optimizers"
    base_config.training.num_epochs = 50
    
    configs: list[OptimizerSpec] = [
        OptimizerSpec(
            name="adamw_cosine",
            description="AdamW with Cosine Annealing",
            optimizer="adamw",
            scheduler="cosine",
            lr=1e-4,
        ),
        OptimizerSpec(
            name="adamw_plateau",
            description="AdamW with ReduceLROnPlateau",
            optimizer="adamw",
            scheduler="reduce_on_plateau",
            lr=1e-4,
        ),
        OptimizerSpec(
            name="sgd_cosine",
            description="SGD with Cosine Annealing",
            optimizer="sgd",
            scheduler="cosine",
            lr=1e-2,
        ),
        OptimizerSpec(
            name="sgd_plateau",
            description="SGD with ReduceLROnPlateau",
            optimizer="sgd",
            scheduler="reduce_on_plateau",
            lr=1e-2,
        ),
        OptimizerSpec(
            name="adamw_high_lr",
            description="AdamW with higher learning rate",
            optimizer="adamw",
            scheduler="cosine",
            lr=5e-4,
        ),
    ]
    
    experiments = []
    for opt_config in configs:
        config = get_default_config()
        config.wandb = base_config.wandb
        config.training = base_config.training
        config.training.learning_rate = opt_config["lr"]
        # Type: ignore because scheduler value is validated at runtime
        config.training.scheduler_type = opt_config["scheduler"]  # type: ignore[assignment]
        
        experiments.append(
            ExperimentConfig(
                name=f"optimizer_{opt_config['name']}",
                description=opt_config["description"],
                base_config=config,
                variations={"optimizer": opt_config["optimizer"]},
                tags=["optimizer", "scheduler"],
            )
        )
    
    return experiments


def get_augmentation_experiments() -> list[ExperimentConfig]:
    """Experiment 7: Data Augmentation Study.
    
    Which augmentations improve identity invariance?
    Test: Flipping, Rotation, Scaling, Color jitter, Grayscale, etc.
    """
    base_config = get_default_config()
    base_config.wandb.enabled = True
    base_config.wandb.project = "jaguar-reid-augmentation"
    
    augmentation_configs: list[AugmentationSpec] = [
        AugmentationSpec(
            name="no_augmentation",
            description="Baseline without augmentation",
            augmentations=[],
        ),
        AugmentationSpec(
            name="flip_only",
            description="Horizontal flip only",
            augmentations=["horizontal_flip"],
        ),
        AugmentationSpec(
            name="geometric",
            description="Geometric augmentations (flip, rotate, scale)",
            augmentations=["horizontal_flip", "rotation", "scaling"],
        ),
        AugmentationSpec(
            name="color",
            description="Color augmentations (jitter, grayscale)",
            augmentations=["color_jitter", "grayscale"],
        ),
        AugmentationSpec(
            name="all_augmentations",
            description="All augmentations combined",
            augmentations=["horizontal_flip", "rotation", "scaling", "color_jitter", "grayscale"],
        ),
    ]
    
    experiments = []
    for aug_config in augmentation_configs:
        config = get_default_config()
        config.wandb = base_config.wandb
        config.training = base_config.training
        
        experiments.append(
            ExperimentConfig(
                name=f"augmentation_{aug_config['name']}",
                description=aug_config["description"],
                base_config=config,
                variations={"augmentations": aug_config["augmentations"]},
                tags=["augmentation"],
            )
        )
    
    return experiments


def get_seed_stability_experiments() -> list[ExperimentConfig]:
    """Experiment 9: Seed Stability Study.
    
    How stable is training across different random seeds?
    Run best configuration with 10 different seeds.
    """
    base_config = get_default_config()
    base_config.wandb.enabled = True
    base_config.wandb.project = "jaguar-reid-stability"
    base_config.training.num_epochs = 50
    
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 5926, 8192, 16384]
    
    experiments = [
        ExperimentConfig(
            name="seed_stability_best_model",
            description="Best model configuration across 10 seeds",
            base_config=base_config,
            seeds=seeds,
            tags=["stability", "seeds"],
        )
    ]
    
    return experiments


def get_embedding_dimension_experiments() -> list[ExperimentConfig]:
    """Additional: Embedding Dimension Study.
    
    What's the optimal embedding dimension for re-identification?
    """
    base_config = get_default_config()
    base_config.wandb.enabled = True
    base_config.wandb.project = "jaguar-reid-embedding-dim"
    
    dimensions = [64, 128, 256, 512, 1024]
    
    experiments = []
    for dim in dimensions:
        config = get_default_config()
        config.wandb = base_config.wandb
        config.model.embedding_dim = dim
        
        experiments.append(
            ExperimentConfig(
                name=f"embedding_dim_{dim}",
                description=f"Embedding dimension: {dim}",
                base_config=config,
                tags=["embedding_dimension"],
            )
        )
    
    return experiments


def get_all_experiments() -> dict[str, list[ExperimentConfig]]:
    """Get all experiment configurations organized by category."""
    return {
        "backbone": get_backbone_experiments(),
        "loss": get_loss_experiments(),
        "optimizer": get_optimizer_experiments(),
        "augmentation": get_augmentation_experiments(),
        "seed_stability": get_seed_stability_experiments(),
        "embedding_dim": get_embedding_dimension_experiments(),
    }


def save_experiment_config(experiment: ExperimentConfig, output_path: Path) -> None:
    """Save experiment configuration to YAML file."""
    import json
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict for serialization
    config_dict = {
        "name": experiment.name,
        "description": experiment.description,
        "seeds": experiment.seeds,
        "tags": experiment.tags,
        "variations": experiment.variations,
        "base_config": {
            "backbone": experiment.base_config.backbone.__dict__,
            "model": experiment.base_config.model.__dict__,
            "training": experiment.base_config.training.__dict__,
            "dataset": experiment.base_config.dataset.__dict__,
            "wandb": experiment.base_config.wandb.__dict__,
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
