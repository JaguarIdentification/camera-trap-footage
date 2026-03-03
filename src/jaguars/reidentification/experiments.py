"""Experiment configurations for jaguar re-identification.

This module defines experiment configurations for systematic evaluation
of different components: backbones, losses, datasets, and hyperparameters.
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
    requires_pk_sampling: bool


class DatasetSpec(TypedDict):
    """Type specification for dataset variant configuration."""

    name: str
    description: str
    dataset_name: str
    patches_field: str | None
    embeddings_field: str | None


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
    group: str | None = None  # Wandb group for organization


def get_backbone_experiments(base_config: ReidentificationConfig | None = None) -> list[ExperimentConfig]:
    """Backbone Comparison Experiments.

    Which backbone gives best mAP-efficiency tradeoff?
    Test different backbones with fixed ArcFace loss.
    """
    if base_config is None:
        base_config = get_default_config()
        base_config.wandb.enabled = True
        base_config.wandb.entity = "jaguars"
        base_config.wandb.project = "camera-trap-reidentification"
        base_config.wandb.tags = ["backbones"]
        base_config.training.num_epochs = 50

        # Dataset settings
        base_config.dataset.source = "fiftyone"
        base_config.dataset.fo_dataset_name = "JID_Master_Dataset"
        base_config.dataset.fo_split_field = "closed_set_split"
        base_config.dataset.fo_patches_field = "sam3_segmentations"
        base_config.dataset.fo_label_field = "ground_truth"

        # Fixed loss: ArcFace
        base_config.training.loss_name = "arcface"
        base_config.model.arcface_margin = 0.5
        base_config.model.arcface_scale = 64.0

    backbones: list[BackboneSpec] = [
        # DINOv3 variants (timm versions)
        BackboneSpec(
            name="vit_large_patch16_dinov3.lvd1689m",
            embedding_dim=1024,
            description="DINOv3 Large (1024-dim, latest self-supervised)",
        ),
        BackboneSpec(
            name="vit_base_patch16_dinov3.lvd1689m",
            embedding_dim=768,
            description="DINOv3 Base (768-dim, efficient high-quality)",
        ),
        # MiewID wildlife re-ID models
        BackboneSpec(
            name="conservationxlabs/miewid-msv2",
            embedding_dim=2152,
            description="MiewID-MSv2 (wildlife re-ID specialist)",
        ),
        BackboneSpec(
            name="conservationxlabs/miewid-msv3",
            embedding_dim=2152,
            description="MiewID-MSv3 (wildlife re-ID specialist)",
        ),
        # DINOv2 variants (timm versions)
        BackboneSpec(
            name="vit_large_patch14_dinov2.lvd142m",
            embedding_dim=1024,
            description="DINOv2 Large (1024-dim, best quality)",
        ),
        BackboneSpec(
            name="vit_base_patch14_dinov2.lvd142m",
            embedding_dim=768,
            description="DINOv2 Base (768-dim, balanced)",
        ),
        BackboneSpec(
            name="vit_small_patch14_dinov2.lvd142m",
            embedding_dim=384,
            description="DINOv2 Small (384-dim, fast)",
        ),
        # MegaDescriptor animal re-ID models
        BackboneSpec(
            name="hf-hub:BVRA/MegaDescriptor-L-384",
            embedding_dim=1536,
            description="MegaDescriptor Large 384 (animal re-ID specialist)",
        ),
        BackboneSpec(
            name="hf-hub:BVRA/MegaDescriptor-B-224",
            embedding_dim=768,
            description="MegaDescriptor Base 224 (animal re-ID specialist)",
        ),
        # CNN baselines
        BackboneSpec(
            name="resnet50",
            embedding_dim=2048,
            description="ResNet50 (25M params, CNN baseline)",
        ),
        BackboneSpec(
            name="convnext_base",
            embedding_dim=1024,
            description="ConvNeXt Base (modern CNN)",
        ),
        BackboneSpec(
            name="convnextv2_base.fcmae_ft_in22k_in1k",
            embedding_dim=1024,
            description="ConvNeXtV2 Base (88M params, modern CNN v2)",
        ),
        BackboneSpec(
            name="efficientnet_b3",
            embedding_dim=1536,
            description="EfficientNet B3 (efficient CNN)",
        ),
        BackboneSpec(
            name="hf-hub:timm/efficientnetv2_rw_m.agc_in1k",
            embedding_dim=2152,
            description="EfficientNetV2-RW-M (efficient CNN v2)",
        ),
    ]

    import copy

    experiments = []
    for backbone in backbones:
        config = copy.deepcopy(base_config)
        config.wandb.run_name = backbone["name"]
        config.backbone.name = backbone["name"]
        config.backbone.embedding_dim = backbone["embedding_dim"]
        config.backbone.pretrained = True

        # Set appropriate input size based on model
        if "dinov2" in backbone["name"]:
            config.backbone.input_size = 518
        elif "dinov3" in backbone["name"]:
            config.backbone.input_size = 512
        elif "miewid" in backbone["name"].lower():
            config.backbone.input_size = 440
        elif "megadescriptor" in backbone["name"].lower():
            if "384" in backbone["name"]:
                config.backbone.input_size = 384
            elif "224" in backbone["name"]:
                config.backbone.input_size = 224
            else:
                config.backbone.input_size = 384  # Default for MegaDescriptor
        elif "convnextv2" in backbone["name"].lower():
            config.backbone.input_size = 224
        elif "resnet" in backbone["name"] or "convnext" in backbone["name"]:
            config.backbone.input_size = 224
        elif "efficientnetv2" in backbone["name"].lower():
            config.backbone.input_size = 320
        elif "efficientnet_b3" in backbone["name"]:
            config.backbone.input_size = 288
        else:
            config.backbone.input_size = 224  # Default

        # Always compute features for fair backbone comparison
        config.dataset.fo_embeddings_field = None

        experiments.append(
            ExperimentConfig(
                name=f"backbone_{backbone['name'].replace('/', '_')}",
                description=backbone["description"],
                base_config=config,
                group="backbone_comparison",
            )
        )

    return experiments


def get_loss_experiments(base_config: ReidentificationConfig | None = None) -> list[ExperimentConfig]:
    """Loss Function Comparison Experiments.

    Which loss function is most effective for jaguar re-ID?
    Tests ArcFace variants, SubCenter, Triplet, Combined, Cross-Entropy, Focal.
    Split into standard batching and PK sampling groups.

    Args:
        base_config: Optional base configuration to use. If None, creates a default config
                     with standard dataset/backbone settings. Pass a config from your notebook
                     to preserve wandb settings, paths, etc.
    """
    if base_config is None:
        base_config = get_default_config()

        # Wandb settings
        base_config.wandb.enabled = True
        base_config.wandb.entity = "jaguars"
        base_config.wandb.project = "camera-trap-reidentification"
        base_config.training.num_epochs = 50

        # Dataset settings
        base_config.dataset.source = "fiftyone"
        base_config.dataset.fo_dataset_name = "JID_Master_Dataset"
        base_config.dataset.fo_split_field = "closed_set_split"
        base_config.dataset.fo_patches_field = "sam3_segmentations"
        base_config.dataset.fo_label_field = "ground_truth"
        base_config.dataset.fo_embeddings_field = "embeddings_BVRA_MegaDescriptor_L_384"

        # Backbone settings
        base_config.backbone.name = "vit_large_patch14_dinov2.lvd142m"
        base_config.backbone.pretrained = True
        base_config.backbone.input_size = 518
        base_config.backbone.embedding_dim = 1024

    # Standard batching losses (random sampling)
    standard_losses: list[LossSpec] = [
        # ArcFace variants (recommended for re-ID)
        LossSpec(
            name="arcface",
            description="ArcFace standard margin (m=0.5, s=64)",
            config_updates={"training": {"loss_name": "arcface"}, "model": {"arcface_margin": 0.5, "arcface_scale": 64.0}},
            requires_pk_sampling=False,
        ),
        LossSpec(
            name="arcface_soft",
            description="ArcFace soft margin (m=0.3, s=64)",
            config_updates={"training": {"loss_name": "arcface"}, "model": {"arcface_margin": 0.3, "arcface_scale": 64.0}},
            requires_pk_sampling=False,
        ),
        LossSpec(
            name="arcface_hard",
            description="ArcFace hard margin (m=0.7, s=64)",
            config_updates={"training": {"loss_name": "arcface"}, "model": {"arcface_margin": 0.7, "arcface_scale": 64.0}},
            requires_pk_sampling=False,
        ),
        LossSpec(
            name="subcenter_arcface",
            description="SubCenter ArcFace (handles intra-class variation)",
            config_updates={"training": {"loss_name": "subcenter_arcface"}, "model": {"arcface_margin": 0.5, "arcface_scale": 30.0}},
            requires_pk_sampling=False,
        ),
        # Baselines for comparison
        LossSpec(
            name="cross_entropy",
            description="Cross-Entropy (classification baseline)",
            config_updates={"training": {"loss_name": "cross_entropy"}},
            requires_pk_sampling=False,
        ),
        LossSpec(
            name="focal",
            description="Focal Loss (handles class imbalance)",
            config_updates={"training": {"loss_name": "focal", "focal_gamma": 2.0}},
            requires_pk_sampling=False,
        ),
    ]

    # PK sampling losses (structured batching)
    pk_sampling_losses: list[LossSpec] = [
        # Combined ArcFace + Triplet
        LossSpec(
            name="arcface_triplet",
            description="ArcFace + Triplet (PK sampling, P=8, K=4)",
            config_updates={
                "training": {
                    "loss_name": "arcface_triplet",
                    "triplet_weight": 0.5,
                    "triplet_mining": "hard",
                    "use_pk_sampler": True,
                    "pk_p": 8,
                    "pk_k": 4,
                }
            },
            requires_pk_sampling=True,
        ),
        # Triplet-only
        LossSpec(
            name="triplet_hard",
            description="Triplet with hard mining (PK sampling, P=8, K=4)",
            config_updates={
                "training": {
                    "loss_name": "triplet",
                    "triplet_margin": 0.3,
                    "triplet_mining": "hard",
                    "use_pk_sampler": True,
                    "pk_p": 8,
                    "pk_k": 4,
                }
            },
            requires_pk_sampling=True,
        ),
        LossSpec(
            name="triplet_semi_hard",
            description="Triplet with semi-hard mining (PK sampling, P=8, K=4)",
            config_updates={
                "training": {
                    "loss_name": "triplet",
                    "triplet_margin": 0.3,
                    "triplet_mining": "semi-hard",
                    "use_pk_sampler": True,
                    "pk_p": 8,
                    "pk_k": 4,
                }
            },
            requires_pk_sampling=True,
        ),
    ]

    experiments = []

    # Standard batching experiments
    for loss in standard_losses:
        # Use deepcopy of base_config to inherit all settings (dataset, backbone, etc.)
        import copy

        config = copy.deepcopy(base_config)
        config.dataset.use_pk_sampling = False
        config.wandb.run_name = loss["name"]
        config.wandb.tags.append("standard_batching")

        # Apply config updates
        config_updates = loss["config_updates"]
        for key, value in config_updates.items():
            if key == "model":
                for k, v in value.items():
                    setattr(config.model, k, v)
            if key == "training":
                for k, v in value.items():
                    setattr(config.training, k, v)

        experiments.append(
            ExperimentConfig(
                name=f"loss_{loss['name']}",
                description=loss["description"],
                base_config=config,
                group="loss_comparison_standard",
            )
        )

    # PK sampling experiments
    for loss in pk_sampling_losses:
        import copy

        config = copy.deepcopy(base_config)
        config.dataset.use_pk_sampling = True
        config.dataset.p_classes = 8
        config.dataset.k_samples_per_class = 4
        config.wandb.run_name = loss["name"]
        config.wandb.tags.append("pk_sampling")

        # Apply config updates
        config_updates = loss["config_updates"]
        for key, value in config_updates.items():
            if key == "model":
                for k, v in value.items():
                    setattr(config.model, k, v)
            if key == "training":
                for k, v in value.items():
                    setattr(config.training, k, v)

        experiments.append(
            ExperimentConfig(
                name=f"loss_{loss['name']}_pk",
                description=loss["description"],
                base_config=config,
                group="loss_comparison_pk",
            )
        )

    return experiments


def get_dataset_experiments(base_config: ReidentificationConfig | None = None) -> list[ExperimentConfig]:
    """Dataset Variant Comparison Experiments.

    Which dataset processing yields best re-ID performance?
    Tests different combinations of segmentation and deduplication.
    """
    if base_config is None:
        base_config = get_default_config()
        base_config.wandb.enabled = True
        base_config.wandb.entity = "jaguars"
        base_config.wandb.project = "camera-trap-reidentification"
        base_config.wandb.tags = ["datasets"]
        base_config.training.num_epochs = 50

        # Fixed backbone
        base_config.backbone.name = "vit_large_patch14_dinov2.lvd142m"
        base_config.backbone.embedding_dim = 1024
        base_config.backbone.pretrained = True
        base_config.backbone.input_size = 518

        # Fixed loss
        base_config.training.loss_name = "arcface"
        base_config.model.arcface_margin = 0.5
        base_config.model.arcface_scale = 64.0

        # FiftyOne common settings
        base_config.dataset.source = "fiftyone"
        base_config.dataset.fo_split_field = "closed_set_split"
        base_config.dataset.fo_label_field = "ground_truth"

    datasets: list[DatasetSpec] = [
        DatasetSpec(
            name="master",
            description="Master dataset (segmented, deduplicated)",
            dataset_name="JID_Master_Dataset",
            patches_field="sam3_segmentations",
            embeddings_field="embeddings_BVRA_MegaDescriptor_L_384",
        ),
        DatasetSpec(
            name="segmented_deduplicated",
            description="Segmented and deduplicated",
            dataset_name="JID_Segmented_Deduplicated",
            patches_field="sam3_segmentations",
            embeddings_field="embeddings_BVRA_MegaDescriptor_L_384",
        ),
        DatasetSpec(
            name="segmented",
            description="Segmented only (with duplicates)",
            dataset_name="JID_Segmented",
            patches_field="sam3_segmentations",
            embeddings_field="embeddings_BVRA_MegaDescriptor_L_384",
        ),
        DatasetSpec(
            name="not_segmented_deduplicated",
            description="Not segmented, deduplicated (full frames)",
            dataset_name="JID_Not_Segmented_Deduplicated",
            patches_field=None,
            embeddings_field=None,
        ),
        DatasetSpec(
            name="not_segmented",
            description="Not segmented (full frames with duplicates)",
            dataset_name="JID_Not_Segmented",
            patches_field=None,
            embeddings_field=None,
        ),
    ]

    import copy

    experiments = []
    for dataset in datasets:
        config = copy.deepcopy(base_config)
        config.dataset.fo_dataset_name = dataset["dataset_name"]
        config.dataset.fo_patches_field = dataset["patches_field"]
        config.dataset.fo_embeddings_field = dataset["embeddings_field"]
        config.wandb.run_name = dataset["name"]

        experiments.append(
            ExperimentConfig(
                name=f"dataset_{dataset['name']}",
                description=dataset["description"],
                base_config=config,
                group="dataset_comparison",
            )
        )

    return experiments


def get_hyperparameter_sweep_config(base_config: ReidentificationConfig | None = None) -> dict[str, Any]:
    """Hyperparameter Sweep Configuration for Wandb.

    Bayesian search over critical hyperparameters using best model from previous experiments.
    Use this after identifying best backbone, loss, and dataset.

    Returns:
        Wandb sweep configuration dict for wandb.sweep()
    """
    if base_config is None:
        base_config = get_default_config()
        base_config.wandb.enabled = True
        base_config.wandb.entity = "jaguars"
        base_config.wandb.project = "camera-trap-reidentification"
        base_config.wandb.tags = ["hyperparameters"]

        # Use best performing configuration from prior experiments
        base_config.backbone.name = "vit_large_patch14_dinov2.lvd142m"
        base_config.backbone.embedding_dim = 1024
        base_config.backbone.input_size = 518
        base_config.training.loss_name = "arcface"

        base_config.dataset.source = "fiftyone"
        base_config.dataset.fo_dataset_name = "JID_Master_Dataset"
        base_config.dataset.fo_split_field = "closed_set_split"
        base_config.dataset.fo_patches_field = "sam3_segmentations"
        base_config.dataset.fo_label_field = "ground_truth"
        base_config.dataset.fo_embeddings_field = "embeddings_BVRA_MegaDescriptor_L_384"

    sweep_config = {
        "name": "jaguar_reid_hyperparam_sweep",
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "validation/map", "goal": "maximize"},
        "parameters": {
            # Learning rate
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3,
            },
            # ArcFace margin
            "arcface_margin": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.7,
            },
            # ArcFace scale
            "arcface_scale": {
                "values": [30.0, 64.0, 128.0],
            },
            # Batch size
            "batch_size": {
                "values": [16, 32, 64],
            },
            # Weight decay
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2,
            },
            # Dropout rate
            "dropout": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.5,
            },
            # Warmup epochs
            "warmup_epochs": {
                "values": [0, 5, 10],
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10,
        },
    }

    return sweep_config


def get_optimizer_experiments(base_config: ReidentificationConfig | None = None) -> list[ExperimentConfig]:
    """Experiment 5: Optimizer + Scheduler Study.

    Which optimizer + LR schedule is most stable and accurate?
    Test: AdamW, SGD × (Cosine, OneCycle, ReduceOnPlateau).
    """
    if base_config is None:
        base_config = get_default_config()
        base_config.wandb.enabled = True
        base_config.wandb.entity = "jaguars"
        base_config.wandb.project = "camera-trap-reidentification"
        base_config.wandb.tags = ["optimizers"]
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

    import copy

    experiments = []
    for opt_config in configs:
        config = copy.deepcopy(base_config)
        config.training.learning_rate = opt_config["lr"]
        # Type: ignore because scheduler value is validated at runtime
        config.training.scheduler_type = opt_config["scheduler"]  # type: ignore[assignment]
        config.wandb.run_name = opt_config["name"]

        experiments.append(
            ExperimentConfig(
                name=f"optimizer_{opt_config['name']}",
                description=opt_config["description"],
                base_config=config,
                variations={"optimizer": opt_config["optimizer"]},
            )
        )

    return experiments


def get_augmentation_experiments(base_config: ReidentificationConfig | None = None) -> list[ExperimentConfig]:
    """Experiment 7: Data Augmentation Study.

    Which augmentations improve identity invariance?
    Test: Flipping, Rotation, Scaling, Color jitter, Grayscale, etc.
    """
    if base_config is None:
        base_config = get_default_config()
        base_config.wandb.enabled = True
        base_config.wandb.entity = "jaguars"
        base_config.wandb.project = "camera-trap-reidentification"
        base_config.wandb.tags = ["augmentations"]

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

    import copy

    experiments = []
    for aug_config in augmentation_configs:
        config = copy.deepcopy(base_config)
        config.wandb.run_name = aug_config["name"]

        experiments.append(
            ExperimentConfig(
                name=f"augmentation_{aug_config['name']}",
                description=aug_config["description"],
                base_config=config,
                variations={"augmentations": aug_config["augmentations"]},
            )
        )

    return experiments


def get_seed_stability_experiments(base_config: ReidentificationConfig | None = None) -> list[ExperimentConfig]:
    """Experiment 9: Seed Stability Study.

    How stable is training across different random seeds?
    Run best configuration with 10 different seeds.
    """
    if base_config is None:
        base_config = get_default_config()
        base_config.wandb.enabled = True
        base_config.wandb.entity = "jaguars"
        base_config.wandb.project = "camera-trap-reidentification"
        base_config.wandb.tags = ["seeds"]
        base_config.training.num_epochs = 50

    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 5926, 8192, 16384]

    import copy

    experiments = [
        ExperimentConfig(
            name="seed_stability_best_model",
            description="Best model configuration across 10 seeds",
            base_config=copy.deepcopy(base_config),
            seeds=seeds,
        )
    ]

    return experiments


def get_embedding_dimension_experiments(base_config: ReidentificationConfig | None = None) -> list[ExperimentConfig]:
    """Additional: Embedding Dimension Study.

    What's the optimal embedding dimension for re-identification?
    """
    if base_config is None:
        base_config = get_default_config()
        base_config.wandb.enabled = True
        base_config.wandb.entity = "jaguars"
        base_config.wandb.project = "camera-trap-reidentification"
        base_config.wandb.tags = ["embedding_dimensions"]

    dimensions = [64, 128, 256, 512, 1024]

    import copy

    experiments = []
    for dim in dimensions:
        config = copy.deepcopy(base_config)
        config.model.embedding_dim = dim
        config.wandb.run_name = f"embedding_dim_{dim}"

        experiments.append(
            ExperimentConfig(
                name=f"embedding_dim_{dim}",
                description=f"Embedding dimension: {dim}",
                base_config=config,
            )
        )

    return experiments


def get_baseline_experiments(base_config: ReidentificationConfig | None = None) -> list[ExperimentConfig]:
    """Baseline Experiments.

    Simple baselines to compare against: random embeddings and backbone-only (no training).

    Args:
        base_config: Optional base configuration to use. If None, creates default config.
    """
    if base_config is None:
        base_config = get_default_config()
        base_config.wandb.enabled = True
        base_config.wandb.entity = "jaguars"
        base_config.wandb.project = "camera-trap-reidentification"
        base_config.wandb.tags = ["baselines"]
        base_config.training.num_epochs = 0  # No training for baselines

        # Dataset settings
        base_config.dataset.source = "fiftyone"
        base_config.dataset.fo_dataset_name = "JID_Master_Dataset"
        base_config.dataset.fo_split_field = "closed_set_split"
        base_config.dataset.fo_patches_field = "sam3_segmentations"
        base_config.dataset.fo_label_field = "ground_truth"
        base_config.dataset.fo_embeddings_field = "embeddings_BVRA_MegaDescriptor_L_384"

        # Backbone settings
        base_config.backbone.name = "vit_large_patch14_dinov2.lvd142m"
        base_config.backbone.pretrained = True
        base_config.backbone.input_size = 518  # DINOv2 uses 518
        base_config.backbone.embedding_dim = 1024  # DINOv2-Large is 1024, not 1536

    import copy

    experiments = []

    # Random baseline
    random_config = copy.deepcopy(base_config)
    random_config.baseline_mode = "random_baseline"
    random_config.wandb.run_name = "baseline_random"
    experiments.append(
        ExperimentConfig(
            name="baseline_random",
            description="Random embeddings (sanity check - should be near 0% mAP)",
            base_config=random_config,
            group="baselines",
        )
    )

    # Backbone-only baselines for each backbone
    backbones: list[BackboneSpec] = [
        # MiewID wildlife re-ID models
        BackboneSpec(
            name="conservationxlabs/miewid-msv2",
            embedding_dim=2152,
            description="MiewID-MSv2",
        ),
        BackboneSpec(
            name="conservationxlabs/miewid-msv3",
            embedding_dim=2152,
            description="MiewID-MSv3",
        ),
        # MegaDescriptor animal re-ID models (our main models)
        BackboneSpec(
            name="hf-hub:BVRA/MegaDescriptor-L-384",
            embedding_dim=1536,
            description="BVRA MegaDescriptor Large 384",
        ),
        BackboneSpec(
            name="hf-hub:BVRA/MegaDescriptor-B-224",
            embedding_dim=768,
            description="BVRA MegaDescriptor Base 224",
        ),
        # DINOv2 models
        BackboneSpec(
            name="vit_large_patch14_dinov2.lvd142m",
            embedding_dim=1024,
            description="DINOv2 Large",
        ),
        BackboneSpec(
            name="vit_base_patch14_dinov2.lvd142m",
            embedding_dim=768,
            description="DINOv2 Base",
        ),
        BackboneSpec(
            name="vit_small_patch14_dinov2.lvd142m",
            embedding_dim=384,
            description="DINOv2 Small",
        ),
        # CNN models
        BackboneSpec(
            name="resnet50",
            embedding_dim=2048,
            description="ResNet50",
        ),
        BackboneSpec(
            name="convnext_base",
            embedding_dim=1024,
            description="ConvNeXt Base",
        ),
        BackboneSpec(
            name="convnextv2_base.fcmae_ft_in22k_in1k",
            embedding_dim=1024,
            description="ConvNeXtV2 Base",
        ),
        BackboneSpec(
            name="efficientnet_b3",
            embedding_dim=1536,
            description="EfficientNet B3",
        ),
        BackboneSpec(
            name="hf-hub:timm/efficientnetv2_rw_m.agc_in1k",
            embedding_dim=2152,
            description="EfficientNetV2-RW-M",
        ),
        # DINOv3 models (latest self-supervised ViT)
        BackboneSpec(
            name="vit_large_patch16_dinov3.lvd1689m",
            embedding_dim=1024,
            description="DINOv3 Large (1024-dim, latest self-supervised)",
        ),
        BackboneSpec(
            name="vit_base_patch16_dinov3.lvd1689m",
            embedding_dim=768,
            description="DINOv3 Base (768-dim, efficient high-quality)",
        ),
    ]

    for backbone in backbones:
        config = copy.deepcopy(base_config)
        config.backbone.name = backbone["name"]
        config.backbone.embedding_dim = backbone["embedding_dim"]
        config.baseline_mode = "backbone_only"  # type: ignore[assignment]
        config.wandb.run_name = backbone["name"]
        

        # Set appropriate input size based on model
        if "dinov2" in backbone["name"]:
            config.backbone.input_size = 518
        elif "dinov3" in backbone["name"]:
            config.backbone.input_size = 512
        elif "miewid" in backbone["name"].lower():
            config.backbone.input_size = 440
        elif "megadescriptor" in backbone["name"].lower():
            if "384" in backbone["name"]:
                config.backbone.input_size = 384
            elif "224" in backbone["name"]:
                config.backbone.input_size = 224
            else:
                config.backbone.input_size = 384
        elif "convnextv2" in backbone["name"].lower():
            config.backbone.input_size = 224
        elif "resnet" in backbone["name"] or "convnext" in backbone["name"]:
            config.backbone.input_size = 224
        elif "efficientnetv2" in backbone["name"].lower():
            config.backbone.input_size = 320
        elif "efficientnet_b3" in backbone["name"]:
            config.backbone.input_size = 288
        else:
            config.backbone.input_size = 224  # Default

        # Update embeddings field if needed
        # Only MegaDescriptor-L-384 has precomputed embeddings currently
        if "megadescriptor" in backbone["name"].lower() and "l" in backbone["name"].lower() and "384" in backbone["name"]:
            config.dataset.fo_embeddings_field = "embeddings_BVRA_MegaDescriptor_L_384"
        else:
            config.dataset.fo_embeddings_field = None  # Will compute on the fly

        experiments.append(
            ExperimentConfig(
                name=f"baseline_{backbone['name'].replace('/', '_')}",
                description=f"{backbone['description']} without fine-tuning",
                base_config=config,
                group="baselines",
            )
        )

    return experiments


def get_all_experiments() -> dict[str, list[ExperimentConfig]]:
    """Get all experiment configurations organized by category.

    Returns:
        Dict with experiment categories: baseline, backbone, loss, dataset.
        Use get_hyperparameter_sweep_config() separately for Wandb sweeps.
    """
    return {
        "baseline": get_baseline_experiments(),
        "backbone": get_backbone_experiments(),
        "loss": get_loss_experiments(),
        "dataset": get_dataset_experiments(),
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
        "tags": experiment.base_config.wandb.tags,
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
