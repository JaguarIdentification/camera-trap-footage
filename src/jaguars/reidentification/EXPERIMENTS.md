# Re-identification Experiments

This directory contains experiment configurations and scripts for systematic evaluation of the jaguar re-identification system.

## Overview

The experiment framework supports systematic comparison of:
- **Backbones**: Different feature extraction architectures
- **Loss functions**: Various metric learning objectives
- **Optimizers**: Training optimization strategies
- **Augmentations**: Data augmentation techniques
- **Embedding dimensions**: Optimal feature space size
- **Seed stability**: Reproducibility across random seeds

## Quick Start

### Run All Experiments

```bash
python -m jaguars.reidentification.run_experiments --output-dir experiments/outputs
```

### Run Specific Category

```bash
# Backbone comparison
python -m jaguars.reidentification.run_experiments --category backbone

# Loss function comparison
python -m jaguars.reidentification.run_experiments --category loss

# Optimizer study
python -m jaguars.reidentification.run_experiments --category optimizer

# Augmentation study
python -m jaguars.reidentification.run_experiments --category augmentation

# Seed stability
python -m jaguars.reidentification.run_experiments --category seed_stability

# Embedding dimension
python -m jaguars.reidentification.run_experiments --category embedding_dim
```

### Dry Run (Preview Experiments)

```bash
python -m jaguars.reidentification.run_experiments --dry-run
```

### Resume from Specific Experiment

```bash
python -m jaguars.reidentification.run_experiments --resume-from backbone_resnet50
```

## Experiment Categories

### 1. Backbone Comparison (`backbone`)

**Research Question**: Which backbone gives the best mAP-efficiency tradeoff?

**Experiments** (5 total):
- `backbone_vit_small_patch14_dinov2.lvd142m`: DINOv2 Small (22M params, fast)
- `backbone_vit_base_patch14_dinov2.lvd142m`: DINOv2 Base (86M params, balanced)
- `backbone_vit_large_patch14_dinov2.lvd142m`: MegaDescriptor Large (304M params, accurate)
- `backbone_resnet50`: ResNet50 (25M params, baseline CNN)
- `backbone_efficientnet_b0`: EfficientNet-B0 (5M params, efficient)

**Metrics**: mAP@5, CMC@1/5/10, inference time, memory usage

### 2. Loss Function Comparison (`loss`)

**Research Question**: Which loss function best fits jaguar re-identification?

**Experiments** (4 total):
- `loss_arcface`: Standard ArcFace (margin=0.5, scale=64)
- `loss_arcface_cosface_hybrid`: Hybrid ArcFace + CosFace
- `loss_large_margin_arcface`: ArcFace with larger margin (0.7)
- `loss_small_margin_arcface`: ArcFace with smaller margin (0.3)

**Future extensions**: Triplet, Center, Contrastive, Focal, Circle losses

**Metrics**: mAP@5, CMC curves, convergence speed

### 3. Optimizer Study (`optimizer`)

**Research Question**: Which optimizer + LR schedule is most stable and accurate?

**Experiments** (5 total):
- `optimizer_adamw_cosine`: AdamW + Cosine Annealing (lr=1e-4)
- `optimizer_adamw_plateau`: AdamW + ReduceLROnPlateau (lr=1e-4)
- `optimizer_sgd_cosine`: SGD + Cosine Annealing (lr=1e-2)
- `optimizer_sgd_plateau`: SGD + ReduceLROnPlateau (lr=1e-2)
- `optimizer_adamw_high_lr`: AdamW with higher LR (lr=5e-4)

**Metrics**: Final mAP, training stability, convergence speed

### 4. Augmentation Study (`augmentation`)

**Research Question**: Which augmentations improve identity invariance?

**Experiments** (5 total):
- `augmentation_no_augmentation`: Baseline without augmentation
- `augmentation_flip_only`: Horizontal flip only
- `augmentation_geometric`: Flip + rotation + scaling
- `augmentation_color`: Color jitter + grayscale
- `augmentation_all_augmentations`: All augmentations combined

**Metrics**: mAP@5, robustness to pose/lighting variations

### 5. Seed Stability Study (`seed_stability`)

**Research Question**: How stable is training across different random seeds?

**Experiments** (1 configuration, 10 seeds):
- `seed_stability_best_model`: Best model across seeds [42, 123, 456, 789, 1024, 2048, 3141, 5926, 8192, 16384]

**Metrics**: Mean ± std mAP, variance in embeddings

### 6. Embedding Dimension Study (`embedding_dim`)

**Research Question**: What's the optimal embedding dimension for re-identification?

**Experiments** (5 total):
- `embedding_dim_64`: 64-dimensional embeddings
- `embedding_dim_128`: 128-dimensional embeddings
- `embedding_dim_256`: 256-dimensional embeddings
- `embedding_dim_512`: 512-dimensional embeddings
- `embedding_dim_1024`: 1024-dimensional embeddings

**Metrics**: mAP vs dimension, memory usage, inference speed

## Experiment Structure

Each experiment run creates the following structure:

```
experiments/outputs/
├── {experiment_name}/
│   └── seed_{seed}/
│       ├── config.json              # Full experiment configuration
│       ├── checkpoints/             # Model checkpoints
│       │   ├── best_model.pth
│       │   └── last_model.pth
│       ├── embeddings/              # Saved embeddings
│       │   ├── train_embeddings.npz
│       │   └── val_embeddings.npz
│       ├── wandb/                   # WandB logs
│       └── results.json             # Final metrics
```

## Configuration Files

Experiment configurations are defined in [`experiments.py`](experiments.py):

```python
from jaguars.reidentification.experiments import (
    get_backbone_experiments,
    get_loss_experiments,
    get_optimizer_experiments,
    get_augmentation_experiments,
    get_seed_stability_experiments,
    get_embedding_dimension_experiments,
    get_all_experiments,
)

# Get all experiments
all_experiments = get_all_experiments()

# Get specific category
backbone_experiments = get_backbone_experiments()
```

## WandB Integration

All experiments are logged to Weights & Biases for easy comparison:

- **Projects**: Each category has its own project (e.g., `jaguar-reid-backbones`)
- **Runs**: Named as `{experiment_name}_seed_{seed}`
- **Tags**: Automatic tagging by category and seed
- **Metrics**: mAP, CMC, loss curves, learning rates tracked

View results at: https://wandb.ai/{your_team}/jaguar-reid-*

## Analysis Scripts

After running experiments, analyze results with:

```python
import json
from pathlib import Path

# Load results
results_dir = Path("experiments/outputs")
for exp_dir in results_dir.glob("*/seed_*/results.json"):
    with open(exp_dir) as f:
        results = json.load(f)
        print(f"{exp_dir.parent.parent.name}: mAP = {results['map']:.4f}")
```

## Best Practices

1. **Start with dry run**: Always preview experiments first
2. **Monitor WandB**: Check runs during execution for failures
3. **Use resume**: If interrupted, resume from last experiment
4. **Save outputs**: Keep experiment outputs for post-hoc analysis
5. **Document findings**: Update this README with insights

## Future Experiments

Planned but not yet implemented:

- **Postprocessing**: Query expansion, re-ranking strategies
- **Background robustness**: Cropped vs full frame performance
- **Model fusion**: Ensemble of different backbones
- **Search method**: Exhaustive vs approximate nearest neighbors
- **Temporal information**: Video-based re-ID with temporal aggregation

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```python
config.training.batch_size = 16  # Default is 32
```

### Slow Training

Use smaller backbone or fewer epochs:
```python
config.backbone.name = "efficientnet_b0"  # Lightweight
config.training.num_epochs = 20  # Fewer epochs
```

### WandB Issues

Disable WandB for debugging:
```python
config.wandb.enabled = False
```

## Contact

For questions or issues, refer to the main project README or ARCHITECTURE.md.
