# Re-identification Experiments

Systematic evaluation of the jaguar re-identification system through controlled experiments.

## Overview

The experiment framework supports comparison of:
- **Baselines**: Random and backbone-only baselines
- **Backbones**: Feature extraction architectures (MegaDescriptor, ResNet, ConvNeXt, EfficientNet)
- **Losses**: Metric learning objectives (ArcFace, Triplet, Combined, Cross-Entropy, Focal)
- **Datasets**: Processing variants (segmented/not segmented, deduplicated/with duplicates)
- **Hyperparameters**: Bayesian optimization of best configuration

## Experiment Structure

Each experiment category has its own notebook in `notebooks/`:

1. **`baseline_experiments.ipynb`**: Sanity checks and untrained baselines
2. **`backbone_experiments.ipynb`**: Compare 6 backbone architectures
3. **`loss_experiments.ipynb`**: Compare 9 loss functions (standard + PK sampling)
4. **`dataset_experiments.ipynb`**: Compare 5 dataset processing variants

All experiments log to a **single Wandb project** (`camerate-trap-reidentification`) with different **groups** and **tags** for organization.

## Wandb Organization

### Project Structure
- **Project**: `camerate-trap-reidentification` (all experiments)
- **Groups**: Category-specific grouping
  - `baselines`: Random and backbone-only baselines
  - `backbone_comparison`: Architecture comparison
  - `loss_comparison_standard`: Losses with standard batching
  - `loss_comparison_pk`: Losses with PK sampling
  - `dataset_comparison`: Dataset variant comparison
- **Tags**: Experiment-specific labels
  - `["baseline", "random"]` for random baseline
  - `["baseline", "backbone_only", "BVRA/MegaDescriptor-L-384"]` for specific backbone baseline
  - `["loss_exp", "arcface", "standard_batching"]` for ArcFace loss
  - `["backbone_exp", "resnet50"]` for ResNet50 backbone
  - `["dataset_exp", "master"]` for master dataset

### Why One Project?
- **Easier Comparison**: Compare across experiment types in one view
- **Better Tracking**: Track overall project progress
- **Simpler Management**: One place for all jaguar re-ID experiments
- **Flexible Filtering**: Use groups/tags to filter by category

## Experiment Categories

### 1. Baseline Experiments
**Notebook**: `notebooks/baseline_experiments.ipynb`  
**Purpose**: Sanity checks and performance lower bounds

**Experiments**:
- **Random Baseline** (1): Random embeddings (should get ~0% mAP)
- **Backbone-Only** (5): Pre-trained features without fine-tuning
  - MegaDescriptor-L-384 (1536-dim)
  - MegaDescriptor-S-384 (384-dim)
  - ResNet50 (2048-dim)
  - ConvNeXt Base (1024-dim)
  - EfficientNet B3 (1536-dim)

**Fixed**: Dataset (JID_Master), no training

**Expected Outcome**: Backbone-only should beat random; establishes minimum viable performance

---

### 2. Backbone Experiments
**Notebook**: `notebooks/backbone_experiments.ipynb`  
**Purpose**: Which architecture gives best mAP-efficiency tradeoff?

**Experiments** (6 total):
1. **MegaDescriptor-L-384** (1536-dim) - Best quality
2. **MegaDescriptor-B-384** (768-dim) - Balanced
3. **MegaDescriptor-S-384** (384-dim) - Fast
4. **ResNet50** (2048-dim) - CNN baseline
5. **ConvNeXt Base** (1024-dim) - Modern CNN
6. **EfficientNet B3** (1536-dim) - Efficient CNN

**Fixed**: Loss (ArcFace m=0.5, s=64), Dataset (JID_Master), Epochs (50)

**Key Metrics**: Validation mAP, training time, model size

**Expected Outcome**: MegaDescriptor-L likely best mAP; CNNs may be faster

---

### 3. Loss Function Experiments
**Notebook**: `notebooks/loss_experiments.ipynb`  
**Purpose**: Which loss function is most effective?

**Standard Batching Group** (6 experiments):
- ArcFace (m=0.5, s=64) - Standard
- ArcFace (m=0.3, s=64) - Soft margin
- ArcFace (m=0.7, s=64) - Hard margin
- SubCenter ArcFace - Handles intra-class variation
- Cross-Entropy - Classification baseline
- Focal Loss - Class imbalance handling

**PK Sampling Group** (3 experiments, P=8, K=4):
- Triplet Hard Mining
- Triplet Semi-Hard Mining
- ArcFace + Triplet Combined

**Fixed**: Backbone (MegaDescriptor-L-384), Dataset (JID_Master), Epochs (50)

**Key Metrics**: Validation mAP, loss convergence, training stability

**Expected Outcome**: ArcFace variants likely best; Triplet may help with hard negatives

---

### 4. Dataset Variant Experiments
**Notebook**: `notebooks/dataset_experiments.ipynb`  
**Purpose**: Which preprocessing yields best performance?

**Experiments** (5 total):
1. **master**: Segmented + Deduplicated (recommended)
2. **segmented_deduplicated**: Segmented + Deduplicated
3. **segmented**: Segmented only (with duplicates)
4. **not_segmented_deduplicated**: Full frames, deduplicated
5. **not_segmented**: Full frames with duplicates

**Fixed**: Backbone (MegaDescriptor-L-384), Loss (ArcFace), Epochs (50)

**Key Metrics**: Validation mAP, training convergence, data leakage indicators

**Expected Outcome**: Segmented + deduplicated should perform best; duplicates may cause overfitting

---

### 5. Hyperparameter Sweeps
**Configuration**: `get_hyperparameter_sweep_config()` in [experiments.py](experiments.py)  
**Purpose**: Fine-tune best configuration from previous experiments

**Use this AFTER identifying**:
- Best backbone (from backbone experiments)
- Best loss (from loss experiments)
- Best dataset (from dataset experiments)

**Sweep Method**: Bayesian optimization

**Search Space**:
- `learning_rate`: log-uniform [1e-5, 1e-3]
- `arcface_margin`: uniform [0.3, 0.7]
- `arcface_scale`: [30.0, 64.0, 128.0]
- `batch_size`: [16, 32, 64]
- `weight_decay`: log-uniform [1e-5, 1e-2]
- `dropout`: uniform [0.0, 0.5]
- `warmup_epochs`: [0, 5, 10]

**Metric**: Maximize `validation/map`

**Early Termination**: Hyperband (min_iter=10)

**How to Run**:
```python
import wandb
from jaguars.reidentification.experiments import get_hyperparameter_sweep_config

# Get sweep configuration
sweep_config = get_hyperparameter_sweep_config()

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="camerate-trap-reidentification")

# Run sweep agent
wandb.agent(sweep_id, function=train_function)
```

---

## Usage Examples

### Running Individual Notebooks

```bash
# 1. Baseline experiments
jupyter notebook notebooks/baseline_experiments.ipynb

# 2. Backbone comparison
jupyter notebook notebooks/backbone_experiments.ipynb

# 3. Loss comparison
jupyter notebook notebooks/loss_experiments.ipynb

# 4. Dataset comparison
jupyter notebook notebooks/dataset_experiments.ipynb
```

### Using Experiments in Code

```python
from jaguars.reidentification.config import get_default_config
from jaguars.reidentification.experiments import (
    get_baseline_experiments,
    get_backbone_experiments,
    get_loss_experiments,
    get_dataset_experiments,
    get_hyperparameter_sweep_config,
)
from jaguars.reidentification.training.train import run_processing as run_training

# Option 1: Use default config
experiments = get_backbone_experiments()

# Option 2: Pass custom base config
config = get_default_config()
config.wandb.enabled = True
config.wandb.entity = "jaguars"
config.wandb.project = "camera-trap-reidentification"
config.training.num_epochs = 100
experiments = get_backbone_experiments(base_config=config)

# Run experiments
for exp in experiments:
    results = run_training(exp.base_config)
    print(f"{exp.name}: mAP = {results['validation/map']}")
```

### Accessing Wandb Results

View all experiments at:
```
https://wandb.ai/<your-entity>/camerate-trap-reidentification
```

**Filter by group**:
- Baselines: `group:baselines`
- Backbone comparison: `group:backbone_comparison`
- Loss standard: `group:loss_comparison_standard`
- Loss PK sampling: `group:loss_comparison_pk`
- Dataset comparison: `group:dataset_comparison`

**Filter by tags**:
- All baseline experiments: `tags:baseline`
- All ArcFace experiments: `tags:arcface`
- All backbone experiments: `tags:backbone_exp`

---

## Recommended Workflow

### Phase 1: Baselines (Day 1)
1. Run `baseline_experiments.ipynb`
2. Verify random baseline ≈ 0% mAP (sanity check)
3. Record backbone-only performance (lower bound)

### Phase 2: Architecture Selection (Days 2-3)
4. Run `backbone_experiments.ipynb`
5. Analyze mAP vs training time
6. Select best backbone for next phases

### Phase 3: Loss Function Selection (Days 4-7)
7. Run `loss_experiments.ipynb` (standard batching first)
8. Run PK sampling experiments
9. Identify best loss function

### Phase 4: Dataset Optimization (Days 8-10)
10. Run `dataset_experiments.ipynb`
11. Verify segmentation + deduplication benefits
12. Confirm no data leakage

### Phase 5: Hyperparameter Tuning (Days 11-14)
13. Use best (backbone, loss, dataset) from previous phases
14. Run Wandb sweep with `get_hyperparameter_sweep_config()`
15. Fine-tune for production

---

## Implementation Details

### File Structure
```
src/jaguars/reidentification/
├── experiments.py              # Experiment definitions
├── EXPERIMENTS.md              # This file
└── training/
    └── train.py                # Training loop with baseline support

notebooks/
├── baseline_experiments.ipynb   # Baseline experiments
├── backbone_experiments.ipynb   # Backbone comparison
├── loss_experiments.ipynb       # Loss comparison
└── dataset_experiments.ipynb    # Dataset comparison
```

### Key Functions

**`experiments.py`**:
- `get_baseline_experiments(base_config)` → List[ExperimentConfig]
- `get_backbone_experiments(base_config)` → List[ExperimentConfig]
- `get_loss_experiments(base_config)` → List[ExperimentConfig]
- `get_dataset_experiments(base_config)` → List[ExperimentConfig]
- `get_hyperparameter_sweep_config(base_config)` → dict (Wandb sweep config)
- `get_all_experiments()` → dict[str, List[ExperimentConfig]]

### ExperimentConfig Structure

```python
@dataclass
class ExperimentConfig:
    name: str                          # Unique experiment name
    description: str                   # Human-readable description
    base_config: ReidentificationConfig  # Full training config
    tags: list[str]                    # Wandb tags
    group: str | None = None           # Wandb group
    seeds: list[int] = field(default_factory=lambda: [42])  # Random seeds
    variations: dict = field(default_factory=dict)  # Additional variations
```

---

## Best Practices

### 1. Always Pass Base Config
```python
# ✓ Good: Inherit notebook settings
config = get_default_config()
config.wandb.enabled = True
config.wandb.entity = "jaguars"
config.wandb.project = "camera-trap-reidentification"
experiments = get_backbone_experiments(base_config=config)

# ✗ Bad: Uses default project name
experiments = get_backbone_experiments()
```

### 2. Use Groups and Tags
- Groups organize experiments by category
- Tags enable fine-grained filtering
- Both are set automatically by experiment functions

### 3. Monitor Wandb During Training
```python
# Check experiment progress
import wandb
api = wandb.Api()
runs = api.runs("your-entity/camerate-trap-reidentification", filters={"group": "baselines"})
for run in runs:
    print(f"{run.name}: {run.summary.get('validation/map', 'N/A')}")
```

### 4. Save Experiment Metadata
```python
# Log experiment configuration to Wandb
import wandb
wandb.config.update({
    "experiment_name": experiment.name,
    "experiment_tags": experiment.tags,
    "experiment_group": experiment.group,
})
```

---

## Troubleshooting

### Issue: Experiments not appearing in Wandb
- **Solution**: Verify `config.wandb.enabled = True`
- Check `config.wandb.project` is set correctly

### Issue: PK sampling crashes
- **Solution**: Ensure dataset has enough samples per class (at least K=4)
- Check `use_pk_sampler=True` is set in config

### Issue: Out of memory
- **Solution**: Reduce batch size in config
- Use smaller backbone (MegaDescriptor-S instead of L)

### Issue: Low validation mAP across all experiments
- **Solution**: Check dataset loading (verify `fo_embeddings_field` is correct)
- Ensure train/val splits are set up properly
- Run random baseline to verify it's near 0%

---

## Results Interpretation

### What mAP Score is Good?

| mAP Range | Interpretation |
|-----------|---------------|
| 0-10%     | Random baseline level (problem with setup) |
| 10-40%    | Baseline untrained features |
| 40-70%    | Good training, learning is happening |
| 70-85%    | Strong performance for wildlife re-ID |
| 85%+      | Excellent performance (rare, check for leakage) |

### Comparing Results

1. **Within-category**: Use Wandb groups to compare experiments in same category
2. **Cross-category**: Compare best from each phase
3. **Statistical significance**: Run multiple seeds (varies seeds in ExperimentConfig)

---

## Future Extensions

Potential additions to experiment framework:
- **Multi-GPU Training**: Distributed experiments
- **Cross-Validation**: K-fold validation
- **Open-Set Evaluation**: Test on unseen individuals
- **Temporal Splits**: Train on earlier videos, test on later ones
- **Site-Based Splits**: Train on some locations, test on others

---

## References

For implementation details, see:
- [experiments.py](experiments.py) - Experiment definitions
- [config.py](config.py) - Configuration dataclasses
- [training/train.py](training/train.py) - Training loop

For usage examples, see notebooks in `notebooks/`
