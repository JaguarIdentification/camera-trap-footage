# Copilot Instructions for Jaguar Identification (JID)

## Project Overview
Jaguar re-identification from camera trap footage using deep learning. The system processes videos and images through segmentation, creates training datasets, and trains re-ID models to identify individual jaguars.

**Key Technologies**: Python 3.10+, PyTorch, FiftyOne, SAM3, DVC, WandB

## Architecture

### Three-Module Structure
1. **ingestion** ([src/jaguars/ingestion](../src/jaguars/ingestion)): Data loading, processing, and export
   - Loaders: CSV labels, PPTX slides → FiftyOne datasets
   - Processing: video sampling, segmentation (SAM3), embeddings, splits, deduplication
   - Export: Disk, FiftyOne, HuggingFace
   
2. **reidentification** ([src/jaguars/reidentification](../src/jaguars/reidentification)): Model training and evaluation
   - Backbones: DINOv2, ResNet, EfficientNet, MegaDescriptor
   - Losses: ArcFace, CosFace
   - Extensive experiment framework (34 configurations) in [experiments.py](../src/jaguars/reidentification/experiments.py)
   
3. **segmentation**: SAM3/YOLO integration (service module)

**Critical Files**:
- [ARCHITECTURE.md](../ARCHITECTURE.md): System design and module boundaries
- [IMPLEMENTATION_SUMMARY.md](../src/jaguars/reidentification/IMPLEMENTATION_SUMMARY.md): Re-ID implementation details
- [EXPERIMENTS.md](../src/jaguars/reidentification/EXPERIMENTS.md): Experiment documentation

### Data Flow (Grouped Dataset Pattern)
```
Videos (group slice) → Frame Sampling → Images (group slice) → Segmentation → Crops
→ Training Dataset → Re-ID Model
```

FiftyOne uses grouped datasets with `group_field="group"`. Always use `select_group_slices("image")` or `select_group_slices("video")` to access specific media types.

## Critical Developer Knowledge

### Environment Setup
```bash
conda activate jid  # Python 3.10+ environment
# Use /sc/home/philipp.kolbe/conda3/envs/jid/bin/python for terminal commands
```

### Dataset Status Checking
Before running pipeline steps, **always check what's already been done**:
```bash
python scripts/check_dataset_state.py  # Shows completed ingestion steps
```

### Running Pipelines

**Ingestion** (notebooks/ingestion_pipeline.ipynb):
1. CSV/PPTX loading → FiftyOne
2. Video frame sampling
3. SAM3 segmentation (detects jaguars)
4. Embedding computation  
5. Train/val/test splits
6. Deduplication (near-duplicate detection)
7. Export variants

**Re-identification Training** (notebooks/reidentification_training.ipynb):
- Run experiments one-by-one or in batches
- WandB tracking enabled
- 6 experiment categories: backbone, loss, optimizer, augmentation, seed_stability, embedding_dim

### Common Patterns

**Processing Module Template** (see [deduplicate.py](../src/jaguars/ingestion/processing/deduplicate.py)):
```python
def run_processing(
    dataset_name: str = JID_MASTER_DATASET,
    dry_run: bool = False,
    verbose: bool = False,
    # ... specific parameters
) -> fo.Dataset | None:
    """Process dataset with specific operation."""
    logger = setup_logger("module_name", level=logging.DEBUG if verbose else logging.INFO)
    
    # Validate, process, save summary
    if dry_run:
        logger.info("DRY RUN: Would process...")
        return None
    
    dataset = get_or_create_dataset(dataset_name)
    # ... processing logic
    dataset.save()
    return dataset
```

**Experiment Configuration**:
```python
def get_experiment_group() -> list[ExperimentConfig]:
    base_config = get_default_config()
    base_config.wandb.enabled = True
    base_config.wandb.project = "jaguar-reid-category"
    
    return [
        ExperimentConfig(
            name="descriptive_name",
            description="What this tests",
            base_config=modified_config,
            tags=["category", "variant"],
        )
    ]
```

## Project-Specific Conventions

### Code Style (enforced by mypy + ruff)
- **Type hints required**: All function signatures, use `from collections.abc import Iterator` not `typing.Iterator`
- **Modern types**: `list[str]` not `List[str]`, `tuple[int, ...]` not `Tuple[int, ...]`
- **Docstrings**: Google style, no line break after `"""`, period after summary, one blank line before details
- **No `Any` type** unless absolutely necessary
- **Line length**: 120 characters max

Example:
```python
def process_samples(
    dataset: fo.Dataset,
    batch_size: int = 32,
) -> dict[str, int]:
    """Process samples in batches and return statistics.
    
    Args:
        dataset: FiftyOne dataset to process
        batch_size: Number of samples per batch
        
    Returns:
        Dictionary with processing statistics
    """
```

### Naming Conventions
- Datasets: `JID_Master_Dataset` (constant in [common/config.py](../src/jaguars/common/config.py))
- Fields: `snake_case` (e.g., `sam3_segmentations`, `closed_set_split`)
- Slices: `"image"`, `"video"` (group slices)
- Tags: lowercase (e.g., `"train"`, `"pptx"`, `"duplicate"`)

### Testing Requirements
- Unit tests in `tests/unit/{module}/`
- Use pytest fixtures for common setups
- Mock external dependencies (models, APIs)
- Run before committing: `pytest tests/unit/`

### Memory Management
**Critical**: Large datasets cause OOM errors
- Use `batch_size` parameters (default 16-32)
- Process FiftyOne views in chunks, not full datasets
- Deduplication: Fixed in [deduplicate.py](../src/jaguars/ingestion/processing/deduplicate.py) with batching

## Integration Points

### FiftyOne Brain
- **Deduplication**: `fob.compute_near_duplicates(view, embeddings=field, thresh=0.95)`
- **Similarity**: `fob.compute_similarity(view, embeddings=field, brain_key="sim_index")`
- Always operate on `view = dataset.select_group_slices("image")` not full dataset

### WandB Experiment Tracking
```python
config.wandb.enabled = True
config.wandb.project = "jaguar-reid-{category}"
config.wandb.run_name = experiment.name
config.wandb.tags = ["backbone", "dinov2"]
```

### DVC Data Management  
```bash
dvc pull  # Download tracked data
dvc push  # Upload after processing
```

Data in `data/` is DVC-tracked. Do NOT commit large files to git.

## Common Tasks

### Add New Experiment
1. Edit [experiments.py](../src/jaguars/reidentification/experiments.py)
2. Add to relevant category function (e.g., `get_backbone_experiments()`)
3. Document in [EXPERIMENTS.md](../src/jaguars/reidentification/EXPERIMENTS.md)
4. Run via notebook: `notebooks/reidentification_training.ipynb`

### Add New Processing Step
1. Create module in `src/jaguars/ingestion/processing/`
2. Follow [deduplicate.py](../src/jaguars/ingestion/processing/deduplicate.py) template
3. Add to [pipeline.py](../src/jaguars/ingestion/pipeline.py)
4. Update [ingestion_pipeline.ipynb](../notebooks/ingestion_pipeline.ipynb)
5. Write unit tests

### Debug Pipeline Issues
1. Check dataset state: `python scripts/check_dataset_state.py`
2. Use `--dry-run` flag to preview operations
3. Enable verbose logging: `--verbose` or `verbose=True`
4. Inspect in FiftyOne App: `fo.launch_app(dataset)`

## Troubleshooting

**OOM during deduplication**: Reduce `batch_size` parameter (default now 16)
**Tests failing**: Models download large weights, test OOM is common—use mocks
**Missing FiftyOne dataset**: Run ingestion pipeline first
**WandB not logging**: Check `config.wandb.enabled = True` and API key
- Data ingestion and processing are handled through dedicated modules in `ingestion` and `processing` directories.

## Conclusion
This document should serve as a foundational guide for AI agents to navigate and contribute effectively to the Jaguar Identification project. For further details, refer to the respective module READMEs and the main [README.md](../README.md).