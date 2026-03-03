# Jaguar Identification (JID) Architecture Plan

## 1. Directory Structure

We will transition the repository to the following structured layout. Each feature area has its own directory containing `preprocessing`, `training`, `evaluation`, and an `interface` (CLI/API).

```text
JID/
├── data/                       # DVC tracked data
│   ├── raw/                    # Input videos/images
│   ├── processed/              # Intermediate steps (optional dump)
│   ├── test_fixtures/          # Small dataset for E2E tests
│   └── models/                 # Model weights (DVC tracked)
├── src/
│   ├── common/                 # Shared utilities
│   ├── segmentation/           # Segmentation Module
│   │   ├── __init__.py
│   │   ├── interface.py        # Generic Segmentor Class
│   │   ├── sam3.py             # SAM3 Implementation
│   │   ├── yoloe.py            # YOLOE Implementation
│   │   └── README.md
│   ├── jaguar_orientation/     # Orientation Project
│   │   ├── pipeline.py         # Main Pipeline Script
│   │   ├── training/           # Classification model training
│   │   ├── evaluation/
│   │   └── README.md
│   ├── jaguar_reidentification/# Re-ID Project
│   │   ├── pipeline.py         # Main Pipeline Script
│   │   ├── training/           # Metric learning, Triplet loss
│   │   ├── evaluation/         # mAP, CMC calculations
│   │   └── README.md
│   ├── ingestion/              # Ingesting data into FiftyOne datasets
|       |── pipeline.py         # Ingest all data sources
|       |── loaders/            # Loaders for videos, pptx, csv, hf
|       |── processing/         # Sampling, splitting, deduplication
|       |── README.md
│   ├── fiftyone/               # App launching scripts
│   └── example_module.py       # Template module for new features
├── tests/
│   ├── e2e/                    # Full pipeline tests on `test_fixtures`
│   └── unit/                   # Per-module tests
├── Makefile                    # Automation
└── README.md                   # Project Root Documentation
```

---

## 2. Module Design Pattern (The "Contract")

Every script/module that performs a data processing step must follow this pattern to ensure it is callable via CLI, Notebook, and Pipeline.

### Common Logging Strategy
To ensure consistent logging across all modules, we define a shared logging strategy. Each module should use the `logging` library and configure it as follows:

```python
import logging
from pathlib import Path

def setup_logger(module_name: str, log_file: Path | None = None, level: int = logging.INFO):
    """Sets up a logger with optional file logging."""
    logger = logging.getLogger(f"jid_logger.{module_name}")
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger
```

---

### Module Template

```python
# src/example_module.py

from pathlib import Path
import json
import argparse
import fiftyone
import logging
from common.logging_utils import setup_logger
import wandb  # store config, metrics, summaries and models as artifacts for training and evaluation modules

MODULE_NAME = "example_module"

def validate_resources(input_path: Path, model_path: Path):
    """Checks if inputs exist."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    # or check that all properties exist on datasets (csv or fiftyone)

def write_summary(summary_data: dict, summary_location: str, to_wandb: bool = False):
    """Generates summary report."""
    with open(f"data/{summary_location}/summary.json", "w") as f:
        json.dump(summary_data, f)
    
    if to_wandb:
        wandb.log({"summary": summary_data})

def run_processing(
    dataset_name: str,
    input_path: Path,
    model_path: Path,
    param_a: int = 10,
    summary_location: str | None = f"summaries/{MODULE_NAME}.json",
    checkpoint_dataset: bool = False,
    wandb_project: str = "JID_Project",
    dry_run: bool = False,
    verbose: bool = False,
) -> fiftyone.Dataset | None:
    """
    Core Logic. 
    Returns: The updated/created FiftyOne dataset object.
    """
    logger = setup_logger(module=MODULE_NAME, log_file=Path(f"logs/{MODULE_NAME}_run_{dataset_name}.log"), level=logging.DEBUG if verbose else logging.INFO)
    logger.info("Starting processing...")

    validate_resources(input_path, model_path)
    
    if not dry_run:
        if wandb_project:
            wandb.init(project=wandb_project, name=f"{MODULE_NAME}_run_{dataset_name}", config={
                "input_path": str(input_path),
                "model_path": str(model_path),
                "param_a": param_a,
            })

        # 1. Load/Create Dataset
        # 2. Process
        # 3. possibly Save/Persist.

        summary_data = {"status": "success", "processed": True}  # Example summary. Usually computed as an output of "load" and "process" steps above.
        if summary_location:
            write_summary(summary_data, summary_location, to_wandb=bool(wandb_project))

        if wandb_project:
            wandb.finish()

        if checkpoint_dataset:
            # Checkpoint fiftyone dataset by appending "_module"

        logger.info("Processing completed successfully.")

        return fiftyone.Dataset(name=dataset_name)  # Placeholder return. Either return copied or modified dataset.
    else:
        logger.info("Dry run enabled - no changes made.")
        return None

def main():
    """CLI Entrypoint"""
    parser = argparse.ArgumentParser(description="Clean Module Description")
    parser.add_argument("--dataset-name", required=True, type=str, help="Name of the FiftyOne dataset to create or modify.")
    parser.add_argument("--input-path", required=True, type=Path, help="Path to input data (files or fiftyone dataset).")
    parser.add_argument("--model-path", required=True, type=Path, help="Path to model weights or configuration.")
    parser.add_argument("--param-a", type=int, default=10, help="An example parameter for processing.")

    parser.add_argument("--wandb-project", required=True, type=str, help="Weights & Biases project name for logging.")
    parser.add_argument("--summary-location", required=True, type=str, help="Location to save summary report.")
    parser.add_argument("--checkpoint-dataset", action="store_true", help="Flag to checkpoint the FiftyOne dataset by appending '_module'.")
    parser.add_argument("--dry-run", action="store_true", help="If set, no changes will be made; useful for testing.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    args = parser.parse_args()
    
    run_processing(
        dataset_name=args.dataset_name,
        input_path=args.input_path,
        model_path=args.model_path,
        param_a=args.param_a,

        checkpoint_dataset=args.checkpoint_dataset,
        wandb_project=args.wandb_project,
        summary_location=args.summary_location,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
```

---

## 3. Data Strategy: FiftyOne & Modalities

### Dataset Management
Instead of writing images to disk at every step, we will maintain persistent FiftyOne datasets.
1.  **Raw Ingestion Dataset:** Contains mixed modalities.
    *   FiftyOne supports Group datasets (slices), but simpler is often better for ML pipelines.
    *   **Strategy:** We will have a `JID_Master_Dataset` dataset (videos and images in different group slices).
    *   When sampling frames from videos, we add them to `JID_Master_Dataset` with a metadata field `source_video_id` linking back to the video dataset.
2.  **Versioning:**
    *   `dataset.clone("JID_Preprocessing_Step1")` can be used if we need to snapshot states.
    *   Final training sets are exported to HuggingFace or DVC-tracked folders.

### HuggingFace Integration
*   Use `dataset.load_from_hub()` and `dataset.push_to_hub()` for final training sets.

---

## 4. Pipelines Detail

### A. Segmentation
Doesn't hold state. It takes a dataset, runs a model (YOLOE or SAM or SAM2 or SAM3), adds labels/masks to the dataset, and returns it.

1. **Inference:** Load the specified model and run on the input dataset.
2. **Evaluation:** Check how many of the jaguar images detected a jaguar.
3. **Human Check:** Manually verify segmentations via fiftyone app (probably not part of this module).

### B. Jaguar Orientation
1.  **Ingest:** Load image dataset.
2.  **Tagging:** If orientation labels not existing (or flag to overwrite): Helper interface in FiftyOne to manually tag Left/Right/Front.
3.  **Augmentation:** Standard image augmentations (flips also need to flip label!).
4.  **Train:** Train classifier (ResNet/ViT).
5.  **Inference:** Exposed as a function `predict_orientation(image_batch)`.
6.  **Evaluation:** Standard classification metrics. Compare pretrained baselines against different trained models.

### C. Jaguar Re-Identification (Main Target)
This pipeline trains the Re-ID model. We should assume that the input is a FiftyOne dataset `JID_Master_Images` already contains cropped jaguar images from segmentation including an orientation label (if this is required for this training config).

1.  **Training:**
    *   Different Embedding Backbones
    *   Different losses (Triplet Loss, ArcFace, etc.)
    *   Model-Fusion techniques (fusing models from high-res images vs. camera trap images): early, intermediate, late fusion.
    *   Include Orientation as training label or not.
    *   Unbalanced vs. Balanced sampling strategies.
2.  **Inference**:
    *   Given a query image, compute embedding and retrieve nearest neighbors from gallery.
3.  **Evaluation:**
    *   Standard Re-ID metrics: mAP, CMC.
    *   Visualize retrieval results in FiftyOne.

### D. Ingestion
Handles loading data from various sources into FiftyOne datasets.
1.  **Ingestion (Unified Loader):**
    *   `loaders.ingest_videos(path)` -> Updates `JID_Source_Videos`
    *   `loaders.ingest_pptx(path)` -> Extracts to temp, Updates `JID_Master_Images` (source_type="pptx"). May already contain crops.
    *   `loaders.ingest_hf(repo_id)` -> Updates `JID_Master_Images` (source_type="professional")
    *   `loaders.ingest_csv_images(path)` -> Updates `JID_Master_Images` (source_type="csv")
    *   `loaders.ingest_open_set(path)` -> Updates `JID_Master_Images` (source_type="open_set", split="test"). Used for open-set evaluation from unknown jaguars from a different project.
2.  **Sampling:**
    *   Iterate `JID_Source_Videos` -> Extract Frames -> Add to `JID_Master_Images` (source_type="video_frame").
3.  **Splitting:**
    *   Stratified split by Jaguar ID (Identity).
    *   Ensure video frames from same sequence stay in same split.
    *   Split early to avoid data leakage.
    *   Different split strategies stored in different attributes (`open`, `closed`, `random`)
4.  **Deduplication:**
    *   Use image embeddings to remove near-duplicates. 
    *   Use fiftyone built-in functions to compute cross-split deduplication.
5.  **Augmentation:**
    *   Basic augmentations (flips, crops, color jitter).
    *   Domain adaptation augmentations (grayscale conversion, noise (lighting) addition).
    *   Possibly convert professional and daylight images to grayscale to match night camera trap domain. Or lighting adaptations.
6.  **Export:**
    *   Push `JID_Master_Images` and `JID_ReID_Crops` to HuggingFace or DVC-tracked folder structure.
---

## 5. Testing Strategy

1.  **Fixtures:** A `data/test_fixtures` folder containing:
    *   1 short video (5 sec).
    *   2 dummy images.
    *   1 dummy PPTX.
2.  **E2E Test:** A script that runs full pipelines on `test_fixtures` and asserts that the final dataset exists and contains >0 samples.
3.  **Unit Tests:** Tests per module 
