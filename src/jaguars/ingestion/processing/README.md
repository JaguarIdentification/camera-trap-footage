# Data Processing

This module handles transformations on the FiftyOne datasets *after* ingestion and *before* training.

## Tasks & Implementation Plan

### 1. `sample.py` (Completed ✅)
**Purpose**: Extract frames from videos in FiftyOne grouped datasets.

**Features**:
- Function `sample_video_frames(dataset_name, fps, max_frames, output_format)`.
- Extracts frames from videos in `JID_Master_Dataset` (or any grouped dataset).
- Adds sampled frames to the SAME GROUP as their source video (grouped dataset architecture).
- Links frames to source video via `source_video_id` metadata.
- Uses OpenCV for frame extraction.
- Follows the standard module pattern (logging, CLI, summary generation).

**Usage**:
```python
from jaguars.ingestion.processing.sample import run_processing

run_processing(
    dataset_name="JID_Master_Dataset",
    fps=1.0,  # Frames per second (0 = first frame only)
    max_frames=None,  # Limit frames per video
    output_format="jpg",
    summary_location=Path("summary.json"),
    verbose=True,
)
```

**CLI**:
```bash
python -m jaguars.ingestion.processing.sample \
    --dataset JID_Master_Dataset \
    --fps 1.0 \
    --format jpg \
    --verbose
```

### 2. `split.py` (New)
**Purpose**: Train/Test Splitting.
**Requirements**:
- Function `create_splits(dataset_name: str, method="stratified_group")`.
- **Logic**:
    - Group by Jaguar Identity (Stratified).
    - Group by Video Sequence (Group): Don't put frames from same video in train AND test via `source_video_id`.
    - Tag samples with `tags=["train"]` or `tags=["test"]`.
    - Store split strategy name in dataset info.

### 3. `deduplicate.py` (New)
**Purpose**: Remove redundant frames.
**Requirements**:
- **Migration**: Incorporate logic from `src/jaguar_reidentification/data_preprocessing/deduplicate.py` (Old location).
- Function `detect_duplicates(dataset_name: str)`.
- Use FiftyOne's embedding computation (e.g. MobileNet) or perceptual hash.
- Mark duplicates with `tags=["duplicate"]`. Do not delete immediately, just filter later.

### 4. `augment.py` (New)
**Purpose**: Offline augmentation (if needed) or preparation for online augmentation.
**Requirements**:
- **Migration**: Refactor `src/jaguar_reidentification/data_preprocessing/augment_camera_trap.py` (Old location).
- Domain adaptation: Logic to convert "Day/Professional" images to look like "Night/Trap" images (grayscale, noise, gamma).
