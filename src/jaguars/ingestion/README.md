# Data Ingestion

This module handles loading data from various sources (Videos, PPTX, CSV, HF) into FiftyOne datasets.

## Architecture
The goal is to strictly separate "loading data" from "processing data". This module uses a **grouped dataset architecture** where videos and images are stored in the same FiftyOne dataset with different "slices" (image slice and video slice).

### 1. `loaders.py`
**Purpose**: Unified entry points for loading data types.
**Functions to Implement/Update**:
- `ingest_pptx_slides(pptx_path: Path) -> None`:
    - **Migration**: Refactor logic from `src/jaguar_reidentification/data_preprocessing/pptx_extract.py` (Old location).
    - Extracts slides/images from PPTX.
    - Adds metadata (slide number, original file).
    - Adds to `JID_Master_Images` with `source_type="pptx"`.

- `ingest_csv_labels(csv_path: Path) -> None`:
    - **Migration**: Refactor logic from `src/jaguar_reidentification/data_preprocessing/clean_labels.py` (Old location).
    - Matches filenames in CSV to samples in `JID_Master_Images`.
    - Adds/Updates labels (Identity, Orientation, etc.) on the FiftyOne samples.

- `ingest_professional_images(folder_path: Path) -> None`:
    - Loads images from a folder for open-set evaluation.
    - Adds to `JID_Master_Images` with `source_type="professional"`.

- `ingest_open_set_images(folder_path: Path) -> None`:
    - Loads images from a folder for open-set evaluation.
    - Adds to `JID_Master_Images` with `source_type="open_set"`.

### 2. `processing/` Submodule
**Purpose**: Data preprocessing steps after ingestion.

- `sample.py`: 
    - Samples frames from videos in grouped FiftyOne datasets.
    - Adds sampled frames to the SAME GROUP as their source video.
    - Links frames to source video via `source_video_id` metadata.
    - Uses OpenCV for frame extraction.
    - Full CLI support and logging.
    - See `processing/README.md` for usage examples.
- `compute_metadata.py`:
    - Computes and updates metadata for videos/images in the dataset.
- `split.py`:
    - closed set train/test splitting.
    - optionally open set
- `add_embeddings.py`:
    - Adds image embeddings to FiftyOne samples using a pre-trained model.
- `deduplicate.py`:
    - Implement deduplication using image embeddings.
- `augment.py` (optional):
    - Implement data augmentation techniques.

### 3. Logging
- Ensure all functions use `logging_utils.setup_logger` for logging instead of `print()`.

### 4. Testing
- unit tests for each loader and processing function.