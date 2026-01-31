# Jaguar Re-Identification (Re-ID)

Centerpiece project for identifying individual jaguars.

## Pipeline Structure (`pipeline.py`)
The pipeline coordinates the data flow:
1. **Ingestion**: `ingestion.loaders`
2. **Sampling**: `processing.sampling`
3. **Segmentation**: `src.segmentation` (External call)
4. **Filtering**: Filter out non-jaguar detections (ViewStage).
5. **Cropping**: Create `JID_ReID_Crops` dataset from detections.
6. **Splitting**: `processing.splitting`
7. **Training**: `training.train_reid`

## Tasks & Implementation Plan

### 1. `pipeline.py`
- Refactor to use `logging_utils`.
- Implement steps 4 (Filtering) and 5 (Cropping) using FiftyOne views and `dataset.clone()`.
- Call `processing.splitting`.

### 2. `training/`
- Implement Re-ID model training (PyTorch).
- Inputs: `JID_ReID_Crops` (accessed via FiftyOne).
- Outputs: Model weights in `data/models/reid/`.

### 3. `evaluation/`
- Implement mAP and CMC rank k calculation.
- Visualization code for FiftyOne App (Embeddings view).
