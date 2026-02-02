# Jaguar Identification (JID)

## Overview
This project focuses on Re-Identifying Jaguars from various data sources including camera trap videos, images, and jaguar ID guides.

## Architecture
The project is divided into three main components:
1. **Jaguar Re-Identification**: The core pipeline for building training datasets and training re-id models.
2. **Jaguar Orientation**: A module for determining the orientation (Left/Right/Front) of a jaguar, used to filter re-id data.
3. **Segmentation**: A service module providing SAM3 and YOLOE segmentation capabilities.

For a detailed technical overview, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Getting Started

### Prerequisites
*   Python 3.10+
*   `uv` (recommended) or `pip` or `conda` for package management.
*   FFMPEG is required for video processing. Please install it by following the instructions [here](https://ffmpeg.org/download.html).
    * If you use conda: `conda install -c conda-forge ffmpeg -y`
*   DVC is used for data versioning. Please install it by following the instructions [here](https://dvc.org/doc/install).

### Installation
```bash
# Install dependencies
uv pip install -r requirements.txt
pip install -e .

# Install generic SAM3 dependencies
pip install -q git+https://github.com/huggingface/transformers.git#egg=transformers
```

## Data management with DVC

We use DVC to keep large datasets out of Git while tracking them reproducibly.

- Initialize DVC locally (run once already):

```bash
dvc init

# TODO configure a remote (S3, GCS, Azure, SSH, etc.) - only once
dvc remote add -d myremote s3://my-bucket/path

git add .dvc .dvcignore
git commit -m "chore: init dvc"
```

- Authenticate (everyone once)

- Add a (large) dataset and push to remote (example):
```bash
##  DVC START           ##
dvc add data/large-dataset
###      GIT START     ###
git add data/large-dataset.dvc .gitignore
git commit -m "chore: track dataset with dvc"
git push
###       GIT END      ###
dvc push
##  DVC END             ##
```

**Note: How dvc commands always wrap around the git commands!**
**It is essential that these are applied in the correct order to avoid issues with dvc!**

- To get data on another machine, run:

```bash
dvc pull
```

The Makefile includes a `data` target that runs `dvc pull` for convenience.

## Project Structure
*   `src/jaguars/ingestion`: Data ingestion and preprocessing.
*   `src/jaguars/reidentification`: Main reidentification pipeline logic.
*   `src/jaguars/orientation`: Orientation classification.
*   `src/jaguars/segmentation`: Object detection/segmentation wrappers.
*   `src/jaguars/common`: Shared utilities.
*   `data/`: Data storage (dvc tracked).

## Further Resources
- [Huggingface Jaguar Dataset](https://huggingface.co/datasets/jaguaridentification/jaguars)
- [Camera Trap Data](https://drive.google.com/drive/folders/1Ztn79beQBbZAKcn_EdCT6QDsroFpqUq-)
- [JaguarIdentification ReidentificationModels GitHub](https://github.com/JaguarIdentification/ReidentificationModels)
- [Notebook to perform embeddings-based exploration and segmentation on FiftyOne](https://github.com/andandandand/practical-computer-vision/blob/main/notebooks/Jaguar_Identification_Embeddings_Based_Exploration.ipynb)

### Project Documents
- [Project Proposal](https://docs.google.com/document/d/1BytfANvJylhKjgfPAI0kiPqtGNK5TTIx7T2rFmMZVIw)
- [Project Plan](https://docs.google.com/document/d/1AqqHKnq8Na6zL1ObF2P69lKjYgMv5NLRFdO7Kjik5n0)
- [Project Slides](https://docs.google.com/presentation/d/1KX2jqEfPrJ5lMHUPYPqMlipiOfwCisV-8KoubId0wlk)

## Authors
- Mehdi Gouasmi (https://github.com/D-i-n-o)
- Philipp Kolbe (https://github.com/philippkolbe)
- Supervisor: Antonio Rueda-Toicen (https://github.com/andandandand)
