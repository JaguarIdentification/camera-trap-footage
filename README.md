# Jaguar Re-Identification
ML project to Re-Identify Jaguars as part of HPI Project Seminar.

## Setup

### Requirements
This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management (recommended).

DVC is used for data versioning. Please install it by following the instructions [here](https://dvc.org/doc/install).

FFMPEG is required for video processing. Please install it by following the instructions [here](https://ffmpeg.org/download.html).

### Quick Setup with uv (Recommended)

1. Install uv (if not already installed):
```bash
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Note: PyTorch will be installed automatically with CPU support. For GPU support, install PyTorch separately following [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Running Quality Checks

Use the provided Makefile to run the tests, linter, formatter or MyPy type checker.  
```bash
make all
```

### Update Environment

**With uv:**
```bash
uv pip install -r requirements.txt --upgrade
```

## Makefile handy targets

Setup:
- `make env` — create the venv environment from `environment.yml`.
- `make update` — updates dependencies (`uv pip install -r requirements.txt --upgrade`).
- `make install` — pip install the package in editable mode inside the active environment.
- `make data` — pulls data tracked by DVC (requires DVC initialized / configured remote).

Analysis & Tests:
- `make format`
- `make lint`
- `make mypy`
- `make all` (runs all of the above)
- `make test`

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

- TODO Authenticate (everyone once)

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
