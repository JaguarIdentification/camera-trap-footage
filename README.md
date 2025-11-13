# Jaguar Re-Identification
ML project to Re-Identify Jaguars as part of HPI Project Seminar.

## Setup

1. Create the environment:
```bash
   conda env create -f "environment.yml"
```
2. Activate it:
```bash
   conda activate jid
```

Use the provided Makefile to run the tests, linter, formatter or MyPy type checker.  
```bash
make all
```

### Update Environment
In the unlikely case that someone changed the environment.yml file, you can update the environment with the following command: (environment should be active)
```bash
conda env update --file environment.yml --prune
```

## Makefile handy targets

Setup:
- `make env` — create the conda environment from `environment.yml`.
- `make update` — updates the conda environment from `environment.yml`.
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
- [Project Plan](https://docs.google.com/document/d/1AqqHKnq8Na6zL1ObF2P69lKjYgMv5NLRFdO7Kjik5n0/edit?tab=t.0#heading=h.3a0tiv3a1n6v)
- [Project Slides](https://docs.google.com/presentation/d/1KX2jqEfPrJ5lMHUPYPqMlipiOfwCisV-8KoubId0wlk/edit?slide=id.p52#slide=id.p52)

## Authors
- Mehdi Gouasmi (https://github.com/D-i-n-o)
- Philipp Kolbe (https://github.com/philippkolbe)
- Supervisor: Antonio Rueda-Toicen (https://github.com/andandandand)
