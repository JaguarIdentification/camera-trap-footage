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

### Final curated FiftyOne snapshot

The intended immutable v1 snapshot, `JaguarCameraTrap_Final_Curated_v1`, is
bounded by 1,322 unique final artifacts in
`data/intermediate/v1/fo_jaguars/labeled_segmented_jaguars_final_curated_v1`.
This deterministic export is derived from, but never modifies, the original
1,367-sample `labeled_segmented_jaguars_primitive` export.

Audit the approved curation policy without writing:

```bash
uv run python -m jaguars.visualization.final_curation --dry-run
```

It must report 39 exact-content duplicate drops, six confirmed false-positive
drops, three audited bbox clips, and 1,322 retained unique paths/hashes. The
retained set contains 1,108 populated and 214 null identities across 59
non-null labels.

After review, materialize the export atomically:

```bash
uv run python -m jaguars.visualization.final_curation --create
```

Media is hardlinked to the untouched source export; it is never copied or
re-encoded. Unsupported cross-filesystem hardlinks fail. Four retained records
(`000001-143`, `000002-144`, `000010-18`, and `000005-126`) are deliberately
not re-segmented: their malformed annotations are omitted and they are marked
`review_required=True`, `review_status="pending"`, and
`needs_annotation_review`.

Audit the materialized export before snapshot creation:

```bash
uv run python -m jaguars.visualization.final_dataset --dry-run
```

The snapshot command creates, verifies, and launches on `localhost:5151`:

```bash
uv run python -m jaguars.visualization.final_dataset
```

The operational modes and replacement guards are:

- `--create-only` creates and verifies without launching the App.
- `--launch-only` launches an existing snapshot without rebuilding or
  re-auditing its media.
- Ordinary creation refuses to continue if the final dataset name already
  exists.
- `--overwrite` is required to replace that exact persistent dataset. It shows
  the existing and proposed sample counts and requires interactive
  confirmation; noninteractive replacement additionally requires `--yes`.
  `--yes` is invalid without `--overwrite`.

Both `/Volumes/Extreme SSD` and `/Volumes/CameraTrapPython` must be mounted
filesystems. All generated FiftyOne state is constrained below
`/Volumes/CameraTrapPython/fiftyone`, including:

- database:
  `/Volumes/CameraTrapPython/fiftyone/var/lib/mongo`
- timestamped JSON reports and `latest.json`:
  `/Volumes/CameraTrapPython/fiftyone/JaguarCameraTrap_Final_Curated_v1`
- dataset/download defaults:
  `/Volumes/CameraTrapPython/fiftyone/datasets`

The curation sidecar records the source JSON hash, policy version, every kept
and dropped path/reason, hash group and representative, clip, stripped review
annotation, count, and retained media hash. Snapshot reports record configured
exports and manifests, storage paths, identity/validation/lineage counts,
phase/failure, constructed fields, and views.

The snapshot has exactly nine saved views:

- `All final samples`
- `Lineage issues`
- `Closed-set train`
- `Closed-set val`
- `Closed-set test`
- `Open-set train`
- `Open-set val`
- `Open-set test`
- `Annotation review`

Creation is strict: missing or unreadable media, duplicate canonical paths or
SHA-256 content, malformed bounding boxes or masks, invalid enrichment values,
terminal count or identity-count drift, and inconsistent `jaguar_id` /
`ground_truth` identity all block publication. Missing or ambiguous lineage is
reported but does not remove a retained sample. Missing annotations are valid
only for the four fully marked pending-review records; the other 1,318 samples
must have valid nonempty body boxes and segmentations.

## Project Structure
*   `src/jaguars/ingestion`: Data ingestion and preprocessing.
*   `src/jaguars/reidentification`: Main reidentification pipeline logic.
*   `src/jaguars/orientation`: Orientation classification.
*   `src/jaguars/segmentation`: Object detection/segmentation wrappers.
*   `src/jaguars/common`: Shared utilities.
*   `data/`: Data storage (dvc tracked).

## Further Resources
- [Huggingface Jaguar Dataset](https://huggingface.co/datasets/jaguaridentification/jaguars)
- [Camera Trap Data](https://drive.google.com/drive/folders/1Mnc0e3pL4ib3rnDQWZB48oomxj6P8rF_)
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
