# External uv and DVC Setup Design

## Goal

Fully configure the repository with `uv` and DVC, download the tracked 35.96 GB dataset, and keep all large machine-local artifacts under `/Volumes/Extreme SSD/CuratedCameraTrapData`.

## Storage layout

The external storage root will be:

```text
/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/
└── data/       # Materialized DVC dataset

/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage-python.sparsebundle
└── APFS volume mounted at /Volumes/CameraTrapPython
    ├── dvc-cache/  # DVC object cache
    ├── uv-cache/   # uv package and build cache
    └── venv/       # uv-managed Python environment
```

The repository will contain local symlinks:

```text
camera-trap-footage/
├── data -> /Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data
└── .venv -> /Volumes/CameraTrapPython/venv
```

This preserves the paths expected by existing source code, notebooks, the Makefile, and `data.dvc`, while avoiding large copies on the internal disk.

## Python environment

The external SSD filesystem does not support uv's advisory lock and package-copy behavior reliably, so a dynamically growing APFS sparse bundle will be stored on that SSD and mounted at `/Volumes/CameraTrapPython`. `uv` will create the environment inside that APFS volume, using the already installed Homebrew Python 3.10 interpreter. Dependencies will be installed from `requirements.txt`, which already includes the editable local package and DVC with Google Drive support. On macOS ARM64, `eva-decord==0.6.1` will provide the compatible `decord` import because upstream `decord==0.6.0` has no wheel for this platform. The additional Transformers Git dependency documented in the README will then be installed into the same environment.

After a restart, mount the environment before using the repository:

```bash
hdiutil attach "/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage-python.sparsebundle"
```

The setup will use a Python version compatible with the project's `>=3.10` requirement and the available dependency set. The repository's source dependency declarations will not be rewritten merely to perform a local installation.

## DVC configuration and data flow

Machine-specific DVC configuration will be stored in `.dvc/config.local`:

```ini
[cache]
    dir = /Volumes/CameraTrapPython/dvc-cache
```

The existing default Google Drive remote in `.dvc/config` remains unchanged. DVC will download objects to the APFS-backed external cache and check out the tracked `data` output through the repository symlink into the external `data/` directory. The pull uses one worker (`--jobs 1`) because the Google Drive backend's default high concurrency races on temporary cache paths in this environment.

If Google authentication requires interactive browser authorization, setup will pause only for that authorization and then resume the pull.

## Verification

The completed setup must satisfy all of the following:

1. `.venv` and `data` resolve beneath the approved external storage root.
2. DVC reports its cache directory as the external `dvc-cache/` directory.
3. Python can import the local `jaguars` package and DVC from the uv-managed environment.
4. `dvc pull data.dvc` exits successfully.
5. `dvc status` reports that the tracked data is unchanged.
6. The materialized data size and file count are consistent with `data.dvc` (35,961,045,643 bytes and 13,001 files, allowing filesystem reporting differences for allocated size).

## Repository impact

The environment, DVC cache, and dataset remain untracked machine-local state. `.venv`, `data`, DVC cache content, and `.dvc/config.local` are already covered by the repository's ignore conventions, their external location, or DVC defaults. Apart from this design document, no project source files need modification.
