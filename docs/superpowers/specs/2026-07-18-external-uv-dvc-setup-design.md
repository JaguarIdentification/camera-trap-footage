# External uv and DVC Setup Design

## Goal

Fully configure the repository with `uv` and DVC, download the tracked 35.96 GB dataset, and keep all large machine-local artifacts under `/Volumes/Extreme SSD/CuratedCameraTrapData`.

## Storage layout

The external storage root will be:

```text
/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/
├── data/       # Materialized DVC dataset
├── dvc-cache/  # DVC object cache
└── venv/       # uv-managed Python environment
```

The repository will contain local symlinks:

```text
camera-trap-footage/
├── data -> /Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data
└── .venv -> /Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/venv
```

This preserves the paths expected by existing source code, notebooks, the Makefile, and `data.dvc`, while avoiding large copies on the internal disk.

## Python environment

`uv` will create the environment at the external `venv/` path. Dependencies will be installed from `requirements.txt`, which already includes the editable local package and DVC with Google Drive support. The additional Transformers Git dependency documented in the README will then be installed into the same environment.

The setup will use a Python version compatible with the project's `>=3.10` requirement and the available dependency set. The repository's source dependency declarations will not be rewritten merely to perform a local installation.

## DVC configuration and data flow

Machine-specific DVC configuration will be stored in `.dvc/config.local`:

```ini
[cache]
    dir = /Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/dvc-cache
```

The existing default Google Drive remote in `.dvc/config` remains unchanged. DVC will download objects to the external cache and check out the tracked `data` output through the repository symlink into the external `data/` directory.

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

The environment, cache, and dataset remain untracked machine-local state. `.venv`, `data`, DVC cache content, and `.dvc/config.local` are already covered by the repository's ignore conventions or DVC defaults. Apart from this design document, no project source files need modification.
