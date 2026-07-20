# FiftyOne Full Local Dataset Design

## Goal

Provide one repeatable command that incrementally catalogs and launches every valid local image and video under `data` in FiftyOne without copying the media. The catalog must preserve every physical path as its own sample, expose available split and capture-time provenance, and keep all generated state on external storage.

## Scope

The catalog includes every readable local media path under `data`, including raw files, intermediate files, derived crops and segmentations, sampled frames, screenshots, and media copied into existing FiftyOne exports. Exact duplicate content at different paths remains visible as separate samples.

The catalog excludes:

- AppleDouble files whose names begin with `._`
- Broken symlinks
- Git LFS pointer files and Parquet shards
- CSV, JSON, PowerPoint, spreadsheet, and other non-media files
- Unsupported image and video extensions

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff`. Supported video extensions are `.mp4`, `.avi`, `.mov`, and `.mkv`. Matching is case-insensitive.

The implementation does not run segmentation, embedding extraction, video-frame sampling, deduplication, or model inference. It does not replace or mutate existing exported FiftyOne datasets.

## Architecture

The feature consists of three focused components and a CLI launcher.

### Media discovery

The discovery component recursively walks the resolved `data` directory and emits one record per supported physical media path. Each record contains:

- The normalized absolute filepath
- A stable key derived from that normalized path
- Media type (`image` or `video`)
- Extension
- File size and modification time
- Repository-relative path
- Source collection and derived variant inferred from directory structure

Discovery must be deterministic: records are sorted by normalized path before synchronization. It skips unsupported files and AppleDouble companions without attempting to decode them. A validation stage performs lightweight media readability checks and records failures in the run report.

### Metadata enrichment

The enrichment component reads both split-bearing manifests:

- `data/intermediate/v1/labels_with_splits.csv`
- `data/intermediate/v1/pptx_extracted_labels_with_splits.csv`

It joins manifest rows to discovered media using a precedence-ordered set of normalized absolute paths, paths relative to the manifest dataset root, and original filenames. Ambiguous filename-only matches are not guessed: the sample records all candidate manifest row identifiers and receives a `provenance_status` value describing the ambiguity.

When available, samples expose:

- Jaguar identity
- Closed-set split
- Open-set split
- Camera-trap site, camera ID, camera model, latitude, and longitude
- Sighting ID
- Original filename and raw source path
- Manifest date, time, and combined datetime
- Embedded EXIF datetime for images
- A timestamp agreement status (`match`, `mismatch`, `manifest_only`, `exif_only`, or `missing`)

Manifest and EXIF timestamps remain separate fields. The system does not silently declare either source authoritative, does not invent a timezone, and does not overwrite conflicting values.

### Persistent incremental synchronization

The synchronizer creates or loads a persistent mixed-media FiftyOne dataset named `JaguarCameraTrap_Full_Local`. The normalized-path stable key is unique within that dataset.

Each run computes four sets:

- New paths to insert
- Existing paths whose filesystem or enriched metadata changed
- Existing paths that are unchanged
- Catalog samples whose files are currently missing

New and changed records are written in bounded batches. Unchanged records are not rewritten. Missing records are retained and marked unavailable by default so that existing annotations are preserved. They are deleted only when the operator explicitly supplies `--prune-missing`.

The synchronizer never collapses samples by content hash. Two paths containing identical bytes remain separate catalog entries.

An exclusive synchronization lock prevents concurrent writers. The FiftyOne App can launch against an existing catalog without acquiring the writer lock when `--no-sync` is used.

## External-storage layout

All generated FiftyOne state must remain outside the repository and internal system disk:

- FiftyOne/MongoDB database: `/Volumes/CameraTrapPython/fiftyone/var/lib/mongo`
- Catalog reports and synchronization lock: `/Volumes/CameraTrapPython/fiftyone/JaguarCameraTrap_Full_Local`
- FiftyOne dataset/download defaults, if used: `/Volumes/CameraTrapPython/fiftyone/datasets`

Media continues to live under the repository's `data` symlink on `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data`; synchronization stores file references rather than media copies.

The launcher must validate that `/Volumes/Extreme SSD` and `/Volumes/CameraTrapPython` are actual mounted filesystems, not merely existing directories. It must refuse to start synchronization if the configured database, report, or dataset-default directory resolves outside `/Volumes/CameraTrapPython/fiftyone`.

## Command-line interface

The repository exposes a single module command for normal use:

```bash
uv run python -m jaguars.visualization.full_dataset
```

Default behavior:

1. Validate mounts and external-storage configuration.
2. Inventory and validate all supported local media.
3. Print counts by extension, source collection, and variant.
4. Incrementally synchronize `JaguarCameraTrap_Full_Local`.
5. Launch the FiftyOne App on `http://localhost:5151`.
6. Keep the process alive until interrupted and then close the App session cleanly.

Supported options:

- `--dry-run`: inventory and report proposed synchronization changes without starting MongoDB writes or launching the App
- `--sync-only`: synchronize and exit without launching the App
- `--no-sync`: launch the existing catalog without inventory or synchronization
- `--prune-missing`: permanently delete catalog samples whose files are missing
- `--address ADDRESS`: App bind address, default `localhost`
- `--port PORT`: App port, default `5151`
- `--dataset-name NAME`: dataset name, default `JaguarCameraTrap_Full_Local`

`--sync-only` and `--no-sync` are mutually exclusive. `--prune-missing` is valid only when synchronization runs. Destructive pruning prints the exact count before deletion and requires an explicit confirmation mechanism unless a separate noninteractive confirmation flag is deliberately supplied.

## Source and variant classification

Every sample receives `source_collection` and `variant` fields derived from its repository-relative path. The initial classification rules cover:

- `raw`: source camera-trap and source-document media
- `intermediate_files`: cleaned or converted working media
- `screenshots`: screenshot-derived media
- `fiftyone_export`: media stored inside an exported FiftyOne dataset
- `huggingface_materialized`: actual media materialized from the Hugging Face package, if later present
- `other`: supported media not matching a more specific rule

Variants include `original`, `video`, `sampled_frame`, `cropped_body`, `cropped_head`, `segmented_body`, `segmented_head`, `pptx_extracted`, and `unknown`. Classification is descriptive and does not exclude any valid media.

## Error handling and reporting

Startup errors are fatal and actionable:

- Missing or incorrectly mounted external volumes
- Database or report paths resolving outside the approved external root
- Dataset schema incompatible with the required stable-key field
- Another synchronization process holding the lock
- App port unavailable

An unreadable individual media file is nonfatal. It is omitted from insertion or marked unavailable when already cataloged, and its filepath and error appear in a JSON run report under the external report directory.

Each run report records:

- Start and finish time
- Dataset name and FiftyOne version
- Resolved storage and data paths
- Inventory counts and validation failures
- New, updated, unchanged, missing, and pruned counts
- Manifest join counts, ambiguous joins, and timestamp-agreement counts
- App address and port when launched

Temporary failures do not delete existing samples. A failed batch stops synchronization with the already committed batches recorded in the report; the next run safely resumes from observed catalog state.

## Testing strategy

Unit tests use temporary directories and pure records for:

- Recursive discovery and deterministic ordering
- Case-insensitive extension matching
- AppleDouble, unsupported-file, broken-symlink, and LFS-pointer exclusion
- Stable path keys and separate records for identical bytes at different paths
- Source and variant classification
- Manifest joins, including ambiguous filename matches
- Closed/open split mapping
- Manifest and EXIF timestamp preservation and agreement status
- Synchronization diff calculation for new, changed, unchanged, and missing samples
- CLI option validation
- Mount and external-root safety guards

An integration test creates a small temporary persistent FiftyOne dataset with image and video fixtures. It verifies first synchronization, idempotent second synchronization, metadata update, missing-file marking, annotation preservation, and explicit pruning. Tests must use an isolated temporary FiftyOne database and must never connect to or mutate `JaguarCameraTrap_Full_Local`.

## Acceptance criteria

The feature is accepted when all of the following are demonstrated:

1. Independent filesystem inventory and discovery report the same count of supported, valid local media paths.
2. Every valid physical path is represented, including duplicate content stored at different paths.
3. A second synchronization inserts zero duplicate samples.
4. Missing files are marked unavailable by default and pruned only through the explicit destructive option.
5. Both image and video samples render in the FiftyOne App.
6. Closed-set and open-set train/validation/test fields are filterable where manifest assignments exist.
7. Manifest and EXIF capture-time fields are visible independently, and timestamp mismatches are filterable.
8. FiftyOne database, reports, locks, and dataset defaults resolve under `/Volumes/CameraTrapPython/fiftyone`.
9. No media is copied and no existing exported FiftyOne dataset is modified.
10. `http://localhost:5151` responds while the launcher is running, and interrupting the command closes the session cleanly without deleting the persistent catalog.

## Known dataset constraints

The local Hugging Face `.parquet` shards are Git LFS pointer files rather than materialized Parquet data. They are intentionally excluded from this media catalog. If the real LFS objects are materialized later and contain embedded images rather than ordinary media files, importing those embedded records is a separate feature and is not part of this design.

The available split manifests cover only labeled subsets of the full local media tree. Unmatched samples remain visible with empty split and identity fields; the launcher does not invent assignments.
