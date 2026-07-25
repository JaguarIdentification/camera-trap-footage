# FiftyOne Final Curated Dataset Design

## Goal

Create a persistent, frozen FiftyOne snapshot containing only the terminal artifacts of the camera-trap curation process. The snapshot is named `JaguarCameraTrap_Final_Curated_v1` and contains exactly the 1,367 samples exported by `data/intermediate/v1/fo_jaguars/labeled_segmented_jaguars_primitive`.

## Final-artifact boundary

Each sample uses the terminal segmented jaguar image as its primary media and retains the final `bboxes_body` and `segmentations_body` annotations. The terminal export contains 1,120 populated `jaguar_id` values and 247 null values. A null identity remains empty unless the approved exact lineage rules produce one unique compatible identity. When identity is resolved, the dataset stores both `jaguar_id` and a FiftyOne `ground_truth` `Classification` whose label equals it; otherwise both fields remain empty.

The snapshot excludes raw videos, sampled frames, screenshots, earlier crops and segmentations, duplicate candidates, embeddings, deduplication scores, cache paths, temporary identifiers, random fields, and pipeline timestamps.

All 1,367 terminal samples remain in the snapshot. Missing or ambiguous upstream lineage never removes a terminal sample.

## Enrichment and lineage

Terminal samples are enriched from upstream exports and the two terminal split manifests:

- `data/intermediate/v1/labels_with_splits.csv`
- `data/intermediate/v1/pptx_extracted_labels_with_splits.csv`

Lineage matching uses only these precedence-ordered exact methods:

1. Preserved FiftyOne or source sample ID
2. Normalized source filepath
3. Export-relative filepath
4. Exact filename when it is unique across all candidate records

The implementation must not use fuzzy filenames, inferred numeric prefixes, image similarity, or best guesses. Every sample receives `lineage_status` with `matched`, `ambiguous`, or `missing`, plus a `lineage_match_method` when matched.

The curated enrichment schema contains:

- Identity: optional `jaguar_id` and `ground_truth`
- Evaluation: `closed_set_split`, `open_set_split`
- Capture grouping: `sighting_id`
- Location and device: `site`, `location`, `camera_id`, `camera_side`, `camera_model`, `latitude`, `longitude`
- Capture time: `capture_date`, `capture_time`, `capture_datetime`
- Provenance: `original_filename`, `source_media_path`, `source_type`, `lineage_status`, `lineage_match_method`
- Integrity: `sha256`, file size, width, height

`closed_set_split` and `open_set_split` are independent string fields containing `train`, `val`, or `test`. The closed-set protocol evaluates identities represented in training. The open-set protocol also evaluates identities withheld from training.

## Storage and immutability

FiftyOne references the existing terminal images without copying them. Every image receives a SHA-256 digest so later media drift can be detected.

Generated state stays on external storage:

- FiftyOne database: `/Volumes/CameraTrapPython/fiftyone/var/lib/mongo`
- Reports: `/Volumes/CameraTrapPython/fiftyone/JaguarCameraTrap_Final_Curated_v1`
- Dataset/download defaults: `/Volumes/CameraTrapPython/fiftyone/datasets`

Creation refuses to run unless `/Volumes/Extreme SSD` and `/Volumes/CameraTrapPython` are actual mounted filesystems and every configured generated-state path resolves below `/Volumes/CameraTrapPython/fiftyone`.

The snapshot is immutable during ordinary operation. If the final dataset already exists, creation fails. Replacement requires `--overwrite`, prints existing and proposed sample counts, and requires interactive confirmation. Noninteractive replacement additionally requires `--yes`. Replacement first builds and validates an ownership-unique staging dataset while the old final remains published. It then renames the old final to an ownership-unique backup, promotes staging, and deletes the backup only after successful promotion. Promotion failure restores the old final and cleans only owned staging state; media is never deleted.

## Atomic creation

Creation builds a temporary persistent FiftyOne dataset. All source and constructed-dataset checks run before the temporary dataset is renamed to `JaguarCameraTrap_Final_Curated_v1`. A failed build removes only its temporary dataset record and retains a failure report. Transactional replacement applies the same validation before any rename and rolls back the old final if promotion fails. It never removes an existing final before the replacement is complete or changes any media.

## Command-line interface

The normal command is:

```bash
uv run python -m jaguars.visualization.final_dataset
```

Default behavior validates storage, audits source consistency, creates the snapshot when absent, writes a JSON report, launches the FiftyOne App on `localhost:5151`, and remains alive until interrupted.

Supported modes are:

- `--dry-run`: audit and report without database writes or App launch
- `--create-only`: create and verify without launching the App
- `--launch-only`: launch an existing snapshot without rebuilding it
- `--overwrite`: explicitly replace the exact final dataset
- `--yes`: allow noninteractive overwrite; valid only with `--overwrite`
- `--address` and `--port`: configure the App listener

Mutually incompatible modes fail during argument validation.

## Saved views

The snapshot includes:

- `All final samples`
- `Lineage issues`
- Closed-set `train`, `val`, and `test`
- Open-set `train`, `val`, and `test`

Jaguar identities remain filterable through `jaguar_id`; the implementation does not create hundreds of per-identity saved views.

## Integrity policy

Creation fails if:

- Any referenced final image is missing or unreadable
- Final filepaths or content hashes are duplicated
- Populated `jaguar_id` and `ground_truth.label` disagree, or only one of the pair is populated
- A body mask or bounding box is malformed
- The constructed sample count differs from the terminal export count

Missing and ambiguous lineage are reported but are not fatal.

Each run report records resolved paths, source and constructed counts, hash and media validation, lineage counts and methods, field-population counts, saved-view creation, and any failure.

## Components

The implementation uses focused modules under `jaguars.visualization`:

- Export parsing converts the terminal FiftyOne export into plain immutable records without opening the production database.
- Lineage enrichment builds exact indexes over upstream exports and manifests, applies precedence rules, and returns enrichment plus status.
- Validation checks media, hashes, annotations, schema invariants, and snapshot counts.
- Snapshot creation maps validated records to FiftyOne samples, creates saved views, and performs the atomic rename.
- The CLI owns storage configuration, mode validation, reporting, overwrite confirmation, and App lifecycle.

Pure parsing, enrichment, and validation code remains independently testable without FiftyOne database state.

## Testing and acceptance

Unit tests cover export parsing, exact and ambiguous lineage joins, schema mapping, content hashes, malformed annotations, duplicate detection, CLI validation, overwrite safety, and external-path guards.

An integration test uses a temporary isolated FiftyOne database and image fixtures to verify atomic creation, schema, annotations, saved views, immutability, and cleanup. Tests never connect to or mutate the production database.

Before production creation, a real-data `--dry-run` must verify:

- 1,367 terminal samples discovered
- 1,120 terminal identities populated and all 247 null terminal identities preserved unless uniquely resolved by exact lineage
- 1,367 unique readable media paths
- 1,367 unique SHA-256 values
- Valid final identity, bounding-box, and segmentation fields
- Complete lineage-status accounting

After creation, verification must confirm the same sample count, all required fields and saved views, identity agreement, content hashes, and successful App launch.
