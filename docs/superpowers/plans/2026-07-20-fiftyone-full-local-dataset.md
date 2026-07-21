# FiftyOne Full Local Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one safe, repeatable command that incrementally catalogs every valid local image/video path under `data` and launches the persistent catalog in FiftyOne without copying media.

**Architecture:** Pure discovery and provenance modules build deterministic catalog records without importing FiftyOne. A narrow FiftyOne adapter owns persistence and synchronization, while a CLI module validates external mounts, configures external state, coordinates synchronization, writes reports, and launches the App.

**Tech Stack:** Python 3.10, pathlib, dataclasses, csv, Pillow EXIF, `fcntl`, FiftyOne 1.19.0, pytest, ruff, mypy

## Global Constraints

- Include every readable local media path under `data`; identical bytes at different paths remain separate samples.
- Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`; supported video extensions are `.mp4`, `.avi`, `.mov`, `.mkv`, case-insensitively.
- Exclude AppleDouble files, broken symlinks, unsupported files, Git LFS pointers, and Parquet shards.
- Do not copy media or run inference, embeddings, segmentation, sampling, or deduplication.
- Dataset name defaults to `JaguarCameraTrap_Full_Local`; normalized absolute filepath is the stable identity.
- MongoDB, reports, locks, and dataset defaults must resolve beneath `/Volumes/CameraTrapPython/fiftyone`.
- Media remains beneath the repository `data` symlink backed by `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data`.
- Require both `/Volumes/Extreme SSD` and `/Volumes/CameraTrapPython` to be actual mounted filesystems before synchronization or launch.
- Missing media is marked unavailable by default; deletion requires `--prune-missing` plus explicit confirmation.
- Preserve manifest and EXIF timestamps separately; never infer timezone or silently choose an authoritative timestamp.
- Existing exported FiftyOne datasets must not be modified.

---

### Task 1: Deterministic Media Discovery and Classification

**Files:**
- Create: `src/jaguars/visualization/__init__.py`
- Create: `src/jaguars/visualization/catalog.py`
- Create: `tests/unit/visualization/__init__.py`
- Create: `tests/unit/visualization/test_catalog.py`

**Interfaces:**
- Produces: `CatalogRecord`, `DiscoveryFailure`, `DiscoveryResult`, `stable_path_key()`, `classify_path()`, and `discover_media()`.
- Consumes: A resolved data root and filesystem paths only; no FiftyOne imports.

- [ ] **Step 1: Write failing discovery tests**

Create `tests/unit/visualization/test_catalog.py` with fixtures that write a valid JPEG via Pillow, a minimal video fixture as nonempty bytes, an AppleDouble file, an unsupported CSV, a 134-byte Git LFS pointer named `.parquet`, and a broken symlink. Assert:

```python
result = discover_media(data_root)
assert [record.relative_path for record in result.records] == [
    "intermediate/v1/files/frame.JPG",
    "raw/17_11_2025/sites/site-a/clip.mp4",
]
assert result.records[0].media_type == "image"
assert result.records[1].media_type == "video"
assert result.records[0].stable_key == stable_path_key(result.records[0].filepath)
assert not result.failures
```

Add separate tests that two byte-identical files have different stable keys, extension matching is case-insensitive, and classification returns:

```python
assert classify_path(Path("raw/17_11_2025/sites/a/cat.jpg")) == ("raw", "original")
assert classify_path(Path("intermediate/v1/screenshots/a.png")) == ("screenshots", "unknown")
assert classify_path(Path("intermediate/v1/fo_jaguars/sam3/data/a.jpg")) == ("fiftyone_export", "segmented_body")
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run:

```bash
.venv/bin/pytest tests/unit/visualization/test_catalog.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'jaguars.visualization'`.

- [ ] **Step 3: Implement immutable catalog records and stable identity**

In `catalog.py`, define frozen dataclasses with typed fields:

```python
@dataclass(frozen=True)
class CatalogRecord:
    stable_key: str
    filepath: Path
    relative_path: str
    media_type: Literal["image", "video"]
    extension: str
    size_bytes: int
    modified_ns: int
    source_collection: str
    variant: str

@dataclass(frozen=True)
class DiscoveryFailure:
    filepath: Path
    reason: str

@dataclass(frozen=True)
class DiscoveryResult:
    records: tuple[CatalogRecord, ...]
    failures: tuple[DiscoveryFailure, ...]
```

Implement `stable_path_key(path)` as SHA-256 of `str(path.resolve(strict=True))`. Implement the specification's source/variant rules in `classify_path(relative_path)` using path components, not substring matching against the entire absolute path.

- [ ] **Step 4: Implement deterministic discovery**

Implement:

```python
def discover_media(data_root: Path) -> DiscoveryResult:
    """Discover supported readable media beneath a resolved data root."""
```

Require `data_root.resolve(strict=True).is_dir()`. Walk with `Path.rglob("*")`, reject any path with an AppleDouble basename, skip symlinks whose targets do not exist, and only admit supported extensions. Confirm images with `PIL.Image.verify()`. For videos, require a regular file and nonzero size; detailed video decoding belongs to FiftyOne metadata computation, not discovery. Sort accepted records and failures by normalized absolute path.

- [ ] **Step 5: Run focused checks**

Run:

```bash
.venv/bin/pytest tests/unit/visualization/test_catalog.py -q
.venv/bin/ruff check src/jaguars/visualization/catalog.py tests/unit/visualization/test_catalog.py
.venv/bin/mypy src/jaguars/visualization/catalog.py
```

Expected: all tests pass; ruff and mypy exit 0.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/jaguars/visualization/__init__.py src/jaguars/visualization/catalog.py tests/unit/visualization/__init__.py tests/unit/visualization/test_catalog.py
git commit -m "feat: discover local media for FiftyOne"
```

### Task 2: Manifest and Capture-Time Provenance Enrichment

**Files:**
- Create: `src/jaguars/visualization/provenance.py`
- Create: `tests/unit/visualization/test_provenance.py`

**Interfaces:**
- Consumes: `CatalogRecord` tuples and the two manifest paths.
- Produces: `EnrichedRecord`, `ManifestRow`, `load_manifest_rows()`, `enrich_records()`, and `timestamp_agreement()`.

- [ ] **Step 1: Write failing provenance tests**

Build temporary split CSVs with the actual uppercase headers and assert exact-path matching takes precedence over filename-only matching. Cover an unambiguous filename fallback, two rows sharing one filename, unmatched media, missing timestamps, and conflicting timestamps.

Required assertions include:

```python
assert enriched.closed_set_split == "train"
assert enriched.open_set_split == "test"
assert enriched.manifest_datetime == "2024-07-11 19:11:03"
assert enriched.exif_datetime == "2023-07-11 19:11:03"
assert enriched.timestamp_status == "mismatch"
assert ambiguous.provenance_status == "ambiguous"
assert ambiguous.manifest_candidate_ids == ("camera:1", "pptx:4")
```

Test every timestamp status: `match`, `mismatch`, `manifest_only`, `exif_only`, and `missing`.

- [ ] **Step 2: Run the tests and verify failure**

```bash
.venv/bin/pytest tests/unit/visualization/test_provenance.py -q
```

Expected: collection fails because `jaguars.visualization.provenance` does not exist.

- [ ] **Step 3: Implement manifest normalization**

Define frozen `ManifestRow` and `EnrichedRecord` dataclasses. `EnrichedRecord` contains the underlying `CatalogRecord`, provenance status, candidate IDs, identity, both split fields, camera/site/location fields, coordinates, original filename/path, sighting ID, manifest date/time/datetime, EXIF datetime, and timestamp status.

Implement:

```python
def load_manifest_rows(paths: Sequence[Path]) -> tuple[ManifestRow, ...]:
    ...

def enrich_records(records: Sequence[CatalogRecord], manifest_rows: Sequence[ManifestRow]) -> tuple[EnrichedRecord, ...]:
    ...
```

Read CSV with `encoding="utf-8-sig"`. Normalize headers with `.strip()` while preserving values. Generate stable manifest row IDs from manifest kind and one-based data row number.

- [ ] **Step 4: Implement precedence-safe joins and EXIF extraction**

Build indexes once per enrichment call:

1. Resolved `FILE PATH`/`RAW FILE PATH` candidates relative to the manifest dataset roots.
2. Normalized repository-relative paths.
3. Case-folded `ORIGINAL FILE NAME`/`FILE NAME` fallback.

Accept a match only when the highest-precedence nonempty candidate set contains exactly one row. Multiple candidates yield `ambiguous`; no candidates yield `unmatched`. Read image EXIF keys in precedence order `DateTimeOriginal`, `DateTimeDigitized`, `DateTime`, preserving the raw EXIF string after normalizing only its separator into `YYYY-MM-DD HH:MM:SS` when parseable.

- [ ] **Step 5: Run focused checks**

```bash
.venv/bin/pytest tests/unit/visualization/test_provenance.py -q
.venv/bin/ruff check src/jaguars/visualization/provenance.py tests/unit/visualization/test_provenance.py
.venv/bin/mypy src/jaguars/visualization/provenance.py
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/jaguars/visualization/provenance.py tests/unit/visualization/test_provenance.py
git commit -m "feat: enrich FiftyOne catalog provenance"
```

### Task 3: External Storage Guards and Run Reporting

**Files:**
- Create: `src/jaguars/visualization/runtime.py`
- Create: `tests/unit/visualization/test_runtime.py`

**Interfaces:**
- Produces: `RuntimePaths`, `validate_runtime_paths()`, `exclusive_sync_lock()`, and `write_run_report()`.
- Consumes: Explicit mount roots, data path, dataset name, and serializable report dictionaries.

- [ ] **Step 1: Write failing runtime tests**

Use dependency injection for the mount table rather than relying on `/Volumes` in unit tests. Assert that validation rejects an existing directory not present in the supplied mount set, rejects database/report roots escaping the approved root through symlinks, and accepts paths beneath the approved APFS mount.

Test `exclusive_sync_lock()` with `fcntl.flock(..., LOCK_EX | LOCK_NB)` by acquiring the same lock twice and expecting a domain-specific `SynchronizationLockedError`. Test atomic report writing by asserting the final JSON parses and no temporary file remains.

- [ ] **Step 2: Run and verify failure**

```bash
.venv/bin/pytest tests/unit/visualization/test_runtime.py -q
```

Expected: collection fails because `jaguars.visualization.runtime` does not exist.

- [ ] **Step 3: Implement runtime path validation**

Define:

```python
@dataclass(frozen=True)
class RuntimePaths:
    data_root: Path
    state_root: Path
    database_dir: Path
    dataset_dir: Path
    report_dir: Path
    lock_path: Path
```

`validate_runtime_paths()` must parse actual mountpoints from `mount` output in production and accept a supplied mount set in tests. Resolve every configured path with `strict=False`, use `Path.is_relative_to()` against `/Volumes/CameraTrapPython/fiftyone`, and require the resolved data root beneath the configured Extreme SSD data root.

- [ ] **Step 4: Implement locking and reports**

Use only the standard library `fcntl` module. Keep the lock file handle open for the entire context manager lifetime. Write JSON reports to a same-directory temporary file, `flush()`, `os.fsync()`, and `os.replace()` into a filename containing UTC start time and dataset slug.

- [ ] **Step 5: Run focused checks and commit**

```bash
.venv/bin/pytest tests/unit/visualization/test_runtime.py -q
.venv/bin/ruff check src/jaguars/visualization/runtime.py tests/unit/visualization/test_runtime.py
.venv/bin/mypy src/jaguars/visualization/runtime.py
git add src/jaguars/visualization/runtime.py tests/unit/visualization/test_runtime.py
git commit -m "feat: guard external FiftyOne runtime state"
```

### Task 4: Incremental FiftyOne Synchronizer

**Files:**
- Create: `src/jaguars/visualization/fiftyone_catalog.py`
- Create: `tests/unit/visualization/test_fiftyone_catalog.py`

**Interfaces:**
- Consumes: `EnrichedRecord` values and a persistent FiftyOne dataset name.
- Produces: `SyncPlan`, `SyncResult`, `plan_sync()`, and `synchronize_dataset()`.

- [ ] **Step 1: Write failing pure diff tests**

Define serialized existing rows as plain mappings so `plan_sync()` stays testable without MongoDB. Assert:

```python
plan = plan_sync(discovered, existing)
assert tuple(r.catalog.stable_key for r in plan.new) == ("new-key",)
assert tuple(r.catalog.stable_key for r in plan.changed) == ("changed-key",)
assert plan.unchanged_keys == ("same-key",)
assert plan.missing_keys == ("gone-key",)
```

Changes must be detected from a deterministic fingerprint over filesystem facts and enriched metadata, not FiftyOne internal timestamps.

- [ ] **Step 2: Write isolated FiftyOne integration tests**

Monkeypatch/configure FiftyOne to use a temporary database directory before importing the adapter. Create tiny image and video fixtures and verify:

- First sync inserts both paths.
- Second sync inserts zero and reports both unchanged.
- Editing a metadata field updates exactly one sample.
- Removing a fixture marks `available=False` without deleting the sample.
- A custom annotation field survives update and missing marking.
- `prune_missing=True, confirmed=False` raises `PruneConfirmationRequired`.
- Confirmed prune deletes only missing samples.

Use a unique dataset name per test and delete it in `finally`.

- [ ] **Step 3: Run and verify failures**

```bash
.venv/bin/pytest tests/unit/visualization/test_fiftyone_catalog.py -q
```

Expected: collection fails because the adapter does not exist.

- [ ] **Step 4: Implement schema and pure synchronization planning**

Define `SyncPlan` and `SyncResult` dataclasses. Implement `plan_sync()` with stable-key dictionaries and sorted tuple outputs. Define catalog-owned fields including `catalog_key`, `catalog_fingerprint`, `available`, `relative_path`, `source_collection`, `variant`, provenance fields, split fields, and timestamps.

Refuse to operate when an existing dataset has a non-string `catalog_key` field or duplicate catalog keys. Do not alter fields outside the catalog-owned allowlist.

- [ ] **Step 5: Implement batched create/update/missing/prune operations**

Create a grouped persistent dataset when absent by adding a `group` field with `image` and `video` slices. Give each physical path its own `fo.Group`, and set the sample's group element to the slice matching `media_type`; this preserves one catalog sample per path while satisfying FiftyOne's mixed-media model. Build `fo.Sample` values referencing absolute filepaths without copying. Use batches of 500 for insertion/update. Update only catalog-owned fields on existing samples. Mark missing samples unavailable in a separate batch. Require both `prune_missing=True` and `confirmed=True` before deletion.

- [ ] **Step 6: Run focused checks and commit**

```bash
.venv/bin/pytest tests/unit/visualization/test_fiftyone_catalog.py -q
.venv/bin/ruff check src/jaguars/visualization/fiftyone_catalog.py tests/unit/visualization/test_fiftyone_catalog.py
.venv/bin/mypy src/jaguars/visualization/fiftyone_catalog.py
git add src/jaguars/visualization/fiftyone_catalog.py tests/unit/visualization/test_fiftyone_catalog.py
git commit -m "feat: synchronize persistent FiftyOne catalog"
```

### Task 5: CLI Orchestration and FiftyOne App Launch

**Files:**
- Create: `src/jaguars/visualization/full_dataset.py`
- Create: `tests/unit/visualization/test_full_dataset.py`

**Interfaces:**
- Consumes: Tasks 1-4 interfaces.
- Produces: `build_parser()`, `run()`, and `main()`; executable with `python -m jaguars.visualization.full_dataset`.

- [ ] **Step 1: Write failing CLI tests**

Assert parser defaults and invalid combinations:

```python
args = build_parser().parse_args([])
assert args.dataset_name == "JaguarCameraTrap_Full_Local"
assert args.address == "localhost"
assert args.port == 5151
```

Test that `--sync-only --no-sync`, `--no-sync --prune-missing`, and an out-of-range port cause parser errors. With mocked component interfaces, assert stage order is validate → discover → enrich → synchronize → launch. Assert `--dry-run` never imports/starts FiftyOne or writes database state, `--sync-only` does not launch, and `--no-sync` skips discovery and synchronization.

- [ ] **Step 2: Run and verify failure**

```bash
.venv/bin/pytest tests/unit/visualization/test_full_dataset.py -q
```

Expected: collection fails because the CLI module does not exist.

- [ ] **Step 3: Implement argument validation and external configuration**

Construct default `RuntimePaths` from the exact Global Constraints. Before importing `fiftyone`, set:

```python
os.environ["FIFTYONE_DATABASE_DIR"] = str(paths.database_dir)
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(paths.dataset_dir)
os.environ["FIFTYONE_DEFAULT_DATASET_DIR"] = str(paths.dataset_dir)
```

Then import FiftyOne lazily. Validate `1 <= port <= 65535`. Add `--yes-prune` as the deliberate noninteractive confirmation flag; without it, prompt with the exact missing count and require the literal response `prune`.

- [ ] **Step 4: Implement orchestration, summaries, and clean session lifetime**

`run()` records UTC start/finish times and all report fields required by the spec. Print inventory counts with `collections.Counter`. For launch:

```python
session = fo.launch_app(dataset, address=args.address, port=args.port)
try:
    session.wait()
finally:
    session.close()
```

Catch `KeyboardInterrupt` as a clean exit. Convert known mount, lock, schema, and port failures into concise stderr messages and exit code 2; unexpected exceptions retain traceback logging and exit code 1.

- [ ] **Step 5: Run focused checks and commit**

```bash
.venv/bin/pytest tests/unit/visualization/test_full_dataset.py -q
.venv/bin/ruff check src/jaguars/visualization/full_dataset.py tests/unit/visualization/test_full_dataset.py
.venv/bin/mypy src/jaguars/visualization/full_dataset.py
git add src/jaguars/visualization/full_dataset.py tests/unit/visualization/test_full_dataset.py
git commit -m "feat: launch full local dataset in FiftyOne"
```

### Task 6: Operator Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Create: `docs/fiftyone-full-local-dataset.md`
- Test: `tests/unit/visualization/`

**Interfaces:**
- Consumes: Completed CLI.
- Produces: Operator instructions and verified production launch.

- [ ] **Step 1: Document prerequisites and commands**

Add a concise README link and create `docs/fiftyone-full-local-dataset.md` covering:

```bash
uv run python -m jaguars.visualization.full_dataset --dry-run
uv run python -m jaguars.visualization.full_dataset --sync-only
uv run python -m jaguars.visualization.full_dataset
uv run python -m jaguars.visualization.full_dataset --no-sync
```

Document both required mounts, the external state layout, first-sync expectations, persistent incremental behavior, split/time-provenance filters, clean shutdown, explicit pruning, and the limitation that local Hugging Face Parquet files are LFS pointers.

- [ ] **Step 2: Run the visualization test suite**

```bash
.venv/bin/pytest tests/unit/visualization -q
.venv/bin/ruff check src/jaguars/visualization tests/unit/visualization
.venv/bin/mypy src/jaguars/visualization
```

Expected: all tests pass and both static checks exit 0.

- [ ] **Step 3: Run a production dry run and record independent inventory**

```bash
uv run python -m jaguars.visualization.full_dataset --dry-run
find -L data -type f ! -name '._*' | awk 'BEGIN{IGNORECASE=1} /\.(jpg|jpeg|png|bmp|tif|tiff|mp4|avi|mov|mkv)$/ {count++} END{print count}'
```

Expected: the CLI valid-media count equals the independent inventory after unreadable-media exclusions are reported explicitly.

- [ ] **Step 4: Synchronize the production catalog twice**

```bash
uv run python -m jaguars.visualization.full_dataset --sync-only
uv run python -m jaguars.visualization.full_dataset --sync-only
```

Expected: first run creates/updates the catalog; second run reports `new=0` and no duplicate stable keys. Both run reports are written beneath `/Volumes/CameraTrapPython/fiftyone/JaguarCameraTrap_Full_Local`.

- [ ] **Step 5: Launch and verify the App**

Start:

```bash
uv run python -m jaguars.visualization.full_dataset --no-sync
```

In another shell, run:

```bash
curl --fail --silent --show-error http://localhost:5151/ >/dev/null
```

Expected: curl exits 0. In the App, verify image and video samples render; `closed_set_split`, `open_set_split`, `timestamp_status`, `source_collection`, and `variant` are filterable. Interrupt the launcher and verify the session closes while the persistent dataset remains.

- [ ] **Step 6: Run broader non-FiftyOne regression tests and document baseline failures**

```bash
.venv/bin/pytest tests/unit/common tests/unit/reidentification/test_config.py -q
```

Expected: these focused unaffected suites pass. Do not claim the repository-wide suite is clean unless `.venv/bin/pytest tests` also passes; the pre-existing project baseline previously contained unrelated dependency and macOS/FiftyOne failures.

- [ ] **Step 7: Commit documentation**

```bash
git add README.md docs/fiftyone-full-local-dataset.md
git commit -m "docs: explain full local FiftyOne viewer"
```
