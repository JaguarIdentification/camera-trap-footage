# FiftyOne Final Curated Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a guarded CLI that creates and launches the frozen `JaguarCameraTrap_Final_Curated_v1` FiftyOne snapshot from the 1,367 terminal segmented artifacts, with exact lineage enrichment and integrity validation.

**Architecture:** Pure modules parse exports, index lineage, validate annotations/media, and construct immutable records without touching a database. A FiftyOne adapter atomically creates the persistent snapshot and saved views. A thin CLI owns mount/storage checks, reporting, overwrite confirmation, and App lifecycle.

**Tech Stack:** Python 3.10, FiftyOne, Pillow, pytest, standard-library `argparse`, `csv`, `dataclasses`, `hashlib`, `json`, and `pathlib`

---

## File structure

- Create `src/jaguars/visualization/__init__.py`: visualization package marker.
- Create `src/jaguars/visualization/final_records.py`: typed terminal/export records and parsing.
- Create `src/jaguars/visualization/final_lineage.py`: exact lineage indexes and enrichment.
- Create `src/jaguars/visualization/final_validation.py`: media, hash, annotation, duplicate, and path guards.
- Create `src/jaguars/visualization/final_snapshot.py`: FiftyOne sample mapping, saved views, and atomic creation.
- Create `src/jaguars/visualization/final_dataset.py`: CLI, configuration, reports, confirmation, and App lifecycle.
- Create `tests/unit/visualization/__init__.py`: test package marker.
- Create `tests/unit/visualization/conftest.py`: small export/image fixtures.
- Create `tests/unit/visualization/test_final_records.py`: terminal export parsing tests.
- Create `tests/unit/visualization/test_final_lineage.py`: exact matching and enrichment tests.
- Create `tests/unit/visualization/test_final_validation.py`: validation and storage safety tests.
- Create `tests/unit/visualization/test_final_dataset.py`: CLI-mode and overwrite tests.
- Create `tests/integration/visualization/test_final_snapshot.py`: isolated FiftyOne snapshot test.
- Modify `README.md`: document the final-snapshot command and safety model.

### Task 1: Parse terminal artifacts into plain records

**Files:**
- Create: `src/jaguars/visualization/__init__.py`
- Create: `src/jaguars/visualization/final_records.py`
- Create: `tests/unit/visualization/__init__.py`
- Create: `tests/unit/visualization/conftest.py`
- Create: `tests/unit/visualization/test_final_records.py`

- [ ] **Step 1: Write failing export-parser tests**

Create fixtures with two tiny JPEGs and a `samples.json` containing `filepath`, optional `jaguar_id`, `bboxes_body`, and `segmentations_body`. Test path resolution, source IDs, parsed annotations, deterministic order, preservation of missing/null identity, and rejection of malformed populated identity:

```python
def test_load_terminal_records_resolves_paths_and_annotations(terminal_export: Path) -> None:
    records = load_terminal_records(terminal_export)
    assert [record.relative_filepath for record in records] == ["data/a.jpg", "data/b.jpg"]
    assert records[0].jaguar_id == "F11"
    assert records[0].bboxes_body["detections"][0]["label"] == "jaguar"
    assert records[0].segmentations_body["detections"][0]["mask"]


def test_load_terminal_records_rejects_missing_identity(terminal_export: Path) -> None:
    samples_path = terminal_export / "samples.json"
    payload = json.loads(samples_path.read_text())
    del payload["samples"][0]["jaguar_id"]
    samples_path.write_text(json.dumps(payload))
    with pytest.raises(TerminalExportError, match="jaguar_id"):
        load_terminal_records(terminal_export)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/unit/visualization/test_final_records.py -v
```

Expected: collection fails because `jaguars.visualization.final_records` does not exist.

- [ ] **Step 3: Implement typed records and parsing**

Implement frozen dataclasses and a parser that preserves annotation dictionaries but drops transient FiftyOne fields:

```python
@dataclass(frozen=True)
class TerminalRecord:
    source_id: str
    filepath: Path
    relative_filepath: str
    jaguar_id: str | None
    bboxes_body: dict[str, Any]
    segmentations_body: dict[str, Any]


def load_terminal_records(export_dir: Path) -> list[TerminalRecord]:
    payload = json.loads((export_dir / "samples.json").read_text(encoding="utf-8"))
    records = [_parse_terminal_sample(export_dir, sample) for sample in payload["samples"]]
    return sorted(records, key=lambda record: record.relative_filepath)
```

Resolve media below the export directory, reject path traversal, require nonempty identities and annotation containers, and extract Mongo IDs from `{"$oid": "..."}` without carrying `_dataset_id`, `_rand`, timestamps, or tags.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/visualization/test_final_records.py -v
```

Expected: all parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaguars/visualization tests/unit/visualization
git commit -m "feat: parse final curated artifacts"
```

### Task 2: Implement deterministic lineage enrichment

**Files:**
- Create: `src/jaguars/visualization/final_lineage.py`
- Create: `tests/unit/visualization/test_final_lineage.py`
- Modify: `tests/unit/visualization/conftest.py`

- [ ] **Step 1: Write failing lineage tests**

Test precedence, unique filename fallback, ambiguity, missing lineage, CSV normalization, both split fields, and match-method reporting:

```python
def test_exact_relative_path_precedes_filename_match() -> None:
    index = LineageIndex.from_candidates(
        [
            candidate("one", export_relative_path="data/a.jpg", original_filename="same.jpg"),
            candidate("two", export_relative_path="data/b.jpg", original_filename="same.jpg"),
        ]
    )
    result = index.enrich(terminal("data/a.jpg"))
    assert result.status == "matched"
    assert result.match_method == "export_relative_filepath"
    assert result.fields["sighting_id"] == "one"


def test_duplicate_filename_is_ambiguous() -> None:
    index = LineageIndex.from_candidates(
        [candidate("one", original_filename="same.jpg"), candidate("two", original_filename="same.jpg")]
    )
    result = index.enrich(terminal("data/same.jpg"))
    assert result.status == "ambiguous"
    assert result.match_method is None


def test_manifest_fields_add_open_and_closed_splits(tmp_path: Path) -> None:
    result = load_manifest_candidates(manifest_with_splits(tmp_path))
    assert result[0].closed_set_split == "train"
    assert result[0].open_set_split == "test"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/unit/visualization/test_final_lineage.py -v
```

Expected: failure because the lineage API is missing.

- [ ] **Step 3: Implement lineage candidates and indexes**

Define normalized final fields and multi-value indexes:

```python
@dataclass(frozen=True)
class Enrichment:
    status: Literal["matched", "ambiguous", "missing"]
    match_method: str | None
    fields: Mapping[str, Scalar]


MATCHERS = (
    "source_id",
    "normalized_source_filepath",
    "export_relative_filepath",
    "unique_filename",
)
```

Parse upstream `samples.json` records from `exports/segmented_deduplicated`, `exports/segmented`, `exports/deduplicated`, and `ingested`, then parse both CSV manifests. Normalize alternate column names into the approved schema. Build indexes whose values are lists so ambiguity is explicit. At each precedence level: return one candidate when unique, return `ambiguous` when multiple distinct candidates match, otherwise continue. Merge a unique export candidate with a unique manifest candidate only when their identity and original filename agree.

- [ ] **Step 4: Run lineage tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/visualization/test_final_lineage.py -v
```

Expected: all lineage tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaguars/visualization/final_lineage.py tests/unit/visualization
git commit -m "feat: enrich final artifacts with exact lineage"
```

### Task 3: Validate media, annotations, duplicates, and storage

**Files:**
- Create: `src/jaguars/visualization/final_validation.py`
- Create: `tests/unit/visualization/test_final_validation.py`

- [ ] **Step 1: Write failing validation tests**

Cover SHA-256, image dimensions, unreadable files, duplicate path/hash rejection, bbox bounds, required RLE masks, mount checks through an injected mount predicate, and generated paths escaping the approved root:

```python
def test_validate_media_returns_integrity_fields(image_path: Path) -> None:
    integrity = validate_media(image_path)
    assert integrity.sha256 == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert integrity.width == 8
    assert integrity.height == 6
    assert integrity.size_bytes == image_path.stat().st_size


def test_duplicate_hashes_are_rejected(records_with_same_bytes: list[ValidatedRecord]) -> None:
    with pytest.raises(IntegrityError, match="duplicate SHA-256"):
        validate_unique_records(records_with_same_bytes)


def test_storage_paths_must_remain_under_external_root(tmp_path: Path) -> None:
    with pytest.raises(StorageSafetyError, match="outside"):
        validate_storage_paths(tmp_path / "mongo", Path("/Volumes/CameraTrapPython/fiftyone"))
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/unit/visualization/test_final_validation.py -v
```

Expected: failure because validation functions do not exist.

- [ ] **Step 3: Implement validation**

Use Pillow `Image.verify()` followed by a fresh open for dimensions, stream SHA-256 in 1 MiB chunks, require normalized `[x, y, width, height]` values within image-relative bounds, and require every segmentation detection to contain a nonempty serialized mask. Aggregate all errors before raising:

```python
def validate_unique_records(records: Sequence[ValidatedRecord]) -> None:
    duplicate_paths = _duplicates(record.terminal.filepath for record in records)
    duplicate_hashes = _duplicates(record.integrity.sha256 for record in records)
    if duplicate_paths or duplicate_hashes:
        raise IntegrityError(_format_duplicate_error(duplicate_paths, duplicate_hashes))
```

Implement `validate_mounts(paths, is_mount=os.path.ismount)` and a `Path.resolve().is_relative_to(approved_root.resolve())` guard for database, report, and dataset-default directories.

- [ ] **Step 4: Run validation tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/visualization/test_final_validation.py -v
```

Expected: all validation tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaguars/visualization/final_validation.py tests/unit/visualization/test_final_validation.py
git commit -m "feat: validate final snapshot integrity"
```

### Task 4: Build the atomic FiftyOne snapshot

**Files:**
- Create: `src/jaguars/visualization/final_snapshot.py`
- Create: `tests/integration/visualization/test_final_snapshot.py`

- [ ] **Step 1: Write a failing isolated integration test**

Configure a temporary FiftyOne database before importing FiftyOne, create three image fixtures, and test fields, annotations, saved views, count validation, rename, and temporary cleanup:

```python
def test_create_snapshot_is_atomic_and_complete(
    isolated_fiftyone: None,
    validated_records: list[ValidatedRecord],
) -> None:
    dataset = create_snapshot(
        records=validated_records,
        dataset_name="test_final_snapshot",
        temporary_name="test_final_snapshot__building",
    )
    assert dataset.name == "test_final_snapshot"
    assert dataset.persistent
    assert len(dataset) == 3
    assert dataset.first().ground_truth.label == dataset.first().jaguar_id
    assert set(dataset.list_saved_views()) == EXPECTED_SAVED_VIEWS
    assert "test_final_snapshot__building" not in fo.list_datasets()
```

Also force validation failure and assert neither final nor owned staging dataset remains. Add adversarial real-database cases for generated staging-name collisions, constructor races, and rename exceptions both before and after persistence. Assert cleanup occurs only when the generated dataset ID and unguessable metadata token prove ownership; an unproven post-metadata constructor artifact must be retained and reported rather than deleted.

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```bash
uv run pytest tests/integration/visualization/test_final_snapshot.py -v
```

Expected: failure because snapshot creation is missing.

- [ ] **Step 3: Implement sample mapping and atomic creation**

Map approved fields explicitly and reconstruct FiftyOne labels from the parsed dictionaries:

```python
sample = fo.Sample(filepath=str(record.terminal.filepath))
sample["jaguar_id"] = record.resolved_jaguar_id
sample["ground_truth"] = (
    None
    if record.resolved_jaguar_id is None
    else fo.Classification(label=record.resolved_jaguar_id)
)
sample["bboxes_body"] = deserialize_detections(record.terminal.bboxes_body)
sample["segmentations_body"] = deserialize_detections(record.terminal.segmentations_body)
sample["lineage_status"] = record.enrichment.status
sample["lineage_match_method"] = record.enrichment.match_method
```

Define all approved sample fields before batched insertion, populate integrity and enrichment fields, create eight saved views, validate count and identity agreement, set `persistent = True`, and rename the temporary dataset only after validation. Explicitly collision-check the generated build name, persist an unguessable ownership token in dataset metadata, and require a database ID/token match before cleanup. Re-query database state after rename exceptions instead of trusting the mutable in-memory dataset name.

- [ ] **Step 4: Run the integration test and verify GREEN**

Run:

```bash
uv run pytest tests/integration/visualization/test_final_snapshot.py -v
```

Expected: all snapshot integration tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaguars/visualization/final_snapshot.py tests/integration/visualization
git commit -m "feat: create atomic final FiftyOne snapshot"
```

### Task 5: Add guarded CLI, reporting, and overwrite policy

**Files:**
- Create: `src/jaguars/visualization/final_dataset.py`
- Create: `tests/unit/visualization/test_final_dataset.py`

- [ ] **Step 1: Write failing CLI and confirmation tests**

Test incompatible modes, `--yes` without `--overwrite`, existing-dataset refusal, exact-name overwrite, interactive decline, dry-run avoiding FiftyOne writes, and report serialization:

```python
@pytest.mark.parametrize(
    "argv",
    [
        ["--dry-run", "--launch-only"],
        ["--create-only", "--launch-only"],
        ["--yes"],
    ],
)
def test_invalid_mode_combinations_exit(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_overwrite_requires_exact_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "wrong-name")
    assert confirm_overwrite("JaguarCameraTrap_Final_Curated_v1", 1367, 1367) is False
```

- [ ] **Step 2: Run the CLI tests and verify RED**

Run:

```bash
uv run pytest tests/unit/visualization/test_final_dataset.py -v
```

Expected: failure because the CLI module does not exist.

- [ ] **Step 3: Implement configuration, orchestration, and JSON reports**

Provide constants for the approved data/export/storage paths and dataset name. Configure FiftyOne environment variables before importing `fiftyone`. Keep orchestration injectable for tests:

```python
def run(args: argparse.Namespace, services: Services = DEFAULT_SERVICES) -> int:
    paths = resolve_paths(args)
    validate_runtime(paths, args)
    audit = build_validated_records(paths)
    report = RunReport.from_audit(audit)
    if args.dry_run:
        write_report(report, paths.report_dir)
        return 0
    dataset = create_or_replace_snapshot(audit, args, services)
    verify_snapshot(dataset, audit)
    write_report(report.completed(dataset), paths.report_dir)
    if not args.create_only:
        launch_and_wait(dataset, args.address, args.port)
    return 0
```

On overwrite, audit first, show existing/proposed counts, and require the exact dataset name interactively unless `--yes`. Pass the confirmed replacement request to the snapshot adapter, which builds and validates unique staging while the old final remains published, then performs a rollback-safe old-to-backup/staging-to-final swap. It deletes only the owned old backup after successful promotion and never deletes media. `--launch-only` validates mounts and loads the existing snapshot without auditing media. Install SIGINT-safe App session cleanup.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/visualization/test_final_dataset.py -v
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaguars/visualization/final_dataset.py tests/unit/visualization/test_final_dataset.py
git commit -m "feat: add final dataset CLI"
```

### Task 6: Document operation and run static/unit verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the documented command**

Document the snapshot boundary, exact default command, modes, external mounts, overwrite behavior, and report location:

```markdown
### Final curated FiftyOne snapshot

Create and launch the immutable v1 snapshot:

`uv run python -m jaguars.visualization.final_dataset`

Use `--dry-run` before first creation. The command references terminal media in place and stores all generated FiftyOne state below `/Volumes/CameraTrapPython/fiftyone`.
```

- [ ] **Step 2: Run formatting and lint checks**

Run:

```bash
uv run black --check src/jaguars/visualization tests/unit/visualization tests/integration/visualization
uv run ruff check src/jaguars/visualization tests/unit/visualization tests/integration/visualization
uv run mypy src/jaguars/visualization
```

Expected: all commands exit 0. Fix only issues in files introduced by this feature.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run pytest tests/unit/visualization tests/integration/visualization -v
```

Expected: all visualization tests pass.

- [ ] **Step 4: Run the complete unit suite**

Run:

```bash
uv run pytest tests/unit -v
```

Expected: all unit tests pass.

- [ ] **Step 5: Commit**

```bash
git add README.md src/jaguars/visualization tests/unit/visualization tests/integration/visualization
git commit -m "docs: document final curated snapshot"
```

### Task 7: Audit and create the real frozen snapshot

**Files:**
- Runtime report: `/Volumes/CameraTrapPython/fiftyone/JaguarCameraTrap_Final_Curated_v1/*.json`

- [ ] **Step 1: Run the real-data dry-run**

Run:

```bash
uv run python -m jaguars.visualization.final_dataset --dry-run
```

Expected: exit 0; report shows exactly 1,367 terminal samples, 1,367 unique readable paths, 1,367 unique hashes, valid identities/annotations, and lineage totals summing to 1,367. If the unique-hash requirement fails, stop and report the duplicate files; do not weaken the approved invariant.

- [ ] **Step 2: Inspect the dry-run report**

Run:

```bash
jq '{counts, lineage, validation, paths}' /Volumes/CameraTrapPython/fiftyone/JaguarCameraTrap_Final_Curated_v1/latest.json
```

Expected: no validation errors, approved resolved storage paths, and complete lineage accounting.

- [ ] **Step 3: Create without launching**

Run:

```bash
uv run python -m jaguars.visualization.final_dataset --create-only
```

Expected: exit 0 and persistent dataset `JaguarCameraTrap_Final_Curated_v1` created atomically. If the name already exists, stop rather than overwriting unless the user separately authorizes the approved overwrite flow.

- [ ] **Step 4: Verify the production snapshot**

Run:

```bash
uv run python -m jaguars.visualization.final_dataset --launch-only --port 5151
```

Expected: FiftyOne launches with 1,367 samples and the eight saved views. Verify `http://localhost:5151` responds, then interrupt once and confirm clean App shutdown without dataset deletion.

- [ ] **Step 5: Review the final diff and status**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only pre-existing unrelated user changes remain uncommitted.
