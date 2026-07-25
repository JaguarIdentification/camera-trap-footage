# FiftyOne Final Curated Dataset Implementation Plan

> Execute test-first in the isolated `final-curated-fiftyone` worktree. The
> original terminal export and production FiftyOne database are read-only.

**Goal:** Deterministically curate the defective 1,367-record terminal export
into a 1,322-record hardlinked export, then use that export as the only default
source for the frozen `JaguarCameraTrap_Final_Curated_v1` snapshot.

**Architecture:** A pure curation planner hashes and groups source records,
applies the approved representative/exclusion/clip/review policy, and returns a
validated immutable plan. A filesystem adapter materializes that plan
atomically with hardlinks and a complete sidecar. Existing parser, validator,
snapshot, and CLI modules consume the curated export and enforce the
media-only review exception.

## Approved acceptance target

- Source export remains unchanged:
  `data/intermediate/v1/fo_jaguars/labeled_segmented_jaguars_primitive`
- Curated target:
  `data/intermediate/v1/fo_jaguars/labeled_segmented_jaguars_final_curated_v1`
- Source records: 1,367
- Exact-content groups/drops: 39
- Confirmed false-positive drops: 6
- Curated records, unique paths, and unique hashes: 1,322
- Populated/null identities: 1,108 / 214
- Distinct non-null identities: 59
- Ordinary annotation-complete samples: 1,318
- Media-only pending-review samples: 4
- Audited bbox clips: 3
- Saved FiftyOne views: 9

No review case is re-segmented. `000001-143`, `000002-144`, `000010-18`, and
`000005-126` remain as media-only records with their malformed annotations
removed and explicit pending-review fields/tag.

## Task 1: Deterministic curation planning

**Files:**

- Create `src/jaguars/visualization/final_curation.py`
- Create `tests/unit/visualization/test_final_curation.py`

1. Write failing tests for semantic normalization, conflicting identities and
   annotations, selection preference order, and lexicographic tie-breaking.
2. Implement exact SHA-256 grouping and representative selection.
3. Write failing synthetic-plan tests covering one exact duplicate, one false
   positive, one clip, one review sample, and frozen counts.
4. Implement `CurationPolicy`, `CurationPlan`, drop/hash-group diagnostics, and
   count validation.
5. Run:

   ```bash
   uv run pytest tests/unit/visualization/test_final_curation.py -v
   ```

## Task 2: Atomic hardlink export construction

1. Write failing tests for hardlink inode equality, source immutability,
   deterministic rewritten samples, clips, stripped review annotations,
   sidecar completeness, unsupported hardlinks, partial-build cleanup, and
   guarded overwrite.
2. Materialize a sibling staging directory using `os.link` only.
3. Write deterministic sample IDs, paths internal to the target, minimal
   metadata, and `curation_report.json`.
4. Refuse an existing target unless `--create --overwrite` is confirmed;
   require `--yes` for noninteractive replacement.
5. Rename only after the complete staging export is written and validated.
6. Load the synthetic result through the terminal parser and batch validator.

## Task 3: Pending-review parser and validator contract

**Files:**

- Modify `src/jaguars/visualization/final_records.py`
- Modify `src/jaguars/visualization/final_validation.py`
- Modify corresponding unit tests

1. Write failing tests showing annotations may be absent only when:
   `review_required=True`, `review_reason` is nonempty,
   `review_status="pending"`, and `needs_annotation_review` is tagged.
2. Extend `TerminalRecord` with optional annotations and frozen review fields.
3. Preserve only the approved review tag; continue dropping transient source
   tags and system fields.
4. Reject partial annotation pairs, incomplete review metadata, or missing
   ordinary annotations.
5. Validate media and hashes normally for review samples.

## Task 4: Snapshot schema and saved review view

**Files:**

- Modify `src/jaguars/visualization/final_snapshot.py`
- Modify `tests/integration/visualization/test_final_snapshot.py`

1. Write failing isolated-FiftyOne tests for media-only review mapping,
   `BooleanField`/review strings, tag persistence, and `Annotation review`.
2. Add `review_required`, `review_reason`, and `review_status` to the approved
   schema.
3. Map review records with null body annotations and the review tag.
4. Require annotations on ordinary samples and the full review contract on
   review samples during prepublication validation.
5. Add `Annotation review` as the ninth saved view.
6. Run the full isolated integration file without any production database
   configuration.

## Task 5: Frozen snapshot defaults and reporting

**Files:**

- Modify `src/jaguars/visualization/final_dataset.py`
- Modify `tests/unit/visualization/test_final_dataset.py`
- Replace `tests/unit/visualization/test_final_real_data.py`

1. Change the default terminal source to the curated export.
2. Freeze expected counts at 1,322 / 1,108 / 214.
3. Report actual body-annotation and review-field population.
4. Replace the known-failure original-export test with a read-only curation
   acceptance test.
5. Keep a second real-data test that becomes active once the curated target
   exists and requires parser/validator green.

## Task 6: Safe curation CLI

1. Default `python -m jaguars.visualization.final_curation` to a read-only
   audit.
2. Support explicit `--dry-run`, `--create`, `--source`, `--target`,
   `--overwrite`, and `--yes`.
3. Load exact lineage for representative preference when operating on the
   default source.
4. Print the complete report on dry-run without writing.
5. Never open FiftyOne or touch its database.

## Task 7: Documentation and verification

**Files:**

- Update `README.md`
- Update the approved design and this plan

Run fresh verification:

```bash
uv run pytest tests/unit/visualization -v
uv run pytest tests/integration/visualization/test_final_snapshot.py -v
uv run black --check src/jaguars/visualization tests/unit/visualization tests/integration/visualization
uv run ruff check src/jaguars/visualization tests/unit/visualization tests/integration/visualization
uv run mypy src/jaguars/visualization
git diff --check
```

Run the mounted-data acceptance audit without writes:

```bash
uv run python -m jaguars.visualization.final_curation --dry-run
```

Expected: 1,322 curated records/hashes, 1,108 populated and 214 null
identities, 59 labels, 39 duplicate drops, six false-positive drops, three
clips, and four pending-review records.

## Task 8: Controlled real creation (controller only)

Do not execute this task during implementation or review.

After code review, the controller may materialize the real curated export:

```bash
uv run python -m jaguars.visualization.final_curation --create
```

Then require the real parser/validator test and snapshot dry-run to pass:

```bash
uv run pytest tests/unit/visualization/test_final_real_data.py -v
uv run python -m jaguars.visualization.final_dataset --dry-run
```

Only after both are green may the controller create the production snapshot:

```bash
uv run python -m jaguars.visualization.final_dataset --create-only
```

Final production verification requires 1,322 samples, 1,322 unique hashes,
1,318 ordinary annotation pairs, four pending-review samples, and all nine
saved views. Launch separately with `--launch-only`. Never overwrite an
existing curated export or snapshot without a distinct explicit
authorization.
