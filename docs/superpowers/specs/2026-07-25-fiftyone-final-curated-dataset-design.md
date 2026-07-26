# FiftyOne Final Curated Dataset Design

## Goal

Create a persistent, frozen FiftyOne snapshot named
`JaguarCameraTrap_Final_Curated_v1` containing exactly 1,322 unique retained
artifacts from the terminal camera-trap export. The original 1,367-sample
export at
`data/intermediate/v1/fo_jaguars/labeled_segmented_jaguars_primitive` remains
unchanged. A deterministic curation step materializes the independently
loadable terminal export
`/Volumes/CameraTrapPython/fiftyone/exports/JaguarCameraTrap_Final_Curated_v1`,
which is the only default source for the final snapshot.

## Curation policy

The curation policy is versioned and deterministic:

- Group samples by exact media SHA-256 and retain one representative from each
  of the 39 two-sample duplicate groups.
- Require populated identities within a hash group to agree.
- Require semantic annotations to agree after ignoring generated detection
  IDs.
- Prefer an annotation-valid record, then a populated identity, then a unique
  compatible exact-lineage match, then the lexicographically smallest
  export-relative filepath.
- Record every representative and dropped member in
  `curation_report.json`.
- Exclude the six confirmed false positives:
  `000005-11`, `000010-8`, `000010-9`, `000015-9`, `000030-25`, and
  `000030-6`.
- Apply the audited in-bounds body-box values to `000004-120`, `000005-61`,
  and `000025-40`. Their segmentation boxes and masks remain unchanged.
- Keep `000001-143`, `000002-144`, `000010-18`, and `000005-126` without
  re-segmentation. Remove their invalid `bboxes_body` and
  `segmentations_body`; set `review_required=True`, a nonempty
  `review_reason`, `review_status="pending"`, and the FiftyOne tag
  `needs_annotation_review`.

The result must contain exactly 1,322 unique paths and hashes, 1,108 populated
identities, 214 null identities, and 59 distinct non-null labels. The four
pending-review samples are media-only. The remaining 1,318 samples require
valid nonempty body boxes and segmentations.

## Curation storage and atomicity

The curated export is metadata-only. Its source media remains on the Extreme
SSD exFAT volume, while its three JSON files are published on the
CameraTrapPython APFS state volume at
`/Volumes/CameraTrapPython/fiftyone/exports/JaguarCameraTrap_Final_Curated_v1`.
Each curated `samples.json` filepath is the canonical absolute path of
retained media below the original terminal export's `data` directory.
Creation never copies, symlinks, hardlinks, or re-encodes media and never
modifies the original export. The target contains only `samples.json`,
`metadata.json`, and `curation_report.json`.

The terminal parser remains export-local by default. The curated snapshot
workflow explicitly supplies the original `data` directory as its approved
media root, requires every resolved path to be a strict descendant, and
rejects traversal or symlink escapes. The snapshot CLI is the authoritative
loader; generic FiftyOne-dataset importers that require a target-local `data`
directory are not supported by copying media into place.

Dry-run planning performs no writes. Creation builds a sibling temporary
directory, writes deterministic `samples.json`, minimal `metadata.json`, and
`curation_report.json`, validates the complete result, and only then renames
it to the target. An existing target is refused unless `--create --overwrite`
is explicitly supplied and confirmed; noninteractive overwrite additionally
requires `--yes`, and piped text is never accepted as confirmation. A lexical
target that is itself a symlink or has an unsafe ancestor/descendant
relationship with the source is rejected before staging or removal.
Before creating the target parent, lock, or staging directory, creation
requires `/Volumes/CameraTrapPython` to be an actual mount, requires the
target to resolve as a strict descendant of
`/Volumes/CameraTrapPython/fiftyone/exports`, and verifies that the target
volume advertises atomic exclusive directory rename. The deepest existing
target ancestor must remain on the CameraTrapPython filesystem; nested mounts
are rejected. Custom targets must pass the same root, filesystem, and
capability checks. The source exFAT volume does not advertise this capability
and is never used for curated metadata publication.
Creation pins the mount and target-parent identity with directory descriptors.
Parent creation, locking, staging creation, JSON writes, and no-clobber renames
are descriptor-relative, and the logical parent identity is rechecked around
the staged build. An ancestor replacement therefore cannot redirect writes
outside the approved filesystem.
Materialization holds an exclusively created sibling lock through staging and
publication. The CLI pins an existing target before asking for confirmation
and passes that exact expected state into materialization. The pin records
whether the target was absent or, for a confirmed replacement, its canonical
path, directory device/inode, report digest, and source identity. The exact
state is rechecked immediately before rename. After the target is renamed to
backup, the backup identity is checked again before promotion or deletion. An
unexpected backup is restored when the target remains absent or retained
under an explicit recovery name when restoration would collide. A
concurrently created or changed target is preserved, only the owned staging
directory is retired from its operational name, and lock cleanup never
unlinks a foreign replacement. Staging cleanup pins device/inode/type
immediately after creation and preserves any later entry that occupies the
former staging pathname. Because macOS has no inode-conditional unlink, owned
staging/backup directories—including their metadata—and released lock files
are atomically moved to unique inert hidden tombstones and retained. No
cleanup path unlinks their contents. The move is verified against the pinned
identity; a last-moment foreign replacement is restored or moved to a
recovery name and is never deleted.
After each publication rename, the promoted directory must match the pinned
staging identity and the logical target parent must still match its pinned
identity. Otherwise creation fails with both the unexpected entry and owned
export preserved at their recoverable locations. Once the promotion rename
returns successfully, cleanup never alters the pinned staging descriptor,
even if either post-publication check fails.
Replacement keeps the old target recoverable until the new directory is
ready.

The sidecar records the source `samples.json` SHA-256, policy version, counts,
kept and dropped paths, reasons, exact-hash groups, representatives, audited
clips, stripped review annotations, and every retained media hash.

## Final-artifact boundary

Each ordinary snapshot sample uses the curated segmented image as primary
media and retains final `bboxes_body` and `segmentations_body`. Pending-review
samples retain their media, identity, lineage, and review fields but have no
body annotations. A resolved identity is stored both as plain `jaguar_id` and
as a FiftyOne `ground_truth` `Classification`; both are null when unresolved.

The snapshot excludes raw videos, sampled frames, screenshots, earlier crops
and segmentations, exact-content duplicates, confirmed false positives,
embeddings, deduplication scores, cache paths, temporary identifiers, random
fields, and pipeline timestamps.

## Enrichment and lineage

Curated samples are enriched from upstream exports and the two split
manifests:

- `data/intermediate/v1/labels_with_splits.csv`
- `data/intermediate/v1/pptx_extracted_labels_with_splits.csv`

Lineage matching uses only these precedence-ordered exact methods:

1. Preserved FiftyOne or source sample ID
2. Normalized source filepath
3. Export-relative filepath
4. Exact filename when globally unique

No fuzzy filenames, numeric-prefix inference, image similarity, or best guess
is allowed. Every sample receives `lineage_status` (`matched`, `ambiguous`, or
`missing`) and, when matched, `lineage_match_method`.

The enrichment boundary contains identity and ground truth; closed- and
open-set splits; sighting, site, location, camera, and capture-time fields;
source provenance; and SHA-256, byte size, width, and height. Missing or
ambiguous lineage is reported but does not remove a retained sample.

## FiftyOne storage and immutability

FiftyOne references the canonical original terminal media paths in place.
Generated state stays on
external storage:

- Database: `/Volumes/CameraTrapPython/fiftyone/var/lib/mongo`
- Curated metadata export:
  `/Volumes/CameraTrapPython/fiftyone/exports/JaguarCameraTrap_Final_Curated_v1`
- Reports:
  `/Volumes/CameraTrapPython/fiftyone/JaguarCameraTrap_Final_Curated_v1`
- Dataset/download defaults: `/Volumes/CameraTrapPython/fiftyone/datasets`

Snapshot creation requires `/Volumes/Extreme SSD` and
`/Volumes/CameraTrapPython` to be mounted filesystems. Every generated-state
path must resolve below `/Volumes/CameraTrapPython/fiftyone`. FiftyOne is
configured and revalidated before import; inherited URI/private-port
configuration or a redirected database is rejected.

Ordinary creation refuses an existing final dataset. `--overwrite` prints
existing and proposed counts and requires confirmation; `--yes` is required
for noninteractive replacement. Confirmation pins the original persisted
dataset ID. Snapshot staging, ownership tokens, promotion, recovery, and
backup cleanup follow the existing compare-and-swap transaction; media is
never deleted.

## Commands

Audit or materialize the curated terminal export:

```bash
uv run python -m jaguars.visualization.final_curation --dry-run
uv run python -m jaguars.visualization.final_curation --create
```

Audit, create, or launch the frozen snapshot:

```bash
uv run python -m jaguars.visualization.final_dataset --dry-run
uv run python -m jaguars.visualization.final_dataset --create-only
uv run python -m jaguars.visualization.final_dataset
```

The snapshot CLI also supports `--launch-only`, guarded `--overwrite`,
noninteractive `--yes`, and App `--address`/`--port`.

## Saved views

The snapshot has exactly nine saved views:

- `All final samples`
- `Lineage issues`
- Closed-set `train`, `val`, and `test`
- Open-set `train`, `val`, and `test`
- `Annotation review`

## Integrity policy

Creation fails for missing or unreadable media, duplicate canonical paths or
hashes, count or frozen identity drift, invalid enrichment, identity/
ground-truth disagreement, or malformed ordinary annotations. Missing
annotations are allowed only when `review_required=True`, the reason is
nonempty, the status is `pending`, and the review tag is present. Review
samples must omit both annotation fields.

## Testing and acceptance

Unit tests cover representative selection and conflicts, semantic annotation
normalization, exact counts, clips, exclusions, metadata-only sidecars,
approved-root media references and escapes, source immutability, atomic
failure cleanup, symlink-safe guarded overwrite, TTY confirmation, parser and
validator compatibility, CLI defaults, APFS target-root/capability guards,
owned staging cleanup, and review contracts.

An isolated FiftyOne integration suite verifies schema, media-only review
samples, all nine views, immutable construction, ownership-safe cleanup, and
replacement recovery without touching production.

The required read-only real-data curation audit verifies:

- 1,367 source records
- 39 exact-content duplicate drops
- 6 confirmed false-positive drops
- 1,322 retained unique paths and hashes
- 1,108 populated and 214 null identities
- 59 non-null labels
- 3 audited bbox clips
- 4 pending-review media-only samples
- CameraTrapPython supports atomic exclusive rename while the source Extreme
  SSD volume does not

After the curated export is materialized, its parser/validator audit must be
green before production snapshot creation. Production verification then
confirms 1,322 samples, 1,318 populated annotation pairs, four pending-review
records, identity agreement, required fields, unique hashes, all nine saved
views, and successful App launch.
