# APFS Curated Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the metadata-only curated export on the APFS FiftyOne state
volume while keeping all media references on the untouched Extreme SSD source
export.

**Architecture:** The curation planner remains write-free and
filesystem-independent. Creation validates the actual CameraTrapPython mount,
the approved APFS export root, and the volume's atomic exclusive-rename
capability before creating a lock or staging directory. It pins the mount and
target parent with directory descriptors and performs every write and rename
relative to those descriptors. Staging cleanup pins directory identity,
and atomically retires the complete owned directory to an inert tombstone
without deleting its metadata. It never recursively removes a foreign
pathname replacement.

**Tech Stack:** Python 3.10+, `ctypes` macOS volume-capability queries,
`pathlib`, pytest, FiftyOne's existing parser/validator.

---

### Task 1: Freeze APFS defaults

**Files:**

- Modify: `src/jaguars/visualization/final_curation.py`
- Modify: `src/jaguars/visualization/final_dataset.py`
- Test: `tests/unit/visualization/test_final_curation.py`
- Test: `tests/unit/visualization/test_final_dataset.py`

- [ ] **Step 1: Write failing default-path tests**

Assert that both CLIs use:

```python
Path(
    "/Volumes/CameraTrapPython/fiftyone/exports/"
    "JaguarCameraTrap_Final_Curated_v1"
)
```

and that the snapshot loader still approves
`labeled_segmented_jaguars_primitive/data` as its media root.

- [ ] **Step 2: Run the focused tests and verify path assertions fail**

```bash
pytest tests/unit/visualization/test_final_curation.py \
  tests/unit/visualization/test_final_dataset.py -q
```

- [ ] **Step 3: Change only the two default terminal-export constants**

Keep the primitive source and original-media-root defaults unchanged.

- [ ] **Step 4: Re-run the focused tests and verify green**

### Task 2: Validate target storage before writes

**Files:**

- Modify: `src/jaguars/visualization/final_curation.py`
- Test: `tests/unit/visualization/test_final_curation.py`
- Test: `tests/unit/visualization/test_final_real_data.py`

- [ ] **Step 1: Write failing storage-policy tests**

Cover:

```python
validate_target_storage(
    target,
    mount_root=mount_root,
    approved_root=approved_root,
    is_mount=lambda path: path == mount_root,
    capability_probe=lambda path: True,
)
```

The validator must reject an unmounted CameraTrapPython root, a target equal
to or outside the approved `fiftyone/exports` root, and a volume without
atomic no-clobber rename. A materialization test must prove rejection occurs
before parent creation, lock acquisition, or staging.

- [ ] **Step 2: Run each new test and verify the intended failure**

- [ ] **Step 3: Implement the read-only capability query and validator**

On macOS, request `ATTR_VOL_CAPABILITIES` with `getattrlist()` and require the
valid `VOL_CAP_INT_RENAME_EXCL` bit. Unsupported creation platforms fail
closed before staging while write-free planning remains portable. Validate
the actual mount, absolute strict-descendant target, same-filesystem ancestry,
and capability before creating target-parent components. Pin the mount and
target parent and use descriptor-relative operations so ancestor replacement
cannot redirect a write.

- [ ] **Step 4: Add a mounted-data test**

When both volumes are mounted, assert the CameraTrapPython APFS mount supports
exclusive rename and the Extreme SSD exFAT source mount does not. The test
must perform no writes and skip cleanly when either mount is absent.

- [ ] **Step 5: Re-run focused tests and verify green**

### Task 3: Preserve foreign staging replacements

**Files:**

- Modify: `src/jaguars/visualization/final_curation.py`
- Test: `tests/unit/visualization/test_final_curation.py`

- [ ] **Step 1: Write a failing regression test**

After a successful staging-to-target rename, recreate the former
`.building-*` pathname with a foreign sentinel and record its inode. Assert
materialization does not remove or mutate that directory.

- [ ] **Step 2: Run the test and verify it fails because the sentinel is gone**

- [ ] **Step 3: Pin and revalidate staging ownership**

Create staging relative to the pinned parent descriptor and immediately
capture its device, inode, and directory type. Atomically move the complete
owned directory to a unique inert tombstone and verify the moved identity;
never delete its metadata during cleanup. Apply the same retirement pattern
to released locks and old backups because macOS provides no inode-conditional
unlink. If a pathname names a foreign replacement at any check or move,
restore or retain that entry rather than deleting it. After promotion, verify
both the target/staging identity and the logical parent identity before
reporting success or removing the preserved backup. Once promotion returns,
mark the staging phase as published before running those checks so failure
cleanup cannot alter an owned export that was moved back to the former staging
name.

- [ ] **Step 4: Re-run the regression and curation tests**

### Task 4: Update operational documentation

**Files:**

- Modify: `docs/superpowers/specs/2026-07-25-fiftyone-final-curated-dataset-design.md`
- Modify: `docs/superpowers/plans/2026-07-25-fiftyone-final-curated-dataset.md`
- Modify: `README.md`

- [ ] Replace the obsolete exFAT curated-target path with the absolute APFS
      export path.
- [ ] State that only metadata moves; canonical media references remain below
      the primitive export on Extreme SSD.
- [ ] Document mount/root/capability validation and write-free dry-run
      behavior.
- [ ] Keep controlled production creation explicitly out of implementation.

### Task 5: Verify and commit

- [ ] Run:

```bash
pytest tests/unit/visualization -q
pytest tests/integration/visualization/test_final_snapshot.py -q
black --check src/jaguars/visualization tests/unit/visualization \
  tests/integration/visualization
ruff check src/jaguars/visualization tests/unit/visualization \
  tests/integration/visualization
mypy src/jaguars/visualization
git diff --check
```

- [ ] Confirm `git status` contains only intended source, test, and
      documentation changes.
- [ ] Commit without creating the real curated export or opening the
      production FiftyOne database.
