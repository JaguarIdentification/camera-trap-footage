# Remove Obsolete External Caches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely remove the three obsolete partial cache directories from the external disk without touching the downloaded dataset or the active APFS-backed Python and DVC storage.

**Architecture:** Treat deletion as a guarded filesystem maintenance operation. Resolve and compare every active repository target before deletion, remove only an explicit allowlist of stale paths, then verify the active environment, DVC cache, and 13,001-file dataset remain intact.

**Tech Stack:** macOS shell utilities (`readlink`, `test`, `du`, `find`, `rm`), uv, DVC

## Global Constraints

- Preserve `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data` and its 13,001 dataset files.
- Preserve `/Volumes/CameraTrapPython/venv` and `/Volumes/CameraTrapPython/dvc-cache`.
- Delete only the exact stale paths listed in Task 2; do not use globs, unresolved variables, or recursive deletion against a parent directory.
- The external volume and `/Volumes/CameraTrapPython` must both be mounted before deletion begins.
- Expected recoverable capacity is approximately 5.7 GB as reported by `du -sh` before cleanup.

---

### Task 1: Preflight and Prove the Targets Are Obsolete

**Files:**
- Inspect: `.venv`
- Inspect: `data`
- Inspect: `.dvc/config.local`
- Inspect: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/venv`
- Inspect: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/uv-cache`
- Inspect: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/dvc-cache`

**Interfaces:**
- Consumes: Repository symlinks and local DVC configuration.
- Produces: Verified active and stale path sets for the deletion gate.

- [ ] **Step 1: Confirm both required volumes are mounted**

Run:

```bash
test -d '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data'
test -d '/Volumes/CameraTrapPython/venv'
test -d '/Volumes/CameraTrapPython/dvc-cache'
```

Expected: all three commands exit with status 0 and print nothing. Stop immediately if any command fails.

- [ ] **Step 2: Resolve the active repository targets**

Run from the repository root:

```bash
readlink data
readlink .venv
git config --file .dvc/config.local cache.dir
```

Expected output, in order:

```text
/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data
/Volumes/CameraTrapPython/venv
/Volumes/CameraTrapPython/dvc-cache
```

Stop if any value differs. In particular, neither `.venv` nor the DVC cache may resolve into one of the stale host-filesystem directories.

- [ ] **Step 3: Confirm the exact stale allowlist and its current size**

Run:

```bash
du -sh '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/venv' '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/uv-cache' '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/dvc-cache'
```

Expected: all three paths exist. The sizes observed while writing this plan were 2.7 GB, 1.9 GB, and 1.1 GB respectively. Size changes do not authorize widening the deletion scope.

### Task 2: Remove Only the Explicitly Approved Stale Paths

**Files:**
- Delete: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/venv`
- Delete: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/uv-cache`
- Delete: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/dvc-cache`
- Delete: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/._venv`
- Delete: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/._uv-cache`
- Delete: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/._dvc-cache`

**Interfaces:**
- Consumes: The verified allowlist from Task 1.
- Produces: External host directory containing only the dataset and its filesystem metadata.

- [ ] **Step 1: Obtain explicit approval for the destructive command**

Present the six paths above to the user and request approval for their permanent removal. Do not execute Task 2 without approval.

- [ ] **Step 2: Delete each stale directory using literal absolute paths**

Run exactly:

```bash
rm -rf '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/venv'
rm -rf '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/uv-cache'
rm -rf '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/dvc-cache'
```

Expected: each command exits with status 0. These removals are permanent rather than recoverable through Trash.

- [ ] **Step 3: Delete the matching AppleDouble companion files**

Run exactly:

```bash
rm -f '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/._venv'
rm -f '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/._uv-cache'
rm -f '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/._dvc-cache'
```

Expected: each command exits with status 0. Do not delete `._data` because it accompanies the retained dataset directory.

### Task 3: Verify Cleanup and Active Storage Integrity

**Files:**
- Verify: `.venv`
- Verify: `data`
- Verify: `.dvc/config.local`
- Verify absence: the six paths deleted in Task 2

**Interfaces:**
- Consumes: Cleaned external directory and active APFS-backed storage.
- Produces: Evidence that capacity was recovered without damaging the configured repository.

- [ ] **Step 1: Verify every stale path is absent**

Run:

```bash
test ! -e '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/venv'
test ! -e '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/uv-cache'
test ! -e '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/dvc-cache'
test ! -e '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/._venv'
test ! -e '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/._uv-cache'
test ! -e '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/._dvc-cache'
```

Expected: all six commands exit with status 0.

- [ ] **Step 2: Confirm the retained external directory entries**

Run:

```bash
find '/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage' -maxdepth 1 -mindepth 1 -print
```

Expected: `data` remains. `._data` may also remain because the host filesystem creates AppleDouble metadata; none of the six deleted stale paths may appear.

- [ ] **Step 3: Verify the active Python environment**

Run from the repository root:

```bash
.venv/bin/python --version
uv run python -c 'import jaguars; print("jaguars import OK")'
```

Expected: Python reports `3.10.19`, and the import command prints `jaguars import OK`.

- [ ] **Step 4: Verify DVC and dataset integrity**

Run from the repository root:

```bash
.venv/bin/dvc status
find -L data -type f ! -name '._*' | wc -l
```

Expected:

```text
Data and pipelines are up to date.
13001
```

If DVC reports missing or changed data, stop and report the discrepancy; do not run `dvc pull` as part of this cleanup.

- [ ] **Step 5: Report the completed deletion and recovery characteristics**

Report that the three obsolete directories and their AppleDouble companions were permanently removed, that approximately 5.7 GB was targeted for recovery, and include the exact DVC status and dataset file count from Step 4. Do not claim exact free-space recovery unless it was separately measured before and after at the filesystem level.
