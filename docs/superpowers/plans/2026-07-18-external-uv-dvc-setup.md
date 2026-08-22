# External uv and DVC Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure the current checkout with a uv environment, DVC cache, and full materialized dataset stored under `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage`.

**Architecture:** An external directory holds the materialized dataset. An APFS sparse bundle stored on the external SSD holds the Python environment, uv cache, and DVC cache because the host external filesystem is incompatible with uv's lock/package-copy semantics and DVC's concurrent temporary cache files. Repository-local `data` and `.venv` symlinks preserve the paths expected by the code and tools, while `.dvc/config.local` redirects DVC's cache without modifying the shared remote configuration.

**Tech Stack:** uv, Python 3.10+, DVC 3 with Google Drive support, Git, POSIX symlinks

---

### Task 1: Prepare external storage and repository links

**Files:**
- Create: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data/`
- Create: `/Volumes/CameraTrapPython/dvc-cache/`
- Create: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage-python.sparsebundle`
- Create: `/Volumes/CameraTrapPython/venv/`
- Create: `.venv` symlink
- Create: `data` symlink

- [ ] **Step 1: Create the external directories**

Run:

```bash
mkdir -p "/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data"
```

Expected: exit code 0.

- [ ] **Step 2: Create and verify the data symlink**

Run:

```bash
ln -s "/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data" data
test "$(readlink data)" = "/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data"
```

Expected: exit code 0 and `data` resolves to the approved external path.

### Task 2: Create and populate the uv environment

**Files:**
- Create: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/venv/`
- Create: `.venv` symlink

- [ ] **Step 1: Create the external environment**

Run:

```bash
hdiutil create -size 100g -type SPARSEBUNDLE -fs APFS -volname CameraTrapPython "/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage-python.sparsebundle"
hdiutil attach "/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage-python.sparsebundle"
uv venv "/Volumes/CameraTrapPython/venv" --python /opt/homebrew/bin/python3.10
ln -s "/Volumes/CameraTrapPython/venv" .venv
```

Expected: uv creates a Python 3.10 environment and `.venv` resolves to it.

- [ ] **Step 2: Install the declared project dependencies**

Run:

```bash
uv pip install --python .venv/bin/python -r <(sed '/^decord>=/d' requirements.txt) eva-decord==0.6.1
```

Expected: exit code 0, including editable installation of `jaguars` and installation of DVC Google Drive support. On macOS ARM64, `eva-decord==0.6.1` supplies the compatible `decord` import because upstream `decord==0.6.0` has no wheel for this platform.

- [ ] **Step 3: Install the README's additional Transformers dependency**

Run:

```bash
uv pip install --python .venv/bin/python "git+https://github.com/huggingface/transformers.git#egg=transformers"
```

Expected: exit code 0.

- [ ] **Step 4: Verify core imports**

Run:

```bash
.venv/bin/python -c "import dvc, jaguars; print(dvc.__version__)"
```

Expected: exit code 0 and a DVC version is printed.

### Task 3: Configure DVC and pull the dataset

**Files:**
- Create: `.dvc/config.local`
- Populate: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/dvc-cache/`
- Populate: `/Volumes/Extreme SSD/CuratedCameraTrapData/camera-trap-footage/data/`

- [ ] **Step 1: Configure the external cache locally**

Run:

```bash
.venv/bin/dvc config --local cache.dir "/Volumes/CameraTrapPython/dvc-cache"
.venv/bin/dvc cache dir
```

Expected: DVC reports the approved external cache path.

- [ ] **Step 2: Pull and materialize the complete tracked dataset**

Run:

```bash
.venv/bin/dvc pull --jobs 1 data.dvc
```

Expected: exit code 0. One worker avoids the Google Drive backend's temporary-file race under its default high concurrency. If Google Drive requires interactive authentication, complete that authorization and resume this command.

### Task 4: Verify the completed setup

**Files:**
- Verify: `.venv`
- Verify: `data`
- Verify: `.dvc/config.local`
- Verify: `data.dvc`

- [ ] **Step 1: Verify paths and DVC state**

Run:

```bash
readlink .venv
readlink data
.venv/bin/dvc cache dir
.venv/bin/dvc status
```

Expected: both symlinks and the cache point under the approved external root; DVC reports `Data and pipelines are up to date.`

- [ ] **Step 2: Verify dataset size and file count**

Run:

```bash
du -shL data
find -L data -type f | wc -l
```

Expected: approximately 34 GiB of logical data and 13,001 files.

- [ ] **Step 3: Verify repository state**

Run:

```bash
git status --short --branch
```

Expected: no large artifact is shown as untracked; only intentional documentation changes, if any, are present.
