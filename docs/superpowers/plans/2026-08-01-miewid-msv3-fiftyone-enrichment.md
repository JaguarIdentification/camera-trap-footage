# MiewID MSv3 FiftyOne Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible MiewID MSv3 foreground-cutout embeddings to the persistent final FiftyOne dataset and publish a verified update to its existing private Hugging Face repository.

**Architecture:** A focused enrichment module owns model revision resolution, preprocessing, resumable inference, validation, provenance, and metadata-only checkpointing. The existing publishing script is extended to update an existing repository with a compare-and-swap precondition and to verify embeddings after a Hub round trip. Pure validation helpers are separated from FiftyOne and model I/O so safety behavior is unit-testable.

**Tech Stack:** Python, PyTorch MPS, Transformers custom models, FiftyOne, Hugging Face Hub, pytest.

---

### Task 1: Pure embedding contracts

**Files:**
- Create: `src/jaguars/visualization/miewid_msv3.py`
- Create: `tests/unit/visualization/test_miewid_msv3.py`

- [ ] Write failing tests for vector validation, SHA-256 checksums, the four segmentation exceptions, resumable selection, conflicting provenance, and complete-dataset publication gates.
- [ ] Run `uv run pytest tests/unit/visualization/test_miewid_msv3.py -q` and confirm failures are caused by the missing module/API.
- [ ] Implement constants and pure helpers minimally, including the 2,152-dimensional raw `float32` contract.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Checkpoint and M4 inference workflow

**Files:**
- Modify: `src/jaguars/visualization/miewid_msv3.py`
- Modify: `tests/unit/visualization/test_miewid_msv3.py`

- [ ] Write failing tests for checkpoint manifest creation, immutable revision enforcement, official 440×440 preprocessing, idempotent write planning, and explicit overwrite behavior.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Implement metadata-only FiftyOne export, pinned `AutoModel` loading with `trust_remote_code=True`, MPS-first adaptive batching, limited CPU retry, per-sample saves, statuses/checksums, and dataset-level provenance.
- [ ] Add a CLI with `preflight`, `checkpoint`, `infer`, and `validate` actions and an `--overwrite-embeddings` guard.
- [ ] Re-run the focused tests and relevant existing visualization tests.

### Task 3: Existing-repository publication

**Files:**
- Modify: `scripts/publish_final_dataset_to_hub.py`
- Create: `tests/unit/visualization/test_publish_final_dataset_to_hub.py`

- [ ] Write failing tests for expected remote-head enforcement, enriched dataset validation, and exact round-trip checksum comparison.
- [ ] Run the focused tests and confirm the failures.
- [ ] Change upload semantics from repository creation to an update of the existing private repository, update its description, and add the expected-head precondition.
- [ ] Extend round-trip verification to require 1,318 valid embeddings, four explicit exceptions, matching checksums, unchanged media hashes, and privacy.
- [ ] Re-run focused tests and the full unit suite.

### Task 4: Local mutation and publication

**Files:**
- Runtime state only: `/Volumes/CameraTrapPython/fiftyone`
- Remote state only: `andandandand/jaguar-camera-trap-final-curated-v1`

- [ ] Run preflight and record the immutable model revision and current Hub dataset head.
- [ ] Create and inspect the metadata-only checkpoint before schema mutation.
- [ ] Run MPS/CPU consistency sampling, then resumable MPS inference at adaptive batch size 16.
- [ ] Run complete local validation and record vector/status counts and checksums.
- [ ] Upload only if the remote head still matches preflight.
- [ ] Round-trip load the new revision, validate every contract, remove the temporary verification dataset, and report the resulting Hub commit.
