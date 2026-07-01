# Upload File-Descriptor Leak Fix — Implementation Plan

> **SUPERSEDED (2026-07-01).** This plan's premise (unclosed `UploadFile` leaking FDs) was
> **refuted** during diagnosis — the framework closes upload spools automatically, and the leak
> test in Task 1 cannot reproduce (Task 1 was never committed). The real leak is ROCm/MIOpen
> kernel compiles. Do **not** execute this plan. See
> `docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md`; its plan carries over
> the only surviving piece (Task 2, the `nofile` ulimit).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the file-descriptor leak by closing every uploaded `UploadFile`, proven by a regression test, and raise the `nofile` ceiling as defense-in-depth.

**Architecture:** A single FastAPI yield-dependency (`uploaded_file`) owns the upload lifecycle and closes the file in a `finally` that runs on every exit path. All five upload endpoints switch from `File(...)` to `Depends(uploaded_file("..."))`. A Linux `/proc`-based test counts `(deleted)` FDs across repeated >1 MB uploads to catch any endpoint that leaks. All six docker-compose files get a raised `nofile` ulimit.

**Tech Stack:** Python 3.12, FastAPI (>=0.128), Starlette `UploadFile`/`SpooledTemporaryFile`, pytest + `fastapi.testclient.TestClient`, docker-compose.

## Global Constraints

- Tests run inside a virtualenv (see Prerequisites); never run `pytest` against system Python.
- Do not change dependency pins: `fastapi>=0.128.0,<1.0.0`, `python-multipart>=0.0.21,<1.0.0`, `aiofiles>=24.1.0,<25.0.0`.
- Single-worker runtime (`workers=1`) — do not introduce anything requiring multiple workers.
- Commit style is Conventional Commits (`fix:`, `test:`, `chore(docker):`, `docs:`) as in the existing history.
- Preserve each endpoint's OpenAPI upload-parameter description verbatim.
- Work happens on branch `fix/upload-fd-leak` (already created; the design spec is committed there).

## Prerequisites (one-time, this session)

All `pip`/`pytest` commands below assume this venv is active.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## File Structure

- **Modify** `app/dependencies.py` — add the `uploaded_file` yield-dependency factory (+ imports). Responsibility: own the upload lifecycle / guarantee close.
- **Modify** `app/main.py` — migrate the 5 upload endpoints to `Depends(uploaded_file(...))`; drop the now-unused `File` import; import `uploaded_file`.
- **Modify** `tests/test_endpoints.py` — add the FD-leak regression test (reuses the existing `client`/`mock_model_manager` fixtures; placed here rather than a new module to avoid moving fixtures to `conftest.py`).
- **Modify** `docker/docker-compose-{amd,cpu,nvidia}.yml` (service `yolo-api`) and `docker/deploy/docker-compose-{amd,cpu,nvidia}.yml` (service `vision-api`) — add `ulimits.nofile`.

---

## Task 1: Close uploaded files (fix + regression test)

**Files:**
- Modify: `tests/test_endpoints.py` (imports near top; new test at end of file)
- Modify: `app/dependencies.py` (imports at top; new factory at end)
- Modify: `app/main.py:14` (imports), `app/main.py:22` (imports), and the 5 endpoint signatures at lines `428`, `496`, `675`, `799`, `907`

**Interfaces:**
- Produces: `uploaded_file(description: str = "Uploaded file") -> Callable` in `app/dependencies.py` — a factory returning an async generator dependency that yields `starlette.datastructures.UploadFile` and awaits `file.close()` in `finally`. Used as `file: UploadFile = Depends(uploaded_file("<desc>"))`.
- Consumes: existing `client` and `mock_model_manager` pytest fixtures in `tests/test_endpoints.py`; `main.extract_frames_from_video` (patched in the test).

- [ ] **Step 1: Add imports for the regression test**

In `tests/test_endpoints.py`, the current top is:

```python
import io
from contextlib import asynccontextmanager
```

Change it to add `gc`, `os`, `sys`:

```python
import gc
import io
import os
import sys
from contextlib import asynccontextmanager
```

(`pytest`, `patch`, `AsyncMock` are already imported in this file.)

- [ ] **Step 2: Write the failing regression test**

Append to the end of `tests/test_endpoints.py`:

```python
# --- Upload file-descriptor leak regression ---

def _deleted_fd_count() -> int:
    """Count file descriptors pointing at deleted files — the leak signature."""
    fd_dir = f"/proc/{os.getpid()}/fd"
    count = 0
    for name in os.listdir(fd_dir):
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue
        if target.endswith("(deleted)"):
            count += 1
    return count


# 2 MB > Starlette's 1 MB SpooledTemporaryFile threshold, so every upload rolls
# over to an on-disk (unlinked) temp file whose FD leaks unless the endpoint closes it.
_BIG_PAYLOAD = b"\x00" * (2 * 1024 * 1024)

_UPLOAD_ENDPOINTS = [
    ("/detect", "big.bmp", "image/bmp"),
    ("/detect/visualize", "big.bmp", "image/bmp"),
    ("/detect/video", "big.mp4", "video/mp4"),
    ("/extract/frames", "big.mp4", "video/mp4"),
    ("/detect/video/visualize", "big.mp4", "video/mp4"),
]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="FD counting via /proc is Linux-only",
)
@pytest.mark.parametrize("path, filename, content_type", _UPLOAD_ENDPOINTS)
def test_upload_endpoint_does_not_leak_fds(client, path, filename, content_type):
    """Every upload endpoint must close its UploadFile so no FD leaks.

    Regression: the SpooledTemporaryFile rollover leaked one FD per >1 MB upload,
    exhausting RLIMIT_NOFILE and jamming the container.
    """

    def _post():
        return client.post(
            path,
            files={"file": (filename, io.BytesIO(_BIG_PAYLOAD), content_type)},
        )

    # Video endpoints run ffmpeg synchronously; stub it out so the test is fast and
    # only the UploadFile lifecycle is exercised (status codes are irrelevant here).
    with patch("main.extract_frames_from_video", new=AsyncMock(return_value=[])):
        # Warm up: first requests open steady-state FDs (lazy imports, caches).
        _post()
        _post()

        responses = []
        gc.collect()
        gc.disable()  # keep any cycle-retained temp files alive so a real leak is visible
        try:
            baseline = _deleted_fd_count()
            for _ in range(20):
                responses.append(_post())
            leaked = _deleted_fd_count() - baseline
        finally:
            gc.enable()

    assert leaked <= 1, f"{path} leaked {leaked} deleted-file FDs across 20 uploads"
```

- [ ] **Step 3: Run the test and verify it FAILS (red)**

Run: `python -m pytest tests/test_endpoints.py::test_upload_endpoint_does_not_leak_fds -v`
Expected: FAIL — each parametrization asserts `leaked <= 1` but reports ~20 leaked FDs (the current code never closes the upload). If it unexpectedly passes, STOP and investigate before implementing (the leak must reproduce for this to be a valid regression test).

- [ ] **Step 4: Add the `uploaded_file` dependency**

In `app/dependencies.py`, the current imports are:

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request
```

Replace them with:

```python
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import File, Request, UploadFile
```

Then append to the end of `app/dependencies.py`:

```python
def uploaded_file(description: str = "Uploaded file"):
    """Factory for a FastAPI dependency that yields an uploaded file and always closes it.

    FastAPI runs the finally block after the response is sent on every exit path
    (success, HTTPException, unhandled error, StreamingResponse completion), so the
    SpooledTemporaryFile backing an upload larger than 1 MB never leaks a file descriptor.
    Each call returns a distinct inner function, so FastAPI's per-request dependency
    cache never collides across endpoints.
    """

    async def _dependency(
        file: UploadFile = File(..., description=description),
    ) -> AsyncIterator[UploadFile]:
        try:
            yield file
        finally:
            await file.close()

    return _dependency
```

- [ ] **Step 5: Migrate `app/main.py` imports**

Line 14 — remove `File`:

```python
from fastapi import FastAPI, UploadFile, Query, Request, Depends, HTTPException
```

Line 22 — add `uploaded_file`:

```python
from dependencies import get_model_manager, get_job_manager, uploaded_file
```

- [ ] **Step 6: Migrate the two image endpoints (`/detect` line 428 and `/detect/visualize` line 799)**

These two lines are identical, so replace all occurrences. Old (appears twice, 8-space indent):

```python
        file: UploadFile = File(..., description="Image for analysis"),
```

New:

```python
        file: UploadFile = Depends(uploaded_file("Image for analysis")),
```

- [ ] **Step 7: Migrate the three video endpoints (lines 496, 675, 907)**

`/detect/video` (line 496, 8-space indent) — old → new:

```python
        file: UploadFile = File(..., description="Video file for analysis"),
```
```python
        file: UploadFile = Depends(uploaded_file("Video file for analysis")),
```

`/extract/frames` (line 675, 8-space indent) — old → new:

```python
        file: UploadFile = File(..., description="Video file for frame extraction"),
```
```python
        file: UploadFile = Depends(uploaded_file("Video file for frame extraction")),
```

`/detect/video/visualize` (line 907, 4-space indent) — old → new:

```python
    file: UploadFile = File(..., description="Video file for annotation"),
```
```python
    file: UploadFile = Depends(uploaded_file("Video file for annotation")),
```

- [ ] **Step 8: Run the regression test and verify it PASSES (green)**

Run: `python -m pytest tests/test_endpoints.py::test_upload_endpoint_does_not_leak_fds -v`
Expected: PASS — all 5 parametrizations report `leaked <= 1`.

- [ ] **Step 9: Run the full suite and verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: PASS — the `File(...)` → `Depends(uploaded_file(...))` swap must not break existing endpoint tests (e.g. `TestAnnotateVideo`), which still post files and expect their original status codes.

- [ ] **Step 10: Commit**

```bash
git add app/dependencies.py app/main.py tests/test_endpoints.py
git commit -m "fix: close uploaded files to stop file-descriptor leak"
```

---

## Task 2: Raise `nofile` ulimit in all compose files

**Files:**
- Modify: `docker/docker-compose-amd.yml`, `docker/docker-compose-cpu.yml`, `docker/docker-compose-nvidia.yml`
- Modify: `docker/deploy/docker-compose-amd.yml`, `docker/deploy/docker-compose-cpu.yml`, `docker/deploy/docker-compose-nvidia.yml`

**Interfaces:**
- Consumes/Produces: none (deployment config only). Independent of Task 1.

- [ ] **Step 1: Add `ulimits` to each of the 6 compose files**

Every file has exactly one line `    restart: unless-stopped` (4-space indent, service level). In **each** of the 6 files, replace that line:

```yaml
    restart: unless-stopped
```

with:

```yaml
    restart: unless-stopped
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
```

Apply this identical edit to all six files listed above.

- [ ] **Step 2: Validate every compose file still parses and has the ulimit**

Run:

```bash
python - <<'PY'
import yaml, glob, sys
ok = True
for f in sorted(glob.glob("docker/docker-compose-*.yml") + glob.glob("docker/deploy/docker-compose-*.yml")):
    d = yaml.safe_load(open(f))
    svc = next(iter(d["services"].values()))
    n = svc.get("ulimits", {}).get("nofile", {})
    good = n.get("soft") == 65536 and n.get("hard") == 65536
    ok &= good
    print(("OK  " if good else "FAIL"), f, n)
sys.exit(0 if ok else 1)
PY
```

Expected: all 6 lines print `OK ... {'soft': 65536, 'hard': 65536}` and exit code 0.

- [ ] **Step 3: Commit**

```bash
git add docker/docker-compose-amd.yml docker/docker-compose-cpu.yml docker/docker-compose-nvidia.yml \
        docker/deploy/docker-compose-amd.yml docker/deploy/docker-compose-cpu.yml docker/deploy/docker-compose-nvidia.yml
git commit -m "chore(docker): raise nofile ulimit to 65536 in all compose files"
```

---

## Self-Review

**1. Spec coverage:**
- Goal 1 (close every UploadFile on all exit paths) → Task 1 Steps 4–7 (dependency + 5 migrations). ✓
- Goal 2 (regression test reproducing the leak) → Task 1 Steps 2–3 (red) and 8 (green). ✓
- Goal 3 (raise nofile in all compose files) → Task 2. ✓
- Spec "Files touched" list → all covered (dependencies.py, main.py, test file, 6 compose files). Deviation: the test lives in `tests/test_endpoints.py` (reuse fixtures) instead of a new `tests/test_upload_cleanup.py`; noted in File Structure. ✓
- Spec edge cases (StreamingResponse, background job, idempotent close) → handled by the single `finally: await file.close()`; no extra tasks needed. ✓

**2. Placeholder scan:** No TBD/TODO; every code and command step contains complete content. ✓

**3. Type consistency:** `uploaded_file(description: str)` is defined in Task 1 Step 4 and consumed with the exact same call form `Depends(uploaded_file("..."))` in Steps 6–7. Import added in Step 5 matches the definition module (`dependencies`). Test helper `_deleted_fd_count()` defined and used within the same file. ✓
