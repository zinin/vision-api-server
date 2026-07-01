# MIOpen FD-Leak Containment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ROCm/MIOpen kernel-compile FD leak harmless: raise the `nofile` ceiling in all compose files, persist the MIOpen caches on volumes so kernels compile once per volume lifetime, and expose FD usage in `/health` with an early-warning log.

**Architecture:** No app-logic changes beyond `/health`. Containment lives in deployment config (ulimits + named volumes in docker-compose) plus a small observability addition in `app/main.py` (`_fd_stats()` helper, two new `/health` fields, threshold WARNING). The refuted `UploadFile` fix is dropped; its uncommitted test stub is reverted.

**Tech Stack:** Python 3.12, FastAPI, pytest + `fastapi.testclient.TestClient`, docker-compose YAML, PyYAML (already in `.venv` via ultralytics).

**Spec:** `docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md`. The SUPERSEDED banners on the old design/plan are already committed (`5c4df1f`) — not part of this plan.

## Global Constraints

- Tests run inside the repo virtualenv only (`.venv`); never run `pytest` against system Python.
- Do not change dependency pins in `requirements*.txt`.
- Single-worker runtime (`workers=1`) — introduce nothing that requires multiple workers.
- Conventional Commits (`feat:`, `chore(docker):`, `docs:`) as in existing history.
- Work on branch `fix/upload-fd-leak`.
- Do NOT touch the live prod container `deploy-vision-api-1` (no traffic, no restarts). Deploy/rebuild is user-side after merge.
- `MIOPEN_FIND_MODE` stays commented out everywhere — it is a documented emergency lever, never enabled by default.

## Prerequisites (one-time, this session)

```bash
source .venv/bin/activate   # .venv already exists in the repo root with all deps installed
```

If `.venv` is missing, create it first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## File Structure

- **Modify** `tests/test_endpoints.py` — first revert the uncommitted FD-leak stub (tests a refuted bug), then add `/health` FD-stats tests (imports at top; new test class at end).
- **Modify** `app/main.py` — `import resource`; `_fd_stats()` helper above the `/health` endpoint; two new fields + WARNING in `health()`.
- **Modify** `docker/docker-compose-{amd,cpu,nvidia}.yml` (service `yolo-api`) and `docker/deploy/docker-compose-{amd,cpu,nvidia}.yml` (service `vision-api`) — `ulimits.nofile` in all six; the two AMD files also get `miopen-cache`/`miopen-config` volumes and a commented `MIOPEN_FIND_MODE` line.
- **Modify** `docker/deploy/.env.example`, `CLAUDE.md`, `.claude/rules/api.md` — documentation.

---

## Task 1: `/health` FD observability (+ revert the refuted-test stub)

**Files:**
- Modify: `tests/test_endpoints.py` (revert uncommitted changes, then edit imports at top; append test class at end)
- Modify: `app/main.py` (imports at top; helper + endpoint changes around line 405)

**Interfaces:**
- Produces: `_fd_stats() -> tuple[int | None, int]` in `app/main.py` — returns `(open_fds, soft_limit)`; `open_fds` is `None` where `/proc/self/fd` is unavailable. `/health` response gains `"open_fds": int | None` and `"fd_soft_limit": int`. Tests patch it as `patch("main._fd_stats", return_value=(900, 1000))`.
- Consumes: existing `client` fixture in `tests/test_endpoints.py`; module logger `logger = logging.getLogger(__name__)` in `app/main.py` (logger name is `main`).

- [ ] **Step 1: Revert the uncommitted FD-leak test stub**

The working tree has ~71 uncommitted lines in `tests/test_endpoints.py` from the superseded plan (an FD-leak regression test that cannot reproduce — the framework closes upload spools). Discard them (approved decision):

```bash
git checkout -- tests/test_endpoints.py
git status --short tests/test_endpoints.py
```

Expected: `git status` prints nothing for the file (clean).

- [ ] **Step 2: Write the failing `/health` tests**

In `tests/test_endpoints.py`, the current top is:

```python
import io
from contextlib import asynccontextmanager
```

Change it to:

```python
import io
import logging
import sys
from contextlib import asynccontextmanager
```

(`pytest`, `patch` are already imported in this file.)

Append to the end of `tests/test_endpoints.py`:

```python
# --- GET /health fd observability ---

class TestHealthFdStats:
    def test_health_reports_fd_stats(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["fd_soft_limit"] > 0
        if sys.platform.startswith("linux"):
            assert data["open_fds"] > 0

    def test_health_warns_when_fd_usage_high(self, client, caplog):
        with patch("main._fd_stats", return_value=(900, 1000)):
            with caplog.at_level(logging.WARNING, logger="main"):
                resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["open_fds"] == 900
        assert resp.json()["fd_soft_limit"] == 1000
        assert "fd usage" in caplog.text.lower()

    def test_health_no_warning_at_normal_usage(self, client, caplog):
        with patch("main._fd_stats", return_value=(100, 1000)):
            with caplog.at_level(logging.WARNING, logger="main"):
                resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["open_fds"] == 100
        assert "fd usage" not in caplog.text.lower()
```

- [ ] **Step 3: Run the tests and verify they FAIL (red)**

Run: `python -m pytest tests/test_endpoints.py::TestHealthFdStats -v`

Expected: 3 FAILED — `test_health_reports_fd_stats` with `KeyError: 'fd_soft_limit'`; the other two with `AttributeError: <module 'main' ...> does not have the attribute '_fd_stats'`.

- [ ] **Step 4: Implement `_fd_stats` and the `/health` fields**

In `app/main.py`, the imports currently start with:

```python
import os
import shutil
```

Change to:

```python
import os
import resource
import shutil
```

Still in `app/main.py`, directly above the `/health` endpoint (currently `@app.get("/health", tags=["Health"])` at line ~405), insert:

```python
def _fd_stats() -> tuple[int | None, int]:
    """Return (open_fds, soft_limit); open_fds is None where /proc is unavailable."""
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        open_fds = len(os.listdir("/proc/self/fd"))
    except OSError:
        return None, soft_limit
    return open_fds, soft_limit


```

Then replace the body of `health()`. Old:

```python
@app.get("/health", tags=["Health"])
async def health(model_manager: ModelManager = Depends(get_model_manager)):
    """Service health check endpoint."""

    # Check ffmpeg availability
    ffmpeg_available = True
    try:
        VideoFrameExtractor()
    except RuntimeError:
        ffmpeg_available = False

    return {
        "status": "healthy",
        "models_loaded": len(model_manager._preloaded) + len(model_manager._cached),
        "preloaded_count": len(model_manager._preloaded),
        "cached_count": len(model_manager._cached),
        "default_device": model_manager.default_device,
        "video_processing": ffmpeg_available
    }
```

New:

```python
@app.get("/health", tags=["Health"])
async def health(model_manager: ModelManager = Depends(get_model_manager)):
    """Service health check endpoint."""

    # Check ffmpeg availability
    ffmpeg_available = True
    try:
        VideoFrameExtractor()
    except RuntimeError:
        ffmpeg_available = False

    open_fds, fd_soft_limit = _fd_stats()
    if open_fds is not None and open_fds >= 0.8 * fd_soft_limit:
        logger.warning(
            "High fd usage: %d open fds >= 80%% of soft limit %d — possible fd leak",
            open_fds,
            fd_soft_limit,
        )

    return {
        "status": "healthy",
        "models_loaded": len(model_manager._preloaded) + len(model_manager._cached),
        "preloaded_count": len(model_manager._preloaded),
        "cached_count": len(model_manager._cached),
        "default_device": model_manager.default_device,
        "video_processing": ffmpeg_available,
        "open_fds": open_fds,
        "fd_soft_limit": fd_soft_limit
    }
```

- [ ] **Step 5: Run the tests and verify they PASS (green)**

Run: `python -m pytest tests/test_endpoints.py::TestHealthFdStats -v`
Expected: 3 PASSED.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass (the `/health` change adds fields only; no existing test asserts the exact key set).

- [ ] **Step 7: Commit**

```bash
git add app/main.py tests/test_endpoints.py
git commit -m "feat: expose fd usage in /health with high-usage warning"
```

---

## Task 2: Raise `nofile` ulimit (6 files) + persist MIOpen caches (2 AMD files)

**Files:**
- Modify: `docker/docker-compose-amd.yml`, `docker/docker-compose-cpu.yml`, `docker/docker-compose-nvidia.yml`
- Modify: `docker/deploy/docker-compose-amd.yml`, `docker/deploy/docker-compose-cpu.yml`, `docker/deploy/docker-compose-nvidia.yml`

**Interfaces:**
- Consumes/Produces: none (deployment config only). Independent of Task 1.

- [ ] **Step 1: Add `ulimits` to each of the 6 compose files**

Every file has exactly one line `    restart: unless-stopped` (4-space indent, service level). In **each** of the 6 files, replace that line with:

```yaml
    restart: unless-stopped
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
```

- [ ] **Step 2: Add MIOpen cache volumes + commented `MIOPEN_FIND_MODE` to `docker/docker-compose-amd.yml`**

Three edits in the dev AMD file.

Environment block — old:

```yaml
      HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-}
      YOLO_DEVICE: 'cuda:0'
```

New:

```yaml
      HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-}
      # Emergency lever against the MIOpen fd leak: fewer kernel compiles, at inference-perf cost.
      # MIOPEN_FIND_MODE: FAST
      YOLO_DEVICE: 'cuda:0'
```

Service volumes — old:

```yaml
    volumes:
      - models:/models
    restart: unless-stopped
```

New:

```yaml
    volumes:
      - models:/models
      - miopen-cache:/root/.cache/miopen    # compiled GPU kernels (MIOpen fd-leak containment)
      - miopen-config:/root/.config/miopen  # MIOpen find-db + lock files
    restart: unless-stopped
```

Top-level volumes (end of file) — old:

```yaml
volumes:
  models:
```

New:

```yaml
volumes:
  models:
  miopen-cache:
  miopen-config:
```

- [ ] **Step 3: Add MIOpen cache volumes + commented `MIOPEN_FIND_MODE` to `docker/deploy/docker-compose-amd.yml`**

Three edits in the deploy AMD file.

Environment block — old:

```yaml
      HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-}
    restart: unless-stopped
```

New:

```yaml
      HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-}
      # Emergency lever against the MIOpen fd leak: fewer kernel compiles, at inference-perf cost.
      # MIOPEN_FIND_MODE: FAST
    restart: unless-stopped
```

(Note: Step 1 already inserted `ulimits` after `restart: unless-stopped` here; keep both — the comment lines go *before* `restart:`.)

Service volumes — old:

```yaml
    volumes:
      - models:/models
```

New:

```yaml
    volumes:
      - models:/models
      - miopen-cache:/root/.cache/miopen    # compiled GPU kernels (MIOpen fd-leak containment)
      - miopen-config:/root/.config/miopen  # MIOpen find-db + lock files
```

Top-level volumes (end of file) — old:

```yaml
volumes:
  models:
```

New:

```yaml
volumes:
  models:
  miopen-cache:
  miopen-config:
```

- [ ] **Step 4: Validate all compose files**

Run (venv active, PyYAML present via ultralytics):

```bash
python - <<'PY'
import glob, sys, yaml

files = sorted(glob.glob("docker/docker-compose-*.yml") + glob.glob("docker/deploy/docker-compose-*.yml"))
assert len(files) == 6, files
ok = True
for f in files:
    d = yaml.safe_load(open(f))
    svc = next(iter(d["services"].values()))
    n = (svc.get("ulimits") or {}).get("nofile") or {}
    good = n.get("soft") == 65536 and n.get("hard") == 65536
    if "amd" in f:
        mounts = svc.get("volumes") or []
        declared = d.get("volumes") or {}
        good &= "miopen-cache:/root/.cache/miopen" in mounts
        good &= "miopen-config:/root/.config/miopen" in mounts
        good &= {"miopen-cache", "miopen-config"} <= set(declared)
    ok &= good
    print(("OK  " if good else "FAIL"), f)
sys.exit(0 if ok else 1)
PY
grep -l '# MIOPEN_FIND_MODE: FAST' docker/docker-compose-amd.yml docker/deploy/docker-compose-amd.yml
```

Expected: six `OK` lines and exit 0; `grep -l` prints both AMD file paths.

- [ ] **Step 5: Commit**

```bash
git add docker/docker-compose-amd.yml docker/docker-compose-cpu.yml docker/docker-compose-nvidia.yml \
        docker/deploy/docker-compose-amd.yml docker/deploy/docker-compose-cpu.yml docker/deploy/docker-compose-nvidia.yml
git commit -m "chore(docker): raise nofile ulimit, persist MIOpen caches on AMD"
```

---

## Task 3: Documentation (`api.md`, `CLAUDE.md`, `.env.example`)

**Files:**
- Modify: `.claude/rules/api.md` (the `### GET /health` response example)
- Modify: `CLAUDE.md` (Configuration env block)
- Modify: `docker/deploy/.env.example` (append at end)

**Interfaces:**
- Consumes: field names `open_fds` / `fd_soft_limit` exactly as produced by Task 1.

- [ ] **Step 1: Update the `/health` response example in `.claude/rules/api.md`**

Old:

```markdown
### GET /health

Health check.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "preloaded_count": 1,
  "cached_count": 1,
  "default_device": "cuda:0",
  "video_processing": true
}
```
```

New:

```markdown
### GET /health

Health check.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "preloaded_count": 1,
  "cached_count": 1,
  "default_device": "cuda:0",
  "video_processing": true,
  "open_fds": 123,
  "fd_soft_limit": 65536
}
```

`open_fds` counts `/proc/self/fd` entries (`null` where `/proc` is unavailable, e.g. non-Linux dev).
A WARNING is logged when `open_fds` reaches 80% of `fd_soft_limit` — early signal of an FD leak.
```

- [ ] **Step 2: Add the `MIOPEN_FIND_MODE` line to `CLAUDE.md`**

In the Configuration env block, old:

```
STABILIZER_MAX_STALENESS=5.0    # Max seconds before track becomes stale
```

New:

```
STABILIZER_MAX_STALENESS=5.0    # Max seconds before track becomes stale
# MIOPEN_FIND_MODE=FAST         # AMD emergency lever: fewer MIOpen kernel compiles (the ROCm
                                # fd-leak trigger) at inference-perf cost. Keep commented unless
                                # the fd-leak rate does not drop after the cache-volume fix.
```

- [ ] **Step 3: Append the knob entry to `docker/deploy/.env.example`**

Append at the end of the file (after the `HSA_OVERRIDE_GFX_VERSION` block):

```
# AMD only: emergency lever against the ROCm/MIOpen fd leak. If the leak rate does not drop
# after the miopen-cache volumes + raised nofile ulimit, uncomment `MIOPEN_FIND_MODE: FAST`
# in docker-compose-amd.yml (fewer GPU-kernel compiles, at some inference-performance cost).
```

- [ ] **Step 4: Commit**

```bash
git add .claude/rules/api.md CLAUDE.md docker/deploy/.env.example
git commit -m "docs: document /health fd fields and MIOPEN_FIND_MODE lever"
```

---

## Post-merge (user-side, informational — not plan tasks)

1. Rebuild images (`:latest` base pulls ROCm 7.2.4 — accepted) and redeploy with the updated compose files.
2. Verify: `docker exec deploy-vision-api-1 sh -c 'ulimit -n'` → 65536; `miopen-cache`/`miopen-config` volumes mounted; `/health` returns `open_fds`/`fd_soft_limit`.
3. Watch 2–3 days: `docker exec deploy-vision-api-1 sh -c "ls -l /proc/1/fd | grep -c '(deleted)'"` — expect the growth rate to fall from ~52/day to ~0 after cache warm-up; the cache volume grows then stabilizes (MB scale).
4. If the rate does not drop: MIOpen attribution is wrong → resume diagnosis (prod stays protected by the 65536 ceiling). Optional lever: uncomment `MIOPEN_FIND_MODE: FAST` in the AMD compose.

## Self-Review

**1. Spec coverage:**
- Spec §1 (ulimit ×6) → Task 2 Step 1. ✓
- Spec §2 (MIOpen volumes, AMD ×2, top-level declarations) → Task 2 Steps 2–3. ✓
- Spec §3 (`/health` fields + 80% WARNING + api.md) → Task 1 Steps 2–5, Task 3 Step 1. ✓
- Spec §4 (FIND_MODE: commented compose lines, `.env.example`, CLAUDE.md) → Task 2 Steps 2–3, Task 3 Steps 2–3. ✓
- Spec §5 (revert stub → Task 1 Step 1; SUPERSEDED banners → already committed in `5c4df1f`, noted in header). ✓
- Spec Testing (field/warning tests, full suite, compose YAML validation) → Task 1 Steps 3/5/6, Task 2 Step 4. ✓
- Spec Rollout → Post-merge section (user-side). ✓

**2. Placeholder scan:** No TBD/TODO; every code/config step shows complete content and exact commands. ✓

**3. Type consistency:** `_fd_stats() -> tuple[int | None, int]` defined in Task 1 Step 4; patched in tests as `patch("main._fd_stats", return_value=(900, 1000))` (Step 2) — same shape. Response keys `open_fds`/`fd_soft_limit` identical across Task 1 code, Task 1 tests, and Task 3 docs. ✓
