# MIOpen FD-Leak Containment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ROCm/MIOpen kernel-compile FD leak harmless: raise the `nofile` ceiling in all compose files, persist the MIOpen caches on volumes so kernels compile once per volume lifetime, and expose FD usage in `/health` with an early-warning log.

**Architecture:** No app-logic changes beyond `/health`. Containment lives in deployment config (ulimits + named volumes in docker-compose) plus a small observability addition in `app/main.py` (`_fd_stats()` helper, two new `/health` fields, hourly-rate-limited threshold WARNING collected before the ffmpeg probe). The refuted `UploadFile` fix is dropped; its test stub was already discarded — Task 1 Step 1 only verifies the file is clean.

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
source .venv/bin/activate
python --version    # HEALTH CHECK — must print a version, see below
```

The repo's `.venv` may exist but be BROKEN: at review time `.venv/bin/python` was a symlink to a
missing `/usr/bin/python3.13` (system Python is 3.12) — `source activate` succeeds silently and
the first `pytest` dies with "No such file or directory". If the health check fails (or `.venv`
is missing entirely), rebuild it:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## File Structure

- **Modify** `tests/test_endpoints.py` — verify the tree is clean (the superseded FD-leak stub was already discarded), then add `/health` FD-stats tests (imports at top; new test class at end).
- **Create** `tests/test_compose.py` — permanent compose-invariant test (Task 2).
- **Modify** `app/main.py` — guarded `import resource`; `_fd_stats()` helper + module-level warning timestamp above the `/health` endpoint; two new fields + rate-limited WARNING collected before the ffmpeg probe in `health()`.
- **Modify** `docker/docker-compose-{amd,cpu,nvidia}.yml` (service `yolo-api`) and `docker/deploy/docker-compose-{amd,cpu,nvidia}.yml` (service `vision-api`) — `ulimits.nofile` in all six; the two AMD files also get `miopen-cache`/`miopen-config` volumes and a commented `MIOPEN_FIND_MODE` line.
- **Modify** `docker/deploy/.env.example`, `CLAUDE.md`, `.claude/rules/api.md` — documentation.

---

## Task 1: `/health` FD observability (+ revert the refuted-test stub)

**Files:**
- Modify: `tests/test_endpoints.py` (verify clean, then edit imports at top; append test class at end)
- Modify: `app/main.py` (imports at top; helper + endpoint changes around line 405)

**Interfaces:**
- Produces: `_fd_stats() -> tuple[int | None, int]` in `app/main.py` — returns `(open_fds, soft_limit)`; `open_fds` is `None` where `/proc/self/fd` is unavailable; `(None, 0)` where the `resource` module is absent (Windows). `/health` response gains `"open_fds": int | None` and `"fd_soft_limit": int`. Tests patch it as `patch("main._fd_stats", return_value=(900, 1000))`. The WARNING is rate-limited via module-level `_last_fd_warning_ts: float | None` (tests reset it to `None` in an autouse fixture).
- Consumes: existing `client` fixture in `tests/test_endpoints.py`; module logger `logger = logging.getLogger(__name__)` in `app/main.py` — logger name is `main` because `tests/conftest.py` puts `app/` on `sys.path` and the module imports as `main`. `time` is ALREADY imported in `app/main.py` (do not duplicate). `VideoFrameExtractor` is already imported into `main`'s namespace (`from video_utils import ...`), so `patch("main.VideoFrameExtractor", ...)` works.

- [ ] **Step 1: Verify `tests/test_endpoints.py` is clean**

The superseded plan's FD-leak test stub was already discarded before the branch was pushed, so on
a fresh checkout this is a verification, not an action:

```bash
git status --short tests/test_endpoints.py
```

Expected: no output (file clean) → move on. If the file IS dirty (unexpected local edits), run
`git diff tests/test_endpoints.py` first and discard only stub-related changes — do NOT run a
blanket `git checkout --` without inspecting the diff.

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

Append to the end of `tests/test_endpoints.py`. The logger name is `"main"` (kept in one place as
the `MAIN_LOGGER` constant) because `tests/conftest.py` adds `app/` to `sys.path` and the module
imports as `main` — if the module is ever moved/renamed, update the constant:

```python
# --- GET /health fd observability ---

MAIN_LOGGER = "main"


class TestHealthFdStats:
    @pytest.fixture(autouse=True)
    def _reset_fd_warning_state(self):
        import main
        main._last_fd_warning_ts = None
        yield

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
            with caplog.at_level(logging.WARNING, logger=MAIN_LOGGER):
                resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["open_fds"] == 900
        assert resp.json()["fd_soft_limit"] == 1000
        assert "fd usage" in caplog.text.lower()

    def test_health_warning_rate_limited(self, client, caplog):
        with patch("main._fd_stats", return_value=(900, 1000)):
            with caplog.at_level(logging.WARNING, logger=MAIN_LOGGER):
                client.get("/health")          # first hit over the threshold logs
                caplog.clear()
                client.get("/health")          # second hit within the hour must NOT
        assert "fd usage" not in caplog.text.lower()

    def test_health_no_warning_at_normal_usage(self, client, caplog):
        with patch("main._fd_stats", return_value=(100, 1000)):
            with caplog.at_level(logging.WARNING, logger=MAIN_LOGGER):
                resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["open_fds"] == 100
        assert "fd usage" not in caplog.text.lower()

    def test_health_handles_missing_procfs(self, client, caplog):
        with patch("main._fd_stats", return_value=(None, 1000)):
            with caplog.at_level(logging.WARNING, logger=MAIN_LOGGER):
                resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["open_fds"] is None
        assert "fd usage" not in caplog.text.lower()

    def test_health_survives_emfile_in_ffmpeg_check(self, client):
        with patch("main._fd_stats", return_value=(100, 1000)):
            with patch("main.VideoFrameExtractor", side_effect=OSError(24, "Too many open files")):
                resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_processing"] is False
        assert data["open_fds"] == 100
        assert data["fd_soft_limit"] == 1000
```

- [ ] **Step 3: Run the tests and verify they FAIL (red)**

Run: `python -m pytest tests/test_endpoints.py::TestHealthFdStats -v`

Expected: 6 FAILED (red) — `test_health_reports_fd_stats` with `KeyError: 'fd_soft_limit'`;
`test_health_survives_emfile_in_ffmpeg_check` fails because today's `/health` catches only
`RuntimeError` (the patched `OSError` escapes → 500, or the fields are missing → `KeyError`);
the four `patch("main._fd_stats", ...)` tests with `AttributeError: <module 'main' ...> does not
have the attribute '_fd_stats'`. The exact failure shape may vary — the point of red is that every
test fails BEFORE the implementation exists.

- [ ] **Step 4: Implement `_fd_stats` and the `/health` fields**

In `app/main.py`, the imports currently start with:

```python
import os
import shutil
```

Change to (guarded import — `resource` is Unix-only, a bare `import resource` would break
`main.py` and the whole test suite on Windows; `time` is already imported on the next line of the
file — do NOT add a duplicate):

```python
import os
import shutil
try:
    import resource
except ImportError:  # resource is Unix-only (absent on Windows)
    resource = None
```

Still in `app/main.py`, directly above the `/health` endpoint (currently `@app.get("/health", tags=["Health"])` at line ~405), insert:

```python
_last_fd_warning_ts: float | None = None


def _fd_stats() -> tuple[int | None, int]:
    """Return (open_fds, soft_limit); open_fds is None where /proc is unavailable."""
    if resource is None:
        return None, 0
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

New. Three deliberate properties: (1) FD stats are collected FIRST — at EMFILE the ffmpeg probe
below is the thing that fails, and the numbers/warning must still surface (the original prod
symptom was `/health` itself dying of EMFILE); (2) the probe catches `OSError` in addition to
`RuntimeError` for the same reason; (3) the warning is rate-limited to once per hour because the
compose healthcheck hits this endpoint every 30 s (~2880 identical lines/day otherwise):

```python
@app.get("/health", tags=["Health"])
async def health(model_manager: ModelManager = Depends(get_model_manager)):
    """Service health check endpoint."""
    global _last_fd_warning_ts

    # FD stats before anything that spawns subprocesses: at EMFILE the ffmpeg
    # probe below fails, and these numbers must still make it into the response.
    open_fds, fd_soft_limit = _fd_stats()
    if open_fds is not None and fd_soft_limit > 0 and open_fds >= 0.8 * fd_soft_limit:
        now = time.monotonic()
        if _last_fd_warning_ts is None or now - _last_fd_warning_ts >= 3600.0:
            _last_fd_warning_ts = now
            logger.warning(
                "High fd usage: %d open fds >= 80%% of soft limit %d — possible fd leak",
                open_fds,
                fd_soft_limit,
            )

    # Check ffmpeg availability (spawns ffmpeg/ffprobe → OSError covers EMFILE)
    ffmpeg_available = True
    try:
        VideoFrameExtractor()
    except (RuntimeError, OSError):
        ffmpeg_available = False

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
Expected: 6 PASSED.

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

Every file has exactly one line `    restart: unless-stopped` (4-space indent, service level) — verify before editing: `grep -c 'restart: unless-stopped' <file>` must print `1` for each; if it prints more, STOP and inspect instead of guessing which occurrence to replace. In **each** of the 6 files, replace that line with:

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

- [ ] **Step 4: Add the permanent compose-invariant test and run it**

A one-off validation script would let a future compose refactor silently drop the ulimits or the
MIOpen volumes — the regression would resurface only as EMFILE weeks later in prod. So the same
checks become a committed test. Create `tests/test_compose.py` with exactly:

```python
"""Deployment-config invariants for the MIOpen FD-leak containment.

Guards docker-compose files against refactors that would silently drop the
`nofile` ulimits or the AMD MIOpen cache volumes (see
docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md).
"""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")  # PyYAML arrives transitively via ultralytics

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILES = sorted((REPO_ROOT / "docker").glob("docker-compose-*.yml")) + sorted(
    (REPO_ROOT / "docker" / "deploy").glob("docker-compose-*.yml")
)
AMD_FILES = [p for p in COMPOSE_FILES if "amd" in p.name]


def _service(data):
    return next(iter(data["services"].values()))


def test_expected_compose_files_present():
    assert len(COMPOSE_FILES) == 6, COMPOSE_FILES
    assert len(AMD_FILES) == 2, AMD_FILES


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_nofile_ulimit_is_65536(path):
    svc = _service(yaml.safe_load(path.read_text()))
    nofile = (svc.get("ulimits") or {}).get("nofile") or {}
    assert nofile.get("soft") == 65536, path
    assert nofile.get("hard") == 65536, path


@pytest.mark.parametrize("path", AMD_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_amd_miopen_cache_volumes(path):
    data = yaml.safe_load(path.read_text())
    svc = _service(data)
    mounts = svc.get("volumes") or []
    assert "miopen-cache:/root/.cache/miopen" in mounts, path
    assert "miopen-config:/root/.config/miopen" in mounts, path
    declared = data.get("volumes") or {}
    assert {"miopen-cache", "miopen-config"} <= set(declared), path
```

(YAML inline comments on the volume lines do not reach the parsed values, so the string equality
above is exact.)

Run:

```bash
python -m pytest tests/test_compose.py -v
grep -l '# MIOPEN_FIND_MODE: FAST' docker/docker-compose-amd.yml docker/deploy/docker-compose-amd.yml
```

Expected: all tests PASS (9 items: 1 presence + 6 ulimit + 2 AMD-volume); `grep -l` prints both AMD file paths.

- [ ] **Step 5: Commit**

```bash
git add docker/docker-compose-amd.yml docker/docker-compose-cpu.yml docker/docker-compose-nvidia.yml \
        docker/deploy/docker-compose-amd.yml docker/deploy/docker-compose-cpu.yml docker/deploy/docker-compose-nvidia.yml \
        tests/test_compose.py
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

`open_fds` counts `/proc/self/fd` entries (`null` where `/proc` is unavailable, e.g. non-Linux dev;
`fd_soft_limit` is `0` where the `resource` module is absent). Values are a point-in-time snapshot
and reflect the process's `RLIMIT_NOFILE` (in Docker — whatever `ulimits` grants; on a bare host —
the shell default). A WARNING is logged (at most once per hour) when `open_fds` reaches 80% of
`fd_soft_limit` — early signal of an FD leak.
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
                                # fd-leak trigger) at inference-perf cost (generic fallback
                                # kernels). NOT read from .env — enable by uncommenting the line
                                # in docker-compose-amd.yml. Keep off unless the fd-leak rate
                                # does not drop after the cache-volume fix.
```

- [ ] **Step 3: Append the knob entry to `docker/deploy/.env.example`**

Append at the end of the file (after the `HSA_OVERRIDE_GFX_VERSION` block):

```
# AMD only: emergency lever against the ROCm/MIOpen fd leak. If the leak rate does not drop
# after the miopen-cache volumes + raised nofile ulimit, uncomment `MIOPEN_FIND_MODE: FAST`
# in docker-compose-amd.yml (fewer GPU-kernel compiles, at some inference-performance cost).
# Intentionally NOT a ${MIOPEN_FIND_MODE:-} pass-through: compose interpolation would always
# inject the variable (empty when unset), and MIOpen's parsing of an empty value is unverified —
# editing the compose file is the safer switch.
```

- [ ] **Step 4: Commit**

```bash
git add .claude/rules/api.md CLAUDE.md docker/deploy/.env.example
git commit -m "docs: document /health fd fields and MIOPEN_FIND_MODE lever"
```

---

## Post-merge (user-side, informational — not plan tasks)

1. Rebuild images (`:latest` base pulls ROCm 7.2.4 — accepted) and redeploy with the updated compose files. Do NOT use `docker compose down -v` on routine redeploys — `-v` deletes the warmed MIOpen cache volumes.
2. Verify: `docker exec deploy-vision-api-1 sh -c 'ulimit -n'` → 65536; `miopen-cache`/`miopen-config` volumes mounted; `/health` returns `open_fds`/`fd_soft_limit` (`curl -s localhost:3001/health | jq '.open_fds, .fd_soft_limit'` — also handy as a periodic manual probe; nothing consumes the WARNING automatically).
3. Expected first-start pattern: the volumes start empty → the leak continues at the organic ~52 FDs/day while the cache warms over the active shape space; this is NOT a failure of the fix. Only later restarts (warm cache) should show ~0/day immediately.
4. Watch 2–3 days: `docker exec deploy-vision-api-1 sh -c "ls -l /proc/1/fd | grep -c '(deleted)'"` — expect the growth rate to fall from ~52/day to ~0 after cache warm-up; the cache volume grows then stabilizes (MB scale).
5. Optional: once the cache volume stops growing, one deliberate container restart clears the warm-up-leaked FDs while keeping the cache (a single reset, not an autoheal loop).
6. If the rate does not drop: MIOpen attribution is wrong → resume diagnosis (prod stays protected by the 65536 ceiling). Optional lever: uncomment `MIOPEN_FIND_MODE: FAST` in the AMD compose (edit the file — deliberately not read from `.env`).
7. Rollback (if ever needed): remove the `ulimits`/volume lines from compose + redeploy; `docker volume rm miopen-cache miopen-config` resets the cache; the `/health` fields are additive and need no rollback.

## Before opening the PR (user's global workflow — do not skip)

As the final change on the branch, remove the working documents so they never appear in the PR
diff (they stay available in branch history):

```bash
git rm -r docs/superpowers/
git commit -m "docs: drop superpowers working documents before PR"
git push
```

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
