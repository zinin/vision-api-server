# Design: Contain the ROCm/MIOpen file-descriptor leak

- **Date:** 2026-07-01
- **Status:** Approved (ready for implementation plan)
- **Supersedes:** `2026-07-01-fd-leak-upload-close-design.md` (its premise — unclosed `UploadFile` —
  was refuted during diagnosis; only the `nofile` ulimit idea survives and is carried over here)
- **Topic:** Make the MIOpen kernel-compile FD leak harmless: persistent kernel cache, higher
  `nofile` ceiling, FD observability

## Problem / Root cause

The production container `deploy-vision-api-1` (AMD/ROCm image) accumulated 997 file descriptors
pointing at `/tmp/#<inode> (deleted)` O_TMPFILE temp files over ~19 days, hit the default
`RLIMIT_NOFILE` soft limit of 1024, and every code path that spawns a subprocess (including
`/health` → ffmpeg check) started failing with `OSError: [Errno 24] Too many open files`.

Diagnosis (the working session's raw findings file was not committed and is gone; this Problem
section is the surviving consolidated record) attributed the leak with high confidence to the
documented upstream ROCm/MIOpen bug (MIOpen #2223 / ROCm #2289):
`hipModuleLoad`-era code leaks FDs to compiled-kernel temp files when MIOpen **compiles a new GPU
kernel**. The originally suspected cause — upload endpoints not closing `UploadFile` — was
**refuted**: FastAPI/Starlette closes upload spools automatically (verified live under heavy prod
traffic and by a version sweep of fastapi 0.128–0.139 / starlette 0.50–1.3.1), and `app/` contains
no anonymous-temp-file mechanism at all.

Trigger mechanism, verified empirically on 2026-07-01 in `.venv` with real `yolo26n.pt` weights and
the app's exact call form (`model.predict(source=<numpy image>, imgsz=1024)`):

- ultralytics `predict` defaults to `rect=True` — a **method default** of `Model.predict()`
  (`engine/model.py`: `custom = {..., "rect": True}  # method defaults`), which overrides
  `cfg/default.yaml`'s `rect: False` (that one applies to train/val); for a `.pt` model this makes
  `LetterBox(auto=True)`, which pads to the *minimal* stride-32 rectangle instead of a square.
- Each new image **aspect-ratio bucket** therefore produces a new input tensor shape
  (16:9 → 576×1024, 4:3 → 768×1024, 9:16 → 1024×576, 1:1 → 1024×1024, …) → a new set of conv
  problem configs → MIOpen kernel compiles → leaked O_TMPFILE FD(s).
- The same aspect ratio at different source resolutions maps to the **same** shape, so the shape
  space is bounded (~64 buckets per model per `imgsz`; `imgsz` itself is client-controlled 32–2016).
- The observed rate (~52 FDs/day, far below the request rate) matches slow exploration of the shape
  space; leaked file sizes (1.4–2.0 MB) match compiled kernel code objects.
- The MIOpen kernel cache lives in the **ephemeral container layer** (`/root/.cache/miopen`,
  `/root/.config/miopen`), so every container recreation forgets all compiled kernels and restarts
  the warm-up leak from zero.

Constraints: the upstream bug has no confirmed fixed version (still present in the deployed
ROCm 7.2.2); the prod GPU (gfx1030 consumer iGPU via `HSA_OVERRIDE_GFX_VERSION=10.3.0`) is an
AMD-unsupported configuration, so "upgrade and hope" is a weak bet. We cannot fix MIOpen from this
repo — we **contain** it.

## Goals

1. Push FD exhaustion out of practical reach for this leak: raise the ceiling ~64× (organic
   traffic: ~19 days-to-EMFILE becomes ~3.4 years; see §1 for the adversarial worst case, which
   the ceiling alone does not fully cover).
2. Kernel compiles (the leak trigger) happen once per **cache lifetime**, not once per container:
   persist the MIOpen caches on volumes.
3. FD usage is observable (health payload + early-warning log), so this or any future FD leak is
   visible months before it bites instead of silently maturing for 19 days.

## Non-goals

- Fixing or upgrading MIOpen/ROCm. The base image stays `rocm/pytorch:latest`; a rebuild will pull
  ROCm 7.2.4 (minor bump over 7.2.2) — accepted by the user.
- Changing inference geometry (`rect=False` / square input): rejected — up to 1.78× more pixels per
  inference on the dominant 16:9 traffic for marginal FD benefit.
- The `uploaded_file` explicit-close dependency from the superseded design: premise refuted, dropped.
- Pre-fix confirmation traffic against prod (mechanism already confirmed locally; the ulimit
  backstop is cause-agnostic anyway).
- Autoheal / restart-on-unhealthy orchestration; periodic cron recycling.

## Design

### 1. `nofile` ulimit 1024 → 65536 (all 6 compose files)

In `docker/docker-compose-{amd,cpu,nvidia}.yml` (service `yolo-api`) and
`docker/deploy/docker-compose-{amd,cpu,nvidia}.yml` (service `vision-api`), after
`restart: unless-stopped`:

```yaml
ulimits:
  nofile:
    soft: 65536
    hard: 65536
```

Cause-agnostic backstop: at the observed ~52 FDs/day, time-to-EMFILE goes from ~19 days to
~3.4 years, and any *other* future slow leak gets the same headroom. (Host hard limit is 524288, so
65536 is safely grantable.)

Sizing rationale: the app's legitimate peak FD usage is in the low hundreds (10 queued video jobs,
4 executor workers, FFmpeg pipes, upload spools) — 65536 sits ~2 orders of magnitude above that,
and the raised ceiling cannot mask an ordinary descriptor bug: the 80% health warning (§3) fires
long before any legitimate workload could approach it.

Worst-case bound (the ~1000-FD warm-up estimate reflects *organic* traffic, not an upper bound):
`imgsz` is client-controlled (32–2016 → ~63 stride-32 values) and multiplies the aspect-bucket
space per model. A client deliberately sweeping shapes could push the one-time compile count into
the tens of thousands — same order as the 65536 ceiling itself. Practical exposure is low (internal
API, organic traffic uses a fixed `imgsz`), but the ceiling alone is not an adversarial-proof
guarantee. Each leaked FD also pins its deleted 1.4–2.0 MB temp file: the observed 997 FDs held
~1.7 GB of invisible disk on the container's writable layer, and shape-sweeping traffic would
scale that number too.

### 2. Persistent MIOpen caches (the two AMD compose files)

In `docker/docker-compose-amd.yml` and `docker/deploy/docker-compose-amd.yml`, add to the service:

```yaml
volumes:
  - miopen-cache:/root/.cache/miopen    # compiled kernel binaries (versioned subdirs)
  - miopen-config:/root/.config/miopen  # find-db (solver choices) + lock files
```

and declare `miopen-cache` / `miopen-config` in the top-level `volumes:` block (alongside the
existing `models` volume; both files have one).

Facts verified read-only in the live container: `HOME=/root`, process runs as uid 0, binary cache
already keyed by version subdirectory (`3.5.1.dabb6df2b9`) so a future ROCm bump simply creates a
sibling subdir — old entries waste a few MB at worst. Find-db filenames embed the MIOpen version
too (`gfx1030_3.HIP.3_5_1_…ufdb.txt`).

Effect: each conv config compiles **once per volume lifetime**. The leak shrinks to a one-time
warm-up (~1000 FDs ≈ 1.5% of the new limit, matching the historical full-exploration total) and is
no longer restarted by container recreation. Bonus: warm restarts skip re-tuning, so
first-inference latency after a deploy improves.

Volume lifecycle / limitations:

- `docker compose down -v` deletes named volumes — including the warmed kernel cache. Avoid `-v`
  on routine redeploys; cache survival across container recreations is the whole point.
- Size is MB-scale per MIOpen version (kernel binaries + find-db text). Versioned subdirs
  accumulate across ROCm upgrades; stale entries waste a few MB at worst. After a major upgrade it
  is safe (and optional) to `docker volume rm miopen-cache miopen-config` and re-warm.
- One container per volume pair: MIOpen keeps lock files under `~/.config/miopen`; sharing the
  volumes between concurrently running replicas is untested (lock contention) and out of scope —
  the deployment is single-container by design (`workers=1`, in-memory job state).

### 3. `/health` FD observability (`app/main.py`)

Extend the existing `/health` response (plain dict, `app/main.py:405`) with:

- `open_fds` — `len(os.listdir("/proc/self/fd"))`; `None` where `/proc` is unavailable
  (e.g. macOS dev machines).
- `fd_soft_limit` — `resource.getrlimit(resource.RLIMIT_NOFILE)[0]`; `0` where the `resource`
  module itself is absent (Windows — the import is guarded so `main.py` still imports there).

and log a `WARNING` through the module logger when `open_fds >= 0.8 * fd_soft_limit`,
**rate-limited to one record per hour** (module-level `time.monotonic()` stamp): the compose
healthcheck hits `/health` every 30 s, and an un-throttled warning would emit ~2880 identical
lines/day once over the threshold. The healthcheck turns this into a free periodic probe: the next
leak shows up in logs and in the health payload months before EMFILE.

Ordering requirement: `health()` collects `_fd_stats()` (and logs the warning) **before** the
ffmpeg availability probe, and that probe catches `OSError` in addition to `RuntimeError`.
`VideoFrameExtractor._verify_ffmpeg` spawns `ffmpeg`/`ffprobe` via `subprocess.run`, which is
exactly what fails with `EMFILE` (Errno 24) in the terminal phase of an FD leak — with the old
ordering the endpoint would 500 and the new fields would vanish right when they matter (that was
the original prod symptom).

Sketch:

```python
try:
    import resource
except ImportError:  # resource is Unix-only (absent on Windows)
    resource = None


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

`/health` adds the two fields; the endpoint's behaviour is otherwise unchanged (still returns 200
with `status: healthy`, including when the ffmpeg probe fails). Update the `/health` response
example in `.claude/rules/api.md`, noting that `open_fds` is a point-in-time snapshot and that the
values reflect the process's `RLIMIT_NOFILE` (in Docker — whatever `ulimits` grants).

### 4. `MIOPEN_FIND_MODE` — documented emergency knob, NOT set by default

MIOpen's default find mode stays in effect. Forcing `MIOPEN_FIND_MODE=FAST` would reduce compiles
further but permanently pins generic fallback kernels — a real inference-performance risk on this
already-weak iGPU (per MIOpen's "Find modes" documentation, FAST skips exhaustive solver tuning in
favour of heuristics; the penalty was not benchmarked here — expect slower convolutions on unlucky
configs), while the cache + ulimit already contain the leak. Instead:

- a commented-out `# MIOPEN_FIND_MODE: FAST` line with a one-line explanation in the `environment:`
  block of both AMD compose files;
- an entry in `docker/deploy/.env.example` and one line in `CLAUDE.md`'s configuration section:
  *emergency lever if the post-deploy leak rate does not drop*.

Activation mechanism — deliberately **edit-the-compose-file** (uncomment the line), NOT a
`${MIOPEN_FIND_MODE:-}` pass-through from `.env`: compose interpolation would always inject the
variable (empty string when unset), and how MIOpen parses an empty value is unverified. The docs
say this explicitly so a future "improvement" does not convert the lever into a pass-through.

### 5. Housekeeping

- Verify `tests/test_endpoints.py` is clean — the superseded plan's FD-leak test stub (a test for
  a bug that does not exist; the framework closes upload spools) was already discarded before the
  branch was pushed. If uncommitted changes are nevertheless present, inspect `git diff` first and
  drop only the stub.
- The superseded design and plan get a `SUPERSEDED` banner pointing here (done alongside this spec).

## Testing

- `/health` tests in `tests/test_endpoints.py` (existing `client` fixture): new fields present;
  on Linux `open_fds > 0` and `fd_soft_limit > 0`; warning path unit-tested by patching `_fd_stats`
  to return a value at/above 80% of the limit and asserting the log record (`caplog`); rate-limit
  covered (second hit within the hour logs nothing); `open_fds=None` branch covered (no procfs →
  `null` in JSON, no warning); EMFILE resilience covered by patching `main.VideoFrameExtractor`
  with `side_effect=OSError(24, ...)` and asserting 200 + FD fields + `video_processing: false`.
- Full suite (`python -m pytest tests/ -v`) inside `.venv`.
- Compose invariants as a **permanent test** `tests/test_compose.py` (PyYAML arrives transitively
  via ultralytics; guard with `pytest.importorskip("yaml")`): all 6 files have `ulimits.nofile`
  soft/hard == 65536; the two AMD files mount and declare the `miopen-*` volumes. A one-off
  validation script would let a future compose refactor silently drop the containment — the
  regression would resurface only as EMFILE weeks later.

## Rollout & validation (deploy is user-side)

1. Merge to `master`; rebuild images; redeploy with the updated compose files. Avoid
   `docker compose down -v` on routine redeploys — `-v` deletes the warmed cache volumes.
2. Immediately verify: `ulimit -n` inside the container = 65536; `miopen-cache`/`miopen-config`
   volumes mounted; `/health` returns `open_fds`/`fd_soft_limit`
   (`curl -s localhost:3001/health | jq '.open_fds, .fd_soft_limit'` — also handy as a periodic
   manual probe, since nothing consumes the WARNING automatically).
3. Expected first-start pattern: the volumes start **empty**, so the leak initially continues at
   the organic ~52 FDs/day while the cache warms across the active shape space — this is not a
   failure of the fix. Only subsequent restarts (warm cache) should show ~0/day from the start.
4. Watch for 2–3 days (read-only): `(deleted)` count in `/proc/1/fd` should fall from ~52/day to
   ~0 once the cache warms; the cache volume grows then stabilizes (MB scale).
5. Optional: once the cache volume stops growing, one deliberate container restart clears the
   FDs leaked during warm-up while keeping the warmed cache (a single reset, not an autoheal loop).
6. If the rate does **not** drop, the MIOpen attribution is wrong → resume diagnosis; prod is
   protected by the raised ceiling meanwhile. Optional lever: `MIOPEN_FIND_MODE=FAST`.
7. Rollback is safe at any point: remove the `ulimits`/volume lines from compose and redeploy
   (previous behaviour returns); `docker volume rm miopen-cache miopen-config` resets to an empty
   cache; the `/health` fields are purely additive.

## Files touched

- `docker/docker-compose-{amd,cpu,nvidia}.yml` — `ulimits`; AMD file also gets the two MIOpen
  volumes and the commented `MIOPEN_FIND_MODE` line.
- `docker/deploy/docker-compose-{amd,cpu,nvidia}.yml` — same.
- `docker/deploy/.env.example` — `MIOPEN_FIND_MODE` doc entry.
- `app/main.py` — guarded `import resource`; `_fd_stats` helper + two `/health` fields +
  hourly-rate-limited threshold warning, collected before the ffmpeg probe (probe also catches
  `OSError`).
- `.claude/rules/api.md` — `/health` response example.
- `CLAUDE.md` — one config line for the `MIOPEN_FIND_MODE` knob.
- `tests/test_endpoints.py` — verify clean (stub already discarded); add `/health` field, warning,
  rate-limit, no-procfs and EMFILE tests.
- `tests/test_compose.py` — new permanent compose-invariant test.
- `docs/superpowers/specs/2026-07-01-fd-leak-upload-close-design.md`,
  `docs/superpowers/plans/2026-07-01-fd-leak-upload-close.md` — `SUPERSEDED` banners.
