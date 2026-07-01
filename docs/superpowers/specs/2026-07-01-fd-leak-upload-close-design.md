# Design: Fix file-descriptor leak in upload endpoints

> **SUPERSEDED (2026-07-01).** The premise of this design was **refuted** during root-cause
> diagnosis: FastAPI/Starlette closes upload spools automatically (verified live and across
> fastapi 0.128–0.139), so unclosed `UploadFile`s are not the leak. The real source is a
> ROCm/MIOpen kernel-compile FD leak. Do **not** implement this design; see
> `2026-07-01-fd-leak-miopen-containment-design.md`. Only the `nofile` ulimit idea survives and
> is carried into the new design.

- **Date:** 2026-07-01
- **Status:** Superseded (premise refuted; do not implement)
- **Topic:** Reliably close `UploadFile` handles; regression test; `nofile` defense-in-depth

## Problem / Root cause

The production container `deploy-vision-api-1` (AMD image) was stuck `unhealthy` for ~31 h.
`/health` returned HTTP 500 `OSError: [Errno 24] Too many open files` every 30 s.

Evidence chain gathered during diagnosis:

- `/health` → `VideoFrameExtractor()` → `_verify_ffmpeg()` → `subprocess.run` → `os.pipe()` → `EMFILE`.
- Process pinned at **1018 / 1024** open FDs (soft `RLIMIT_NOFILE` = 1024, hard = 524288).
- **997** of those FDs were `(deleted)` anonymous temp files: `/tmp/#<inode> (deleted)`.
- These are `SpooledTemporaryFile` rollovers: FastAPI/Starlette `UploadFile` spools any upload
  larger than 1 MB to an on-disk temp file (`tempfile.TemporaryFile`, unlinked-on-create on Linux).
- The upload endpoints call `await file.read()` but **never** `await file.close()`. Every image/video
  upload > 1 MB leaked one FD. Over ~19 days of traffic the process hit the 1024 ceiling on
  2026-06-30 08:36:08 UTC and could no longer spawn `ffmpeg` (or any subprocess).
- `RestartCount = 0`, `OOMKilled = false`: soft failure, so `restart: unless-stopped` never fired
  (it reacts to process exit, not to `unhealthy`).

The leak exists in current `master`, not only in the deployed image — all five upload endpoints lack
`file.close()`.

Affected endpoints (all in `app/main.py`):

| Endpoint | Read pattern |
|---|---|
| `/detect` | `validate_and_decode_image(file, ...)` (reads in `image_utils`) |
| `/detect/video` | inline `await file.read()` |
| `/extract/frames` | inline `await file.read()` |
| `/detect/visualize` | `validate_and_decode_image(file, ...)` |
| `/detect/video/visualize` | streams `await file.read(1 MB)` chunks to a temp file |

## Goals

1. Guarantee every uploaded `UploadFile` is closed after the request, on every exit path
   (success, `HTTPException`, unhandled error, `StreamingResponse` completion).
2. Add a regression test that reproduces the leak and fails without the fix.
3. Raise `nofile` in all compose files as defense-in-depth.

## Non-goals

- Autoheal / orchestrator restart-on-`unhealthy` (explicitly declined by user).
- Refactoring `image_utils.py` (its `seek(0)` stays; harmless).
- The separate `ffmpeg_pipe` `close()` hardening (unrelated code path).
- Deploy (rebuild image + redeploy) — performed on the user's side after merge.

## Design

### 1. `uploaded_file` yield-dependency (`app/dependencies.py`)

A single choke point that owns the file lifecycle. A factory preserves each endpoint's OpenAPI
parameter description.

```python
from collections.abc import AsyncIterator
from fastapi import File, UploadFile

def uploaded_file(description: str = "Uploaded file"):
    """Factory: FastAPI dependency that yields the uploaded file and guarantees it is closed.

    The finally block runs after the response is sent on every exit path, so the
    SpooledTemporaryFile backing large uploads never leaks a file descriptor.
    """
    async def _dep(file: UploadFile = File(..., description=description)) -> AsyncIterator[UploadFile]:
        try:
            yield file
        finally:
            await file.close()
    return _dep
```

Rationale: DRY (one definition), idiomatic FastAPI, automatically covers future endpoints, and
handles all exit paths. Each `uploaded_file("...")` call returns a distinct inner function, so
FastAPI's per-request dependency cache never collides across endpoints.

### 2. Migrate the five endpoints (`app/main.py`)

Replace `file: UploadFile = File(..., description="X")` with
`file: UploadFile = Depends(uploaded_file("X"))`, keeping the existing descriptions:

- `/detect` — `"Image for analysis"`
- `/detect/video` — `"Video file for analysis"`
- `/extract/frames` — `"Video file for frame extraction"`
- `/detect/visualize` — `"Image for analysis"`
- `/detect/video/visualize` — `"Video file for annotation"`

Import `uploaded_file` from `dependencies`. Remove the now-unused `File` import from `main.py`
(keep `UploadFile` — still used as the type annotation). `image_utils.py` is unchanged: closing
happens at the dependency boundary, after the endpoint (and any `validate_and_decode_image` reads)
have finished; there are no re-reads after the response.

### 3. Defense-in-depth: `nofile` ulimit (6 compose files)

Add to the service in each file (`docker/docker-compose-{amd,cpu,nvidia}.yml` → service `yolo-api`;
`docker/deploy/docker-compose-{amd,cpu,nvidia}.yml` → service `vision-api`):

```yaml
ulimits:
  nofile:
    soft: 65536
    hard: 65536
```

This is not the fix — it raises the ceiling so any future FD leak takes far longer to bite and
gives monitoring time to catch it.

## Testing

New file `tests/test_upload_cleanup.py`:

- Linux-gated: `@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="…/proc…")`.
- Uses the existing `client` fixture (mocked `model_manager`, no-op lifespan).
- Counts **only `(deleted)` FDs** in `/proc/<pid>/fd` — the precise leak signature — to isolate
  from unrelated FDs (sockets, ffmpeg pipes):

```python
def _deleted_fd_count() -> int:
    fd_dir = f"/proc/{os.getpid()}/fd"
    n = 0
    for name in os.listdir(fd_dir):
        try:
            if "(deleted)" in os.readlink(os.path.join(fd_dir, name)):
                n += 1
        except OSError:
            pass
    return n
```

- Parametrized over all five endpoints (catches a forgotten migration). A >1 MB payload with the
  right extension forces `SpooledTemporaryFile` rollover. Warm up with 2 requests, then loop 20
  requests, assert `_deleted_fd_count()` delta ≤ small slack (e.g. 1). The response status is
  ignored — only FDs matter — so video endpoints need no ffmpeg mock (read+close happens before
  extraction; the endpoint may 4xx/5xx and that is fine).

Also run the full suite (`python -m pytest tests/ -v`): the `File(...)` → `Depends(...)` swap must
not break existing endpoint tests.

## Error handling / edge cases

- `await file.close()` is safe/idempotent in Starlette (calling it when already closed is a no-op);
  kept in `finally` so it runs on all exits and does not raise for normal temp files.
- `StreamingResponse` (`/detect/visualize`): teardown runs after streaming completes; the streamed
  JPEG comes from the already-decoded in-memory image, so closing the file cannot affect it.
- `/detect/video/visualize`: the background job processes the *saved* temp file, not the
  `UploadFile`; closing the upload after the 202 response is correct.

## Files touched

- `app/dependencies.py` — add `uploaded_file` factory (+ imports).
- `app/main.py` — migrate 5 endpoints; adjust imports.
- `tests/test_upload_cleanup.py` — new regression test.
- `docker/docker-compose-{amd,cpu,nvidia}.yml` — add `ulimits.nofile`.
- `docker/deploy/docker-compose-{amd,cpu,nvidia}.yml` — add `ulimits.nofile`.

## Rollout

1. Merge to `master`.
2. Rebuild the AMD/NVIDIA/CPU images and redeploy (user side).
3. The already-restarted container is healthy now; the rebuild ships the permanent fix + the higher
   `nofile` ceiling.
