# Job Cancellation for `/detect/video/visualize`

## Goal

Allow clients to cancel a video annotation job started via `POST /detect/video/visualize`, both when it is still queued and when it is actively being processed. Cancellation is cooperative: the running annotation stops at the next frame boundary and transitions the job to a new terminal status `CANCELLED`.

## Non-Goals

- Pausing / resuming jobs.
- Cancelling jobs across multiple worker processes (the system is designed for `workers=1` with in-memory state — cancellation inherits the same constraint).
- Hard-killing the FFmpeg subprocess to shorten cancellation latency (cleanup happens via existing context managers).

## Status Model

A new terminal status is introduced:

```python
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"   # NEW
```

`CANCELLED` is handled identically to `COMPLETED` and `FAILED` for TTL cleanup: `completed_at` is set to the cancellation time, and `cleanup_expired` removes the job record and directory once `ttl_seconds` elapse.

The `Job` dataclass gains one field:

```python
cancel_event: threading.Event = field(default_factory=threading.Event)
```

`JobStats` remains `None` for cancelled jobs — no statistics are produced because work was aborted.

## API

```
POST /jobs/{job_id}/cancel
```

- No request body.
- Tag: `Jobs`.

### Responses

| Current status | Response | Body |
|----------------|----------|------|
| `QUEUED` | `200 OK` | `JobStatusResponse` with `status: "cancelled"` |
| `PROCESSING` | `200 OK` | `JobStatusResponse` with `status: "processing"` (worker flips to `cancelled` when it observes the event) |
| `CANCELLED` | `200 OK` | `JobStatusResponse` (idempotent no-op) |
| `COMPLETED` | `409 Conflict` | `{"detail": "Cannot cancel job in terminal status 'completed'"}` |
| `FAILED` | `409 Conflict` | `{"detail": "Cannot cancel job in terminal status 'failed'"}` |
| unknown / TTL-cleaned | `404 Not Found` | `{"detail": "Job not found"}` |

Note: a client calling `/cancel` on a `PROCESSING` job will see `status: "processing"` in the immediate response. A follow-up `GET /jobs/{job_id}` transitions to `cancelled` once the annotator reaches the next cancellation checkpoint (sub-second in typical conditions, see Latency section).

### Race with normal completion

If `/cancel` arrives while the annotator is finalising the last frame, the job may have already transitioned to `COMPLETED` by the time the event would have been observed. This is correct behaviour: the client should check `GET /jobs/{job_id}` after `/cancel` and treat `COMPLETED` as "work finished before cancellation took effect". This is documented in the endpoint docstring.

## Cancellation Flow

### JobManager

New method:

```python
def request_cancel(self, job_id: str) -> Job:
    """Idempotent cancel request.

    Raises:
        KeyError: job_id unknown.
        ValueError: job is in a non-cancellable terminal status
                    (COMPLETED or FAILED).
    """
```

Behaviour:

- `QUEUED` → `cancel_event.set()`, status → `CANCELLED`, `completed_at = now`, **delete `input_path`** if it exists (prevents `input.mp4` leak — the worker's `finally` block never runs for skipped-queued jobs). The worker later pulls the id from the queue, sees `status != QUEUED`, and skips it.
- `PROCESSING` → `cancel_event.set()` only. Status transition to `CANCELLED` is done by the worker after `JobCancelledError` is raised from `annotate()` (or after a pre-annotate path observes the event — see Worker section).
- `CANCELLED` → no-op, return the job as-is.
- `COMPLETED` / `FAILED` → `raise ValueError`.

Modified method — `mark_processing` becomes a conditional compare-and-set:

```python
def mark_processing(self, job_id: str) -> bool:
    """CAS: flip QUEUED → PROCESSING atomically. Returns True if the transition
    happened, False if the job was no longer QUEUED (e.g. cancelled between
    the worker's guard check and this call). Worker must treat False as skip."""
```

This closes the race where `request_cancel` arrives after the worker's `status != QUEUED` guard but before `mark_processing` — without CAS, the unconditional assignment would overwrite `CANCELLED` with `PROCESSING`.

New method:

```python
def mark_cancelled(self, job_id: str) -> None:
    """Called by the worker after JobCancelledError (or after pre-annotate
    cancel observation): status=CANCELLED, completed_at=now."""
```

### Annotator

A new exception class lives in `app/video_annotator.py`:

```python
class JobCancelledError(Exception):
    """Raised inside annotate() when cancel_event is set."""
```

`VideoAnnotator.annotate(...)` gains an optional parameter:

```python
def annotate(
    self,
    input_path: Path,
    output_path: Path,
    params: AnnotationParams,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> AnnotationStats:
```

`cancel_event` is passed through to `_pass1_collect` and `_pass2_render`. Both methods check the event at the top of their `while True` loops — **before** calling `decoder.read_frame()` — so:

- In pass 1, no further YOLO batch is launched after cancellation.
- In pass 2, no further frame is decoded or written to the encoder.

```python
while True:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelledError()
    frame = decoder.read_frame()
    if frame is None:
        break
    ...
```

Additionally, `annotate()` itself performs **one extra check between the two passes** — after `DetectionStabilizer.stabilize()` returns and before `_pass2_render` spins up the new FFmpeg decoder + encoder subprocesses:

```python
# after stabilizer
if cancel_event is not None and cancel_event.is_set():
    raise JobCancelledError()
self._pass2_render(...)
```

This avoids an unnecessary FFmpeg startup (which is non-trivial for GPU-accelerated encoders) when cancellation arrived near the end of pass 1.

FFmpeg subprocesses are cleaned up via the existing `FFmpegDecoder.__exit__` / `FFmpegEncoder.__exit__` when the exception propagates.

When `cancel_event is None`, behaviour is identical to the current code (backwards-compatible).

### Worker (`_annotation_worker` in `app/main.py`)

Four changes:

1. Immediately after `await job_manager.get_next_job_id()`, add a skip for queued jobs whose status was flipped to `CANCELLED` while they were sitting in the queue. Use the new CAS `mark_processing` to close the race where `/cancel` arrives between the guard check and the status flip:

   ```python
   job_id = await job_manager.get_next_job_id()
   job = job_manager.get_job(job_id)
   if job is None or not job_manager.mark_processing(job_id):
       logger.info(f"Job {job_id} cancelled while queued, skipping")
       continue
   ```

   `mark_processing` returns `False` if the job is no longer `QUEUED`. Input file cleanup for queued-cancelled jobs already happened inside `request_cancel`, so the skip branch does not need to touch disk.

2. After loading the model (which can take seconds to minutes on cold-start / download) and before constructing the annotator, observe a possibly-arrived cancel:

   ```python
   model_entry = await model_manager.get_model(model_name)
   if job.cancel_event.is_set():
       logger.info(f"Job {job_id} cancelled during model load")
       job_manager.mark_cancelled(job_id)
       continue
   ```

3. Pass `cancel_event` into `annotate(...)` and catch `JobCancelledError` separately:

   ```python
   try:
       stats = await loop.run_in_executor(
           executor,
           lambda: annotator.annotate(
               input_path=job.input_path,
               output_path=output_path,
               params=params,
               progress_callback=progress_cb,
               cancel_event=job.cancel_event,
           ),
       )
       job_manager.mark_completed(job_id, output_path, stats={...})
   except JobCancelledError:
       logger.info(f"Job {job_id} cancelled during processing")
       job_manager.mark_cancelled(job_id)
       _cleanup_partial_output(output_path)
   except Exception as e:
       logger.error(f"Annotation failed for job {job_id}: {e}", exc_info=True)
       job_manager.mark_failed(job_id, str(e))
   ```

   `_cleanup_partial_output` wraps `output_path.unlink()` in `try/except OSError` with a warning log — a missing file or a permission error must not promote the job to `FAILED`.

4. Cancel has precedence over pre-annotate failures. In every pre-annotate `except` branch that currently calls `mark_failed` (model-load errors, annotator construction errors), first check `job.cancel_event.is_set()`; if it is, call `mark_cancelled` instead:

   ```python
   except (RuntimeError, ValueError) as e:
       if job.cancel_event.is_set():
           logger.info(f"Job {job_id} cancelled during model load (suppressed {type(e).__name__})")
           job_manager.mark_cancelled(job_id)
       else:
           job_manager.mark_failed(job_id, f"Model error: {e}")
       continue
   ```

   This gives clients a consistent contract: once `/cancel` has been accepted for a `PROCESSING` job, the terminal status is `CANCELLED` regardless of whether the worker was inside `get_model`, initialising the annotator, or inside `annotate()`. The existing `JobCancelledError` branch already enforces this symmetrically.

The `finally` block that deletes `input.mp4` remains unchanged — it runs for every PROCESSING path (success, cancel-during-processing, failure). For queued-cancelled jobs the input is deleted earlier in `request_cancel`.

## Cleanup Behaviour (post-cancel)

- `input.mp4` — for `QUEUED→CANCELLED` jobs deleted inside `request_cancel`; for `PROCESSING→CANCELLED` jobs deleted by the worker's existing `finally` block.
- `output.mp4` (if partially written in pass 2) is deleted explicitly in the `JobCancelledError` branch via `_cleanup_partial_output` (swallows `OSError` with warning log).
- The job record itself stays in `_jobs` with `status=CANCELLED` until TTL expiry, mirroring `COMPLETED`/`FAILED`. This allows `GET /jobs/{id}` to return `cancelled` for observability after the fact.
- `GET /jobs/{id}/download` on a cancelled job returns `400` via the existing check (`job.status != COMPLETED`).

## Cancellation Latency

Latency has two distinct components and the design document distinguishes them explicitly:

**Checkpoint latency** — time from `cancel_event.set()` to the thread raising `JobCancelledError` (or the worker observing the event in the pre-annotate gap). Bounded by:

- Inside `annotate`: one per-frame iteration — pass 1 = one YOLO inference (<200 ms on GPU, up to a few seconds on CPU for big models); pass 2 = one decode + render + encode frame (tens of ms).
- Pre-annotate: the duration of a single `get_model()` call; the post-`get_model` check is immediate.

Checkpoint latency is typically well under 1 second on GPU; on CPU with large models it can be a few seconds.

**Terminal-transition latency** — time from `JobCancelledError` being raised to the job actually reaching `status=CANCELLED` (observable via `GET /jobs/{id}`). The exception must propagate through the `with FFmpegDecoder(...):` / `with FFmpegEncoder(...):` context managers, which wait for the subprocesses to exit:

- `FFmpegDecoder.close()` timeout: up to ~10 seconds (`app/ffmpeg_pipe.py`).
- `FFmpegEncoder.close()` timeout: up to ~300 seconds. Encoders commonly need to flush buffers and rewrite the MP4 `moov` atom on close; in pathological cases (software encode of a 4K stream near the timeout) this can dominate the overall latency.

Clients polling `GET /jobs/{id}` after `/cancel` should therefore expect:

- typical GPU-path: observed `CANCELLED` within 1–2 seconds;
- CPU / large-model cases: several seconds;
- worst case (pass 2, large encoder buffer): tens of seconds up to a few minutes.

Hard-killing the FFmpeg subprocesses to shorten this latency is explicitly a non-goal (see Non-Goals).

No per-frame overhead in the hot loop worth worrying about — `threading.Event.is_set()` is a fast atomic read.

## Error Handling & Edge Cases

| Case | Behaviour |
|------|-----------|
| Cancel between `get_next_job_id()` and `mark_processing()` | Worker's guard reads `QUEUED` and calls the new CAS `mark_processing`. If `request_cancel` flipped the status to `CANCELLED` in between, CAS returns `False` and the worker skips. Input file was already deleted by `request_cancel`. |
| Cancel during `get_model()` (model load / download) | Worker observes `cancel_event.is_set()` immediately after `get_model()` returns and calls `mark_cancelled` without starting the annotator. |
| Cancel followed by pre-annotate failure (e.g. model-load error) | Pre-annotate `except` branches check `cancel_event.is_set()` first. If set, call `mark_cancelled`; else `mark_failed`. Cancel has precedence. |
| Cancel arrives after annotator finishes last frame | Event set, but `annotate()` already returned `stats`. Worker takes success path → `COMPLETED`. Documented as expected race. |
| Repeated `/cancel` on a `CANCELLED` job | `request_cancel` returns the job unchanged. 200 OK with current state. `threading.Event.set()` is idempotent. |
| Unknown `job_id` | `request_cancel` → `KeyError` → 404. |
| TTL-cleaned `job_id` | Same as unknown → 404. |
| `/cancel` during worker's `mark_completed` / `mark_failed` call | Both `request_cancel` and `mark_*` run in the event loop (worker awaits executor), so they serialise via single-threaded asyncio. No lock needed. |
| `output.mp4` unlink fails (`OSError`) after `JobCancelledError` | `_cleanup_partial_output` swallows the error with a warning log; status stays `CANCELLED`. TTL cleanup removes the leftover file later. |

## Thread Safety

- `threading.Event` is the correct primitive for this cross-thread handoff: the event is set from the asyncio event-loop thread (inside `request_cancel`) and read from an executor worker thread (inside `annotate`). It is safe and lock-free for this pattern.
- `Job.status` is only written from the event-loop thread (endpoints, `_annotation_worker`). The executor thread reads only `job.cancel_event`. No additional synchronisation is required.
- The `QUEUED → PROCESSING` transition is the only case where two actors potentially race on `Job.status` (both in the event-loop thread — `request_cancel` and `mark_processing`). The new CAS form of `mark_processing` resolves this by checking the expected `QUEUED` state before writing `PROCESSING`.

## Logging

| Site | Message |
|------|---------|
| `JobManager.request_cancel` | `Job {id}: cancel requested (was {prev_status})` |
| Worker, queued-skip branch | `Job {id} cancelled while queued, skipping` |
| Worker, `JobCancelledError` branch | `Job {id} cancelled during processing` |

All at `INFO` level.

## Testing

Counts are indicative; each task in the plan explicitly lists its tests.

### `tests/test_job_manager.py`

1. `request_cancel` on QUEUED → status `CANCELLED`, event set, `completed_at` set, **`input_path` deleted**.
2. `request_cancel` on PROCESSING → event set, status remains `PROCESSING`.
3. `request_cancel` idempotent on CANCELLED.
4. `request_cancel` on COMPLETED → `ValueError`.
5. `request_cancel` on FAILED → `ValueError`.
6. `request_cancel` on unknown id → `KeyError`.
7. `mark_processing` CAS: returns `True` on `QUEUED`, flips to `PROCESSING`.
8. `mark_processing` CAS: returns `False` on `CANCELLED` / `COMPLETED` / `FAILED`, does not overwrite.
9. `mark_cancelled` sets status and `completed_at`.
10. `cleanup_expired` includes CANCELLED.

### `tests/test_video_annotator.py`

11. `annotate` raises `JobCancelledError` when event set during pass 1 (after ~2 predict calls).
12. `annotate` raises `JobCancelledError` when event set between pass 1 and pass 2 — pass 2 decoder/encoder are never constructed.
13. `annotate` raises `JobCancelledError` when event set during pass 2 rendering.
14. `annotate` with `cancel_event=None` runs to completion (backwards compat).

### HTTP-level (`tests/test_endpoints.py`)

15. `POST /jobs/{id}/cancel` on queued job → 200 with `status="cancelled"`.
16. `POST /jobs/{id}/cancel` idempotent on cancelled job → 200.
17. `POST /jobs/{id}/cancel` on unknown id → 404.
18. `POST /jobs/{id}/cancel` on completed job → 409.
19. `POST /jobs/{id}/cancel` on failed job → 409.

### Worker integration (`tests/test_worker.py`)

20. Worker skips jobs cancelled while queued — annotator never called; queue drained (`_queue.empty()`); input file gone (asserted implicitly via `request_cancel` already having removed it).
21. Worker, when annotator raises `JobCancelledError`, calls `mark_cancelled` (not `mark_failed`), passes `cancel_event=job.cancel_event` into `annotate`, and deletes any partial output.
22. Worker cancels the job when `cancel_event` is set immediately after `get_model()` returns (model-load window path) — annotator never constructed.
23. Worker routes pre-annotate failure to `CANCELLED` instead of `FAILED` when `cancel_event.is_set()` — precedence.

No end-to-end cancellation test with a real video is included (too slow / flaky for CI). Manual verification via `curl` is documented in `api.md`.

## Files Touched

| File | Change |
|------|--------|
| `app/job_manager.py` | Add `CANCELLED` status; `cancel_event` field; `request_cancel` (deletes `input_path` for QUEUED); make `mark_processing` a CAS returning `bool`; add `mark_cancelled`; include `CANCELLED` in `cleanup_expired`. |
| `app/video_annotator.py` | Add `JobCancelledError`; accept `cancel_event` in `annotate`, `_pass1_collect`, `_pass2_render`; check at top of each per-frame loop; one additional check in `annotate()` between pass 1 and pass 2. |
| `app/main.py` | Add `/jobs/{job_id}/cancel` endpoint; worker: CAS-based skip for cancelled-while-queued; post-`get_model()` cancel observation; pass `cancel_event`; catch `JobCancelledError`; pre-annotate `except` branches route to `mark_cancelled` if event set; partial-output cleanup helper (`try/except OSError`). |
| `app/models.py` | Update `JobStatusResponse.status` field description to include `"cancelled"`. |
| `.claude/rules/api.md` | Document new endpoint. |
| `CLAUDE.md` | Add `/jobs/{job_id}/cancel` row to endpoints table. |
| `tests/test_job_manager.py` | Add tests 1–10. |
| `tests/test_video_annotator.py` | Add tests 11–14. |
| `tests/test_endpoints.py` | Add tests 15–19. |
| `tests/test_worker.py` | Add tests 20–23; update `_run_worker_until_job_done` helper to treat `CANCELLED` as terminal. |

## Out of Scope / Future Work

- Bulk-cancel (`DELETE /jobs?status=queued`) — not needed now.
- Cancellation reason / audit trail — single status + log line is sufficient.
- Hard subprocess kill for lower latency — can be added later without breaking this API.
