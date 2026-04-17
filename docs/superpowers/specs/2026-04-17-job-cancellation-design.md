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

- `QUEUED` → `cancel_event.set()`, status → `CANCELLED`, `completed_at = now`. The worker later pulls the id from the queue, sees `status != QUEUED`, and skips it.
- `PROCESSING` → `cancel_event.set()` only. Status transition to `CANCELLED` is done by the worker after `JobCancelledError` is raised from `annotate()`.
- `CANCELLED` → no-op, return the job as-is.
- `COMPLETED` / `FAILED` → `raise ValueError`.

New method:

```python
def mark_cancelled(self, job_id: str) -> None:
    """Called by the worker after JobCancelledError: status=CANCELLED, completed_at=now."""
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

FFmpeg subprocesses are cleaned up via the existing `FFmpegDecoder.__exit__` / `FFmpegEncoder.__exit__` when the exception propagates.

When `cancel_event is None`, behaviour is identical to the current code (backwards-compatible).

### Worker (`_annotation_worker` in `app/main.py`)

Two changes:

1. Immediately after `await job_manager.get_next_job_id()`, add a skip for queued jobs whose status was flipped to `CANCELLED` while they were sitting in the queue:

   ```python
   job_id = await job_manager.get_next_job_id()
   job = job_manager.get_job(job_id)
   if job is None or job.status != JobStatus.QUEUED:
       logger.info(f"Job {job_id} cancelled while queued, skipping")
       continue
   ```

2. Pass `cancel_event` into `annotate(...)` and catch `JobCancelledError` separately:

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

   `_cleanup_partial_output` is a small helper that does `output_path.unlink(missing_ok=True)`.

The `finally` block that deletes `input.mp4` remains unchanged — it runs regardless.

## Cleanup Behaviour (post-cancel)

- `input.mp4` is deleted by the existing worker `finally` block.
- `output.mp4` (if partially written in pass 2) is deleted explicitly in the `JobCancelledError` branch.
- The job record itself stays in `_jobs` with `status=CANCELLED` until TTL expiry, mirroring `COMPLETED`/`FAILED`. This allows `GET /jobs/{id}` to return `cancelled` for observability after the fact.
- `GET /jobs/{id}/download` on a cancelled job returns `400` via the existing check (`job.status != COMPLETED`).

## Cancellation Latency

With the cancel-event check placed **at the top of the per-frame loops** in both passes, the observed latency is bounded by a single frame's processing time:

- Pass 1: one YOLO inference (typically <200 ms on GPU, up to a few seconds on CPU for big models).
- Pass 2: one decode + render + encode frame (typically tens of ms).

Typical end-to-end: well under 1 second. Pathological (CPU, large model, 4K): a few seconds.

No per-frame overhead worth worrying about — `threading.Event.is_set()` is a fast atomic read.

## Error Handling & Edge Cases

| Case | Behaviour |
|------|-----------|
| Cancel between `get_next_job_id()` and `mark_processing()` | `request_cancel` sees `QUEUED`, flips to `CANCELLED`. Worker's new guard (`status != QUEUED`) skips the job. |
| Cancel arrives after annotator finishes last frame | Event set, but `annotate()` already returned `stats`. Worker takes success path → `COMPLETED`. Documented as expected race. |
| Repeated `/cancel` on a `CANCELLED` job | `request_cancel` returns the job unchanged. 200 OK with current state. `threading.Event.set()` is idempotent. |
| Unknown `job_id` | `request_cancel` → `KeyError` → 404. |
| TTL-cleaned `job_id` | Same as unknown → 404. |
| `/cancel` during worker's `mark_completed` / `mark_failed` call | Both `request_cancel` and `mark_*` run in the event loop (worker awaits executor), so they serialise via single-threaded asyncio. No lock needed. |

## Thread Safety

- `threading.Event` is the correct primitive for this cross-thread handoff: the event is set from the asyncio event-loop thread (inside `request_cancel`) and read from an executor worker thread (inside `annotate`). It is safe and lock-free for this pattern.
- `Job.status` is only written from the event-loop thread (endpoints, `_annotation_worker`). The executor thread only reads `job.cancel_event`. No additional synchronisation is required.

## Logging

| Site | Message |
|------|---------|
| `JobManager.request_cancel` | `Job {id}: cancel requested (was {prev_status})` |
| Worker, queued-skip branch | `Job {id} cancelled while queued, skipping` |
| Worker, `JobCancelledError` branch | `Job {id} cancelled during processing` |

All at `INFO` level.

## Testing

### `tests/test_job_manager.py` (existing file)

1. `test_request_cancel_queued` — create job (QUEUED) → `request_cancel` → status `CANCELLED`, event set, `completed_at` not None.
2. `test_request_cancel_processing` — force `status = PROCESSING` → `request_cancel` → event set, status remains `PROCESSING`.
3. `test_request_cancel_idempotent` — cancel twice → no raise, status stays `CANCELLED`.
4. `test_request_cancel_terminal_completed` — mark completed → `request_cancel` raises `ValueError`.
5. `test_request_cancel_terminal_failed` — mark failed → `request_cancel` raises `ValueError`.
6. `test_request_cancel_unknown` — unknown id → `KeyError`.
7. `test_cleanup_expired_includes_cancelled` — cancelled job past TTL is removed along with its directory.

### `tests/test_video_annotator.py` (existing file)

8. `test_annotate_respects_cancel_event_pass1` — mock YOLO + decoder yielding N frames; set `cancel_event` after 3 reads → `JobCancelledError` raised, FFmpeg subprocess terminated.
9. `test_annotate_respects_cancel_event_pass2` — event set after pass 1 completes → `JobCancelledError` from pass 2.
10. `test_annotate_without_cancel_event` — `cancel_event=None` → runs to completion, backwards-compat preserved.

### HTTP-level (`tests/test_endpoints.py`)

11. `POST /jobs/{id}/cancel` on queued job → 200, body matches `JobStatusResponse(status="cancelled")`.
12. `POST /jobs/{id}/cancel` on unknown id → 404.
13. `POST /jobs/{id}/cancel` on completed job → 409.

### Worker integration (`tests/test_worker.py`)

14. Worker skips jobs whose status became `CANCELLED` while queued (no annotator call).
15. Worker catches `JobCancelledError` from annotator and calls `mark_cancelled` (not `mark_failed`).

No end-to-end cancellation test with a real video is included (too slow / flaky for CI). Manual verification via `curl` is documented in `api.md`.

## Files Touched

| File | Change |
|------|--------|
| `app/job_manager.py` | Add `CANCELLED` status, `cancel_event` field, `request_cancel`, `mark_cancelled`. |
| `app/video_annotator.py` | Add `JobCancelledError`; accept `cancel_event` in `annotate`, `_pass1_collect`, `_pass2_render`; check at top of each loop. |
| `app/main.py` | Add `/jobs/{job_id}/cancel` endpoint; worker: skip-guard for cancelled-while-queued; pass `cancel_event`; catch `JobCancelledError`; partial-output cleanup helper. |
| `app/models.py` | No change (`JobStatusResponse.status` is `str`). |
| `.claude/rules/api.md` | Document new endpoint. |
| `CLAUDE.md` | Add `/jobs/{job_id}/cancel` row to endpoints table. |
| `tests/test_job_manager.py` | Add tests 1–7. |
| `tests/test_video_annotator.py` | Add tests 8–10. |
| `tests/test_endpoints.py` | Add tests 11–13. |
| `tests/test_worker.py` | Add tests 14–15. |

## Out of Scope / Future Work

- Bulk-cancel (`DELETE /jobs?status=queued`) — not needed now.
- Cancellation reason / audit trail — single status + log line is sufficient.
- Hard subprocess kill for lower latency — can be added later without breaking this API.
