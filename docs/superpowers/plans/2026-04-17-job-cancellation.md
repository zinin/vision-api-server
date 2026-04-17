# Job Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /jobs/{job_id}/cancel` to abort a queued or processing video annotation job; introduce a new terminal status `CANCELLED` with cooperative cancellation checked on every frame.

**Architecture:** A `threading.Event` on each `Job` is set by the event-loop thread in `JobManager.request_cancel` and polled by the executor thread at the top of both decode loops in `VideoAnnotator.annotate`, with one additional check in `annotate()` between pass 1 and pass 2. A new `JobCancelledError` is raised on the worker, which then calls `mark_cancelled` and deletes any partial output. Queue-time cancellation skips the job in the worker dispatch loop via a CAS `mark_processing`. Cancel takes precedence over pre-annotate failures — if the event is set, any pre-annotate exception routes to `CANCELLED` instead of `FAILED`.

**Tech Stack:** Python 3.12, FastAPI, asyncio, `threading.Event`, pytest.

**Spec:** `docs/superpowers/specs/2026-04-17-job-cancellation-design.md`

---

## Status: ALL TASKS COMPLETE

All 7 plan tasks are implemented, reviewed (internal + 7 external reviewers), and all "valid" findings from external review have been applied.

---

## Task 1: Add `CANCELLED` status and `cancel_event` field

✅ Done — see commit: `644b20b`

## Task 2: Implement `JobManager.request_cancel` and CAS `mark_processing`

✅ Done — see commit: `363b0fe`

## Task 3: Implement `JobManager.mark_cancelled` and TTL inclusion

✅ Done — see commit: `f68144b`

## Task 4: Add `JobCancelledError` and cancellation checks in `VideoAnnotator.annotate`

✅ Done — see commit: `d8c74e7`

## Task 5: Worker — skip queued-but-cancelled, handle `JobCancelledError`

✅ Done — see commits: `fbe8c03`, `9e857e1` (fixup: distinct log for missing-job vs queued-cancel skip)

## Task 6: Add `POST /jobs/{job_id}/cancel` endpoint

✅ Done — see commits: `cf1556c`, `551e849` (fixup: exception chaining + PROCESSING test)

## Task 7: Documentation and OpenAPI schema

✅ Done — see commits: `4fe4e44`, `77ff9f9` (fixup: clarify example + 409 body)

---

## External code-review fixes (iteration over original plan)

Applied after running `/external-code-review default` against all 7 reviewers (Claude, Codex, Gemini, CCS×5). 225/226 tests pass; 1 pre-existing env failure unrelated.

- `114f713` — fix(jobs): count cancelled-queued jobs against capacity (close bypass). **Codex Critical** — `check_queue_capacity` used `status==QUEUED`, but `request_cancel` left the id in `asyncio.Queue`. Changed to `self._queue.qsize()`.
- `406b58c` — fix(worker): terminalize job on setup-time exception (cancel precedence). **Codex Important** — setup code between post-`get_model` check and inner try could leave jobs stuck in PROCESSING. Wrapped in try/except with cancel precedence.
- `fc597d0` — fix(jobs): mark_cancelled refuses to overwrite terminal states. Defensive guard against future misuse.
- `81a131e` — refactor(worker): distinguish post-model-load cancel log from failure path. Disambiguate grep-able messages.
- `8936a25` — refactor(api): remove dead COMPLETED branch from cancel endpoint. `request_cancel` never returns COMPLETED.
- `628a171` — refactor(worker): simplify partial output cleanup with missing_ok.
- `c96690e` — test(api): double-cancel on PROCESSING job is idempotent.
- `24cbb9d` — docs(jobs): document cross-thread invariant on JobManager.
- `dfac161` — refactor(jobs): debug-level log for idempotent cancel no-op.
- `ab1471d` — docs(api): explain cancel latency components and completion race.
- `49c7277` — refactor(models): use Literal for JobStatusResponse.status (OpenAPI enum).

---

## Remaining (post-plan) work

The code work is done. What remains is branch-finishing hygiene per user's global CLAUDE.md convention:

1. `git rm docs/superpowers/specs/2026-04-17-job-cancellation-design.md`
2. `git rm docs/superpowers/specs/2026-04-17-job-cancellation-review-iter-1.md`
3. `git rm docs/superpowers/specs/2026-04-17-job-cancellation-review-merged-iter-1.md`
4. `git rm docs/superpowers/plans/2026-04-17-job-cancellation.md` (this file)
5. Commit removal
6. Optionally squash fixup commits into their parent feature commits
7. Open PR via `gh pr create`

The `superpowers:finishing-a-development-branch` skill handles this.
