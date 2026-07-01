> **SUPERSEDED (2026-07-01) — DO NOT EXECUTE.** The premise of the referenced plan (unclosed
> `UploadFile`) was refuted during diagnosis; the real leak is ROCm/MIOpen kernel compiles.
> This prompt is kept for history only. Use
> `2026-07-01-fd-leak-miopen-containment-execution-prompt.md` instead.

## TASK

Execute the implementation plan for the **upload file-descriptor leak fix** in the `vision-api-server` project.

Use `/superpowers:subagent-driven-development` skill for execution.

Work happens on the existing branch `fix/upload-fd-leak` (the design spec and plan are already committed there).

## DOCUMENTS

- Design: `docs/superpowers/specs/2026-07-01-fd-leak-upload-close-design.md`
- Plan: `docs/superpowers/plans/2026-07-01-fd-leak-upload-close.md`

Read both documents first.

## IMPORTANT: DO NOT START WORK YET

After reading the documents:
1. Confirm you have loaded all context.
2. Summarize your understanding briefly.
3. **WAIT for user instruction before taking any action.**

Do NOT begin implementation until the user explicitly tells you to start.

## SESSION CONTEXT

**Why this fix exists (root cause, established by live diagnosis):**
The production container `deploy-vision-api-1` (AMD image) was stuck `unhealthy` for ~31 h. `/health`
returned HTTP 500 `OSError: [Errno 24] Too many open files`. The uvicorn process was pinned at
**1018 / 1024** open FDs (`RLIMIT_NOFILE` soft = 1024); **997** of them were `(deleted)` anonymous
`/tmp/#<inode>` files — leaked `UploadFile` `SpooledTemporaryFile` rollovers. Every upload > 1 MB that
was never closed leaked one FD; over ~19 days the process hit the ceiling and could no longer spawn
`ffmpeg` (or any subprocess). `restart: unless-stopped` never fired because a health 500 is a *soft*
failure, not a process exit. The container was already restarted during the diagnosis session and is
healthy now — this plan is the **permanent code fix**, not incident response.

**Key decisions and rationale:**
- **Close mechanism = a yield-dependency *factory* `uploaded_file(description)`** in `app/dependencies.py`.
  FastAPI runs its `finally: await file.close()` after the response on every exit path (success,
  HTTPException, unhandled error, StreamingResponse completion). The factory form is deliberate: it
  preserves each endpoint's per-parameter OpenAPI description while centralizing the close in one place.
- **Test = FD-count integration test** that counts only `(deleted)` FDs in `/proc/<pid>/fd` — the exact
  leak signature — to isolate the UploadFile leak from unrelated FDs (sockets, ffmpeg pipes).
- **`nofile` ulimit raised to 65536** in all 6 compose files. This is defense-in-depth, NOT the fix.

**Rejected alternatives (do not "improve" the plan back into these):**
- Per-endpoint `try/finally` — repetitive, easy to forget in a new endpoint.
- Global middleware closing `request._form` — reaches into Starlette internals.
- An assert-`close()`-was-called unit test — tests the mechanism, not the outcome.
- A separate `tests/test_upload_cleanup.py` — rejected in favor of adding the test to
  `tests/test_endpoints.py` so it reuses the existing `client` / `mock_model_manager` fixtures without
  moving them to `conftest.py` (that move is a fiddly cross-file import refactor with no payoff here).
- "Just pull the latest image" — rejected: current `master` also lacks `file.close()`, so a rebuild
  alone does not fix it.

**Edge cases / things to watch:**
- **The red-verify step (Task 1 Step 3) is a hard gate.** The leak MUST reproduce under `TestClient`
  before you implement. It reproduces because the pinned `fastapi>=0.128` does not auto-close uploads
  (proven — production leaked 997 FDs on that line). The test calls `gc.disable()` during the measured
  loop so cycle-retained temp files stay visible. If the test unexpectedly PASSES before the fix, STOP
  and investigate — do not implement on top of a non-reproducing test.
- The leak only manifests for uploads **> 1 MB** (Starlette's 1 MB spool threshold); the test uses a
  2 MB payload. Smaller uploads stay in memory and never touch an FD.
- Video endpoints run `ffmpeg` synchronously; the test patches `main.extract_frames_from_video` with
  `AsyncMock(return_value=[])` so it is fast and only the UploadFile lifecycle is exercised. Status
  codes are irrelevant to the FD test (image endpoints 400 on the garbage payload; `/detect/video/visualize`
  returns 429 once `job_manager_for_tests` queue of 10 fills — all still parse + close the upload).
- The two `/detect` and `/detect/visualize` image-parameter lines are **identical** (`main.py:428` and
  `:799`) — edit with replace-all.
- No local `.venv` exists; create it first (`python3 -m venv .venv && source .venv/bin/activate &&
  pip install -r requirements.txt -r requirements-dev.txt`). Deps are heavy (torch, ultralytics, cv2).
- The FD-count test is Linux-only (guarded by `skipif`).
- Git commits in this repo surface an auto-configured identity warning — informational, ignore it.
- Deploy (rebuild + redeploy of the AMD/NVIDIA/CPU images) is on the user's side, outside this plan.

## PLAN QUALITY WARNING

The plan was written for a large task and may contain:
- Errors or inaccuracies in implementation details
- Oversights about edge cases or dependencies
- Assumptions that don't match the actual codebase
- Missing steps or incomplete instructions

**If you notice any issues during implementation:**
1. STOP before proceeding with the problematic step.
2. Clearly describe the problem you found.
3. Explain why the plan doesn't work or seems incorrect.
4. Ask the user how to proceed.

Do NOT silently work around plan issues or make significant deviations without user approval.
