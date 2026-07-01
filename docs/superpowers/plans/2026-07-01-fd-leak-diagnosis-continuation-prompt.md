## TASK

Continue the **root-cause diagnosis** of the production file-descriptor leak in `vision-api-server`
(project: `/home/zinin/git/vision-api-server`, branch `fix/upload-fd-leak`).

This is NOT normal plan execution. The original plan
(`docs/superpowers/plans/2026-07-01-fd-leak-upload-close.md`) rests on a **REFUTED premise** — it
assumed unclosed `UploadFile`s cause the leak, but FastAPI auto-closes form uploads across the entire
dependency pin (verified empirically), so the leak does not reproduce that way and the plan's
regression test cannot go red. A prior session pivoted from executing that plan to diagnosing the
REAL source. Diagnosis is well advanced but the exact leaking path is **not yet pinpointed**.

## CRITICAL: DO NOT START WORKING

**STOP. READ THIS CAREFULLY.**

After loading all context below, you MUST:
1. Read `.superpowers/sdd/diagnosis-findings.md` **FIRST** — it is the full diagnosis state.
2. Read the design + plan for background (knowing they rest on a refuted premise).
3. Report a brief summary of what you understood.
4. **WAIT for explicit user instructions** before ANY action.

**DO NOT:**
- Start implementing or changing code.
- Run commands beyond reading files.
- **Send ANY traffic to the production container `deploy-vision-api-1` without explicit user OK** —
  it is a live prod service on this host.
- Re-investigate anything in the "Refuted hypotheses" list.
- Assume the next step.

The user will tell you exactly what to do.

## PRIMARY ARTIFACT — read first

`.superpowers/sdd/diagnosis-findings.md` — established facts, refuted hypotheses (do NOT re-investigate),
open leads (top suspect: **ROCm/MIOpen temp files**), the decisive next step, and housekeeping.

## DOCUMENTS (background only; premise already refuted)

- Design: `docs/superpowers/specs/2026-07-01-fd-leak-upload-close-design.md`
- Plan: `docs/superpowers/plans/2026-07-01-fd-leak-upload-close.md`

## STATE SUMMARY

- **Leak is REAL and LIVE** on container `deploy-vision-api-1` (AMD/ROCm; runs on THIS host, docker
  29.5.3 accessible). Signature: `/tmp/#<inode> (deleted)`, **O_TMPFILE**, sizes **~1.4–2.0 MB**;
  historically **997** → EMFILE → `/health` 500. Deployed image built **2026-04-18 == current master**.
  Deployed deps: **fastapi 0.136.0 / starlette 1.0.0 / python-multipart 0.0.26**.
- **REFUTED (verified — don't redo):** unclosed-UploadFile on the pin (sweep fastapi 0.128–0.139 all
  `leaked=0`); starlette-1.0.0 regression (exact deployed combo in a minimal app `leaked=0`); CPU
  inference library leak (YOLO/cv2 `leaked=0`); app-code anonymous temp files (none in git history);
  custom middleware / BackgroundTasks / double `request.form()`; `image_utils.validate_and_decode_image`.
- **Key insight:** a minimal FastAPI app with the exact deployed deps does NOT leak, but the real
  container DOES → the trigger is **app code or environment, not the framework**. Under load the live
  `(deleted)` count hovers 0–2 (in-flight spools closing) → the leak is **slow/rare (~52/day)**.
- **TOP suspect (untested):** ROCm/MIOpen scratch temp files — prod is AMD/ROCm; the CPU inference test
  can't see it; live fds show MIOpen lock-files. If true, the plan's UploadFile fix is irrelevant.
  **Second suspect:** video endpoints under real ffmpeg (no ffmpeg on the dev host).
- **DECISIVE next step:** controlled test on the LIVE container — hit one endpoint type at a time,
  sample `ls -l /proc/1/fd | grep -c '(deleted)'` before/after, see whose delta sticks
  (image→MIOpen vs video→ffmpeg). Touches prod traffic → **requires explicit user OK**.

## HOUSEKEEPING / WORKING TREE

- `tests/test_endpoints.py` — **uncommitted** Steps 1–2 from the blocked Task 1 (FD test that can't
  reproduce locally). Decide: revert, or repurpose as an isolation/version-pinned test. Nothing committed.
- Project `.venv/` at repo root (fastapi 0.138.2). Scratch venvs/repro scripts were under the previous
  session's scratchpad (throwaway — may not exist in the new session).
- Branch `fix/upload-fd-leak`, BASE `cd17f3f`. No commits made this session.
- Task 2 (raise `nofile` ulimit → 65536 in 6 compose files) remains valid defense-in-depth regardless
  of root cause.

## PLAN QUALITY WARNING

The original plan's **core premise is already known to be refuted** (framework auto-closes uploads).
Treat the plan as background, not as instructions to execute. More generally the plan may contain
inaccuracies, wrong assumptions, or missing steps.

**If you notice any issue:** STOP, describe it, explain why, and ask the user how to proceed. Do NOT
silently work around it or make significant deviations without approval.

## INSTRUCTIONS

1. Read `.superpowers/sdd/diagnosis-findings.md`, then the design/plan for background.
2. Provide a brief summary of what you understood (state, refuted hypotheses, top suspect, next step).
3. **STOP and WAIT** — do not proceed with any investigation or change.
4. Ask: "С чего продолжаем — контролируемый live-тест контейнера, ветка MIOpen, или иное?"
