## TASK

Execute the implementation plan for the **ROCm/MIOpen FD-leak containment** in the
`vision-api-server` project.

Use `/superpowers:subagent-driven-development` (recommended) or `/superpowers:executing-plans`
for execution.

Work happens on the existing branch **`fix/upload-fd-leak`** (the design spec, the plan, and this
prompt are already committed and pushed there — `git pull` first if needed). All paths in this
prompt and in the plan are **relative to the repo root** (this machine's absolute path differs
from the machine where the plan was written — do not assume any `/home/...` prefix).

Converse in **Russian** (user preference).

## DOCUMENTS

- Design spec: `docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md`
- Plan: `docs/superpowers/plans/2026-07-01-fd-leak-miopen-containment.md`
- Background (do NOT execute; premise refuted, kept for history):
  `docs/superpowers/specs/2026-07-01-fd-leak-upload-close-design.md`,
  `docs/superpowers/plans/2026-07-01-fd-leak-upload-close.md`

Read the spec and the plan first.

## IMPORTANT: DO NOT START WORK YET

After reading the documents:
1. Confirm you have loaded all context.
2. Summarize your understanding briefly.
3. **WAIT for user instruction before taking any action.**

Do NOT begin implementation until the user explicitly tells you to start.

## SESSION CONTEXT

**Why this fix exists (root cause, established by live diagnosis + local verification):**
The production container (AMD/ROCm image, runs on a DIFFERENT host — not this machine) accumulated
997 `(deleted)` O_TMPFILE FDs over ~19 days, hit `RLIMIT_NOFILE` soft = 1024, and every subprocess
spawn (including the `/health` ffmpeg check) failed with `OSError: [Errno 24] Too many open files`.
Root cause: the documented upstream **ROCm/MIOpen kernel-compile FD leak** (MIOpen #2223 /
ROCm #2289) — MIOpen leaks FD(s) each time it compiles a NEW GPU kernel. The trigger was verified
empirically: ultralytics `predict` defaults to `rect=True` → letterbox to the minimal stride-32
rectangle → **each new image aspect-ratio bucket = new input shape = new conv configs = kernel
compiles**. The kernel cache lives in the ephemeral container layer, so every container recreation
restarts the leak from zero. The originally suspected cause (endpoints not closing `UploadFile`)
was **REFUTED** — FastAPI/Starlette closes upload spools automatically (proven live and across
fastapi 0.128–0.139).

**Strategy — contain, not fix (the bug is upstream, no fixed version exists):**
1. `nofile` ulimit 1024 → **65536** in all 6 compose files (cause-agnostic backstop: ~19 days to
   EMFILE becomes ~3.4 years at the observed ~52 FDs/day).
2. Persistent named volumes `miopen-cache:/root/.cache/miopen` + `miopen-config:/root/.config/miopen`
   in the two AMD compose files — each kernel compiles once per **volume** lifetime, not per container.
3. `/health` gains `open_fds` / `fd_soft_limit` fields + a WARNING log at ≥80% of the soft limit
   (the compose healthcheck polls `/health` every 30 s → free periodic probe; the last leak matured
   silently for 19 days).

**Key decisions and rationale (user-approved during brainstorming — do not revisit):**
- **Do NOT change inference geometry** (`rect=False` / square input was rejected: up to 1.78× more
  pixels per inference on the dominant 16:9 traffic, real throughput cost on a weak iGPU).
- **Base image stays `rocm/pytorch:latest`** (user decision; a rebuild pulls ROCm 7.2.4 — accepted).
- **`MIOPEN_FIND_MODE` is NEVER enabled by default** — it stays a *commented-out* line in the two
  AMD compose files + doc entries. Enabling FAST would permanently pin generic fallback kernels
  (inference-perf risk). It is an emergency lever only if the post-deploy leak rate does not drop.
- **The `uploaded_file` close-dependency from the superseded plan must NOT be (re)introduced** —
  its premise is refuted; the framework already closes uploads.
- MIOpen paths/ownership were verified read-only on the live prod container: `HOME=/root`, uid 0,
  binary cache under a version subdir (`3.5.1.dabb6df2b9`), find-db under `~/.config/miopen`.
  These are prod facts — do not "re-verify" them on this machine.

**Edge cases / things to watch:**
- **Task 1 Step 1 (revert of the old test stub) is likely a no-op on a fresh pull** — the stub was
  already discarded before the branch was pushed. Run the step's verification (`git status --short
  tests/test_endpoints.py` → clean) and move on.
- Tests run **inside a virtualenv only**. If `.venv` is missing on this machine, create it per the
  plan's Prerequisites (`python3 -m venv .venv && source .venv/bin/activate && pip install -r
  requirements.txt -r requirements-dev.txt`). Deps are heavy (torch, ultralytics, cv2).
- `ffmpeg` may be absent on this machine — fine: `/health` then reports `video_processing: false`
  but still returns 200 `"healthy"`; the new tests only assert the FD fields and status.
- The warning tests patch `patch("main._fd_stats", return_value=(...))` and capture
  `caplog.at_level(logging.WARNING, logger="main")` — the app module is `main` (tests do
  `from main import app`), so the module logger's name is `main`.
- The YAML validation step needs PyYAML — already present in the venv via the ultralytics
  dependency chain.
- In the deploy AMD compose, apply Task 2 Step 1 (ulimits) before Step 3; the commented
  `MIOPEN_FIND_MODE` lines go INSIDE the `environment:` mapping (YAML comments are legal there).
- Git commits in this repo surface an auto-configured identity warning — informational, ignore it.
- **Deploy (rebuild + redeploy) and all post-merge prod checks are user-side, on the prod host** —
  the "Post-merge" section of the plan is informational; do not attempt to reach any prod container
  from this machine.
- Push the branch after the final task completes (the user merges via PR).

## PLAN QUALITY WARNING

The plan was written in another session and may contain:
- Errors or inaccuracies in implementation details
- Oversights about edge cases or dependencies
- Assumptions that don't match the actual codebase or this machine
- Missing steps or incomplete instructions

**If you notice any issues during implementation:**
1. STOP before proceeding with the problematic step.
2. Clearly describe the problem you found.
3. Explain why the plan doesn't work or seems incorrect.
4. Ask the user how to proceed.

Do NOT silently work around plan issues or make significant deviations without user approval.
