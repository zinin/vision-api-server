## TASK

Continue executing the implementation plan for the **ROCm/MIOpen FD-leak containment** in the
`vision-api-server` project.

Use `/superpowers:subagent-driven-development` (recommended) or `/superpowers:executing-plans`
for execution. Work happens on the existing branch **`fix/upload-fd-leak`** (deliberate decision:
the branch is NOT renamed; the PR title will carry the honest name).

Converse in **Russian** (user preference).

## CRITICAL: DO NOT START WORKING

**STOP. READ THIS CAREFULLY.**

After loading all context below, you MUST:
1. Read the documents and understand the context
2. Report what you understood (brief summary)
3. **WAIT for explicit user instructions** before taking ANY action

**DO NOT:**
- Start implementing tasks
- Make any code changes
- Run any commands (except reading documents)
- Assume what task to work on next

**The user will tell you exactly what to do.** Until then, only read and summarize.

## DOCUMENTS

- Design: `docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md`
- Plan: `docs/superpowers/plans/2026-07-01-fd-leak-miopen-containment.md`

Both were UPDATED by design-review iteration 1 and committed (`ea18a33` auto-fixes, `446fad6`
decisions + log). Read both. Review artifacts (reference only, do not re-litigate):
`docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-review-iter-1.md` (36 issues with
decisions), `...-review-merged-iter-1.md` (raw reviewer output).

## PROGRESS

**Completed:**
- [x] Design + plan written, committed, pushed (earlier sessions)
- [x] Design review iteration 1 (`mesh-design-review`, 4 external reviewers: codex gpt-5.5,
      claude-self, qwen, minimax; 3 more timed out): 36 deduplicated issues → 20 auto-fixes,
      4 user decisions, 3 auto-resolutions, 9 dismissals. Documents updated accordingly.

**Remaining (implementation has NOT started — no code was touched yet):**
- [ ] Task 1: `/health` FD observability — TDD, 7 tests, guarded `import resource`,
      `_fd_stats()` triple `(open_fds, fd_deleted, soft_limit)`, hourly-rate-limited WARNING
      collected BEFORE the ffmpeg probe, probe catches `(RuntimeError, OSError)`
- [ ] Task 2: `ulimits.nofile` 65536 in all 6 compose files + MIOpen cache volumes in the 2 AMD
      files + permanent `tests/test_compose.py`
- [ ] Task 3: documentation (`.claude/rules/api.md`, `CLAUDE.md`, `docker/deploy/.env.example`)
- [ ] Push the branch after the final task (user merges via PR)

## SESSION CONTEXT

Facts verified this session (trust them, do not re-derive):
- `.venv` in the repo root is BROKEN: `.venv/bin/python` symlinks to a missing
  `/usr/bin/python3.13` (system Python is 3.12.3). The plan's Prerequisites now include a health
  check; expect to rebuild the venv (deps are heavy: torch, ultralytics, cv2 — takes minutes).
- `tests/test_endpoints.py` is CLEAN — Task 1 Step 1 is verify-only, expect a no-op.
- `app/main.py`: `import time` already exists (line 3); `VideoFrameExtractor` is already imported
  into `main`'s namespace (line 44) — the plan's `patch("main.VideoFrameExtractor", ...)` and
  `time.monotonic()` usage are valid as written.
- No `tmpfs` anywhere in `docker/` — container `/tmp` is overlayfs writable layer (the disk-
  pressure wording in the spec §1 is correct).

Review decisions locked with the user (do NOT revisit, do NOT "improve"):
- CRITICAL-1 → variant B: NO pre-deploy warm-cache experiment. The "cache hit does not leak"
  assumption is documented as open in spec §2; rollout monitoring is two-axis
  (`fd_deleted` vs `open_fds`) and answers it in prod.
- `fd_deleted` field in `/health` ACCEPTED — `_fd_stats()` returns a triple; all 7 tests in the
  plan already reflect this.
- `imgsz` normalization/allowlist NOT needed (clients use a fixed imgsz; worst case is documented
  in spec §1 only).
- Single WARNING threshold 80% (no INFO/ERROR gradation), rate-limited to once per hour.
- No `scripts/verify_fd_containment.sh` — post-merge checks stay as copy-paste one-liners in the
  plan.
- Rejected (do not reintroduce): `fd_usage_pct` derived field, caching the ffmpeg probe, prewarm
  container, subtracting 1 from listdir count, mocking stdlib instead of `_fd_stats`.
- The `uploaded_file` close-fix from the superseded plan remains REFUTED — never reintroduce.
- `MIOPEN_FIND_MODE` stays a commented-out compose line (never enabled by default, deliberately
  NOT a `${...:-}` pass-through from `.env`).

False-positive inoculations (already reflected in the spec; if a reviewer/subagent raises these,
they are wrong): `rect=True` IS the effective default — it is a method default of
`Model.predict()` (verified in ultralytics 8.4.14), `cfg/default.yaml`'s `rect: False` applies to
train/val; `caplog.at_level(..., logger="main")` DOES capture the warning regardless of
`LOG_LEVEL` (propagation is not filtered by root logger level).

Operational rules:
- Tests run ONLY inside `.venv`; do not touch dependency pins.
- Do NOT touch the live prod container; deploy and post-merge checks are user-side.
- Conventional Commits; git identity warning in this repo is informational — ignore.
- Before opening the PR (final step, see the plan's "Before opening the PR" section):
  `git rm -r docs/superpowers/` + commit + push — plan/spec/review docs must not appear in the
  PR diff.
- A future review iteration (if the user asks for one) must feed
  `...-review-iter-1.md` as PREVIOUS REVIEW DECISIONS to block repeat findings.

## PLAN QUALITY WARNING

The plan was written and then heavily amended by review in other sessions and may still contain:
- Errors or inaccuracies in implementation details
- Oversights about edge cases or dependencies
- Assumptions that don't match the actual codebase or this machine
- Missing steps or incomplete instructions

**If you notice any issues during implementation:**
1. STOP before proceeding with the problematic step
2. Clearly describe the problem you found
3. Explain why the plan doesn't work or seems incorrect
4. Ask the user how to proceed

Do NOT silently work around plan issues or make significant deviations without user approval.

## INSTRUCTIONS

1. Read the documents listed above
2. Understand current progress and session context
3. Provide a brief summary of what you understood
4. **STOP and WAIT** — do NOT proceed with any implementation
5. Ask: "С чего начинаем?"
