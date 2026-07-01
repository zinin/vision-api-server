## TASK

**Brainstorm the FIX** for the production file-descriptor leak in `vision-api-server`
(project `/home/zinin/git/vision-api-server`, branch `fix/upload-fd-leak`).

The **root-cause diagnosis is COMPLETE** (root cause identified with high confidence — see below).
This session's job is to run a **`superpowers:brainstorming`** session to design the fix. This is NOT
execution of the old plan — that plan's premise is **refuted**.

Converse in **Russian** (user preference).

## CRITICAL: START WITH BRAINSTORMING — DO NOT IMPLEMENT

After loading all context below, you MUST:
1. Read `.superpowers/sdd/diagnosis-findings.md` **FIRST** (full diagnosis, ROOT CAUSE, confirmation
   status, fix directions, upgrade assessment). Then skim the refuted design/plan for background.
2. Report a brief summary of what you understood.
3. Invoke **`superpowers:brainstorming`** and work through the fix design WITH the user — ask questions
   one at a time, explore trade-offs. Converge before proposing a design/plan.

**DO NOT:**
- Start implementing or changing code.
- Write an implementation plan (`superpowers:writing-plans`) before brainstorming converges and the
  user approves the design.
- **Send ANY traffic (API requests) to the prod container `deploy-vision-api-1`** — it is live prod on
  THIS host. Read-only `docker inspect` / `docker exec` (ls/cat/env, no traffic) was authorized last
  session; **sending API traffic still needs explicit user OK.**
- Re-investigate anything in the "Refuted hypotheses" list.

## PRIMARY ARTIFACT — read first

`.superpowers/sdd/diagnosis-findings.md` — full diagnosis state. Contains: "ROOT CAUSE — high confidence"
section, established facts, refuted hypotheses (do NOT re-investigate), honest confirmation status, fix
directions, and the upgrade assessment. (File is untracked/on-disk — read it directly.)

## BACKGROUND DOCS (premise REFUTED — do not execute)

- Design: `docs/superpowers/specs/2026-07-01-fd-leak-upload-close-design.md`
- Plan: `docs/superpowers/plans/2026-07-01-fd-leak-upload-close.md`

Both assume an unclosed `UploadFile` causes the leak — **REFUTED**. Only their **Task 2 (raise `nofile`
ulimit)** survives as valid defense-in-depth.

## ROOT CAUSE (high confidence)

The `/tmp/#<inode> (deleted)` **O_TMPFILE** leak is the **ROCm/MIOpen `hipModuleLoad`/`hipModuleUnload`
file-descriptor leak** (upstream MIOpen #2223 / ROCm #2289), triggered when MIOpen compiles/loads a **new
GPU kernel** — **NOT** the `UploadFile` spool and **NOT** app code.

Evidence:
- **App code is clean:** `app/` has zero `os.dup`/`.fileno()`/`.file` spool access/subprocess-`stdin=<fd>`
  and no `tempfile.TemporaryFile`/`SpooledTemporaryFile`/`mkstemp`. `video_utils.py:430` closes its
  `NamedTemporaryFile` via `with` before `os.unlink`.
- **Framework closes upload spools — proven live:** under heavy real upload traffic the `/proc/1/fd`
  `(deleted)` count oscillates 0↔2 and **returns to 0** (transient in-flight spools, not a leak).
- **Rate ~52/day ≪ upload rate (thousands/day)** ⇒ the permanent leak is a RARE event, not per-request →
  matches occasional new-kernel compiles, not spool close-failures.
- **Prod stack (read-only inspected):** ROCm **7.2.2**, torch **2.10.0+rocm7.2.2**, MIOpen **3.5.1**,
  **gfx1030** via `HSA_OVERRIDE_GFX_VERSION=10.3.0` (an **unsupported** consumer-iGPU config), `/tmp` =
  overlay rootfs, **no `MIOPEN_*`/`TMPDIR` env**, kernel cache `~/.cache/miopen` in the ephemeral
  container layer. `libMIOpen`/`libamd_comgr`/`libhiprtc` loaded and active in PID 1.
- **Upstream match:** MIOpen compiles kernels to `/tmp/miopen-*` / `/tmp/comgr-*` on kernel-DB miss and
  "forgets to close" fds → "Too many open files". Documented known issue ROCm 5.5/5.6 → 6.0, **no
  confirmed fix version**, still present in 7.2.2.
- **Historical failure:** `OSError: [Errno 24] Too many open files` storm 2026-07-01 15:42–43, cleared
  by the 15:44 restart; historically 997 `(deleted)` fds → EMFILE → `/health` 500.

**Confirmation status (honest):** a read-only ~7-min live fd watch caught only TRANSIENT churn (count → 0);
the slow permanent leak (~52/day ≈ 1 per 27 min) was **not directly isolated**. Clean proofs still open if
100% certainty is wanted: (a) multi-hour read-only baseline-drift watch, or (b) controlled **`<1 MB`-image
`/detect` test** — a sub-1 MB upload cannot spool, so any permanent O_TMPFILE afterwards = MIOpen (needs
traffic OK). For *fixing*, this proof is not strictly required.

## FIX DIRECTIONS TO BRAINSTORM

1. **`nofile` ulimit `1024 → 65536`** (6 compose files; old-plan Task 2). **Biggest lever** —
   ~19-days-to-EMFILE becomes ~3+ years of headroom at the same leak rate. Cheap, safe,
   version-independent. Strong candidate for "do this regardless".
2. **Fewer MIOpen compiles:** `MIOPEN_FIND_MODE=FAST` (avoid exhaustive tuning — the worst leaker);
   **persistent volume** for `~/.cache/miopen` (+ `~/.config/miopen`) so kernels survive restarts.
3. **Fixed inference shape (VERIFY FIRST, free, no prod):** if ultralytics `predict` uses
   rectangular/aspect-preserving letterbox, each new image aspect-ratio → new conv config → new compile
   → one leaked fd (the likely ~52/day driver). Forcing square `imgsz×imgsz` would collapse configs → leak
   ~stops. **Verify ultralytics behavior in `.venv` before relying on this.**
4. **Pinned ROCm upgrade (weak/optional):** `rocm/pytorch:latest` now = ROCm **7.2.4** (minor bump from
   7.2.2), fix **unconfirmed**, unsupported HW. Only as a **tested** experiment (throwaway repro container),
   **pinned** not `:latest`. Not the primary fix.
5. **Optional stopgap:** periodic process recycle (cron restart) before the ceiling.

**Open questions for brainstorming:** which combination to ship; whether to run a confirmatory test
(the `<1 MB` test) before committing; keep or drop the refuted `uploaded_file` explicit-close change;
verify the ultralytics shape mechanism first; how to validate the fix; deploy/rebuild is user-side.

## ENVIRONMENT / CONSTRAINTS

- Prod `deploy-vision-api-1` runs on THIS host (docker 29.5.3), **heavy live traffic**, `LOG_LEVEL=DEBUG`,
  `YOLO_MODELS={"yolo26s.pt":"cuda:0","yolo26x.pt":"cuda:0"}`, only `/models` volume mounted.
- Read-only `docker inspect`/`exec` authorized; **sending API traffic is NOT** (needs explicit OK).
- Branch `fix/upload-fd-leak`, base `cd17f3f`. Tests run inside a venv only (`.venv/`, see plan Prerequisites).
- **Working tree:** `tests/test_endpoints.py` has an **uncommitted** blocked Task-1 stub (an FD-leak test
  that cannot reproduce locally, + `gc/os/sys` imports) — decide **revert vs repurpose**.
  `.superpowers/sdd/diagnosis-findings.md` was updated this session (the diagnosis record).
- **Deploy (rebuild image + redeploy) is user-side.**

## INSTRUCTIONS

1. Read `.superpowers/sdd/diagnosis-findings.md` (primary), skim the refuted design/plan for background.
2. Provide a brief summary of what you understood (root cause, fix directions, confirmation status).
3. Invoke **`superpowers:brainstorming`** and drive the fix design WITH the user (Russian).
4. Do **NOT** implement or write a plan until brainstorming converges and the user approves.
5. Open with: «Диагностика завершена. Давай побрейнштормим фикс — с чего начнём: подтвердить механизм
   формы в ultralytics, обсудить состав фикса (ulimit + MIOpen-настройки), или что-то ещё?»
