## TASK

Continue executing the implementation plan for Video Annotation feature (VAS-2). All implementation, two rounds of code review, and documentation are done — only manual integration test remains.

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

- Design: `docs/plans/2026-02-14-video-annotation-design.md`
- Plan: `docs/plans/2026-02-14-video-annotation.md`

Read both documents to understand the full picture.

## PROGRESS

**Completed:**
- [x] Task 1: Update requirements (opencv-contrib, aiofiles, httpx) and set up test infrastructure — `af74edb`
- [x] Task 2: Add configuration settings to config.py — `eb3958a`
- [x] Task 3: Add Pydantic response models to models.py — `529388e`
- [x] Task 4: Create JobManager (job_manager.py) — `b3cb025` + test fix `0e048f9`
- [x] Task 5: Create VideoAnnotator (video_annotator.py) + make visualization.py methods public — `a9ac08f`
- [x] Task 6: Add API endpoints and worker loop to main.py + get_job_manager dependency — `a0fea76`
- [x] Task 7: Update CLAUDE.md documentation — `7ad108b`
- [x] Code review round 1 (4 agents: Claude, Codex, CCS/GLM-4.7, Gemini) — 11 issues fixed — `4763068`
- [x] Code review round 2 (4 agents: same) — 3 issues fixed — `7869f78`
- [x] Added Testing section to CLAUDE.md — `0856b67`

**Remaining:**
- [ ] Task 8: Manual integration test (requires running server with YOLO model + FFmpeg)

## SESSION CONTEXT

Key facts about the implementation:

1. **Branch**: `feature/VAS-2`. Current HEAD: `0856b67`.
2. **15 tests pass**: 1 config + 3 models + 11 job_manager. Run with `pip install -r requirements-dev.txt && python -m pytest tests/ -v`.
3. **VideoAnnotator has no unit tests** — requires YOLO model, video files and FFmpeg. Tested only via manual integration (Task 8).
4. **3 design review iterations were done BEFORE implementation** (78 findings total, all processed). The plan already incorporates all accepted fixes.
5. **Code review round 1 (4 agents).** Results:
   - 19 unique findings total
   - 11 marked "Справедливо" — all fixed in `4763068`
   - 3 marked "Спорно" (job orphan on shutil.move failure, thread-safety comment, multi-worker warning) — left as-is for v1
   - 5 marked "Ложное срабатывание"
6. **Code review round 2 (4 agents).** Results:
   - 19 unique findings total
   - 3 marked "Справедливо" — all fixed in `7869f78`:
     - `max_queued_jobs` now has `Field(default=10, ge=1)` validation
     - Model validated at submission time before upload (immediate 400 instead of delayed failure)
     - pytest/pytest-asyncio moved to `requirements-dev.txt`
   - 6 marked "Спорно" — left as-is for v1:
     - Queue not atomic with upload (TOCTOU, low impact for internal API)
     - Thread-safety progress_callback (repeat from round 1, CPython GIL safe)
     - Failed trackers kept active (minor CPU waste, max 4 frames)
     - create_job before shutil.move (repeat from round 1, worker handles failure)
     - No tests for VideoAnnotator (known, by design)
     - Duplicate ffmpeg checks in lifespan (cosmetic)
   - 10 marked "Ложное срабатывание"
7. **Key fixes across both review rounds** (`4763068` + `7869f78`):
   - `startup_sweep()` safety guard against dangerous paths (`/tmp`, `/var`, etc.)
   - `video_only.mp4` unlink wrapped in try/except OSError
   - `video_job_ttl` minimum 60s, `default_detect_every` upper bound 300, `max_queued_jobs` minimum 1
   - Version synced to 2.2.0
   - `font_scale` calculated once per video instead of per detection
   - Imports moved to top-level, unused import removed, redundant except simplified
   - Progress skipped when `total_frames` unknown
   - Model validated at submission time in `annotate_video` endpoint
   - Dev dependencies separated to `requirements-dev.txt`
8. **Endpoint naming**: `/detect/video/visualize` (matches existing `/detect/visualize` for images).
9. **No tracker fallback**: Only CSRT, no KCF. `_create_csrt_tracker()` handles OpenCV version differences.
10. **YOLO.track was investigated and rejected**: It runs detection on EVERY frame (no `detect_every`). CSRT kept for performance.
11. **Python venv**: `.venv/bin/python -m pytest tests/ -v` — system python3 does NOT have pytest.
12. **workers=1 is a hard requirement** — in-memory job state not shared across processes.

### Files changed in this feature (for reference)

**New files:**
- `app/job_manager.py` — JobManager, Job dataclass, queue, TTL cleanup
- `app/video_annotator.py` — YOLO + CSRT tracker annotation pipeline
- `tests/test_config.py` — config settings tests
- `tests/test_models.py` — Pydantic response model tests
- `tests/test_job_manager.py` — JobManager tests
- `tests/__init__.py` — empty
- `tests/conftest.py` — sys.path setup
- `requirements-dev.txt` — dev dependencies (includes -r requirements.txt)

**Modified files:**
- `app/main.py` — +3 endpoints, worker loop, JobManager init in lifespan, model validation at submission
- `app/config.py` — +4 env variables with Field validation
- `app/models.py` — +3 Pydantic response models
- `app/visualization.py` — renamed _draw_detection → draw_detection, _calculate_adaptive_font_scale → calculate_adaptive_font_scale
- `app/dependencies.py` — +get_job_manager
- `requirements.txt` — opencv-contrib, aiofiles, httpx (pytest moved to dev)
- `CLAUDE.md` — updated docs + testing section

## PLAN QUALITY WARNING

The plan was written for a large task and may contain:
- Errors or inaccuracies in implementation details
- Oversights about edge cases or dependencies
- Assumptions that don't match the actual codebase
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
5. Ask: "What would you like me to work on?"

## SPECIAL INSTRUCTIONS

### Code Review Policy

**DO NOT launch code review agents automatically.** Only run code review when the user explicitly requests it.

When the user asks for code review:

**Step 1: Ask which reviewers to use:**

Use AskUserQuestion with multiSelect: true and header: "Reviewers":
- Question: "Какие code review агенты запустить?"
- Options (all checked by default):
  - **superpowers:code-reviewer** — основной ревью (Claude)
  - **codex-code-reviewer** — Codex CLI ревью
  - **ccs-code-reviewer** — CCS ревью (PROFILE=glmt)
  - **gemini-code-reviewer** — Gemini CLI ревью

User can deselect agents they don't want to run.

**Step 2: Ask how to run the reviews:**

Use AskUserQuestion with these options:
- **Background tasks (Recommended)** — независимые агенты в фоне, можно продолжать работу пока они выполняются
- **Team of reviewers** — создать команду code reviewers через TeamCreate

**Step 3a: If "Background tasks" selected:**

Launch **only selected** agents **in parallel** in a single message, ALL with `run_in_background: true`.

After launching, display:
```
N code review агентов запущены параллельно в фоне:
  [list only selected agents with descriptions]

Ожидаю результаты. Вы можете продолжать работу — я сообщу, когда ревью завершатся.
Если хотите отменить ожидание какого-то агента, скажите об этом.
```

**Do NOT block user input.** Continue accepting user instructions while agents work.
When each agent completes, read its output_file.
After all agents finish (or user cancels some), proceed to **Step 4: Process Results**.

**Step 3b: If "Team of reviewers" selected:**

1. Create a team via TeamCreate with name `code-review`
2. Create tasks via TaskCreate (one per selected reviewer)
3. Spawn teammates via Task tool with `team_name: "code-review"` — only selected agents
4. Assign tasks to teammates
5. Wait for all to complete, then proceed to **Step 4: Process Results**
6. Shut down the team when done

**Step 4: Process Results**

After collecting results from all reviewers:

1. **Deduplicate:** If multiple agents found the same issue (same file, same problem), merge into one entry. Note all agents that found it.

2. **Analyze each issue:** For every finding, check against the actual codebase:
   - Is the issue real? (read the code, verify the claim)
   - Is the severity level correct? (Critical/Important/Minor)
   - Could this be a false positive or misunderstanding of the codebase?

3. **Present a summary table:**

| Суть проблемы | Уровень | Кто нашёл | Вердикт |
|---|---|---|---|
| Описание проблемы + `file:line` | Critical / Important / Minor | [перечислить нашедших] | Справедливо / Ложное срабатывание / Спорно (пояснение) |

4. **For each "Спорно" verdict**, briefly explain why you are unsure.

5. **Offer to fix only issues marked "Справедливо":**
   ```
   Справедливых замечаний: N. Хотите, чтобы я исправил их?
   ```
   Wait for user confirmation before making any changes.
