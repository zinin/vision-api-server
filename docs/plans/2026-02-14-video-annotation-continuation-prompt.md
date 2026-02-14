## TASK

Run code review for Video Annotation feature (VAS-2). All implementation tasks are complete except Task 8 (manual integration test).

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

**Remaining:**
- [ ] Task 8: Manual integration test (requires running server with YOLO model + FFmpeg)

## SESSION CONTEXT

Key facts about the implementation:

1. **Branch**: `feature/VAS-2`. Current HEAD: `7ad108b`.
2. **15 tests pass**: 1 config + 3 models + 11 job_manager. Run with `.venv/bin/python -m pytest tests/ -v`.
3. **VideoAnnotator has no unit tests** — requires YOLO model, video files and FFmpeg. Tested only via manual integration (Task 8).
4. **3 design review iterations were done BEFORE implementation** (78 findings total, all processed). The plan already incorporates all accepted fixes.
5. **Endpoint naming**: `/detect/video/visualize` (matches existing `/detect/visualize` for images).
6. **No tracker fallback**: Only CSRT, no KCF. `_create_csrt_tracker()` handles OpenCV version differences.
7. **YOLO.track was investigated and rejected**: It runs detection on EVERY frame (no `detect_every`). CSRT kept for performance.
8. **Python venv**: `.venv/bin/python -m pytest tests/ -v` — system python3 does NOT have pytest.
9. **workers=1 is a hard requirement** — in-memory job state not shared across processes.

### Files changed in this feature (for code review scope)

**New files:**
- `app/job_manager.py` — JobManager, Job dataclass, queue, TTL cleanup
- `app/video_annotator.py` — YOLO + CSRT tracker annotation pipeline
- `tests/test_config.py` — config settings tests
- `tests/test_models.py` — Pydantic response model tests
- `tests/test_job_manager.py` — JobManager tests
- `tests/__init__.py` — empty
- `tests/conftest.py` — sys.path setup

**Modified files:**
- `app/main.py` — +3 endpoints, worker loop, JobManager init in lifespan
- `app/config.py` — +4 env variables
- `app/models.py` — +3 Pydantic response models
- `app/visualization.py` — renamed _draw_detection → draw_detection, _calculate_adaptive_font_scale → calculate_adaptive_font_scale
- `app/dependencies.py` — +get_job_manager
- `requirements.txt` — opencv-contrib, aiofiles, httpx
- `CLAUDE.md` — updated docs

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
