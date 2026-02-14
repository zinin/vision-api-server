## TASK

Continue executing the implementation plan for Video Annotation (VAS-2).

Use `/superpowers:subagent-driven-development` skill for execution.

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
- Review iteration 1: `docs/plans/2026-02-14-video-annotation-review-iter-1.md`
- Review iteration 2: `docs/plans/2026-02-14-video-annotation-review-iter-2.md`
- Review iteration 3: `docs/plans/2026-02-14-video-annotation-review-iter-3.md`

Read all documents to understand the full picture.

## PROGRESS

**Completed:**
- [x] Design document written
- [x] Implementation plan written (8 tasks)
- [x] Design review iteration 1 completed (3 agents: Codex, Gemini, CCS)
- [x] All 24 review findings processed, decisions made
- [x] Design and plan documents updated with review fixes (iter 1)
- [x] Design review iteration 2 completed (3 agents: Codex, Gemini, CCS)
- [x] All 27 review findings processed (19 new, 3 repeats, 5 false positives)
- [x] Design and plan documents updated with review fixes (iter 2)
- [x] Design review iteration 3 completed (3 agents: Codex, Gemini, CCS)
- [x] All 27 review findings processed (11 new, 8 repeats, 7 false positives)
- [x] Design and plan documents updated with review fixes (iter 3)
- [x] Task 1: Update requirements (opencv-contrib, aiofiles, httpx) and set up test infrastructure — `af74edb`
- [x] Task 2: Add configuration settings to config.py — `eb3958a`
- [x] Task 3: Add Pydantic response models to models.py — `529388e`
- [x] Task 4: Create JobManager (job_manager.py) — `b3cb025` + test fix `0e048f9`
- [x] Task 5: Create VideoAnnotator (video_annotator.py) + make visualization.py methods public — `a9ac08f`
- [x] Task 6: Add API endpoints and worker loop to main.py + get_job_manager dependency — `a0fea76`

**Remaining (2 implementation tasks):**
- [ ] Task 7: Update CLAUDE.md documentation
- [ ] Task 8: Manual integration test

## SESSION CONTEXT

Key decisions from brainstorming and review sessions:

1. **Endpoint naming**: User explicitly renamed from `/detect/video/annotate` to `/detect/video/visualize` to match existing convention (`/detect/visualize` for images).

2. **No tracker fallback needed**: Both CSRT and KCF are always available with `opencv-contrib-python-headless`. Plan uses only CSRT, no fallback logic. The `_create_csrt_tracker()` helper handles OpenCV version differences (`cv2.TrackerCSRT` vs `cv2.legacy.TrackerCSRT`).

3. **cv2.VideoCapture is streaming**: Does NOT load all frames into memory. Decodes one frame at a time (~6 MB for 1080p). Total RAM ~50 MB per job regardless of video length.

4. **cv2.VideoWriter writes to disk**: Frame-by-frame, no memory accumulation. Intermediate file `video_only.mp4` is deleted after FFmpeg audio merge.

5. **No existing tests**: The project had zero tests before this work. Task 1 set up test infrastructure from scratch. pytest and pytest-asyncio are already in requirements.txt.

6. **opencv-contrib is a drop-in replacement**: Changing `opencv-python-headless` to `opencv-contrib-python-headless` adds the `cv2.legacy` tracking module without breaking anything.

7. **Branch**: Work is on `feature/VAS-2`. Current HEAD: `a0fea76`.

8. **CLAUDE.md Jira note**: CLAUDE.md says `project_key: "FV"` but the actual project key is `VAS`. Use `VAS` for Jira operations.

9. **YOLO.track investigated and rejected**: YOLO.track runs detection on EVERY frame (no native `detect_every`). CSRT is kept because `detect_every` is a core performance feature — YOLO runs only every Nth frame, CSRT tracks between.

10. **Python venv**: pytest и зависимости установлены в `.venv`. Для запуска тестов использовать: `.venv/bin/python -m pytest tests/ -v`. Системный `python3` / `python3.13` НЕ имеет pytest.

11. **Текущие тесты**: 15 тестов проходят (1 config + 3 models + 11 job_manager).

12. **Code quality review на Task 4 нашёл**: naive datetime в тесте test_cleanup_expired — исправлено в отдельном коммите `0e048f9` (datetime.now() → datetime.now(tz=timezone.utc)).

13. **Task 5 без тестов**: VideoAnnotator не имеет unit-тестов — требует YOLO модель, видеофайлы и FFmpeg. Тестируется интеграционно в Task 8.

14. **Task 6 реализован субагентом**: Все 6 шагов выполнены точно по плану. Модуль загружается, 14 роутов зарегистрированы (включая 3 новых: `/detect/video/visualize`, `/jobs/{job_id}`, `/jobs/{job_id}/download`). 15 тестов проходят без регрессий.

### Review Fixes Applied to Plan (Iteration 1)

These changes are already in the plan documents. Implementers should follow the updated plan:

- **C2 (Enum bug)**: `job.status == "completed"` → `job.status == JobStatus.COMPLETED`. `JobStatus` imported in main.py.
- **C3+I1 (Streaming upload)**: Submit path rewritten — streaming upload in 1MB chunks to temp file, validate size on the fly, create_job only after successful write. No more `await file.read()` for 500MB files.
- **C4 (worker_task scope)**: `worker_task` stored in `app.state.worker_task`, not local variable. Cleanup uses `hasattr(app.state, "worker_task")`.
- **C5 (FFmpeg fallback)**: `FileNotFoundError` added to except clause in `_merge_audio`.
- **I2 (MAX_CONCURRENT_JOBS)**: Removed from config, tests, and design. Always single worker.
- **I3 (Startup sweep)**: `startup_sweep()` method added to JobManager — deletes all dirs in VIDEO_JOBS_DIR on startup. Test added.
- **I6 (Public methods)**: `_draw_detection` → `draw_detection`, `_calculate_adaptive_font_scale` → `calculate_adaptive_font_scale` in visualization.py (Task 5 Step 0).
- **I7 (detect_every default)**: Endpoint default changed to `None`, resolved from `settings.default_detect_every` in body.
- **S3 (FileResponse)**: Download endpoint uses `FileResponse` instead of `StreamingResponse(open(...))`. Need `FileResponse` import.
- **ffprobe**: VideoAnnotator uses `_get_video_metadata()` with ffprobe primary, cv2 fallback for VFR video reliability.
- **I4 (Deployment)**: `workers=1` documented as hard requirement.
- **Q1-Q2**: Jobs don't survive restart (by design). Audio is best effort (no audio = not an error).
- **S1 (Tests TODO)**: VideoAnnotator unit tests deferred, TODO noted in Task 5.

### Review Fixes Applied to Plan (Iteration 2)

These changes are ALSO already in the plan documents:

- **C-NEW-1 (writer try/finally)**: `writer` initialized as `None`, release called in `finally` block. After successful release in happy path `writer = None` to avoid double release.
- **C-NEW-2 (input cleanup isolation)**: `unlink(input.mp4)` moved out of the main try block in worker. Separate try/except with `logger.warning`. Failed unlink doesn't overwrite job status to failed.
- **C-NEW-3 (tmp file cleanup)**: Endpoint upload wrapped in try/except/finally for tmp file cleanup. `startup_sweep` extended to also delete `.tmp` files. Test updated (`assert removed == 3`).
- **I-NEW-2 (queue pre-check)**: New `check_queue_capacity()` method in JobManager. Called in endpoint BEFORE upload starts. `create_job()` also calls it (double-check after upload).
- **I-NEW-3 (ffprobe validation)**: After parsing ffprobe output, validates `width>0, height>0, fps>0`. Falls back to cv2 if invalid.
- **I-NEW-4 (aiofiles)**: Sync `f.write()` replaced with `aiofiles.open()` + `await f.write()`. `aiofiles` added to requirements.txt in Task 1.
- **I-NEW-6 (UTC timezone)**: `datetime.now()` → `datetime.now(tz=timezone.utc)` everywhere. `from datetime import datetime, timezone` import.
- **I-NEW-7 (cv2 fallback try/finally)**: cv2 fallback in `_get_video_metadata()` wrapped in try/finally with `cap.release()`.
- **I-NEW-8 (tracker bbox validation)**: Added `if w <= 0 or h <= 0: continue` in `_update_trackers()`.
- **I-NEW-9 (service executor)**: `run_in_executor(None, ...)` → `run_in_executor(executor, ...)` where `executor = get_executor(settings.max_executor_workers).executor`.
- **I-NEW-10 (mark_processing warning)**: Added `logger.warning` if job status != QUEUED when mark_processing is called.
- **S-LOGGING**: Added `logger.info` at start of `annotate()` with key parameters.
- **S-FFMPEG-CHECK**: Added `shutil.which("ffmpeg")` / `shutil.which("ffprobe")` check in lifespan with `logger.warning`.
- **S-SLOTS**: `@dataclass(slots=True)` for Job, AnnotationParams, AnnotationStats.
- **REPEAT-I2**: Removed `MAX_CONCURRENT_JOBS` from Task 7 docs configuration.
- **S-TRACKER-COLOR**: Confirmed solved — `(tracker, det)` tuple preserves class_id, so `draw_detection` uses correct color by class.

### Review Fixes Applied to Plan (Iteration 3)

These changes are ALSO already in the plan documents:

- **C-3-1 (executor bug)**: `model_manager._executor` → `get_executor(settings.max_executor_workers).executor`. Added `from inference_utils import get_executor` to imports. ModelManager has no `_executor` attribute; the executor lives in `inference_utils.py` as `InferenceExecutor` singleton.
- **C-3-2 (per-job finally)**: Worker restructured with inner try/finally. `finally` always cleans up `input_path`, even on model load failure. Fixes input file leak when `continue` after model error.
- **I-3-1 (move inside try)**: `shutil.move()` moved inside the try block, right after `create_job()`. On failure, tmp file is cleaned in except handlers.
- **I-3-2 (config validation)**: `default_detect_every: int = Field(default=5, ge=1)` with `from pydantic import field_validator, Field` import. Prevents ZeroDivisionError from `frame_num % 0`.

### Accepted Risks (No Changes)

- **C1**: mp4v codec kept — file is for download, not browser playback.
- **I5**: ModelManager eviction risk accepted — worker holds reference, GC won't delete. Preloaded models are never evicted.
- **I8**: No graceful shutdown cleanup — startup sweep handles orphans on next start.
- **startup_sweep safety**: VIDEO_JOBS_DIR=/tmp/vision_jobs is specific enough, risk minimal.

### Deferred (Out of MVP Scope)

- Duration/frame limit for abuse protection
- Queue position/ETA in response
- Worker health endpoint
- GPU encoding
- 410 Gone for expired jobs
- Classes validation against model.names
- Smoke test for CV2/FFmpeg integration
- Cancel endpoint (DELETE /jobs/{job_id})
- Disk space pre-check
- detect_every > total_frames validation
- NamedTemporaryFile for upload
- Monitoring metrics
- In-flight uploads counting toward queue limit
- VFR/rotated video metadata cross-check
- Content-based input validation (beyond extension)
- FFmpeg stderr last 500 chars instead of first 500
- Model pre-validation before upload

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
