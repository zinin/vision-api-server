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

Read all documents to understand the full picture.

## PROGRESS

**Completed:**
- [x] Design document written
- [x] Implementation plan written (8 tasks)
- [x] Design review iteration 1 completed (3 agents: Codex, Gemini, CCS)
- [x] All 24 review findings processed, decisions made
- [x] Design and plan documents updated with review fixes

**Remaining (all 8 implementation tasks):**
- [ ] Task 1: Update requirements (opencv-contrib) and set up test infrastructure
- [ ] Task 2: Add configuration settings to config.py
- [ ] Task 3: Add Pydantic response models to models.py
- [ ] Task 4: Create JobManager (job_manager.py)
- [ ] Task 5: Create VideoAnnotator (video_annotator.py) + make visualization.py methods public
- [ ] Task 6: Add API endpoints and worker loop to main.py
- [ ] Task 7: Update CLAUDE.md documentation
- [ ] Task 8: Manual integration test

## SESSION CONTEXT

Key decisions from brainstorming and review sessions:

1. **Endpoint naming**: User explicitly renamed from `/detect/video/annotate` to `/detect/video/visualize` to match existing convention (`/detect/visualize` for images).

2. **No tracker fallback needed**: Both CSRT and KCF are always available with `opencv-contrib-python-headless`. Plan uses only CSRT, no fallback logic. The `_create_csrt_tracker()` helper handles OpenCV version differences (`cv2.TrackerCSRT` vs `cv2.legacy.TrackerCSRT`).

3. **cv2.VideoCapture is streaming**: Does NOT load all frames into memory. Decodes one frame at a time (~6 MB for 1080p). Total RAM ~50 MB per job regardless of video length.

4. **cv2.VideoWriter writes to disk**: Frame-by-frame, no memory accumulation. Intermediate file `video_only.mp4` is deleted after FFmpeg audio merge.

5. **No existing tests**: The project has zero tests. Task 1 sets up test infrastructure from scratch. pytest and pytest-asyncio are already in requirements.txt.

6. **opencv-contrib is a drop-in replacement**: Changing `opencv-python-headless` to `opencv-contrib-python-headless` adds the `cv2.legacy` tracking module without breaking anything.

7. **Branch**: Work is on `feature/VAS-2`. Current commits include the design doc and plan.

8. **CLAUDE.md Jira note**: CLAUDE.md says `project_key: "FV"` but the actual project key is `VAS`. Use `VAS` for Jira operations.

9. **YOLO.track investigated and rejected**: YOLO.track runs detection on EVERY frame (no native `detect_every`). CSRT is kept because `detect_every` is a core performance feature — YOLO runs only every Nth frame, CSRT tracks between.

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

### Accepted Risks (No Changes)

- **C1**: mp4v codec kept — file is for download, not browser playback.
- **I5**: ModelManager eviction risk accepted — worker holds reference, GC won't delete. Preloaded models are never evicted.
- **I8**: No graceful shutdown cleanup — startup sweep handles orphans on next start.

### Deferred (Out of MVP Scope)

- Duration/frame limit for abuse protection
- Queue position/ETA in response
- Worker health endpoint
- GPU encoding
- 410 Gone for expired jobs
- Classes validation against model.names

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
