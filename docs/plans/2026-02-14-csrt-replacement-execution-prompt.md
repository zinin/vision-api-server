## TASK

Execute the implementation plan for replacing CSRT tracker with YOLO + hold mode in VideoAnnotator.

Use `/superpowers:subagent-driven-development` skill for execution.

## DOCUMENTS

- Design: `docs/plans/2026-02-14-csrt-replacement-design.md`
- Plan: `docs/plans/2026-02-14-csrt-replacement-plan.md`

Read both documents first.

## IMPORTANT: DO NOT START WORK YET

After reading the documents:
1. Confirm you have loaded all context
2. Summarize your understanding briefly
3. **WAIT for user instruction before taking any action**

Do NOT begin implementation until the user explicitly tells you to start.

## SESSION CONTEXT

### Key Decisions

1. **`model.predict()` over `model.track()`** — model instance is shared between annotation worker and `/detect` API endpoints. `model.track()` stores state in `model.predictor.trackers` which would cause thread-safety issues. `model.predict()` is stateless.

2. **Hold mode over alternative trackers** — MOSSE (~1ms) and KCF (~3-5ms) were considered but rejected. Hold is simpler (remove code vs replace code) and with detect_every=5 at 30fps, bboxes refresh every ~170ms — acceptable visual quality.

3. **`default_detect_every` stays at 5** — user explicitly chose to keep GPU economy by default rather than changing to 1.

4. **`opencv-contrib-python-headless` stays** in requirements.txt — still needed for cv2.VideoCapture, cv2.VideoWriter, etc. Only the CSRT tracker usage is removed.

### Performance Context

- Current (CSRT): detect_every=5 → ~5.4 fps (CSRT at 220ms/frame dominates)
- After (Hold): detect_every=5 → ~90+ fps (hold is ~0ms, only YOLO every 5th frame)
- After (YOLO every frame): detect_every=1 → ~20-33 fps

### Edge Cases

- `current_detections` starts as empty list. First frame (frame_num=0) always triggers YOLO (0 % N == 0). So hold frames never display stale data from a previous video — they only reuse detections from the current video's detection frames.
- `AnnotationStats.tracked_frames` field semantics change from "CSRT-tracked frames" to "hold frames" — the field name stays the same, no API-breaking change.

### What NOT to Change

- `app/config.py` — NO changes (default_detect_every=5 stays)
- `app/main.py` — NO changes
- `app/visualization.py` — NO changes
- `app/models.py`, `app/job_manager.py` — NO changes
- API contract unchanged — `detect_every` parameter still accepted (range 1-300)

### Branch

- Working branch: `feature/VAS-2`
- 26 commits ahead of `origin/feature/VAS-2`
- All 77 tests currently passing

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
