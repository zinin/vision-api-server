## TASK

Execute the implementation plan for YOLO26 migration (VAS-5).

Use `/superpowers:subagent-driven-development` skill for execution.

## DOCUMENTS

- Design: `docs/plans/2026-02-15-yolo26-migration-design.md`
- Plan: `docs/plans/2026-02-15-yolo26-migration-plan.md`

Read both documents first.

## IMPORTANT: DO NOT START WORK YET

After reading the documents:
1. Confirm you have loaded all context
2. Summarize your understanding briefly
3. **WAIT for user instruction before taking any action**

Do NOT begin implementation until the user explicitly tells you to start.

## SESSION CONTEXT

### Key Decisions
- **Approach:** Minimal migration — only update dependency version and replace model name references. No code logic changes.
- **Default model:** `yolo26s.pt` (Small) replaces `yolo11s.pt` everywhere.
- **Backward compatibility:** Not needed. Old models (yolo11, yolov8) are not supported — full cutover to YOLO26.
- **ultralytics version:** `>=8.4.0,<9.0.0` — YOLO26 was introduced in v8.4.0 (Jan 14, 2026). Latest is v8.4.14.

### Rejected Alternatives
- **Approach 2 (+ optimization):** Adding explicit `end2end` config parameter was rejected as YAGNI — ultralytics handles NMS-free transparently.
- **Approach 3 (+ multitask):** Adding segmentation/pose support was rejected — too large scope, separate ticket.

### Why This Works Without Code Changes
- The project uses only generic ultralytics APIs: `YOLO()`, `model.predict()`, `model.to()`, `model.names`, `result.boxes.xyxy/cls/conf`
- All of these are 100% backward compatible in YOLO26
- YOLO26's NMS-free end-to-end inference is handled internally by ultralytics — the `predict()` return format is identical
- Tests are mock-based (mock YOLO results), so they don't depend on real model behavior

### Edge Cases
- `yolo11x.pt` references in Docker compose files must also be updated to `yolo26x.pt` (not just `s` variant)
- The design doc itself (`2026-02-15-yolo26-migration-design.md`) references `yolo11` in context — this is expected and should NOT be changed
- `docker/deploy/.env.example` lists all model sizes (n/s/m/l/x) — update all of them

### Branch
Working branch: `feature/VAS-5`

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
