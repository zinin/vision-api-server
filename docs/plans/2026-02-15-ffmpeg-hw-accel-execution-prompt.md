## TASK

Execute the implementation plan for FFmpeg hardware-accelerated video pipeline (VAS-2).

Use `/superpowers:subagent-driven-development` skill for execution.

## DOCUMENTS

- Design: `docs/plans/2026-02-15-ffmpeg-hw-accel-design.md`
- Plan: `docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md`

Read both documents first.

## IMPORTANT: DO NOT START WORK YET

After reading the documents:
1. Confirm you have loaded all context
2. Summarize your understanding briefly
3. **WAIT for user instruction before taking any action**

Do NOT begin implementation until the user explicitly tells you to start.

## PROGRESS

**Completed:**
- [x] Design review iteration 1 — 3 external reviewers (Codex/Gemini/CCS), 17 findings applied to design+plan

**Remaining (all 9 implementation tasks):**
- [ ] Task 1: `VIDEO_HW_ACCEL` + `VAAPI_DEVICE` config settings
- [ ] Task 2: `app/hw_accel.py` — hardware acceleration detection module
- [ ] Task 3: `app/ffmpeg_pipe.py` — FFmpegDecoder (pipe-based video decoding)
- [ ] Task 4: `app/ffmpeg_pipe.py` — FFmpegEncoder (pipe-based encoding + audio merge)
- [ ] Task 5: Refactor VideoAnnotator to use FFmpeg pipes (largest/riskiest)
- [ ] Task 6: Wire hw_config into lifespan and annotation worker
- [ ] Task 7: Update Docker images (NVIDIA compose + AMD VAAPI packages)
- [ ] Task 8: Update CLAUDE.md documentation
- [ ] Task 9: Full test suite verification

## SESSION CONTEXT

### Key decisions and rationale

- **Full FFmpeg pipe chosen** (Approach 2) over encode-only pipe (Approach 1) and OpenCV CUDA (Approach 3). User explicitly chose maximum performance over simpler implementation.
- **Automatic fallback chain:** NVIDIA (NVDEC/NVENC) → AMD (VAAPI) → CPU. User wants it to "just work" — if NVIDIA is available use it, if AMD use AMD, otherwise CPU.
- **Double encoding is the biggest waste** in current pipeline: cv2.VideoWriter writes mp4v (CPU), then FFmpeg re-encodes to h264/h265 (CPU). The new pipeline eliminates this entirely — one FFmpeg encoder process does it all, including audio merge.

### Rejected alternatives

- **OpenCV CUDA** (`cv2.cudacodec`): Requires custom OpenCV build, fragile across versions, poor AMD support. Too complex for the benefit.
- **PyAV** (Python FFmpeg bindings): Hardware acceleration support is less mature than FFmpeg CLI. Adds dependency.
- **Encode-only pipe** (Approach 1): Simpler but leaves CPU decoding. User wanted full GPU pipeline.

### Technical edge cases

- **Frames go through CPU memory** for bbox drawing — this is unavoidable without CUDA drawing kernels (not worth the complexity).
- **VAAPI encoding from raw pipe** requires special FFmpeg flags: `-vaapi_device /dev/dri/renderD128 -vf 'format=nv12,hwupload'`. NVENC does NOT need this — it accepts CPU frames and handles upload internally.
- **Standard `apt-get install ffmpeg`** on Ubuntu 24.04 MAY have NVENC/VAAPI support compiled in, but the NVIDIA runtime libs (`libnvidia-encode.so`, `libnvidia-decode.so`) are mounted by nvidia-container-toolkit at container startup. The detection module handles this gracefully — if not available, falls back to CPU.
- **AMD av1 encoding** has no VAAPI encoder — falls back to CPU `libsvtav1` for av1 codec on AMD.
- **FFmpegEncoder.close()** must read stderr BEFORE wait() to avoid deadlock on large stderr output. The plan's implementation handles this correctly.
- **The `_merge_audio()` method is deleted** — the encoder pipe merges audio from the original file in a single step using `-i original.mp4 -map 1:a:0?`.

### Review iteration 1 — applied changes

17 findings from 3 reviewers were applied to design/plan. Key changes:

1. **Stderr daemon thread** — both FFmpegDecoder and FFmpegEncoder use `_drain_stderr()` daemon thread to prevent pipe buffer deadlock (found by all 3 reviewers)
2. **Frame `.copy()`** — `np.frombuffer` returns read-only array; `.copy()` added for OpenCV drawing compatibility (Codex)
3. **NVIDIA compose `video` capability** — `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` required for NVENC/NVDEC (Codex)
4. **Codec-specific HW detection** — `detect_hw_accel(codec=...)` checks encoder for configured codec, not just h264 (Codex)
5. **FFmpeg mandatory check** — `shutil.which("ffmpeg")` at startup since cv2 fallback is gone (Codex)
6. **Auto-rotation** — FFmpeg auto-rotates by default; verify with rawvideo pipe (Gemini)
7. **VAAPI device configurable** — `VAAPI_DEVICE` env var, default `/dev/dri/renderD128` (CCS+Codex)
8. **AMD rate control** — `-qp` added for VAAPI encoders (Codex)
9. **Metadata passthrough** — `-map_metadata 0` in encoder command (Gemini+CCS)
10. **Process health checks** — `process.poll()` in `read_frame()` and `write_frame()` (CCS+Gemini)
11. **Subprocess lifecycle** — `terminate/kill` on timeout in `close()` methods (all)

### Explicitly out of scope (Non-Goals)

- Runtime encoder fallback (if GPU encoder fails mid-job → job error, not automatic CPU retry)
- VFR preservation — CFR normalization acceptable
- Graceful subprocess shutdown on job cancellation
- Async frame pipelining (3 threads)
- NVENC preset/RC configuration beyond defaults

### Warnings

- Task 5 (VideoAnnotator refactor) is the riskiest — it's the largest change and touches the core pipeline. The test rewrite must be thorough.
- The `CODEC_MAP` dict in `video_annotator.py` should be removed since encode args now come from `HWAccelConfig`.
- After Task 5, `cv2` import in `video_annotator.py` may become unused (only `numpy` needed). Check and remove if so.
- The `shutil` import in `video_annotator.py` was only used by `_merge_audio()` fallback — remove it too.

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
