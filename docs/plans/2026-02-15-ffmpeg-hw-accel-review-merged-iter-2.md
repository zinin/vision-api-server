# Merged Design Review — Iteration 2

## codex-executor (gpt-5.3-codex)

### Critical Issues

1. **HW-detection is static-only, no runtime probe.** The `detect_hw_accel()` function relies on `ffmpeg -hwaccels` and `ffmpeg -encoders` output, but never actually tests whether the backend works. This produces false-positive GPU selections in containers or hosts where the device is listed but non-functional, breaking the "just works" contract. References: design line 46, plan lines 300, 325.

2. **CPU fallback does not validate encoder availability.** When falling back to CPU, the code never checks whether `libx264`/`libx265`/`libsvtav1` is actually compiled into the FFmpeg binary. Result: the startup detection "succeeds" with CPU, but encoding fails at job runtime. References: plan lines 243, 268, 349.

3. **`ffprobe` not checked at startup (only `ffmpeg` is).** The pipeline depends on `ffprobe` for metadata extraction, but the fail-fast startup check only verifies `ffmpeg`. This creates a "startup OK, job fails later" scenario. References: plan line 1042, 857; `app/video_annotator.py:192`.

4. **Broken test snippets will block TDD process:**
   - Missing `pytest`/`ValidationError` imports in Task 1 tests (plan line 37).
   - `BytesIO.close.assert_called()` is impossible on a real `BytesIO` -- needs to be a Mock (plan lines 394, 456).
   - Mock encoder lacks `poll.return_value=None`, so `write_frame()` will treat the process as "crashed" (plan lines 590, 754).

### Concerns

1. **No explicit output `pix_fmt` on encoder.** Without `-pix_fmt yuv420p`, FFmpeg may default to `yuv444p` for H.264/H.265, causing playback failures on many clients. References: plan lines 730, 739.

2. **Task 6 worker test is too vague.** It says "add comment/verify" without a concrete assertion on `hw_config` being passed to `VideoAnnotator`. High risk of silent regression. Reference: plan line 1030.

3. **cv2 metadata fallback may be lost.** The plan suggests removing `cv2` dependency "if no remaining references", but `cv2` currently serves as a fallback for metadata extraction when `ffprobe` fails. Removing it could degrade resilience. References: plan line 1007; `app/video_annotator.py:232`.

4. **"Cached detect" API surface is unclear.** The design mentions caching the HW detection result, but in the plan it is only implicitly stored in `app.state` with no documented interface for reuse elsewhere. Reference: design line 46, plan line 310.

### Suggestions

1. Add an **active runtime probe at startup** -- a short test encode/decode for each candidate backend+codec pair -- before committing to an `HWAccelType`.

2. Add **CPU encoder validation** for the selected `VIDEO_CODEC` at startup (or within `detect_hw_accel`) with a clear fail-fast error message.

3. Explicitly set **`-pix_fmt yuv420p`** as baseline compatibility profile for H.264/H.265 in the encoder command, and add a test verifying the command includes it.

4. Strengthen the **test plan**: add a real integration smoke test with a small video file (with and without audio), plus an explicit test for `VideoAnnotator(..., hw_config=...)` constructor in the worker.

### Questions

1. Should `ffprobe` be treated as a mandatory startup dependency on par with `ffmpeg`?
2. What is the target compatibility level for output files -- "maximum quality" or "maximum client playback compatibility" (`yuv420p`)?
3. Is GPU backend selection without a runtime probe acceptable, given the stated expectation of "auto = just works"?
4. What should happen when the CPU encoder for the selected `VIDEO_CODEC` is missing: fail at startup, or automatically switch to a different codec?

---

## gemini-executor

### Critical Issues

1. **Incorrect FFmpeg argument ordering for AMD/VAAPI (FFmpegEncoder)** -- In the plan's `app/ffmpeg_pipe.py` (Task 4), encoder arguments from `hw_config.get_encode_args()` are appended *after* the input file (`-i pipe:0`). But for AMD, the returned list starts with `-vaapi_device`, which is a **global** parameter that must appear *before* the first `-i`. The current plan would produce `ffmpeg ... -i pipe:0 ... -vaapi_device ... output.mp4`, leading to an error or the device being ignored. The fix is to split `HWAccelConfig` arguments into global (inserted before `-i`) and codec-specific (inserted before the output file), or use the modern `-init_hw_device vaapi=va:/dev/dri/renderD128 -filter_hw_device va` syntax.

### Concerns

1. **Memory churn when writing frames** -- `FFmpegEncoder.write_frame` calls `frame.tobytes()`, creating a full copy (~25MB for 4K BGR24) on top of the numpy array. Could use `memoryview(frame)` instead if subprocess supports it.

2. **Audio re-encoding (-c:a aac)** -- Hardcoded AAC transcoding wastes CPU when the source is already AAC. Using `-c:a copy` would be more efficient, but `aac` is acceptable for v1 simplicity.

3. **Pixel format padding in rawvideo pipe** -- `FFmpegDecoder` uses `-pix_fmt bgr24`, which should produce packed pixels without stride padding for `pipe:1`, but worth monitoring for skew artifacts in testing.

### Suggestions

1. **Restructure HWAccelConfig** into `global_args`, `input_args`, and `output_args` instead of a single `get_encode_args()` method.

2. **Use `-init_hw_device`** syntax for VAAPI as the modern standard approach.

3. **Verify Docker NVIDIA dependencies** for ffmpeg hardware detection libraries.

### Questions

1. Does `-map 1:a:0?` correctly synchronize audio from the source file (stream 1) with video from the pipe (stream 0) when YOLO processing runs slower than real-time? (Likely yes due to FFmpeg buffering, but worth verifying.)

---

## ccs-executor (glmt)

### Critical Issues

1. **Frame size mismatch in FFmpegDecoder** -- `FFmpegDecoder` accepts `width` and `height` in the constructor but does not verify they match the actual decoded frame size from FFmpeg. If `_get_video_metadata` returns different dimensions than what FFmpeg decodes (due to rotation, padding, or ffprobe bugs), `read_frame()` will read the wrong number of bytes, causing frame corruption.

2. **`_drain_stderr` incompatible with `text=False`** -- The Popen is created with `text=False` (binary mode), so `process.stderr` yields `bytes`, not `str`. But `_drain_stderr` appends to `list[str]`, creating a type mismatch. The `for line in process.stderr` iteration with binary mode may also behave differently than expected.

3. **Race condition on stderr access** -- In `close()`, after `_stderr_thread.join(timeout=5)`, the code reads `_stderr_lines` for error messages. But the daemon thread might still be writing to `_stderr_lines` during the join timeout, and there is no synchronization (lock/deque).

### Concerns

1. AMD VAAPI decode with `-hwaccel_device` may not work on older FFmpeg versions
2. NVENC preset hardcoded as "p4" -- not configurable
3. FFmpegDecoder does not detect mid-stream resolution changes
4. Asymmetric timeouts: decoder `wait(timeout=10)` vs encoder `wait(timeout=300)` -- encoder might hang for 5 minutes
5. VAAPI `-vf format=nv12,hwupload` filter chain may be inefficient

### Suggestions

1. Remove width/height from FFmpegDecoder; let FFmpeg determine the size or detect it from the first frame
2. Add validation of first decoded frame size against ffprobe metadata
3. Use `threading.Lock` or `collections.deque(maxlen=...)` for thread-safe stderr collection
4. Add configurable NVENC preset (`nvenc_preset: str = "p4"`)
5. Unify subprocess timeouts into named constants
6. Consider PyAV (libav binding) as an alternative to subprocess pipes

### Questions

1. Why pass width/height to FFmpegDecoder when FFmpeg already knows the video dimensions?
2. Should the first frame be validated against metadata to detect mismatches early?
3. Is NVENC preset "p4" sufficient for all use cases, or should it be configurable?
4. Should runtime GPU encoder fallback be reconsidered?
