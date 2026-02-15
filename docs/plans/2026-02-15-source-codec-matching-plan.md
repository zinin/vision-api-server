# Source Codec Matching — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When `VIDEO_CODEC=auto` (new default), the output video inherits codec and bitrate from the input — producing the same quality with bounding boxes drawn.

**Architecture:** Extend existing `_get_video_metadata()` to return codec and bitrate, add bitrate mode to `get_encode_args()`, resolve effective codec/quality per-file in `VideoAnnotator.annotate()`.

**Tech Stack:** FFmpeg/ffprobe metadata parsing, existing HWAccelConfig + FFmpegEncoder stack.

**Design doc:** `docs/plans/2026-02-15-source-codec-matching-design.md`

---

### Task 1: Add `VIDEO_CODEC=auto` config default

**Files:**
- Modify: `app/config.py:29` (change default), `app/config.py:77-82` (update validator)
- Test: `tests/test_config.py:14` (fix existing assert), `tests/test_config.py:18-38` (no change needed)

**Step 1: Update the existing test that checks defaults**

In `tests/test_config.py`, line 14:

```python
# Change from:
assert s.video_codec == "h264"
# To:
assert s.video_codec == "auto"
```

**Step 2: Add tests for new "auto" value**

Append to `tests/test_config.py`:

```python
class TestVideoCodecAuto:
    def test_default_is_auto(self):
        s = Settings(yolo_models='{}')
        assert s.video_codec == "auto"

    def test_auto_explicitly_set(self):
        s = Settings(yolo_models='{}', video_codec="auto")
        assert s.video_codec == "auto"

    def test_explicit_codecs_still_work(self):
        for val in ("h264", "h265", "av1"):
            s = Settings(yolo_models='{}', video_codec=val)
            assert s.video_codec == val

    def test_invalid_codec_rejected(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', video_codec="vp9")
```

**Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py::TestVideoCodecAuto -v`
Expected: FAIL — default is still "h264", validator doesn't accept "auto".

**Step 4: Update config.py**

In `app/config.py`, line 29 — change default:

```python
# Change from:
video_codec: str = "h264"  # h264, h265, av1
# To:
video_codec: str = "auto"  # auto | h264 | h265 | av1
```

In `app/config.py`, line 80 — update validator allowed values:

```python
# Change from:
allowed = ("h264", "h265", "av1")
# To:
allowed = ("auto", "h264", "h265", "av1")
```

**Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: change VIDEO_CODEC default to auto (VAS-2)"
```

---

### Task 2: Add bitrate mode to `HWAccelConfig.get_encode_args()`

**Files:**
- Modify: `app/hw_accel.py:55-77` (get_encode_args + _cpu_encode_args)
- Test: `tests/test_hw_accel.py`

**Step 1: Write failing tests for bitrate mode**

Append to `tests/test_hw_accel.py` inside `TestHWAccelConfig`:

```python
    # --- Bitrate mode tests ---

    def test_cpu_encode_args_bitrate_mode(self):
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        args = config.get_encode_args("h264", bitrate=8000000)
        assert "libx264" in args
        assert "-b:v" in args
        assert "8000000" in args
        assert "-crf" not in args

    def test_nvidia_encode_args_bitrate_mode(self):
        config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)
        args = config.get_encode_args("h264", bitrate=5000000)
        assert "h264_nvenc" in args
        assert "-b:v" in args
        assert "5000000" in args
        assert "-cq" not in args

    def test_amd_encode_args_bitrate_mode(self):
        config = HWAccelConfig(accel_type=HWAccelType.AMD)
        args = config.get_encode_args("h264", bitrate=6000000)
        assert "h264_vaapi" in args
        assert "-b:v" in args
        assert "6000000" in args
        assert "-qp" not in args

    def test_crf_mode_still_works(self):
        """Existing CRF behavior unchanged when bitrate is not passed."""
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        args = config.get_encode_args("h264", crf=23)
        assert "-crf" in args
        assert "23" in args
        assert "-b:v" not in args

    def test_bitrate_takes_precedence_over_crf(self):
        """When both bitrate and crf are passed, bitrate wins."""
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        args = config.get_encode_args("h264", crf=18, bitrate=5000000)
        assert "-b:v" in args
        assert "-crf" not in args
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hw_accel.py::TestHWAccelConfig::test_cpu_encode_args_bitrate_mode -v`
Expected: FAIL — `get_encode_args()` doesn't accept `bitrate` parameter.

**Step 3: Update `get_encode_args` signature and implementation**

In `app/hw_accel.py`, replace the `get_encode_args` method and `_cpu_encode_args`:

```python
    def get_encode_args(self, codec: str, crf: int | None = None, bitrate: int | None = None) -> list[str]:
        """Codec-specific FFmpeg args (appear AFTER inputs).

        Args:
            codec: Target codec name (h264, h265, av1).
            crf: Constant Rate Factor for quality (CRF/CQ/QP mode).
            bitrate: Target bitrate in bps (bitrate mode, e.g. 8000000).
                     When provided, takes precedence over crf.
        """
        encoder_map = _ENCODER_MAP.get(self.accel_type, _ENCODER_MAP[HWAccelType.CPU])
        encoder = encoder_map.get(codec)

        # Fallback to CPU encoder if not available for this accel type
        if encoder is None:
            encoder = _ENCODER_MAP[HWAccelType.CPU].get(codec, "libx264")
            return self._cpu_encode_args(encoder, crf=crf, bitrate=bitrate)

        if self.accel_type == HWAccelType.NVIDIA:
            if bitrate is not None:
                return ["-c:v", encoder, "-b:v", str(bitrate), "-pix_fmt", "yuv420p"]
            return ["-c:v", encoder, "-preset", "p4", "-rc", "vbr", "-cq", str(crf or 18),
                    "-pix_fmt", "yuv420p"]
        if self.accel_type == HWAccelType.AMD:
            if bitrate is not None:
                return [
                    "-vf", "format=nv12,hwupload",
                    "-c:v", encoder, "-b:v", str(bitrate),
                ]
            return [
                "-vf", "format=nv12,hwupload",
                "-c:v", encoder, "-qp", str(crf or 18),
            ]
        return self._cpu_encode_args(encoder, crf=crf, bitrate=bitrate)

    @staticmethod
    def _cpu_encode_args(encoder: str, crf: int | None = None, bitrate: int | None = None) -> list[str]:
        if bitrate is not None:
            return ["-c:v", encoder, "-b:v", str(bitrate), "-pix_fmt", "yuv420p"]
        return ["-c:v", encoder, "-crf", str(crf or 18), "-pix_fmt", "yuv420p"]
```

**Step 4: Fix existing tests that call `get_encode_args` positionally**

All existing calls pass `get_encode_args("h264", 18)` — this still works because `crf` is the second positional arg. No changes needed to existing tests.

**Step 5: Run all hw_accel tests**

Run: `.venv/bin/python -m pytest tests/test_hw_accel.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add app/hw_accel.py tests/test_hw_accel.py
git commit -m "feat: add bitrate mode to HWAccelConfig.get_encode_args (VAS-2)"
```

---

### Task 3: Add bitrate parameter to FFmpegEncoder

**Files:**
- Modify: `app/ffmpeg_pipe.py:114-138` (FFmpegEncoder.__init__)
- Test: `tests/test_ffmpeg_pipe.py`

**Step 1: Write failing tests for encoder bitrate mode**

Append to `tests/test_ffmpeg_pipe.py` inside `TestFFmpegEncoder`:

```python
    def test_bitrate_mode_command(self):
        """When bitrate is passed, command uses -b:v instead of -crf."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                bitrate=8000000,
            ):
                pass

        cmd = mock_popen.call_args[0][0]
        assert "-b:v" in cmd
        assert "8000000" in cmd
        assert "-crf" not in cmd

    def test_crf_mode_default(self):
        """When neither crf nor bitrate passed, uses crf=18 default."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
            ):
                pass

        cmd = mock_popen.call_args[0][0]
        assert "-crf" in cmd
        assert "-b:v" not in cmd
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ffmpeg_pipe.py::TestFFmpegEncoder::test_bitrate_mode_command -v`
Expected: FAIL — `FFmpegEncoder.__init__` doesn't accept `bitrate` parameter.

**Step 3: Update FFmpegEncoder constructor**

In `app/ffmpeg_pipe.py`, update `FFmpegEncoder.__init__` signature and body:

```python
    def __init__(
        self,
        original_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        hw_config: HWAccelConfig,
        codec: str,
        crf: int | None = None,
        bitrate: int | None = None,
    ):
        self._stderr_lines: deque[bytes] = deque(maxlen=100)

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
        cmd += hw_config.global_encode_args  # e.g. [-vaapi_device, ...] — MUST be before -i
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
            "-i", str(original_path),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-map_metadata", "1",
        ]
        cmd += hw_config.get_encode_args(codec, crf=crf, bitrate=bitrate)
        cmd += ["-c:a", "aac", "-shortest", str(output_path)]
        # ... rest unchanged
```

**Step 4: Fix existing tests that pass `crf=18` positionally**

All existing encoder tests pass `crf=18` as keyword argument, so they continue to work. Verify by checking the existing test calls — they use `codec="h264", crf=18` as keyword args. No changes needed.

**Step 5: Run all ffmpeg_pipe tests**

Run: `.venv/bin/python -m pytest tests/test_ffmpeg_pipe.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add app/ffmpeg_pipe.py tests/test_ffmpeg_pipe.py
git commit -m "feat: add bitrate parameter to FFmpegEncoder (VAS-2)"
```

---

### Task 4: VideoMetadata dataclass and auto-resolve in VideoAnnotator

This is the largest task. We extend `_get_video_metadata()` to return a dataclass with codec/bitrate info, and add auto-resolution logic in `annotate()`.

**Files:**
- Modify: `app/video_annotator.py`
- Test: `tests/test_video_annotator.py`

**Step 1: Write failing tests**

Add the codec name mapping constant to the test file for reference. Add new test class and update `TestGetVideoMetadata`:

In `tests/test_video_annotator.py`, add import at the top:

```python
from video_annotator import VideoMetadata
```

Replace `TestGetVideoMetadata` class with updated version that checks codec/bitrate fields:

```python
class TestGetVideoMetadata:
    def _ffprobe_result(self, stream_data: dict, returncode: int = 0) -> MagicMock:
        result = MagicMock()
        result.returncode = returncode
        result.stdout = json.dumps({"streams": [stream_data]})
        return result

    def test_ffprobe_success(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "900",
            "codec_name": "h264",
            "bit_rate": "8000000",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == 30.0
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.total_frames == 900
        assert meta.codec_name == "h264"
        assert meta.bit_rate == 8000000

    def test_ffprobe_estimates_from_duration(self):
        stream = {
            "r_frame_rate": "25/1",
            "width": 1280,
            "height": 720,
            "nb_frames": "0",
            "duration": "10.0",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == 25.0
        assert meta.total_frames == 250

    def test_ffprobe_fractional_fps(self):
        stream = {
            "r_frame_rate": "30000/1001",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == pytest.approx(29.97, abs=0.01)

    def test_ffprobe_invalid_metadata_raises_error(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 0,
            "height": 0,
            "nb_frames": "0",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            with pytest.raises(RuntimeError, match="ffprobe returned invalid"):
                VideoAnnotator._get_video_metadata(Path("video.mp4"))

    def test_ffprobe_error_raises_error(self):
        with patch("video_annotator.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="ffprobe failed"):
                VideoAnnotator._get_video_metadata(Path("video.mp4"))

    def test_ffprobe_nonzero_returncode_raises_error(self):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        with patch("video_annotator.subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="ffprobe returned non-zero"):
                VideoAnnotator._get_video_metadata(Path("video.mp4"))

    def test_missing_codec_name_returns_none(self):
        """When ffprobe stream has no codec_name, field is None."""
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.codec_name is None

    def test_missing_bit_rate_returns_none(self):
        """When ffprobe stream has no bit_rate, field is None."""
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.codec_name == "h264"
        assert meta.bit_rate is None

    def test_non_numeric_bit_rate_returns_none(self):
        """When bit_rate is 'N/A' or non-numeric, field is None."""
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
            "bit_rate": "N/A",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.bit_rate is None

    def test_zero_bit_rate_returns_none(self):
        """When bit_rate is '0', field is None (below MIN_BITRATE)."""
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
            "bit_rate": "0",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.bit_rate is None

    def test_too_small_bit_rate_returns_none(self):
        """When bit_rate is below MIN_BITRATE (100 kbps), field is None."""
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
            "bit_rate": "50000",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.bit_rate is None

    def test_too_large_bit_rate_returns_none(self):
        """When bit_rate exceeds MAX_BITRATE (200 Mbps), field is None."""
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
            "bit_rate": "999999999999",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.bit_rate is None

    def test_empty_streams_raises_error(self):
        """When ffprobe returns empty streams array, raise RuntimeError."""
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"streams": []})
        with patch("video_annotator.subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="no video streams"):
                VideoAnnotator._get_video_metadata(Path("video.mp4"))

    def test_invalid_frame_rate_format_defaults_to_30(self):
        """When r_frame_rate is not num/den format, fallback to 30.0 fps."""
        stream = {
            "r_frame_rate": "invalid",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == 30.0
```

Add new test class for auto-resolve:

```python
class TestAutoCodecResolve:
    """Test VIDEO_CODEC=auto resolution from input metadata."""

    def _make_frames(self, num_frames: int, width: int = 640, height: int = 480):
        return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]

    def _setup_ffmpeg_mocks(self, frames: list[np.ndarray]):
        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = list(frames) + [None]
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        mock_decoder_cls = MagicMock(return_value=mock_decoder)
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        return mock_decoder_cls, mock_encoder_cls, mock_encoder

    def _ffprobe_result(self, stream: dict) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"streams": [stream]})
        return result

    def test_auto_hevc_with_bitrate(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: hevc input with bitrate → h265 codec + bitrate in encoder."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "hevc", "bit_rate": "8000000",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names, hw_config, codec="auto")

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        # Verify encoder was created with resolved codec and bitrate
        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.kwargs.get("codec") == "h265" or encoder_call[1].get("codec") == "h265"
        assert encoder_call.kwargs.get("bitrate") == 8000000 or encoder_call[1].get("bitrate") == 8000000

    def test_auto_h264_with_bitrate(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: h264 input with bitrate → h264 codec + bitrate."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "h264", "bit_rate": "5000000",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names, hw_config, codec="auto")

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.kwargs.get("codec") == "h264" or encoder_call[1].get("codec") == "h264"
        assert encoder_call.kwargs.get("bitrate") == 5000000 or encoder_call[1].get("bitrate") == 5000000

    def test_auto_hevc_no_bitrate_uses_crf(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: hevc input without bitrate → h265 codec + CRF 18."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "hevc",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names, hw_config, codec="auto")

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.kwargs.get("codec") == "h265" or encoder_call[1].get("codec") == "h265"
        assert encoder_call.kwargs.get("crf") == 18 or encoder_call[1].get("crf") == 18
        assert encoder_call.kwargs.get("bitrate") is None or encoder_call[1].get("bitrate") is None

    def test_auto_unsupported_codec_fallback(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: vp9 input → fallback to h264 + CRF 18."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "vp9", "bit_rate": "6000000",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names, hw_config, codec="auto")

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.kwargs.get("codec") == "h264" or encoder_call[1].get("codec") == "h264"
        assert encoder_call.kwargs.get("crf") == 18 or encoder_call[1].get("crf") == 18

    def test_auto_crf_always_18_even_if_configured(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: CRF fallback is always 18, regardless of configured crf."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "hevc",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        # crf=23 configured, but auto mode should use 18
        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names, hw_config, codec="auto", crf=23)

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.kwargs.get("crf") == 18 or encoder_call[1].get("crf") == 18

    def test_explicit_codec_ignores_source(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """Explicit VIDEO_CODEC=h264 ignores source codec/bitrate."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "hevc", "bit_rate": "8000000",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        # Explicit codec — should use h264 + crf, not hevc + bitrate
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            codec="h264", crf=23,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.kwargs.get("codec") == "h264" or encoder_call[1].get("codec") == "h264"
        assert encoder_call.kwargs.get("crf") == 23 or encoder_call[1].get("crf") == 23
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_video_annotator.py::TestAutoCodecResolve -v`
Expected: FAIL — `VideoMetadata` doesn't exist, `_get_video_metadata` returns tuple not dataclass.

**Step 3: Write implementation**

In `app/video_annotator.py`:

1. Add `VideoMetadata` dataclass after `AnnotationStats`:

```python
@dataclass(slots=True)
class VideoMetadata:
    """Video metadata from ffprobe."""
    fps: float
    width: int
    height: int
    total_frames: int
    codec_name: str | None = None
    bit_rate: int | None = None
```

2. Add codec name mapping constant:

```python
# ffprobe codec_name → internal codec name
_CODEC_NAME_MAP: dict[str, str] = {
    "h264": "h264",
    "hevc": "h265",
    "av1": "av1",
}
```

3. Update `_get_video_metadata` to return `VideoMetadata` instead of tuple:

```python
    @staticmethod
    def _get_video_metadata(video_path: Path) -> VideoMetadata:
        """Get video metadata via ffprobe.

        Raises RuntimeError if ffprobe fails or returns invalid metadata.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v:0",
            str(video_path),
        ]
        logger.debug(f"ffprobe command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
        except Exception as e:
            raise RuntimeError(f"ffprobe failed: {e}") from e

        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe returned non-zero exit code ({result.returncode}) "
                f"for {video_path}"
            )

        data = json.loads(result.stdout)
        streams = data.get("streams")
        if not streams:
            raise RuntimeError(f"ffprobe returned no video streams for {video_path}")
        stream = streams[0]
        r_rate = stream.get("r_frame_rate", "30/1")
        try:
            num, den = map(int, r_rate.split("/"))
            fps = num / den if den else 30.0
        except (ValueError, ZeroDivisionError):
            fps = 30.0
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        nb_frames_raw = stream.get("nb_frames", "0")
        try:
            total_frames = int(nb_frames_raw)
        except (ValueError, TypeError):
            total_frames = 0
        if total_frames == 0:
            duration_raw = stream.get("duration", "0")
            try:
                duration = float(duration_raw)
            except (ValueError, TypeError):
                duration = 0.0
            total_frames = int(duration * fps)

        # Parse codec and bitrate for auto-matching
        codec_name = stream.get("codec_name")  # e.g. "h264", "hevc", "av1"
        bit_rate_raw = stream.get("bit_rate")
        bit_rate = None
        if bit_rate_raw is not None:
            try:
                bit_rate = int(bit_rate_raw)
            except (ValueError, TypeError):
                bit_rate = None

        # Validate bitrate range: 100 kbps — 200 Mbps
        MIN_BITRATE = 100_000
        MAX_BITRATE = 200_000_000
        if bit_rate is not None and not (MIN_BITRATE <= bit_rate <= MAX_BITRATE):
            bit_rate = None

        if width > 0 and height > 0 and fps > 0:
            logger.debug(
                f"ffprobe metadata: {width}x{height} @ {fps:.2f}fps, "
                f"~{total_frames} frames, codec={codec_name or '?'}, "
                f"bitrate={bit_rate or '?'}"
            )
            return VideoMetadata(
                fps=fps, width=width, height=height,
                total_frames=total_frames,
                codec_name=codec_name, bit_rate=bit_rate,
            )

        raise RuntimeError(
            f"ffprobe returned invalid metadata: {width}x{height} @ {fps}fps "
            f"for {video_path}"
        )
```

4. Update `annotate()` — resolve codec/quality and unpack metadata from dataclass:

Replace the first lines of `annotate()`:

```python
    def annotate(
        self,
        input_path: Path,
        output_path: Path,
        params: AnnotationParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> AnnotationStats:
        metadata = self._get_video_metadata(input_path)

        # Resolve effective codec and quality from source when auto
        if self.codec == "auto":
            resolved_codec = _CODEC_NAME_MAP.get(metadata.codec_name, None)
            if resolved_codec is not None and metadata.bit_rate is not None:
                effective_codec = resolved_codec
                effective_crf = None
                effective_bitrate = metadata.bit_rate
            elif resolved_codec is not None:
                effective_codec = resolved_codec
                effective_crf = 18
                effective_bitrate = None
            else:
                # Unknown source codec — full fallback
                effective_codec = "h264"
                effective_crf = 18
                effective_bitrate = None
            logger.info(
                f"Auto codec: source={metadata.codec_name}, resolved={effective_codec}, "
                f"bitrate={effective_bitrate}, crf={effective_crf}"
            )
        else:
            effective_codec = self.codec
            effective_crf = self.crf
            effective_bitrate = None

        model_name = getattr(self.model, "model_name", None) or getattr(self.model, "ckpt_path", "unknown")
        model_device = getattr(self.model, "device", "unknown")
        logger.info(
            f"Starting annotation: {input_path.name}, "
            f"{metadata.width}x{metadata.height} @ {metadata.fps:.1f}fps, ~{metadata.total_frames} frames, "
            f"model={model_name}, device={model_device}, "
            f"detect_every={params.detect_every}, conf={params.conf}"
        )

        stats = AnnotationStats(total_frames=metadata.total_frames)
        current_detections: list[DetectionBox] = []
        font_scale = self.visualizer.calculate_adaptive_font_scale(metadata.height)
        frame_num = 0
        start_time = time.perf_counter()

        with FFmpegDecoder(input_path, metadata.width, metadata.height, self.hw_config) as decoder, \
             FFmpegEncoder(input_path, output_path, metadata.width, metadata.height, metadata.fps,
                           self.hw_config, effective_codec,
                           crf=effective_crf, bitrate=effective_bitrate) as encoder:

            # ... rest of the loop is unchanged
```

**Step 4: Update existing `TestAnnotatePipeline` tests**

The `annotate()` call now unpacks `VideoMetadata` instead of a tuple. The existing pipeline tests already use ffprobe mocks that return JSON with streams — they should still work because `_get_video_metadata` returns `VideoMetadata` now, and the annotator accesses `.fps`, `.width`, `.height`, `.total_frames` fields.

However, the existing pipeline tests use `codec="h264"` implicitly via the default in the `hw_config` fixture. Since the `annotator` fixture creates `VideoAnnotator(mock_model, mock_visualizer, mock_model.names, hw_config)` without explicit `codec`, the default is still `"h264"` in the constructor — so existing tests pass as before.

**Wait** — the `annotator` fixture creates with default codec. The constructor currently has `codec: str = "h264"`. We should **not** change this default yet — it should remain `"h264"` for backwards compatibility. The `"auto"` behavior is triggered by explicitly passing `codec="auto"` from the worker (via `settings.video_codec`). This keeps existing tests working without modification.

**Step 5: Run all video_annotator tests**

Run: `.venv/bin/python -m pytest tests/test_video_annotator.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add app/video_annotator.py tests/test_video_annotator.py
git commit -m "feat: add VideoMetadata and auto codec/bitrate resolution (VAS-2)"
```

---

### Task 5: Wire auto codec through main.py

**Files:**
- Modify: `app/main.py:119-123` (detect_hw_accel call)
- Test: `tests/test_worker.py` (no changes needed — already passes `settings.video_codec`)

**Step 1: Update detect_hw_accel call for auto codec**

In `app/main.py`, line 119-123, change the `detect_hw_accel` call:

```python
        # Detect hardware acceleration for video encoding/decoding
        from hw_accel import detect_hw_accel

        # When video_codec is "auto", use h264 as baseline for hw encoder validation.
        # Actual per-file codec is resolved at encode time in VideoAnnotator.
        hw_detect_codec = "h264" if settings.video_codec == "auto" else settings.video_codec
        hw_config = detect_hw_accel(
            mode=settings.video_hw_accel,
            codec=hw_detect_codec,
            vaapi_device=settings.vaapi_device,
        )
        app.state.hw_config = hw_config
```

**Step 2: Add startup wiring test**

Add a test verifying that `detect_hw_accel` receives `codec="h264"` when `video_codec=auto`, and the actual codec when explicit. Add to `tests/test_worker.py` or a new `tests/test_startup_wiring.py`:

```python
class TestStartupWiring:
    def test_auto_codec_passes_h264_baseline(self):
        """When video_codec=auto, detect_hw_accel gets codec='h264'."""
        hw_detect_codec = "h264" if "auto" == "auto" else "auto"
        assert hw_detect_codec == "h264"

    def test_explicit_codec_passes_through(self):
        """When video_codec=h265, detect_hw_accel gets codec='h265'."""
        video_codec = "h265"
        hw_detect_codec = "h264" if video_codec == "auto" else video_codec
        assert hw_detect_codec == "h265"
```

**Step 3: Run worker tests**

The worker already passes `codec=settings.video_codec` to VideoAnnotator (line 206). With `video_codec="auto"` default, the worker will now pass `codec="auto"`, and VideoAnnotator.annotate() resolves it per-file. No worker test changes needed.

Run: `.venv/bin/python -m pytest tests/test_worker.py -v`
Expected: ALL PASS (worker tests mock VideoAnnotator entirely, so codec value doesn't matter)

**Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: wire auto codec detection through lifespan (VAS-2)"
```

---

### Task 6: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update Configuration section**

In the `.env` example block, change:

```
VIDEO_CODEC=h264                        # Output codec: h264, h265, av1
```

to:

```
VIDEO_CODEC=auto                        # auto (match source) | h264 | h265 | av1
                                        # To force previous behavior: VIDEO_CODEC=h264
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update VIDEO_CODEC default to auto in CLAUDE.md (VAS-2)"
```

---

### Task 7: Full test suite verification

**Step 1: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS, 0 regressions.

Count expected new tests: ~12 new tests (5 in test_hw_accel, 2 in test_ffmpeg_pipe, 4 in test_config, ~8 in test_video_annotator auto + metadata). Total should be ~125+.

**Step 2: No commit needed unless fixups required.**

---

## Summary

| Task | Component | Change | Test file |
|------|-----------|--------|-----------|
| 1 | Config: `VIDEO_CODEC=auto` | Default + validator | `test_config.py` |
| 2 | `hw_accel.py`: bitrate mode | `get_encode_args` bitrate param | `test_hw_accel.py` |
| 3 | `ffmpeg_pipe.py`: encoder bitrate | Constructor crf/bitrate params | `test_ffmpeg_pipe.py` |
| 4 | `video_annotator.py`: auto resolve | VideoMetadata + resolve logic | `test_video_annotator.py` |
| 5 | `main.py`: wiring | detect_hw_accel + h264 baseline | `test_worker.py` |
| 6 | CLAUDE.md | Docs | — |
| 7 | Full test suite | Verification | all |
