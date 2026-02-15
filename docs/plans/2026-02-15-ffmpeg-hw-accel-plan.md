# FFmpeg Hardware-Accelerated Video Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace cv2-based video I/O in VideoAnnotator with FFmpeg pipe-based pipeline supporting GPU-accelerated decoding (NVDEC/VAAPI) and encoding (NVENC/VAAPI) with automatic fallback to CPU.

**Architecture:** Two new modules (`hw_accel.py` for hardware detection, `ffmpeg_pipe.py` for FFmpeg subprocess pipes) that replace `cv2.VideoCapture`/`cv2.VideoWriter` in `VideoAnnotator`. Detection runs once at startup, result is cached. Encoder pipe also merges audio, eliminating the intermediate file and double encoding.

**Tech Stack:** FFmpeg subprocess pipes, numpy for frame I/O, existing YOLO + DetectionVisualizer stack unchanged.

**Design doc:** `docs/plans/2026-02-15-ffmpeg-hw-accel-design.md`

---

### Task 1: Add `VIDEO_HW_ACCEL` config setting

**Files:**
- Modify: `app/config.py:24-31` (add setting near other video settings)
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

In `tests/test_config.py`, add:

```python
class TestVideoHwAccel:
    def test_default_auto(self):
        s = Settings(yolo_models='{}')
        assert s.video_hw_accel == "auto"

    def test_valid_values(self):
        for val in ("auto", "nvidia", "amd", "cpu"):
            s = Settings(yolo_models='{}', video_hw_accel=val)
            assert s.video_hw_accel == val

    def test_invalid_value_rejected(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', video_hw_accel="vulkan")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py::TestVideoHwAccel -v`
Expected: FAIL — `video_hw_accel` field does not exist yet.

**Step 3: Write minimal implementation**

In `app/config.py`, add to `Settings` class after `video_crf`:

```python
video_hw_accel: str = "auto"  # auto | nvidia | amd | cpu
vaapi_device: str = "/dev/dri/renderD128"  # VAAPI render device path

@field_validator("video_hw_accel")
@classmethod
def validate_video_hw_accel(cls, v: str) -> str:
    allowed = ("auto", "nvidia", "amd", "cpu")
    if v not in allowed:
        raise ValueError(f"video_hw_accel must be one of: {allowed}")
    return v
```

Also add `vaapi_device` test to `TestVideoHwAccel`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py::TestVideoHwAccel -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add VIDEO_HW_ACCEL config setting (VAS-2)"
```

---

### Task 2: Hardware acceleration detection module

**Files:**
- Create: `app/hw_accel.py`
- Test: `tests/test_hw_accel.py`

**Step 1: Write the failing tests**

Create `tests/test_hw_accel.py`:

```python
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from hw_accel import HWAccelType, HWAccelConfig, detect_hw_accel


class TestHWAccelConfig:
    def test_cpu_decode_args_empty(self):
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        assert config.decode_args == []

    def test_nvidia_decode_args(self):
        config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)
        assert "-hwaccel" in config.decode_args
        assert "cuda" in config.decode_args

    def test_amd_decode_args(self):
        config = HWAccelConfig(accel_type=HWAccelType.AMD)
        assert "-hwaccel" in config.decode_args
        assert "vaapi" in config.decode_args

    def test_cpu_encode_args_h264(self):
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        args = config.get_encode_args("h264", 18)
        assert "-c:v" in args
        assert "libx264" in args

    def test_nvidia_encode_args_h264(self):
        config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)
        args = config.get_encode_args("h264", 18)
        assert "h264_nvenc" in args

    def test_nvidia_encode_args_h265(self):
        config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)
        args = config.get_encode_args("h265", 18)
        assert "hevc_nvenc" in args

    def test_amd_encode_args_h264(self):
        config = HWAccelConfig(accel_type=HWAccelType.AMD)
        args = config.get_encode_args("h264", 18)
        assert "h264_vaapi" in args

    def test_amd_encode_args_av1_falls_back_to_cpu(self):
        config = HWAccelConfig(accel_type=HWAccelType.AMD)
        args = config.get_encode_args("av1", 18)
        assert "libsvtav1" in args

    def test_amd_encode_args_include_vaapi_device(self):
        config = HWAccelConfig(accel_type=HWAccelType.AMD)
        args = config.get_encode_args("h264", 18)
        assert "-vaapi_device" in args


class TestDetectHwAccel:
    def _mock_subprocess(self, hwaccels_output: str, encoders_output: str):
        """Helper to mock subprocess.run for ffmpeg queries."""
        def side_effect(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            cmd_str = " ".join(cmd)
            if "-hwaccels" in cmd_str:
                result.stdout = hwaccels_output
            elif "-encoders" in cmd_str:
                result.stdout = encoders_output
            else:
                result.stdout = ""
            return result
        return side_effect

    def test_detect_nvidia(self):
        hwaccels = "Hardware acceleration methods:\ncuda\n"
        encoders = "...\n V..... h264_nvenc           NVIDIA NVENC H.264 encoder\n"
        with patch("hw_accel.subprocess.run", side_effect=self._mock_subprocess(hwaccels, encoders)):
            config = detect_hw_accel("auto")
        assert config.accel_type == HWAccelType.NVIDIA

    def test_detect_amd(self):
        hwaccels = "Hardware acceleration methods:\nvaapi\n"
        encoders = "...\n V..... h264_vaapi           H.264/AVC (VAAPI)\n"
        with patch("hw_accel.subprocess.run", side_effect=self._mock_subprocess(hwaccels, encoders)):
            config = detect_hw_accel("auto")
        assert config.accel_type == HWAccelType.AMD

    def test_detect_cpu_fallback(self):
        hwaccels = "Hardware acceleration methods:\n"
        encoders = "...\n V..... libx264\n"
        with patch("hw_accel.subprocess.run", side_effect=self._mock_subprocess(hwaccels, encoders)):
            config = detect_hw_accel("auto")
        assert config.accel_type == HWAccelType.CPU

    def test_force_cpu(self):
        """When forced to cpu, don't probe — just return CPU config."""
        config = detect_hw_accel("cpu")
        assert config.accel_type == HWAccelType.CPU

    def test_force_nvidia(self):
        hwaccels = "Hardware acceleration methods:\ncuda\n"
        encoders = "...\n V..... h264_nvenc           NVIDIA NVENC\n"
        with patch("hw_accel.subprocess.run", side_effect=self._mock_subprocess(hwaccels, encoders)):
            config = detect_hw_accel("nvidia")
        assert config.accel_type == HWAccelType.NVIDIA

    def test_force_nvidia_not_available_falls_back(self):
        hwaccels = "Hardware acceleration methods:\n"
        encoders = "...\n"
        with patch("hw_accel.subprocess.run", side_effect=self._mock_subprocess(hwaccels, encoders)):
            config = detect_hw_accel("nvidia")
        assert config.accel_type == HWAccelType.CPU

    def test_ffmpeg_not_found(self):
        with patch("hw_accel.subprocess.run", side_effect=FileNotFoundError):
            config = detect_hw_accel("auto")
        assert config.accel_type == HWAccelType.CPU
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hw_accel.py -v`
Expected: FAIL — `hw_accel` module does not exist.

**Step 3: Write implementation**

Create `app/hw_accel.py`:

```python
import logging
import subprocess
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HWAccelType(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    CPU = "cpu"


# Encoder mapping: (codec_config) -> encoder name per accel type
_ENCODER_MAP: dict[HWAccelType, dict[str, str]] = {
    HWAccelType.NVIDIA: {
        "h264": "h264_nvenc",
        "h265": "hevc_nvenc",
        "av1": "av1_nvenc",
    },
    HWAccelType.AMD: {
        "h264": "h264_vaapi",
        "h265": "hevc_vaapi",
        # av1: no VAAPI encoder — fallback handled in get_encode_args
    },
    HWAccelType.CPU: {
        "h264": "libx264",
        "h265": "libx265",
        "av1": "libsvtav1",
    },
}


@dataclass(frozen=True)
class HWAccelConfig:
    accel_type: HWAccelType
    vaapi_device: str = "/dev/dri/renderD128"

    @property
    def decode_args(self) -> list[str]:
        if self.accel_type == HWAccelType.NVIDIA:
            return ["-hwaccel", "cuda"]
        if self.accel_type == HWAccelType.AMD:
            return ["-hwaccel", "vaapi", "-hwaccel_device", self.vaapi_device]
        return []

    @property
    def global_encode_args(self) -> list[str]:
        """Global FFmpeg args that MUST appear BEFORE -i (e.g. -vaapi_device)."""
        if self.accel_type == HWAccelType.AMD:
            return ["-vaapi_device", self.vaapi_device]
        return []

    def get_encode_args(self, codec: str, crf: int) -> list[str]:
        """Codec-specific FFmpeg args (appear AFTER inputs)."""
        encoder_map = _ENCODER_MAP.get(self.accel_type, _ENCODER_MAP[HWAccelType.CPU])
        encoder = encoder_map.get(codec)

        # Fallback to CPU encoder if not available for this accel type
        if encoder is None:
            encoder = _ENCODER_MAP[HWAccelType.CPU].get(codec, "libx264")
            return self._cpu_encode_args(encoder, crf)

        if self.accel_type == HWAccelType.NVIDIA:
            return ["-c:v", encoder, "-preset", "p4", "-rc", "vbr", "-cq", str(crf),
                    "-pix_fmt", "yuv420p"]
        if self.accel_type == HWAccelType.AMD:
            return [
                "-vf", "format=nv12,hwupload",
                "-c:v", encoder, "-qp", str(crf),
            ]
        return self._cpu_encode_args(encoder, crf)

    @staticmethod
    def _cpu_encode_args(encoder: str, crf: int) -> list[str]:
        return ["-c:v", encoder, "-crf", str(crf), "-pix_fmt", "yuv420p"]


def _ffmpeg_query(args: list[str]) -> str:
    """Run ffmpeg with given args, return stdout."""
    try:
        result = subprocess.run(
            ["ffmpeg"] + args,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _has_hwaccel(name: str) -> bool:
    output = _ffmpeg_query(["-hide_banner", "-hwaccels"])
    return name in output


def _has_encoder(name: str) -> bool:
    output = _ffmpeg_query(["-hide_banner", "-encoders"])
    return name in output


def detect_hw_accel(mode: str = "auto", codec: str = "h264", vaapi_device: str = "/dev/dri/renderD128") -> HWAccelConfig:
    """Detect available hardware acceleration.

    Args:
        mode: "auto", "nvidia", "amd", or "cpu"
        codec: configured video codec ("h264", "h265", "av1") — used to check encoder availability
        vaapi_device: VAAPI render device path

    Returns:
        HWAccelConfig with detected acceleration type.
    """
    if mode == "cpu":
        logger.info("Hardware acceleration: forced CPU")
        return HWAccelConfig(accel_type=HWAccelType.CPU)

    # Check NVIDIA — verify encoder for configured codec exists
    if mode in ("auto", "nvidia"):
        try:
            nvidia_encoder = _ENCODER_MAP[HWAccelType.NVIDIA].get(codec)
            if _has_hwaccel("cuda") and nvidia_encoder and _has_encoder(nvidia_encoder):
                logger.info("Hardware acceleration: NVIDIA (NVDEC/NVENC)")
                return HWAccelConfig(accel_type=HWAccelType.NVIDIA, vaapi_device=vaapi_device)
        except Exception as e:
            logger.warning(f"NVIDIA detection failed: {e}")
        if mode == "nvidia":
            logger.warning("NVIDIA requested but not available, falling back to CPU")

    # Check AMD — verify encoder for configured codec exists (av1 has no VAAPI encoder)
    if mode in ("auto", "amd"):
        try:
            amd_encoder = _ENCODER_MAP.get(HWAccelType.AMD, {}).get(codec)
            if _has_hwaccel("vaapi") and amd_encoder and _has_encoder(amd_encoder):
                logger.info("Hardware acceleration: AMD (VAAPI)")
                return HWAccelConfig(accel_type=HWAccelType.AMD, vaapi_device=vaapi_device)
        except Exception as e:
            logger.warning(f"AMD/VAAPI detection failed: {e}")
        if mode == "amd":
            logger.warning("AMD/VAAPI requested but not available, falling back to CPU")

    # CPU fallback — validate that the encoder for configured codec is available
    cpu_encoder = _ENCODER_MAP[HWAccelType.CPU].get(codec)
    if cpu_encoder and not _has_encoder(cpu_encoder):
        raise RuntimeError(
            f"CPU encoder '{cpu_encoder}' for codec '{codec}' not found in FFmpeg. "
            f"Install FFmpeg with '{cpu_encoder}' support or change VIDEO_CODEC."
        )

    logger.info("Hardware acceleration: CPU (software codecs)")
    return HWAccelConfig(accel_type=HWAccelType.CPU)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hw_accel.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add app/hw_accel.py tests/test_hw_accel.py
git commit -m "feat: add hardware acceleration detection module (VAS-2)"
```

---

### Task 3: FFmpeg pipe decoder

**Files:**
- Create: `app/ffmpeg_pipe.py`
- Test: `tests/test_ffmpeg_pipe.py`

**Step 1: Write the failing tests for FFmpegDecoder**

Create `tests/test_ffmpeg_pipe.py`:

```python
import subprocess
from unittest.mock import patch, MagicMock, PropertyMock
from io import BytesIO

import numpy as np
import pytest

from ffmpeg_pipe import FFmpegDecoder
from hw_accel import HWAccelConfig, HWAccelType


class TestFFmpegDecoder:
    def _make_mock_process(self, frames: list[np.ndarray]):
        """Create mock Popen that yields raw frame bytes then EOF."""
        raw_data = b"".join(f.tobytes() for f in frames)
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(raw_data)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        return mock_proc

    def test_reads_frames(self):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        mock_proc = self._make_mock_process(frames)
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 640, 480, config) as decoder:
                read_frames = []
                while True:
                    frame = decoder.read_frame()
                    if frame is None:
                        break
                    read_frames.append(frame)

        assert len(read_frames) == 3
        assert read_frames[0].shape == (480, 640, 3)

    def test_eof_returns_none(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 640, 480, config) as decoder:
                assert decoder.read_frame() is None

    def test_nvidia_decode_args(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegDecoder("input.mp4", 640, 480, config):
                pass
        cmd = mock_popen.call_args[0][0]
        assert "-hwaccel" in cmd
        assert "cuda" in cmd

    def test_cpu_no_hwaccel_args(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegDecoder("input.mp4", 640, 480, config):
                pass
        cmd = mock_popen.call_args[0][0]
        assert "-hwaccel" not in cmd

    def test_cleanup_on_exit(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 640, 480, config):
                pass

        mock_proc.stdout.close.assert_called()
        mock_proc.wait.assert_called()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ffmpeg_pipe.py::TestFFmpegDecoder -v`
Expected: FAIL — module doesn't exist.

**Step 3: Write FFmpegDecoder implementation**

Create `app/ffmpeg_pipe.py`:

```python
import logging
import subprocess
import threading
from collections import deque
from pathlib import Path

import numpy as np

from hw_accel import HWAccelConfig

logger = logging.getLogger(__name__)


def _drain_stderr(process: subprocess.Popen, collected: deque[bytes]) -> None:
    """Daemon thread target: read stderr line-by-line to prevent pipe buffer deadlock.
    Collects bytes (Popen uses text=False). Thread-safe via deque + GIL."""
    try:
        for line in process.stderr:
            collected.append(line)
    except (ValueError, OSError):
        pass  # pipe closed


def _format_stderr(lines: deque[bytes], max_lines: int = 10) -> str:
    """Decode last N stderr lines for error messages."""
    tail = list(lines)[-max_lines:]
    return b"".join(tail).decode("utf-8", errors="replace")[:500]


class FFmpegDecoder:
    """Decode video frames via FFmpeg subprocess pipe.

    Usage:
        with FFmpegDecoder(path, w, h, config) as decoder:
            while (frame := decoder.read_frame()) is not None:
                process(frame)
    """

    def __init__(
        self,
        input_path: str | Path,
        width: int,
        height: int,
        hw_config: HWAccelConfig,
    ):
        self._input_path = str(input_path)
        self._width = width
        self._height = height
        self._frame_size = width * height * 3  # BGR24
        self._stderr_lines: deque[bytes] = deque(maxlen=100)

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
        cmd += hw_config.decode_args
        cmd += ["-i", str(input_path), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

        logger.debug(f"FFmpegDecoder command: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )
        # Start daemon thread to drain stderr and prevent deadlock
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._process, self._stderr_lines), daemon=True
        )
        self._stderr_thread.start()

    def read_frame(self) -> np.ndarray | None:
        """Read one BGR24 frame (writable copy). Returns None on EOF."""
        raw = self._process.stdout.read(self._frame_size)
        if not raw or len(raw) < self._frame_size:
            # Check if process crashed (vs normal EOF)
            if self._process.poll() is not None and self._process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg decoder crashed (rc={self._process.returncode}): "
                    f"{_format_stderr(self._stderr_lines)}"
                )
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        ).copy()  # .copy() makes array writable for OpenCV drawing

    def close(self) -> None:
        if self._process.stdout:
            self._process.stdout.close()
        self._stderr_thread.join(timeout=5)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        if self._process.returncode and self._process.returncode != 0:
            logger.warning(f"FFmpeg decoder exited with code {self._process.returncode}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ffmpeg_pipe.py::TestFFmpegDecoder -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add app/ffmpeg_pipe.py tests/test_ffmpeg_pipe.py
git commit -m "feat: add FFmpegDecoder for pipe-based video decoding (VAS-2)"
```

---

### Task 4: FFmpeg pipe encoder

**Files:**
- Modify: `app/ffmpeg_pipe.py` (add FFmpegEncoder class)
- Modify: `tests/test_ffmpeg_pipe.py` (add encoder tests)

**Step 1: Write the failing tests for FFmpegEncoder**

Append to `tests/test_ffmpeg_pipe.py`:

```python
from ffmpeg_pipe import FFmpegEncoder


class TestFFmpegEncoder:
    def _make_mock_process(self, returncode: int = 0):
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.wait.return_value = returncode
        mock_proc.returncode = returncode
        return mock_proc

    def test_write_frame(self):
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegEncoder(
                output_path="output.mp4",
                original_path="input.mp4",
                width=640, height=480, fps=30.0,
                hw_config=config, codec="h264", crf=18,
            ) as encoder:
                encoder.write_frame(frame)

        mock_proc.stdin.write.assert_called_once()
        written = mock_proc.stdin.write.call_args[0][0]
        assert len(written) == 640 * 480 * 3

    def test_cpu_encode_command(self):
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                output_path="output.mp4",
                original_path="input.mp4",
                width=640, height=480, fps=30.0,
                hw_config=config, codec="h264", crf=18,
            ):
                pass
        cmd = mock_popen.call_args[0][0]
        assert "libx264" in cmd
        assert "pipe:0" in cmd

    def test_nvidia_encode_command(self):
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                output_path="output.mp4",
                original_path="input.mp4",
                width=640, height=480, fps=30.0,
                hw_config=config, codec="h264", crf=18,
            ):
                pass
        cmd = mock_popen.call_args[0][0]
        assert "h264_nvenc" in cmd

    def test_audio_merge_in_command(self):
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                output_path="output.mp4",
                original_path="input.mp4",
                width=640, height=480, fps=30.0,
                hw_config=config, codec="h264", crf=18,
            ):
                pass
        cmd = mock_popen.call_args[0][0]
        # Should have two inputs: pipe:0 for video, input.mp4 for audio
        assert cmd.count("-i") == 2
        assert "input.mp4" in cmd
        assert "-map" in cmd
        assert "aac" in cmd

    def test_cleanup_on_exit(self):
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegEncoder(
                output_path="output.mp4",
                original_path="input.mp4",
                width=640, height=480, fps=30.0,
                hw_config=config, codec="h264", crf=18,
            ):
                pass

        mock_proc.stdin.close.assert_called()
        mock_proc.wait.assert_called()

    def test_nonzero_exit_raises(self):
        mock_proc = self._make_mock_process(returncode=1)
        mock_proc.stderr.read.return_value = "encoding error"
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="FFmpeg encoder failed"):
                with FFmpegEncoder(
                    output_path="output.mp4",
                    original_path="input.mp4",
                    width=640, height=480, fps=30.0,
                    hw_config=config, codec="h264", crf=18,
                ) as encoder:
                    pass  # exit triggers close which checks returncode
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ffmpeg_pipe.py::TestFFmpegEncoder -v`
Expected: FAIL — `FFmpegEncoder` doesn't exist yet.

**Step 3: Write FFmpegEncoder implementation**

Append to `app/ffmpeg_pipe.py`:

```python
class FFmpegEncoder:
    """Encode video frames via FFmpeg subprocess pipe with audio merge.

    Usage:
        with FFmpegEncoder(out, orig, w, h, fps, config, codec, crf) as enc:
            enc.write_frame(frame)
    """

    def __init__(
        self,
        output_path: str | Path,
        original_path: str | Path,
        width: int,
        height: int,
        fps: float,
        hw_config: HWAccelConfig,
        codec: str,
        crf: int,
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
            "-map_metadata", "0",
        ]
        cmd += hw_config.get_encode_args(codec, crf)  # codec-specific args (after inputs)
        cmd += ["-c:a", "aac", "-shortest", str(output_path)]

        logger.debug(f"FFmpegEncoder command: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )
        # Start daemon thread to drain stderr and prevent deadlock
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._process, self._stderr_lines), daemon=True
        )
        self._stderr_thread.start()

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one BGR24 frame. Checks process health before writing."""
        if self._process.poll() is not None:
            raise RuntimeError(
                f"FFmpeg encoder crashed (rc={self._process.returncode}): "
                f"{_format_stderr(self._stderr_lines)}"
            )
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except OSError:
            pass
        self._stderr_thread.join(timeout=10)
        try:
            self._process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        if self._process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoder failed (rc={self._process.returncode}): "
                f"{_format_stderr(self._stderr_lines, max_lines=20) or 'no stderr'}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ffmpeg_pipe.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add app/ffmpeg_pipe.py tests/test_ffmpeg_pipe.py
git commit -m "feat: add FFmpegEncoder for pipe-based video encoding with audio merge (VAS-2)"
```

---

### Task 5: Refactor VideoAnnotator to use FFmpeg pipes

**Files:**
- Modify: `app/video_annotator.py`
- Modify: `tests/test_video_annotator.py`

This is the largest task. We replace cv2.VideoCapture/VideoWriter with FFmpegDecoder/FFmpegEncoder and remove `_merge_audio()`.

**Step 1: Update tests to expect the new pipeline**

The key changes to `tests/test_video_annotator.py`:

1. Remove `TestMergeAudio` class (method deleted).
2. Update `TestAnnotatePipeline` — replace cv2 mocks with ffmpeg_pipe mocks:
   - Instead of mocking `cv2.VideoCapture`, mock `ffmpeg_pipe.FFmpegDecoder`
   - Instead of mocking `cv2.VideoWriter` + `cv2.VideoWriter_fourcc`, mock `ffmpeg_pipe.FFmpegEncoder`
   - The encoder mock should track `write_frame` calls instead of `writer.write`
3. VideoAnnotator constructor now takes `hw_config` parameter.
4. Remove the `test_cannot_open_video` test (cv2.VideoCapture.isOpened no longer used — FFmpeg fails differently).
5. Add a new test for FFmpeg decoder failure (subprocess error).

Replace `TestAnnotatePipeline` with updated tests using this pattern:

```python
from hw_accel import HWAccelConfig, HWAccelType

class TestAnnotatePipeline:
    def _setup_decoder_mock(self, num_frames, width=640, height=480):
        """Create mock FFmpegDecoder that yields N frames then None."""
        frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]
        returns = frames + [None]

        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = returns
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)
        return mock_decoder

    def _setup_encoder_mock(self):
        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)
        return mock_encoder

    def test_full_pipeline(self, mock_model, mock_visualizer, tmp_path):
        num_frames = 6
        detect_every = 3
        mock_decoder = self._setup_decoder_mock(num_frames)
        mock_encoder = self._setup_encoder_mock()
        hw_config = HWAccelConfig(accel_type=HWAccelType.CPU)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        # ffprobe mock for _get_video_metadata
        ffprobe_stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": str(num_frames),
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names,
            hw_config=hw_config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", return_value=mock_decoder),
            patch("video_annotator.FFmpegEncoder", return_value=mock_encoder),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=detect_every),
            )

        assert stats.total_frames == num_frames
        assert stats.detected_frames == 2  # frames 0, 3
        assert stats.tracked_frames == 4   # frames 1, 2, 4, 5
        assert mock_encoder.write_frame.call_count == num_frames
```

Similarly update `test_detect_every_1`, `test_hold_reuses_detections`, `test_hold_clears_on_empty_detection`. Pattern is the same — replace cv2 mocks with decoder/encoder mocks.

Add FFmpeg failure test:

```python
    def test_decoder_failure_propagates(self, mock_model, mock_visualizer, tmp_path):
        hw_config = HWAccelConfig(accel_type=HWAccelType.CPU)
        mock_decoder = MagicMock()
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.read_frame.side_effect = RuntimeError("decode error")
        mock_decoder.__exit__ = MagicMock(return_value=False)

        ffprobe_stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480, "nb_frames": "100",
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config=hw_config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", return_value=mock_decoder),
            patch("video_annotator.FFmpegEncoder", return_value=MagicMock()),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            with pytest.raises(RuntimeError, match="decode error"):
                annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_video_annotator.py::TestAnnotatePipeline -v`
Expected: FAIL — VideoAnnotator doesn't accept hw_config yet, doesn't use FFmpegDecoder/FFmpegEncoder.

**Step 3: Refactor VideoAnnotator**

Changes to `app/video_annotator.py`:

1. Add imports:
```python
from ffmpeg_pipe import FFmpegDecoder, FFmpegEncoder
from hw_accel import HWAccelConfig
```

2. Remove `CODEC_MAP` dict (no longer needed — encode args come from HWAccelConfig).

3. Update constructor to accept `hw_config`:
```python
def __init__(
    self,
    model: Any,
    visualizer: DetectionVisualizer,
    class_names: dict[int, str],
    hw_config: HWAccelConfig,
    codec: str = "h264",
    crf: int = 18,
):
    self.model = model
    self.visualizer = visualizer
    self.class_names = class_names
    self.hw_config = hw_config
    self.codec = codec
    self.crf = crf
```

4. Replace `annotate()` method body — use FFmpegDecoder/FFmpegEncoder:
```python
def annotate(self, input_path, output_path, params, progress_callback=None):
    fps, width, height, total_frames = self._get_video_metadata(input_path)
    # ... logging ...

    stats = AnnotationStats(total_frames=total_frames)
    current_detections = []
    font_scale = self.visualizer.calculate_adaptive_font_scale(height)
    frame_num = 0
    start_time = time.perf_counter()

    with FFmpegDecoder(input_path, width, height, self.hw_config) as decoder, \
         FFmpegEncoder(output_path, input_path, width, height, fps,
                       self.hw_config, self.codec, self.crf) as encoder:

        while True:
            frame = decoder.read_frame()
            if frame is None:
                break

            if frame_num % params.detect_every == 0:
                results = self.model.predict(...)
                current_detections = self._extract_detections(results, params.classes)
                stats.detected_frames += 1
                stats.total_detections += len(current_detections)
            else:
                stats.tracked_frames += 1
                stats.total_detections += len(current_detections)

            self._draw_detections(frame, current_detections, params, font_scale)
            encoder.write_frame(frame)
            frame_num += 1

            if progress_callback and total_frames > 0 and frame_num % 10 == 0:
                progress = int((frame_num / total_frames) * 100)
                progress_callback(min(progress, 99))

    stats.total_frames = frame_num
    stats.processing_time_ms = int((time.perf_counter() - start_time) * 1000)
    # ... logging ...
    return stats
```

5. Remove `_merge_audio()` method entirely.
6. Remove `import shutil` (no longer needed for copy fallback).
7. Remove `import cv2` if no other references remain.

**Step 4: Run all video annotator tests**

Run: `python -m pytest tests/test_video_annotator.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add app/video_annotator.py tests/test_video_annotator.py
git commit -m "refactor: replace cv2 I/O with FFmpeg pipes in VideoAnnotator (VAS-2)"
```

---

### Task 6: Wire hw_config into application startup and annotation worker

**Files:**
- Modify: `app/main.py:78-137` (lifespan) and `app/main.py:170-274` (_annotation_worker)

**Step 1: Write the failing test**

In `tests/test_worker.py`, verify the worker passes hw_config to VideoAnnotator. If `test_worker.py` doesn't test this yet, add:

```python
# Verify VideoAnnotator receives hw_config from settings
```

Check existing test_worker.py first — if it uses mock VideoAnnotator, verify the constructor call includes hw_config.

**Step 2: Update lifespan**

In `app/main.py` lifespan, after job_manager setup:

1. **Verify FFmpeg and ffprobe are installed** (mandatory after this change):
```python
import shutil
if not shutil.which("ffmpeg"):
    raise RuntimeError("FFmpeg is required but not found in PATH")
if not shutil.which("ffprobe"):
    raise RuntimeError("ffprobe is required but not found in PATH")
```

2. **Detect hardware acceleration** passing configured codec and vaapi device:
```python
from hw_accel import detect_hw_accel

hw_config = detect_hw_accel(
    mode=settings.video_hw_accel,
    codec=settings.video_codec,
    vaapi_device=settings.vaapi_device,
)
app.state.hw_config = hw_config
logger.info(f"Video hardware acceleration: {hw_config.accel_type.value}")
```

**Step 3: Update _annotation_worker**

In `_annotation_worker()`, pass `hw_config` to VideoAnnotator:

```python
hw_config = app.state.hw_config

annotator = VideoAnnotator(
    model=model_entry.model,
    visualizer=model_entry.visualizer,
    class_names=model_entry.model.names,
    hw_config=hw_config,
    codec=settings.video_codec,
    crf=settings.video_crf,
)
```

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add app/main.py
git commit -m "feat: wire hw_config into lifespan and annotation worker (VAS-2)"
```

---

### Task 7: Update Docker images for hardware acceleration

**Files:**
- Modify: `docker/amd/Dockerfile` (add VAAPI packages)
- Modify: `docker/docker-compose-nvidia.yml` (add `video` capability)
- `docker/nvidia/Dockerfile` — no changes needed (nvidia-container-toolkit provides libs at runtime)

**Step 1: Update AMD Dockerfile**

Add VAAPI packages to `apt-get install`:

```dockerfile
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    python3-pip \
    ffmpeg gcc g++ \
    libva-dev mesa-va-drivers vainfo \
    && rm -rf /var/lib/apt/lists/*
```

**Step 2: Update NVIDIA docker-compose**

In `docker/docker-compose-nvidia.yml`, add `video` to `NVIDIA_DRIVER_CAPABILITIES` (required for NVENC/NVDEC access):

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

**Step 3: Verify NVIDIA Dockerfile**

Check that the nvidia/cuda base image + nvidia-container-toolkit will provide `libnvidia-encode.so`. No Dockerfile changes expected — the NVENC/NVDEC libs are mounted at runtime by nvidia-container-toolkit.

**Step 4: Commit**

```bash
git add docker/amd/Dockerfile docker/docker-compose-nvidia.yml
git commit -m "build: add VAAPI packages and NVIDIA video capability for hw accel (VAS-2)"
```

---

### Task 8: Update CLAUDE.md and configuration docs

**Files:**
- Modify: `CLAUDE.md` (add `VIDEO_HW_ACCEL` to config table, update architecture table with new modules)

**Step 1: Add new modules to architecture table**

```markdown
| `app/hw_accel.py` | Hardware acceleration detection (NVIDIA/AMD/CPU) |
| `app/ffmpeg_pipe.py` | FFmpeg pipe-based video decoder/encoder |
```

**Step 2: Add config entry**

```markdown
VIDEO_HW_ACCEL=auto                     # auto, nvidia, amd, cpu
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add hw_accel and ffmpeg_pipe to CLAUDE.md (VAS-2)"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Manual smoke test (if Docker/GPU available)**

```bash
# Check hw detection output in logs
LOG_LEVEL=DEBUG python -c "from hw_accel import detect_hw_accel; print(detect_hw_accel('auto'))"
```

**Step 3: Final commit if any fixups needed**

---

## Summary

| Task | Component | New/Modify | Test file |
|------|-----------|------------|-----------|
| 1 | Config: `VIDEO_HW_ACCEL` | Modify `config.py` | `test_config.py` |
| 2 | `hw_accel.py` — detection | Create | `test_hw_accel.py` |
| 3 | `ffmpeg_pipe.py` — decoder | Create | `test_ffmpeg_pipe.py` |
| 4 | `ffmpeg_pipe.py` — encoder | Modify | `test_ffmpeg_pipe.py` |
| 5 | `video_annotator.py` — refactor | Modify | `test_video_annotator.py` |
| 6 | `main.py` — wire hw_config | Modify | `test_worker.py` |
| 7 | Docker — VAAPI packages | Modify Dockerfile | manual |
| 8 | CLAUDE.md docs | Modify | — |
| 9 | Full test pass | — | all |
