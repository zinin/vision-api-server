"""VideoAnnotator.annotate() end to end with real ffmpeg on a lavfi clip; YOLO is a stub.

Skipped when ffmpeg/ffprobe are missing. Covers what the mocked tests cannot:
real pipes, the reader and writer threads, yuv420p frames drawn in place,
libx264 encoding and the audio merge.
"""
import json
import logging
import shutil
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from detection_stabilizer import StabilizerConfig
from hw_accel import HWAccelConfig, HWAccelType
from video_annotator import AnnotationParams, AnnotationStats, VideoAnnotator
from visualization import DetectionVisualizer

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)

WIDTH, HEIGHT, FPS, SECONDS = 320, 240, 10, 2
BOX = (40, 60, 200, 180)  # x1, y1, x2, y2 in full-frame pixels
BLUE = (255, 0, 0)  # BGR colour of class 0 in DetectionVisualizer's palette
# Video length of the short-audio clip. ffmpeg keeps taking frames until it acts on
# the end of the audio: about 40 on an idle host, up to about 250 on two CPUs.
LONG_SECONDS = 60
# Source frames the vfr clip keeps, at their original times: gaps of 0.1 s to 0.8 s.
VFR_FRAMES = (0, 1, 2, 6, 7, 15, 19)


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, box, class_id, conf):
        self.xyxy = _Tensor([box])
        self.cls = _Tensor([class_id])
        self.conf = _Tensor([conf])

    def __len__(self):
        return 1


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _StubModel:
    """Answers every frame with BOX, expressed in the coordinates of the frame it got."""

    names = {0: "person"}

    def __init__(self):
        self.frame_shapes = []

    def predict(self, source, **kwargs):
        results = []
        for frame in source:
            self.frame_shapes.append(frame.shape)
            sx, sy = frame.shape[1] / WIDTH, frame.shape[0] / HEIGHT
            box = (BOX[0] * sx, BOX[1] * sy, BOX[2] * sx, BOX[3] * sy)
            results.append(_Result(_Boxes(box, 0, 0.9)))
        return results


def _make_clip(path: Path, *, seconds: int = SECONDS, audio_seconds: float | None = SECONDS,
               pix_fmt: str = "yuv420p", vfr: bool = False, ramp: bool = False,
               jitter: bool = False) -> Path:
    """A gray lavfi clip. ``audio_seconds=None`` leaves the audio out; ``vfr``
    keeps only VFR_FRAMES, each with its original timestamp (0, 0.1, 0.2, 0.6 s, ...);
    ``ramp`` replaces the gray with a horizontal luma ramp from 0 to 255; ``jitter``
    stamps frame 1 80 ms early, as a camera's clock may, so that ffmpeg reads
    r_frame_rate as 50 while the clip keeps its FPS frames a second."""
    if ramp:
        # ffmpeg treats gray as full range, so a full-range pix_fmt stores the ramp unchanged
        video = f"nullsrc=s={WIDTH}x{HEIGHT}:r={FPS}:d={seconds},format=gray,geq=lum='X*255/{WIDTH - 1}'"
    else:
        video = f"color=c=gray:s={WIDTH}x{HEIGHT}:r={FPS}:d={seconds}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", video]
    if audio_seconds is not None:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_seconds}"]
    if vfr:
        keep = "+".join(f"eq(n\\,{n})" for n in VFR_FRAMES)
        cmd += ["-vf", f"select='{keep}'", "-fps_mode", "passthrough"]
    if jitter:
        # millisecond time bases, or the encoder rounds the early stamp back onto the grid
        cmd += ["-vf", f"settb=1/1000,setpts='(N/{FPS} - if(eq(N\\,1)\\,0.08\\,0))/TB'",
                "-fps_mode", "passthrough", "-enc_time_base", "1/1000", "-video_track_timescale", "1000"]
    cmd += ["-c:v", "libx264", "-pix_fmt", pix_fmt]
    if audio_seconds is not None:
        cmd += ["-c:a", "aac"]
    subprocess.run(cmd + [str(path)], check=True, timeout=120)
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("annotate")
    return {
        "audio": _make_clip(root / "audio.mp4"),
        "no_audio": _make_clip(root / "no_audio.mp4", audio_seconds=None),
        # IP cameras often record full-range yuvj420p; pass 2 asks ffmpeg for yuv420p
        "full_range": _make_clip(root / "full_range.mp4", pix_fmt="yuvj420p", ramp=True),
        # -shortest ends the encoder and closes its pipe while frames are still coming
        "short_audio": _make_clip(root / "short_audio.mp4", seconds=LONG_SECONDS, audio_seconds=1),
        "vfr": _make_clip(root / "vfr.mp4", audio_seconds=None, vfr=True),
        # r_frame_rate reads as 50; the audio outlasts the video, so -shortest keeps every frame
        "jitter": _make_clip(root / "jitter.mp4", audio_seconds=SECONDS + 1, jitter=True),
    }


def _annotator(model: "_StubModel", batch_size: int) -> VideoAnnotator:
    return VideoAnnotator(
        model, DetectionVisualizer(model.names), model.names,
        HWAccelConfig(accel_type=HWAccelType.CPU), codec="h264", crf=18,
        stabilizer_config=StabilizerConfig(), batch_size=batch_size,
    )


def _annotate_within(annotator: VideoAnnotator, clip: Path, output: Path,
                     params: AnnotationParams, timeout: float = 120.0) -> AnnotationStats:
    """annotate() on a daemon thread, so that a hang fails the test instead of blocking the suite."""
    outcome: dict = {}

    def run() -> None:
        try:
            outcome["stats"] = annotator.annotate(clip, output, params)
        except BaseException as exc:  # re-raised on the test's thread
            outcome["error"] = exc

    thread = threading.Thread(target=run, name="annotate", daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"annotate() still running after {timeout:.0f} s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["stats"]


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,nb_read_frames,r_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def _frame(path: Path, index: int, pix_fmt: str = "bgr24") -> np.ndarray:
    """Frame ``index`` as ffmpeg decodes it: BGR, or with ``pix_fmt="gray"``
    its luma alone, full range whatever the range of the video."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"],
        check=True, capture_output=True, timeout=60,
    ).stdout
    shape = (HEIGHT, WIDTH) if pix_fmt == "gray" else (HEIGHT, WIDTH, 3)
    return np.frombuffer(raw, np.uint8).reshape(shape)


def _assert_box_colour(frame: np.ndarray) -> None:
    left_edge = frame[100:140, BOX[0] - 1:BOX[0] + 2].astype(int).reshape(-1, 3).mean(axis=0)
    assert np.abs(left_edge - BLUE).max() < 50, left_edge  # the box, in its class colour


def _assert_box_drawn(frame: np.ndarray) -> None:
    _assert_box_colour(frame)
    assert np.abs(frame[10:30, 260:300].astype(int) - 128).max() < 12  # gray far from the box


@pytest.mark.parametrize("batch_size,imgsz", [(1, 320), (4, 160)])
def test_annotate_draws_boxes_and_keeps_every_frame_and_the_audio(clips, tmp_path, batch_size, imgsz):
    model = _StubModel()
    output = tmp_path / "annotated.mp4"
    stats = _annotator(model, batch_size).annotate(
        clips["audio"], output, AnnotationParams(conf=0.5, imgsz=imgsz, detect_every=1, line_width=6),
    )

    source, result = _probe(clips["audio"]), _probe(output)
    assert stats.total_frames == int(source["video"]["nb_read_frames"]) == FPS * SECONDS
    assert int(result["video"]["nb_read_frames"]) == stats.total_frames
    assert "audio" in result
    # imgsz 160 halves the 320x240 frames before YOLO sees them
    assert set(model.frame_shapes) == {(imgsz * HEIGHT // WIDTH, imgsz, 3)}
    _assert_box_drawn(_frame(output, 10))


def test_clip_without_audio(clips, tmp_path):
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["no_audio"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    result = _probe(output)
    assert "audio" not in result
    assert int(result["video"]["nb_read_frames"]) == stats.total_frames
    _assert_box_drawn(_frame(output, 5))


def test_full_range_source(clips, tmp_path):
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["full_range"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    assert int(_probe(output)["video"]["nb_read_frames"]) == stats.total_frames
    _assert_box_colour(_frame(output, 5))
    # Below the box the output shows the source's ramp at its brightness: full-range
    # levels passed on as limited range would clip the dark and bright ends of it.
    below_box = slice(BOX[3] + 20, HEIGHT)
    source = _frame(clips["full_range"], 5, pix_fmt="gray")[below_box].astype(int)
    result = _frame(output, 5, pix_fmt="gray")[below_box].astype(int)
    assert source.min() < 5 and source.max() > 250  # the ramp covers the whole range
    diff = np.abs(result - source)
    assert diff.max() <= 3, f"{diff.max()} levels off, in {(diff.max(axis=0) > 3).sum()} of {WIDTH} columns"


def test_audio_shorter_than_video_ends_the_output_early(clips, tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="ffmpeg_pipe")
    output = tmp_path / "annotated.mp4"
    stats = _annotate_within(
        _annotator(_StubModel(), 4), clips["short_audio"], output,
        AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    written = int(_probe(output)["video"]["nb_read_frames"])
    assert stats.total_frames == FPS * LONG_SECONDS
    assert FPS // 2 <= written < stats.total_frames  # cut after the audio, long before the video ends
    # the encoder exited while frames were still coming; both early-EOF branches log this
    assert "FFmpegEncoder: clean exit" in caplog.text
    _assert_box_drawn(_frame(output, 2))


def test_variable_frame_rate_keeps_passes_aligned(clips, tmp_path):
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["vfr"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    written = int(_probe(output)["video"]["nb_read_frames"])
    # the pipe decoders resample the irregular frames to a constant rate: a pass that
    # decoded them as stored (-fps_mode passthrough) would count differently
    assert stats.total_frames != int(_probe(clips["vfr"])["video"]["nb_read_frames"])
    assert written == stats.total_frames
    _assert_box_drawn(_frame(output, written - 1))  # boxes reach the last frame


def test_misread_frame_rate_decodes_each_frame_once(clips, tmp_path):
    """With r_frame_rate misread as 50, decoders left to pick their own rate made five
    frames of each, YOLO ran on all of them, and the encoder, playing them at the real
    10 fps, cut the slow-motion result at the end of the audio."""
    source = _probe(clips["jitter"])
    assert Fraction(source["video"]["r_frame_rate"]) > 2 * FPS  # the clip is what it claims
    stored = int(source["video"]["nb_read_frames"])
    assert stored == FPS * SECONDS
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["jitter"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    result = _probe(output)
    # ffmpeg 8 repeats the last of these millisecond-stamped frames once where 6.1 stops
    assert stored <= stats.total_frames <= stored + 1
    assert int(result["video"]["nb_read_frames"]) == stats.total_frames  # nothing cut
    assert "audio" in result
    _assert_box_drawn(_frame(output, stats.total_frames - 1))
