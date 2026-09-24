"""VideoAnnotator.annotate() end to end with real ffmpeg on a lavfi clip; YOLO is a stub.

Skipped when ffmpeg/ffprobe are missing. Covers what the mocked tests cannot:
real pipes, the reader and writer threads, yuv420p frames drawn in place,
libx264 encoding and the audio merge.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from detection_stabilizer import StabilizerConfig
from hw_accel import HWAccelConfig, HWAccelType
from video_annotator import AnnotationParams, VideoAnnotator
from visualization import DetectionVisualizer

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)

WIDTH, HEIGHT, FPS, SECONDS = 320, 240, 10, 2
BOX = (40, 60, 200, 180)  # x1, y1, x2, y2 in full-frame pixels
BLUE = (255, 0, 0)  # BGR colour of class 0 in DetectionVisualizer's palette


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


def _make_clip(path: Path, *, audio_seconds: float | None = SECONDS, pix_fmt: str = "yuv420p",
               vfr: bool = False) -> Path:
    """A gray lavfi clip. ``audio_seconds=None`` leaves the audio out; ``vfr``
    keeps every third frame with its original timestamp (0, 0.3, 0.6 s, ...)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r={FPS}:d={SECONDS}"]
    if audio_seconds is not None:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_seconds}"]
    if vfr:
        cmd += ["-vf", "select='not(mod(n\\,3))'", "-fps_mode", "passthrough"]
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
        "full_range": _make_clip(root / "full_range.mp4", pix_fmt="yuvj420p"),
        # -shortest ends the encoder before the last frames are written
        "short_audio": _make_clip(root / "short_audio.mp4", audio_seconds=1),
        "vfr": _make_clip(root / "vfr.mp4", audio_seconds=None, vfr=True),
    }


def _annotator(model: "_StubModel", batch_size: int) -> VideoAnnotator:
    return VideoAnnotator(
        model, DetectionVisualizer(model.names), model.names,
        HWAccelConfig(accel_type=HWAccelType.CPU), codec="h264", crf=18,
        stabilizer_config=StabilizerConfig(), batch_size=batch_size,
    )


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,nb_read_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def _frame(path: Path, index: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        check=True, capture_output=True, timeout=60,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH, 3)


def _assert_box_drawn(frame: np.ndarray) -> None:
    frame = frame.astype(int)
    left_edge = frame[100:140, BOX[0] - 1:BOX[0] + 2].reshape(-1, 3).mean(axis=0)
    assert np.abs(left_edge - BLUE).max() < 50, left_edge  # the box, in its class colour
    assert np.abs(frame[10:30, 260:300] - 128).max() < 12  # gray far from the box


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
    _assert_box_drawn(_frame(output, 5))


def test_audio_shorter_than_video_ends_the_output_early(clips, tmp_path):
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["short_audio"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    written = int(_probe(output)["video"]["nb_read_frames"])
    assert stats.total_frames == FPS * SECONDS
    assert FPS // 2 <= written < stats.total_frames  # cut near the 1 s of audio
    _assert_box_drawn(_frame(output, 2))


def test_variable_frame_rate_keeps_passes_aligned(clips, tmp_path):
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["vfr"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    written = int(_probe(output)["video"]["nb_read_frames"])
    assert written == stats.total_frames
    _assert_box_drawn(_frame(output, written - 1))  # boxes reach the last frame
