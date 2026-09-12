"""Integration tests on synthetic lavfi clips. Skipped when ffmpeg/ffprobe are missing."""
import asyncio
import base64
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import main
import video_utils
from config import Settings, get_settings
from dependencies import get_job_manager, get_model_manager
from frame_selection import SelectionParams
from job_manager import JobManager
from main import app
from video_utils import VideoFrameExtractor, VideoInfo, extract_frames_from_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)

STATIC_SRC = "color=c=gray:s=320x240:r=10:d=10"
# a 30×30 white box slides right at 60 px/s while t in [1, 3]; overlay evaluates x per frame
MOTION_SRC = (
    "color=c=black:s=320x240:r=10:d=10[bg];"
    "color=c=white:s=30x30:r=10:d=10[box];"
    "[bg][box]overlay=x='20+60*t':y=100:eval=frame:enable='between(t,1,3)'"
)
# per-pixel random noise on every frame: a storm for the metric
STORM_SRC = "nullsrc=s=320x240:r=10:d=10,geq=lum='random(1)*255':cb=128:cr=128"
LONG_SRC = "color=c=gray:s=320x240:r=10:d=40"
# a saturated colour: the only clip that tells BGR from RGB
RED_SRC = "color=c=red:s=320x240:r=10:d=3"
# the moving box of MOTION_SRC on a 40 s recording: the grid alone fills the budget
LONG_MOTION_SRC = (
    "color=c=black:s=320x240:r=10:d=40[bg];"
    "color=c=white:s=30x30:r=10:d=40[box];"
    "[bg][box]overlay=x='20+60*t':y=100:eval=frame:enable='between(t,1,3)'"
)
# every third frame of a 10 fps source, timestamps kept: irregular pts 0, 0.3, 0.6, …
VFR_SRC = "testsrc=size=320x240:rate=10:duration=10"


def _make_clip(path: Path, source: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", source,
         "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=120,
    )
    return path


@pytest.fixture(scope="module")
def clips_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("clips")


@pytest.fixture(scope="module")
def static_clip(clips_dir):
    return _make_clip(clips_dir / "static.mp4", STATIC_SRC)


@pytest.fixture(scope="module")
def motion_clip(clips_dir):
    return _make_clip(clips_dir / "motion.mp4", MOTION_SRC)


@pytest.fixture(scope="module")
def storm_clip(clips_dir):
    return _make_clip(clips_dir / "storm.mp4", STORM_SRC)


@pytest.fixture(scope="module")
def long_clip(clips_dir):
    return _make_clip(clips_dir / "long.mp4", LONG_SRC)


@pytest.fixture(scope="module")
def red_clip(clips_dir):
    return _make_clip(clips_dir / "red.mp4", RED_SRC)


@pytest.fixture(scope="module")
def long_motion_clip(clips_dir):
    return _make_clip(clips_dir / "long_motion.mp4", LONG_MOTION_SRC)


@pytest.fixture(scope="module")
def vfr_clip(clips_dir):
    """Variable frame rate: ``select`` drops two frames out of three and
    ``-fps_mode passthrough`` keeps the original timestamps."""
    out = clips_dir / "vfr.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", VFR_SRC,
         "-vf", "select='not(mod(n\\,3))'", "-fps_mode", "passthrough",
         "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(out)],
        check=True, timeout=120,
    )
    return out


@pytest.fixture(scope="module")
def multi_track_clip(clips_dir):
    """Two video tracks of different sizes; the first one is what ffprobe reports."""
    out = clips_dir / "multi_track.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=gray:s=160x120:r=10:d=1",
         "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=10:d=1",
         "-map", "0:v", "-map", "1:v",
         "-disposition:v:0", "0", "-disposition:v:1", "default",
         "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(out)],
        check=True, timeout=120,
    )
    return out


@pytest.fixture(scope="module")
def rotated_clip(clips_dir, static_clip):
    """The static clip with a 90° display-rotation matrix, stream-copied."""
    out = clips_dir / "rotated.mp4"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90",
         "-i", str(static_clip), "-c", "copy", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg lacks -display_rotation: {result.stderr.strip()[:200]}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "v:0", str(out)],
        capture_output=True, text=True, check=True,
    )
    if '"rotation"' not in probe.stdout:
        pytest.skip("ffmpeg did not write rotation metadata")
    return out


@pytest.fixture(scope="module")
def extractor():
    return VideoFrameExtractor()


@asynccontextmanager
async def _noop_lifespan(app: FastAPI):
    yield


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    entry = MagicMock()
    entry.model.names = {0: "person"}
    entry.model.predict.return_value = []  # no detections on synthetic clips
    entry.model_name = "yolo26n.pt"
    mm.get_model = AsyncMock(return_value=entry)
    mm._preloaded = {"yolo26n.pt": entry}
    mm._cached = {}
    return mm


@pytest.fixture
def client(tmp_path, mock_model_manager):
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    app.dependency_overrides[get_settings] = lambda: Settings(yolo_models="{}", video_jobs_dir=str(tmp_path))
    app.dependency_overrides[get_job_manager] = lambda: JobManager(
        jobs_dir=str(tmp_path), ttl_seconds=3600, max_queued=10
    )
    app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.router.lifespan_context = original_lifespan
        app.dependency_overrides.clear()


def _upload(path: Path):
    return {"file": (path.name, path.read_bytes(), "video/mp4")}


def _by_reason(selected, reason):
    return [f for f in selected if f.reason == reason]


def _recording_extractor(recorded: dict):
    """Stand-in for extract_frames_from_video: records its keyword arguments, then fails."""
    async def _extract(**kwargs):
        recorded.update(kwargs)
        raise RuntimeError("boom")

    return _extract


class TestScan:
    def test_static_clip_first_and_grid(self, extractor, static_clip):
        result = extractor.scan(str(static_clip), SelectionParams())

        assert [f.index for f in result.selected] == [0, 40, 80]
        assert [f.reason for f in result.selected] == ["first", "grid", "grid"]
        assert result.stats.total_frames == 100
        assert result.stats.storm is False
        assert result.stats.counts == {"first": 1, "grid": 2}
        assert result.stats.pass1_seconds > 0
        assert result.pts[40] == pytest.approx(4.0, abs=0.11)
        assert len(result.pts) == len(result.blob) == 100
        assert max(result.blob) == 0.0
        assert result.info.duration == pytest.approx(10.0, abs=0.1)
        assert (result.info.width, result.info.height) == (320, 240)

    def test_motion_clip_adds_motion_frames(self, extractor, motion_clip):
        result = extractor.scan(str(motion_clip), SelectionParams())

        motion = _by_reason(result.selected, "motion")
        assert 1 <= len(motion) <= 3, result.selected
        times = [result.pts[f.index] for f in motion]
        assert all(0.9 <= t <= 3.2 for t in times), times
        for earlier, later in zip(times, times[1:]):
            assert later - earlier >= 1.0 - 1e-3
        assert [f.index for f in result.selected if f.reason != "motion"] == [0, 40, 80]
        assert len(result.selected) <= 6
        assert result.stats.counts["motion"] == len(motion)

    def test_storm_clip_grid_only(self, extractor, storm_clip):
        result = extractor.scan(str(storm_clip), SelectionParams())

        assert result.stats.storm is True
        assert result.stats.median_blob > 0.02
        assert [f.reason for f in result.selected] == ["first", "grid", "grid"]

    def test_long_clip_thins_grid(self, extractor, long_clip):
        result = extractor.scan(str(long_clip), SelectionParams())

        assert len(result.selected) == 6
        assert result.selected[0].index == 0
        assert result.pts[result.selected[-1].index] == pytest.approx(36.0, abs=0.11)
        assert result.stats.total_frames == 400

    def test_audio_only_file_raises_value_error(self, extractor, tmp_path):
        audio = tmp_path / "audio.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)],
            check=True, timeout=60,
        )
        with pytest.raises(ValueError, match="no video stream"):
            extractor.scan(str(audio), SelectionParams())

    def test_garbage_file_raises_value_error(self, extractor, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a video at all")
        with pytest.raises(ValueError, match="could not be read"):
            extractor.scan(str(bad), SelectionParams())

    def test_expired_deadline_raises_runtime_error(self, long_clip):
        extractor = VideoFrameExtractor(timeout=0.0)
        with pytest.raises(RuntimeError, match="timed out"):
            extractor.scan(str(long_clip), SelectionParams())


class TestExtractFrames:
    def test_static_clip_frames(self, extractor, static_clip):
        result = extractor.extract_frames(str(static_clip), SelectionParams())

        assert [f.frame_number for f in result.frames] == [0, 40, 80]
        assert [f.reason for f in result.frames] == ["first", "grid", "grid"]
        assert [round(f.timestamp, 1) for f in result.frames] == [0.0, 4.0, 8.0]
        for frame in result.frames:
            assert frame.image.shape == (240, 320, 3)
            assert frame.image.dtype == np.uint8
            assert frame.image.flags.writeable
        assert result.info.duration == pytest.approx(10.0, abs=0.1)
        assert result.stats.pass2_seconds > 0
        assert result.stats.counts == {"first": 1, "grid": 2}

    def test_frames_are_the_selected_ones(self, extractor, motion_clip):
        """The white box is on screen only for t in [1, 3]: motion frames inside
        that window contain white pixels, frame 0 is black."""
        result = extractor.extract_frames(str(motion_clip), SelectionParams())

        by_number = {f.frame_number: f for f in result.frames}
        assert by_number[0].image.max() < 60
        visible = [f for f in result.frames if f.reason == "motion" and 1.0 <= f.timestamp <= 3.0]
        assert visible
        assert all(f.image.max() > 200 for f in visible)

    def test_rotated_clip_uses_display_dimensions(self, extractor, rotated_clip):
        info = extractor.get_video_info(str(rotated_clip))
        assert (info.width, info.height, info.rotation) == (240, 320, 90)

        result = extractor.extract_frames(str(rotated_clip), SelectionParams())
        assert result.frames[0].image.shape == (320, 240, 3)

    def test_red_clip_frame_is_bgr(self, extractor, red_clip):
        """Channel order is BGR the whole way out, as YOLO and cv2 expect. A red
        source frame must come back blue-low/red-high; a cvtColor(RGB2BGR) anywhere
        in the path would swap them, and every other fixture clip is gray."""
        result = extractor.extract_frames(str(red_clip), SelectionParams())

        blue, green, red = (int(v) for v in result.frames[0].image[120, 160])
        assert red > 180, (blue, green, red)
        assert blue < 60 and green < 60, (blue, green, red)

    def test_max_frames_two_keeps_first_and_last_grid_frame(self, extractor, static_clip):
        result = extractor.extract_frames(str(static_clip), SelectionParams(max_frames=2))
        assert [f.frame_number for f in result.frames] == [0, 80]

    def test_async_wrapper_returns_result(self, static_clip):
        result = asyncio.run(extract_frames_from_video(static_clip.read_bytes(), SelectionParams(max_frames=2)))
        assert [f.frame_number for f in result.frames] == [0, 80]
        assert result.info.duration == pytest.approx(10.0, abs=0.1)

    def test_vfr_clip_timestamps_are_the_scanned_pts(self, extractor, vfr_clip):
        """Irregular timestamps: the frames come back with the pts of pass 1,
        never a frame index times a nominal frame rate."""
        scan = extractor.scan(str(vfr_clip), SelectionParams())
        result = extractor.extract_frames(str(vfr_clip), SelectionParams())

        assert result.frames
        assert [f.frame_number for f in result.frames] == [f.index for f in scan.selected]
        assert [f.timestamp for f in result.frames] == [scan.pts[f.frame_number] for f in result.frames]

    def test_multi_track_clip_decodes_the_probed_stream(self, extractor, multi_track_clip):
        """ffprobe reads v:0, so ffmpeg must decode v:0 too — not the default-disposition track."""
        result = extractor.extract_frames(str(multi_track_clip), SelectionParams())

        assert result.frames
        for frame in result.frames:
            assert frame.image.shape == (120, 160, 3)

    def test_long_motion_clip_keeps_one_motion_frame(self, extractor, long_motion_clip):
        """40 s of grid would fill the budget on its own; the held-back slot still
        buys the strongest peak, which lies inside the [1, 3] s movement window."""
        result = extractor.extract_frames(str(long_motion_clip), SelectionParams())

        assert len(result.frames) == 6
        assert result.stats.counts["motion"] == 1
        motion = [f for f in result.frames if f.reason == "motion"]
        assert 1.0 <= motion[0].timestamp <= 3.0, motion[0].timestamp

    def test_frame_grab_mismatch_raises_runtime_error(self, extractor, static_clip, monkeypatch):
        """If the second pass returns frames that do not match the scan, fail loudly."""
        scan = extractor.scan(str(static_clip), SelectionParams())
        wrong_pts = [t + 0.5 for t in scan.pts]  # every pts off by half a second
        with pytest.raises(RuntimeError, match="does not match"):
            extractor._grab_frames(str(static_clip), scan.info, scan.selected, wrong_pts,
                                   deadline=time.monotonic() + extractor.timeout)


class TestPipelineGuards:
    """The failure branches of both passes, exercised on real ffmpeg output."""

    def test_wrong_frame_dimensions_raise_runtime_error(self, extractor, static_clip):
        scan = extractor.scan(str(static_clip), SelectionParams())
        swapped = VideoInfo(
            duration=scan.info.duration, width=scan.info.height, height=scan.info.width,
            fps=scan.info.fps, codec=scan.info.codec,
        )
        with pytest.raises(RuntimeError, match="differs from the expected"):
            extractor._grab_frames(str(static_clip), swapped, scan.selected, scan.pts,
                                   deadline=time.monotonic() + extractor.timeout)

    def test_expired_deadline_in_pass_two_raises_runtime_error(self, extractor, static_clip):
        """The deadline is enforced while the frames are streamed, not by a communicate() timeout."""
        scan = extractor.scan(str(static_clip), SelectionParams())
        with pytest.raises(RuntimeError, match="timed out"):
            extractor._grab_frames(str(static_clip), scan.info, scan.selected, scan.pts,
                                   deadline=time.monotonic() - 1.0)

    def test_missing_showinfo_lines_truncate_the_scan(self, extractor, static_clip, monkeypatch):
        """Fewer timestamps than decoded frames: both lists are cut to the shorter one."""
        real_parse = video_utils.parse_showinfo_line
        kept: list[int] = []

        def every_other_line(line):
            parsed = real_parse(line)
            if parsed is None:
                return None
            kept.append(len(kept))
            return parsed if len(kept) % 2 else None

        info = extractor.get_video_info(str(static_clip))
        monkeypatch.setattr(video_utils, "parse_showinfo_line", every_other_line)
        pts, blob = extractor._scan_motion(str(static_clip), info,
                                           deadline=time.monotonic() + extractor.timeout)

        assert len(pts) == len(blob) == (len(kept) + 1) // 2

    def test_no_showinfo_lines_raise_runtime_error(self, extractor, static_clip, monkeypatch):
        info = extractor.get_video_info(str(static_clip))
        monkeypatch.setattr(video_utils, "parse_showinfo_line", lambda line: None)

        with pytest.raises(RuntimeError, match="without showinfo timestamps"):
            extractor._scan_motion(str(static_clip), info,
                                   deadline=time.monotonic() + extractor.timeout)


class TestExtractFramesEndpoint:
    def test_static_clip_response(self, client, static_clip):
        response = client.post("/extract/frames", files=_upload(static_clip))

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True
        assert body["video_duration"] == pytest.approx(10.0, abs=0.1)
        assert body["video_resolution"] == [320, 240]
        assert body["frames_extracted"] == 3
        assert [f["frame_number"] for f in body["frames"]] == [0, 40, 80]
        assert [f["reason"] for f in body["frames"]] == ["first", "grid", "grid"]
        assert [round(f["timestamp"], 1) for f in body["frames"]] == [0.0, 4.0, 8.0]
        first = body["frames"][0]
        assert (first["width"], first["height"]) == (320, 240)
        jpeg = base64.b64decode(first["image_base64"])
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (240, 320, 3)

    def test_red_clip_jpeg_is_bgr(self, client, red_clip):
        """The same channel-order pin through the endpoint: cv2.imencode is handed
        BGR, so decoding the returned JPEG must give back a red pixel."""
        response = client.post("/extract/frames", files=_upload(red_clip))

        assert response.status_code == 200, response.text
        jpeg = base64.b64decode(response.json()["frames"][0]["image_base64"])
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        blue, green, red = (int(v) for v in image[120, 160])
        assert red > 180, (blue, green, red)
        assert blue < 60 and green < 60, (blue, green, red)

    def test_legacy_client_query_still_works(self, client, static_clip):
        """frigate-analyzer sends scene_threshold until it is updated; FastAPI ignores it."""
        response = client.post(
            "/extract/frames?scene_threshold=0.05&min_interval=1.0&max_frames=50&quality=85",
            files=_upload(static_clip),
        )
        assert response.status_code == 200, response.text
        assert response.json()["frames_extracted"] == 3

    def test_max_frames_caps_output(self, client, static_clip):
        response = client.post("/extract/frames?max_frames=2", files=_upload(static_clip))
        assert response.status_code == 200, response.text
        assert [f["frame_number"] for f in response.json()["frames"]] == [0, 80]

    def test_max_gap_changes_grid(self, client, static_clip):
        response = client.post("/extract/frames?max_gap=2", files=_upload(static_clip))
        assert response.status_code == 200, response.text
        assert [f["frame_number"] for f in response.json()["frames"]] == [0, 20, 40, 60, 80]

    def test_out_of_range_max_gap_rejected(self, client, static_clip):
        response = client.post("/extract/frames?max_gap=0.1", files=_upload(static_clip))
        assert response.status_code == 422

    def test_extraction_failure_returns_500(self, client, static_clip, monkeypatch):
        monkeypatch.setattr(main, "extract_frames_from_video", AsyncMock(side_effect=RuntimeError("boom")))

        response = client.post("/extract/frames", files=_upload(static_clip))

        assert response.status_code == 500
        assert response.json()["detail"].startswith("Failed to extract frames")

    def test_configured_timeout_reaches_the_extractor(self, client, tmp_path, static_clip, monkeypatch):
        recorded: dict = {}
        app.dependency_overrides[get_settings] = lambda: Settings(
            yolo_models="{}", video_jobs_dir=str(tmp_path), video_extract_timeout=10.0
        )
        monkeypatch.setattr(main, "extract_frames_from_video", _recording_extractor(recorded))

        response = client.post("/extract/frames", files=_upload(static_clip))

        assert response.status_code == 500
        assert recorded["timeout"] == 10.0

    def test_audio_only_file_rejected_with_422(self, client, tmp_path):
        audio = tmp_path / "audio.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)],
            check=True, timeout=60,
        )
        response = client.post("/extract/frames", files=_upload(audio))
        assert response.status_code == 422
        assert "no video stream" in response.json()["detail"]


class TestDetectVideoEndpoint:
    def test_static_clip_response(self, client, static_clip):
        response = client.post("/detect/video", files=_upload(static_clip))

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["frames_analyzed"] == 3
        assert body["video_duration"] == pytest.approx(10.0, abs=0.1)
        assert body["video_resolution"] == [320, 240]
        assert [f["frame_number"] for f in body["frames"]] == [0, 40, 80]
        assert [f["reason"] for f in body["frames"]] == ["first", "grid", "grid"]
        assert body["total_detections"] == 0
        assert body["model"] == "yolo26n.pt"

    def test_selection_params_are_passed_through(self, client, static_clip):
        response = client.post("/detect/video?max_gap=2&max_frames=3", files=_upload(static_clip))
        assert response.status_code == 200, response.text
        assert [f["frame_number"] for f in response.json()["frames"]] == [0, 40, 80]

    def test_extraction_failure_returns_500(self, client, static_clip, monkeypatch):
        monkeypatch.setattr(main, "extract_frames_from_video", AsyncMock(side_effect=RuntimeError("boom")))

        response = client.post("/detect/video", files=_upload(static_clip))

        assert response.status_code == 500
        assert response.json()["detail"].startswith("Failed to extract frames")

    def test_configured_timeout_reaches_the_extractor(self, client, tmp_path, static_clip, monkeypatch):
        recorded: dict = {}
        app.dependency_overrides[get_settings] = lambda: Settings(
            yolo_models="{}", video_jobs_dir=str(tmp_path), video_extract_timeout=10.0
        )
        monkeypatch.setattr(main, "extract_frames_from_video", _recording_extractor(recorded))

        response = client.post("/detect/video", files=_upload(static_clip))

        assert response.status_code == 500
        assert recorded["timeout"] == 10.0
