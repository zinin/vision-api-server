"""Integration tests on synthetic lavfi clips. Skipped when ffmpeg/ffprobe are missing."""
import shutil
import subprocess
from pathlib import Path

import pytest

from frame_selection import SelectionParams
from video_utils import VideoFrameExtractor

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
def extractor():
    return VideoFrameExtractor()


def _by_reason(selected, reason):
    return [f for f in selected if f.reason == reason]


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
