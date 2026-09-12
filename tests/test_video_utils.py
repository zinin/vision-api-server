import asyncio
import json
import os
import subprocess
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

import video_utils
from frame_selection import SelectionParams
from video_utils import (
    ShowinfoFrame,
    VideoFrameExtractor,
    VideoInfo,
    extract_frames_from_video,
    parse_probe_output,
    parse_showinfo,
    parse_showinfo_line,
)


def _probe_run(payload: dict) -> MagicMock:
    return MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")


class TestExtractFramesNoVideoStream:
    """Verify that files without a video stream raise ValueError early: the check
    lives in ``get_video_info``, so the probe output is what the test drives."""

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch("video_utils.subprocess.run")
    def test_audio_only_file_raises_value_error(self, mock_run, mock_verify):
        mock_run.return_value = _probe_run({"streams": [], "format": {"duration": "0.15"}})

        extractor = VideoFrameExtractor()
        with pytest.raises(ValueError, match="no video stream"):
            extractor.extract_frames("/tmp/fake.mp4", SelectionParams())

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch("video_utils.subprocess.run")
    def test_zero_width_raises_value_error(self, mock_run, mock_verify):
        mock_run.return_value = _probe_run({
            "streams": [{"codec_name": "h264", "width": 0, "height": 720, "avg_frame_rate": "30/1"}],
            "format": {"duration": "10.0"},
        })

        extractor = VideoFrameExtractor()
        with pytest.raises(ValueError, match="no video stream"):
            extractor.extract_frames("/tmp/fake.mp4", SelectionParams())

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch("video_utils.subprocess.run")
    def test_zero_height_raises_value_error(self, mock_run, mock_verify):
        mock_run.return_value = _probe_run({
            "streams": [{"codec_name": "h264", "width": 1280, "height": 0, "avg_frame_rate": "30/1"}],
            "format": {"duration": "10.0"},
        })

        extractor = VideoFrameExtractor()
        with pytest.raises(ValueError, match="no video stream"):
            extractor.extract_frames("/tmp/fake.mp4", SelectionParams())


class TestExtractFramesFromVideoTempFile:
    def test_temp_file_is_removed_when_the_write_fails(self, tmp_path, monkeypatch):
        """A failing write (ENOSPC/EIO) must not leak the delete=False temp file."""
        created: list[str] = []
        real_named_temp_file = tempfile.NamedTemporaryFile

        def failing_temp_file(*args, **kwargs):
            tmp = real_named_temp_file(*args, **{**kwargs, "dir": str(tmp_path)})
            created.append(tmp.name)

            def _write(_data):
                raise OSError("disk full")

            tmp.write = _write
            return tmp

        monkeypatch.setattr(video_utils.tempfile, "NamedTemporaryFile", failing_temp_file)

        with pytest.raises(OSError, match="disk full"):
            asyncio.run(extract_frames_from_video(b"x", SelectionParams()))

        assert created
        assert not os.path.exists(created[0])


class TestScanMotionGuards:
    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    def test_degenerate_aspect_ratio_raises_value_error(self, mock_verify):
        """Wider than 640:1 scales to zero height: fail before ffmpeg is started,
        otherwise the read loop never sees EOF. A property of the file, so ValueError
        (422), not RuntimeError (500) — the client must not retry it."""
        info = VideoInfo(duration=1.0, width=1300, height=1, fps=10.0, codec="h264")

        with pytest.raises(ValueError, match="zero-height"):
            VideoFrameExtractor()._scan_motion(
                "/tmp/fake.mp4", info, deadline=time.monotonic() + 1.0
            )


def _probe(stream: dict, fmt: dict | None = None) -> dict:
    return {"streams": [stream], "format": fmt if fmt is not None else {}}


class TestParseProbeOutput:
    def test_small_long_video(self):
        info = parse_probe_output(_probe(
            {"codec_name": "h264", "width": 64, "height": 48, "avg_frame_rate": "25/1",
             "r_frame_rate": "25/1", "duration": "150.000000"},
            {"duration": "150.000000"},
        ))
        assert (info.width, info.height, info.duration) == (64, 48, 150.0)
        assert (info.fps, info.codec, info.rotation) == (25.0, "h264", 0)

    def test_codec_with_digits(self):
        info = parse_probe_output(_probe(
            {"codec_name": "av1", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001", "duration": "1.0"}
        ))
        assert info.codec == "av1"
        assert info.fps == pytest.approx(29.97, abs=0.01)

    def test_rotation_swaps_dimensions(self):
        info = parse_probe_output(_probe(
            {"codec_name": "hevc", "width": 320, "height": 240, "avg_frame_rate": "10/1",
             "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]},
            {"duration": "10.0"},
        ))
        assert (info.width, info.height, info.rotation) == (240, 320, 90)

    def test_rotation_from_tags(self):
        info = parse_probe_output(_probe({"width": 320, "height": 240, "tags": {"rotate": "270"}}))
        assert (info.width, info.height, info.rotation) == (240, 320, 270)

    def test_rotation_180_keeps_dimensions(self):
        info = parse_probe_output(_probe({"width": 320, "height": 240, "side_data_list": [{"rotation": 180}]}))
        assert (info.width, info.height, info.rotation) == (320, 240, 180)

    def test_missing_duration_is_zero(self):
        info = parse_probe_output(_probe({"width": 320, "height": 240}))
        assert info.duration == 0.0

    def test_stream_duration_fallback(self):
        info = parse_probe_output(_probe({"width": 320, "height": 240, "duration": "8.0"}))
        assert info.duration == 8.0

    def test_format_duration_wins_over_stream(self):
        info = parse_probe_output(_probe({"width": 320, "height": 240, "duration": "8.0"}, {"duration": "16.0"}))
        assert info.duration == 16.0

    def test_zero_fps_denominator_falls_back(self):
        info = parse_probe_output(_probe(
            {"width": 320, "height": 240, "avg_frame_rate": "0/0", "r_frame_rate": "12500/1000"}
        ))
        assert info.fps == 12.5

    def test_missing_codec_is_unknown(self):
        assert parse_probe_output(_probe({"width": 320, "height": 240})).codec == "unknown"

    def test_no_streams_raises(self):
        with pytest.raises(ValueError, match="no video stream"):
            parse_probe_output({"streams": [], "format": {"duration": "3.0"}})

    def test_zero_dimensions_raise(self):
        with pytest.raises(ValueError, match="no video stream"):
            parse_probe_output(_probe({"width": 0, "height": 240}))


class TestGetVideoInfo:
    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch("video_utils.subprocess.run")
    def test_ffprobe_failure_raises_value_error(self, mock_run, mock_verify):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Invalid data found")

        extractor = VideoFrameExtractor()
        with pytest.raises(ValueError, match="could not be read as a valid video"):
            extractor.get_video_info("/tmp/fake.mp4")

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch("video_utils.subprocess.run")
    def test_invalid_json_raises_value_error(self, mock_run, mock_verify):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")
        with pytest.raises(ValueError, match="could not be read as a valid video"):
            VideoFrameExtractor().get_video_info("/tmp/fake.mp4")

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch("video_utils.subprocess.run")
    def test_timeout_raises_runtime_error(self, mock_run, mock_verify):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)
        with pytest.raises(RuntimeError, match="timed out"):
            VideoFrameExtractor().get_video_info("/tmp/fake.mp4")

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch("video_utils.subprocess.run")
    def test_parses_json_output(self, mock_run, mock_verify):
        payload = {
            "streams": [{"codec_name": "hevc", "width": 2880, "height": 1620, "avg_frame_rate": "25/2"}],
            "format": {"duration": "16.0"},
        }
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")

        info = VideoFrameExtractor().get_video_info("/tmp/fake.mp4")

        assert info == VideoInfo(duration=16.0, width=2880, height=1620, fps=12.5, codec="hevc", rotation=0)
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "ffprobe"
        assert "-print_format" in cmd and "json" in cmd
        assert "-show_streams" in cmd and "-show_format" in cmd
        assert mock_run.call_args.kwargs["timeout"] == 30.0


SHOWINFO_SAMPLE = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'seg.mp4':
  Duration: 00:00:16.00, start: 0.000000, bitrate: 2517 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (hevc (native) -> rawvideo (native))
[Parsed_showinfo_1 @ 0x5581] config in time_base: 1/12800, frame_rate: 25/2
[Parsed_showinfo_1 @ 0x5581] n:   0 pts:      0 pts_time:0       duration:   1024 duration_time:0.08 fmt:gray cl:unspecified sar:1/1 s:640x360 i:P iskey:1 type:I checksum:0848BE85 plane_checksum:[0848BE85] mean:[128] stdev:[0.0]
[Parsed_showinfo_1 @ 0x5581] n:   1 pts:   1024 pts_time:0.08    duration:   1024 duration_time:0.08 fmt:gray cl:unspecified sar:1/1 s:640x360 i:P iskey:0 type:P checksum:0848BE85 plane_checksum:[0848BE85] mean:[128] stdev:[0.0]
[Parsed_showinfo_1 @ 0x5581] n:   2 pts:   -512 pts_time:-0.04   duration:   1024 duration_time:0.08 fmt:gray cl:unspecified sar:1/1 s:640x360 i:P iskey:0 type:P checksum:0848BE85 plane_checksum:[0848BE85] mean:[128] stdev:[0.0]
[Parsed_showinfo_1 @ 0x5581] n:   3 pts:      2 pts_time:2.22222e-05 duration:   1024 duration_time:0.08 fmt:gray cl:unspecified sar:1/1 s:640x360 i:P iskey:0 type:P checksum:0848BE85 plane_checksum:[0848BE85] mean:[128] stdev:[0.0]
[out#0/rawvideo @ 0x5582] video:54000KiB audio:0KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: 0.000000%
"""


class TestParseShowinfo:
    def test_parses_frames_and_ignores_other_lines(self):
        frames = parse_showinfo(SHOWINFO_SAMPLE)
        assert frames[:3] == [
            ShowinfoFrame(pts=0.0, width=640, height=360),
            ShowinfoFrame(pts=0.08, width=640, height=360),
            ShowinfoFrame(pts=-0.04, width=640, height=360),
        ]
        assert len(frames) == 4
        assert frames[3].pts == pytest.approx(2.22222e-05)
        assert (frames[3].width, frames[3].height) == (640, 360)

    def test_line_without_frame_data_is_none(self):
        assert parse_showinfo_line("  Stream #0:0 -> #0:0 (hevc (native) -> rawvideo (native))") is None
        assert parse_showinfo_line("[Parsed_showinfo_1 @ 0x1] config in time_base: 1/12800, frame_rate: 25/2") is None

    def test_positive_exponent_parses_as_a_float(self):
        line = "[Parsed_showinfo_0 @ 0x1] n:   9 pts:  1280000 pts_time:1e+02 duration: 1024 duration_time:0.08 fmt:gray s:640x360 i:P iskey:0 type:P"
        assert parse_showinfo_line(line) == ShowinfoFrame(pts=100.0, width=640, height=360)

    def test_pts_field_of_the_line_is_not_taken_for_size(self):
        line = "[Parsed_showinfo_0 @ 0x1] n:   5 pts:   5120 pts_time:0.4 duration: 1024 duration_time:0.08 fmt:bgr24 s:2880x1620 i:P iskey:0 type:P"
        assert parse_showinfo_line(line) == ShowinfoFrame(pts=0.4, width=2880, height=1620)
