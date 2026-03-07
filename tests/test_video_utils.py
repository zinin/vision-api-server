import subprocess
from unittest.mock import patch, MagicMock

import pytest

from video_utils import VideoFrameExtractor, VideoInfo


class TestExtractFramesNoVideoStream:
    """Verify that files without a video stream raise ValueError early."""

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch.object(VideoFrameExtractor, "get_video_info")
    def test_audio_only_file_raises_value_error(self, mock_info, mock_verify):
        mock_info.return_value = VideoInfo(
            duration=0.15, width=0, height=0, fps=30.0, codec="unknown"
        )

        extractor = VideoFrameExtractor()
        with pytest.raises(ValueError, match="no video stream"):
            extractor.extract_frames("/tmp/fake.mp4")

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch.object(VideoFrameExtractor, "get_video_info")
    def test_zero_width_raises_value_error(self, mock_info, mock_verify):
        mock_info.return_value = VideoInfo(
            duration=10.0, width=0, height=720, fps=30.0, codec="h264"
        )

        extractor = VideoFrameExtractor()
        with pytest.raises(ValueError, match="no video stream"):
            extractor.extract_frames("/tmp/fake.mp4")

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch.object(VideoFrameExtractor, "get_video_info")
    def test_zero_height_raises_value_error(self, mock_info, mock_verify):
        mock_info.return_value = VideoInfo(
            duration=10.0, width=1280, height=0, fps=30.0, codec="h264"
        )

        extractor = VideoFrameExtractor()
        with pytest.raises(ValueError, match="no video stream"):
            extractor.extract_frames("/tmp/fake.mp4")


class TestGetVideoInfoInvalidFile:
    """Verify that ffprobe failures for invalid files raise ValueError."""

    @patch.object(VideoFrameExtractor, "_verify_ffmpeg")
    @patch("video_utils.subprocess.run")
    def test_ffprobe_failure_raises_value_error(self, mock_run, mock_verify):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="ffprobe", stderr="Invalid data found"
        )

        extractor = VideoFrameExtractor()
        with pytest.raises(ValueError, match="could not be read as a valid video"):
            extractor.get_video_info("/tmp/fake.mp4")
