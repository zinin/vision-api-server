import pytest
from pydantic import ValidationError

from config import Settings


def test_video_job_settings_defaults():
    s = Settings(yolo_models="{}")
    assert s.video_job_ttl == 3600
    assert s.video_jobs_dir == "/tmp/vision_jobs"
    assert s.max_queued_jobs == 10
    assert s.default_detect_every == 5
    assert s.log_level == "INFO"
    assert s.video_codec == "auto"
    assert s.video_crf == 18


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

    def test_vaapi_device_default(self):
        s = Settings(yolo_models='{}')
        assert s.vaapi_device == "/dev/dri/renderD128"

    def test_vaapi_device_custom(self):
        s = Settings(yolo_models='{}', vaapi_device="/dev/dri/renderD129")
        assert s.vaapi_device == "/dev/dri/renderD129"


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
