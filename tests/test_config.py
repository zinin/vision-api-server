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


class TestVideoExtractTimeout:
    def test_default(self):
        assert Settings(yolo_models="{}").video_extract_timeout == 300.0

    def test_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models="{}", video_extract_timeout=5)

    def test_minimum_accepted(self):
        assert Settings(yolo_models="{}", video_extract_timeout=10).video_extract_timeout == 10.0


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


class TestStabilizerSettings:
    def test_defaults(self):
        s = Settings(yolo_models='{}')
        assert s.stabilizer_conf_factor == 0.4
        assert s.stabilizer_iou_threshold == 0.3
        assert s.stabilizer_min_vote_conf == 0.3
        assert s.stabilizer_grace_center == 2.0
        assert s.stabilizer_grace_edge == 0.5
        assert s.stabilizer_center_zone == 0.6
        assert s.stabilizer_max_staleness == 5.0

    def test_custom_values(self):
        s = Settings(
            yolo_models='{}',
            stabilizer_conf_factor=0.3,
            stabilizer_iou_threshold=0.5,
            stabilizer_grace_center=3.0,
        )
        assert s.stabilizer_conf_factor == 0.3
        assert s.stabilizer_iou_threshold == 0.5
        assert s.stabilizer_grace_center == 3.0

    def test_conf_factor_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_conf_factor=0.0)

    def test_conf_factor_must_be_lte_1(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_conf_factor=1.5)

    def test_iou_threshold_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_iou_threshold=0.0)

    def test_max_staleness_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_max_staleness=0.0)

    def test_grace_non_negative(self):
        s = Settings(yolo_models='{}', stabilizer_grace_center=0.0, stabilizer_grace_edge=0.0)
        assert s.stabilizer_grace_center == 0.0

    def test_grace_negative_rejected(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_grace_center=-1.0)


class TestVideoInferenceSettings:
    def test_defaults_are_auto(self):
        s = Settings(yolo_models="{}")
        assert s.video_fp16 == "auto"
        assert s.video_batch_size == "auto"

    @pytest.mark.parametrize("raw,expected", [
        ("", "auto"), ("auto", "auto"), ("AUTO", "auto"),
        ("true", "true"), ("True", "true"), ("false", "false"), (True, "true"),
    ])
    def test_fp16_values(self, raw, expected):
        assert Settings(yolo_models="{}", video_fp16=raw).video_fp16 == expected

    @pytest.mark.parametrize("raw", ["yes", "1", "fp16"])
    def test_fp16_rejects_other_values(self, raw):
        with pytest.raises(ValidationError):
            Settings(yolo_models="{}", video_fp16=raw)

    @pytest.mark.parametrize("raw,expected", [
        ("", "auto"), ("auto", "auto"), ("8", "8"), (" 16 ", "16"), (4, "4"), ("1", "1"), ("64", "64"),
    ])
    def test_batch_size_values(self, raw, expected):
        assert Settings(yolo_models="{}", video_batch_size=raw).video_batch_size == expected

    @pytest.mark.parametrize("raw", ["0", "65", "-1", "eight", "2.5"])
    def test_batch_size_rejects_other_values(self, raw):
        with pytest.raises(ValidationError):
            Settings(yolo_models="{}", video_batch_size=raw)

    def test_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("VIDEO_FP16", "false")
        monkeypatch.setenv("VIDEO_BATCH_SIZE", "2")
        s = Settings(yolo_models="{}")
        assert (s.video_fp16, s.video_batch_size) == ("false", "2")
