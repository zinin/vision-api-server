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

    def test_amd_global_encode_args_include_vaapi_device(self):
        """vaapi_device is in global_encode_args (before -i), not get_encode_args."""
        config = HWAccelConfig(accel_type=HWAccelType.AMD)
        assert "-vaapi_device" in config.global_encode_args

    def test_cpu_global_encode_args_empty(self):
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        assert config.global_encode_args == []

    def test_nvidia_global_encode_args_empty(self):
        config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)
        assert config.global_encode_args == []

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
        """When forced to cpu, don't probe -- just return CPU config."""
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
