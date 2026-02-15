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
        # av1: no VAAPI encoder -- fallback handled in get_encode_args
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


def detect_hw_accel(
    mode: str = "auto",
    codec: str = "h264",
    vaapi_device: str = "/dev/dri/renderD128",
) -> HWAccelConfig:
    """Detect available hardware acceleration.

    Args:
        mode: "auto", "nvidia", "amd", or "cpu"
        codec: configured video codec ("h264", "h265", "av1") -- used to check encoder availability
        vaapi_device: VAAPI render device path

    Returns:
        HWAccelConfig with detected acceleration type.
    """
    if mode == "cpu":
        logger.info("Hardware acceleration: forced CPU")
        return HWAccelConfig(accel_type=HWAccelType.CPU)

    # Check NVIDIA -- verify encoder for configured codec exists
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

    # Check AMD -- verify encoder for configured codec exists (av1 has no VAAPI encoder)
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

    # CPU fallback -- validate encoder availability only in auto mode
    # (skip when falling back from a forced GPU mode or when ffmpeg is absent)
    if mode == "auto":
        cpu_encoder = _ENCODER_MAP[HWAccelType.CPU].get(codec)
        if cpu_encoder:
            encoder_output = _ffmpeg_query(["-hide_banner", "-encoders"])
            # Only validate when ffmpeg actually responded (non-empty output)
            if encoder_output and cpu_encoder not in encoder_output:
                raise RuntimeError(
                    f"CPU encoder '{cpu_encoder}' for codec '{codec}' not found in FFmpeg. "
                    f"Install FFmpeg with '{cpu_encoder}' support or change VIDEO_CODEC."
                )

    logger.info("Hardware acceleration: CPU (software codecs)")
    return HWAccelConfig(accel_type=HWAccelType.CPU)
