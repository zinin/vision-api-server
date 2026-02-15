# Source Codec Matching — Auto Output Settings from Input Video

**Date:** 2026-02-15
**Scope:** `/detect/video/visualize` endpoint — VideoAnnotator output encoding
**Ticket:** VAS-2 (расширение)

## Problem

Currently `VIDEO_CODEC` and `VIDEO_CRF` are global settings with fixed defaults (h264, CRF 18). When a user uploads an h265 video at 8 Mbps, the output is always h264 at CRF 18 — codec and quality don't match the source.

## Solution

When `VIDEO_CODEC=auto` (new default), probe the input video via ffprobe and encode the output with the same codec and approximately the same bitrate. The output should be as close to the original as possible — just with bounding boxes drawn.

### Behavior Matrix

| `VIDEO_CODEC` | Codec source | Quality source |
|---|---|---|
| `auto` (default) | From input file (ffprobe) | Bitrate from input (`-b:v`) |
| `h264`/`h265`/`av1` | From config (explicit) | CRF from `VIDEO_CRF` |

### Fallback Chain

| Input codec | Bitrate available | Output |
|---|---|---|
| h264/hevc/av1 | yes | Same codec + `-b:v` from source |
| h264/hevc/av1 | no | Same codec + CRF 18 |
| vp9/mpeg4/other | yes | h264 + CRF 18 |
| vp9/mpeg4/other | no | h264 + CRF 18 |

Unsupported input codec → full fallback to h264 + CRF 18 (current behavior). Missing bitrate → CRF 18 with matched codec.

## Components

### 1. Config Change

`app/config.py`:
- `video_codec: str = "auto"` — new default
- Validator accepts `("auto", "h264", "h265", "av1")`

### 2. Extended ffprobe Metadata

`VideoAnnotator._get_video_metadata()` currently returns `(fps, width, height, total_frames)`.

New return type:

```python
@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    total_frames: int
    codec_name: str | None   # ffprobe codec_name: "h264", "hevc", "av1", "vp9", etc.
    bit_rate: int | None      # video stream bit_rate in bps, e.g. 8000000
```

ffprobe already runs — just parse two additional fields from `stream`:
- `stream["codec_name"]` → `"h264"`, `"hevc"`, `"av1"`, `"vp9"`, etc.
- `stream["bit_rate"]` → `"8234567"` (string, may be absent)

**Codec name mapping (ffprobe → internal):**

| ffprobe `codec_name` | Internal `codec` |
|---|---|
| `h264` | `h264` |
| `hevc` | `h265` |
| `av1` | `av1` |
| anything else | fallback → `h264` |

### 3. Bitrate Mode in Encode Args

`HWAccelConfig.get_encode_args(codec, crf=None, bitrate=None)`:

When `bitrate` is provided, use target bitrate instead of CRF/CQ/QP:

| Accel | CRF mode (current) | Bitrate mode (new) |
|---|---|---|
| **CPU** | `-c:v libx264 -crf 18` | `-c:v libx264 -b:v 8000000` |
| **NVIDIA** | `-c:v h264_nvenc -rc vbr -cq 18` | `-c:v h264_nvenc -b:v 8000000` |
| **AMD** | `-c:v h264_vaapi -qp 18` | `-c:v h264_vaapi -b:v 8000000` |

`FFmpegEncoder.__init__` signature: `crf: int | None = None, bitrate: int | None = None` — passes through to `get_encode_args()`.

### 4. Auto-Resolve in VideoAnnotator

`VideoAnnotator.__init__`:
- `codec: str = "auto"` (was `"h264"`)
- `crf: int = 18` unchanged

`VideoAnnotator.annotate()`:
1. Call `_get_video_metadata()` → `VideoMetadata`
2. If `self.codec == "auto"`:
   - Map `metadata.codec_name` → internal codec (h264/h265/av1). Unknown → h264.
   - If mapped codec is known AND `metadata.bit_rate` is available → use bitrate mode
   - If mapped codec is known but no bitrate → use codec + CRF 18
   - If unknown codec → h264 + CRF 18
3. If `self.codec != "auto"` → use `self.codec` + `self.crf` (current behavior, metadata ignored)
4. Create `FFmpegEncoder` with resolved codec + crf/bitrate

### 5. HW Accel Startup Detection

`detect_hw_accel()` at startup:
- When `video_codec == "auto"` → pass `codec="h264"` for encoder availability check (h264 is the most universal baseline)
- The actual per-file codec is resolved at encode time
- `get_encode_args()` already handles fallback to CPU encoder when GPU encoder for a specific codec is unavailable (e.g., AMD + av1 → CPU libsvtav1)

### 6. Worker Wiring

`main.py` `_annotation_worker`:
- Passes `codec=settings.video_codec` (which may now be `"auto"`)
- No other changes needed — VideoAnnotator handles resolution internally

## Non-Goals

- Audio codec matching (always AAC)
- Per-request codec/quality override via API parameters
- VBR/CBR mode selection
- Runtime encoder fallback (GPU fails mid-job → error, not CPU retry)

## Testing

- `test_config.py`: video_codec=auto default, validation includes "auto"
- `test_hw_accel.py`: `get_encode_args` with bitrate parameter (NVIDIA/AMD/CPU)
- `test_video_annotator.py`: auto codec resolution from ffprobe, fallback to h264, missing bitrate
- `test_ffmpeg_pipe.py`: FFmpegEncoder with bitrate parameter
