# FFmpeg Hardware-Accelerated Video Pipeline

**Date:** 2026-02-15
**Scope:** `/detect/video/visualize` endpoint — VideoAnnotator
**Ticket:** VAS-2 (расширение)

## Problem

VideoAnnotator использует CPU для всех операций с видео:
1. `cv2.VideoCapture` — CPU-декодирование
2. `cv2.VideoWriter` с `mp4v` — CPU-кодирование (промежуточный файл)
3. FFmpeg `libx264`/`libx265` — CPU-перекодирование + merge аудио

Двойное кодирование (mp4v → h264) — расточительно. GPU простаивает между YOLO-инференсами.

## Solution

Заменить cv2.VideoCapture и cv2.VideoWriter на FFmpeg subprocess pipes с аппаратным ускорением.

### Новый пайплайн

```
FFmpeg decode (NVDEC/VAAPI/CPU) → pipe → Python numpy → draw bbox → pipe → FFmpeg encode (NVENC/VAAPI/CPU) + audio merge
```

- Один FFmpeg-процесс для декодирования (stdout pipe → Python)
- Один FFmpeg-процесс для кодирования (Python → stdin pipe) с merge аудио из оригинала
- Промежуточный файл `video_only.mp4` исчезает
- Фреймы проходят через CPU-память для отрисовки bbox (неизбежно)

### Fallback chain

NVIDIA (NVDEC/NVENC) → AMD (VAAPI) → CPU (software codecs)

## Components

### 1. `app/hw_accel.py` — Hardware Acceleration Detection

**HWAccelType** (enum): `NVIDIA`, `AMD`, `CPU`

**HWAccelConfig** (dataclass):
- `accel_type: HWAccelType`
- `decode_args: list[str]` — FFmpeg args for decoder
- `encode_args_fn(codec, crf) → list[str]` — FFmpeg args for encoder by codec/quality

**Detection** (run once at startup, cached):
1. `ffmpeg -hwaccels` → parse available hwaccels (cuda, vaapi)
2. `ffmpeg -encoders` → check for configured codec's encoder (e.g. `h264_nvenc`, `hevc_nvenc`, `av1_nvenc` depending on `VIDEO_CODEC`)
3. Select best available, cache result

When `VIDEO_HW_ACCEL` is forced (`nvidia`/`amd`) but hardware not available → log warning, fallback to CPU.

**FFmpeg is mandatory** after this change (no more cv2 fallback). Startup must verify FFmpeg is installed.

**Codec mapping:**

| Config codec | NVIDIA | AMD | CPU |
|---|---|---|---|
| h264 | `h264_nvenc` | `h264_vaapi` | `libx264` |
| h265 | `hevc_nvenc` | `hevc_vaapi` | `libx265` |
| av1 | `av1_nvenc` | fallback CPU | `libsvtav1` |

**NVENC** accepts regular CPU frames (handles upload internally).
**VAAPI** requires `-vaapi_device <device> -vf 'format=nv12,hwupload'`.
Default device: `/dev/dri/renderD128`, overridable via `VAAPI_DEVICE` env var.
**AMD VAAPI rate control**: use `-qp <crf>` (no `-crf` support in VAAPI encoders).

### 2. `app/ffmpeg_pipe.py` — FFmpeg Pipe Decoder/Encoder

#### FFmpegDecoder

Context manager. Launches FFmpeg subprocess that decodes video and pipes raw BGR24 frames to stdout.

```python
cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning']
cmd += hw_config.decode_args  # e.g. ['-hwaccel', 'cuda']
cmd += ['-i', input_path, '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
```

FFmpeg auto-rotates by default (applies display rotation metadata). Raw output already correct orientation.

**Stderr handling**: daemon thread reads stderr asynchronously to prevent pipe buffer deadlock.
**Frame buffer**: `np.frombuffer` returns read-only array → `.copy()` required for OpenCV drawing.
**Process health**: `read_frame()` checks `process.poll()` to detect mid-stream crashes.

API:
- `__enter__` → starts process, starts stderr drain thread
- `read_frame() → np.ndarray | None` — reads one frame (writable copy), None on EOF
- `__exit__` → terminates process, joins stderr thread, closes pipes

#### FFmpegEncoder

Context manager. Launches FFmpeg subprocess that accepts raw BGR24 frames via stdin, encodes with target codec, and merges audio from original video.

```python
cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}', '-r', str(fps),
    '-i', 'pipe:0',
    '-i', str(original_path),
    '-map', '0:v:0', '-map', '1:a:0?',
    '-map_metadata', '0',
]
cmd += hw_config.encode_args(codec, crf)
cmd += ['-c:a', 'aac', '-shortest', str(output_path)]
```

**Stderr handling**: daemon thread reads stderr asynchronously (same as decoder).

API:
- `__enter__` → starts process, starts stderr drain thread
- `write_frame(frame: np.ndarray)` — writes one frame; checks `process.poll()` before write
- `__exit__` → closes stdin, reads remaining stderr, waits for completion, checks return code

### 3. VideoAnnotator Changes

`annotate()` method:
- Replace `cv2.VideoCapture` → `FFmpegDecoder`
- Replace `cv2.VideoWriter` → `FFmpegEncoder`
- Remove `_merge_audio()` — encoder handles it
- Remove intermediate file logic (`video_only.mp4`, cleanup)
- Constructor receives `hw_config: HWAccelConfig`

### 4. Docker Changes

**NVIDIA** (`docker/nvidia/Dockerfile` + `docker-compose-nvidia.yml`):
- System FFmpeg on Ubuntu 24.04 is typically compiled with `--enable-nvenc`/`--enable-cuda`
- nvidia-container-toolkit mounts `libnvidia-encode.so`/`libnvidia-decode.so` automatically
- **docker-compose**: add `video` to `NVIDIA_DRIVER_CAPABILITIES` (required for NVENC/NVDEC)
- Verify at build/runtime; if not available, auto-fallback to CPU

**AMD** (`docker/amd/Dockerfile`):
- Add: `apt-get install libva-dev mesa-va-drivers vainfo`
- System FFmpeg on Ubuntu typically compiled with `--enable-vaapi`

**CPU** — no changes.

### 5. Configuration

New `.env` parameters:
```
VIDEO_HW_ACCEL=auto  # auto | nvidia | amd | cpu
VAAPI_DEVICE=/dev/dri/renderD128  # VAAPI render device (default: /dev/dri/renderD128)
```

`auto` (default) — detect automatically. Can force specific backend.
Forced mode (`nvidia`/`amd`) falls back to CPU with warning if hardware unavailable.

Add to `app/config.py` Settings class.

## Non-Goals

- GPU-accelerated bbox drawing (requires CUDA kernels, overkill)
- Changes to `/detect/video` or `/extract/frames` endpoints (different pipeline, lower priority)
- Custom FFmpeg builds in Docker (use system FFmpeg, fallback to CPU if no hw support)
- Runtime encoder fallback (if GPU encoder fails mid-job → job error, not automatic CPU retry)
- VFR (variable frame rate) preservation — CFR normalization is acceptable
- Graceful subprocess shutdown on job cancellation (deferred to future)

## Testing

- Unit tests: mock subprocess.Popen, verify correct FFmpeg commands per HWAccelType
- Integration: test with CPU fallback (always available)
- Manual: verify GPU acceleration on NVIDIA/AMD hardware via FFmpeg logs
