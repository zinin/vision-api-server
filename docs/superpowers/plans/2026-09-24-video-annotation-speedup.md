# Video Annotation Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ускорить `/detect/video/visualize` примерно в 2.6 раза на NVIDIA (эталонный ролик 2560×1920, yolo26x@1024: 86 → 33 с) и не сломать AMD и CPU.

**Architecture:** Два прохода через процессы ffmpeg остаются. В проходе 1 поток-читатель (`ThreadedFrameReader`) читает кадры BGR и уменьшает их до размера инференса ровно как `LetterBox` в Ultralytics, а основной поток гонит YOLO батчами (`BatchDetector`, FP16 и размер батча по настройкам `VIDEO_FP16`/`VIDEO_BATCH_SIZE`). В проходе 2 кадры идут в yuv420p от декодера до энкодера, рамки рисуются прямо на плоскостях Y/U/V, запись идёт в потоке-писателе (`ThreadedFrameWriter`). Видео-задачи получают собственный экземпляр модели (`ModelManager.get_video_model`).

**Tech Stack:** Python 3.13 (`.venv`), FastAPI, Ultralytics YOLO ≥ 8.4.150, PyTorch, OpenCV (`cv2`), numpy, ffmpeg/ffprobe через `subprocess`, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-24-video-annotation-speedup-design.md`

## Global Constraints

- Тесты только в `.venv`: `.venv/bin/python -m pytest tests/ -v` из корня репозитория. Каждый шаг «Run» ниже подразумевает этот интерпретатор.
- Модули в `app/` импортируют друг друга по голым именам (`from frame_threads import ...`), потому что `app/` лежит в `sys.path` (см. `tests/conftest.py`). В `app/` нет `__init__.py`, так и оставить.
- Комментарии и докстринги в коде на английском, как во всём `app/`. Коммиты по образцу репозитория: `feat(video): ...`, `test(video): ...`, `docs: ...`, `chore: ...`.
- `ultralytics>=8.4.150,<9.0.0`: 8.4.150 — первая версия с параметром `predict(quantize=16)`, в 8.4.149 его нет. Устаревший `half=True` пишет WARNING на каждом вызове, его не использовать.
- `VIDEO_FP16`: `auto` | `true` | `false`. `VIDEO_BATCH_SIZE`: `auto` | целое 1–64. Регистр не важен, пустое значение означает `auto`. `auto`: NVIDIA (устройство модели `cuda` и `torch.version.hip is None`) — FP16 и батч 8; ROCm и CPU — FP32 и батч 1. `VIDEO_FP16=true` на CPU — WARNING и FP32.
- Кадр для инференса уменьшает только `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` до `inference_size()`. Скейлеры ffmpeg в проходе 1 запрещены: они меняют детекции (на 400 кадрах слабых детекций 287 вместо 485).
- Проход 2: `yuv420p` от декодера до энкодера. Цвет рамок — BT.601 limited range, как в текущем энкодере.
- Не меняются: API и ответы, `AnnotationStats`, разбивка прогресса (проход 1 — 0–80 %, проход 2 — 80–99 %), политика кодека и битрейта, откат NVENC → CPU, семантика отмены. Энкодер при отмене не убивается.
- Видео-задачи берут модель только через `ModelManager.get_video_model`, `/detect` и `/detect/video` — через `get_model`.
- Никаких записей камер в git: репозиторий публичный. Тестовые клипы генерируются `ffmpeg -f lavfi` во временный каталог, интеграционные тесты пропускаются без `ffmpeg`/`ffprobe`.
- `tests/test_supervisor.py::test_smoke_killpg_reaches_a_grandchild` падает внутри контейнера NVIDIA-образа и без этих изменений (окружение контейнера). На хосте он проходит. Ориентир — прогон на хосте.
- Перед созданием PR все файлы из `docs/superpowers/` удаляются из ветки через `git rm` и коммитятся (Task 12). Плановые документы не должны попасть в диф PR.
- Субагентам не назначать модель haiku.
- Задачи 10–12 выполняет основная сессия вместе с владельцем, а не субагент: они трогают GPU владельца, его хосты и PR.

## Review Focus

Пять классов входа, которые спека подразумевает, а модульные тесты на моках не проверяют. Каждый пункт закрыт тестом в задаче-владельце:

1. **Запись с IP-камеры в полном диапазоне `yuvj420p`** (так пишут многие камеры) — проход 2 просит у ffmpeg `yuv420p`. Рамка должна появиться своим цветом, число кадров не должно измениться. Тест `test_full_range_source`, Task 8.
2. **Звук короче видео** — `-shortest` завершает энкодер раньше, чем кончились кадры. `annotate()` должен закончиться успешно, без зависания и падения потока-писателя, результат обрезан по звуку. Тест `test_audio_shorter_than_video_ends_the_output_early`, Task 8.
3. **Переменный fps** — проходы 1 и 2 должны насчитать одни и те же кадры, иначе рамки съедут. Рамка должна быть на последнем кадре, число кадров результата равно числу кадров прохода 1. Тест `test_variable_frame_rate_keeps_passes_aligned`, Task 8.
4. **Запись без звука** — в результате нет аудио, ошибки нет. Тест `test_clip_without_audio`, Task 8.
5. **Клип короче одного батча** (кадров меньше `VIDEO_BATCH_SIZE`) — один вызов `predict` со всеми кадрами, каждый кадр размечен и записан. Тест `test_clip_shorter_than_one_batch`, Task 7.

---

## Карта файлов

| Файл | Ответственность | Задачи |
|---|---|---|
| `app/config.py` | `video_fp16`, `video_batch_size` и их валидаторы | 1 |
| `docker/docker-compose-*.yml`, `docker/deploy/docker-compose-*.yml` | проброс `VIDEO_FP16`, `VIDEO_BATCH_SIZE` | 1 |
| `requirements.txt` | `ultralytics>=8.4.150` | 1 |
| `app/ffmpeg_pipe.py` | `frame_shape()`, `_grow_pipe()`, `pix_fmt`, `read_into`, `abort`, запись без копии | 2 |
| `app/frame_threads.py` (новый) | `ThreadedFrameReader`, `ThreadedFrameWriter` | 3 |
| `app/batch_inference.py` (новый) | `InferenceMode`, `resolve_inference_mode`, `model_stride`, `inference_size`, `resize_for_inference`, `extract_detections`, `BatchDetector` | 4 |
| `app/visualization.py` | BT.601-помощники, `yuv420_planes`, `draw_detection_yuv420`, общая геометрия подписи | 5 |
| `app/model_manager.py` | `get_video_model`, выгрузка по TTL и при остановке | 6 |
| `app/video_annotator.py` | оба прохода на новых компонентах, `fp16`/`batch_size`, логи | 7 |
| `app/main.py` | worker: `get_video_model` + `resolve_inference_mode` | 9 |
| `tests/test_config.py`, `tests/test_compose.py` | настройки, проброс | 1 |
| `tests/test_ffmpeg_pipe.py` | новые возможности декодера и энкодера | 2 |
| `tests/test_frame_threads.py` (новый) | потоки чтения и записи | 3 |
| `tests/test_batch_inference.py` (новый) | ресайз против `LetterBox`, режим, батчи, OOM | 4 |
| `tests/test_visualization.py` | отрисовка в YUV | 5 |
| `tests/test_model_manager.py` (новый) | `get_video_model` | 6 |
| `tests/test_video_annotator.py` | моки на `read_into`, новые тесты прохода 1 | 7 |
| `tests/test_video_annotation_integration.py` (новый) | `annotate()` с настоящим ffmpeg | 8 |
| `tests/test_worker.py` | `get_video_model`, режим инференса | 9 |
| `CLAUDE.md`, `.claude/rules/api.md`, `.claude/rules/docker.md`, `docker/deploy/.env.example` | документация | 1, 3, 4, 7, 8 |

---

### Task 1: Настройки `VIDEO_FP16` и `VIDEO_BATCH_SIZE`, проброс в compose, версия ultralytics

**Files:**
- Modify: `app/config.py`
- Modify: `docker/docker-compose-nvidia.yml`, `docker/docker-compose-amd.yml`, `docker/docker-compose-cpu.yml`, `docker/deploy/docker-compose-nvidia.yml`, `docker/deploy/docker-compose-amd.yml`, `docker/deploy/docker-compose-cpu.yml`
- Modify: `requirements.txt`, `docker/deploy/.env.example`, `.claude/rules/docker.md`, `CLAUDE.md`
- Test: `tests/test_config.py`, `tests/test_compose.py`

**Interfaces:**
- Consumes: ничего из других задач.
- Produces: `Settings.video_fp16: str` — нормализованная строка `"auto"`, `"true"` или `"false"`; `Settings.video_batch_size: str` — `"auto"` или десятичная строка `"1"`…`"64"`. Их читает worker в Task 9 и передаёт в `resolve_inference_mode` из Task 4.

- [ ] **Step 1: Написать падающие тесты настроек**

Дописать в конец `tests/test_config.py`:

```python


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
```

- [ ] **Step 2: Написать падающий тест проброса в compose**

В `tests/test_compose.py` перед строкой `SUPERVISOR_CMD = ...` добавить:

```python
# Video annotation inference knobs; every compose file forwards them so .env reaches the app.
VIDEO_INFERENCE_PASSTHROUGH = ("VIDEO_FP16", "VIDEO_BATCH_SIZE")
```

В конец файла дописать:

```python


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_compose_forwards_the_video_inference_settings(path):
    # empty value = auto in app/config.py, so the defaults stay per-device
    env = _environment(_service(yaml.safe_load(path.read_text())))
    for key in VIDEO_INFERENCE_PASSTHROUGH:
        assert env.get(key) == "${%s:-}" % key, (path, key)
```

- [ ] **Step 3: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_compose.py -v`
Expected: FAIL. `TestVideoInferenceSettings::test_defaults_are_auto` падает с `AttributeError: 'Settings' object has no attribute 'video_fp16'`, а шесть `test_every_compose_forwards_the_video_inference_settings[...]` — с `AssertionError`.

- [ ] **Step 4: Добавить настройки**

В `app/config.py` после строки `    vaapi_device: str = "/dev/dri/renderD128"  # VAAPI render device path` вставить:

```python
    # Video annotation inference; "auto" picks by the model's device (see batch_inference)
    video_fp16: str = "auto"  # auto | true | false
    video_batch_size: str = "auto"  # auto | 1..64 frames per YOLO call
```

Там же перед `    @field_validator("yolo_model_ttl")` вставить:

```python
    @field_validator("video_fp16", mode="before")
    @classmethod
    def validate_video_fp16(cls, v) -> str:
        value = str(v).strip().lower() or "auto"
        if value not in ("auto", "true", "false"):
            raise ValueError("video_fp16 must be one of: auto, true, false")
        return value

    @field_validator("video_batch_size", mode="before")
    @classmethod
    def validate_video_batch_size(cls, v) -> str:
        value = str(v).strip().lower() or "auto"
        if value == "auto":
            return value
        if not value.isdigit() or not 1 <= int(value) <= 64:
            raise ValueError("video_batch_size must be auto or an integer from 1 to 64")
        return str(int(value))

```

`mode="before"` нужен, чтобы `Settings(video_batch_size=8)` и `Settings(video_fp16=True)` тоже проходили: pydantic v2 сам не приводит `int` к `str`.

- [ ] **Step 5: Пробросить переменные во всех шести compose-файлах**

В пяти файлах с окружением в виде словаря (`docker/docker-compose-nvidia.yml`, `docker/docker-compose-amd.yml`, `docker/deploy/docker-compose-nvidia.yml`, `docker/deploy/docker-compose-amd.yml`, `docker/deploy/docker-compose-cpu.yml`) сразу после строки `VIDEO_HW_ACCEL: ...` этого файла вставить три строки с тем же отступом (6 пробелов):

```yaml
      # Video annotation inference; empty = auto (FP16 + batch 8 on NVIDIA, FP32 + batch 1 elsewhere)
      VIDEO_FP16: ${VIDEO_FP16:-}
      VIDEO_BATCH_SIZE: ${VIDEO_BATCH_SIZE:-}
```

Строки-якоря: `VIDEO_HW_ACCEL: ${VIDEO_HW_ACCEL:-nvidia}` (оба nvidia-файла), `VIDEO_HW_ACCEL: ${VIDEO_HW_ACCEL:-amd}` (оба amd-файла), `VIDEO_HW_ACCEL: cpu` (`docker/deploy/docker-compose-cpu.yml`).

В `docker/docker-compose-cpu.yml` окружение задано списком. После строки `      - VIDEO_HW_ACCEL=cpu` вставить:

```yaml
      # Video annotation inference; empty = auto (FP32 + batch 1 on CPU)
      - VIDEO_FP16=${VIDEO_FP16:-}
      - VIDEO_BATCH_SIZE=${VIDEO_BATCH_SIZE:-}
```

- [ ] **Step 6: Документация настроек**

В `docker/deploy/.env.example` после строки `# VIDEO_CRF=18` вставить:

```
# Video annotation inference (/detect/video/visualize). Empty or auto picks by the model's device:
# FP16 and batch 8 on NVIDIA, FP32 and batch 1 on AMD ROCm and CPU. FP16 on CPU is ignored.
# A batch that does not fit into video memory is halved automatically.
# VIDEO_FP16=auto          # auto | true | false
# VIDEO_BATCH_SIZE=auto    # auto | 1-64 frames per YOLO call
```

(перед блоком вставки оставить одну пустую строку, как между остальными блоками файла).

В `.claude/rules/docker.md`, в блоке «Environment Configuration», заменить

````
WATCHDOG_ENABLED=true
```
````

на

````
WATCHDOG_ENABLED=true

# Video annotation inference: empty or auto = FP16 + batch 8 on NVIDIA, FP32 + batch 1 on AMD/CPU
VIDEO_FP16=auto
VIDEO_BATCH_SIZE=auto
```
````

В `CLAUDE.md`, в блоке «Configuration», после строки `VAAPI_DEVICE=/dev/dri/renderD128        # VAAPI render device path` вставить:

```
VIDEO_FP16=auto                         # auto | true | false: FP16 YOLO in video annotation
VIDEO_BATCH_SIZE=auto                   # auto | 1-64 frames per YOLO call in video annotation
                                        # auto = FP16 + batch 8 on NVIDIA, FP32 + batch 1 on AMD/CPU
```

- [ ] **Step 7: Поднять версию ultralytics и обновить venv**

В `requirements.txt` заменить `ultralytics>=8.4.0,<9.0.0` на `ultralytics>=8.4.150,<9.0.0`.

Run: `.venv/bin/pip install -r requirements.txt -r requirements-dev.txt && .venv/bin/python -c "import ultralytics; print(ultralytics.__version__)"`
Expected: версия `8.4.150` или новее (в `.venv` до этого стояла 8.4.14).

- [ ] **Step 8: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_compose.py -v`
Expected: PASS.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS целиком.

- [ ] **Step 9: Commit**

```bash
git add app/config.py tests/test_config.py tests/test_compose.py requirements.txt \
  docker/docker-compose-nvidia.yml docker/docker-compose-amd.yml docker/docker-compose-cpu.yml \
  docker/deploy/docker-compose-nvidia.yml docker/deploy/docker-compose-amd.yml docker/deploy/docker-compose-cpu.yml \
  docker/deploy/.env.example .claude/rules/docker.md CLAUDE.md
git commit -m "feat(config): VIDEO_FP16 and VIDEO_BATCH_SIZE for video annotation inference"
```

---

### Task 2: `ffmpeg_pipe` — форматы кадра, чтение в готовый буфер, запись без копии, `abort()`

**Files:**
- Modify: `app/ffmpeg_pipe.py`
- Test: `tests/test_ffmpeg_pipe.py`

**Interfaces:**
- Consumes: ничего из других задач.
- Produces:
  - `frame_shape(width: int, height: int, pix_fmt: str) -> tuple[int, ...]` — `(h, w, 3)` для `"bgr24"`, `(w*h + 2*ceil(w/2)*ceil(h/2),)` для `"yuv420p"`, иначе `ValueError("Unsupported pix_fmt: ...")`.
  - `_grow_pipe(stream) -> None`.
  - `FFmpegDecoder(input_path, width, height, hw_config, pix_fmt: str = "bgr24")` с атрибутами `frame_shape: tuple[int, ...]`, `frame_size: int` и методами `read_into(buf: np.ndarray) -> bool`, `read_frame() -> np.ndarray | None`, `abort() -> None`.
  - `FFmpegEncoder(original_path, output_path, width, height, fps, hw_config, codec, crf=None, bitrate=None, pix_fmt: str = "bgr24")`, `write_frame(frame) -> bool` пишет `memoryview` кадра.

  Task 3 опирается на утиный тип декодера (`frame_shape`, `read_into`, `abort`) и энкодера (`write_frame`). Task 7 создаёт оба класса с `pix_fmt`.

- [ ] **Step 1: Написать падающие тесты**

В `tests/test_ffmpeg_pipe.py` заменить шапку файла

```python
import subprocess
from unittest.mock import patch, MagicMock, call
from io import BytesIO

import numpy as np
import pytest

from ffmpeg_pipe import FFmpegDecoder, FFmpegEncoder
from hw_accel import HWAccelConfig, HWAccelType
```

на

```python
import logging
import os
import subprocess
from unittest.mock import patch, MagicMock, call
from io import BytesIO

import numpy as np
import pytest

import ffmpeg_pipe
from ffmpeg_pipe import FFmpegDecoder, FFmpegEncoder, _grow_pipe, frame_shape
from hw_accel import HWAccelConfig, HWAccelType


class TestFrameShape:
    def test_bgr24(self):
        assert frame_shape(640, 480, "bgr24") == (480, 640, 3)

    def test_yuv420p(self):
        assert frame_shape(640, 480, "yuv420p") == (640 * 480 * 3 // 2,)

    def test_yuv420p_odd_size_rounds_chroma_up(self):
        assert frame_shape(641, 481, "yuv420p") == (641 * 481 + 2 * 321 * 241,)

    def test_unknown_pix_fmt(self):
        with pytest.raises(ValueError, match="Unsupported pix_fmt"):
            frame_shape(640, 480, "nv12")


class _ChunkedStdout(BytesIO):
    """A pipe that hands out at most ``chunk`` bytes per readinto() call."""

    def __init__(self, data: bytes, chunk: int):
        super().__init__(data)
        self._chunk = chunk

    def readinto(self, b):
        view = memoryview(b)[:self._chunk]
        return super().readinto(view)


class TestGrowPipe:
    def test_real_pipe_grows_to_1_mib(self):
        f_getpipe = getattr(ffmpeg_pipe.fcntl, "F_GETPIPE_SZ", None)
        if f_getpipe is None:
            pytest.skip("F_GETPIPE_SZ is Linux-only")
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd, "rb") as stream:
                _grow_pipe(stream)
                assert ffmpeg_pipe.fcntl.fcntl(stream.fileno(), f_getpipe) == 1 << 20
        finally:
            os.close(write_fd)

    def test_streams_without_a_pipe_are_ignored(self):
        _grow_pipe(BytesIO(b""))
        _grow_pipe(MagicMock())
```

В классе `TestFFmpegDecoder` после последнего теста `test_amd_decode_args` (перед строкой `class TestFFmpegEncoder:`) добавить:

```python
    def test_yuv420p_output_and_flat_frames(self):
        size = 640 * 480 * 3 // 2
        mock_proc = self._make_mock_process([])
        mock_proc.stdout = BytesIO(bytes(range(256)) * (size // 256) + bytes(size % 256))
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegDecoder("input.mp4", 640, 480, config, pix_fmt="yuv420p") as decoder:
                assert decoder.frame_shape == (size,)
                assert decoder.frame_size == size
                frame = decoder.read_frame()
                assert decoder.read_frame() is None

        cmd = mock_popen.call_args[0][0]
        assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
        assert frame.shape == (size,)
        assert frame[:3].tolist() == [0, 1, 2]

    def test_read_into_fills_the_buffer_in_place(self):
        frames = [np.full((4, 6, 3), i, dtype=np.uint8) for i in (7, 9)]
        mock_proc = self._make_mock_process(frames)
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        buf = np.zeros((4, 6, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                assert decoder.read_into(buf) is True
                assert (buf == 7).all()
                assert decoder.read_into(buf) is True
                assert (buf == 9).all()
                assert decoder.read_into(buf) is False

    def test_read_into_retries_short_reads(self):
        frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        mock_proc = self._make_mock_process([])
        mock_proc.stdout = _ChunkedStdout(frame.tobytes(), chunk=5)
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        buf = np.zeros_like(frame)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                assert decoder.read_into(buf) is True
        np.testing.assert_array_equal(buf, frame)

    def test_read_into_rejects_a_wrong_size_buffer(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                with pytest.raises(ValueError, match="a frame needs 72"):
                    decoder.read_into(np.zeros(10, dtype=np.uint8))

    def test_read_into_raises_when_ffmpeg_crashed(self):
        mock_proc = self._make_mock_process([])
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                with pytest.raises(RuntimeError, match="FFmpeg decoder crashed"):
                    decoder.read_into(np.zeros((4, 6, 3), dtype=np.uint8))

    def test_abort_kills_ffmpeg_and_close_stays_quiet(self, caplog):
        mock_proc = self._make_mock_process([])
        mock_proc.returncode = -9
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with caplog.at_level(logging.WARNING, logger="ffmpeg_pipe"):
                with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                    decoder.abort()

        mock_proc.kill.assert_called_once()
        assert not any("exited with code" in r.message for r in caplog.records)

```

В `TestFFmpegEncoder.test_write_frame` заменить две последние строки

```python
        assert result is True
        mock_proc.stdin.write.assert_called_once_with(frame.tobytes())
```

на

```python
        assert result is True
        mock_proc.stdin.write.assert_called_once()
        written = mock_proc.stdin.write.call_args.args[0]
        # A memoryview of the frame itself: no tobytes() copy of every frame
        assert isinstance(written, memoryview)
        assert written.tobytes() == frame.tobytes()
```

В конец класса `TestFFmpegEncoder` (конец файла) дописать:

```python

    def test_yuv420p_input_format(self):
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                crf=18,
                pix_fmt="yuv420p",
            ) as encoder:
                encoder.write_frame(np.zeros(640 * 480 * 3 // 2, dtype=np.uint8))

        cmd = mock_popen.call_args[0][0]
        pipe_input = cmd.index("pipe:0")
        # -pix_fmt before -i pipe:0 describes the raw input
        assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
        assert cmd.index("-pix_fmt") < pipe_input

    def test_unknown_pix_fmt_is_rejected_before_ffmpeg_starts(self):
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        with patch("ffmpeg_pipe.subprocess.Popen") as mock_popen:
            with pytest.raises(ValueError, match="Unsupported pix_fmt"):
                FFmpegEncoder("input.mp4", "output.mp4", 640, 480, 30.0, config, "h264", pix_fmt="rgb48")
        mock_popen.assert_not_called()
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_ffmpeg_pipe.py -v`
Expected: FAIL с `ImportError: cannot import name '_grow_pipe' from 'ffmpeg_pipe'` (весь модуль не собирается).

- [ ] **Step 3: Добавить `frame_shape` и `_grow_pipe`**

В `app/ffmpeg_pipe.py` заменить шапку

```python
import logging
import subprocess
import threading
from collections import deque
from pathlib import Path

import numpy as np

from hw_accel import HWAccelConfig

logger = logging.getLogger(__name__)
```

на

```python
import logging
import math
import subprocess
import threading
from collections import deque
from pathlib import Path

try:
    import fcntl
except ImportError:  # fcntl is Unix-only
    fcntl = None

import numpy as np

from hw_accel import HWAccelConfig

logger = logging.getLogger(__name__)

# 1 MiB is the default /proc/sys/fs/pipe-max-size, so any process may ask for it.
_PIPE_SIZE = 1 << 20


def frame_shape(width: int, height: int, pix_fmt: str) -> tuple[int, ...]:
    """numpy shape of one rawvideo frame: (h, w, 3) for bgr24, flat for yuv420p."""
    if pix_fmt == "bgr24":
        return (height, width, 3)
    if pix_fmt == "yuv420p":
        chroma = ((width + 1) // 2) * ((height + 1) // 2)
        return (width * height + 2 * chroma,)
    raise ValueError(f"Unsupported pix_fmt: {pix_fmt}")


def _grow_pipe(stream) -> None:
    """Best effort: enlarge a pipe's kernel buffer from 64 KiB to 1 MiB.

    A bigger buffer lets ffmpeg and Python hand a frame over in fewer, larger
    steps. Streams that are not pipes (test doubles, non-Linux) are skipped.
    """
    setpipe = getattr(fcntl, "F_SETPIPE_SZ", None)
    if setpipe is None:
        return
    try:
        fcntl.fcntl(stream.fileno(), setpipe, _PIPE_SIZE)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
```

- [ ] **Step 4: Переписать начало `FFmpegDecoder`**

Заменить в `app/ffmpeg_pipe.py` всё от строки `class FFmpegDecoder:` до строки `    def close(self) -> None:` декодера (не включая её) на:

```python
class FFmpegDecoder:
    """Decode video frames via FFmpeg subprocess pipe.

    ``pix_fmt`` is ``bgr24`` (frames shaped (h, w, 3)) or ``yuv420p`` (flat
    frames: the Y, U and V planes back to back). ``frame_shape`` and
    ``frame_size`` describe one frame.

    Usage:
        with FFmpegDecoder(path, w, h, config) as decoder:
            while (frame := decoder.read_frame()) is not None:
                process(frame)
    """

    def __init__(
        self,
        input_path: str | Path,
        width: int,
        height: int,
        hw_config: HWAccelConfig,
        pix_fmt: str = "bgr24",
    ):
        self._input_path = str(input_path)
        self._width = width
        self._height = height
        self.frame_shape = frame_shape(width, height, pix_fmt)
        self.frame_size = math.prod(self.frame_shape)
        self._stderr_lines: deque[bytes] = deque(maxlen=100)
        self._aborted = False

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
        cmd += hw_config.decode_args
        cmd += ["-i", str(input_path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]

        logger.debug(f"FFmpegDecoder command: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )
        _grow_pipe(self._process.stdout)
        # Start daemon thread to drain stderr and prevent deadlock
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._process, self._stderr_lines), daemon=True
        )
        self._stderr_thread.start()

    def read_into(self, buf: np.ndarray) -> bool:
        """Read the next frame straight into ``buf``. Returns False on EOF.

        ``buf`` must be a C-contiguous uint8 array of ``frame_size`` bytes.
        Filling it in place skips the bytes object and the copy that
        ``read_frame`` would make. Raises RuntimeError if ffmpeg crashed.
        """
        view = memoryview(buf).cast("B")
        if len(view) != self.frame_size:
            raise ValueError(
                f"Buffer holds {len(view)} bytes, a frame needs {self.frame_size}"
            )
        got = 0
        while got < self.frame_size:
            n = self._process.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        if got == self.frame_size:
            return True
        # Short read: normal EOF or a crash
        if self._process.poll() is not None and self._process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg decoder crashed ({rc_to_str(self._process.returncode)}): "
                f"{_format_stderr(self._stderr_lines)}"
            )
        return False

    def read_frame(self) -> np.ndarray | None:
        """Read one frame into a new writable array. Returns None on EOF."""
        frame = np.empty(self.frame_shape, dtype=np.uint8)
        return frame if self.read_into(frame) else None

    def abort(self) -> None:
        """Kill ffmpeg so that a read blocked in another thread returns.

        ``close()`` cannot unblock such a read: closing the pipe waits for the
        buffered reader's lock, which the reading thread holds.
        """
        self._aborted = True
        self._process.kill()

```

- [ ] **Step 5: Приглушить лог убитого декодера**

В `FFmpegDecoder.close()` заменить

```python
            logger.warning(
                "FFmpeg decoder did not exit after SIGKILL; process may be leaked"
            )
        elif self._process.returncode != 0:
            logger.warning(f"FFmpeg decoder exited with code {self._process.returncode}")
```

на

```python
            logger.warning(
                "FFmpeg decoder did not exit after SIGKILL; process may be leaked"
            )
        elif self._aborted:
            logger.debug(f"FFmpeg decoder aborted ({rc_to_str(self._process.returncode)})")
        elif self._process.returncode != 0:
            logger.warning(f"FFmpeg decoder exited with code {self._process.returncode}")
```

- [ ] **Step 6: Научить энкодер `pix_fmt` и записи без копии**

В `app/ffmpeg_pipe.py` заменить начало класса энкодера

```python
class FFmpegEncoder:
    """Encode raw BGR24 frames via FFmpeg subprocess pipe with audio merge.

    Usage:
        with FFmpegEncoder(original, output, w, h, fps, config, codec, crf) as enc:
            for frame in frames:
                enc.write_frame(frame)
    """

    def __init__(
        self,
        original_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        hw_config: HWAccelConfig,
        codec: str,
        crf: int | None = None,
        bitrate: int | None = None,
    ):
```

на

```python
class FFmpegEncoder:
    """Encode raw frames via FFmpeg subprocess pipe with audio merge.

    ``pix_fmt`` names the layout of the frames written to the pipe:
    ``bgr24`` or ``yuv420p`` (see ``frame_shape``).

    Usage:
        with FFmpegEncoder(original, output, w, h, fps, config, codec, crf) as enc:
            for frame in frames:
                enc.write_frame(frame)
    """

    def __init__(
        self,
        original_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        hw_config: HWAccelConfig,
        codec: str,
        crf: int | None = None,
        bitrate: int | None = None,
        pix_fmt: str = "bgr24",
    ):
        frame_shape(width, height, pix_fmt)  # rejects an unsupported pix_fmt early
```

Там же заменить `            "-f", "rawvideo", "-pix_fmt", "bgr24",` на `            "-f", "rawvideo", "-pix_fmt", pix_fmt,`.

После строк

```python
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
        )
```

вставить строку `        _grow_pipe(self._process.stdin)`.

В `write_frame` заменить докстринг-строку `        """Write one BGR24 frame to the encoder.` на `        """Write one frame (laid out as ``pix_fmt``) to the encoder.`, а строки

```python
        try:
            self._process.stdin.write(frame.tobytes())
```

на

```python
        try:
            # A memoryview hands the array's own memory to write(): no copy.
            self._process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
```

- [ ] **Step 7: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_ffmpeg_pipe.py -v`
Expected: PASS.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS целиком. `VideoAnnotator` пока вызывает `read_frame()`, а тот теперь работает через `read_into`.

- [ ] **Step 8: Commit**

```bash
git add app/ffmpeg_pipe.py tests/test_ffmpeg_pipe.py
git commit -m "feat(video): yuv420p frames, in-place reads, copy-free writes and abort in ffmpeg_pipe"
```

---

### Task 3: Потоки чтения и записи `frame_threads.py`

**Files:**
- Create: `app/frame_threads.py`
- Create: `tests/test_frame_threads.py`
- Modify: `CLAUDE.md` (таблица Architecture)

**Interfaces:**
- Consumes: утиный тип декодера из Task 2 — атрибут `frame_shape`, методы `read_into(buf) -> bool`, `abort()`; утиный тип энкодера — `write_frame(buf) -> bool`.
- Produces:
  - `ThreadedFrameReader(decoder, transform: Callable[[int, np.ndarray], Any], queue_size: int)` — контекстный менеджер; итерация отдаёт `(frame_num, transform(frame_num, frame))`; исключения фонового потока поднимаются в итераторе; выход из `with` до конца вызывает `decoder.abort()`.
  - `ThreadedFrameWriter(encoder, frame_shape: tuple[int, ...], pool_size: int = 4, queue_size: int = 2)` — контекстный менеджер с методами `acquire() -> np.ndarray`, `release(buf)`, `submit(buf) -> bool` (False после раннего EOF энкодера), `close(drop: bool = False)`.

  Оба использует Task 7.

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_frame_threads.py`:

```python
import threading
import time

import numpy as np
import pytest

from frame_threads import ThreadedFrameReader, ThreadedFrameWriter


class FakeDecoder:
    """read_into() fills the buffer with the frame number.

    ``fail_at`` raises on that read; ``block_at`` blocks that read until
    abort() is called, like a pipe read that only killing ffmpeg ends.
    """

    def __init__(self, frames=5, shape=(2, 3), fail_at=None, block_at=None):
        self.frame_shape = shape
        self._frames = frames
        self._fail_at = fail_at
        self._block_at = block_at
        self.reads = 0
        self.aborted = threading.Event()

    def read_into(self, buf):
        n = self.reads
        if n == self._block_at:
            self.aborted.wait(timeout=5)
            raise RuntimeError("FFmpeg decoder crashed (killed by signal 9)")
        if n == self._fail_at:
            raise RuntimeError("FFmpeg decoder crashed (rc=1)")
        if n >= self._frames:
            return False
        buf.fill(n)
        self.reads += 1
        return True

    def abort(self):
        self.aborted.set()


def _copy(frame_num, frame):
    return frame.copy()


class TestThreadedFrameReader:
    def test_yields_transformed_frames_in_order(self):
        decoder = FakeDecoder(frames=5)
        with ThreadedFrameReader(decoder, _copy, queue_size=2) as frames:
            items = list(frames)
        assert [n for n, _ in items] == [0, 1, 2, 3, 4]
        # The buffer is reused, so each payload must be the copy made for its frame.
        assert [int(payload[0, 0]) for _, payload in items] == [0, 1, 2, 3, 4]

    def test_none_from_transform_still_counts_the_frame(self):
        decoder = FakeDecoder(frames=4)

        def even_only(frame_num, frame):
            return frame.copy() if frame_num % 2 == 0 else None

        with ThreadedFrameReader(decoder, even_only, queue_size=2) as frames:
            items = list(frames)
        assert [(n, payload is None) for n, payload in items] == [
            (0, False), (1, True), (2, False), (3, True),
        ]

    def test_decoder_error_is_raised_in_the_consumer(self):
        decoder = FakeDecoder(frames=5, fail_at=2)
        received = []
        with pytest.raises(RuntimeError, match="decoder crashed"):
            with ThreadedFrameReader(decoder, _copy, queue_size=2) as frames:
                for n, _ in frames:
                    received.append(n)
        assert received == [0, 1]

    def test_transform_error_is_raised_in_the_consumer(self):
        def broken(frame_num, frame):
            raise ValueError("bad frame")

        with pytest.raises(ValueError, match="bad frame"):
            with ThreadedFrameReader(FakeDecoder(frames=3), broken, queue_size=2) as frames:
                list(frames)

    def test_leaving_early_aborts_a_blocked_read(self):
        decoder = FakeDecoder(frames=10, block_at=1)
        started = time.monotonic()
        with pytest.raises(KeyError):
            with ThreadedFrameReader(decoder, _copy, queue_size=2) as frames:
                for _ in frames:
                    raise KeyError("consumer gave up")
        assert decoder.aborted.is_set()
        assert time.monotonic() - started < 2

    def test_leaving_early_with_a_full_queue_does_not_hang(self):
        decoder = FakeDecoder(frames=1000)
        started = time.monotonic()
        with ThreadedFrameReader(decoder, _copy, queue_size=1) as frames:
            next(iter(frames))
            time.sleep(0.2)  # let the reader fill the queue and block on put()
        assert decoder.aborted.is_set()
        assert time.monotonic() - started < 2

    def test_reaching_the_end_does_not_abort(self):
        decoder = FakeDecoder(frames=3)
        with ThreadedFrameReader(decoder, _copy, queue_size=2) as frames:
            list(frames)
        assert not decoder.aborted.is_set()


class FakeEncoder:
    """Records a copy of every frame; ``eof_after`` / ``fail_at`` / ``delay`` shape it."""

    def __init__(self, eof_after=None, fail_at=None, delay=0.0):
        self.frames = []
        self._eof_after = eof_after
        self._fail_at = fail_at
        self._delay = delay

    def write_frame(self, frame):
        n = len(self.frames)
        if n == self._fail_at:
            raise RuntimeError("[hevc_nvenc] InitializeEncoder failed: out of memory (10)")
        if self._eof_after is not None and n >= self._eof_after:
            return False
        if self._delay:
            time.sleep(self._delay)
        self.frames.append(frame.copy())
        return True


SHAPE = (4,)


class TestThreadedFrameWriter:
    def test_writes_all_frames_in_order(self):
        encoder = FakeEncoder()
        with ThreadedFrameWriter(encoder, SHAPE, pool_size=3, queue_size=2) as writer:
            for n in range(10):
                buf = writer.acquire()
                buf.fill(n)
                assert writer.submit(buf) is True
        assert [int(f[0]) for f in encoder.frames] == list(range(10))

    def test_clean_exit_waits_for_a_slow_encoder(self):
        encoder = FakeEncoder(delay=0.05)
        with ThreadedFrameWriter(encoder, SHAPE, pool_size=4, queue_size=2) as writer:
            for n in range(4):
                buf = writer.acquire()
                buf.fill(n)
                writer.submit(buf)
        assert len(encoder.frames) == 4

    def test_submit_returns_false_after_encoder_eof(self):
        encoder = FakeEncoder(eof_after=2)
        results = []
        with ThreadedFrameWriter(encoder, SHAPE, pool_size=2, queue_size=1) as writer:
            for n in range(20):
                buf = writer.acquire()
                buf.fill(n)
                ok = writer.submit(buf)
                results.append(ok)
                if not ok:
                    break
        assert results[-1] is False
        assert len(encoder.frames) == 2

    def test_encoder_error_is_raised_with_its_message(self):
        encoder = FakeEncoder(fail_at=0)
        with pytest.raises(RuntimeError, match="out of memory"):
            with ThreadedFrameWriter(encoder, SHAPE, pool_size=2, queue_size=1) as writer:
                for n in range(5):
                    buf = writer.acquire()
                    writer.submit(buf)

    def test_exception_in_the_body_drops_queued_frames(self):
        encoder = FakeEncoder(delay=0.1)
        with pytest.raises(KeyError):
            with ThreadedFrameWriter(encoder, SHAPE, pool_size=4, queue_size=3) as writer:
                for n in range(4):
                    buf = writer.acquire()
                    buf.fill(n)
                    writer.submit(buf)
                raise KeyError("cancelled")
        assert len(encoder.frames) < 4

    def test_release_returns_an_unused_buffer(self):
        encoder = FakeEncoder()
        with ThreadedFrameWriter(encoder, SHAPE, pool_size=1, queue_size=1) as writer:
            buf = writer.acquire()
            writer.release(buf)
            again = writer.acquire()  # would block forever if release() lost it
            writer.release(again)
        assert encoder.frames == []

    def test_buffers_have_the_requested_shape(self):
        with ThreadedFrameWriter(FakeEncoder(), (3, 5, 3), pool_size=1) as writer:
            buf = writer.acquire()
            assert buf.shape == (3, 5, 3)
            assert buf.dtype == np.uint8
            writer.release(buf)
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_frame_threads.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'frame_threads'`.

- [ ] **Step 3: Написать модуль**

Создать `app/frame_threads.py`:

```python
"""Background threads that keep the ffmpeg pipes busy while the main thread computes.

Pass 1 of the video annotator reads frames on a ``ThreadedFrameReader`` thread
while the main thread runs YOLO; pass 2 writes frames on a
``ThreadedFrameWriter`` thread while the main thread reads and draws the next
one. Pipe I/O, cv2 and torch release the GIL, so the threads really overlap.
"""
import logging
import queue
import threading
from typing import Any, Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

_POLL_SECONDS = 0.1
_JOIN_SECONDS = 10.0
_END = object()


class _Failure:
    """An exception raised on a background thread, handed to the main thread."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


class ThreadedFrameReader:
    """Read decoder frames on a background thread.

    The thread reads each frame into one reusable buffer of
    ``decoder.frame_shape``, calls ``transform(frame_num, frame)`` and queues
    ``(frame_num, result)``. The buffer is overwritten by the next read, so
    ``transform`` must return a new object, or None to only count the frame.

    Iterate over the reader to consume the frames; an exception from the
    thread is re-raised by the iterator. Leaving the ``with`` block before the
    end kills the decoder through ``decoder.abort()`` so that a read blocked in
    the pipe returns, then joins the thread.
    """

    def __init__(
        self,
        decoder: Any,
        transform: Callable[[int, np.ndarray], Any],
        queue_size: int,
    ):
        self._decoder = decoder
        self._transform = transform
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._finished = False  # the consumer got the end marker or the failure
        self._thread = threading.Thread(target=self._run, name="frame-reader", daemon=True)
        self._thread.start()

    def _put(self, item: Any) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=_POLL_SECONDS)
                return True
            except queue.Full:
                pass
        return False

    def _run(self) -> None:
        try:
            buf = np.empty(self._decoder.frame_shape, dtype=np.uint8)
            frame_num = 0
            while not self._stop.is_set() and self._decoder.read_into(buf):
                if not self._put((frame_num, self._transform(frame_num, buf))):
                    return
                frame_num += 1
            self._put(_END)
        except BaseException as exc:  # handed to the consumer, re-raised there
            self._put(_Failure(exc))

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        while True:
            try:
                item = self._queue.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    self._finished = True
                    raise RuntimeError("Frame reader thread stopped without an end marker")
                continue
            if item is _END:
                self._finished = True
                return
            if isinstance(item, _Failure):
                self._finished = True
                raise item.exc
            yield item

    def close(self) -> None:
        if not self._finished:
            self._stop.set()
            self._decoder.abort()
        self._thread.join(timeout=_JOIN_SECONDS)
        if self._thread.is_alive():
            logger.warning(f"Frame reader thread did not stop within {_JOIN_SECONDS:.0f}s")

    def __enter__(self) -> "ThreadedFrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ThreadedFrameWriter:
    """Write frames to an encoder on a background thread.

    ``acquire()`` hands out a buffer of ``frame_shape`` from a fixed pool,
    ``submit()`` queues it, and the thread returns it to the pool once
    ``encoder.write_frame`` is done with it. ``submit()`` returns False after
    the encoder finished early (``write_frame`` returned False, e.g. on
    ``-shortest``); callers stop feeding frames then.

    An exception from ``write_frame`` is re-raised by the next ``acquire()``,
    ``submit()`` or clean exit from the ``with`` block. Leaving the block with
    an exception drops the frames still queued; a clean exit writes them all.
    """

    def __init__(
        self,
        encoder: Any,
        frame_shape: tuple[int, ...],
        pool_size: int = 4,
        queue_size: int = 2,
    ):
        self._encoder = encoder
        self._pool: queue.Queue = queue.Queue()
        for _ in range(pool_size):
            self._pool.put(np.empty(frame_shape, dtype=np.uint8))
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._drop = threading.Event()
        self._eof = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="frame-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            buf = self._queue.get()
            if buf is _END:
                return
            try:
                if self._error is None and not self._drop.is_set() and not self._eof.is_set():
                    if not self._encoder.write_frame(buf):
                        self._eof.set()
            except BaseException as exc:  # re-raised on the main thread
                self._error = exc
            finally:
                self._pool.put(buf)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def acquire(self) -> np.ndarray:
        """A free buffer from the pool; blocks while all of them are queued."""
        self._raise_if_failed()
        return self._pool.get()

    def release(self, buf: np.ndarray) -> None:
        """Return a buffer that will not be submitted."""
        self._pool.put(buf)

    def submit(self, buf: np.ndarray) -> bool:
        """Queue ``buf`` for writing. False once the encoder has finished."""
        self._raise_if_failed()
        if self._eof.is_set():
            self._pool.put(buf)
            return False
        self._queue.put(buf)
        return True

    def close(self, drop: bool = False) -> None:
        if drop:
            self._drop.set()
        self._queue.put(_END)
        # No timeout: the encoder closes its stdin right after this returns, so
        # the thread must be done with it. A slow CPU encoder can take seconds
        # per 4K frame; a hung one blocked the old single-threaded loop too.
        self._thread.join()

    def __enter__(self) -> "ThreadedFrameWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close(drop=exc_type is not None)
        if exc_type is None:
            self._raise_if_failed()
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_frame_threads.py -v`
Expected: PASS, весь файл за пару секунд.

- [ ] **Step 5: Строка в CLAUDE.md**

В таблице «Architecture» в `CLAUDE.md` после строки `` | `app/ffmpeg_pipe.py` | FFmpeg pipe-based video decoder/encoder | `` вставить:

```
| `app/frame_threads.py` | Reader/writer threads that overlap ffmpeg pipe I/O with inference and drawing |
```

- [ ] **Step 6: Commit**

```bash
git add app/frame_threads.py tests/test_frame_threads.py CLAUDE.md
git commit -m "feat(video): reader and writer threads around the ffmpeg pipes"
```

---

### Task 4: Батчевый инференс `batch_inference.py`

**Files:**
- Create: `app/batch_inference.py`
- Create: `tests/test_batch_inference.py`
- Modify: `CLAUDE.md` (таблица Architecture)

**Interfaces:**
- Consumes: `RawDetection` из `app/detection_stabilizer.py` (поля `frame_num, x1, y1, x2, y2, class_id, class_name, confidence`, свойство `bbox`).
- Produces:
  - `InferenceMode(fp16: bool, batch_size: int)` — frozen dataclass.
  - `resolve_inference_mode(fp16: str, batch_size: str, device: str) -> InferenceMode`; `device` — строка вида `"cuda:0"`, `"cpu"`.
  - `model_stride(model) -> int`.
  - `inference_size(width: int, height: int, imgsz: int, stride: int) -> tuple[int, int]` — `(w, h)`.
  - `resize_for_inference(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray` — всегда новый массив.
  - `extract_detections(result, frame_num: int, class_names: dict[int, str], scale: tuple[float, float] = (1.0, 1.0)) -> list[RawDetection]`.
  - `BatchDetector(model, class_names, *, conf, imgsz, max_det, fp16, batch_size, scale)` с методами `add(frame_num, frame)`, `flush()` и атрибутами `detections: dict[int, list[RawDetection]]`, `detected_frames: int`, `batch_size: int`.

  Task 7 использует всё, кроме `resolve_inference_mode`; его использует Task 9.

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_batch_inference.py`:

```python
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.checks import check_imgsz

from batch_inference import (
    BatchDetector,
    InferenceMode,
    extract_detections,
    inference_size,
    model_stride,
    resize_for_inference,
    resolve_inference_mode,
)


def _result(boxes_data):
    """YOLO result double. Each tuple: (x1, y1, x2, y2, class_id, confidence)."""
    result = MagicMock()
    if not boxes_data:
        result.boxes = None
        return result
    boxes = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([b[:4] for b in boxes_data], np.float32)
    boxes.cls.cpu.return_value.numpy.return_value = np.array([b[4] for b in boxes_data], np.float32)
    boxes.conf.cpu.return_value.numpy.return_value = np.array([b[5] for b in boxes_data], np.float32)
    boxes.__len__ = lambda self: len(boxes_data)
    result.boxes = boxes
    return result


class TestInferenceSize:
    @pytest.mark.parametrize("width,height,imgsz,expected", [
        (2560, 1920, 1024, (1024, 768)),
        (1920, 1080, 1024, (1024, 576)),
        (1080, 1920, 1024, (576, 1024)),
        (320, 240, 1024, (1024, 768)),  # LetterBox scales small frames up
        (2560, 1920, 1000, (1024, 768)),  # imgsz is rounded up to a multiple of 32
    ])
    def test_matches_letterbox_arithmetic(self, width, height, imgsz, expected):
        assert inference_size(width, height, imgsz, 32) == expected

    @pytest.mark.parametrize("width,height,imgsz", [
        (2560, 1920, 1024),
        (1920, 1080, 1024),
        (1080, 1920, 1024),
        (1001, 757, 640),
        (320, 240, 1024),
        (2688, 1520, 1000),
        (640, 480, 640),
    ])
    def test_pre_resized_frame_letterboxes_to_the_same_tensor(self, width, height, imgsz):
        """Byte-for-byte guard: if Ultralytics changes how LetterBox resizes,
        the pre-resize on the reader thread would silently change detections."""
        rng = np.random.default_rng(width * height)
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        size = inference_size(width, height, imgsz, 32)
        letterbox = LetterBox(check_imgsz(imgsz, stride=32, min_dim=2), auto=True, stride=32)
        np.testing.assert_array_equal(
            letterbox(image=resize_for_inference(frame, size)),
            letterbox(image=frame),
        )

    def test_same_size_returns_a_copy(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        out = resize_for_inference(frame, (640, 480))
        assert out is not frame
        np.testing.assert_array_equal(out, frame)


class TestModelStride:
    def test_reads_the_largest_stride(self):
        model = MagicMock()
        model.model.stride = torch.tensor([8.0, 16.0, 32.0])
        assert model_stride(model) == 32

    def test_keeps_a_larger_stride(self):
        model = MagicMock()
        model.model.stride = torch.tensor([8.0, 16.0, 32.0, 64.0])
        assert model_stride(model) == 64

    def test_falls_back_to_32(self):
        assert model_stride(MagicMock()) == 32
        assert model_stride(object()) == 32


class TestResolveInferenceMode:
    def test_auto_on_nvidia_is_fp16_batch_8(self):
        with patch("batch_inference.torch.version.hip", None):
            assert resolve_inference_mode("auto", "auto", "cuda:0") == InferenceMode(True, 8)

    def test_auto_on_rocm_is_fp32_batch_1(self):
        with patch("batch_inference.torch.version.hip", "7.2.0"):
            assert resolve_inference_mode("auto", "auto", "cuda:0") == InferenceMode(False, 1)

    def test_auto_on_cpu_is_fp32_batch_1(self):
        assert resolve_inference_mode("auto", "auto", "cpu") == InferenceMode(False, 1)

    def test_explicit_values_override_auto(self):
        with patch("batch_inference.torch.version.hip", "7.2.0"):
            assert resolve_inference_mode("true", "8", "cuda:0") == InferenceMode(True, 8)
        with patch("batch_inference.torch.version.hip", None):
            assert resolve_inference_mode("false", "2", "cuda") == InferenceMode(False, 2)

    def test_fp16_on_cpu_is_refused_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="batch_inference"):
            mode = resolve_inference_mode("true", "4", "cpu")
        assert mode == InferenceMode(False, 4)
        assert "VIDEO_FP16=true ignored" in caplog.text


class TestExtractDetections:
    def test_single_detection(self):
        dets = extract_detections(_result([(10, 20, 100, 200, 0, 0.9)]), 5, {0: "person"})
        assert len(dets) == 1
        assert dets[0].frame_num == 5
        assert dets[0].class_name == "person"
        assert dets[0].confidence == pytest.approx(0.9)
        assert dets[0].bbox == (10, 20, 100, 200)

    def test_scale_maps_boxes_to_the_full_frame(self):
        dets = extract_detections(_result([(10, 20, 100, 200, 0, 0.9)]), 0, {0: "person"}, (2.5, 2.5))
        assert dets[0].bbox == (25, 50, 250, 500)

    def test_unknown_class_id(self):
        dets = extract_detections(_result([(1, 2, 3, 4, 7, 0.5)]), 0, {0: "person"})
        assert dets[0].class_name == "class_7"

    def test_empty_boxes(self):
        assert extract_detections(_result([]), 0, {0: "person"}) == []


def _model(boxes_data=((10, 20, 100, 200, 0, 0.9),), oom_above=None):
    """predict() double answering one result per frame; ``oom_above`` makes
    batches larger than that raise CUDA out of memory."""
    model = MagicMock()

    def predict(source, **kwargs):
        if oom_above is not None and len(source) > oom_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return [_result(list(boxes_data)) for _ in source]

    model.predict.side_effect = predict
    return model


def _detector(model, batch_size=3, fp16=False, scale=(1.0, 1.0)):
    return BatchDetector(
        model, {0: "person"}, conf=0.24, imgsz=640, max_det=100,
        fp16=fp16, batch_size=batch_size, scale=scale,
    )


def _frames(n):
    return [np.full((4, 4, 3), i, np.uint8) for i in range(n)]


class TestBatchDetector:
    def test_batches_frames_and_flushes_the_rest(self):
        model = _model()
        detector = _detector(model, batch_size=3)
        for n, frame in enumerate(_frames(7)):
            detector.add(n * 2, frame)
        detector.flush()

        assert [len(c.kwargs["source"]) for c in model.predict.call_args_list] == [3, 3, 1]
        assert sorted(detector.detections) == [0, 2, 4, 6, 8, 10, 12]
        assert detector.detected_frames == 7

    def test_frames_without_boxes_are_counted_but_not_stored(self):
        detector = _detector(_model(boxes_data=()), batch_size=2)
        for n, frame in enumerate(_frames(3)):
            detector.add(n, frame)
        detector.flush()
        assert detector.detections == {}
        assert detector.detected_frames == 3

    def test_predict_arguments(self):
        model = _model()
        detector = _detector(model, batch_size=1)
        detector.add(0, _frames(1)[0])
        kwargs = model.predict.call_args.kwargs
        assert kwargs["conf"] == 0.24
        assert kwargs["imgsz"] == 640
        assert kwargs["max_det"] == 100
        assert kwargs["verbose"] is False
        assert "quantize" not in kwargs

    def test_fp16_passes_quantize_16(self):
        model = _model()
        detector = _detector(model, batch_size=1, fp16=True)
        detector.add(0, _frames(1)[0])
        assert model.predict.call_args.kwargs["quantize"] == 16

    def test_scale_applies_to_stored_boxes(self):
        detector = _detector(_model(), batch_size=1, scale=(2.0, 2.0))
        detector.add(0, _frames(1)[0])
        assert detector.detections[0][0].bbox == (20, 40, 200, 400)

    def test_out_of_memory_halves_the_batch_and_loses_nothing(self, caplog):
        model = _model(oom_above=2)
        detector = _detector(model, batch_size=8)
        with caplog.at_level(logging.WARNING, logger="batch_inference"):
            for n, frame in enumerate(_frames(8)):
                detector.add(n, frame)
            detector.flush()

        assert detector.batch_size == 2
        assert sorted(detector.detections) == list(range(8))
        assert detector.detected_frames == 8
        assert "out of memory at batch 8" in caplog.text
        assert "out of memory at batch 4" in caplog.text

    def test_out_of_memory_at_batch_1_is_raised(self):
        detector = _detector(_model(oom_above=0), batch_size=1)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            detector.add(0, _frames(1)[0])

    def test_result_count_mismatch_is_an_error(self):
        model = MagicMock()
        model.predict.return_value = [_result([(1, 2, 3, 4, 0, 0.9)])]
        detector = _detector(model, batch_size=2)
        with pytest.raises(RuntimeError, match="1 results for 2 frames"):
            for n, frame in enumerate(_frames(2)):
                detector.add(n, frame)
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_batch_inference.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'batch_inference'`.

- [ ] **Step 3: Написать модуль**

Создать `app/batch_inference.py`:

```python
"""Batched YOLO inference for pass 1 of the video annotator.

``inference_size`` and ``resize_for_inference`` shrink a frame exactly the way
Ultralytics' LetterBox does (same size, cv2 INTER_LINEAR), so ``predict()`` on
the pre-resized frame sees the same input tensor as on the full frame while the
resize runs on the reader thread. ``BatchDetector`` feeds those frames to
``predict()`` in batches and maps the boxes back to the full frame.
"""
import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics.utils.checks import check_imgsz

from detection_stabilizer import RawDetection

logger = logging.getLogger(__name__)

_NVIDIA_AUTO_BATCH = 8


@dataclass(frozen=True, slots=True)
class InferenceMode:
    fp16: bool
    batch_size: int


def resolve_inference_mode(fp16: str, batch_size: str, device: str) -> InferenceMode:
    """Turn the VIDEO_FP16 and VIDEO_BATCH_SIZE values into a mode for one model.

    ``auto`` means FP16 with batch 8 on NVIDIA and FP32 with batch 1 elsewhere.
    ROCm builds of torch also call the GPU ``cuda``; ``torch.version.hip`` tells
    them apart. FP16 on CPU is refused: PyTorch has no fast half kernels there.
    """
    device_type = torch.device(device).type
    nvidia = device_type == "cuda" and torch.version.hip is None
    use_fp16 = nvidia if fp16 == "auto" else fp16 == "true"
    if use_fp16 and device_type == "cpu":
        logger.warning("VIDEO_FP16=true ignored: the model runs on CPU, staying in FP32")
        use_fp16 = False
    if batch_size == "auto":
        size = _NVIDIA_AUTO_BATCH if nvidia else 1
    else:
        size = int(batch_size)
    return InferenceMode(fp16=use_fp16, batch_size=size)


def model_stride(model: Any) -> int:
    """Largest stride of a YOLO model, floored at 32 as Ultralytics does."""
    try:
        stride = int(max(model.model.stride))
    except (AttributeError, TypeError, ValueError):
        stride = 32
    return max(stride, 32)


def inference_size(width: int, height: int, imgsz: int, stride: int) -> tuple[int, int]:
    """(w, h) that Ultralytics' LetterBox resizes a width x height frame to."""
    target_h, target_w = check_imgsz(imgsz, stride=stride, min_dim=2)
    r = min(target_h / height, target_w / width)
    return round(width * r), round(height * r)


def resize_for_inference(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A new array of ``size`` (w, h), resized the way LetterBox resizes."""
    if (frame.shape[1], frame.shape[0]) == size:
        return frame.copy()
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def extract_detections(
    result: Any,
    frame_num: int,
    class_names: dict[int, str],
    scale: tuple[float, float] = (1.0, 1.0),
) -> list[RawDetection]:
    """RawDetections of one YOLO result, boxes multiplied by ``scale`` (sx, sy)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    sx, sy = scale
    detections = []
    for i in range(len(cls)):
        class_id = int(cls[i])
        detections.append(RawDetection(
            frame_num=frame_num,
            x1=int(xyxy[i][0] * sx), y1=int(xyxy[i][1] * sy),
            x2=int(xyxy[i][2] * sx), y2=int(xyxy[i][3] * sy),
            class_id=class_id,
            class_name=class_names.get(class_id, f"class_{class_id}"),
            confidence=float(conf[i]),
        ))
    return detections


class BatchDetector:
    """Run YOLO on frames in batches and collect RawDetections per frame.

    Frames come already resized by ``resize_for_inference``; ``scale`` maps
    their boxes back to the full frame. When the GPU runs out of memory at a
    batch above 1, the batch is halved and the same frames run again.
    """

    def __init__(
        self,
        model: Any,
        class_names: dict[int, str],
        *,
        conf: float,
        imgsz: int,
        max_det: int,
        fp16: bool,
        batch_size: int,
        scale: tuple[float, float],
    ):
        self._model = model
        self._class_names = class_names
        self._conf = conf
        self._imgsz = imgsz
        self._max_det = max_det
        self._fp16 = fp16
        self._scale = scale
        self.batch_size = batch_size
        self.detections: dict[int, list[RawDetection]] = {}
        self.detected_frames = 0
        self._pending: list[tuple[int, np.ndarray]] = []

    def add(self, frame_num: int, frame: np.ndarray) -> None:
        self._pending.append((frame_num, frame))
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        items, self._pending = self._pending, []
        start = 0
        while start < len(items):
            chunk = items[start:start + self.batch_size]
            results = self._predict([frame for _, frame in chunk])
            if results is None:  # out of GPU memory: batch_size was halved, retry
                continue
            if len(results) != len(chunk):
                raise RuntimeError(
                    f"YOLO returned {len(results)} results for {len(chunk)} frames"
                )
            for (frame_num, _), result in zip(chunk, results):
                self.detected_frames += 1
                detections = extract_detections(result, frame_num, self._class_names, self._scale)
                if detections:
                    self.detections[frame_num] = detections
            start += len(chunk)

    def _predict(self, frames: list[np.ndarray]) -> list[Any] | None:
        kwargs: dict[str, Any] = {
            "source": frames, "conf": self._conf, "imgsz": self._imgsz,
            "max_det": self._max_det, "verbose": False,
        }
        if self._fp16:
            kwargs["quantize"] = 16
        try:
            return self._model.predict(**kwargs)
        except torch.cuda.OutOfMemoryError:
            if self.batch_size == 1:
                raise
        # Out of memory at batch > 1. The except block has ended, so the
        # traceback no longer pins the failed batch's tensors.
        old = self.batch_size
        self.batch_size = max(1, old // 2)
        torch.cuda.empty_cache()
        logger.warning(f"GPU out of memory at batch {old}, retrying with batch {self.batch_size}")
        return None
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_batch_inference.py -v`
Expected: PASS. Если упал `test_pre_resized_frame_letterboxes_to_the_same_tensor`, значит, установленная ultralytics ресайзит иначе. Сверить `inference_size` с `LetterBox.get_params` этой версии, а не ослаблять тест.

- [ ] **Step 5: Строка в CLAUDE.md**

В таблице «Architecture» после строки `` | `app/video_annotator.py` | YOLO detection + hold mode video annotation pipeline | `` вставить:

```
| `app/batch_inference.py` | Video pass 1: LetterBox-exact pre-resize, batched predict, FP16/batch mode per device |
```

- [ ] **Step 6: Commit**

```bash
git add app/batch_inference.py tests/test_batch_inference.py CLAUDE.md
git commit -m "feat(video): batched YOLO inference with LetterBox-exact pre-resize"
```

---

### Task 5: Отрисовка на кадре yuv420p

**Files:**
- Modify: `app/visualization.py`
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: ничего из других задач.
- Produces:
  - `bgr_to_yuv601(bgr: tuple[int, int, int]) -> tuple[int, int, int]`;
  - `bgr_patch_to_yuv420(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (стороны патча чётные);
  - `yuv420_planes(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — изменяемые виды Y, U, V плоского кадра;
  - `DetectionVisualizer.draw_detection_yuv420(frame, width, height, det, line_width, show_labels, show_conf, font_scale, text_thickness) -> None` рисует на месте.

  `draw_detection` (BGR) выдаёт те же пиксели, что и раньше. Task 7 вызывает `draw_detection_yuv420` по именам аргументов.

- [ ] **Step 1: Написать падающие тесты**

В `tests/test_visualization.py` заменить шапку

```python
from unittest.mock import MagicMock

import pytest
import numpy as np

from visualization import Color, DetectionBox, DetectionVisualizer, encode_image_to_bytes
```

на

```python
from unittest.mock import MagicMock

import cv2
import pytest
import numpy as np

from visualization import (
    Color,
    DetectionBox,
    DetectionVisualizer,
    bgr_patch_to_yuv420,
    bgr_to_yuv601,
    encode_image_to_bytes,
    yuv420_planes,
)
```

В конец файла дописать:

```python


class TestYuvHelpers:
    @pytest.mark.parametrize("bgr", [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128), (255, 255, 255)])
    def test_bgr_to_yuv601_matches_opencv_i420(self, bgr):
        """OpenCV's I420 conversion is limited-range BT.601 too (1 level of rounding slack)."""
        image = np.full((2, 2, 3), bgr, np.uint8)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        expected = (int(yuv[0]), int(yuv[4]), int(yuv[5]))
        assert all(abs(a - b) <= 1 for a, b in zip(bgr_to_yuv601(bgr), expected))

    def test_patch_conversion_subsamples_chroma(self):
        patch = np.zeros((4, 6, 3), np.uint8)
        patch[:, :] = (255, 0, 0)
        y, u, v = bgr_patch_to_yuv420(patch)
        assert y.shape == (4, 6) and u.shape == (2, 3) and v.shape == (2, 3)
        assert (int(y[0, 0]), int(u[0, 0]), int(v[0, 0])) == bgr_to_yuv601((255, 0, 0))

    def test_planes_are_views_into_the_frame(self):
        frame = np.zeros(5 * 3 + 2 * 3 * 2, np.uint8)  # 5x3 frame, chroma 3x2
        y, u, v = yuv420_planes(frame, 5, 3)
        assert (y.shape, u.shape, v.shape) == ((3, 5), (2, 3), (2, 3))
        v[:] = 7
        assert frame[-6:].tolist() == [7] * 6


def _yuv_frame(width, height, seed=0):
    """A random yuv420p frame and its BGR twin (OpenCV's BT.601 conversion)."""
    rng = np.random.default_rng(seed)
    bgr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    bgr = cv2.GaussianBlur(bgr, (7, 7), 0)  # smooth, like video
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    return yuv, cv2.cvtColor(yuv.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420)


def _touched_mask(visualizer, width, height, det, **kwargs):
    """Pixels the BGR renderer paints for ``det``: draw on a sentinel canvas."""
    canvas = np.full((height, width, 3), (1, 2, 3), np.uint8)
    visualizer.draw_detection(canvas, det, **kwargs)
    return (canvas != (1, 2, 3)).any(axis=2)


class TestDrawDetectionYuv420:
    W, H = 320, 240
    KW = dict(line_width=2, show_labels=True, show_conf=True, font_scale=0.5, text_thickness=1)

    def _draw(self, frame, det, width=None, height=None, **overrides):
        kwargs = {**self.KW, **overrides}
        visualizer = DetectionVisualizer(class_names={0: "person"})
        visualizer.draw_detection_yuv420(frame, width or self.W, height or self.H, det, **kwargs)

    def test_pixels_away_from_the_drawing_are_untouched(self):
        yuv, _ = _yuv_frame(self.W, self.H)
        before = yuv.copy()
        det = DetectionBox(x1=41, y1=57, x2=201, y2=181, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det)

        mask = _touched_mask(DetectionVisualizer({0: "person"}), self.W, self.H, det, **self.KW)
        near = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        y_before, u_before, v_before = yuv420_planes(before, self.W, self.H)
        y_after, u_after, v_after = yuv420_planes(yuv, self.W, self.H)
        np.testing.assert_array_equal(y_after[~near], y_before[~near])
        chroma_near = near.reshape(self.H // 2, 2, self.W // 2, 2).any(axis=(1, 3))
        np.testing.assert_array_equal(u_after[~chroma_near], u_before[~chroma_near])
        np.testing.assert_array_equal(v_after[~chroma_near], v_before[~chroma_near])

    def test_luma_matches_the_bgr_renderer(self):
        """Y is exact: every pixel the BGR renderer paints gets the luma of its colour."""
        yuv, bgr = _yuv_frame(self.W, self.H, seed=1)
        det = DetectionBox(x1=41, y1=57, x2=201, y2=181, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det)
        DetectionVisualizer({0: "person"}).draw_detection(bgr, det, **self.KW)

        reference = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        mask = _touched_mask(DetectionVisualizer({0: "person"}), self.W, self.H, det, **self.KW)
        ours, _, _ = yuv420_planes(yuv, self.W, self.H)
        ref, _, _ = yuv420_planes(reference, self.W, self.H)
        assert np.abs(ours.astype(int) - ref.astype(int))[mask].max() <= 1

    def test_even_aligned_outline_chroma_matches_the_bgr_renderer(self):
        """On even coordinates the half-resolution outline lands on the same
        chroma blocks OpenCV's I420 conversion of the BGR drawing colours."""
        kwargs = {**self.KW, "show_labels": False, "show_conf": False}
        yuv, bgr = _yuv_frame(self.W, self.H, seed=2)
        det = DetectionBox(x1=40, y1=60, x2=200, y2=180, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det, show_labels=False, show_conf=False)
        DetectionVisualizer({0: "person"}).draw_detection(bgr, det, **kwargs)

        reference = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        mask = _touched_mask(DetectionVisualizer({0: "person"}), self.W, self.H, det, **kwargs)
        chroma_mask = mask.reshape(self.H // 2, 2, self.W // 2, 2).any(axis=(1, 3))
        _, u_ours, v_ours = yuv420_planes(yuv, self.W, self.H)
        _, u_ref, v_ref = yuv420_planes(reference, self.W, self.H)
        assert np.abs(u_ours.astype(int) - u_ref.astype(int))[chroma_mask].max() <= 1
        assert np.abs(v_ours.astype(int) - v_ref.astype(int))[chroma_mask].max() <= 1

    def test_label_text_is_white_on_the_class_colour(self):
        yuv = np.zeros(self.W * self.H * 3 // 2, np.uint8)
        det = DetectionBox(x1=40, y1=60, x2=200, y2=180, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det)
        y, _, _ = yuv420_planes(yuv, self.W, self.H)
        label_rows = y[40:60, 40:120]
        assert (label_rows > 200).any()  # white text
        assert (label_rows == bgr_to_yuv601((255, 0, 0))[0]).any()  # class-0 background

    def test_label_is_clipped_at_the_right_edge(self):
        yuv = np.zeros(self.W * self.H * 3 // 2, np.uint8)
        det = DetectionBox(x1=300, y1=60, x2=318, y2=100, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det)
        y, _, _ = yuv420_planes(yuv, self.W, self.H)
        assert (y[40:60, self.W - 1] != 0).any()

    def test_no_label_when_labels_and_confidence_are_hidden(self):
        yuv = np.zeros(self.W * self.H * 3 // 2, np.uint8)
        det = DetectionBox(x1=40, y1=60, x2=200, y2=180, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det, show_labels=False, show_conf=False)
        y, _, _ = yuv420_planes(yuv, self.W, self.H)
        assert (y[20:50, 40:120] == 0).all()  # nothing above the box
        assert (y[60, 40:200] != 0).all()  # the top edge is drawn

    def test_line_width_1(self):
        yuv = np.zeros(self.W * self.H * 3 // 2, np.uint8)
        det = DetectionBox(x1=40, y1=60, x2=200, y2=180, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det, line_width=1, show_labels=False, show_conf=False)
        y, u, _ = yuv420_planes(yuv, self.W, self.H)
        assert (y[120, 40] != 0) and (y[120, 42] == 0)
        assert u[60, 20] == bgr_to_yuv601((255, 0, 0))[1]

    def test_odd_frame_size(self):
        width, height = 321, 241
        yuv = np.zeros(width * height + 2 * 161 * 121, np.uint8)
        det = DetectionBox(x1=250, y1=200, x2=320, y2=240, class_id=0, class_name="person", confidence=0.5)
        self._draw(yuv, det, width=width, height=height)
        y, _, _ = yuv420_planes(yuv, width, height)
        assert (y[240, 250:320] != 0).all()
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_visualization.py -v`
Expected: FAIL с `ImportError: cannot import name 'bgr_patch_to_yuv420' from 'visualization'`.

- [ ] **Step 3: Добавить геометрию подписи и BT.601-помощники**

В `app/visualization.py` сразу после класса `DetectionBox` (перед `class DetectionVisualizer:`) вставить:

```python


@dataclass(frozen=True, slots=True)
class _LabelBox:
    """Label background rectangle (inclusive corners) and the text origin."""

    x0: int
    y0: int
    x1: int
    y1: int
    text_x: int
    text_y: int


# Limited-range BT.601, the matrix swscale applies when the encoder turns
# untagged BGR frames into yuv420p. Rows give Y, U, V from R, G, B in 0..255.
_BT601_FROM_RGB = np.array(
    [
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.0],
        [112.0, -93.786, -18.214],
    ],
    dtype=np.float32,
) / 255.0
_BT601_OFFSET = np.array([16.0, 128.0, 128.0], dtype=np.float32)


def bgr_to_yuv601(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """One BGR colour as limited-range BT.601 (Y, U, V)."""
    rgb = np.array(bgr[::-1], dtype=np.float32)
    y, u, v = np.clip(np.rint(_BT601_FROM_RGB @ rgb + _BT601_OFFSET), 0, 255)
    return int(y), int(u), int(v)


def bgr_patch_to_yuv420(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y, U, V planes of a BGR patch with even sides; U and V are 2x2 means."""
    yuv = patch[..., ::-1].astype(np.float32) @ _BT601_FROM_RGB.T + _BT601_OFFSET
    h, w = patch.shape[:2]
    chroma = yuv[..., 1:].reshape(h // 2, 2, w // 2, 2, 2).mean(axis=(1, 3))

    def to_u8(plane: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(plane), 0, 255).astype(np.uint8)

    return to_u8(yuv[..., 0]), to_u8(chroma[..., 0]), to_u8(chroma[..., 1])


def yuv420_planes(
    frame: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Writable Y, U, V views into a flat yuv420p frame."""
    chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
    y_end = width * height
    u_end = y_end + chroma_w * chroma_h
    return (
        frame[:y_end].reshape(height, width),
        frame[y_end:u_end].reshape(chroma_h, chroma_w),
        frame[u_end:u_end + chroma_w * chroma_h].reshape(chroma_h, chroma_w),
    )
```

- [ ] **Step 4: Перевести BGR-отрисовку на общие помощники и добавить YUV-отрисовку**

В `app/visualization.py` заменить методы `draw_detection` и `_draw_label_with_background` целиком (всё от `    def draw_detection(` до конца `_draw_label_with_background`, перед `def encode_image_to_bytes(`) на:

```python
    def draw_detection(
            self,
            image: np.ndarray,
            det: DetectionBox,
            line_width: int,
            show_labels: bool,
            show_conf: bool,
            font_scale: float,
            text_thickness: int
    ) -> None:
        """Draw a single detection box with optional label."""
        color = self._get_class_color(det.class_id)
        color_tuple = color.as_tuple()

        # Draw bounding box
        cv2.rectangle(
            image,
            (det.x1, det.y1),
            (det.x2, det.y2),
            color_tuple,
            line_width
        )

        label = self._label_text(det, show_labels, show_conf)
        if label is None:
            return
        self._draw_label_with_background(
            image, label, det.x1, det.y1,
            color_tuple, font_scale, text_thickness
        )

    def draw_detection_yuv420(
            self,
            frame: np.ndarray,
            width: int,
            height: int,
            det: DetectionBox,
            line_width: int,
            show_labels: bool,
            show_conf: bool,
            font_scale: float,
            text_thickness: int
    ) -> None:
        """Draw a single detection onto a flat yuv420p frame, in place.

        The box goes onto Y with ``line_width`` and onto U and V at half
        resolution with ``(line_width + 1) // 2``. The label is drawn in BGR
        on a patch aligned to even coordinates, converted with BT.601 and
        pasted: luma exactly over the label rectangle, chroma over every 2x2
        block the rectangle touches.
        """
        y_plane, u_plane, v_plane = yuv420_planes(frame, width, height)
        bgr = self._get_class_color(det.class_id).as_tuple()
        luma, cb, cr = bgr_to_yuv601(bgr)

        cv2.rectangle(y_plane, (det.x1, det.y1), (det.x2, det.y2), luma, line_width)
        chroma_width = (line_width + 1) // 2
        corner1, corner2 = (det.x1 // 2, det.y1 // 2), (det.x2 // 2, det.y2 // 2)
        cv2.rectangle(u_plane, corner1, corner2, cb, chroma_width)
        cv2.rectangle(v_plane, corner1, corner2, cr, chroma_width)

        label = self._label_text(det, show_labels, show_conf)
        if label is None:
            return
        box = self._label_box(label, det.x1, det.y1, font_scale, text_thickness)

        # Patch over the label rectangle, grown to even edges for 4:2:0 chroma.
        px0, py0 = box.x0 - box.x0 % 2, box.y0 - box.y0 % 2
        px1, py1 = box.x1 + 1 + (box.x1 + 1) % 2, box.y1 + 1 + (box.y1 + 1) % 2
        patch = np.empty((py1 - py0, px1 - px0, 3), dtype=np.uint8)
        patch[:] = bgr
        cv2.putText(
            patch, label, (box.text_x - px0, box.text_y - py0), self._font,
            font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA,
        )
        patch_y, patch_u, patch_v = bgr_patch_to_yuv420(patch)

        # Luma: exactly the label rectangle, clipped to the frame.
        x0, y0 = max(box.x0, 0), max(box.y0, 0)
        x1, y1 = min(box.x1 + 1, width), min(box.y1 + 1, height)
        if x0 < x1 and y0 < y1:
            y_plane[y0:y1, x0:x1] = patch_y[y0 - py0:y1 - py0, x0 - px0:x1 - px0]

        # Chroma: the 2x2 blocks under the patch, clipped to the planes.
        cx0, cy0 = max(px0, 0) // 2, max(py0, 0) // 2
        cx1, cy1 = min(px1 // 2, u_plane.shape[1]), min(py1 // 2, u_plane.shape[0])
        if cx0 < cx1 and cy0 < cy1:
            rows = slice(cy0 - py0 // 2, cy1 - py0 // 2)
            cols = slice(cx0 - px0 // 2, cx1 - px0 // 2)
            u_plane[cy0:cy1, cx0:cx1] = patch_u[rows, cols]
            v_plane[cy0:cy1, cx0:cx1] = patch_v[rows, cols]

    @staticmethod
    def _label_text(det: DetectionBox, show_labels: bool, show_conf: bool) -> str | None:
        """The label for a detection, or None when both parts are hidden."""
        parts = []
        if show_labels:
            parts.append(det.class_name)
        if show_conf:
            parts.append(f"{det.confidence:.2f}")
        return " ".join(parts) if parts else None

    def _label_box(
            self, label: str, x: int, y: int, font_scale: float, thickness: int
    ) -> _LabelBox:
        """Where the label background and text go for a box whose top-left is (x, y)."""
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self._font, font_scale, thickness
        )
        padding = 4
        label_y = max(text_h + baseline + padding, y)
        return _LabelBox(
            x0=x,
            y0=label_y - text_h - baseline - padding,
            x1=x + text_w + padding,
            y1=label_y,
            text_x=x + padding // 2,
            text_y=label_y - baseline - padding // 2,
        )

    def _draw_label_with_background(
            self,
            image: np.ndarray,
            label: str,
            x: int,
            y: int,
            bg_color: tuple[int, int, int],
            font_scale: float,
            thickness: int
    ) -> None:
        """Draw text label with background."""
        box = self._label_box(label, x, y, font_scale, thickness)

        # Background rectangle
        cv2.rectangle(image, (box.x0, box.y0), (box.x1, box.y1), bg_color, -1)

        # Text
        cv2.putText(
            image,
            label,
            (box.text_x, box.text_y),
            self._font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )


```

- [ ] **Step 5: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_visualization.py -v`
Expected: PASS.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS целиком. `/detect/visualize` и `draw_yolo_results` используют прежнюю BGR-отрисовку.

- [ ] **Step 6: Commit**

```bash
git add app/visualization.py tests/test_visualization.py
git commit -m "feat(video): draw detections straight onto yuv420p frames"
```

---

### Task 6: Отдельный экземпляр модели для видео-задач

**Files:**
- Modify: `app/model_manager.py`
- Create: `tests/test_model_manager.py`

**Interfaces:**
- Consumes: существующие `ModelManager._load_model_sync(model_name, device) -> ModelEntry`, `_load_model_async`, `CachedModelEntry`, `ModelEntry(model, visualizer, model_name, device)`.
- Produces: `async ModelManager.get_video_model(model_name: str | None = None) -> ModelEntry`. `ValueError`, если имени нет и предзагруженных моделей нет; `RuntimeError("Failed to load model ...")`, если загрузка упала. Вызывает worker в Task 9.

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_model_manager.py`:

```python
import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from model_manager import ModelEntry, ModelManager


def _entry(name: str, device: str) -> ModelEntry:
    return ModelEntry(model=MagicMock(), visualizer=MagicMock(), model_name=name, device=device)


@pytest.fixture
def manager():
    mm = ModelManager(default_device="cpu", ttl_seconds=60)
    mm._preloaded["yolo26s.pt"] = _entry("yolo26s.pt", "cuda:0")
    mm._preloaded["yolo26x.pt"] = _entry("yolo26x.pt", "cuda:0")
    return mm


def _loader(delay: float = 0.0):
    """Stand-in for _load_model_sync: a fresh entry per call, optionally slow."""
    def load(model_name, device):
        if delay:
            time.sleep(delay)
        return _entry(model_name, device)
    return MagicMock(side_effect=load)


class TestGetVideoModel:
    async def test_separate_instance_on_the_preloaded_device(self, manager):
        loader = _loader()
        with patch.object(manager, "_load_model_sync", loader):
            entry = await manager.get_video_model("yolo26x.pt")

        loader.assert_called_once_with("yolo26x.pt", "cuda:0")
        assert entry is not manager._preloaded["yolo26x.pt"]
        assert entry.model is not manager._preloaded["yolo26x.pt"].model
        assert entry.device == "cuda:0"

    async def test_none_means_the_default_model(self, manager):
        loader = _loader()
        with patch.object(manager, "_load_model_sync", loader):
            entry = await manager.get_video_model(None)
        assert entry.model_name == "yolo26s.pt"

    async def test_no_default_model_raises_value_error(self):
        with pytest.raises(ValueError):
            await ModelManager(default_device="cpu").get_video_model(None)

    async def test_not_preloaded_model_uses_the_default_device(self, manager):
        loader = _loader()
        with patch.object(manager, "_load_model_sync", loader):
            await manager.get_video_model("yolo26m.pt")
        loader.assert_called_once_with("yolo26m.pt", "cpu")

    async def test_cached_between_calls(self, manager):
        loader = _loader()
        with patch.object(manager, "_load_model_sync", loader):
            first = await manager.get_video_model("yolo26x.pt")
            second = await manager.get_video_model("yolo26x.pt")
        assert first is second
        assert loader.call_count == 1

    async def test_concurrent_calls_load_once(self, manager):
        loader = _loader(delay=0.1)
        with patch.object(manager, "_load_model_sync", loader):
            first, second = await asyncio.gather(
                manager.get_video_model("yolo26x.pt"),
                manager.get_video_model("yolo26x.pt"),
            )
        assert first is second
        assert loader.call_count == 1

    async def test_load_failure_raises_runtime_error(self, manager):
        loader = MagicMock(side_effect=OSError("file is corrupt"))
        with patch.object(manager, "_load_model_sync", loader):
            with pytest.raises(RuntimeError, match="file is corrupt"):
                await manager.get_video_model("yolo26x.pt")
        assert "yolo26x.pt" not in manager._video_models

    async def test_idle_video_model_is_evicted(self, manager):
        with patch.object(manager, "_load_model_sync", _loader()):
            await manager.get_video_model("yolo26x.pt")
        manager._video_models["yolo26x.pt"].last_used_at -= 120

        assert await manager.cleanup_expired() == 1
        assert "yolo26x.pt" not in manager._video_models
        assert "yolo26x.pt" in manager._preloaded

    async def test_recently_used_video_model_is_kept(self, manager):
        with patch.object(manager, "_load_model_sync", _loader()):
            await manager.get_video_model("yolo26x.pt")
        assert await manager.cleanup_expired() == 0
        assert "yolo26x.pt" in manager._video_models

    async def test_shutdown_clears_video_models(self, manager):
        with patch.object(manager, "_load_model_sync", _loader()):
            await manager.get_video_model("yolo26x.pt")
        await manager.shutdown()
        assert manager._video_models == {}
```

(`asyncio_mode = "auto"` в `pyproject.toml`, поэтому `async def` тесты не требуют маркера.)

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_model_manager.py -v`
Expected: FAIL с `AttributeError: 'ModelManager' object has no attribute 'get_video_model'`.

- [ ] **Step 3: Реализовать `get_video_model`**

В `app/model_manager.py` в `__init__` после строки `        self._cached: dict[str, CachedModelEntry] = {}` вставить:

```python
        # Instances of their own for video annotation jobs, see get_video_model()
        self._video_models: dict[str, CachedModelEntry] = {}
```

Перед `    async def cleanup_expired(self) -> int:` вставить:

```python
    async def get_video_model(self, model_name: str | None = None) -> ModelEntry:
        """A model instance of its own for video annotation jobs.

        Ultralytics converts a model's weights between FP16 and FP32 in place
        and rebuilds its predictor whenever ``quantize`` changes, so the video
        pipeline must not share the instance ``/detect`` uses. The instance is
        loaded from the same file onto the same device as ``get_model`` would
        use, cached, and evicted after ``ttl_seconds`` without a job.

        Raises:
            ValueError: If no model name provided and no default model available.
            RuntimeError: If model loading fails.
        """
        if model_name is None:
            if not self._preloaded:
                raise ValueError("No model specified and no default model available")
            model_name = next(iter(self._preloaded))

        cached = self._video_models.get(model_name)
        if cached is not None:
            cached.touch()
            return cached.entry

        lock_key = f"video:{model_name}"
        async with self._global_lock:
            if lock_key not in self._loading_locks:
                self._loading_locks[lock_key] = asyncio.Lock()
            lock = self._loading_locks[lock_key]

        async with lock:
            cached = self._video_models.get(model_name)
            if cached is not None:
                cached.touch()
                return cached.entry

            preloaded = self._preloaded.get(model_name)
            device = preloaded.device if preloaded is not None else self.default_device
            try:
                entry = await self._load_model_async(model_name, device)
                self._video_models[model_name] = CachedModelEntry(entry=entry)
                logger.info(f"Video model {model_name} loaded on {device} (TTL: {self.ttl_seconds}s)")
                return entry
            except Exception as e:
                logger.error(f"Failed to load video model {model_name}: {e}")
                raise RuntimeError(f"Failed to load model {model_name}: {e}") from e
            finally:
                async with self._global_lock:
                    self._loading_locks.pop(lock_key, None)

```

В `cleanup_expired` перед строкой `        if evicted > 0 and self.default_device.startswith("cuda") and torch.cuda.is_available():` вставить:

```python
        # A job longer than the TTL keeps its own reference to the model, so
        # evicting the entry mid-job only means the next job loads it again.
        expired_video = [
            name for name, cached in self._video_models.items()
            if cached.is_expired(self.ttl_seconds)
        ]
        for model_name in expired_video:
            cached = self._video_models.pop(model_name, None)
            if cached:
                logger.info(f"Evicting idle video model: {model_name}")
                del cached.entry.model
                del cached.entry.visualizer
                evicted += 1

```

В `shutdown` после строки `        self._cached.clear()` вставить:

```python

        for model_name, cached in list(self._video_models.items()):
            logger.debug(f"Unloading video model: {model_name}")
            del cached.entry.model
            del cached.entry.visualizer
        self._video_models.clear()
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_model_manager.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/model_manager.py tests/test_model_manager.py
git commit -m "feat(models): separate model instances for video annotation jobs"
```

---

### Task 7: `VideoAnnotator` на новых компонентах

**Files:**
- Modify: `app/video_annotator.py`
- Test: `tests/test_video_annotator.py`
- Modify: `CLAUDE.md` (строка `ffmpeg_pipe.py`, абзац Video Annotation), `.claude/rules/api.md`

**Interfaces:**
- Consumes: из Task 2 — `FFmpegDecoder(..., pix_fmt=...)` с `frame_shape` и `read_into`, `FFmpegEncoder(..., pix_fmt=...)`, `frame_shape()`; из Task 3 — `ThreadedFrameReader`, `ThreadedFrameWriter`; из Task 4 — `BatchDetector`, `inference_size`, `model_stride`, `resize_for_inference`; из Task 5 — `DetectionVisualizer.draw_detection_yuv420`.
- Produces: `VideoAnnotator(model, visualizer, class_names, hw_config, codec="h264", crf=18, stabilizer_config=None, fp16: bool = False, batch_size: int = 1)`; `VideoAnnotator._draw_detections(frame, width, height, detections, params, font_scale)`. Метод `_extract_raw_detections` удаляется (его заменяет `batch_inference.extract_detections`). Сигнатура `annotate()` и `AnnotationStats` не меняются. Task 9 передаёт `fp16` и `batch_size`.

- [ ] **Step 1: Перевести моки тестов на `read_into`**

Правки в `tests/test_video_annotator.py`.

a) После строки `from detection_stabilizer import StabilizerConfig` добавить строку `from ffmpeg_pipe import frame_shape`.

b) Перед строкой `# --- Fixtures ---` вставить:

```python
def _make_decoder_mock(frames: list[np.ndarray]):
    """FFmpegDecoder double: read_into() succeeds once per frame, then reports EOF.

    The frames only set the count; their content is irrelevant to these tests.
    ``frame_shape`` is filled in by ``_decoder_cls`` from the requested pix_fmt.
    """
    remaining = [len(frames)]

    def read_into(buf):
        if remaining[0] == 0:
            return False
        remaining[0] -= 1
        buf.fill(0)
        return True

    mock_decoder = MagicMock()
    mock_decoder.read_into.side_effect = read_into
    mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
    mock_decoder.__exit__ = MagicMock(return_value=False)
    return mock_decoder


def _decoder_cls(*decoders):
    """FFmpegDecoder stand-in handing out ``decoders`` in order.

    Each one gets the ``frame_shape`` of the pix_fmt it is opened with, so a
    decoder serves pass 1 (bgr24) or pass 2 (yuv420p) alike.
    """
    pending = list(decoders)

    def construct(input_path, width, height, hw_config, pix_fmt="bgr24"):
        decoder = pending.pop(0)
        decoder.frame_shape = frame_shape(width, height, pix_fmt)
        return decoder

    return MagicMock(side_effect=construct)


def _one_result_per_frame(boxes_data: list[tuple]):
    """predict() side effect: the same boxes for every frame of the batch."""
    def predict(source, **kwargs):
        return [_make_yolo_result(boxes_data) for _ in source]
    return predict


```

c) Фикстуру `sample_frame` заменить на

```python
@pytest.fixture
def sample_frame():
    return np.zeros(frame_shape(640, 480, "yuv420p"), dtype=np.uint8)
```

d) Всё от строки `# --- _extract_raw_detections ---` до строки `# --- annotate() pipeline ---` (не включая её) заменить на:

```python
# --- _draw_detections ---

class TestDrawDetections:
    def test_calls_visualizer(self, annotator, mock_visualizer, sample_frame, default_params):
        dets = [
            DetectionBox(x1=10, y1=20, x2=100, y2=200, class_id=0, class_name="person", confidence=0.9),
            DetectionBox(x1=50, y1=60, x2=150, y2=250, class_id=1, class_name="car", confidence=0.8),
        ]
        annotator._draw_detections(sample_frame, 640, 480, dets, default_params, font_scale=0.5)
        assert mock_visualizer.draw_detection_yuv420.call_count == 2
        for call in mock_visualizer.draw_detection_yuv420.call_args_list:
            assert call.kwargs["frame"] is sample_frame
            assert (call.kwargs["width"], call.kwargs["height"]) == (640, 480)
            assert call.kwargs["font_scale"] == 0.5


```

Тесты `TestExtractRawDetections` переехали в `tests/test_batch_inference.py::TestExtractDetections` (Task 4). Тест на `class_filter` не переносится: фильтр классов применяется после стабилизации, в проходе 1 он всегда был `None`.

e) Тело метода `_make_decoder_mock` в классе `TestAnnotatePipeline`

```python
        """Create a single decoder mock instance with its own frame sequence."""
        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = list(frames) + [None]
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)
        return mock_decoder
```

заменить на

```python
        """Create a single decoder mock instance with its own frame count."""
        return _make_decoder_mock(frames)
```

В трёх других классах (`TestAutoCodecResolve`, `TestAnnotateCancellation`, `TestNvencFallback`) тело того же метода

```python
        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = list(frames) + [None]
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)
        return mock_decoder
```

заменить на `        return _make_decoder_mock(frames)` (Edit с `replace_all`, ровно 3 вхождения).

f) Механические замены по всему файлу (Edit с `replace_all`; в скобках — ожидаемое число вхождений, если оно не совпало, остановиться и разобраться):

| Было | Стало |
|---|---|
| `MagicMock(side_effect=[decoder_pass1, decoder_pass2])` (3) | `_decoder_cls(decoder_pass1, decoder_pass2)` |
| `MagicMock(side_effect=[decoder1, decoder2])` (4) | `_decoder_cls(decoder1, decoder2)` |
| `MagicMock(side_effect=[decoder1])` (2) | `_decoder_cls(decoder1)` |
| `MagicMock(side_effect=[dec_p1, dec_p2_nvenc, dec_p2_cpu])` (4) | `_decoder_cls(dec_p1, dec_p2_nvenc, dec_p2_cpu)` |
| `MagicMock(side_effect=[dec_p1, dec_p2])` (1) | `_decoder_cls(dec_p1, dec_p2)` |
| `MagicMock(side_effect=[dec_p1, dec_p2_nvenc, dec_p2_cpu_unused])` (1) | `_decoder_cls(dec_p1, dec_p2_nvenc, dec_p2_cpu_unused)` |
| `        mock_decoder_cls = MagicMock(return_value=mock_decoder)` (1) | `        mock_decoder_cls = _decoder_cls(mock_decoder)` |
| `mock_decoder.read_frame.side_effect = RuntimeError(` (1) | `mock_decoder.read_into.side_effect = RuntimeError(` |
| `"""When FFmpegDecoder.read_frame raises RuntimeError, annotate propagates it."""` (1) | `"""When FFmpegDecoder.read_into raises RuntimeError, annotate propagates it."""` |
| `.read_frame.call_count` (3) | `.read_into.call_count` |
| `mock_visualizer.draw_detection.call_count` (2 — третье вхождение ушло вместе со старым `TestDrawDetections` на шаге d) | `mock_visualizer.draw_detection_yuv420.call_count` |

g) Перед строкой `class TestAutoCodecResolve:` вставить новый класс:

```python
class TestPass1Inference:
    """Pass 1 batches detection frames and feeds YOLO frames already at inference size."""

    def _run(self, mock_model, mock_visualizer, tmp_path, *, num_frames, width=640, height=480,
             params=None, fp16=False, batch_size=1, boxes=((10, 20, 100, 200, 0, 0.9),)):
        mock_model.predict.side_effect = _one_result_per_frame(list(boxes))
        frames = [None] * num_frames
        decoder_cls = _decoder_cls(_make_decoder_mock(frames), _make_decoder_mock(frames))
        encoder = MagicMock()
        encoder.__enter__ = MagicMock(return_value=encoder)
        encoder.__exit__ = MagicMock(return_value=False)
        encoder_cls = MagicMock(return_value=encoder)
        ffprobe = MagicMock()
        ffprobe.returncode = 0
        ffprobe.stdout = json.dumps({"streams": [{
            "r_frame_rate": "30/1", "width": width, "height": height, "nb_frames": str(num_frames),
        }]})
        input_path = tmp_path / "input.mp4"
        input_path.touch()
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, HWAccelConfig(accel_type=HWAccelType.CPU),
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
            fp16=fp16, batch_size=batch_size,
        )
        with (
            patch("video_annotator.FFmpegDecoder", decoder_cls),
            patch("video_annotator.FFmpegEncoder", encoder_cls),
            patch("video_annotator.subprocess.run", return_value=ffprobe),
        ):
            stats = annotator.annotate(input_path, tmp_path / "out.mp4", params or AnnotationParams(detect_every=1))
        return stats, decoder_cls, encoder_cls, encoder

    def test_detection_frames_go_to_predict_in_batches(self, mock_model, mock_visualizer, tmp_path):
        stats, *_ = self._run(mock_model, mock_visualizer, tmp_path, num_frames=7, batch_size=3)
        sizes = [len(c.kwargs["source"]) for c in mock_model.predict.call_args_list]
        assert sizes == [3, 3, 1]
        assert stats.detected_frames == 7

    def test_clip_shorter_than_one_batch(self, mock_model, mock_visualizer, tmp_path):
        stats, _, _, encoder = self._run(mock_model, mock_visualizer, tmp_path, num_frames=3, batch_size=8)
        sizes = [len(c.kwargs["source"]) for c in mock_model.predict.call_args_list]
        assert sizes == [3]
        assert stats.detected_frames == 3
        assert mock_visualizer.draw_detection_yuv420.call_count == 3
        assert encoder.write_frame.call_count == 3

    def test_detect_every_skips_frames_before_batching(self, mock_model, mock_visualizer, tmp_path):
        stats, *_ = self._run(
            mock_model, mock_visualizer, tmp_path, num_frames=10, batch_size=8,
            params=AnnotationParams(detect_every=5),
        )
        sizes = [len(c.kwargs["source"]) for c in mock_model.predict.call_args_list]
        assert sizes == [2]  # frames 0 and 5
        assert stats.detected_frames == 2
        assert stats.total_frames == 10

    def test_fp16_reaches_predict(self, mock_model, mock_visualizer, tmp_path):
        self._run(mock_model, mock_visualizer, tmp_path, num_frames=2, fp16=True)
        assert all(c.kwargs["quantize"] == 16 for c in mock_model.predict.call_args_list)

    def test_fp32_does_not_pass_quantize(self, mock_model, mock_visualizer, tmp_path):
        self._run(mock_model, mock_visualizer, tmp_path, num_frames=2)
        assert all("quantize" not in c.kwargs for c in mock_model.predict.call_args_list)

    def test_frames_arrive_at_inference_size_and_boxes_are_scaled_back(
        self, mock_model, mock_visualizer, tmp_path
    ):
        self._run(
            mock_model, mock_visualizer, tmp_path, num_frames=1, width=1280, height=960,
            params=AnnotationParams(detect_every=1, imgsz=640),
        )
        source = mock_model.predict.call_args.kwargs["source"]
        assert source[0].shape == (480, 640, 3)
        det = mock_visualizer.draw_detection_yuv420.call_args.kwargs["det"]
        assert (det.x1, det.y1, det.x2, det.y2) == (20, 40, 200, 400)

    def test_pass2_streams_yuv420p_both_ways(self, mock_model, mock_visualizer, tmp_path):
        _, decoder_cls, encoder_cls, _ = self._run(mock_model, mock_visualizer, tmp_path, num_frames=2)
        pass1, pass2 = decoder_cls.call_args_list
        assert pass1.kwargs.get("pix_fmt", "bgr24") == "bgr24"
        assert pass2.kwargs["pix_fmt"] == "yuv420p"
        assert encoder_cls.call_args.kwargs["pix_fmt"] == "yuv420p"

    def test_every_frame_reaches_the_encoder(self, mock_model, mock_visualizer, tmp_path):
        _, _, _, encoder = self._run(mock_model, mock_visualizer, tmp_path, num_frames=12, batch_size=4)
        assert encoder.write_frame.call_count == 12
        assert encoder.write_frame.call_args.args[0].shape == (640 * 480 * 3 // 2,)

    def test_gpu_cache_is_released_between_passes(self, mock_model, mock_visualizer, tmp_path):
        with patch("video_annotator.torch.cuda.empty_cache") as empty_cache:
            self._run(mock_model, mock_visualizer, tmp_path, num_frames=2)
        empty_cache.assert_called_once()


```

- [ ] **Step 2: Убедиться, что новые тесты падают**

Run: `.venv/bin/python -m pytest tests/test_video_annotator.py -k "TestPass1Inference or TestDrawDetections" -v`
Expected: FAIL: `TypeError: VideoAnnotator.__init__() got an unexpected keyword argument 'fp16'` и `TypeError` в `test_calls_visualizer`.

Весь файл до Step 5 не запускать: старый `_pass1_collect` вызывает `read_frame()` новых моков, а `MagicMock` никогда не возвращает `None`, и цикл не закончится.

- [ ] **Step 3: Переписать проход 1**

В `app/video_annotator.py`:

1. Импорты: заменить

```python
import numpy as np

from detection_stabilizer import (
    DetectionStabilizer,
    RawDetection,
    StabilizedFrame,
    StabilizerConfig,
)
from ffmpeg_pipe import FFmpegDecoder, FFmpegEncoder
from hw_accel import HWAccelConfig, HWAccelType
```

на

```python
import numpy as np
import torch

from batch_inference import BatchDetector, inference_size, model_stride, resize_for_inference
from detection_stabilizer import (
    DetectionStabilizer,
    RawDetection,
    StabilizedFrame,
    StabilizerConfig,
)
from ffmpeg_pipe import FFmpegDecoder, FFmpegEncoder
from frame_threads import ThreadedFrameReader, ThreadedFrameWriter
from hw_accel import HWAccelConfig, HWAccelType
```

2. Конструктор: заменить

```python
        crf: int = 18,
        stabilizer_config: StabilizerConfig | None = None,
    ):
        self.model = model
        self.visualizer = visualizer
        self.class_names = class_names
        self.hw_config = hw_config
        self.codec = codec
        self.crf = crf
        self.stabilizer_config = stabilizer_config or StabilizerConfig()
```

на

```python
        crf: int = 18,
        stabilizer_config: StabilizerConfig | None = None,
        fp16: bool = False,
        batch_size: int = 1,
    ):
        self.model = model
        self.visualizer = visualizer
        self.class_names = class_names
        self.hw_config = hw_config
        self.codec = codec
        self.crf = crf
        self.stabilizer_config = stabilizer_config or StabilizerConfig()
        self.fp16 = fp16
        self.batch_size = batch_size
```

3. Строку лога в `annotate()`

```python
            f"detect_every={params.detect_every}, conf={params.conf}"
        )
```

заменить на

```python
            f"detect_every={params.detect_every}, conf={params.conf}, "
            f"fp16={self.fp16}, batch={self.batch_size}"
        )
```

4. В `annotate()` после строки `        stats.total_frames = actual_frames` вставить:

```python
        pass1_seconds = time.perf_counter() - start_time
        # Pass 1 leaves its activations in PyTorch's cache; NVDEC and NVENC need
        # that memory in pass 2. A no-op when CUDA was never initialised.
        torch.cuda.empty_cache()
```

5. Метод `_pass1_collect` целиком (от `    def _pass1_collect(` до `        return raw_detections, frame_num`) заменить на:

```python
    def _pass1_collect(
        self,
        input_path: Path,
        metadata: VideoMetadata,
        params: AnnotationParams,
        yolo_conf: float,
        progress_callback: Callable[[int], None] | None,
        stats: AnnotationStats,
        cancel_event: threading.Event | None = None,
    ) -> tuple[dict[int, list[RawDetection]], int]:
        """Pass 1: decode on a reader thread, run YOLO in batches (no disk cache).

        The reader thread shrinks every detection frame to the inference size
        exactly as Ultralytics would, so the model input is unchanged while
        the main thread only waits for the GPU.
        """
        size = inference_size(metadata.width, metadata.height, params.imgsz, model_stride(self.model))
        detect_every = params.detect_every

        def transform(frame_num: int, frame: np.ndarray) -> np.ndarray | None:
            if frame_num % detect_every:
                return None
            return resize_for_inference(frame, size)

        detector = BatchDetector(
            self.model, self.class_names,
            conf=yolo_conf, imgsz=params.imgsz, max_det=params.max_det,
            fp16=self.fp16, batch_size=self.batch_size,
            scale=(metadata.width / size[0], metadata.height / size[1]),
        )
        frame_count = 0

        with FFmpegDecoder(input_path, metadata.width, metadata.height, self.hw_config) as decoder:
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelledError()
            with ThreadedFrameReader(decoder, transform, queue_size=2 * self.batch_size) as frames:
                for frame_num, small in frames:
                    if cancel_event is not None and cancel_event.is_set():
                        raise JobCancelledError()
                    if small is not None:
                        detector.add(frame_num, small)
                    frame_count = frame_num + 1

                    if progress_callback and metadata.total_frames > 0 and frame_count % 10 == 0:
                        progress = int((frame_count / metadata.total_frames) * 80)
                        progress_callback(min(progress, 80))

            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelledError()
            detector.flush()

        stats.detected_frames = detector.detected_frames
        return detector.detections, frame_count
```

Проверка отмены сразу после открытия декодера нужна до запуска потока-читателя: тест `test_cancel_before_pass1_first_iteration_raises` требует, чтобы `read_into` не вызывался ни разу.

6. Метод `_extract_raw_detections` удалить целиком.

- [ ] **Step 4: Переписать проход 2 и итоговый лог**

1. В `annotate()` после строки `        font_scale = self.visualizer.calculate_adaptive_font_scale(metadata.height)` вставить `        pass2_start = time.perf_counter()`.

2. В `annotate()` заменить

```python
        stats.processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        fps_actual = actual_frames / max(stats.processing_time_ms / 1000, 0.001)
        logger.info(
            f"Frame processing complete: {actual_frames} frames in {stats.processing_time_ms}ms "
            f"({fps_actual:.1f} fps), detected={stats.detected_frames}, "
            f"tracked={stats.tracked_frames}, total_detections={stats.total_detections}"
        )
```

на

```python
        pass2_seconds = time.perf_counter() - pass2_start
        stats.processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        fps_actual = actual_frames / max(stats.processing_time_ms / 1000, 0.001)
        logger.info(
            f"Frame processing complete: {actual_frames} frames in {stats.processing_time_ms}ms "
            f"({fps_actual:.1f} fps; pass1 {pass1_seconds:.1f}s "
            f"{actual_frames / max(pass1_seconds, 0.001):.1f} fps, pass2 {pass2_seconds:.1f}s "
            f"{actual_frames / max(pass2_seconds, 0.001):.1f} fps), detected={stats.detected_frames}, "
            f"tracked={stats.tracked_frames}, total_detections={stats.total_detections}"
        )
```

3. Методы `_pass2_render` и `_draw_detections` целиком (до конца файла) заменить на:

```python
    def _pass2_render(
        self,
        input_path: Path,
        output_path: Path,
        metadata: VideoMetadata,
        params: AnnotationParams,
        stabilized: dict[int, StabilizedFrame],
        effective_codec: str,
        effective_crf: int | None,
        effective_bitrate: int | None,
        font_scale: float,
        total_frames: int,
        progress_callback: Callable[[int], None] | None,
        cancel_event: threading.Event | None = None,
        hw_config: HWAccelConfig | None = None,
    ) -> None:
        """Pass 2: decode yuv420p again, draw on the planes, encode on a writer thread.

        Frames stay in yuv420p from decoder to encoder, so neither ffmpeg
        converts colours and only the pixels under the boxes change.

        ``hw_config`` defaults to ``self.hw_config``; it's overridable so
        ``annotate()`` can retry this pass on CPU when NVENC init fails.
        """
        frame_num = 0
        config = hw_config if hw_config is not None else self.hw_config
        width, height = metadata.width, metadata.height

        with FFmpegDecoder(input_path, width, height, config, pix_fmt="yuv420p") as decoder, \
             FFmpegEncoder(input_path, output_path, width, height,
                           metadata.fps, config, effective_codec,
                           crf=effective_crf, bitrate=effective_bitrate,
                           pix_fmt="yuv420p") as encoder, \
             ThreadedFrameWriter(encoder, decoder.frame_shape) as writer:

            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise JobCancelledError()
                frame = writer.acquire()
                if not decoder.read_into(frame):
                    writer.release(frame)
                    break

                if frame_num in stabilized:
                    self._draw_detections(frame, width, height,
                                          stabilized[frame_num].detections, params, font_scale)

                if not writer.submit(frame):
                    # Encoder finished early (e.g. -shortest finalised the
                    # output); further frames would be decoded and drawn in
                    # vain. Break out instead of feeding dead iterations.
                    break
                frame_num += 1

                if progress_callback and total_frames > 0 and frame_num % 10 == 0:
                    progress = 80 + int((frame_num / total_frames) * 19)
                    progress_callback(min(progress, 99))

    def _draw_detections(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        detections: list[DetectionBox],
        params: AnnotationParams,
        font_scale: float,
    ) -> None:
        for det in detections:
            self.visualizer.draw_detection_yuv420(
                frame=frame,
                width=width,
                height=height,
                det=det,
                line_width=params.line_width,
                show_labels=params.show_labels,
                show_conf=params.show_conf,
                font_scale=font_scale,
                text_thickness=1,
            )
```

Порядок контекстов важен: выход идёт в обратном порядке, писатель закрывается и дожидается своего потока раньше, чем энкодер закроет stdin.

- [ ] **Step 5: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_video_annotator.py -v`
Expected: PASS весь файл: старые тесты пайплайна, кодеков, отмены и NVENC-отката плюс `TestPass1Inference`.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS целиком. `tests/test_worker.py` подменяет `VideoAnnotator` целиком, а worker пока создаёт его без `fp16`/`batch_size` (значения по умолчанию False и 1).

Проверить, что в `app/video_annotator.py` не осталось `read_frame`, `write_frame(`, `tobytes` и `_extract_raw_detections`:
Run: `grep -n "read_frame\|write_frame(\|tobytes\|_extract_raw_detections" app/video_annotator.py`
Expected: пустой вывод.

- [ ] **Step 6: Документация**

В таблице «Architecture» в `CLAUDE.md` заменить строку `` | `app/ffmpeg_pipe.py` | FFmpeg pipe-based video decoder/encoder | `` на `` | `app/ffmpeg_pipe.py` | FFmpeg pipe-based video decoder/encoder (bgr24 or yuv420p frames) | ``.

В разделе «Key Patterns» в `CLAUDE.md` заменить абзац

```
**Video Annotation**: Async job API — YOLO every Nth frame + hold mode (reuse detections) for intermediate frames. Single worker, in-memory job state (requires `workers=1`).
```

на

```
**Video Annotation**: Async job API — YOLO every Nth frame + hold mode (reuse detections) for intermediate frames. Single worker, in-memory job state (requires `workers=1`). The worker gets its own model instance (`ModelManager.get_video_model`): Ultralytics flips a model between FP16 and FP32 in place, which would break a concurrent `/detect` on a shared one. Pass 1 decodes BGR on a `ThreadedFrameReader` thread that shrinks detection frames to the inference size exactly like Ultralytics' LetterBox (cv2 INTER_LINEAR; ffmpeg's scalers change detections), and the main thread runs YOLO in batches (`BatchDetector`: `VIDEO_BATCH_SIZE`, `VIDEO_FP16`, the batch halves on GPU out-of-memory). Pass 2 keeps frames in yuv420p from decoder to encoder, draws boxes on the Y/U/V planes (BT.601, the matrix the old BGR→yuv420p conversion used) and writes on a `ThreadedFrameWriter` thread. RTX 3090, yolo26x@1024, 2560×1920, 1601 frames: 77 s → 33 s, pass 2 at the NVENC ceiling.
```

В `.claude/rules/api.md` в пункте «Checkpoint latency» заменить `Bounded by one frame's work (one YOLO inference in pass 1, or one decode+encode frame in pass 2).` на `Bounded by one batch of YOLO inference in pass 1 (`VIDEO_BATCH_SIZE` frames, 8 on NVIDIA by default) or one decoded frame in pass 2.`

- [ ] **Step 7: Commit**

```bash
git add app/video_annotator.py tests/test_video_annotator.py CLAUDE.md .claude/rules/api.md
git commit -m "feat(video): threaded, batched pass 1 and yuv420p pass 2 in VideoAnnotator"
```

---

### Task 8: Интеграционный тест `annotate()` с настоящим ffmpeg

**Files:**
- Create: `tests/test_video_annotation_integration.py`
- Modify: `CLAUDE.md` (раздел Testing)

**Interfaces:**
- Consumes: `VideoAnnotator(..., batch_size=...)` из Task 7, `DetectionVisualizer` из Task 5.
- Produces: ничего для других задач.

Тест закрепляет поведение, реализованное в Task 7, и пункты 1–4 раздела Review Focus. Он должен пройти с первого раза. Падение здесь означает ошибку в задачах 2–7: чинить нужно там, а не ослаблять тест.

- [ ] **Step 1: Написать тест**

Создать `tests/test_video_annotation_integration.py`:

```python
"""VideoAnnotator.annotate() end to end with real ffmpeg on a lavfi clip; YOLO is a stub.

Skipped when ffmpeg/ffprobe are missing. Covers what the mocked tests cannot:
real pipes, the reader and writer threads, yuv420p frames drawn in place,
libx264 encoding and the audio merge.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from detection_stabilizer import StabilizerConfig
from hw_accel import HWAccelConfig, HWAccelType
from video_annotator import AnnotationParams, VideoAnnotator
from visualization import DetectionVisualizer

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)

WIDTH, HEIGHT, FPS, SECONDS = 320, 240, 10, 2
BOX = (40, 60, 200, 180)  # x1, y1, x2, y2 in full-frame pixels
BLUE = (255, 0, 0)  # BGR colour of class 0 in DetectionVisualizer's palette


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, box, class_id, conf):
        self.xyxy = _Tensor([box])
        self.cls = _Tensor([class_id])
        self.conf = _Tensor([conf])

    def __len__(self):
        return 1


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _StubModel:
    """Answers every frame with BOX, expressed in the coordinates of the frame it got."""

    names = {0: "person"}

    def __init__(self):
        self.frame_shapes = []

    def predict(self, source, **kwargs):
        results = []
        for frame in source:
            self.frame_shapes.append(frame.shape)
            sx, sy = frame.shape[1] / WIDTH, frame.shape[0] / HEIGHT
            box = (BOX[0] * sx, BOX[1] * sy, BOX[2] * sx, BOX[3] * sy)
            results.append(_Result(_Boxes(box, 0, 0.9)))
        return results


def _make_clip(path: Path, *, audio_seconds: float | None = SECONDS, pix_fmt: str = "yuv420p",
               vfr: bool = False) -> Path:
    """A gray lavfi clip. ``audio_seconds=None`` leaves the audio out; ``vfr``
    keeps every third frame with its original timestamp (0, 0.3, 0.6 s, ...)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r={FPS}:d={SECONDS}"]
    if audio_seconds is not None:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_seconds}"]
    if vfr:
        cmd += ["-vf", "select='not(mod(n\\,3))'", "-fps_mode", "passthrough"]
    cmd += ["-c:v", "libx264", "-pix_fmt", pix_fmt]
    if audio_seconds is not None:
        cmd += ["-c:a", "aac"]
    subprocess.run(cmd + [str(path)], check=True, timeout=120)
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("annotate")
    return {
        "audio": _make_clip(root / "audio.mp4"),
        "no_audio": _make_clip(root / "no_audio.mp4", audio_seconds=None),
        # IP cameras often record full-range yuvj420p; pass 2 asks ffmpeg for yuv420p
        "full_range": _make_clip(root / "full_range.mp4", pix_fmt="yuvj420p"),
        # -shortest ends the encoder before the last frames are written
        "short_audio": _make_clip(root / "short_audio.mp4", audio_seconds=1),
        "vfr": _make_clip(root / "vfr.mp4", audio_seconds=None, vfr=True),
    }


def _annotator(model: "_StubModel", batch_size: int) -> VideoAnnotator:
    return VideoAnnotator(
        model, DetectionVisualizer(model.names), model.names,
        HWAccelConfig(accel_type=HWAccelType.CPU), codec="h264", crf=18,
        stabilizer_config=StabilizerConfig(), batch_size=batch_size,
    )


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,nb_read_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def _frame(path: Path, index: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        check=True, capture_output=True, timeout=60,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH, 3)


def _assert_box_drawn(frame: np.ndarray) -> None:
    frame = frame.astype(int)
    left_edge = frame[100:140, BOX[0] - 1:BOX[0] + 2].reshape(-1, 3).mean(axis=0)
    assert np.abs(left_edge - BLUE).max() < 50, left_edge  # the box, in its class colour
    assert np.abs(frame[10:30, 260:300] - 128).max() < 12  # gray far from the box


@pytest.mark.parametrize("batch_size,imgsz", [(1, 320), (4, 160)])
def test_annotate_draws_boxes_and_keeps_every_frame_and_the_audio(clips, tmp_path, batch_size, imgsz):
    model = _StubModel()
    output = tmp_path / "annotated.mp4"
    stats = _annotator(model, batch_size).annotate(
        clips["audio"], output, AnnotationParams(conf=0.5, imgsz=imgsz, detect_every=1, line_width=6),
    )

    source, result = _probe(clips["audio"]), _probe(output)
    assert stats.total_frames == int(source["video"]["nb_read_frames"]) == FPS * SECONDS
    assert int(result["video"]["nb_read_frames"]) == stats.total_frames
    assert "audio" in result
    # imgsz 160 halves the 320x240 frames before YOLO sees them
    assert set(model.frame_shapes) == {(imgsz * HEIGHT // WIDTH, imgsz, 3)}
    _assert_box_drawn(_frame(output, 10))


def test_clip_without_audio(clips, tmp_path):
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["no_audio"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    result = _probe(output)
    assert "audio" not in result
    assert int(result["video"]["nb_read_frames"]) == stats.total_frames
    _assert_box_drawn(_frame(output, 5))


def test_full_range_source(clips, tmp_path):
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["full_range"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    assert int(_probe(output)["video"]["nb_read_frames"]) == stats.total_frames
    _assert_box_drawn(_frame(output, 5))


def test_audio_shorter_than_video_ends_the_output_early(clips, tmp_path):
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["short_audio"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    written = int(_probe(output)["video"]["nb_read_frames"])
    assert stats.total_frames == FPS * SECONDS
    assert FPS // 2 <= written < stats.total_frames  # cut near the 1 s of audio
    _assert_box_drawn(_frame(output, 2))


def test_variable_frame_rate_keeps_passes_aligned(clips, tmp_path):
    output = tmp_path / "annotated.mp4"
    stats = _annotator(_StubModel(), 4).annotate(
        clips["vfr"], output, AnnotationParams(conf=0.5, imgsz=320, line_width=6),
    )
    written = int(_probe(output)["video"]["nb_read_frames"])
    assert written == stats.total_frames
    _assert_box_drawn(_frame(output, written - 1))  # boxes reach the last frame
```

- [ ] **Step 2: Запустить тест**

Run: `.venv/bin/python -m pytest tests/test_video_annotation_integration.py -v`
Expected: PASS, 6 тестов примерно за 5–10 с. Если ffmpeg не установлен, тесты пропускаются (SKIPPED), тогда поставить ffmpeg и повторить.

- [ ] **Step 3: Раздел Testing в CLAUDE.md**

В абзаце «Tests cover …» раздела «Testing» в `CLAUDE.md` заменить хвост

```
`tests/test_video_extraction_integration.py`: both ffmpeg passes and both video endpoints on lavfi clips).
```

на

```
`tests/test_video_extraction_integration.py`: both ffmpeg passes and both video endpoints on lavfi clips), and the annotation pipeline pieces (`tests/test_frame_threads.py`, `tests/test_batch_inference.py` incl. a byte-exact check against Ultralytics' LetterBox, `tests/test_model_manager.py`; `tests/test_video_annotation_integration.py`: `annotate()` with real ffmpeg on lavfi clips).
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_video_annotation_integration.py CLAUDE.md
git commit -m "test(video): annotate() end to end with real ffmpeg on lavfi clips"
```

---

### Task 9: Worker берёт свою модель и режим инференса

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `Settings.video_fp16`, `Settings.video_batch_size` (Task 1); `resolve_inference_mode(fp16, batch_size, device) -> InferenceMode` (Task 4); `ModelManager.get_video_model(model_name)` (Task 6); `VideoAnnotator(..., fp16=..., batch_size=...)` (Task 7); `ModelEntry.device` — строка устройства, на которое загружена модель.
- Produces: ничего для других задач.

- [ ] **Step 1: Обновить тесты worker**

В `tests/test_worker.py`:

a) Переименовать все вхождения `get_model` в `get_video_model` (Edit с `replace_all`, 17 вхождений на 14 строках, включая имена помощников `get_model_side_effect`, `real_get_model`, `slow_get_model`). Worker теперь вызывает только `get_video_model`.

b) В фикстуре `mock_model_entry` после строки `    entry.visualizer = MagicMock()` добавить строку `    entry.device = "cpu"`.

c) В `test_continues_after_failure` после строки `        original_entry.visualizer = MagicMock()` добавить строку `        original_entry.device = "cpu"`.

d) Перед строкой `class TestAnnotationWorkerCancellation:` вставить (в конец класса `TestAnnotationWorker`):

```python
    @pytest.mark.asyncio
    async def test_passes_configured_inference_mode(
        self, worker_app, worker_job_manager, tmp_path
    ):
        """VIDEO_FP16 / VIDEO_BATCH_SIZE reach VideoAnnotator as fp16 / batch_size."""
        settings = Settings(
            yolo_models="{}", video_jobs_dir=str(tmp_path), max_executor_workers=1,
            video_fp16="false", video_batch_size="4",
        )
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        mock_annotator_cls = MagicMock()
        mock_annotator_cls.return_value.annotate.return_value = AnnotationStats(total_frames=1)
        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(worker_app, settings, worker_job_manager)

        kwargs = mock_annotator_cls.call_args.kwargs
        assert kwargs["fp16"] is False
        assert kwargs["batch_size"] == 4

    @pytest.mark.asyncio
    async def test_auto_inference_mode_on_cpu_is_fp32_batch_1(
        self, worker_app, worker_settings, worker_job_manager
    ):
        """With the defaults (auto) a model on CPU keeps today's behaviour."""
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        mock_annotator_cls = MagicMock()
        mock_annotator_cls.return_value.annotate.return_value = AnnotationStats(total_frames=1)
        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(worker_app, worker_settings, worker_job_manager)

        kwargs = mock_annotator_cls.call_args.kwargs
        assert kwargs["fp16"] is False
        assert kwargs["batch_size"] == 1
        worker_app.state.model_manager.get_video_model.assert_awaited()


```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_worker.py -k "inference_mode" -v`
Expected: FAIL. Worker вызывает `get_model`, а у мока менеджера есть только `get_video_model` (`AsyncMock`); задача зависает в `PROCESSING`, и `mock_annotator_cls.call_args` равен `None` (`AttributeError: 'NoneType' object has no attribute 'kwargs'`). Каждый такой тест ждёт до 5 с таймаута помощника.

- [ ] **Step 3: Подключить worker**

В `app/main.py`:

1. После строки `from video_annotator import VideoAnnotator, AnnotationParams, JobCancelledError` добавить строку `from batch_inference import resolve_inference_mode`.

2. В `_annotation_worker` заменить `                    model_entry = await model_manager.get_model(model_name)` на `                    model_entry = await model_manager.get_video_model(model_name)`.

3. В том же блоке настройки заменить

```python
                        max_staleness_sec=settings.stabilizer_max_staleness,
                    )
                    annotator = VideoAnnotator(
```

на

```python
                        max_staleness_sec=settings.stabilizer_max_staleness,
                    )
                    mode = resolve_inference_mode(
                        settings.video_fp16, settings.video_batch_size, model_entry.device
                    )
                    annotator = VideoAnnotator(
```

а

```python
                        crf=settings.video_crf,
                        stabilizer_config=stabilizer_config,
                    )
```

на

```python
                        crf=settings.video_crf,
                        stabilizer_config=stabilizer_config,
                        fp16=mode.fp16,
                        batch_size=mode.batch_size,
                    )
```

`resolve_inference_mode` стоит внутри `try` блока настройки. Ошибка в нём превращается в `Setup error`, а при отмене — в `CANCELLED`, как и остальная настройка.

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_worker.py -v`
Expected: PASS.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS целиком (около 600 тестов).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_worker.py
git commit -m "feat(video): annotation worker uses its own model and the configured inference mode"
```

---

### Task 10: Проверка на RTX 3090

Выполняет основная сессия вместе с владельцем. Код не меняется, результаты идут в описание PR.

**Files:** нет.

- [ ] **Step 1: Собрать NVIDIA-образ из ветки**

Run: `docker build -f docker/nvidia/Dockerfile -t vision-api-server:speedup .`
Expected: образ собран. Первый раз скачиваются колёса torch cu130, это долго.

- [ ] **Step 2: Запустить отдельный контейнер рядом с рабочим**

```bash
docker run -d --name vas-speedup --gpus all -p 3002:8000 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e YOLO_MODELS='{"yolo26s.pt":"cuda:0","yolo26x.pt":"cuda:0"}' -e YOLO_DEVICE=cuda:0 \
  -e MODELS_DIR=/models -e LOG_LEVEL=INFO -e VIDEO_HW_ACCEL=nvidia \
  -v deploy_models:/models vision-api-server:speedup
until curl -sf localhost:3002/health >/dev/null; do sleep 5; done; curl -s localhost:3002/health
```

Рабочий контейнер `deploy-vision-api-1` на порту 3001 не трогать.

- [ ] **Step 3: Прогнать ролик с параметрами frigate-analyzer**

Ролик: копия результата задачи `09a599cd4b74` (2560×1920 HEVC, 1601 кадр). Если её уже нет в scratchpad сессии брейншторма, попросить у владельца свежую двухминутную запись той же камеры. Записи в репозиторий не класть.

```bash
VIDEO=/path/to/video.mp4
JOB=$(curl -s -X POST "localhost:3002/detect/video/visualize?conf=0.6&imgsz=1024&max_det=100&detect_every=1&line_width=2&show_labels=true&show_conf=true&model=yolo26x.pt&classes=person,car,motorcycle,truck,bicycle,cat,dog,bird,backpack,horse,sheep,cow,bear,elephant,zebra,giraffe" \
  -F "file=@$VIDEO" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
while :; do S=$(curl -s localhost:3002/jobs/$JOB); echo "$S" | grep -q '"status":"completed"\|"status":"failed"' && break; sleep 3; done; echo "$S"
docker logs vas-speedup 2>&1 | grep -E "Starting annotation|Frame processing complete"
```

Expected:
- `"status":"completed"`;
- в логе `fp16=True, batch=8`;
- `processing_time_ms` около 30–35 тыс.; эталон на продакшене — 76.8 с (исходный файл той же задачи);
- `pass2` около 10 с.

- [ ] **Step 4: Посмотреть результат и отзывчивость**

Во время повторного прогона шага 3 в соседнем терминале:
- `for i in $(seq 10); do /usr/bin/time -f %e curl -s -o /dev/null localhost:3002/health; sleep 2; done` — каждый ответ не дольше доли секунды;
- `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1` — во время прохода 1 загрузка около 90–100 %.

После прогона скачать результат и посмотреть кадр с рамкой (глазами, через Read):

```bash
curl -s -o /tmp/annotated.mp4 localhost:3002/jobs/$JOB/download
ffmpeg -hide_banner -loglevel error -y -i /tmp/annotated.mp4 -vf "select=eq(n\,800),scale=960:-1" -frames:v 1 /tmp/annotated_800.png
```

- [ ] **Step 5: Убрать контейнер и записать результаты**

Run: `docker rm -f vas-speedup`

Записать в черновик описания PR: время до и после, fps проходов, загрузку GPU и наблюдение по кадру.

---

### Task 11: Проверка на AMD и CPU

Выполняет основная сессия. Адреса хостов и доступ даёт владелец; без них задачу не начинать, а спросить.

**Files:** нет.

- [ ] **Step 1: Получить от владельца хосты AMD и CPU**

Спросить адреса, способ доступа и какой ролик использовать (тот же, что в Task 10, или запись с этого хоста).

- [ ] **Step 2: AMD, режим `auto`**

На AMD-хосте в клоне ветки собрать и запустить образ на свободном порту, с теми же переменными, что у рабочего контейнера хоста (`HSA_OVERRIDE_GFX_VERSION` и т. д.):
`docker build -f docker/amd/Dockerfile -t vision-api-server:speedup-amd .`

Прогнать ролик, как в Task 10, Step 3. Проверить:
- в логе `fp16=False, batch=1`: форма тензоров и тип прежние, новых компиляций MIOpen быть не должно;
- время против текущего образа хоста на том же ролике;
- `curl -s localhost:<port>/health`: `fd_deleted` не растёт от задачи к задаче (сигнатура утечки MIOpen).

- [ ] **Step 3: AMD, эксперимент FP16 и батч 8**

Перезапустить контейнер с `-e VIDEO_FP16=true -e VIDEO_BATCH_SIZE=8`, прогнать тот же ролик дважды. Записать время обоих прогонов (первый включает компиляцию MIOpen) и динамику `open_fds`/`fd_deleted`. По итогам предложить владельцу, менять ли `auto` для AMD. Это отдельная задача, в этой ветке `auto` не трогать.

- [ ] **Step 4: CPU**

На CPU-хосте собрать `docker build -f docker/cpu/Dockerfile -t vision-api-server:speedup-cpu .`, прогнать ролик. Убедиться, что задача завершается, в логе `fp16=False, batch=1`, а рамки на месте. Сравнить время с текущим CPU-образом.

- [ ] **Step 5: Убрать временные контейнеры и записать результаты в черновик PR**

---

### Task 12: Подготовка ветки к PR

Выполняет основная сессия.

**Files:**
- Delete: `docs/superpowers/specs/2026-09-24-video-annotation-speedup-design.md`, `docs/superpowers/plans/2026-09-24-video-annotation-speedup.md`

- [ ] **Step 1: Прогнать весь набор тестов**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS целиком.

- [ ] **Step 2: Убрать плановые документы из ветки**

```bash
git rm docs/superpowers/specs/2026-09-24-video-annotation-speedup-design.md \
  docs/superpowers/plans/2026-09-24-video-annotation-speedup.md
git commit -m "chore: remove design and plan documents before PR"
```

Документы остаются в истории ветки.

- [ ] **Step 3: Завершить ветку**

Использовать superpowers:finishing-a-development-branch. PR создаётся только после явного согласия владельца; в описание PR войдут результаты задач 10 и 11.
