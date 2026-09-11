# Motion-Based Frame Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить отбор кадров в `/extract/frames` и `/detect/video` на алгоритм по движению: кадр 0, сетка раз в `max_gap`, пики метрики `blob`, кап `max_frames`.

**Architecture:** Чистый модуль `app/frame_selection.py` считает метрику по паре серых кадров и применяет правило отбора к массивам pts и метрики. `app/video_utils.py` переписывается: ffprobe JSON, первый проход ffmpeg потоком через pipe (серые кадры 640 px, pts из `showinfo`), второй проход через `select` только для выбранных кадров. Эндпоинты в `app/main.py` получают новые параметры, старый select-путь и `scene_threshold` удаляются.

**Tech Stack:** Python 3.14 (`.venv`), FastAPI, numpy, OpenCV (`cv2`), ffmpeg/ffprobe через `subprocess`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-motion-frame-selection-design.md`

## Global Constraints

- Тесты только в `.venv`: `.venv/bin/python -m pytest tests/ -v` из корня репозитория. В каждом шаге «Run» ниже подразумевается этот интерпретатор.
- Модули в `app/` импортируют друг друга по голым именам (`from frame_selection import ...`), потому что `app/` лежит в `sys.path` (см. `tests/conftest.py`). В `app/` нет `__init__.py`, так и оставить.
- Комментарии и докстринги в коде на английском, как во всём `app/`. Прозу плана и коммиты писать по образцу репозитория: `feat(video): ...`, `test(video): ...`, `docs(api): ...`.
- Каждый коммит заканчивается строкой `Claude-Session: https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4`.
- Контракт ошибок: ValueError из экстрактора → 422 только для нечитаемого файла или файла без видеопотока; всё остальное RuntimeError → 500. Клиент по 422 хоронит запись навсегда.
- Список кадров на успешном ответе никогда не пуст: кадр 0 берётся всегда.
- Никаких бинарных фикстур в git: тестовые клипы генерируются `ffmpeg -f lavfi` во временный каталог; интеграционные тесты пропускаются, если `ffmpeg` или `ffprobe` нет в PATH.
- ffmpeg ≥ 5.1 (`-fps_mode`), локально 8.0.1. `-fps_mode passthrough` обязателен в обоих проходах.
- Константы из спеки: `SCAN_WIDTH = 640`, `DIFF_THRESHOLD = 20`, `STORM_MEDIAN_BLOB = 0.02`, `PTS_TOLERANCE = 1e-3`; параметры по умолчанию `max_gap=4.0`, `motion_threshold=0.001`, `min_interval=1.0`, `max_frames=6`; дедлайн извлечения 300 с, таймаут ffprobe 30 с.
- Версия приложения после изменений: `3.0.0` в `FastAPI(version=...)` и в ответе `GET /`.
- Перед созданием PR все файлы из `docs/superpowers/` удаляются из ветки через `git rm` и коммитятся (Task 9). Плановые документы не должны попасть в диф PR.
- Субагентам не назначать модель haiku.

---

## Карта файлов

| Файл | Ответственность | Задачи |
|---|---|---|
| `app/frame_selection.py` (новый) | константы, `SelectionParams`, `SelectedFrame`, `select_frames`, `prepare_frame`, `blob_area` | 1, 2 |
| `app/video_utils.py` (переписывается) | `VideoInfo`, ffprobe JSON, разбор `showinfo`, `VideoFrameExtractor` с проходами 1 и 2, async-обёртка | 3, 4, 5 |
| `app/models.py` | поле `reason`, описания `frame_number`/`timestamp`, удаление `VideoDetectionSettings` | 6 |
| `app/main.py` | query-параметры, оба эндпоинта, версия | 6 |
| `tests/test_frame_selection.py` (новый) | правило и метрика на массивах | 1, 2 |
| `tests/test_video_utils.py` | ffprobe JSON, `showinfo`, ошибки экстрактора на моках | 3, 5 |
| `tests/test_video_extraction_integration.py` (новый) | клипы `lavfi`: проходы 1 и 2, эндпоинты через `TestClient` | 4, 5, 6 |
| `.claude/rules/api.md`, `CLAUDE.md` | документация | 7 |
| `~/vision-api-research/frame-selection/run_impl.py` (вне репозитория) | проверка реализации на корпусе | 8 |

---

### Task 1: Правило отбора `select_frames`

**Files:**
- Create: `app/frame_selection.py`
- Test: `tests/test_frame_selection.py`

**Interfaces:**
- Consumes: ничего из проекта.
- Produces: `SelectionParams(max_gap: float = 4.0, motion_threshold: float = 0.001, min_interval: float = 1.0, max_frames: int = 6)`, `SelectedFrame(index: int, reason: Literal["first", "grid", "motion"])`, `select_frames(pts: Sequence[float], blob: Sequence[float], params: SelectionParams) -> list[SelectedFrame]`, константы `SCAN_WIDTH`, `DIFF_THRESHOLD`, `STORM_MEDIAN_BLOB`, `PTS_TOLERANCE`. Задачи 4 и 5 используют `select_frames` и `STORM_MEDIAN_BLOB`, задача 6 — `SelectionParams`.

- [ ] **Step 1: Написать падающие тесты правила**

Создать `tests/test_frame_selection.py`:

```python
import pytest

from frame_selection import PTS_TOLERANCE, SelectedFrame, SelectionParams, select_frames


def _static(duration: float, fps: float = 10.0):
    """pts and zero blob for a static clip of `duration` seconds."""
    n = int(round(duration * fps))
    pts = [i / fps for i in range(n)]
    return pts, [0.0] * n


def _indices(selected):
    return [f.index for f in selected]


def _reasons(selected):
    return [f.reason for f in selected]


class TestFirstAndGrid:
    def test_single_frame(self):
        assert select_frames([0.0], [0.0], SelectionParams()) == [SelectedFrame(0, "first")]

    def test_static_clip_gives_first_and_grid(self):
        pts, blob = _static(10.0)
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0))
        assert _indices(selected) == [0, 40, 80]
        assert _reasons(selected) == ["first", "grid", "grid"]

    def test_grid_step_is_max_of_gap_and_interval(self):
        pts, blob = _static(10.0)
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, min_interval=5.0))
        assert _indices(selected) == [0, 50]

    def test_pts_jitter_within_tolerance_keeps_grid(self):
        pts, blob = _static(10.0)
        pts[40] = 4.0 - PTS_TOLERANCE / 2  # camera jitter: 3.9995 s
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0))
        assert _indices(selected) == [0, 40, 80]

    def test_empty_input_gives_empty_result(self):
        assert select_frames([], [], SelectionParams()) == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            select_frames([0.0, 0.1], [0.0], SelectionParams())


class TestThinning:
    def test_long_clip_thins_grid_keeping_first_and_last(self):
        pts, blob = _static(40.0)
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, max_frames=6))
        assert _indices(selected) == [0, 80, 160, 200, 280, 360]
        assert _reasons(selected) == ["first", "grid", "grid", "grid", "grid", "grid"]

    def test_max_frames_one_keeps_only_first(self):
        pts, blob = _static(40.0)
        assert select_frames(pts, blob, SelectionParams(max_frames=1)) == [SelectedFrame(0, "first")]

    def test_thinning_leaves_no_budget_for_peaks(self):
        pts, blob = _static(40.0)
        blob[100] = 0.05
        selected = select_frames(pts, blob, SelectionParams(max_gap=4.0, max_frames=6))
        assert "motion" not in _reasons(selected)


class TestStorm:
    def test_storm_returns_grid_only(self):
        pts, _ = _static(10.0)
        blob = [0.0] + [0.05] * (len(pts) - 1)  # rain: every frame changes
        blob[25] = 0.5
        selected = select_frames(pts, blob, SelectionParams())
        assert _indices(selected) == [0, 40, 80]

    def test_below_storm_median_peaks_are_used(self):
        pts, blob = _static(10.0)
        blob[25] = 0.5
        selected = select_frames(pts, blob, SelectionParams())
        assert SelectedFrame(25, "motion") in selected


class TestPeaks:
    def test_larger_peak_wins_the_last_slot(self):
        pts, blob = _static(10.0)
        blob[15] = 0.01
        blob[25] = 0.02
        selected = select_frames(pts, blob, SelectionParams(max_frames=4))
        assert _indices(selected) == [0, 25, 40, 80]
        assert selected[1].reason == "motion"

    def test_peak_at_threshold_is_ignored(self):
        pts, blob = _static(10.0)
        blob[25] = 0.001  # equal to the threshold: not strictly above
        selected = select_frames(pts, blob, SelectionParams(motion_threshold=0.001))
        assert _indices(selected) == [0, 40, 80]

    def test_peak_too_close_to_grid_frame_rejected(self):
        pts, blob = _static(10.0)
        blob[41] = 0.02  # 0.1 s after grid frame 40
        blob[25] = 0.01
        selected = select_frames(pts, blob, SelectionParams(min_interval=1.0))
        assert _indices(selected) == [0, 25, 40, 80]

    def test_peaks_respect_min_interval_between_themselves(self):
        pts, blob = _static(10.0)
        blob[20] = 0.03
        blob[25] = 0.02  # 0.5 s after frame 20
        blob[30] = 0.01  # exactly 1.0 s after frame 20
        selected = select_frames(pts, blob, SelectionParams(min_interval=1.0))
        assert _indices(selected) == [0, 20, 30, 40, 80]

    def test_budget_limits_peaks(self):
        pts, blob = _static(10.0)
        blob[10] = 0.01
        blob[20] = 0.02
        blob[30] = 0.03
        selected = select_frames(pts, blob, SelectionParams(max_frames=4))
        assert _indices(selected) == [0, 30, 40, 80]

    def test_equal_metric_prefers_lower_index(self):
        pts, blob = _static(10.0)
        blob[30] = 0.01
        blob[60] = 0.01
        selected = select_frames(pts, blob, SelectionParams(max_frames=4))
        assert _indices(selected) == [0, 30, 40, 80]

    def test_result_sorted_by_index(self):
        pts, blob = _static(10.0)
        blob[60] = 0.02
        blob[20] = 0.01
        selected = select_frames(pts, blob, SelectionParams())
        assert _indices(selected) == sorted(_indices(selected))
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_frame_selection.py -v`
Expected: ошибка импорта `ModuleNotFoundError: No module named 'frame_selection'`.

- [ ] **Step 3: Реализовать модуль с правилом**

Создать `app/frame_selection.py`:

```python
"""Motion-based frame selection: the ``blob`` metric and the selection rule.

Pure logic that depends on numpy and OpenCV only and never calls ffmpeg.
``video_utils`` decodes the video and feeds per-frame timestamps and metric
values into ``select_frames``.
"""
from dataclasses import dataclass
from statistics import median
from typing import Literal, Sequence

SCAN_WIDTH = 640          # width of the gray frames the metric is computed on
DIFF_THRESHOLD = 20       # brightness levels; |cur - prev| above this counts as changed
STORM_MEDIAN_BLOB = 0.02  # median blob over a segment above this = storm (rain, snow in IR)
PTS_TOLERANCE = 1e-3      # seconds; camera pts jitter tolerance in time comparisons

Reason = Literal["first", "grid", "motion"]


@dataclass(frozen=True)
class SelectionParams:
    """Frame selection parameters, mirrored by the query parameters of the video endpoints."""
    max_gap: float = 4.0            # grid step, seconds
    motion_threshold: float = 0.001  # blob above this is a motion candidate (fraction of the frame)
    min_interval: float = 1.0       # minimum distance between any two selected frames, seconds
    max_frames: int = 6             # cap on the number of selected frames


@dataclass(frozen=True)
class SelectedFrame:
    index: int      # frame index in the source video (0-based, ffmpeg's ``n``)
    reason: Reason


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def select_frames(
    pts: Sequence[float],
    blob: Sequence[float],
    params: SelectionParams,
) -> list[SelectedFrame]:
    """Select frames from per-frame timestamps and motion metric values.

    ``pts[i]`` is the presentation time of frame ``i``; ``blob[i]`` is the
    metric between frames ``i - 1`` and ``i`` (``blob[0]`` is 0). The rule:
    frame 0, then a time grid every ``max(max_gap, min_interval)`` seconds,
    thinned uniformly when it exceeds ``max_frames``; unless the segment is a
    storm (median blob above ``STORM_MEDIAN_BLOB``), the remaining budget is
    filled with the strongest motion peaks at least ``min_interval`` away from
    every selected frame. Deterministic; the result is sorted by index.
    """
    n = len(pts)
    if len(blob) != n:
        raise ValueError(f"pts and blob must have the same length: {n} != {len(blob)}")
    if n == 0:
        return []

    # 1-2: first frame + grid
    step = max(params.max_gap, params.min_interval)
    base = [SelectedFrame(0, "first")]
    last = pts[0]
    for i in range(1, n):
        if pts[i] - last >= step - PTS_TOLERANCE:
            base.append(SelectedFrame(i, "grid"))
            last = pts[i]

    # 3: uniform thinning keeps the first and the last grid frame; no budget is left for peaks
    cap = params.max_frames
    if len(base) > cap:
        if cap == 1:
            return [base[0]]
        positions = [_round_half_up(k * (len(base) - 1) / (cap - 1)) for k in range(cap)]
        return [base[p] for p in positions]

    # 4: storm — nearly every frame changes, the metric is blind, keep the grid only
    if n >= 2 and median(blob[1:]) > STORM_MEDIAN_BLOB:
        return base

    # 5: motion peaks, strongest first, ties by index, NMS by min_interval
    selected = list(base)
    taken = {f.index for f in selected}
    candidates = sorted(
        (i for i in range(n) if blob[i] > params.motion_threshold and i not in taken),
        key=lambda i: (-blob[i], i),
    )
    for i in candidates:
        if len(selected) >= cap:
            break
        if all(abs(pts[i] - pts[f.index]) >= params.min_interval - PTS_TOLERANCE for f in selected):
            selected.append(SelectedFrame(i, "motion"))

    return sorted(selected, key=lambda f: f.index)
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_frame_selection.py -v`
Expected: все 18 тестов PASS.

- [ ] **Step 5: Коммит**

```bash
git add app/frame_selection.py tests/test_frame_selection.py
git commit -m "feat(video): add motion frame selection rule

Claude-Session: https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4"
```

---

### Task 2: Метрика `blob_area`

**Files:**
- Modify: `app/frame_selection.py`
- Test: `tests/test_frame_selection.py`

**Interfaces:**
- Consumes: константу `DIFF_THRESHOLD` из Task 1.
- Produces: `prepare_frame(gray: np.ndarray) -> np.ndarray`, `blob_area(prev: np.ndarray, cur: np.ndarray) -> float` (оба аргумента — результат `prepare_frame`). Задача 4 вызывает их на каждом кадре первого прохода.

- [ ] **Step 1: Написать падающие тесты метрики**

В `tests/test_frame_selection.py` добавить `import numpy as np` перед `import pytest` и расширить импорт из модуля до
`from frame_selection import PTS_TOLERANCE, SelectedFrame, SelectionParams, blob_area, prepare_frame, select_frames`.
Затем добавить в конец файла:

```python
class TestBlobArea:
    def test_identical_frames_give_zero(self):
        frame = prepare_frame(np.full((360, 640), 128, np.uint8))
        assert blob_area(frame, frame) == 0.0

    def test_square_gives_its_dilated_area(self):
        prev = prepare_frame(np.zeros((360, 640), np.uint8))
        cur_raw = np.zeros((360, 640), np.uint8)
        cur_raw[100:140, 100:140] = 255  # 40×40 square
        value = blob_area(prev, prepare_frame(cur_raw))
        # 40×40 = 0.0069 of the frame before blur and dilation, at most 50×50 = 0.0109 after
        assert 0.0069 <= value <= 0.0109

    def test_noise_below_threshold_gives_zero(self):
        rng = np.random.default_rng(0)
        prev = prepare_frame(np.full((360, 640), 128, np.uint8))
        cur = prepare_frame(rng.integers(120, 137, size=(360, 640), dtype=np.uint8))
        assert blob_area(prev, cur) == 0.0

    def test_largest_component_only(self):
        prev = prepare_frame(np.zeros((360, 640), np.uint8))
        cur_raw = np.zeros((360, 640), np.uint8)
        cur_raw[10:20, 10:20] = 255      # small blob, 10×10
        cur_raw[200:260, 300:360] = 255  # big blob, 60×60
        value = blob_area(prev, prepare_frame(cur_raw))
        assert 60 * 60 / (640 * 360) <= value <= 70 * 70 / (640 * 360)

    def test_returns_python_float(self):
        prev = prepare_frame(np.zeros((360, 640), np.uint8))
        cur_raw = np.zeros((360, 640), np.uint8)
        cur_raw[50:90, 50:90] = 255
        assert type(blob_area(prev, prepare_frame(cur_raw))) is float
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_frame_selection.py -v -k BlobArea`
Expected: `ImportError: cannot import name 'blob_area'`.

- [ ] **Step 3: Реализовать метрику**

В `app/frame_selection.py` добавить импорты после `from typing import ...`:

```python
import cv2
import numpy as np
```

и после константы `PTS_TOLERANCE` — ядра:

```python
_KERNEL_OPEN = np.ones((3, 3), np.uint8)
_KERNEL_DILATE = np.ones((7, 7), np.uint8)
```

После класса `SelectedFrame` добавить две функции:

```python
def prepare_frame(gray: np.ndarray) -> np.ndarray:
    """Blur a gray frame 3×3 once. The caller keeps the result: it is the input for two pairs."""
    return cv2.blur(gray, (3, 3))


def blob_area(prev: np.ndarray, cur: np.ndarray) -> float:
    """Area of the largest changed region between two prepared frames, as a fraction of the frame.

    Reproduces ``mask_stats`` from the research scripts, on which the default
    thresholds were tuned: absdiff, 3×3 blur, mask above ``DIFF_THRESHOLD``,
    3×3 opening, 7×7 dilation, largest 8-connected component.
    """
    diff = cv2.blur(cv2.absdiff(prev, cur), (3, 3))
    mask = (diff > DIFF_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_OPEN)
    mask = cv2.dilate(mask, _KERNEL_DILATE)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / float(mask.size)
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_frame_selection.py -v`
Expected: все 23 теста PASS.

- [ ] **Step 5: Коммит**

```bash
git add app/frame_selection.py tests/test_frame_selection.py
git commit -m "feat(video): add blob motion metric

Claude-Session: https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4"
```

---

### Task 3: ffprobe JSON и разбор `showinfo` в `video_utils.py`

**Files:**
- Modify: `app/video_utils.py:1-197` (импорты, `VideoInfo`, `ExtractedFrame`, `get_video_info`, `_get_duration`, `_get_dimensions`)
- Test: `tests/test_video_utils.py`

**Interfaces:**
- Consumes: ничего нового.
- Produces: `VideoInfo(duration, width, height, fps, codec, rotation=0)` (ширина и высота после поворота), `ShowinfoFrame(pts: float, width: int, height: int)`, `parse_showinfo_line(line: str) -> ShowinfoFrame | None`, `parse_showinfo(stderr: str) -> list[ShowinfoFrame]`, `parse_probe_output(data: dict) -> VideoInfo`, константы `NO_VIDEO_STREAM_MESSAGE`, `UNREADABLE_VIDEO_MESSAGE`, `VideoFrameExtractor.PROBE_TIMEOUT = 30.0`, `VideoFrameExtractor.get_video_info(video_path) -> VideoInfo`. Задачи 4 и 5 используют всё перечисленное.

Старый путь `extract_frames` в этой задаче остаётся нетронутым: он всё ещё вызывает `get_video_info`, и новая `VideoInfo` с ним совместима.

- [ ] **Step 1: Переписать тесты `get_video_info` и добавить тесты парсеров**

Заменить содержимое `tests/test_video_utils.py` целиком:

```python
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from video_utils import (
    ShowinfoFrame,
    VideoFrameExtractor,
    VideoInfo,
    parse_probe_output,
    parse_showinfo,
    parse_showinfo_line,
)


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
[out#0/rawvideo @ 0x5582] video:54000KiB audio:0KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: 0.000000%
"""


class TestParseShowinfo:
    def test_parses_frames_and_ignores_other_lines(self):
        assert parse_showinfo(SHOWINFO_SAMPLE) == [
            ShowinfoFrame(pts=0.0, width=640, height=360),
            ShowinfoFrame(pts=0.08, width=640, height=360),
            ShowinfoFrame(pts=-0.04, width=640, height=360),
        ]

    def test_line_without_frame_data_is_none(self):
        assert parse_showinfo_line("  Stream #0:0 -> #0:0 (hevc (native) -> rawvideo (native))") is None
        assert parse_showinfo_line("[Parsed_showinfo_1 @ 0x1] config in time_base: 1/12800, frame_rate: 25/2") is None

    def test_pts_field_of_the_line_is_not_taken_for_size(self):
        line = "[Parsed_showinfo_0 @ 0x1] n:   5 pts:   5120 pts_time:0.4 duration: 1024 duration_time:0.08 fmt:bgr24 s:2880x1620 i:P iskey:0 type:P"
        assert parse_showinfo_line(line) == ShowinfoFrame(pts=0.4, width=2880, height=1620)
```

- [ ] **Step 2: Убедиться, что новые тесты падают**

Run: `.venv/bin/python -m pytest tests/test_video_utils.py -v`
Expected: `ImportError: cannot import name 'ShowinfoFrame'`.

- [ ] **Step 3: Заменить импорты, `VideoInfo` и разбор ffprobe**

В `app/video_utils.py` заменить строки 1–32 (импорты, `VideoInfo`, `ExtractedFrame`) на:

```python
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

NO_VIDEO_STREAM_MESSAGE = (
    "File contains no video stream. Only audio or metadata streams were found."
)
UNREADABLE_VIDEO_MESSAGE = (
    "File could not be read as a valid video. "
    "The file may be corrupted or not a supported video format."
)

_PTS_RE = re.compile(r"pts_time:\s*(-?[0-9.]+)")
_SIZE_RE = re.compile(r"\bs:(\d+)x(\d+)")


@dataclass
class VideoInfo:
    """Video metadata from ffprobe. ``width`` and ``height`` are the display
    dimensions after the rotation metadata is applied, which is what ffmpeg
    outputs because it autorotates by default."""
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    rotation: int = 0


@dataclass
class ExtractedFrame:
    """Extracted frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int


@dataclass(frozen=True)
class ShowinfoFrame:
    """One frame line of ffmpeg's ``showinfo`` filter."""
    pts: float
    width: int
    height: int


def parse_showinfo_line(line: str) -> ShowinfoFrame | None:
    """Parse one ffmpeg stderr line; None unless it is a showinfo frame line."""
    if "showinfo" not in line:
        return None
    pts = _PTS_RE.search(line)
    size = _SIZE_RE.search(line)
    if pts is None or size is None:
        return None
    return ShowinfoFrame(pts=float(pts.group(1)), width=int(size.group(1)), height=int(size.group(2)))


def parse_showinfo(stderr: str) -> list[ShowinfoFrame]:
    """All showinfo frame lines of an ffmpeg stderr dump, in order."""
    frames = []
    for line in stderr.splitlines():
        parsed = parse_showinfo_line(line)
        if parsed is not None:
            frames.append(parsed)
    return frames


def _to_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_fps(value) -> float:
    """ffprobe frame rates are fractions like ``12500/1000``; ``0/0`` means unknown."""
    if value is None:
        return 0.0
    text = str(value)
    if "/" not in text:
        return _to_float(text)
    num, _, den = text.partition("/")
    den_f = _to_float(den)
    return _to_float(num) / den_f if den_f > 0 else 0.0


def _parse_rotation(stream: dict) -> int:
    """Rotation in degrees from the Display Matrix side data or the legacy ``rotate`` tag.
    Only the magnitude matters: the value decides whether width and height swap."""
    for side_data in stream.get("side_data_list") or []:
        if isinstance(side_data, dict) and "rotation" in side_data:
            return abs(int(round(_to_float(side_data["rotation"])))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        return abs(int(round(_to_float(tag)))) % 360
    return 0


def parse_probe_output(data: dict) -> VideoInfo:
    """Build VideoInfo from ``ffprobe -print_format json -show_streams -show_format`` output."""
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(NO_VIDEO_STREAM_MESSAGE)
    stream = streams[0]
    width = _to_int(stream.get("width"))
    height = _to_int(stream.get("height"))
    if width <= 0 or height <= 0:
        raise ValueError(NO_VIDEO_STREAM_MESSAGE)
    rotation = _parse_rotation(stream)
    if rotation in (90, 270):
        width, height = height, width
    fmt = data.get("format") or {}
    duration = _to_float(fmt.get("duration")) or _to_float(stream.get("duration")) or 0.0
    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate")) or 0.0
    codec = stream.get("codec_name") or "unknown"
    return VideoInfo(duration=duration, width=width, height=height, fps=fps, codec=codec, rotation=rotation)
```

Затем заменить методы `get_video_info`, `_get_duration` и `_get_dimensions` класса `VideoFrameExtractor` (строки 81–197 исходного файла) на:

```python
    PROBE_TIMEOUT = 30.0

    def get_video_info(self, video_path: str) -> VideoInfo:
        """Video metadata via ffprobe JSON.

        Raises ValueError when the file is unreadable or has no video stream
        (the HTTP layer maps it to 422) and RuntimeError when ffprobe hangs.
        """
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            "-select_streams", "v:0",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.PROBE_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"ffprobe timed out after {self.PROBE_TIMEOUT:.0f}s") from e

        if result.returncode != 0:
            logger.warning(f"ffprobe failed for video: {video_path}")
            logger.warning(f"ffprobe stderr: {result.stderr}")
            raise ValueError(UNREADABLE_VIDEO_MESSAGE)

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise ValueError(UNREADABLE_VIDEO_MESSAGE) from e

        return parse_probe_output(data)
```

В старом `extract_frames` (пока живом) заменить `raise ValueError("File contains no video stream. " ...)` на `raise ValueError(NO_VIDEO_STREAM_MESSAGE)`; текст сообщения не меняется.

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_video_utils.py tests/test_frame_selection.py -v`
Expected: все PASS (три старых теста `TestExtractFramesNoVideoStream` по-прежнему зелёные, потому что мок `get_video_info` возвращает нулевые размеры и старый `extract_frames` бросает ValueError).

- [ ] **Step 5: Коммит**

```bash
git add app/video_utils.py tests/test_video_utils.py
git commit -m "feat(video): probe metadata via ffprobe JSON, parse showinfo lines

Claude-Session: https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4"
```

---

### Task 4: Первый проход — `VideoFrameExtractor.scan`

**Files:**
- Modify: `app/video_utils.py` (конструктор, новые dataclass'ы, методы `scan`, `_scan`, `_scan_motion`, вспомогательные классы)
- Test: `tests/test_video_extraction_integration.py` (новый)

**Interfaces:**
- Consumes: `select_frames`, `prepare_frame`, `blob_area`, `SelectionParams`, `SelectedFrame`, `SCAN_WIDTH`, `STORM_MEDIAN_BLOB` из `frame_selection`; `parse_showinfo_line`, `ShowinfoFrame`, `VideoInfo`, `NO_VIDEO_STREAM_MESSAGE` из Task 3; `_rc_to_str` из `ffmpeg_pipe`.
- Produces: `SelectionStats(total_frames, median_blob, storm, counts, pass1_seconds, pass2_seconds=0.0)`, `ScanResult(selected, pts, blob, info, stats)`, `VideoFrameExtractor(ffmpeg_path="ffmpeg", ffprobe_path="ffprobe", timeout=300.0)`, `VideoFrameExtractor.scan(video_path, params) -> ScanResult`, внутренние `_scan(video_path, params, deadline)`, `_scan_motion(video_path, info, deadline) -> tuple[list[float], list[float]]`. Task 5 строит `extract_frames` поверх `_scan`, Task 8 вызывает `scan` на корпусе.

- [ ] **Step 1: Написать падающие интеграционные тесты первого прохода**

Создать `tests/test_video_extraction_integration.py`:

```python
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
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_video_extraction_integration.py -v`
Expected: `AttributeError: 'VideoFrameExtractor' object has no attribute 'scan'` (или `TypeError` на `timeout=` в конструкторе).

- [ ] **Step 3: Добавить импорты, dataclass'ы и вспомогательные классы**

В `app/video_utils.py` дополнить импорты (после `import tempfile`):

```python
import threading
import time
from collections import Counter, deque
from statistics import median
```

и после `import numpy as np`:

```python
from ffmpeg_pipe import _rc_to_str
from frame_selection import (
    SCAN_WIDTH,
    STORM_MEDIAN_BLOB,
    SelectedFrame,
    SelectionParams,
    blob_area,
    prepare_frame,
    select_frames,
)
```

После `parse_showinfo` добавить:

```python
@dataclass
class SelectionStats:
    """What the selection did, for the INFO log line and the corpus check."""
    total_frames: int
    median_blob: float
    storm: bool
    counts: dict[str, int]      # selected frames per reason
    pass1_seconds: float
    pass2_seconds: float = 0.0


@dataclass
class ScanResult:
    """Pass 1 output: the selection and everything needed to fetch the frames."""
    selected: list[SelectedFrame]
    pts: list[float]            # presentation time of every decoded frame
    blob: list[float]           # metric of every decoded frame; blob[0] == 0.0
    info: VideoInfo
    stats: SelectionStats


class _StderrCollector:
    """Daemon thread draining an ffmpeg stderr pipe so it never blocks.
    Keeps every showinfo frame and the last non-showinfo lines for error messages."""

    def __init__(self, stream, keep_lines: int = 50):
        self.frames: list[ShowinfoFrame] = []
        self.tail: deque[str] = deque(maxlen=keep_lines)
        self._stream = stream
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for raw in self._stream:
                line = raw.decode("utf-8", errors="replace").rstrip()
                parsed = parse_showinfo_line(line)
                if parsed is not None:
                    self.frames.append(parsed)
                else:
                    self.tail.append(line)
        except (ValueError, OSError):
            pass  # pipe closed

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout)

    def tail_text(self) -> str:
        return "\n".join(self.tail)[-2000:]


class _Deadline:
    """Kills a subprocess when the wall-clock deadline passes, even if the
    main thread is blocked in a pipe read."""

    def __init__(self, process: subprocess.Popen, deadline: float):
        self.expired = False
        self._process = process
        self._timer = threading.Timer(max(0.0, deadline - time.monotonic()), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.expired = True
        try:
            self._process.kill()
        except OSError:
            pass  # already gone

    def cancel(self) -> None:
        self._timer.cancel()


def _finish_process(process: subprocess.Popen, wait_seconds: float = 10.0) -> int | None:
    """Wait for ffmpeg to exit; escalate to SIGKILL. None means it never exited."""
    try:
        return process.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            return process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg did not exit after SIGKILL; process may be leaked")
            return None
```

- [ ] **Step 4: Заменить конструктор и добавить методы первого прохода**

Заменить `__init__` класса `VideoFrameExtractor` (докстринг класса и параметры `scene_threshold`/`min_interval` удаляются):

```python
class VideoFrameExtractor:
    """Motion-based key frame extraction with ffmpeg.

    Pass 1 (``scan``) decodes the whole video into gray 640 px frames through a
    pipe, computes the ``blob`` motion metric per frame and applies
    ``select_frames``. Pass 2 (``extract_frames``) decodes again and fetches
    only the selected frames in full resolution.
    """

    PROBE_TIMEOUT = 30.0

    def __init__(
            self,
            ffmpeg_path: str = "ffmpeg",
            ffprobe_path: str = "ffprobe",
            timeout: float = 300.0,
    ):
        """
        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
            timeout: Wall-clock deadline for one extraction (both passes), seconds
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.timeout = timeout

        self._verify_ffmpeg()
```

`PROBE_TIMEOUT`, добавленный в Task 3 перед `get_video_info`, перенести сюда (одно определение). После `get_video_info` добавить:

```python
    def scan(self, video_path: str, params: SelectionParams) -> ScanResult:
        """Pass 1 only: decode, measure motion, select. No full-resolution frames."""
        return self._scan(video_path, params, deadline=time.monotonic() + self.timeout)

    def _scan(self, video_path: str, params: SelectionParams, deadline: float) -> ScanResult:
        info = self.get_video_info(video_path)
        if info.width == 0 or info.height == 0:
            raise ValueError(NO_VIDEO_STREAM_MESSAGE)
        logger.info(
            f"Video: duration={info.duration:.2f}s, resolution={info.width}x{info.height}, "
            f"fps={info.fps:.2f}, codec={info.codec}"
        )

        started = time.monotonic()
        pts, blob = self._scan_motion(video_path, info, deadline)
        pass1_seconds = time.monotonic() - started

        if info.duration <= 0.0:
            logger.warning(f"ffprobe gave no duration; using the last frame pts {pts[-1]:.3f}s")
            info.duration = pts[-1]
        if any(later < earlier for earlier, later in zip(pts, pts[1:])):
            logger.warning("Frame pts are not monotonic; selection uses them as reported")

        selected = select_frames(pts, blob, params)
        median_blob = float(median(blob[1:])) if len(blob) >= 2 else 0.0
        stats = SelectionStats(
            total_frames=len(pts),
            median_blob=median_blob,
            storm=median_blob > STORM_MEDIAN_BLOB,
            counts=dict(Counter(f.reason for f in selected)),
            pass1_seconds=pass1_seconds,
        )
        return ScanResult(selected=selected, pts=pts, blob=blob, info=info, stats=stats)

    def _scan_motion(
            self, video_path: str, info: VideoInfo, deadline: float
    ) -> tuple[list[float], list[float]]:
        """Stream gray 640 px frames from ffmpeg and compute the blob metric per frame.

        Memory is O(1) in the number of frames: only the previous prepared
        frame and two lists of floats are kept.
        """
        scaled_h = int(round(info.height * SCAN_WIDTH / info.width / 2)) * 2
        frame_size = SCAN_WIDTH * scaled_h
        cmd = [
            self.ffmpeg_path, "-hide_banner", "-nostats", "-loglevel", "info",
            "-an", "-i", video_path,
            "-vf", f"scale={SCAN_WIDTH}:{scaled_h},showinfo",
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        logger.debug(f"Motion scan command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        collector = _StderrCollector(process.stderr)
        killer = _Deadline(process, deadline)
        blob: list[float] = []
        prev = None
        try:
            while True:
                raw = process.stdout.read(frame_size)  # BufferedReader: short only at EOF
                if len(raw) < frame_size:
                    break
                cur = prepare_frame(np.frombuffer(raw, np.uint8).reshape(scaled_h, SCAN_WIDTH))
                blob.append(blob_area(prev, cur) if prev is not None else 0.0)
                prev = cur
        finally:
            process.stdout.close()
            returncode = _finish_process(process)
            killer.cancel()
            collector.join()

        if killer.expired:
            raise RuntimeError(
                f"Frame extraction timed out after {self.timeout:.0f}s during the motion scan"
            )
        if not blob:
            raise RuntimeError(
                f"FFmpeg produced no frames ({_rc_to_str(returncode)}): {collector.tail_text()}"
            )
        if returncode != 0:
            logger.warning(
                f"FFmpeg motion scan exited with {_rc_to_str(returncode)} after {len(blob)} frames: "
                f"{collector.tail_text()}"
            )

        pts = [f.pts for f in collector.frames]
        if len(pts) != len(blob):
            logger.warning(
                f"Motion scan: {len(pts)} showinfo frames vs {len(blob)} decoded frames; "
                f"truncating to the shorter"
            )
            n = min(len(pts), len(blob))
            if n == 0:
                raise RuntimeError("FFmpeg produced frames without showinfo timestamps")
            pts, blob = pts[:n], blob[:n]
        return pts, blob
```

Старый `extract_frames` в этой задаче ещё компилируется, но обращается к удалённым `self.scene_threshold` и `self.min_interval` только при вызове; тесты `TestExtractFramesNoVideoStream` падают раньше, на проверке размеров, поэтому остаются зелёными. Task 5 удаляет старый код.

- [ ] **Step 5: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_video_extraction_integration.py tests/test_video_utils.py -v`
Expected: все PASS. Первый проход по 10-секундному клипу 320×240 занимает меньше секунды.

- [ ] **Step 6: Коммит**

```bash
git add app/video_utils.py tests/test_video_extraction_integration.py
git commit -m "feat(video): stream motion scan through an ffmpeg pipe

Claude-Session: https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4"
```

---

### Task 5: Второй проход, `extract_frames`, async-обёртка, удаление старого пути

**Files:**
- Modify: `app/video_utils.py` (удалить старые `extract_frames`, `_extract_frames_fallback`, `_parse_ffmpeg_timestamps`, `_load_frames`, `extract_frames_from_video`; добавить `_grab_frames`, `extract_frames`, `ExtractionResult`, новую обёртку)
- Test: `tests/test_video_extraction_integration.py`, `tests/test_video_utils.py`

**Interfaces:**
- Consumes: `_scan`, `ScanResult`, `SelectionStats`, `parse_showinfo`, `PTS_TOLERANCE` (импортировать из `frame_selection`), `_rc_to_str`.
- Produces: `ExtractedFrame(image: np.ndarray (BGR uint8 H×W×3), timestamp: float, frame_number: int, reason: str)`, `ExtractionResult(frames: list[ExtractedFrame], info: VideoInfo, stats: SelectionStats)`, `VideoFrameExtractor.extract_frames(video_path, params) -> ExtractionResult`, `async extract_frames_from_video(video_data: bytes, params: SelectionParams) -> ExtractionResult`. Task 6 переключает эндпоинты на них.

После этой задачи и до Task 6 эндпоинты в `main.py` вызывают `extract_frames_from_video` со старыми аргументами; импорт остаётся валидным, существующие тесты эндпоинтов эти вызовы не делают. Задачи 5 и 6 выполнять подряд.

- [ ] **Step 1: Обновить моковые тесты и написать падающие интеграционные тесты второго прохода**

В `tests/test_video_utils.py` в трёх тестах класса `TestExtractFramesNoVideoStream` заменить вызов `extractor.extract_frames("/tmp/fake.mp4")` на `extractor.extract_frames("/tmp/fake.mp4", SelectionParams())` и добавить импорт `from frame_selection import SelectionParams`.

В `tests/test_video_extraction_integration.py` добавить импорты:

```python
import asyncio
import time

import numpy as np

from video_utils import extract_frames_from_video
```

и фикстуру после `long_clip`:

```python
@pytest.fixture(scope="module")
def rotated_clip(clips_dir, static_clip):
    """The static clip with a 90° display-rotation matrix, stream-copied."""
    out = clips_dir / "rotated.mp4"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90",
         "-i", str(static_clip), "-c", "copy", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg lacks -display_rotation: {result.stderr.strip()[:200]}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "v:0", str(out)],
        capture_output=True, text=True, check=True,
    )
    if '"rotation"' not in probe.stdout:
        pytest.skip("ffmpeg did not write rotation metadata")
    return out
```

и класс тестов в конец файла:

```python
class TestExtractFrames:
    def test_static_clip_frames(self, extractor, static_clip):
        result = extractor.extract_frames(str(static_clip), SelectionParams())

        assert [f.frame_number for f in result.frames] == [0, 40, 80]
        assert [f.reason for f in result.frames] == ["first", "grid", "grid"]
        assert [round(f.timestamp, 1) for f in result.frames] == [0.0, 4.0, 8.0]
        for frame in result.frames:
            assert frame.image.shape == (240, 320, 3)
            assert frame.image.dtype == np.uint8
            assert frame.image.flags.writeable
        assert result.info.duration == pytest.approx(10.0, abs=0.1)
        assert result.stats.pass2_seconds > 0
        assert result.stats.counts == {"first": 1, "grid": 2}

    def test_frames_are_the_selected_ones(self, extractor, motion_clip):
        """The white box is on screen only for t in [1, 3]: motion frames inside
        that window contain white pixels, frame 0 is black."""
        result = extractor.extract_frames(str(motion_clip), SelectionParams())

        by_number = {f.frame_number: f for f in result.frames}
        assert by_number[0].image.max() < 60
        visible = [f for f in result.frames if f.reason == "motion" and 1.0 <= f.timestamp <= 3.0]
        assert visible
        assert all(f.image.max() > 200 for f in visible)

    def test_rotated_clip_uses_display_dimensions(self, extractor, rotated_clip):
        info = extractor.get_video_info(str(rotated_clip))
        assert (info.width, info.height, info.rotation) == (240, 320, 90)

        result = extractor.extract_frames(str(rotated_clip), SelectionParams())
        assert result.frames[0].image.shape == (320, 240, 3)

    def test_max_frames_two_keeps_first_and_last_grid_frame(self, extractor, static_clip):
        result = extractor.extract_frames(str(static_clip), SelectionParams(max_frames=2))
        assert [f.frame_number for f in result.frames] == [0, 80]

    def test_async_wrapper_returns_result(self, static_clip):
        result = asyncio.run(extract_frames_from_video(static_clip.read_bytes(), SelectionParams(max_frames=2)))
        assert [f.frame_number for f in result.frames] == [0, 80]
        assert result.info.duration == pytest.approx(10.0, abs=0.1)

    def test_frame_grab_mismatch_raises_runtime_error(self, extractor, static_clip, monkeypatch):
        """If the second pass returns frames that do not match the scan, fail loudly."""
        scan = extractor.scan(str(static_clip), SelectionParams())
        wrong_pts = [t + 0.5 for t in scan.pts]  # every pts off by half a second
        with pytest.raises(RuntimeError, match="does not match"):
            extractor._grab_frames(str(static_clip), scan.info, scan.selected, wrong_pts,
                                   deadline=time.monotonic() + extractor.timeout)
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_video_extraction_integration.py -v -k "ExtractFrames"`
Expected: FAIL: `TypeError` (старый `extract_frames` не принимает `params`, старая `extract_frames_from_video` не знает `params`) и `AttributeError` (`_grab_frames`).

- [ ] **Step 3: Удалить старый путь и добавить второй проход**

В `app/video_utils.py`:

1. Добавить `PTS_TOLERANCE` в импорт из `frame_selection`.
2. Добавить поле `reason: str` в `ExtractedFrame` (после `frame_number`) и после `ScanResult` добавить:

```python
@dataclass
class ExtractionResult:
    """Both passes done: the selected frames in full resolution plus metadata."""
    frames: list[ExtractedFrame]
    info: VideoInfo
    stats: SelectionStats
```

3. Удалить методы `extract_frames`, `_extract_frames_fallback`, `_parse_ffmpeg_timestamps`, `_load_frames` и функцию `extract_frames_from_video` целиком. Удалить неиспользуемые после этого импорты `cv2`, `Path`, `Optional`.
4. После `_scan_motion` добавить:

```python
    def extract_frames(self, video_path: str, params: SelectionParams) -> ExtractionResult:
        """Both passes: select frames by motion, then fetch them in full resolution (BGR)."""
        deadline = time.monotonic() + self.timeout
        scan = self._scan(video_path, params, deadline)

        started = time.monotonic()
        frames = self._grab_frames(video_path, scan.info, scan.selected, scan.pts, deadline)
        scan.stats.pass2_seconds = time.monotonic() - started

        counts = scan.stats.counts
        logger.info(
            f"Frame selection: {len(frames)} frames (first {counts.get('first', 0)}, "
            f"grid {counts.get('grid', 0)}, motion {counts.get('motion', 0)}) of {scan.stats.total_frames}, "
            f"storm={scan.stats.storm}, median_blob={scan.stats.median_blob:.4f}, "
            f"pass1={scan.stats.pass1_seconds:.2f}s, pass2={scan.stats.pass2_seconds:.2f}s"
        )
        return ExtractionResult(frames=frames, info=scan.info, stats=scan.stats)

    def _grab_frames(
            self,
            video_path: str,
            info: VideoInfo,
            selected: list[SelectedFrame],
            pts: list[float],
            deadline: float,
    ) -> list[ExtractedFrame]:
        """Pass 2: decode again, let ``select`` pass only the chosen frames, read them as bgr24.

        Output frames are matched to the selection by pts, never by position,
        and every mismatch raises: silently returning frames with somebody
        else's timestamps is worse than failing.
        """
        if not selected:
            raise RuntimeError("No frames selected")
        expr = "+".join(f"eq(n\\,{f.index})" for f in selected)
        cmd = [
            self.ffmpeg_path, "-hide_banner", "-nostats", "-loglevel", "info",
            "-an", "-i", video_path,
            "-vf", f"select='{expr}',showinfo",
            "-fps_mode", "passthrough",
            "-frames:v", str(len(selected)),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        logger.debug(f"Frame grab command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = process.communicate(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError(
                f"Frame extraction timed out after {self.timeout:.0f}s during the frame grab"
            )

        stderr_text = err.decode("utf-8", errors="replace")
        shown = parse_showinfo(stderr_text)
        tail = "\n".join(line for line in stderr_text.splitlines() if "showinfo" not in line)[-2000:]
        rc = _rc_to_str(process.returncode)

        frame_size = info.width * info.height * 3
        if len(out) % frame_size != 0:
            raise RuntimeError(
                f"FFmpeg frame grab returned {len(out)} bytes, not a multiple of the "
                f"{info.width}x{info.height} frame size ({rc}): {tail}"
            )
        count = len(out) // frame_size
        if count != len(selected) or len(shown) != count:
            raise RuntimeError(
                f"FFmpeg frame grab returned {count} frames and {len(shown)} showinfo lines, "
                f"expected {len(selected)} ({rc}): {tail}"
            )
        if process.returncode != 0:
            logger.warning(f"FFmpeg frame grab exited with {rc} but returned all frames: {tail}")

        frames: list[ExtractedFrame] = []
        used: set[int] = set()
        for k, frame_info in enumerate(shown):
            if (frame_info.width, frame_info.height) != (info.width, info.height):
                raise RuntimeError(
                    f"FFmpeg frame size {frame_info.width}x{frame_info.height} differs from the "
                    f"expected {info.width}x{info.height}"
                )
            match = next(
                (f for f in selected
                 if f.index not in used and abs(pts[f.index] - frame_info.pts) <= PTS_TOLERANCE),
                None,
            )
            if match is None:
                raise RuntimeError(
                    f"Frame with pts {frame_info.pts:.3f} does not match any selected frame"
                )
            used.add(match.index)
            image = (
                np.frombuffer(out, dtype=np.uint8, count=frame_size, offset=k * frame_size)
                .reshape(info.height, info.width, 3)
                .copy()  # writable, independent of the pipe buffer
            )
            frames.append(ExtractedFrame(
                image=image,
                timestamp=pts[match.index],
                frame_number=match.index,
                reason=match.reason,
            ))
        frames.sort(key=lambda f: f.frame_number)
        return frames
```

5. В конец файла добавить новую обёртку:

```python
async def extract_frames_from_video(video_data: bytes, params: SelectionParams) -> ExtractionResult:
    """Async wrapper: write the upload to a temp file and run both passes in the default executor."""
    import asyncio

    def _extract() -> ExtractionResult:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_data)
            tmp_path = tmp.name
        try:
            return VideoFrameExtractor().extract_frames(tmp_path, params)
        finally:
            os.unlink(tmp_path)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _extract)
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `.venv/bin/python -m pytest tests/test_video_extraction_integration.py tests/test_video_utils.py tests/test_frame_selection.py -v`
Expected: все PASS; тест с поворотом либо PASS, либо SKIP с причиной про `-display_rotation`.

- [ ] **Step 5: Проверить, что старого кода не осталось**

Run: `grep -n -E 'scene_threshold|_extract_frames_fallback|_load_frames|_parse_ffmpeg_timestamps|fps=1/|-q:v' app/video_utils.py`
Expected: пустой вывод.

- [ ] **Step 6: Коммит**

```bash
git add app/video_utils.py tests/test_video_extraction_integration.py tests/test_video_utils.py
git commit -m "feat(video): fetch selected frames in a second ffmpeg pass, drop scene detection

Claude-Session: https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4"
```

---

### Task 6: Модели ответа и эндпоинты

**Files:**
- Modify: `app/models.py:59-64` (`FrameDetection`), `app/models.py:121-149` (`VideoDetectionSettings`, `ExtractedFrameData`)
- Modify: `app/main.py:25-48` (импорты), `app/main.py:345-349` (версия), `app/main.py:375-387` (типы query), `app/main.py:399` (версия в `/`), `app/main.py:537-713` (`/detect/video`), `app/main.py:716-837` (`/extract/frames`)
- Test: `tests/test_video_extraction_integration.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `SelectionParams` из `frame_selection`; `extract_frames_from_video(video_data, params) -> ExtractionResult` из Task 5.
- Produces: query-параметры `max_gap`, `motion_threshold`, `min_interval`, `max_frames` на обоих эндпоинтах; поле `reason` в `ExtractedFrameData` и `FrameDetection`; версия `3.0.0`.

- [ ] **Step 1: Написать падающие тесты моделей и эндпоинтов**

В `tests/test_models.py` добавить импорты в начало файла (сейчас там только `from models import JobCreatedResponse, JobStatusResponse, JobStats`):

```python
import pytest
from pydantic import ValidationError

from models import ExtractedFrameData, FrameDetection
```

и тесты в конец файла:

```python
def test_frame_detection_carries_reason():
    frame = FrameDetection(frame_number=40, timestamp=4.0, reason="grid", detections=[], count=0)
    assert frame.model_dump()["reason"] == "grid"


def test_extracted_frame_rejects_unknown_reason():
    with pytest.raises(ValidationError):
        ExtractedFrameData(frame_number=0, timestamp=0.0, reason="scene", image_base64="", width=1, height=1)


def test_video_detection_settings_removed():
    import models
    assert not hasattr(models, "VideoDetectionSettings")
```

В `tests/test_video_extraction_integration.py` добавить импорты:

```python
import base64
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import cv2
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import Settings, get_settings
from dependencies import get_job_manager, get_model_manager
from job_manager import JobManager
from main import app
```

фикстуры (после `extractor`):

```python
@asynccontextmanager
async def _noop_lifespan(app: FastAPI):
    yield


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    entry = MagicMock()
    entry.model.names = {0: "person"}
    entry.model.predict.return_value = []  # no detections on synthetic clips
    entry.model_name = "yolo26n.pt"
    mm.get_model = AsyncMock(return_value=entry)
    mm._preloaded = {"yolo26n.pt": entry}
    mm._cached = {}
    return mm


@pytest.fixture
def client(tmp_path, mock_model_manager):
    app.router.lifespan_context = _noop_lifespan
    app.dependency_overrides[get_settings] = lambda: Settings(yolo_models="{}", video_jobs_dir=str(tmp_path))
    app.dependency_overrides[get_job_manager] = lambda: JobManager(
        jobs_dir=str(tmp_path), ttl_seconds=3600, max_queued=10
    )
    app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _upload(path: Path):
    return {"file": (path.name, path.read_bytes(), "video/mp4")}
```

и тесты в конец файла:

```python
class TestExtractFramesEndpoint:
    def test_static_clip_response(self, client, static_clip):
        response = client.post("/extract/frames", files=_upload(static_clip))

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True
        assert body["video_duration"] == pytest.approx(10.0, abs=0.1)
        assert body["video_resolution"] == [320, 240]
        assert body["frames_extracted"] == 3
        assert [f["frame_number"] for f in body["frames"]] == [0, 40, 80]
        assert [f["reason"] for f in body["frames"]] == ["first", "grid", "grid"]
        assert [round(f["timestamp"], 1) for f in body["frames"]] == [0.0, 4.0, 8.0]
        first = body["frames"][0]
        assert (first["width"], first["height"]) == (320, 240)
        jpeg = base64.b64decode(first["image_base64"])
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (240, 320, 3)

    def test_legacy_client_query_still_works(self, client, static_clip):
        """frigate-analyzer sends scene_threshold until it is updated; FastAPI ignores it."""
        response = client.post(
            "/extract/frames?scene_threshold=0.05&min_interval=1.0&max_frames=50&quality=85",
            files=_upload(static_clip),
        )
        assert response.status_code == 200, response.text
        assert response.json()["frames_extracted"] == 3

    def test_max_frames_caps_output(self, client, static_clip):
        response = client.post("/extract/frames?max_frames=2", files=_upload(static_clip))
        assert response.status_code == 200, response.text
        assert [f["frame_number"] for f in response.json()["frames"]] == [0, 80]

    def test_max_gap_changes_grid(self, client, static_clip):
        response = client.post("/extract/frames?max_gap=2", files=_upload(static_clip))
        assert response.status_code == 200, response.text
        assert [f["frame_number"] for f in response.json()["frames"]] == [0, 20, 40, 60, 80]

    def test_out_of_range_max_gap_rejected(self, client, static_clip):
        response = client.post("/extract/frames?max_gap=0.1", files=_upload(static_clip))
        assert response.status_code == 422

    def test_audio_only_file_rejected_with_422(self, client, tmp_path):
        audio = tmp_path / "audio.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)],
            check=True, timeout=60,
        )
        response = client.post("/extract/frames", files=_upload(audio))
        assert response.status_code == 422
        assert "no video stream" in response.json()["detail"]


class TestDetectVideoEndpoint:
    def test_static_clip_response(self, client, static_clip):
        response = client.post("/detect/video", files=_upload(static_clip))

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["frames_analyzed"] == 3
        assert body["video_duration"] == pytest.approx(10.0, abs=0.1)
        assert body["video_resolution"] == [320, 240]
        assert [f["frame_number"] for f in body["frames"]] == [0, 40, 80]
        assert [f["reason"] for f in body["frames"]] == ["first", "grid", "grid"]
        assert body["total_detections"] == 0
        assert body["model"] == "yolo26n.pt"

    def test_selection_params_are_passed_through(self, client, static_clip):
        response = client.post("/detect/video?max_gap=2&max_frames=3", files=_upload(static_clip))
        assert response.status_code == 200, response.text
        assert [f["frame_number"] for f in response.json()["frames"]] == [0, 40, 80]
```

Последний тест: сетка с шагом 2 с даёт 0, 20, 40, 60, 80 (5 кадров), кап 3 прореживает до первого, среднего и последнего: 0, 40, 80.

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `.venv/bin/python -m pytest tests/test_models.py tests/test_video_extraction_integration.py -v -k "reason or Endpoint or settings_removed"`
Expected: FAIL (`ValidationError` на `reason`, `hasattr` истинен, эндпоинты возвращают 500 из-за старых аргументов `extract_frames_from_video`).

- [ ] **Step 3: Обновить модели**

В `app/models.py` заменить `FrameDetection`:

```python
class FrameDetection(BaseModel):
    """Detection results for a single video frame."""
    frame_number: int = Field(description="Frame index in the source video (0-based)")
    timestamp: float = Field(description="Frame presentation time in seconds")
    reason: Literal["first", "grid", "motion"] = Field(
        description="Why the frame was selected: the first frame, the time grid, or a motion peak"
    )
    detections: list[Detection] = Field(default_factory=list)
    count: int = Field(description="Number of detections in frame")
```

Удалить класс `VideoDetectionSettings` целиком. Заменить `ExtractedFrameData`:

```python
class ExtractedFrameData(BaseModel):
    """Single extracted frame with base64-encoded image data."""
    frame_number: int = Field(description="Frame index in the source video (0-based)")
    timestamp: float = Field(description="Frame presentation time in seconds")
    reason: Literal["first", "grid", "motion"] = Field(
        description="Why the frame was selected: the first frame, the time grid, or a motion peak"
    )
    image_base64: str = Field(description="JPEG image encoded as base64 string")
    width: int = Field(description="Frame width in pixels")
    height: int = Field(description="Frame height in pixels")
```

- [ ] **Step 4: Обновить `main.py`: импорты, версия, типы query-параметров**

Добавить импорт после `from detection_stabilizer import StabilizerConfig`:

```python
from frame_selection import SelectionParams
```

В `FastAPI(...)` заменить `version="2.3.0"` на `version="3.0.0"`; в `root()` заменить `"version": "2.2.0"` на `"version": "3.0.0"`.

Заменить блок «Video-specific query parameters» (`SceneThresholdQuery`, `MinIntervalQuery`, `MaxFramesQuery`) на:

```python
# Frame selection query parameters, shared by /detect/video and /extract/frames
MaxGapQuery = Annotated[
    float,
    Query(ge=0.5, le=30.0, description="Grid step in seconds: a frame is always taken once this much time passed since the previous selected one")
]
MotionThresholdQuery = Annotated[
    float,
    Query(ge=0.0001, le=0.1, description="Motion threshold: area of the largest changed region as a fraction of the frame")
]
MinIntervalQuery = Annotated[
    float,
    Query(ge=0.1, le=30.0, description="Minimum interval between any two selected frames (seconds)")
]
MaxFramesQuery = Annotated[
    int,
    Query(ge=1, le=200, description="Maximum number of frames to select")
]
```

- [ ] **Step 5: Переписать `/detect/video`**

Заменить сигнатуру и докстринг:

```python
@app.post("/detect/video", response_model=VideoDetectionResponse, tags=["Video Detection"])
async def detect_objects_in_video(
        file: UploadFile = File(..., description="Video file for analysis"),
        conf: ConfidenceQuery = 0.5,
        imgsz: ImageSizeQuery = 640,
        max_det: MaxDetQuery = 100,
        max_gap: MaxGapQuery = 4.0,
        motion_threshold: MotionThresholdQuery = 0.001,
        min_interval: MinIntervalQuery = 1.0,
        max_frames: MaxFramesQuery = 6,
        model: ModelQuery = None,
        model_manager: ModelManager = Depends(get_model_manager),
        settings: Settings = Depends(get_settings)
):
    """
    Analyze video using YOLO object detection on motion-selected frames.

    **Frame selection:**
    1. The first frame is always taken (`reason=first`).
    2. A grid frame is taken every `max_gap` seconds (`reason=grid`); a grid longer
       than `max_frames` is thinned uniformly.
    3. The strongest motion peaks above `motion_threshold` fill the remaining budget,
       never closer than `min_interval` to another selected frame (`reason=motion`).
       Segments where nearly every frame changes (rain or snow in IR) get the grid only.

    **Parameters:**
    - **file**: Video file (MP4, AVI, MOV, MKV, WEBM, WMV, FLV)
    - **conf**: Confidence threshold (0.0 - 1.0)
    - **imgsz**: Image size for processing
    - **max_det**: Maximum detections per frame
    - **max_gap**: Grid step in seconds (0.5-30)
    - **motion_threshold**: Motion threshold as a fraction of the frame (0.0001-0.1)
    - **min_interval**: Minimum seconds between selected frames (0.1-30)
    - **max_frames**: Maximum frames to analyze (1-200)
    - **model**: Model name (e.g. yolo26s.pt). If not specified, uses first preloaded model.
    """
```

Заменить лог-строку `Processing video: ...` на:

```python
    logger.info(
        f"Processing video: {file.filename}, conf={conf}, model={model_name}, "
        f"max_gap={max_gap}, motion_threshold={motion_threshold}, "
        f"min_interval={min_interval}, max_frames={max_frames}"
    )
```

Заменить блок от `# Extract frames` до `logger.info(f"Extracted {len(frames)} frames from video")` включительно на:

```python
    # Extract frames
    params = SelectionParams(
        max_gap=max_gap,
        motion_threshold=motion_threshold,
        min_interval=min_interval,
        max_frames=max_frames,
    )
    try:
        extraction = await extract_frames_from_video(video_data=video_data, params=params)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract frames: {str(e)}"
        )

    frames = extraction.frames
    logger.info(f"Extracted {len(frames)} frames from video")
```

Заменить две строки:

```python
    # Estimate video duration from last frame timestamp
    video_duration = frames[-1].timestamp if frames else 0.0
```

на:

```python
    video_duration = extraction.info.duration
```

В построении `FrameDetection` добавить `reason`:

```python
        frame_results.append(FrameDetection(
            frame_number=frame.frame_number,
            timestamp=round(frame.timestamp, 3),
            reason=frame.reason,
            detections=frame_detections,
            count=len(frame_detections)
        ))
```

Комментарий `# Get video dimensions from first frame` и код под ним остаются: `frame.image` теперь BGR, `run_inference` получает его как есть, как и `/detect`.

- [ ] **Step 6: Переписать `/extract/frames`**

Заменить функцию целиком:

```python
@app.post("/extract/frames", response_model=FrameExtractionResponse, tags=["Frame Extraction"])
async def extract_video_frames(
        file: UploadFile = File(..., description="Video file for frame extraction"),
        max_gap: MaxGapQuery = 4.0,
        motion_threshold: MotionThresholdQuery = 0.001,
        min_interval: MinIntervalQuery = 1.0,
        max_frames: MaxFramesQuery = 6,
        quality: Annotated[int, Query(ge=1, le=100, description="JPEG quality")] = 85
):
    """
    Extract motion-selected key frames from video without object detection.

    Returns frames as base64-encoded JPEG images.

    **Frame selection:**
    1. The first frame is always taken (`reason=first`).
    2. A grid frame is taken every `max_gap` seconds (`reason=grid`); a grid longer
       than `max_frames` is thinned uniformly.
    3. The strongest motion peaks above `motion_threshold` fill the remaining budget,
       never closer than `min_interval` to another selected frame (`reason=motion`).
       Segments where nearly every frame changes (rain or snow in IR) get the grid only.

    **Parameters:**
    - **file**: Video file (MP4, AVI, MOV, MKV, WEBM, WMV, FLV)
    - **max_gap**: Grid step in seconds (0.5-30)
    - **motion_threshold**: Motion threshold as a fraction of the frame (0.0001-0.1)
    - **min_interval**: Minimum seconds between selected frames (0.1-30)
    - **max_frames**: Maximum frames to extract (1-200)
    - **quality**: JPEG compression quality (1-100)
    """
    start_time = time.perf_counter()

    # Validate file extension
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            )

    logger.info(
        f"Extracting frames from video: {file.filename}, max_gap={max_gap}, "
        f"motion_threshold={motion_threshold}, min_interval={min_interval}, max_frames={max_frames}"
    )

    # Read video data with size check
    video_data = await file.read()
    if len(video_data) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Video too large. Maximum size: {MAX_VIDEO_SIZE // (1024 * 1024)} MB"
        )

    # Extract frames
    params = SelectionParams(
        max_gap=max_gap,
        motion_threshold=motion_threshold,
        min_interval=min_interval,
        max_frames=max_frames,
    )
    try:
        extraction = await extract_frames_from_video(video_data=video_data, params=params)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract frames: {str(e)}"
        )

    frames = extraction.frames
    logger.info(f"Extracted {len(frames)} frames from video")

    # Get video dimensions from first frame
    video_height, video_width = frames[0].image.shape[:2]

    # Convert frames to base64-encoded JPEG (frames are BGR, which is what cv2 expects)
    frame_results: list[ExtractedFrameData] = []
    for frame in frames:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, jpeg_data = cv2.imencode('.jpg', frame.image, encode_params)

        if not success:
            logger.warning(f"Failed to encode frame {frame.frame_number}")
            continue

        image_base64 = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')

        height, width = frame.image.shape[:2]
        frame_results.append(ExtractedFrameData(
            frame_number=frame.frame_number,
            timestamp=round(frame.timestamp, 3),
            reason=frame.reason,
            image_base64=image_base64,
            width=width,
            height=height
        ))

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    response = FrameExtractionResponse(
        video_duration=round(extraction.info.duration, 3),
        video_resolution=(video_width, video_height),
        frames_extracted=len(frame_results),
        frames=frame_results,
        processing_time_ms=processing_time_ms
    )

    logger.info(
        f"Frame extraction completed: {len(frame_results)} frames, "
        f"{processing_time_ms}ms"
    )

    return response
```

- [ ] **Step 7: Убедиться, что весь набор тестов проходит**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: все PASS (кроме SKIP теста с поворотом на старом ffmpeg). Затем:

Run: `grep -rn -E 'scene_threshold|SceneThresholdQuery|VideoDetectionSettings|COLOR_RGB2BGR' app/`
Expected: пустой вывод.

- [ ] **Step 8: Коммит**

```bash
git add app/main.py app/models.py tests/test_models.py tests/test_video_extraction_integration.py
git commit -m "feat(api): motion-based frame selection parameters on video endpoints

Claude-Session: https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4"
```

---

### Task 7: Документация

**Files:**
- Modify: `.claude/rules/api.md:39-63` (`/detect/video`), `.claude/rules/api.md:85-108` (`/extract/frames`)
- Modify: `CLAUDE.md` (таблица архитектуры, таблица эндпоинтов, Key Patterns)

**Interfaces:** нет кода.

- [ ] **Step 1: Обновить `.claude/rules/api.md`**

В разделе `### POST /detect/video` заменить таблицу параметров и раздел «Frame Extraction Algorithm» на:

```markdown
Video analysis on motion-selected frames.

**Parameters:**
| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `file` | file | required | — | Video file |
| `conf` | float | 0.5 | 0.0-1.0 | Confidence threshold |
| `imgsz` | int | 640 | 32-2016 | Inference image size |
| `max_det` | int | 100 | 1-1000 | Max detections per frame |
| `max_gap` | float | 4.0 | 0.5-30.0 | Grid step, seconds |
| `motion_threshold` | float | 0.001 | 0.0001-0.1 | Motion threshold, fraction of the frame |
| `min_interval` | float | 1.0 | 0.1-30.0 | Min seconds between any two selected frames |
| `max_frames` | int | 6 | 1-200 | Cap on selected frames |
| `model` | string | null | — | Model name |

**Frame selection** (`app/frame_selection.py`):
1. Frame 0 is always taken (`reason: first`).
2. A grid frame every `max_gap` seconds (`reason: grid`). A grid longer than `max_frames` is thinned uniformly, keeping the first and the last grid frame.
3. Motion peaks fill the remaining budget: frames are ranked by `blob`, the area of the largest changed region between neighbouring frames (gray, 640 px wide) as a fraction of the frame, taken while above `motion_threshold` and at least `min_interval` from every selected frame (`reason: motion`). If the median `blob` over the segment exceeds 0.02 (rain, snow in IR), peaks are skipped and only the grid remains.

`frame_number` is the frame index in the source video (0-based), `timestamp` its presentation time, `video_duration` comes from ffprobe. Unknown query parameters (e.g. the removed `scene_threshold`) are ignored.
```

В разделе `### POST /extract/frames` заменить содержимое на:

```markdown
Extract motion-selected key frames without detection.

**Parameters:** `file`, `max_gap`, `motion_threshold`, `min_interval`, `max_frames` as in `/detect/video`, plus `quality` (int, 85, 1-100, JPEG quality).

**Response:**
```json
{
  "success": true,
  "video_duration": 16.0,
  "video_resolution": [2880, 1620],
  "frames_extracted": 4,
  "frames": [
    {"frame_number": 0,   "timestamp": 0.0,  "reason": "first",  "image_base64": "...", "width": 2880, "height": 1620},
    {"frame_number": 31,  "timestamp": 2.48, "reason": "motion", "image_base64": "...", "width": 2880, "height": 1620},
    {"frame_number": 50,  "timestamp": 4.0,  "reason": "grid",   "image_base64": "...", "width": 2880, "height": 1620},
    {"frame_number": 100, "timestamp": 8.0,  "reason": "grid",   "image_base64": "...", "width": 2880, "height": 1620}
  ],
  "processing_time_ms": 1700
}
```

The list is never empty on success: frame 0 is always included. Cost: about 1.7 s for a 16-second 2880×1620 segment (two ffmpeg decodes: a 640 px gray scan and a fetch of the selected frames).
```

В разделе «Status Codes» добавить строку:

```markdown
- `422` — Unreadable video, no video stream, or query parameter out of range
```

- [ ] **Step 2: Обновить `CLAUDE.md`**

В таблице Architecture после строки `app/video_utils.py` заменить её описание на `FFmpeg two-pass frame extraction: motion scan, selected-frame fetch` и добавить строку:

```markdown
| `app/frame_selection.py` | Motion metric (`blob`) and the frame selection rule (grid + motion peaks) |
```

В таблице Endpoints заменить описания:

```markdown
| `/detect/video` | POST | Video detection on motion-selected frames |
| `/extract/frames` | POST | Extract motion-selected frames as base64 |
```

В Key Patterns заменить абзац `**Smart Frames**: ...` на:

```markdown
**Motion Frames**: Two ffmpeg passes. Pass 1 streams gray 640 px frames through a pipe and computes `blob`, the area of the largest changed region between neighbouring frames; pass 2 fetches only the selected frames with `select`. Selection: frame 0, a grid every `max_gap` (4 s), then the strongest motion peaks above `motion_threshold` (0.001) at least `min_interval` (1 s) apart, capped at `max_frames` (6). A segment whose median `blob` exceeds 0.02 (rain, snow in IR) gets the grid only. Real pts from `showinfo`; `video_duration` from ffprobe. Tests use lavfi-generated clips, no binary fixtures.
```

- [ ] **Step 3: Проверить, что документация не ссылается на удалённое**

Run: `grep -rn -E 'scene_threshold|Smart Frames|scene change' CLAUDE.md .claude/rules/api.md`
Expected: пустой вывод.

- [ ] **Step 4: Коммит**

```bash
git add CLAUDE.md .claude/rules/api.md
git commit -m "docs(api): describe motion-based frame selection

Claude-Session: https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4"
```

---

### Task 8: Проверка на корпусе, замер времени, smoke

**Files:**
- Create: `~/vision-api-research/frame-selection/run_impl.py` (вне репозитория, рядом с корпусом)

**Interfaces:**
- Consumes: `VideoFrameExtractor.scan`, `SelectionParams`; функции `load` и структуры сегментов из `~/vision-api-research/frame-selection/evaluate.py` (`seg["key"]`, `seg["k"]`, `seg["tracks"]`, `seg["hit"]`, `seg["n"]`).
- Produces: числа для описания PR.

- [ ] **Step 1: Написать скрипт проверки**

Создать `~/vision-api-research/frame-selection/run_impl.py`:

```python
#!/usr/bin/env python3
"""Score the real VideoFrameExtractor.scan() against the YOLO oracle, reusing evaluate.py.

Run from ~/vision-api-research/frame-selection/ with the repository's venv:
    /opt/github/zinin/vision-api-server/.venv/bin/python run_impl.py
"""
import os
import sys
import time

S = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("VISION_API_REPO", "/opt/github/zinin/vision-api-server")
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, S)

import numpy as np  # noqa: E402

import evaluate  # noqa: E402  (loads metrics + oracle from the directories next to this file)
from frame_selection import SelectionParams  # noqa: E402
from video_utils import VideoFrameExtractor  # noqa: E402


def main() -> None:
    segs = evaluate.load()
    extractor = VideoFrameExtractor()
    params = SelectionParams()
    moving = moving_hit = 0
    frames_per_seg = []
    empty_frames = []
    frame_count_mismatch = []
    reasons = {"first": 0, "grid": 0, "motion": 0}
    storms = 0
    elapsed = 0.0
    for seg in segs:
        path = os.path.join(S, "corpus", seg["key"])
        started = time.time()
        result = extractor.scan(path, params)
        elapsed += time.time() - started
        if result.stats.total_frames != seg["n"]:
            frame_count_mismatch.append((seg["key"], result.stats.total_frames, seg["n"]))
        idx = [f.index for f in result.selected]
        near = set(int(round(i / seg["k"])) * seg["k"] for i in idx)
        for track in seg["tracks"]:
            if track["static"]:
                continue
            moving += 1
            moving_hit += bool(track["fset"] & near)
        frames_per_seg.append(len(idx))
        if not any(seg["hit"]):
            empty_frames.append(len(idx))
        for reason, count in result.stats.counts.items():
            reasons[reason] += count
        storms += result.stats.storm
    fr = np.array(frames_per_seg)
    print(f"segments: {len(segs)}, scan time {elapsed / len(segs):.2f} s/segment")
    print(f"moving events: {moving_hit}/{moving} = {moving_hit / max(moving, 1):.1%}")
    print(f"frames/segment: mean {fr.mean():.2f}, p95 {np.percentile(fr, 95):.1f}, max {fr.max()}")
    print(f"frames on empty segments: mean {np.mean(empty_frames):.2f}")
    print(f"reasons: {reasons}, storm segments: {storms}")
    if frame_count_mismatch:
        print(f"frame count mismatches vs metrics.py ({len(frame_count_mismatch)}):")
        for key, got, expected in frame_count_mismatch[:10]:
            print(f"  {key}: scan {got} vs metrics {expected}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Запустить проверку**

Run:
```bash
cd ~/vision-api-research/frame-selection && /opt/github/zinin/vision-api-server/.venv/bin/python run_impl.py
```
Expected: `moving events` в диапазоне 14/17–16/17 (отчёт: 15/17), `frames/segment` около 4.7 (допуск ±0.3), `frames on empty segments` около 4.7, `frame count mismatches` пусто. Прогон занимает около 4 минут (207 сегментов по 1 с). Результат за пределами допуска — повод для superpowers:systematic-debugging, а не для правки ожиданий; первое, что сверить, — `blob` реализации с массивом `blob` из `metrics/<segment>.json` на одном сегменте.

- [ ] **Step 3: Замерить время полного извлечения на большом сегменте**

Run:
```bash
cd ~/vision-api-research/frame-selection && /opt/github/zinin/vision-api-server/.venv/bin/python - <<'EOF'
import glob, os, subprocess, sys, time
sys.path.insert(0, "/opt/github/zinin/vision-api-server/app")
from frame_selection import SelectionParams
from video_utils import VideoFrameExtractor
# the first 16-second 2880-wide segment of the corpus
for path in sorted(glob.glob("corpus/**/*.mp4", recursive=True)):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width:format=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout.split()
    if probe and probe[0].startswith("2880") and float(probe[-1]) > 15:
        break
ex = VideoFrameExtractor()
t = time.time(); result = ex.extract_frames(path, SelectionParams()); dt = time.time() - t
print(path, f"{dt:.2f}s", result.stats, [ (f.frame_number, round(f.timestamp, 2), f.reason) for f in result.frames])
EOF
```
Expected: около 1.7 с (отчёт: 1.69 с для двухпроходного прототипа), `pass1` около 0.9 с, `pass2` около 0.7 с.

- [ ] **Step 4: Ручной smoke через HTTP**

В одном терминале:
```bash
cd /opt/github/zinin/vision-api-server/app && YOLO_MODELS='{}' YOLO_DEVICE=cpu ../.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8010
```
В другом:
```bash
SEG=$(find ~/vision-api-research/frame-selection/corpus -name '*.mp4' | head -1)
curl -s -F "file=@$SEG" 'http://127.0.0.1:8010/extract/frames?scene_threshold=0.05&min_interval=1.0&max_frames=50&quality=85' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['video_duration'], d['video_resolution'], d['frames_extracted'], [(f['frame_number'], f['timestamp'], f['reason'], len(f['image_base64'])) for f in d['frames']])"
curl -s -o /dev/null -w '%{http_code}\n' -F "file=@$SEG" 'http://127.0.0.1:8010/extract/frames?max_gap=0.1'
curl -s http://127.0.0.1:8010/health | python3 -c "import json,sys; print(json.load(sys.stdin)['video_processing'])"
```
Expected: первый вызов печатает длительность из ffprobe (8 или 16), разрешение записи, число кадров ≤ 50 и список кадров с причинами (запрос со старыми параметрами клиента работает); второй печатает `422`; третий `True`. В логе uvicorn одна строка `Frame selection: ...`. Остановить uvicorn.

- [ ] **Step 5: Зафиксировать числа**

Записать результаты шагов 2 и 3 (события, кадры на сегмент, время) в сообщение для описания PR; в код ничего не коммитить. Скрипт `run_impl.py` остаётся рядом с корпусом.

---

### Task 9: Подготовка ветки к PR

**Files:**
- Delete from git: `docs/superpowers/specs/2026-09-11-motion-frame-selection-design.md`, `docs/superpowers/plans/2026-09-12-motion-frame-selection.md`

**Interfaces:** нет.

Правило пользователя: плановые документы не должны попасть в диф PR, они остаются в истории ветки.

- [ ] **Step 1: Убедиться, что всё зелёное и рабочее дерево чистое**

Run: `.venv/bin/python -m pytest tests/ -q && git status --short`
Expected: все тесты PASS; в статусе только неотслеживаемые файлы (`.mcp.json`, веса, `docs/research/`, старые промпты).

- [ ] **Step 2: Удалить документы из ветки**

```bash
git rm docs/superpowers/specs/2026-09-11-motion-frame-selection-design.md docs/superpowers/plans/2026-09-12-motion-frame-selection.md
git commit -m "chore: remove design and plan documents before PR

Claude-Session: https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4"
git ls-files docs/
```
Expected: `git ls-files docs/` пуст.

- [ ] **Step 3: Передать ветку дальше**

Дальше по superpowers:finishing-a-development-branch: PR в `master` с описанием, включающим числа из Task 8 и ссылку `https://claude.ai/code/session_01Mf2QfwHB9Ygn9mU1WXxgh4` в конце. Отдельным шагом после выката сервера обновляется frigate-analyzer (`DETECT_MAX_FRAMES=6`, удаление `scene_threshold`).
