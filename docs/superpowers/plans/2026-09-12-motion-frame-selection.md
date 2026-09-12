# Motion-Based Frame Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить отбор кадров в `/extract/frames` и `/detect/video` на алгоритм по движению: кадр 0, сетка раз в `max_gap`, пики метрики `blob`, кап `max_frames`.

**Architecture:** Чистый модуль `app/frame_selection.py` считает метрику по паре серых кадров и применяет правило отбора к массивам pts и метрики. `app/video_utils.py` переписывается: ffprobe JSON, первый проход ffmpeg потоком через pipe (серые кадры 640 px, pts из `showinfo`), второй проход через `select` только для выбранных кадров. Эндпоинты в `app/main.py` получают новые параметры, старый select-путь и `scene_threshold` удаляются.

**Tech Stack:** Python 3.14 (`.venv`), FastAPI, numpy, OpenCV (`cv2`), ffmpeg/ffprobe через `subprocess`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-motion-frame-selection-design.md`

**Status (2026-09-12):** Tasks 1-8 complete. The final whole-branch review is done; its single fix wave is commit `0059599` (bounded pass-2 kill drain, ValueError on the degenerate aspect ratio, cancel-before-reap, showinfo tail classifier, tmp_path binding, MaxGapQuery description, pass-2 wording, documented thinning cliff, BGR channel-order test). Two commits landed after it: `647e573` corrects the grid-only threshold in the docs (20 s, not 24 s — the two thresholds are distinct), and `17ca740` fixes the cliff itself under ruling R15, thinning the grid to `max_frames - 1` so one slot always stays free for the strongest motion peak. 448 tests pass; the corpus numbers at `max_frames=6` are unchanged. Remaining: Task 9 and the PR.

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

✅ Done — see commit(s): `f267991`

---

### Task 2: Метрика `blob_area`

✅ Done — see commit(s): `c2c2533`

---

### Task 3: ffprobe JSON и разбор `showinfo` в `video_utils.py`

✅ Done — see commit(s): `473d02b`

---

### Task 4: Первый проход — `VideoFrameExtractor.scan`

✅ Done — see commit(s): `4701d2b` (plus controller ruling R5: shared `median_blob()` helper in `app/frame_selection.py`)

---

### Task 5: Второй проход, `extract_frames`, async-обёртка, удаление старого пути

✅ Done — see commit(s): `9b5acad` (plus rulings R6/R7: `scaled_h` guard, stderr closed after the collector join)

---

### Task 6: Модели ответа и эндпоинты

✅ Done — see commit(s): `dbfba40`

---

### Task 7: Документация

✅ Done — see commit(s): `0e21f90`, `69c6637` (plus ruling R2: README.md updated too)

---

### Task 8: Проверка на корпусе, замер времени, smoke

✅ Done — verification only, no repository commit. Corpus (207 segments): 15/17 moving events, 4.71 frames/segment, 4.66 on empty segments, zero frame-count mismatches vs metrics.py; `extract_frames` on a 16 s 2880×1620 segment 1.46 s (pass1 0.87 s, pass2 0.55 s); HTTP smoke green. Script kept at `~/vision-api-research/frame-selection/run_impl.py`, report in the SDD workspace.

---

### Task 9: Подготовка ветки к PR

**Files:**
- Delete from git: `docs/superpowers/specs/2026-09-11-motion-frame-selection-design.md`, `docs/superpowers/plans/2026-09-12-motion-frame-selection.md`

**Interfaces:** нет.

Правило пользователя: плановые документы не должны попасть в диф PR, они остаются в истории ветки.

- [ ] **Step 1: Убедиться, что всё зелёное и рабочее дерево чистое**

Run: `LOG_LEVEL=INFO .venv/bin/python -m pytest tests/ -q && git status --short`
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
