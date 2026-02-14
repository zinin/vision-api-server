# Merged Design Review — Iteration 1

## codex-executor (gpt-5.3-codex)

### Critical Issues
- `Enum` сравнивается со строкой, из-за чего логика статуса сломана: `download_url` не будет выставляться и `/jobs/{job_id}/download` будет ошибочно отклонять даже завершённые задачи (`docs/plans/2026-02-14-video-annotation.md:1145`, `docs/plans/2026-02-14-video-annotation.md:1172`, `docs/plans/2026-02-14-video-annotation.md:368`).
- Job создаётся до валидации размера и до записи файла; при `413` или I/O-ошибке останется "битая" задача в очереди (`docs/plans/2026-02-14-video-annotation.md:1091`, `docs/plans/2026-02-14-video-annotation.md:1109`, `docs/plans/2026-02-14-video-annotation.md:1116`).
- Заявление про постоянный RAM-профиль противоречит реализации: `await file.read()` загружает весь файл в память (до 500MB) (`docs/plans/2026-02-14-video-annotation-design.md:75`, `docs/plans/2026-02-14-video-annotation-design.md:115`, `docs/plans/2026-02-14-video-annotation.md:1108`).
- Конфиг `MAX_CONCURRENT_JOBS` документирован, но фактически не используется: всегда стартует ровно один воркер (`docs/plans/2026-02-14-video-annotation-design.md:122`, `docs/plans/2026-02-14-video-annotation.md:901`).
- TTL-очистка привязана к in-memory `dict`; после рестарта старые директории в `VIDEO_JOBS_DIR` станут сиротами и не будут очищены (`docs/plans/2026-02-14-video-annotation-design.md:94`, `docs/plans/2026-02-14-video-annotation-design.md:97`, `docs/plans/2026-02-14-video-annotation.md:462`).
- Fallback аудио-merge неполный: не ловится `FileNotFoundError` для `ffmpeg`, поэтому при отсутствии бинаря job упадёт вместо "video-only" (`docs/plans/2026-02-14-video-annotation.md:783`, `docs/plans/2026-02-14-video-annotation.md:805`, `app/main.py:94`).

### Concerns
- `DEFAULT_DETECT_EVERY` из конфига в API практически игнорируется из-за жёсткого дефолта `5` в параметре endpoint (`docs/plans/2026-02-14-video-annotation-design.md:124`, `docs/plans/2026-02-14-video-annotation.md:1051`).
- Нет автоматических тестов на ключевой pipeline (воркер, endpoint-ы, merge, отказоустойчивость); только manual test (`docs/plans/2026-02-14-video-annotation.md:530`, `docs/plans/2026-02-14-video-annotation.md:1243`).
- Потенциально плохая масштабируемость CSRT при большом числе объектов: `max_det` до 1000 + трекер на каждый bbox может быть CPU-бутылочным горлышком (`docs/plans/2026-02-14-video-annotation-design.md:19`, `docs/plans/2026-02-14-video-annotation.md:727`).
- `VideoAnnotator` зависит от приватных методов визуализатора (`_draw_detection`, `_calculate_adaptive_font_scale`), что хрупко для будущих рефакторингов (`docs/plans/2026-02-14-video-annotation.md:766`, `app/visualization.py:157`).
- Дизайн заявляет `ffprobe`-метаданные, но кодовый план реально полагается на `cv2.CAP_PROP_*`, которые часто неточны на VFR-видео (`docs/plans/2026-02-14-video-annotation-design.md:74`, `docs/plans/2026-02-14-video-annotation.md:620`).
- In-memory job manager плохо совместим с multi-process deployment (несколько uvicorn workers/pods): polling/download может попадать в другой процесс и давать 404 (`docs/plans/2026-02-14-video-annotation-design.md:93`).

### Suggestions
- Перестроить submit-путь: сначала потоковая запись upload на диск чанками, затем валидация, и только после успешной записи создавать job.
- Явно зафиксировать модель деплоя: либо hard requirement "1 process/1 pod", либо вынести queue/state в Redis/Postgres.
- Добавить startup sweep для `VIDEO_JOBS_DIR` (удаление сирот/просроченных папок), иначе диск будет деградировать.
- Для скачивания использовать `FileResponse` вместо `StreamingResponse(open(...))` для корректного lifecycle файлового дескриптора и HTTP range.
- Вынести публичный метод рисования одного bbox в `DetectionVisualizer`, не использовать приватные методы напрямую.
- Рассмотреть альтернативу CSRT: `YOLO.track` (ByteTrack/BoT-SORT) или гибридный режим (YOLO чаще при большом количестве объектов).
- Добавить минимальные интеграционные автотесты с коротким фикстурным видео и мокнутой моделью/ffmpeg.

### Questions
- Должны ли задачи/результаты переживать рестарт сервера, или при рестарте нужно гарантированно чистить все job-артефакты?
- Гарантирован ли прод-режим с одним процессом FastAPI (`workers=1`, один pod), или нужна корректная работа при горизонтальном масштабировании?
- Что является требованием по аудио: "best effort" или "обязательно сохранить аудио, иначе failed"?
- Какой целевой SLA по длине видео и числу объектов в кадре (чтобы обосновать CSRT vs более дешёвый/стабильный трекинг)?
- Нужно ли валидировать `classes` относительно `model.names` на этапе submit (и делать case-insensitive matching)?

---

## gemini-executor

### Критические проблемы (Critical Issues)

1. **Несовместимость видеокодека с браузерами**
   В `app/video_annotator.py` используется `cv2.VideoWriter` с кодеком `mp4v` (MPEG-4 Part 2), а на этапе объединения с аудио через FFmpeg используется флаг `-c:v copy`. Большинство современных браузеров не воспроизводят `mp4v` в MP4 контейнере. Требуется H.264 (avc1).
   **Решение:** Заменить `-c:v copy` на `-c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p`.

### Опасения (Concerns)

1. **Дублирование моделей в памяти (ModelManager Eviction)** — Фоновый воркер держит ссылку на модель. Если обработка длится дольше `YOLO_MODEL_TTL`, ModelManager может выгрузить модель и загрузить вторую копию, что может привести к OOM.

2. **Отсутствие лимита на длительность/количество кадров** — 500MB лимит не защищает от "бомб сжатия" (10 часов черного экрана). Один пользователь может занять воркер на часы.

### Предложения (Suggestions)

1. **Тестирование VideoAnnotator** — Добавить unit-тесты с мокнутыми cv2.VideoCapture, cv2.VideoWriter и subprocess.

2. **Queue position/ETA в API response** — Добавить позицию в очереди или ETA в JobCreatedResponse.

### Вопросы (Questions)

1. Предполагается ли использование GPU для кодирования видео? Текущая реализация полностью на CPU.

**Вердикт:** План детальный, архитектурно верный. Если исправить кодек (libx264), можно пускать в работу.

---

## ccs-executor (glmt / GLM-4.7)

### Критические проблемы

1. **Race condition при отмене worker task** — `'worker_task' in dir()` не будет работать после `yield` в lifespan. Нужно хранить в `app.state`.

2. **Ошибка в типе Query-параметров** — Возможная опечатка `Annotated` vs `Annotated` (проверка показала: в проекте используется правильный `Annotated`).

3. **Memory leak при неудачном FFmpeg merge** — `video_only.mp4` не удаляется при timeout или ошибке FFmpeg.

4. **Отсутствие проверки при multiprocess deployment** — Каждый процесс создаст свой JobManager с отдельным состоянием.

### Проблемы (Concerns)

5. **Input file загружается полностью в память** (до 500 MB).
6. **Tracker drift** не обрабатывается между YOLO-фреймами.
7. **Нет валидации `detect_every` относительно длины видео.**
8. **Отсутствие graceful cleanup при shutdown во время активной обработки.**
9. **Использование приватного метода `visualizer._draw_detection`.**

### Предложения

10. Более детальный progress reporting (`current_frame`, `estimated_remaining`).
11. Параметр `max_duration_seconds` для защиты от abuse.
12. Streaming response для коротких видео.
13. Worker health check endpoint.

### Вопросы

14. Возвращать 410 Gone вместо 404 для TTL-удалённых jobs?
15. Гарантирует ли opencv-contrib >= 4.12 наличие CSRT tracker?
16. Поведение при отсутствии audio track в видео?

**Оценка:** Дизайн и план хорошо продуманы, архитектура разумна для single-worker. Нужно address error handling, cleanup и edge cases.
