# Review Iteration 1 — 2026-02-14 16:30

## Источник

- Design: `docs/plans/2026-02-14-video-annotation-design.md`
- Plan: `docs/plans/2026-02-14-video-annotation.md`
- Review agents: codex-executor (gpt-5.3-codex), gemini-executor, ccs-executor (glmt/GLM-4.7)
- Merged output: `docs/plans/2026-02-14-video-annotation-review-merged-iter-1.md`

## Замечания

### [C1] mp4v кодек несовместим с браузерами

> cv2.VideoWriter пишет mp4v, FFmpeg копирует as-is с `-c:v copy`. Большинство браузеров не воспроизводят mp4v.

**Источник:** gemini-executor
**Статус:** Новое
**Ответ:** Оставить mp4v. Файл предназначен для скачивания, VLC его откроет. Браузерная совместимость не нужна.
**Действие:** Нет изменений

---

### [C2] Enum сравнивается со строкой

> `job.status == "completed"` всегда False, т.к. job.status — это JobStatus.COMPLETED (Enum). download_url никогда не выставится.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Исправить в плане
**Действие:** Заменено на `job.status == JobStatus.COMPLETED` и `job.status != JobStatus.COMPLETED`. Добавлен import `JobStatus` в main.py.

---

### [C3] Job создаётся до валидации файла

> Job создаётся до проверки размера файла. При 413 или I/O-ошибке "битый" job остаётся в очереди.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Исправить порядок — streaming upload, затем валидация, затем create_job
**Действие:** Переписан submit-путь: streaming upload чанками (1MB), проверка размера на лету, create_job только после успешной записи на диск.

---

### [C4] Race condition: worker_task scope

> `'worker_task' in dir()` не работает после yield в lifespan — локальная переменная вне scope.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Исправить в плане
**Действие:** Заменено на `app.state.worker_task`. Cleanup использует `hasattr(app.state, "worker_task")`.

---

### [C5] video_only.mp4 не удаляется при ошибке FFmpeg

> При timeout или ошибке FFmpeg, video_only.mp4 не удаляется. Также не ловится FileNotFoundError если ffmpeg отсутствует.

**Источник:** codex-executor, ccs-executor
**Статус:** Новое
**Ответ:** Исправить в плане
**Действие:** Добавлен `FileNotFoundError` в except clause. video_only.mp4 удаляется после основного пути (код уже есть).

---

### [I1] `await file.read()` загружает до 500MB в RAM

> Противоречит заявлению о ~50MB RAM на job.

**Источник:** codex-executor, ccs-executor
**Статус:** Новое
**Ответ:** Исправить — streaming upload чанками на диск
**Действие:** Объединено с C3. Streaming upload через 1MB чанки.

---

### [I2] MAX_CONCURRENT_JOBS не используется

> Конфиг документирован, но всегда стартует ровно один воркер.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Удалить параметр
**Действие:** Удалён из config, тестов, дизайна.

---

### [I3] TTL cleanup не чистит сиротские директории после рестарта

> После рестарта старые директории станут сиротами.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Добавить startup sweep
**Действие:** Добавлен метод `startup_sweep()` в JobManager + тест + вызов в lifespan.

---

### [I4] Multi-process deployment несовместимость

> In-memory JobManager не работает с multiple workers/pods.

**Источник:** codex-executor, ccs-executor
**Статус:** Новое
**Ответ:** Документировать ограничение (workers=1)
**Действие:** Добавлена секция "Deployment constraint" в дизайн.

---

### [I5] ModelManager eviction во время обработки

> Модель может быть выгружена по TTL во время долгой обработки.

**Источник:** gemini-executor
**Статус:** Новое
**Ответ:** Принять риск. Worker держит ссылку, Python GC не удалит модель. Preloaded модели не evict-ятся.
**Действие:** Нет изменений

---

### [I6] Использование приватных методов visualization.py

> VideoAnnotator вызывает `_draw_detection` и `_calculate_adaptive_font_scale` напрямую.

**Источник:** codex-executor, ccs-executor
**Статус:** Новое
**Ответ:** Сделать методы публичными
**Действие:** Добавлен Step 0 в Task 5 — переименовать методы в visualization.py. Обновлены вызовы в video_annotator.py.

---

### [I7] DEFAULT_DETECT_EVERY из конфига игнорируется

> В endpoint жёстко задано `detect_every=5`, конфиг не используется.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Исправить — использовать settings.default_detect_every
**Действие:** Default изменён на None, резолвится из settings в теле endpoint.

---

### [I8] Нет graceful cleanup при shutdown

> При shutdown во время обработки — файлы не удалятся, статус останется processing.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Принять риск — startup sweep покроет при следующем старте.
**Действие:** Нет изменений

---

### [S1] Нет автотестов для VideoAnnotator

> Plan говорит "No unit tests" для Task 5.

**Источник:** codex-executor, gemini-executor
**Статус:** Новое
**Ответ:** Отложить, поставить TODO в коде
**Действие:** Добавлена пометка "TODO: add unit tests with mocked cv2/subprocess later" в Task 5.

---

### [S2] Нет лимита на длительность видео

> 500MB не защищает от "бомб сжатия".

**Источник:** gemini-executor, ccs-executor
**Статус:** Новое
**Ответ:** Отложить
**Действие:** Нет изменений

---

### [S3] StreamingResponse для download

> `StreamingResponse(open(...))` не закрывает fd корректно. FileResponse лучше.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Исправить — использовать FileResponse
**Действие:** Заменено на `FileResponse` в плане и дизайне.

---

### [S5] Queue position/ETA в response

> При MAX_CONCURRENT_JOBS=1 пользователь может долго ждать.

**Источник:** gemini-executor, ccs-executor
**Статус:** Новое
**Ответ:** Отложить — за рамками MVP
**Действие:** Нет изменений

---

### [S6] YOLO.track вместо CSRT

> Нативный tracking Ultralytics может быть проще.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Исследовано. YOLO.track запускает детекцию на каждом кадре (нет detect_every). Решено оставить CSRT.
**Действие:** Нет изменений (CSRT остаётся)

---

### [S7] Worker health endpoint

> GET /jobs/worker/status для мониторинга.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Отложить — за рамками MVP
**Действие:** Нет изменений

---

### [Q1-Q2] Jobs survive restart? Audio policy?

> Вопросы о персистентности и аудио.

**Источник:** codex-executor, ccs-executor
**Статус:** Новое
**Ответ:** Нет + best effort. Jobs теряются при рестарте (ok). Аудио — best effort, нет аудио не ошибка.
**Действие:** Документировано в дизайне.

---

### [Q4] Валидация classes

> Валидировать classes против model.names при submit?

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Нет — невалидные классы просто отфильтруют пустые результаты.
**Действие:** Нет изменений

---

### [Q5] 410 Gone для expired jobs

> Возвращать 410 вместо 404?

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Нет, 404 достаточно.
**Действие:** Нет изменений

---

### [Q6] GPU encoding

> CPU encoding может быть узким местом.

**Источник:** gemini-executor
**Статус:** Новое
**Ответ:** Отложить — CPU достаточно для MVP.
**Действие:** Нет изменений

---

### [Misc] ffprobe vs cv2.CAP_PROP

> Дизайн упоминает ffprobe, код использует cv2. cv2 ненадёжен на VFR видео.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Использовать ffprobe с fallback на cv2
**Действие:** Добавлен метод `_get_video_metadata()` в VideoAnnotator — ffprobe primary, cv2 fallback.

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| `docs/plans/2026-02-14-video-annotation-design.md` | Убран MAX_CONCURRENT_JOBS, добавлен startup sweep, deployment constraint, audio policy, FileResponse, ffprobe, public methods |
| `docs/plans/2026-02-14-video-annotation.md` | C2: Enum fix, C3+I1: streaming upload, C4: app.state.worker_task, C5: FileNotFoundError, I2: убран max_concurrent_jobs, I3: startup_sweep + тест, I6: public methods, I7: settings.default_detect_every, S3: FileResponse, ffprobe metadata, uuid import |

## Статистика

- Всего замечаний: 24
- Новых: 24
- Повторов (автоответ): 0
- Пользователь сказал "стоп": Нет
- Агенты: codex-executor (gpt-5.3-codex), gemini-executor, ccs-executor (glmt/GLM-4.7)
