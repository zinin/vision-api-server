# Review Iteration 2 — 2026-02-14 17:45

## Источник

- Design: `docs/plans/2026-02-14-video-annotation-design.md`
- Plan: `docs/plans/2026-02-14-video-annotation.md`
- Review agents: codex-executor (gpt-5.3-codex), gemini-executor, ccs-executor (glmt/GLM-4.7)
- Merged output: `docs/plans/2026-02-14-video-annotation-review-merged-iter-2.md`

## Замечания

### [C-NEW-1] writer.release() только в happy path

> VideoAnnotator.annotate() вызывает writer.release() только при успешном завершении. При exception в цикле кадров VideoWriter не закрывается — resource leak + повреждённые файлы.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Исправить в плане
**Действие:** `writer` инициализируется как `None`, release вызывается в `finally`. После успешного release в happy path `writer = None` чтобы не вызывать дважды.

---

### [C-NEW-2] Input cleanup в том же try-блоке worker

> После mark_completed, unlink(input.mp4) идёт в том же try. Если unlink падает → job перезаписывается в failed, хотя output.mp4 уже создан.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Вынести cleanup из try
**Действие:** Input cleanup вынесен в отдельный try/except с logger.warning. Не влияет на статус job.

---

### [C-NEW-3] Неполная очистка temp upload файлов

> При I/O ошибке записи/перемещения upload файла, upload_*.tmp остаётся на диске. startup_sweep чистит только директории.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Добавить finally для tmp файла + startup_sweep чистит .tmp файлы
**Действие:** Endpoint обёрнут в try/except/finally с cleanup tmp файла. startup_sweep расширен — удаляет и .tmp файлы. Тест обновлён.

---

### [I-NEW-1] startup_sweep safety при некорректном VIDEO_JOBS_DIR

> startup_sweep удаляет ВСЕ директории. Если VIDEO_JOBS_DIR=/tmp, удалит чужие данные.

**Источник:** codex-executor, gemini-executor, ccs-executor
**Статус:** Новое
**Ответ:** Оставить как есть — VIDEO_JOBS_DIR=/tmp/vision_jobs — специфичная директория, риск минимален.
**Действие:** Нет изменений

---

### [I-NEW-2] Queue limit проверяется после полной загрузки файла

> Клиент может залить 500MB и получить 429.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Проверка до upload
**Действие:** Добавлен `check_queue_capacity()` в JobManager + вызов в endpoint до начала upload. create_job также вызывает check (double check after upload, т.к. между upload и create может пройти время).

---

### [I-NEW-3] ffprobe metadata без валидации

> width/height/fps могут быть 0 при returncode==0. Fallback на cv2 не сработает.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Добавить валидацию
**Действие:** После парсинга ffprobe проверяется width>0, height>0, fps>0. При невалидных значениях — fallback на cv2.

---

### [I-NEW-4] Синхронный f.write() в async endpoint

> Блокирует event loop при записи больших файлов.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Использовать aiofiles
**Действие:** Заменено на `aiofiles.open()` + `await f.write()`. aiofiles добавлен в requirements.txt (Task 1).

---

### [I-NEW-5] OpenCV package conflict в Docker

> opencv-python-headless и opencv-contrib-python-headless конфликтуют при одновременной установке.

**Источник:** gemini-executor
**Статус:** Новое
**Ответ:** Не нужно — Docker собирается чисто.
**Действие:** Нет изменений

---

### [I-NEW-6] Naive datetime без timezone

> datetime.now() без timezone в Job dataclass. Клиент получает ISO без TZ.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Исправить — использовать UTC
**Действие:** `datetime.now()` → `datetime.now(tz=timezone.utc)` во всех местах. Import `timezone` добавлен.

---

### [I-NEW-7] cv2.VideoCapture resource leak в fallback

> В _get_video_metadata() fallback cv2.VideoCapture не закрывается при exception.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Исправить — обернуть в try/finally
**Действие:** cv2 fallback обёрнут в try/finally с cap.release().

---

### [I-NEW-8] Tracker degenerate bbox

> tracker.update() может вернуть success=True но w=0/h=0. Код не проверяет.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Добавить проверку w>0, h>0
**Действие:** Добавлена проверка `if w <= 0 or h <= 0: continue` в _update_trackers().

---

### [I-NEW-9] run_in_executor(None) — дефолтный executor

> Используется дефолтный ThreadPoolExecutor, не контролируемый конфигом.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Использовать executor сервиса
**Действие:** `run_in_executor(None, ...)` → `run_in_executor(executor, ...)` где `executor = model_manager._executor`.

---

### [I-NEW-10] mark_processing() без проверки статуса

> mark_processing() не проверяет что job в статусе QUEUED.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Добавить warning
**Действие:** Добавлен `logger.warning` при неожиданном текущем статусе.

---

### [REPEAT-I2] MAX_CONCURRENT_JOBS в Task 7 docs

> В документацию снова попал MAX_CONCURRENT_JOBS.

**Источник:** codex-executor
**Статус:** Повтор (iter-1, I2)
**Ответ:** Удалить (принятое решение iter-1)
**Действие:** Убран из Task 7 конфигурации.

---

### [REPEAT-Q1Q2] audio_missing flag

> Нет предупреждения клиенту что аудио отсутствует при ошибке FFmpeg.

**Источник:** ccs-executor
**Статус:** Повтор (iter-1, Q1-Q2)
**Ответ:** Аудио — best effort, нет аудио ≠ ошибка (принятое решение iter-1)
**Действие:** Нет изменений

---

### [REPEAT-I8] Worker cancellation / zombie processes

> Worker cancellation может оставить zombie processes или partial files.

**Источник:** ccs-executor
**Статус:** Повтор (iter-1, I8)
**Ответ:** Принятый риск — startup sweep покрывает при следующем старте
**Действие:** Нет изменений

---

### [FP-1] Race condition в create_job()

> Между проверкой очереди и put_nowait другой запрос может пройти.

**Источник:** ccs-executor
**Статус:** Ложное срабатывание
**Ответ:** asyncio однопоточный. Между check и put_nowait нет await → нет race condition.
**Действие:** Нет изменений

---

### [FP-2] asyncio.Queue unbounded (backpressure)

> asyncio.Queue() по умолчанию бесконечна, сервер продолжит принимать видео.

**Источник:** gemini-executor
**Статус:** Ложное срабатывание
**Ответ:** create_job() уже проверяет `queued_count >= max_queued` с RuntimeError. Очередь контролируется.
**Действие:** Нет изменений

---

### [FP-3] detect_every 0/negative

> detect_every=0 → деление на ноль.

**Источник:** gemini-executor
**Статус:** Ложное срабатывание
**Ответ:** Endpoint уже имеет `Query(ge=1, le=300)` → Pydantic валидирует.
**Действие:** Нет изменений

---

### [FP-4] FFmpeg stderr не логируется

> subprocess.run может упасть молча, stderr не логируется.

**Источник:** gemini-executor
**Статус:** Ложное срабатывание (Gemini не читал файлы — 0 tool calls)
**Ответ:** Код уже делает `capture_output=True` и логирует `result.stderr[:500]`.
**Действие:** Нет изменений

---

### [FP-5] startup_sweep удаляет при PROCESSING

> startup_sweep удаляет файлы рабочего job.

**Источник:** ccs-executor
**Статус:** Ложное срабатывание
**Ответ:** sweep вызывается при startup ДО запуска worker. Нет processing jobs в этот момент.
**Действие:** Нет изменений

---

### [S-LOGGING] Логирование параметров в annotate()

> Нет логирования параметров при старте обработки.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Добавить
**Действие:** Добавлен logger.info в начале annotate() с параметрами (resolution, fps, total_frames, detect_every, conf).

---

### [S-FFMPEG-CHECK] FFmpeg startup check

> ffmpeg/ffprobe может отсутствовать. Лучше узнать при старте.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Добавить
**Действие:** Добавлена проверка `shutil.which("ffmpeg")` и `shutil.which("ffprobe")` в lifespan с logger.warning.

---

### [S-SLOTS] @dataclass(slots=True)

> Оптимизация памяти для dataclass.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Добавить
**Действие:** `@dataclass(slots=True)` для Job, AnnotationParams, AnnotationStats.

---

### [S-TRACKER-COLOR] Tracker color consistency

> Цвета bbox могут меняться каждые N кадров при пересоздании trackers.

**Источник:** gemini-executor
**Статус:** Новое
**Ответ:** Проверить при реализации Task 5 — как draw_detection выбирает цвет
**Действие:** TODO: verify at implementation time

---

### [S-DEFERRED] Отложенные suggestions

> Smoke-тест CV2/FFmpeg, Cancel endpoint, detect_every>total_frames validation, NamedTemporaryFile, disk space check, metrics, YOLO verbose, VideoWriter fourcc fallback, existing endpoints memory fix.

**Источник:** gemini-executor, ccs-executor, codex-executor
**Статус:** Новое
**Ответ:** Отложить — за рамками MVP
**Действие:** Нет изменений

---

### [Q-ANSWERED] Вопросы с ответами

> classes str vs list, corrupted video handling, CSRT tracker memory.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** classes=str (совместимость с API), break на ret=False (частичный результат лучше ошибки), max_det=100 + ~50MB RAM приемлемо.
**Действие:** Нет изменений

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| `docs/plans/2026-02-14-video-annotation.md` | Task 1: +aiofiles. Task 4: timezone.utc, slots=True, mark_processing warning, check_queue_capacity(), startup_sweep .tmp cleanup, тесты обновлены. Task 5: writer try/finally, ffprobe validation, cv2 try/finally, tracker bbox check, logging. Task 6: input cleanup отдельный try, aiofiles upload, queue pre-check, executor, FFmpeg startup check. Task 7: убран MAX_CONCURRENT_JOBS. |
| `docs/plans/2026-02-14-video-annotation-design.md` | Pipeline: early queue reject, aiofiles, metadata validation, writer finally, cleanup isolation. JobManager: .tmp cleanup, check_queue_capacity. |

## Статистика

- Всего замечаний: 27
- Новых: 19
- Повторов (автоответ): 3
- Ложных срабатываний: 5
- Пользователь сказал "стоп": Нет
- Агенты: codex-executor (gpt-5.3-codex), gemini-executor, ccs-executor (glmt/GLM-4.7)
