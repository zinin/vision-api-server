# Review Iteration 3 — 2026-02-14 18:00

## Источник

- Design: `docs/plans/2026-02-14-video-annotation-design.md`
- Plan: `docs/plans/2026-02-14-video-annotation.md`
- Review agents: codex-executor (gpt-5.3-codex), gemini-executor, ccs-executor (glmt/GLM-4.7)
- Merged output: `docs/plans/2026-02-14-video-annotation-review-merged-iter-3.md`

## Замечания

### [C-3-1] model_manager._executor не существует

> Plan Task 6 использует `executor = model_manager._executor`, но ModelManager не имеет атрибута `_executor`. Вызовет AttributeError при попытке аннотации. Job застрянет в статусе processing.

**Источник:** codex-executor, gemini-executor, ccs-executor (все 3)
**Статус:** Новое
**Ответ:** Использовать `get_executor(settings.max_executor_workers).executor` из `inference_utils.py`
**Действие:** Заменено в плане Task 6. Добавлен `from inference_utils import get_executor` в imports.

---

### [C-3-2] Input file leak при ошибке загрузки модели

> В worker при model load failure делается `continue`, но cleanup input.mp4 расположен после основного try-блока. Файл остаётся на диске до restart+sweep.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Обернуть тело worker цикла в per-job try/finally
**Действие:** Перестроен worker: добавлен внутренний try/finally, где finally всегда чистит input_path. Теперь cleanup выполняется при любом исходе (model error, annotation error, success).

---

### [I-3-1] shutil.move() вне защищённого try-блока

> `create_job()` вызывается до `shutil.move()`, причём move вынесен за пределы try-блока. Если move падает — битый job в очереди + tmp файл на диске.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Перенести shutil.move внутрь try-блока
**Действие:** `_shutil.move()` перемещён внутрь try-блока сразу после `create_job()`. При ошибке move — tmp файл удаляется в except.

---

### [I-3-2] DEFAULT_DETECT_EVERY=0 в конфиге вызывает ZeroDivisionError

> Config не валидирует default_detect_every. Значение 0 вызовет `frame_num % 0` в цикле кадров VideoAnnotator. Endpoint валидирует Query(ge=1) для user input, но default из config не проверяется.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Добавить `Field(ge=1)` в config.py
**Действие:** `default_detect_every: int = Field(default=5, ge=1)` в плане Task 2. Добавлен import `Field` из pydantic.

---

### [S-3-3] FFmpeg stderr[:500] — обрезает начало

> В _merge_audio логируется `result.stderr[:500]`. Самая полезная информация об ошибках обычно в конце stderr.

**Источник:** gemini-executor
**Статус:** Новое
**Ответ:** Не принято (минорное, не стоит менять для MVP)
**Действие:** Нет изменений

---

### [S-3-4] pip uninstall opencv-python-headless для локальной среды

> При замене opencv-python-headless на opencv-contrib-python-headless в requirements.txt, pip install не удалит старый пакет автоматически. В Docker это не проблема (чистая сборка), но в локальной среде может вызвать конфликт.

**Источник:** gemini-executor
**Статус:** Новое
**Ответ:** Не принято (Docker собирается чисто, локальная среда — ответственность разработчика)
**Действие:** Нет изменений

---

### [S-3-5] Model пре-валидация в endpoint до upload

> Валидировать параметр model синхронно в endpoint до тяжёлой загрузки файла.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Не принято
**Действие:** Нет изменений

---

### [I-3-3] Concurrent uploads bypass queue limit (disk DoS)

> check_queue_capacity() до upload не резервирует слот. Параллельные клиенты могут одновременно загружать большие файлы, даже если очередь полна.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Отложить — create_job() делает повторную проверку. Реальная проблема — disk space, что уже в deferred.
**Действие:** Нет изменений

---

### [I-3-4] VFR/rotated video metadata mismatch

> ffprobe dimensions vs actual frame dimensions для rotated/VFR видео.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Отложить — низкий риск для MVP
**Действие:** Нет изменений

---

### [S-3-6] Extension-only input validation

> Валидация только по расширению файла. Не-видео payload попадёт в pipeline.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Отложить — worker fail gracefully
**Действие:** Нет изменений

---

### [S-3-7] Test gaps for endpoint/worker failure scenarios

> Автотесты не покрывают endpoint/worker failure paths.

**Источник:** codex-executor
**Статус:** Новое (расширение S1 из iter-1)
**Ответ:** Отложить — unit тесты для VideoAnnotator уже deferred (S1 iter-1)
**Действие:** Нет изменений

---

### [INFO-1] Empty detections reset trackers immediately

> При пустых детекциях на detection frame все трекеры очищаются. Корректное поведение (YOLO = ground truth).

**Источник:** gemini-executor
**Статус:** Новое (информационное)
**Ответ:** Принято к сведению, документировать не нужно
**Действие:** Нет изменений

---

### [INFO-2] S-TRACKER-COLOR подтверждён как решённый

> Кортеж `(tracker, det)` в _init_trackers сохраняет class_id → цвет остаётся постоянным.

**Источник:** gemini-executor
**Статус:** Подтверждение решения из iter-2
**Ответ:** S-TRACKER-COLOR TODO можно снять
**Действие:** Нет изменений (TODO будет удалён при имплементации)

---

### [REPEAT-FP-1] Race condition в create_job

> Race condition между check и put_nowait.

**Источник:** ccs-executor (I-5)
**Статус:** Повтор (iter-2, FP-1)
**Ответ:** Ложное срабатывание — asyncio однопоточный, между check и put нет await

---

### [REPEAT-FP-5] startup_sweep удаляет dirs PROCESSING jobs

> startup_sweep может удалить директории обрабатываемых jobs.

**Источник:** ccs-executor (I-6)
**Статус:** Повтор (iter-2, FP-5)
**Ответ:** Ложное срабатывание — sweep вызывается при старте ДО запуска worker

---

### [REPEAT-C4] worker_task не в app.state

> worker_task не сохраняется в app.state.

**Источник:** ccs-executor (I-7)
**Статус:** Повтор (iter-1, C4)
**Ответ:** Уже исправлено в iter-1

---

### [REPEAT-S-LOGGING] Нет логирования параметров

> Нет логирования входящих параметров в worker.

**Источник:** ccs-executor (S-1)
**Статус:** Повтор (iter-2, S-LOGGING)
**Ответ:** Уже добавлено в iter-2

---

### [DEFERRED-1] detect_every > total_frames validation

**Источник:** ccs-executor (I-1)
**Статус:** Повтор (ранее отложено)

---

### [DEFERRED-2] Monitoring metrics (queued_at)

**Источник:** ccs-executor (S-2)
**Статус:** Повтор (ранее отложено — monitoring metrics)

---

### [DEFERRED-3] Disk limit strategy

**Источник:** ccs-executor (Q-2)
**Статус:** Повтор (ранее отложено — disk space pre-check)

---

### [DEFERRED-4] Multi-GPU strategy

**Источник:** ccs-executor (Q-3)
**Статус:** Повтор (ранее отложено — out of scope)

---

### [FP-3-1] Annotated "typo" в main.py

> CCS утверждает опечатку `Annotated` → `Annotated`.

**Источник:** ccs-executor (C-3)
**Статус:** Ложное срабатывание
**Ответ:** Код корректен, `Annotated` используется правильно

---

### [FP-3-2] DetectionBox frozen dataclass — проблема

> CCS отмечает что DetectionBox frozen.

**Источник:** ccs-executor (C-2)
**Статус:** Ложное срабатывание
**Ответ:** Не проблема — план только создаёт новые экземпляры DetectionBox, не мутирует

---

### [FP-3-3] FileResponse import отсутствует

> CCS говорит что FileResponse не импортирован.

**Источник:** ccs-executor (C-4)
**Статус:** Ложное срабатывание
**Ответ:** Уже адресовано в плане Task 6 Step 2

---

### [FP-3-4] Приватные методы не переименованы

> CCS говорит Step 0 не содержит детального кода.

**Источник:** ccs-executor (C-5)
**Статус:** Ложное срабатывание
**Ответ:** Адресовано в плане Task 5 Step 0, инструкция достаточно ясна

---

### [FP-3-5] DetectEveryQuery / ClassesQuery не определены

> CCS говорит что type aliases не определены.

**Источник:** ccs-executor (C-6)
**Статус:** Ложное срабатывание
**Ответ:** Адресовано в плане Task 6 Step 5 (определены прямо в коде endpoints)

---

### [FP-3-6] FileResponse без проверки существования файла

> CCS утверждает что нет проверки exists.

**Источник:** ccs-executor (I-3)
**Статус:** Ложное срабатывание
**Ответ:** Проверка уже есть в плане (`if job.output_path is None or not job.output_path.exists()`)

---

### [FP-3-7] get_job_manager dependency не существует

> CCS утверждает что get_job_manager не существует.

**Источник:** ccs-executor (I-4)
**Статус:** Ложное срабатывание
**Ответ:** Адресовано в плане Task 6 Step 1

---

### [ANSWERED-1] Corrupted video handling

**Источник:** ccs-executor (Q-1)
**Статус:** Повтор (iter-2, Q-ANSWERED)
**Ответ:** ffprobe error → cv2 fallback → cap.isOpened() check → job failed

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| `docs/plans/2026-02-14-video-annotation.md` | C-3-1: `model_manager._executor` → `get_executor().executor`, добавлен import. C-3-2: worker перестроен с per-job try/finally. I-3-1: shutil.move перенесён внутрь try. I-3-2: `Field(ge=1)` для default_detect_every, добавлен import Field. |

## Статистика

- Всего замечаний: 27
- Новых: 11 (4 приняты, 7 отложены/не приняты)
- Повторов (автоответ): 8
- Ложных срабатываний: 7
- Информационных: 2 (подтверждения предыдущих решений)
- Пользователь сказал "стоп": Нет
- Агенты: codex-executor (gpt-5.3-codex), gemini-executor, ccs-executor (glmt/GLM-4.7)
