# Merged Design Review — Iteration 1 (job-cancellation)

Source:
- Design: `docs/superpowers/specs/2026-04-17-job-cancellation-design.md`
- Plan:   `docs/superpowers/plans/2026-04-17-job-cancellation.md`

## codex-executor (gpt-5.4 xhigh)

### [MAJOR-1] Отмена `QUEUED`-job сейчас оставляет `input.mp4` до TTL
> В предложенном потоке queued-cancelled job просто пропускается worker’ом, но cleanup входного файла привязан к `finally`, в который skip-ветка не входит.

**Почему это важно:** В [спеке](/opt/github/zinin/vision-api-server/docs/superpowers/specs/2026-04-17-job-cancellation-design.md:81) и [cleanup-секции](/opt/github/zinin/vision-api-server/docs/superpowers/specs/2026-04-17-job-cancellation-design.md:178) предполагается, что `input.mp4` удалит existing `finally`, но в реальном worker этот `finally` начинается только внутри внутреннего `try` ([app/main.py](/opt/github/zinin/vision-api-server/app/main.py:195), [app/main.py](/opt/github/zinin/vision-api-server/app/main.py:281)). Плановая skip-ветка делает `continue` раньше ([plan](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-04-17-job-cancellation.md:752)). Это даёт утечку больших uploaded-файлов до TTL.

**Рекомендация:** Для `QUEUED` удалять `input_path` сразу в `request_cancel()` или в skip-ветке worker перед `continue`. Отдельно добавить тест, что queued-cancel не оставляет `input.mp4`.

### [MAJOR-2] Заявленная latency для `PROCESSING` занижена уже до входа в `annotate()`
> `mark_processing()` ставится до `await model_manager.get_model(...)`, а `cancel_event` впервые читается только внутри decode-loop’ов `annotate()`.

**Почему это важно:** Спека обещает bound “one frame” и обычно `<1s` ([spec](/opt/github/zinin/vision-api-server/docs/superpowers/specs/2026-04-17-job-cancellation-design.md:183)), но worker переводит job в `PROCESSING` ещё до загрузки модели ([app/main.py](/opt/github/zinin/vision-api-server/app/main.py:192), [app/main.py](/opt/github/zinin/vision-api-server/app/main.py:200)). В `ModelManager` это может быть on-demand download/load с ретраями ([app/model_manager.py](/opt/github/zinin/vision-api-server/app/model_manager.py:89), [app/model_manager.py](/opt/github/zinin/vision-api-server/app/model_manager.py:206)), то есть далеко не “один кадр”.

**Рекомендация:** Либо переносить `mark_processing` ближе к фактическому старту `annotate()`, либо добавлять cancel-checkpoint сразу после `get_model()` и других pre-annotate шагов. Документацию по latency в любом случае нужно ослабить.

### [MAJOR-3] Переход в `CANCELLED` блокируется teardown FFmpeg, а не только checkpoint’ом в цикле
> `JobCancelledError` поднимается внутри `with FFmpegDecoder/FFmpegEncoder`, но worker увидит его только после завершения `__exit__`.

**Почему это важно:** `FFmpegDecoder.close()` ждёт до 10 секунд ([app/ffmpeg_pipe.py](/opt/github/zinin/vision-api-server/app/ffmpeg_pipe.py:81)), `FFmpegEncoder.close()` до 300 секунд ([app/ffmpeg_pipe.py](/opt/github/zinin/vision-api-server/app/ffmpeg_pipe.py:167)). Значит follow-up `GET /jobs/{id}` может долго видеть `processing`, хотя event уже set. Это противоречит [разделу Latency](/opt/github/zinin/vision-api-server/docs/superpowers/specs/2026-04-17-job-cancellation-design.md:185) и делает smoke-test “within ~1 second” в [плане](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-04-17-job-cancellation.md:1072) ненадёжным.

**Рекомендация:** Если hard-kill FFmpeg действительно non-goal, нужно явно документировать, что terminal transition включает teardown и может быть существенно дольше checkpoint latency. Если нужен быстрый `CANCELLED`, понадобится cancellation-aware abort path в FFmpeg wrappers.

### [MAJOR-4] После успешного `/cancel` job всё ещё может закончиться `FAILED`
> В дизайне описана только гонка с `COMPLETED`, но не гонка с ошибками до старта `annotate()`.

**Почему это важно:** Между `mark_processing()` и запуском `annotate()` worker может упасть на `get_model()`/инициализации и пойти в `mark_failed()` ([app/main.py](/opt/github/zinin/vision-api-server/app/main.py:192), [app/main.py](/opt/github/zinin/vision-api-server/app/main.py:202), [app/main.py](/opt/github/zinin/vision-api-server/app/main.py:274)). При этом `/cancel` уже вернул `200 OK` для `PROCESSING` ([spec](/opt/github/zinin/vision-api-server/docs/superpowers/specs/2026-04-17-job-cancellation-design.md:50)). Для клиента это очень неочевидный контракт.

**Рекомендация:** Явно определить precedence. Либо документировать “cancel best-effort, pre-annotate failure may win”, либо после принятой отмены проверять `cancel_event.is_set()` перед `mark_failed()` и давать отмене приоритет.

### [MINOR-1] Дизайн и план расходятся по безопасному удалению partial output
> В spec helper `_cleanup_partial_output` описан как `unlink(missing_ok=True)`, а в плане уже используется `try/except OSError`.

**Почему это важно:** `missing_ok=True` не покрывает `PermissionError`/I/O ошибки. Без `try/except` worker после `mark_cancelled()` может свалиться во внешний error-path с лишним логом и sleep. См. [spec](/opt/github/zinin/vision-api-server/docs/superpowers/specs/2026-04-17-job-cancellation-design.md:172) vs [plan](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-04-17-job-cancellation.md:824).

**Рекомендация:** Зафиксировать в дизайне именно плановый вариант: `try/except OSError` + warning, без изменения terminal status.

### [MINOR-2] В тест-плане не закрыты два самых рискованных сценария
> Сейчас нет теста, который бы проверял cleanup `input.mp4` для queued-cancel, и нет HTTP-level теста для `PROCESSING -> 200 / status="processing"`.

**Почему это важно:** Это как раз самые нестабильные места текущего дизайна: silent disk leak и неинтуитивная API-семантика. В [worker-тестах](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-04-17-job-cancellation.md:655) и [endpoint-тестах](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-04-17-job-cancellation.md:868) этих assert’ов нет.

**Рекомендация:** Добавить минимум два теста: queued-cancel удаляет `input_path`; `POST /jobs/{id}/cancel` для вручную выставленного `PROCESSING` возвращает `200`, `status="processing"` и выставляет `cancel_event`.

### [MINOR-3] OpenAPI/model docs останутся устаревшими относительно нового статуса
> План говорит “No other files change”, но `JobStatusResponse` по-прежнему описывает только `queued, processing, completed, failed`.

**Почему это важно:** Схема и generated docs будут расходиться с реальным API. См. [app/models.py](/opt/github/zinin/vision-api-server/app/models.py:178) и [plan](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-04-17-job-cancellation.md:963).

**Рекомендация:** Обновить description/example в `app/models.py` и добавить маленький model-level test для `status="cancelled"`.

### [NIT-1] В spec и plan не совпадает число тестов
> В spec заявлено “15 tests total”, но сам plan фактически добавляет 21 тест.

**Почему это важно:** Это не ломает реализацию, но затрудняет ревью и трекинг выполнения. См. [spec/testing](/opt/github/zinin/vision-api-server/docs/superpowers/specs/2026-04-17-job-cancellation-design.md:220) и [plan](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-04-17-job-cancellation.md:33).

**Рекомендация:** Либо пересчитать total в spec, либо убрать общее число и оставить только пофайловое перечисление.

---

## gemini-executor

Вот подробный анализ предоставленного дизайна и плана реализации (Plan & Design) в соответствии с вашими требованиями.

### [CRITICAL-1] Утечка ресурсов (`input.mp4`) при пропуске отмененной задачи
> **Контекст:** Раздел «Worker» (шаг 1) и Task 5 предписывают добавить проверку `if job.status != QUEUED: continue` сразу после `get_next_job_id()`. При этом в разделе «Cleanup Behaviour» указано, что `input.mp4` удаляется существующим блоком `finally`.
**Почему это важно:** Если задача была отменена, пока находилась в очереди, воркер извлечет её, увидит статус `CANCELLED` (или отличный от `QUEUED`) и вызовет `continue`. Это приведет к пропуску всего остального тела цикла, **включая блок `finally`**. В результате исходный файл `input.mp4` останется на диске и не будет удален вплоть до срабатывания глобального TTL (что может занять часы, либо вообще не очистить директорию, если TTL удаляет только определенные файлы). При активной отмене задач это приведет к быстрому переполнению диска.
**Рекомендация:** Проверку на статус необходимо поместить *внутрь* блока `try`, чтобы при выходе из него гарантированно сработал `finally`, либо явно удалять `input.mp4` перед вызовом `continue`. В Task 5 также следует обновить тесты: убедиться, что тест `skip_queued_but_cancelled` явно проверяет факт удаления `input.mp4`.

### [MAJOR-1] Гонка состояний между `mark_processing` и `request_cancel`
> **Контекст:** Раздел «Error Handling & Edge Cases» описывает гонку: *«Cancel between get_next_job_id and mark_processing -> request_cancel sees QUEUED, flips to CANCELLED. Worker's guard skips.»*
**Почему это важно:** Дизайн ошибочно предполагает, что проверка воркера (`if job.status != QUEUED`) защитит от этой гонки. Однако гонка может произойти ровно *после* того, как воркер прошел эту проверку (увидев `QUEUED`), но *до* вызова `mark_processing`.
В этом сценарии: 
1. Воркер проверяет статус -> `QUEUED`.
2. Клиент вызывает `/cancel` -> `request_cancel` меняет статус на `CANCELLED` и устанавливает `cancel_event`.
3. Воркер вызывает `mark_processing` -> статус перезаписывается на `PROCESSING`.
Хотя задача в итоге корректно прервется (так как `cancel_event` установлен), для клиента статус прыгнет: `QUEUED -> CANCELLED -> PROCESSING -> CANCELLED`, что может сломать клиентскую логику опроса.
**Рекомендация:** В Task 5 (или Task 2) необходимо модифицировать саму функцию `mark_processing`, чтобы она проверяла текущий статус перед обновлением. Например: переводить в `PROCESSING` только если статус все еще `QUEUED`.

### [MINOR-1] Нарушение инвариантов промежуточных коммитов при TDD
> **Контекст:** В Task 4 `VideoAnnotator` начинает выбрасывать исключение `JobCancelledError`. Логика перехвата этого исключения добавляется в воркер только на следующем шаге — в Task 5.
**Почему это важно:** Хотя это не повлияет на финальный результат PR, такой порядок нарушает принцип работоспособности каждого коммита. Если тесты или система будут запущены после завершения Task 4 (например, в CI), отмена активной задачи приведет к тому, что воркер поймает `JobCancelledError` в базовом блоке `except Exception:`, пометит задачу как `FAILED` (а не `CANCELLED`) и, возможно, не удалит `output.mp4`.
**Рекомендация:** Либо поменять местами Task 4 и Task 5, либо объединить добавление обработчика в воркере и генерацию исключения в аннотаторе в один логический шаг. Воркер должен "научиться" обрабатывать ошибку до (или одновременно с) тем, как она начнет генерироваться.

### [NIT-1] Состояние гонки на уровне файловой системы при удалении файлов
> **Контекст:** В Task 5 предложен код удаления артефактов: `if output_path.exists(): output_path.unlink()`.
**Почему это важно:** Этот классический паттерн («check-then-act») подвержен состоянию гонки (Time-of-check to time-of-use). Если файл будет удален сторонним процессом (или администратором) между `.exists()` и `.unlink()`, приложение упадет с `FileNotFoundError`.
**Рекомендация:** Использовать более безопасный и лаконичный метод из стандартной библиотеки Python 3.8+: `output_path.unlink(missing_ok=True)`. Это позволит избавиться от предварительной проверки `exists()`. То же самое касается очистки `input.mp4`.


---

## ccs-executor (glm)

Все файлы прочитаны. Формирую ревью.

---

### [MAJOR-1] Отсутствует проверка cancel между проходом 1 и проходом 2 в `annotate()`

Между завершением `_pass1_collect` и входом в `_pass2_render` выполняется стабилизация и подсчёт статистики. Если `cancel_event` был установлен в последних итерациях pass 1, код выполнит всю стабилизацию, затем **запустит два FFmpeg-сабпроцесса** (декодер + энкодер) в pass 2, и только потом обнаружит событие на первой итерации while-цикла.

**Почему это важно:** Ненужный запуск двух FFmpeg-процессов + аллокация ресурсов стабилизатора для задания, которое уже отменено. На 4K-видео создание энкодера может быть дорогостоящим (инициализация GPU-кодировщика).

**Рекомендация:** Добавить одну проверку `cancel_event` в `annotate()` перед вызовом `_pass2_render`:

```python
# After stabilization, before pass 2:
if cancel_event is not None and cancel_event.is_set():
    raise JobCancelledError()
self._pass2_render(...)
```

Это также сэкономит CPU на стабилизации, если добавить вторую проверку сразу после `_pass1_collect`.

---

### [MINOR-1] Описание `JobStatusResponse.status` не включает `"cancelled"`

В `app/models.py:181`:
```python
status: str = Field(description="Job status: queued, processing, completed, failed")
```

План (Task 7) обновляет `CLAUDE.md` и `api.md`, но не обновляет описание поля в Pydantic-модели. FastAPI генерирует OpenAPI-схему из этого описания.

**Рекомендация:** Обновить описание на `"Job status: queued, processing, completed, failed, cancelled"` в рамках Task 7.

---

### [MINOR-2] Расхождение в количестве тестов: дизайн заявляет 15, план содержит 21

Дизайн перечисляет 15 тестов (7 + 3 + 3 + 2). План добавляет 21: Task 1 — 3, Task 2 — 6, Task 3 — 2, Task 4 — 3, Task 5 — 2, Task 6 — 5. Дополнительные тесты — это улучшение (больше покрытия), но цифра в дизайне устарела.

**Рекомендация:** Обновить дизайн-документ: заменить «15 tests total» на «21 tests total» и добавить недостающие записи (3 теста Task 1, 1 тест Task 3, 2 дополнительных endpoint-теста).

---

### [MINOR-3] Worker test `test_skip_queued_but_cancelled` не верифицирует, что job был извлечён из очереди

В `tests/test_worker.py:657` тест проверяет `mock_annotator_cls.assert_not_called()` и `status == CANCELLED`. Но `request_cancel` уже установил `status = CANCELLED` до запуска воркера, а `_run_worker_until_job_done` видит `all_done=True` сразу (CANCELLED ∈ terminal). Если воркер не успеет обработать job за 50ms `asyncio.sleep`, тест всё равно пройдёт, но skip-guard не будет фактически проверен.

**Рекомендация:** Добавить проверку, что очередь пуста (воркер действительно извлёк job_id):
```python
assert worker_job_manager._queue.empty()
```

---

### [MINOR-4] Тест `test_cancel_during_processing_marks_cancelled` не проверяет передачу `cancel_event` в `annotate()`

В `tests/test_worker.py:699` mock-annotator всегда бросает `JobCancelledError`, независимо от аргументов. Тест верифицирует путь обработки ошибки, но не проверяет, что воркер реально передаёт `cancel_event=job.cancel_event`.

**Рекомендация:** Добавить assert на вызов:
```python
mock_annotator_cls.return_value.annotate.assert_called_once()
call_kwargs = mock_annotator_cls.return_value.annotate.call_args.kwargs
assert call_kwargs["cancel_event"] is job.cancel_event
```

---

### [NIT-1] Дизайн говорит «set cancel_event after 3 reads», план ставит после 2 predict-вызовов

Дизайн (Testing, пункт 8): «set cancel_event after 3 reads → JobCancelledError raised». План Task 4 `test_cancel_during_pass1_raises`: `fake_predict` устанавливает событие после 2-го вызова (`call_count["n"] >= 2`). С `detect_every=1` это 2 фрейма, а не 3.

**Рекомендация:** Обновить описание в дизайн-документе для согласованности.

---

### Что правильно

- **Threading модель:** `threading.Event` — корректный примитив для event-loop ↔ executor thread. Все записи `job.status` — только в event-loop thread. Гонок нет.
- **API семантика:** Матрица 200/404/409 разумна. PROCESSING → 200 с `"status": "processing"` — честный ответ, клиент опрашивает `GET /jobs/{id}`.
- **Resource cleanup:** FFmpeg-контекстные менеджеры корректно вызывают `__exit__` при `JobCancelledError`. `output_path.unlink()` защищён от `OSError`.
- **Latency:** Оценка «sub-second» реалистична. Между проверками — максимум один `model.predict()` или один decode+encode фрейма.
- **Backwards compatibility:** `cancel_event | None = None` не ломает существующих вызывающих. Тест `test_cancel_event_none_runs_to_completion` покрывает это.
- **План → дизайн консистентность:** План точно реализует дизайн (с дополнительными тестами). TDD red→green шаги корректны. Скрытых зависимостей между задачами нет.


---

## ccs-executor (albb-glm)

# Design Review: Job Cancellation for Vision API Server

Провожу анализ представленного дизайна и плана реализации.

### [CRITICAL-1] Race condition: QUEUED→CANCELLED vs mark_processing

> Между моментом когда worker получает Job-объект и моментом вызова `mark_processing`, `request_cancel` может установить `CANCELLED`, но worker уже прочитал `QUEUED` и proceed к `mark_processing`, который перезапишет `CANCELLED→PROCESSING`.

**Почему это важно:** Приводит к corrupted state — job с `status=PROCESSING` но `cancel_event.is_set()=True`. Annotator начнёт работу, но сразу прервётся на первом check, выбросит `JobCancelledError`, worker вызовет `mark_cancelled`, и job станет `CANCELLED`. Однако: (1) partial output уже мог быть создан, (2) race вызывает неопределённое поведение, (3) дублирование `completed_at` (cancel уже установил его, processing→cancelled установит снова).

Design говорит: *"Cancel between get_next_job_id and mark_processing → request_cancel sees QUEUED, flips to CANCELLED. Worker's guard skips."* Но это предполагает что worker's guard видит CANCELLED status. Если worker прочитал status **до** того как cancel записал, worker видит QUEUED и proceed.

**Рекомендация:** Добавить guard в `mark_processing`:

```python
def mark_processing(self, job_id: str) -> None:
    job = self._jobs[job_id]
    if job.status != JobStatus.QUEUED:
        return  # Already cancelled, skip
    job.status = JobStatus.PROCESSING
```

Альтернатива: compare-and-swap через lock, но выше проще.

---

### [CRITICAL-2] output_path.unlink() failure не обработан

> В `JobCancelledError` handler: `if output_path.exists(): output_path.unlink()` — если unlink() выбросит OSError (permission denied, file still open by FFmpeg, etc.), исключение propagate в generic `Exception` handler → job mark_failed вместо mark_cancelled.

**Почему это важно:** Job с `cancel_event.is_set()=True` становится `FAILED`. Partial output file не удалён — resource leak. Клиент видит `status=failed` вместо `cancelled`. TTL cleanup eventual удалит директорию, но until then file orphaned.

**Рекомендация:** Wrap unlink в try/except:

```python
except JobCancelledError:
    job_manager.mark_cancelled(job_id)
    try:
        if output_path.exists():
            output_path.unlink()
    except OSError:
        logger.warning("Failed to delete partial output", exc_info=True)
```

---

### [MAJOR-1] Cancellation latency bound не учитывает blocking read_frame()

> Event check находится **до** `decoder.read_frame()`. Если executor thread blocked внутри `read_frame()` waiting on FFmpeg pipe, cancellation не будет observed until read_frame returns.

**Почему это важно:** Design заявляет *"bounded by a single frame's processing time"* и *"Typical: well under 1 second."* Но:
- `read_frame()` блокирует на pipe read → может ждать десятки ms (frame time at 30fps) или дольше если FFmpeg slow
- Если cancellation request приходит когда thread blocked в read_frame(), latency = blocking time + inference time + check

Pathological case: 4K video, CPU decode, slow FFmpeg → read_frame could block 100+ ms. Add YOLO inference (several seconds). Total latency could exceed "few seconds" stated.

**Рекомендация:** Update "Cancellation Latency" section чтобы explicitly mention `read_frame()` blocking contribution. Или: clarify что bound = **one full frame cycle** (decode + inference + encode).

---

### [MINOR-1] Thread Safety section некорректен

> Design: *"Executor reads `cancel_event` only."* Но executor thread читает `job.status` в skip guard: `if job.status != QUEUED: continue`.

**Почему это важно:** Documentation inconsistency. Не bug — GIL гарантирует visibility для single reads. Но утверждение misleading.

**Рекомендация:** Update Thread Safety section: *"Executor reads `cancel_event` and reads `status` once in skip guard."*

---

### [MINOR-2] Test coverage gap для race conditions

> Нет test для race QUEUED→CANCELLED vs concurrent mark_processing. Unit tests sequential execution не может expose concurrent race.

**Почему это важно:** CRITICAL-1 race не тестируем. Integration test или explicit concurrency test нужен для confidence.

**Рекомендация:** Add note в plan что race handling verified via code review + `mark_processing` guard, not unit test. Integration test with real concurrent cancellation optional.

---

### [NIT-1] API semantics: 200 OK with status="processing" для PROCESSING job

> Client получает `{"status": "processing"}` когда cancel request на PROCESSING job. Worker asynchronously flips to CANCELLED. Client polling `/jobs/{id}` shortly after sees CANCELLED — potential confusion.

**Почему это важно:** Intentional design, documented. Minor UX consideration.

**Рекомендация:** Optionally: return `{"status": "processing", "cancellation_requested": true}` для clarity. Но YAGNI — current design acceptable.

---

## Summary

2 CRITICAL issues (race + unlink failure), 1 MAJOR (latency bounds), 2 MINOR (documentation). Plan structure TDD-правильный, coverage adequate для объявленного scope, но missing race handling и unlink error.

**Verdict:** Design нуждается в доработке перед implementation — CRITICAL issues должны быть addressed в design doc и reflected в plan.


---

## ccs-executor (albb-qwen)

# Дизайн-ревью: Job Cancellation для Vision API Server

## Обзор

Провёл полный анализ дизайна и плана реализации. Ниже — замечания, от критических до косметических.

---

### [CRITICAL-1] Гонка при отмене QUEUED-джоба между `request_cancel` и worker dispatch

> В дизайне заявлено: «QUEUED → cancel_event.set(), status → CANCELLED... Worker later pulls id from queue, sees status != QUEUED, skips.» Однако в текущей архитектуре `request_cancel` меняет статус на `CANCELLED`, но worker после `get_next_job_id()` проверяет `if job.status != QUEUED: continue`. Это создаёт окно, где worker может уже вытащить job_id из очереди, но ещё не проверить статус — и тогда `mark_processing` будет вызван на уже отменённом джобе.

**Почему это важно:** Worker может ошибочно перевести джоб в `PROCESSING`, хотя клиент уже получил 200 OK со статусом `CANCELLED`. Это нарушает идемпотентность и создаёт несогласованность состояния.

**Рекомендация:** В `request_cancel` для QUEUED-джоба нужно атомарно проверить, не начал ли worker уже обработку. Либо добавить флаг `processing_started: bool`, либо использовать lock вокруг `get_next_job_id()` + проверки статуса. Альтернативно — worker должен проверять статус ПЕРЕД вызовом `mark_processing`, а не после.

---

### [CRITICAL-2] Отсутствие синхронизации `completed_at` между потоками

> В дизайне: «`completed_at` is set to the cancellation time» в `request_cancel` для QUEUED. Но `mark_cancelled` (вызывается из worker-потока) тоже пишет `completed_at = now()`. Это две записи из разных потоков без memory barrier.

**Почему это важно:** В Python GTP обеспечивает атомарность, но нет гарантии видимости значения между потоками без явного sync. Event-loop может увидеть stale значение.

**Рекомендация:** Либо писать `completed_at` только в event-loop (worker сигнализирует о завершении через queue/callback), либо использовать `asyncio.Lock` для защиты записи.

---

### [MAJOR-1] `JobCancelledError` не очищает encoder subprocess

> В дизайне: «FFmpeg subprocesses cleaned up via existing `__exit__`.» Но если `JobCancelledError` поднимается из середины `_pass2_render`, encoder может остаться в процессе записи. Контекстный менеджер `with FFmpegEncoder(...)` закроет stdin, но subprocess может зависнуть на `wait()`.

**Почему это важно:** Зависший encoder = утечка процесса + потенциальная блокировка файла вывода.

**Рекомендация:** Добавить явный `encoder.terminate()` / `encoder.kill()` в `finally`-блок внутри `_pass2_render`, или гарантировать, что `FFmpegEncoder.__exit__()` делает `process.terminate()` с timeout.

---

### [MAJOR-2] Тест 10 (`test_cancel_event_none_runs_to_completion`) не проверяет поведение при `None`

> В плане: «None no-op (3 tests)». Но если `cancel_event=None`, проверка `if cancel_event is not None and cancel_event.is_set()` должна быть полностью прозрачной. Тест должен подтвердить, что annotate проходит оба прохода без единой проверки события.

**Почему это важно:** Без явной проверки можно пропустить регрессию, где добавляется новая проверка без guard на `None`.

**Рекомендация:** В тесте добавить счетчик проверок (mock `cancel_event.is_set` не должен вызываться) или явно.assert_not_called().

---

### [MAJOR-3] Task 5: guard `if job.status != QUEUED: continue` ломает QUEUED→CANCELLED сценарий

> В плане worker должен пропускать джобы, где статус не QUEUED. Но если `request_cancel` перевёл QUEUED→CANCELLED, worker увидит CANCELLED и пропустит. Однако `mark_cancelled` никогда не будет вызван, и джоб останется в состоянии CANCELLED без `completed_at`.

**Почему это важно:** TTL cleanup не сработает, потому что `completed_at` не установлен. Джоб зависнет в памяти навсегда.

**Рекомендация:** Worker должен проверять: `if job.status == CANCELLED: mark_cancelled(job_id); continue` — то есть явно завершать отменённые джобы.

---

### [MINOR-1] Неясно, кто вызывает `request_cancel` — endpoint или middleware

> В дизайне: «Endpoint handler calls request_cancel». Но нет описания, как endpoint получает `JobManager` — через DI, глобал, singleton?

**Почему это важно:** Если `JobManager` создаётся заново на каждый запрос, `request_cancel` будет работать с другим экземпляром, чем worker.

**Рекомендация:** Явно указать в плане Task 6, что endpoint использует тот же instance `JobManager`, что и worker (через `Depends(get_job_manager)` или app.state).

---

### [MINOR-2] Отсутствует тест на race: cancel после последнего кадра

> В дизайне описан edge case: «Cancel arrives after annotator finishes last frame». Но в плане тестов (15 тестов) нет явного теста на эту гонку.

**Почему это важно:** Это единственный сценарий, где клиент может получить `processing` вместо `cancelled` — нужно задокументировать ожидаемое поведение.

**Рекомендация:** Добавить тест 16: mock annotate так, чтобы он завершался между `request_cancel` и проверкой статуса.

---

### [MINOR-3] `JobStats` для CANCELLED джобов — неочевидное поведение

> В дизайне: «`JobStats` remains `None` for cancelled jobs». Но клиент может ожидать хотя бы частичную статистику (сколько кадров обработано до отмены).

**Почему это важно:** Если клиент хочет показать прогресс до отмены, ему придётся парсить логи или метрики.

**Рекомендация:** Либо явно задокументировать «stats unavailable for CANCELLED», либо добавить поле `partial_stats: Optional[JobStats]`.

---

### [NIT-1] Название `JobCancelledError` нарушает Naming Convention проекта

> В кодеbase используются исключения в стиле `VideoProcessingError`, `InferenceTimeoutError`. `JobCancelledError` выбивается.

**Рекомендация:** Переименовать в `JobCancellationError` для консистентности.

---

### [NIT-2] В плане Task 4 не указано, где импортировать `JobCancelledError`

> План говорит «Add exception», но не указывает файл. Судя по контексту, это `app/video_annotator.py`, но worker и endpoint тоже должны импортировать.

**Рекомендация:** Уточнить: «Add to `app/video_annotator.py`, export from `app/__init__.py` или импортировать в `app/worker.py` и `app/main.py`».

---

### [NIT-3] Неясно, будет ли `/cancel` endpoint логгировать отмену

> В дизайне нет требования к логгированию. Для отладки гонок было бы полезно логгировать: «Cancel requested for job X, status=Y».

**Рекомендация:** Добавить в Task 6: «Log cancel request at INFO level with job_id and current status».

---

## Итог

**Критических:** 2 (гонка QUEUED→CANCELLED, синхронизация `completed_at`)
**Мажорных:** 3 (encoder cleanup, тест на None, worker guard)
**Минорных:** 3 (DI, race тест, partial stats)
**Косметических:** 3 (naming, импорт, логирование)

**Вердикт:** Дизайн требует доработки перед реализацией. Критичные гонки нужно устранить на уровне дизайна, иначе реализация унаследует баги.


---

## ccs-executor (albb-kimi)

Теперь у меня достаточно контекста для ревью. Ниже мой анализ:

---

### [CRITICAL-1] `cleanup_expired` не включает `CANCELLED` в проверку TTL

> В текущем коде `app/job_manager.py:122` cleanup_expired проверяет только `COMPLETED` и `FAILED`. Дизайн заявляет, что `CANCELLED` должен обрабатываться идентично для TTL-очистки, но план не явно не добавляет `CANCELLED` в условие.

**Почему это важно:** Если не добавить `CANCELLED` в `cleanup_expired`, отменённые задачи будут оставаться в памяти `_jobs` и на диске навсегда — утечка ресурсов.

**Рекомендация:** В Task 3 явно указать изменение строки 122 в `cleanup_expired`:
```python
if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
```

---

### [MAJOR-1] Семантика 200 OK для PROCESSING → "processing" неинтуитивна

> API возвращает `200 OK` с `status: "processing"` когда клиент запрашивает отмену задачи в процессе. Дизайн описывает это как корректное поведение, но клиент ожидает подтверждения отмены, а получает "ещё обрабатывается".

**Почему это важно:** Клиенту придётся делать дополнительные polling-запросы, чтобы узнать, отменилась ли задача. Это не очевидно из API.

**Рекомендация:** Добавить в ответ флаг `cancellation_requested: true` для PROCESSING статуса, чтобы клиент понимал, что запрос принят и задача будет отменена при первой возможности:
```python
return JobStatusResponse(
    job_id=job.job_id,
    status=job.status.value,  # "processing"
    cancellation_requested=True,  # NEW
    ...
)
```

---

### [MAJOR-2] `output_path.unlink()` не защищён от ошибок

> В дизайне указано удаление `output_path` при `JobCancelledError`, но не указан `missing_ok=True`. Если файл не существует (например, отмена произошла до начала pass 2), будет `FileNotFoundError`.

**Почему это важно:** Исключение при очистке может помешать корректному завершению worker и освобождению ресурсов.

**Рекомендация:** Явно указать в плане:
```python
if output_path.exists():
    output_path.unlink(missing_ok=True)
```

---

### [MAJOR-3] Не учтена отмена между pass1 и pass2

> Дизайн предполагает проверку `cancel_event` только внутри `_pass1_collect` и `_pass2_render`. Но между этими проходами выполняется `DetectionStabilizer.stabilize()` (потенциально тяжёлая операция на CPU), где нет проверки отмены.

**Почему это важно:** Время работы стабилизатора пропорционально числу детекций и может занимать секунды для длинных видео. Клиент не может отменить задачу в этом окне.

**Рекомендация:** Добавить проверку `cancel_event.is_set()` после стабилизации (между pass1 и pass2) и перед рендерингом. Это может потребовать передачи callback или переноса проверки в `annotate()`.

---

### [MINOR-1] Неочевидная семантика `completed_at` для отменённых задач

> Для QUEUED→CANCELLED дизайн устанавливает `completed_at = now()`, что используется для TTL. Но семантически задача не "завершена", а "отменена".

**Почему это важно:** Запутанность при отладке — `completed_at` для отменённой задачи может вводить в заблуждение.

**Рекомендация:** Добавить комментарий в код, объясняющий, что `completed_at` используется как timestamp для TTL (не только для успешного завершения).

---

### [MINOR-2] Отсутствует тест на race: cancel приходит во время stabilizer

> В плане 15 тестов, но ни один не проверяет сценарий, когда отмена приходит между окончанием pass1 и началом pass2.

**Рекомендация:** Добавить test #16 в `test_worker.py`: mock-аннотатор с задержкой в stabilizer, вызов cancel, проверка что задача отменяется после stabilizer (или до, в зависимости от реализации).

---

### [NIT-1] Непоследовательность в документации API

> В дизайне `/cancel` возвращает `JobStatusResponse`, но не указано, что `stats` должен быть `None` для cancelled jobs (дизайн говорит "JobStats remains None for cancelled jobs").

**Рекомендация:** Убедиться, что endpoint `/cancel` корректно обрабатывает `stats=None` при формировании ответа.

---

### [NIT-2] Не указан import для `threading` в `Job`

> Дизайн добавляет `cancel_event: threading.Event`, но не указан import в `app/job_manager.py`.

**Рекомендация:** Добавить `import threading` в Task 1.

---

Общая оценка: дизайн в целом корректен, но требует уточнений в области очистки ресурсов, TTL-обработки для CANCELLED, и добавления проверки отмены между проходами стабилизатора.


---

## ccs-executor (albb-minimax)

Теперь у меня достаточно контекста для полного обзора. Вот мои findings:

---

### [CRITICAL-1] Гонка между guard и mark_processing в воркере

> В реализации Task 5, guard `if job.status != JobStatus.QUEUED` проверяется перед `mark_processing()`. Однако существует TOCTOU (time-of-check to time-of-use): между проверкой и вызовом `mark_processing()` может вызваться `request_cancel`, который установит `status = CANCELLED`.
> 
> Более того, текущая `mark_processing()` в `app/job_manager.py:87-93` только логирует warning при неожиданном статусе, но всё равно устанавливает `status = PROCESSING`:
> ```python
> if job.status != JobStatus.QUEUED:
>     logger.warning(f"Job {job_id} unexpected state for processing: {job.status}")
> job.status = JobStatus.PROCESSING  # Выполняется ВСЕГДА
> ```
> 
> **Почему это важно:** Job может оказаться в статусе PROCESSING даже после отмены, что нарушает логику и может привести к неконсистентному состоянию.
> 
> **Рекомендация:** Переместить guard ПОСЛЕ `mark_processing()`:
> ```python
> job_manager.mark_processing(job_id)
> job = job_manager.get_job(job_id)
> if job is None or job.status != JobStatus.PROCESSING:
>     logger.info(f"Job {job_id} no longer in processing state, skipping")
>     continue
> ```

---

### [MAJOR-1] Отсутствие теста на race condition request_cancel + mark_processing

> План предусматривает тесты 14-15 для воркера, но нет теста, который бы проверял конкретный сценарий гонки: job выбран из очереди, но между проверкой и mark_processing() вызывается request_cancel.
> 
> **Почему это важно:** Без такого теста баг может остаться незамеченным.
> 
> **Рекомендация:** Добавить тест в Task 5, который мокает задержку между get_job() и mark_processing() и вызывает request_cancel в этот момент.

---

### [MAJOR-2] Несоответствие дизайна и плана в ответе для PROCESSING job

> В дизайне (строка 50-56) сказано:
> > PROCESSING → 200 OK с `status: "processing"` (worker flips to cancelled when it observes the event)
> 
> Но в плане Task 6 endpoint возвращает `job.status.value`, который для PROCESSING будет `"processing"`. Это соответствует дизайну.
> 
> **Однако**: дизайн говорит "A client cancelling a PROCESSING job will see `status: "processing"` in the response", но не объясняет, что клиент должен сделать follow-up GET. Это может быть неочевидно без документации.
> 
> **Рекомендация:** Убедиться, что в документации (Task 7) явно указано, что для PROCESSING jobs нужно делать повторный GET для проверки статуса.

---

### [MAJOR-3] mark_processing не проверяет return value

> В реализации `mark_processing()` нет никакой валидации — она просто устанавливает статус. Это означает, что вызов `mark_processing()` на уже CANCELLED job "молча" перезапишет статус на PROCESSING.
> 
> **Почему это важно:** Хотя guard в воркере должен предотвращать это, сама функция не защищена от неправильного использования.
> 
> **Рекомендация:** Рассмотреть добавление проверки в `mark_processing()`, или сделать её более явной в документации.

---

### [MINOR-1] Проверка cancel_event.is_set() дважды на каждой итерации

> В дизайне (строка 121-123) проверка:
> ```python
> while True:
>     if cancel_event is not None and cancel_event.is_set():
>         raise JobCancelledError()
> ```
> 
> Это делается дважды (в pass1 и pass2), на каждой итерации цикла обработки кадров. `cancel_event.is_set()` — быстрая атомарная операция, но проверка `cancel_event is not None` выполняется каждый раз.
> 
> **Почему это важно:** Небольшая, но ненужная проверка на каждой итерации. Для сотен тысяч кадров это может накопиться.
> 
> **Рекомендация:** Оптимизация не критична, но можно вынести проверку до цикла если cancel_event не None, или использовать early return в начале annotate().

---

### [MINOR-2] download_url всегда null для CANCELLED в Task 6

> В реализации endpoint (план строка 945-947):
> ```python
> download_url = None
> if job.status == JobStatus.COMPLETED:
>     download_url = f"/jobs/{job.job_id}/download"
> ```
> 
> Это корректно — для CANCELLED нет output. Но для COMPLETED, FAILED и CANCELLED логика идентична (все возвращают download_url = None).
> 
> **Рекомендация:** Упростить: `download_url = f"/jobs/{job.job_id}/download" if job.status == JobStatus.COMPLETED else None` — текущий код и так работает, просто чуть более многословен.

---

### [MINOR-3] Не проверяется output_path.exists() перед unlink в cleanup_partial_output

> В плане (строка 828-831):
> ```python
> try:
>     if output_path.exists():
>         output_path.unlink()
> except OSError as e:
>     logger.warning(f"Failed to remove partial output for {job_id}: {e}")
> ```
> 
> Здесь уже есть exists() проверка, всё ок.
> 
> **Почему это важно:** Нужно убедиться, что эта проверка реализована как показано в плане.

---

### [NIT-1] Тест 10 в плане — некорректное имя

> В Task 4, тест называется `test_cancel_event_none_runs_to_completion` (план строка 476), но в дизайне (строка 236) он упомянут как Test 10 без явного имени.
> 
> **Рекомендация:** Убедиться, что имена тестов согласованы.

---

### [NIT-2] plan не включает изменение _run_worker_until_job_done helper

> В Task 5, план добавляет тесты, но не упоминает, что helper `_run_worker_until_job_done` должен быть обновлён для распознавания CANCELLED как terminal state. Однако в плане (строка 718-740) этот код ЕСТЬ.
> 
> Это есть в плане — просто убедитесь, что он реализован.

---

### Замечаний нет

За исключением **CRITICAL-1** (гонка между guard и mark_processing), остальные пункты — улучшения или вопросы стиля. Дизайн в целом корректен:

- `threading.Event` — правильный примитив для cross-thread signaling
- API semantics (200/409/404) — соответствует дизайну
- Cancellation latency "sub-second" — реалистично для GPU, приемлемо для CPU
- 15 тестов покрывают основные сценарии
- Backward compatibility `cancel_event=None` — корректно

**Главная проблема — CRITICAL-1 — требует исправления перед реализацией.**
