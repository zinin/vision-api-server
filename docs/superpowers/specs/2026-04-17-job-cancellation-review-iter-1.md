# Review Iteration 1 — 2026-04-17

## Источник

- Design: `docs/superpowers/specs/2026-04-17-job-cancellation-design.md`
- Plan: `docs/superpowers/plans/2026-04-17-job-cancellation.md`
- Review agents: codex-executor (gpt-5.4 xhigh), gemini-executor, ccs-executor × (glm, albb-glm, albb-qwen, albb-kimi, albb-minimax) — 7 ревьюеров
- Merged output: `docs/superpowers/specs/2026-04-17-job-cancellation-review-merged-iter-1.md`

## Замечания

### [CRITICAL] Утечка `input.mp4` при пропуске queued-cancelled job

> `request_cancel` переводит QUEUED в CANCELLED, worker при пикапе делает `continue` и обходит `finally`-блок, в котором удаляется `input.mp4`. Big uploads остаются на диске до TTL.

**Источник:** codex (MAJOR-1), gemini (CRITICAL-1).
**Статус:** Автоисправлено.
**Действие:** В дизайне — `request_cancel` для QUEUED теперь удаляет `input_path` сам. В плане Task 2 добавлена реализация (`try/except OSError`) и тест `test_request_cancel_queued_marks_cancelled_and_deletes_input`.

---

### [CRITICAL] Гонка `request_cancel` ↔ `mark_processing`

> Окно между `if job.status != QUEUED` и `mark_processing(...)`: cancel может прилететь внутри. `mark_processing` перезаписывает CANCELLED → PROCESSING, status клиента прыгает `QUEUED → CANCELLED → PROCESSING → CANCELLED`.

**Источник:** gemini (MAJOR-1), ccs-glm (CRITICAL-1), ccs-albb-glm (CRITICAL-1), ccs-albb-minimax (CRITICAL-1) — 4 ревьюера независимо.
**Статус:** Автоисправлено.
**Действие:** `mark_processing` превращён в CAS, возвращает `bool`. Worker'ский guard теперь: `if not job_manager.mark_processing(job_id): continue`. Обновлены дизайн-секции JobManager / Error Handling / Thread Safety, в плане Task 2 добавлены тесты `test_mark_processing_cas_queued_returns_true`, `test_mark_processing_cas_cancelled_returns_false`, изменён `test_job_lifecycle`.

---

### [MAJOR] Нет cancel-check между pass 1 и pass 2

> Стабилизация между проходами — потенциально секунды CPU-работы; после неё сразу запускаются FFmpeg decoder + encoder (дорогая GPU-инициализация). Cancel observable только с первой итерации pass 2.

**Источник:** ccs-glm (MAJOR-1), ccs-albb-kimi (MAJOR-3), ccs-albb-qwen (MAJOR-3).
**Статус:** Автоисправлено.
**Действие:** В `annotate()` добавлена проверка `cancel_event.is_set()` между вызовом `DetectionStabilizer.stabilize` и `_pass2_render`. В плане Task 4 добавлен тест `test_cancel_between_passes_raises` проверяющий, что pass 2 decoder/encoder не создаются.

---

### [MAJOR] FFmpeg teardown latency не учтён в "sub-second" обещании

> `FFmpegDecoder.close()` ждёт до 10 с, `FFmpegEncoder.close()` — до 300 с. После `JobCancelledError` клиент может видеть `processing` минутами.

**Источник:** codex (MAJOR-3).
**Статус:** Обсуждено с пользователем → вариант (a).
**Ответ:** Ослабить формулировку latency в дизайне. Hard-kill не добавлять (non-goal).
**Действие:** Секция "Cancellation Latency" полностью переписана: разделены "checkpoint latency" (sub-second) и "terminal-transition latency" (включает FFmpeg teardown, до нескольких минут в патологическом случае для encoder). В плане ничего не меняется.

---

### [MAJOR] `mark_processing` до `get_model()` — cancel не observable во время model load

> `get_model()` может занимать секунды-десятки секунд (cold start, download, retries). cancel_event впервые читается только внутри `annotate()`.

**Источник:** codex (MAJOR-2).
**Статус:** Обсуждено с пользователем → вариант (a).
**Ответ:** Добавить явный cancel-check сразу после `get_model()`, перед построением annotator'а. Не двигать `mark_processing` — это изменило бы семантику статуса.
**Действие:** В дизайне Worker section пункт (b). В плане Task 5 Step 3 пункт (d) + тест `test_cancel_during_model_load`.

---

### [MAJOR] Cancel перед pre-annotate failure → неконсистентный терминальный статус

> Клиент сделал `/cancel`, но до `annotate()` упал `get_model()`. Worker пишет FAILED, хотя клиент просил отмены. Поведение зависит от того, где именно в pipeline упал worker.

**Источник:** codex (MAJOR-4).
**Статус:** Обсуждено с пользователем → вариант (a).
**Ответ:** Cancel имеет приоритет. Все pre-annotate `except`-ветки проверяют `cancel_event.is_set()` и в этом случае вызывают `mark_cancelled` вместо `mark_failed`. Симметрично с `JobCancelledError`-веткой.
**Действие:** В дизайне Worker section пункт (c) + Error Handling table. В плане Task 5 Step 3 пункт (c) + тест `test_cancel_precedence_over_model_load_failure`.

---

### [MINOR] `JobStatusResponse.status` field description устарел

> Pydantic-описание в `app/models.py:181` перечисляет только `queued, processing, completed, failed`. OpenAPI-схема FastAPI будет stale после добавления `cancelled`.

**Источник:** ccs-glm (MINOR-1), ccs-albb-qwen (NIT-1).
**Статус:** Автоисправлено.
**Действие:** В плане Task 7 добавлен новый Step 1 для правки `app/models.py`, в дизайне Files Touched соответственно обновлён.

---

### [MINOR] Тест `test_skip_queued_but_cancelled` не проверяет реальное срабатывание skip-guard

> Поскольку `request_cancel` уже ставит CANCELLED, `_run_worker_until_job_done` видит terminal сразу и может вернуться до того, как worker пикапнул id. Тест проходит, но skip-guard не exercised.

**Источник:** ccs-glm (MINOR-3).
**Статус:** Автоисправлено.
**Действие:** В тест добавлены `assert worker_job_manager._queue.empty()` и проверка, что input-файл удалён `request_cancel`-ом.

---

### [MINOR] Тест `test_cancel_during_processing_marks_cancelled` не проверяет передачу `cancel_event`

> mock annotate всегда кидает JobCancelledError независимо от аргументов, тест не верифицирует, что worker реально передаёт `cancel_event=job.cancel_event`.

**Источник:** ccs-glm (MINOR-4).
**Статус:** Автоисправлено.
**Действие:** В тест добавлен `assert call_kwargs["cancel_event"] is job.cancel_event`.

---

### [MINOR] Дизайн и план расходятся по безопасному удалению partial output

> Spec говорит `unlink(missing_ok=True)`, плане `try/except OSError`. `missing_ok` не покрывает `PermissionError`.

**Источник:** codex (MINOR-1).
**Статус:** Автоисправлено.
**Действие:** В дизайне формулировка изменена — `_cleanup_partial_output` обёрнут в `try/except OSError`, соответствует плану.

---

### [MINOR] Расхождение в количестве тестов (15 в дизайне vs 21+ в плане)

**Источник:** codex (NIT-1), ccs-glm (MINOR-2).
**Статус:** Автоисправлено.
**Действие:** В дизайне раздел Testing полностью перепронумерован (теперь 23 теста), цифра "15 tests total" убрана.

---

### [NIT] Расхождение "3 reads" (дизайн) vs "2 predict calls" (план)

**Источник:** ccs-glm (NIT-1).
**Статус:** Автоисправлено.
**Действие:** В дизайне описание теста 11 переформулировано ("during pass 1 (after ~2 predict calls)"), согласовано с фактическим кодом теста в плане.

---

### [DISMISSED] `cancellation_requested` флаг в ответе для PROCESSING

**Источник:** ccs-albb-kimi (MAJOR-1).
**Статус:** Отклонено. Пользователь ранее утвердил вариант "200 + status='processing'" (brainstorming, Q5/B). Добавлять доп. поле — YAGNI.

### [DISMISSED] Переименовать `JobCancelledError` → `JobCancellationError`

**Источник:** ccs-albb-glm (NIT-1).
**Статус:** Отклонено. В кодовой базе нет `VideoProcessingError`/`InferenceTimeoutError` (ревьюер ошибся), так что исходный аргумент не валиден. Bikeshed.

### [DISMISSED] `JobStats.partial_stats` для cancelled jobs

**Источник:** ccs-albb-glm (MINOR-3).
**Статус:** Отклонено. YAGNI, вне scope.

### [DISMISSED] "Memory barrier для `completed_at`"

**Источник:** ccs-albb-glm (CRITICAL-2).
**Статус:** Отклонено. Неверно: в CPython присваивание reference атомарно под GIL. Никаких явных барьеров не требуется.

### [DISMISSED] Encoder subprocess zombie в `__exit__`

**Источник:** ccs-albb-glm (MAJOR-1).
**Статус:** Отклонено. Existing `FFmpegEncoder.__exit__` уже делает `process.terminate()` + wait с таймаутом. Кодекс справедливо поднял этот же вопрос как latency-claim (не zombie), и это уже покрыто решением (a) по вопросу A.

### [DISMISSED] Task 4 перед Task 5 ломает per-commit invariant

**Источник:** gemini (MINOR-1).
**Статус:** Отклонено. В строгом TDD red→green между коммитами временная несогласованность — нормальный артефакт. Objective invariant измеряется на конец серии задач, не на каждый коммит.

### [DISMISSED] Микро-оптимизация двойной проверки `cancel_event is not None`

**Источник:** ccs-albb-minimax (MINOR-1).
**Статус:** Отклонено. `is None` — O(1) проверка reference, overhead несущественен; clarity важнее.

### [DISMISSED] DI JobManager в endpoint

**Источник:** ccs-albb-glm (MINOR-1).
**Статус:** Отклонено. План уже использует `Depends(get_job_manager)` (Task 6).

### [DISMISSED] Логирование cancel request в endpoint

**Источник:** ccs-albb-glm (NIT-3).
**Статус:** Отклонено. В дизайне уже есть таблица логирования с записью в `JobManager.request_cancel`.

### [DISMISSED] Race-условие test (CRITICAL race via concurrency)

**Источник:** ccs-glm (MINOR-2), ccs-albb-minimax (MAJOR-1).
**Статус:** Отклонено как отдельный тест. Race полностью устраняется CAS `mark_processing` (проверяется unit-тестом `test_mark_processing_cas_cancelled_returns_false`). Concurrent integration test для асинхронной гонки чрезмерен — CAS-инвариант доказуем чисто unit-тестом.

---

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| `docs/superpowers/specs/2026-04-17-job-cancellation-design.md` | JobManager section: добавлены input_path cleanup в `request_cancel` и CAS `mark_processing`. Annotator section: описан between-passes check. Worker section: 4 пункта (CAS guard, post-get_model check, `JobCancelledError`, precedence over pre-annotate failure). Cleanup Behaviour: обновлено место удаления `input.mp4`. Cancellation Latency: разделено на checkpoint + terminal-transition с реалистичными границами. Error Handling: 3 новых кейса. Thread Safety: упомянут CAS. Testing: полная переномерация (23 теста). Files Touched: обновлены описания всех файлов. |
| `docs/superpowers/plans/2026-04-17-job-cancellation.md` | Header Architecture обновлён. File Structure расширено. Task 2 переименован ("CAS mark_processing"), добавлены 4 новых теста и CAS-реализация, обновлён `test_job_lifecycle`. Task 4: добавлен between-passes check в annotate + тест `test_cancel_between_passes_raises`. Task 5: CAS guard, шаги (c)(d)(e), тесты `test_cancel_during_processing_marks_cancelled` и `test_skip_queued_but_cancelled` обогащены ассертами, добавлены `test_cancel_during_model_load` и `test_cancel_precedence_over_model_load_failure`. Task 7: новый Step 1 для `app/models.py`, перенумерован commit. |

## Статистика

- Всего замечаний: **26**
- Автоисправлено: **11** (клирные фиксы и согласования)
- Обсуждено с пользователем: **3** (A, B, C — все ответы (a))
- Отклонено: **11** (в том числе одна формально-критическая ложная тревога)
- Повторов (автоответ): 0 (это первая итерация)
- Пользователь сказал «стоп»: Нет
- Агенты: codex-executor (gpt-5.4 xhigh), gemini-executor, ccs-executor × (glm, albb-glm, albb-qwen, albb-kimi, albb-minimax)
