# Merged Design Review — Iteration 3

## codex-executor (gpt-5.3-codex)

### Critical Issues
1. В плане используется `model_manager._executor` (`docs/plans/2026-02-14-video-annotation.md:1113`), но в текущем `ModelManager` такого поля нет (`app/model_manager.py:50`). Это даст `AttributeError` до внутреннего `try` (`docs/plans/2026-02-14-video-annotation.md:1114`), и job может застрять в `processing` (внешний обработчик только логирует, `docs/plans/2026-02-14-video-annotation.md:1154`).
2. При ошибке загрузки модели делается `continue` (`docs/plans/2026-02-14-video-annotation.md:1085`), а cleanup входного файла расположен ниже (`docs/plans/2026-02-14-video-annotation.md:1144`). В итоге большие `input.mp4` остаются на диске при ошибочных `model`.
3. `create_job()` вызывается до перемещения файла (`docs/plans/2026-02-14-video-annotation.md:1256`), а `_shutil.move(...)` вынесен вне защищенного блока (`docs/plans/2026-02-14-video-annotation.md:1280`). При падении `move` получите 500 + "битый" job в очереди/хранилище.
4. Лимит очереди не ограничивает фазу upload при конкуренции: проверка до загрузки (`docs/plans/2026-02-14-video-annotation.md:1230`) не резервирует слот. Много параллельных клиентов могут одновременно записывать большие `.tmp`, что дает DoS по диску/IO, несмотря на `MAX_QUEUED_JOBS`.

### Important Issues
1. Новые настройки добавляются без валидации (`docs/plans/2026-02-14-video-annotation.md:96`). Если `DEFAULT_DETECT_EVERY=0`, будет деление по модулю на ноль в цикле кадров (`docs/plans/2026-02-14-video-annotation.md:706`).
2. Метаданные могут быть некорректны для VFR/rotated видео: fps берется из `r_frame_rate` (`docs/plans/2026-02-14-video-annotation.md:779`), а `VideoWriter` открывается по размерам ffprobe (`docs/plans/2026-02-14-video-annotation.md:690`) без сверки с фактическим первым кадром.
3. Валидация входа по сути только по расширению файла (`docs/plans/2026-02-14-video-annotation.md:1214`). Не-видео payload попадет в async pipeline и зря съест worker/очередь.
4. План тестов не покрывает endpoint/worker-сценарии отказов (Task 2-4 покрывают только config/models/job_manager; интеграция в Task 8 ручная). Критичные баги выше автотестами не ловятся.

### Suggestions
1. Сделать submit flow транзакционным: резерв слота -> upload -> валидация медиа -> `create_job` -> move, с гарантированным rollback.
2. Перестроить `_annotation_worker` в per-job `try/except/finally`, где `finally` всегда чистит `input_path`.
3. Синхронизировать стратегию запуска и импорты (в Task 8 используется `uvicorn app.main:app`, `docs/plans/2026-02-14-video-annotation.md:1410`, а код опирается на абсолютные импорты вида `from config import ...`, `app/main.py:15`).

### Questions
1. Нужно ли валидировать `model` синхронно в `POST /detect/video/visualize` до тяжелой загрузки файла?
2. Должны ли "in-flight uploads" считаться частью лимита очереди, чтобы лимит реально защищал диск/IO?
3. Какой контракт по VFR/rotation: обязаны сохранять точную длительность и ориентацию, или допустимы отклонения?

---

## gemini-executor

### Critical Issues
1. `model_manager._executor` не существует в ModelManager — вызовет AttributeError. Нужно либо добавить ThreadPoolExecutor в ModelManager, либо использовать None.

### Important Issues
1. OpenCV package conflict — pip install не удалит старый opencv-python-headless автоматически. Рекомендация: pip uninstall -y opencv-python-headless в Task 1.
2. Empty detections reset trackers — при пустых детекциях на detection frame все трекеры очищаются мгновенно. Корректное поведение для MVP но стоит документировать.

### Suggestions
1. S-TRACKER-COLOR подтверждён как решённый — (tracker, det) кортеж сохраняет class_id.
2. job_manager.shutdown() в lifespan — уже учтено в плане.
3. FFmpeg stderr[:500] — лучше логировать последние 500 символов, т.к. самая важная информация обычно в конце.

### Questions
1. Будете ли добавлять _executor в ModelManager или использовать None?

---

## ccs-executor (glmt/GLM-4.7)

### Critical Issues
1. C-1: `model_manager._executor` не существует — AttributeError. Решение: использовать get_executor() из inference_utils.py.
2. C-2: DetectionBox — frozen dataclass. Не проблема для текущего плана (только создание), но учитывать при расширении.
3. C-3: Опечатка Annotated в main.py. [Ложное срабатывание — код использует Annotated корректно]
4. C-4: Отсутствие FileResponse import. [Адресовано в плане Task 6 Step 2]
5. C-5: Приватные методы не переименованы детально. [Адресовано в плане Task 5 Step 0]
6. C-6: DetectEveryQuery/ClassesQuery не определены. [Адресовано в плане Task 6 Step 5]

### Important Issues
1. I-1: detect_every > total_frames validation. [Ранее отложено]
2. I-2: Tracker drift при высоком detect_every.
3. I-3: FileResponse без проверки существования. [Адресовано в плане — проверка уже есть]
4. I-4: get_job_manager dependency не существует. [Адресовано в плане Task 6 Step 1]
5. I-5: Race condition в check_queue_capacity → create_job. [FP-1 из iter-2]
6. I-6: startup_sweep удаляет dirs до worker. [FP-5 из iter-2]
7. I-7: worker_task не в app.state. [C4 из iter-1]

### Suggestions
1. S-1: Логирование параметров. [S-LOGGING из iter-2]
2. S-2: queued_at timestamp. [Отложено — monitoring]
3. S-3: Naming convention (visualize vs visualise).

### Questions
1. Q-1: Corrupted video handling. [Отвечено iter-2]
2. Q-2: Disk limit strategy. [Отложено]
3. Q-3: Multi-GPU strategy. [Отложено]
