# Merged Design Review — Iteration 2

## codex-executor (gpt-5.3-codex)

### Critical Issues
- `docs/plans/2026-02-14-video-annotation.md:506` (`startup_sweep`) удаляет **все** директории в `VIDEO_JOBS_DIR` без проверки формата job-id. При ошибочной конфигурации (`VIDEO_JOBS_DIR=/tmp` и т.п.) это может снести чужие данные. В `design` это же решение зафиксировано в `docs/plans/2026-02-14-video-annotation-design.md:99`.
- `docs/plans/2026-02-14-video-annotation.md:1078` → `:1099`: статус ставится `completed`, а затем очистка `input.mp4` идет в том же `try`. Если `unlink()` упадет, job будет переписан в `failed`, хотя `output.mp4` уже создан. Это ломает консистентность API.
- `docs/plans/2026-02-14-video-annotation.md:657` → `:724`: в `VideoAnnotator.annotate()` `writer.release()` вызывается только в happy-path (`:710`). При исключении в цикле кадров writer не гарантированно закрывается, и промежуточные файлы могут остаться/быть повреждены.
- `docs/plans/2026-02-14-video-annotation.md:1186` → `:1222`: обработка temp upload-файла неполная. Ловится только `HTTPException` и `RuntimeError` от queue overflow; I/O ошибки записи/перемещения оставят `upload_*.tmp` на диске. При этом `startup_sweep` удаляет только директории (`:511`), не такие файлы.

### Important Issues
- `docs/plans/2026-02-14-video-annotation.md:1180` → `:1219`: ограничение очереди применяется после полной загрузки файла. При заполненной очереди клиенты все равно могут залить сотни МБ, что создает лишнюю нагрузку на диск/сеть и плохой UX.
- `docs/plans/2026-02-14-video-annotation.md:752` → `:759`: метаданные ffprobe принимаются без валидации (width/height/fps могут быть 0/некорректными при `returncode==0`), и fallback на cv2 не сработает. Это приведет к ложным отказам `VideoWriter`.
- `docs/plans/2026-02-14-video-annotation.md:1188` → `:1197`: синхронный `f.write()` внутри async endpoint блокирует event loop на больших загрузках.
- `docs/plans/2026-02-14-video-annotation.md:1066` → `:1070`: используется `run_in_executor(None, ...)` (дефолтный executor), а не контролируемый executor сервиса. Пул потоков и его поведение не управляются конфигом.
- `docs/plans/2026-02-14-video-annotation.md:1329`: в документацию снова попал `MAX_CONCURRENT_JOBS`, что конфликтует с текущим решением про отказ от этого параметра.

### Suggestions
- Ввести "reservation slot" перед upload: быстрый отказ при переполненной очереди, затем commit job только после успешной записи файла.
- Разделить переходы состояния и cleanup: после `mark_completed` ошибки удаления входного файла логировать отдельно, не переводя job в `failed`.
- Сделать безопасный sweep: удалять только директории, соответствующие шаблону job-id, и только внутри строго валидированного базового пути; отдельно чистить `upload_*.tmp`.
- Добавить минимальные интеграционные тесты на критичные сценарии: overflow очереди, ошибка `move`, падение в середине annotate, корректность статуса при cleanup-ошибках.

### Questions
- Какое целевое поведение при заполненной очереди: отказ **до загрузки файла** или допускается текущий "upload then 429"?
- Должен ли `failed` job хранить `input.mp4` для диагностики, или удаляем сразу после ошибки?
- Нужно ли жестко валидировать `VIDEO_JOBS_DIR` (например, запрет системных путей вроде `/tmp`, `/`, `/var`)?
- Подтверждаете ли удаление `MAX_CONCURRENT_JOBS` из Task 7 документации, чтобы не расходиться с принятым решением?

---

## gemini-executor

### Critical Issues
1. **Опасность `startup_sweep` при некорректной конфигурации** — startup_sweep удалит все файлы/директории если VIDEO_STORAGE_PATH сконфигурирован как системная директория.
2. **Конфликт пакетов OpenCV** — opencv-python-headless и opencv-contrib-python-headless конфликтуют в namespace Python если установлены одновременно.
3. **Отсутствие ограничения размера очереди (Backpressure)** — asyncio.Queue() по умолчанию бесконечна, при наплыве сервер продолжит принимать видео.

### Important Issues
4. **Валидация `detect_every`** — 0 или отрицательное число → деление на ноль.
5. **Обработка ошибок FFmpeg** — subprocess.run может упасть молча, stderr не логируется.

### Suggestions
6. **Smoke-тест для CV2/FFmpeg** — тест с 10 черными кадрами через пайплайн.
7. **Визуальный скачок цветов** — цвета рамок меняются каждые N кадров если привязаны к tracker ID.

---

## ccs-executor (glmt/GLM-4.7)

### Critical Issues
- **C1:** Race condition в `create_job()` между проверкой и put_nowait. Рекомендуют asyncio.Lock.
- **C2:** `startup_sweep()` удаляет без проверки состояний. Если сервер остановлен при PROCESSING — удалит файлы рабочего job.
- **C3:** `_merge_audio()` fallback копирует video_only без предупреждения. Нет флага audio_missing.
- **C4:** Существующие endpoints (`/detect/video`, `/extract/frames`) загружают файл в RAM. Out of scope.

### Important Issues
- **I1:** Naive datetime без timezone.
- **I2:** cv2.VideoCapture не закрывается при exception в fallback path _get_video_metadata.
- **I3:** Нет валидации detect_every <= total_frames.
- **I4:** mark_processing() не проверяет текущий статус.
- **I5:** Worker cancellation может оставить zombie processes.
- **I6:** Minor naming: переменная `evicted`.

### Suggestions
- S1: NamedTemporaryFile для upload.
- S2: Логирование параметров в annotate().
- S3: Conditional YOLO verbose при DEBUG.
- S4: Метрики мониторинга.
- S5: @dataclass(slots=True).
- S6: FFmpeg availability check на startup.
- S7: VideoWriter fourcc fallback.

### Questions
- Q1: Memory overhead для 100 CSRT trackers.
- Q2: Поведение при corrupted video.
- Q3: classes как str vs list[str].
- Q4: Tracker drift с degenerate bbox.
- Q5: Cancel endpoint (DELETE /jobs/{job_id}).
- Q6: Disk space pre-check.
