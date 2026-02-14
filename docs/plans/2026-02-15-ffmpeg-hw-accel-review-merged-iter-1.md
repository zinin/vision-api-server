# Merged Design Review — Iteration 1

## codex-executor (gpt-5.3-codex)

### Critical Issues
- `NVIDIA` часть плана опирается на неверное предположение, что "ничего менять не нужно": в плане это явно зафиксировано (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:1053`), но текущий compose не включает `video` capability (`docker/docker-compose-nvidia.yml:16`). С `compute,utility` NVENC/NVDEC часто недоступны, и автодетект даст ложноположительный результат.
- В сниппете декодера кадр создается через `np.frombuffer(raw, ...)` (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:507`), это обычно read-only массив для `bytes`. Далее кадр модифицируется при отрисовке (`app/video_annotator.py:149`, `app/visualization.py:172`), что может падать на OpenCV.
- Обнаружение ускорения неполное и противоречит design: design требует проверку декодера (`docs/plans/2026-02-15-ffmpeg-hw-accel-design.md:49`), а плановый код проверяет только `hwaccels` и `h264_*` энкодеры (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:322`, `docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:333`). Для `video_codec=h265/av1` (`app/config.py:29`) это приведет к runtime-фейлам.
- Заявлен fallback chain (`docs/plans/2026-02-15-ffmpeg-hw-accel-design.md:33`), но в плане fallback только на этапе detect (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:994`), а при реальном сбое энкодера сразу exception (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:723`). Требование "just work" этим не выполняется.
- В декодере `stderr` запущен в `PIPE` (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:499`), но не дренируется во время работы, и `wait(timeout=10)` без `terminate/kill` (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:516`). Это риск дедлоков и зависших процессов.
- После удаления `_merge_audio()` (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:953`) ffmpeg становится критически обязательным, но текущий startup ведет себя как "optional warning" (`app/main.py:111`, `app/main.py:121`). Без пересмотра политики вы получите "сервис поднялся, но джобы стабильно падают".

### Concerns
- VAAPI путь захардкожен как `/dev/dri/renderD128` (`docs/plans/2026-02-15-ffmpeg-hw-accel-design.md:61`, `docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:273`). На части хостов будет другой render node.
- Для AMD-ветки `crf` фактически игнорируется (нет эквивалента rate-control в аргументах, см. `docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:271`). Параметр качества `VIDEO_CRF` (`app/config.py:30`) становится непредсказуемым по смыслу.
- Есть риск A/V рассинхрона на VFR: пайплайн фиксирует `-r` из метаданных (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:697`), а текущая метадата берется из `r_frame_rate` (`app/video_annotator.py:209`), что не всегда равно реальному таймингу потока.
- В плане есть некорректные тестовые сниппеты: `pytest`/`ValidationError` не импортированы (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:37`), а `BytesIO.close.assert_called()` невозможен (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:384`, `docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:448`).
- Task 6 слишком расплывчат: "add comment/verify" без конкретных assertions (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:981`). Это слабый контроль регрессий.
- Design и Plan расходятся по AMD пакетам: design включает `vainfo`, план его не добавляет (`docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md:1047`).

### Suggestions
- Делать не только "list-based detect", а активный probe backend+codec (микро encode/decode test) и хранить capability matrix.
- Реализовать runtime retry: `selected -> next backend -> CPU` на уровне `VideoAnnotator`, а не только в startup detect.
- Исправить жизненный цикл subprocess: writable frame buffer, `terminate/kill` на timeout, отдельное дренирование `stderr` (или `DEVNULL` для декодера).
- Обновить NVIDIA compose: добавить `video` в `NVIDIA_DRIVER_CAPABILITIES`, и явно задокументировать это рядом с VAS-2.
- Явно определить политику качества/совместимости: `-pix_fmt yuv420p` для CPU-кодеков, `-crf`/`-qp` стратегия по backend'ам.
- Добавить минимальные интеграционные тесты на реальном ffmpeg (короткий ролик): no-audio, VFR, corrupt input, forced backend unavailable.

### Questions
- При `VIDEO_HW_ACCEL=nvidia|amd` ожидается строгий fail-fast или все равно fallback на CPU?
- Должна ли задача завершаться "успешно без аудио", если merge/audio encode сломался, или это всегда `FAILED`?
- Требуется ли точное сохранение таймингов для VFR (PTS-level), или допускается CFR-нормализация?
- Нужно ли поддерживать хосты с несколькими render node (не только `/dev/dri/renderD128`)?

---

## gemini-executor (gemini-3-pro-preview)

### Critical Issues

1. **Риск взаимной блокировки (Deadlock) в `FFmpegEncoder`**
   - Суть: В плане указано: "FFmpegEncoder.close() must read stderr BEFORE wait()". Этого недостаточно. Если FFmpeg начнет выдавать предупреждения в `stderr` *во время* работы цикла `write_frame`, буфер `stderr` (обычно 64КБ) заполнится. FFmpeg приостановит чтение `stdin`. Python-скрипт заблокируется на `process.stdin.write()`. Классический deadlock.
   - Решение: Необходимо читать `stderr` асинхронно в отдельном потоке (daemon thread) на протяжении всего времени жизни процесса.

2. **Отсутствие обработки ротации видео (Metadata Rotation)**
   - Суть: Видео с телефонов часто записаны "боком" с флагом ротации (90°). FFmpegDecoder с `-f rawvideo` декодирует кадры "как есть" (stored orientation). YOLO будет детектить на перевёрнутом изображении, итоговое видео будет лежать на боку.
   - Решение: Добавить авто-ротацию при декодировании (`-vf "transpose=..."` или встроенный авто-поворот).

### Concerns

1. **A/V Sync с VFR видео** — форсирование CFR при VFR input может вызвать рассинхрон звука на длинных видео.
2. **BGR24 pipe bandwidth** — для 4K 60fps ~1.5 ГБ/сек через pipe. Альтернатива: YUV420p/NV12 (2x меньше).
3. **Зависимость `read_frame` от ffprobe** — если ffprobe сообщит одни размеры, а decoder выдаст с padding, будет desync/мусор.

### Suggestions

1. Добавить `write_frame` health check — `process.poll()` перед записью.
2. Использовать `-map_metadata 0` для сохранения глобальных тегов файла.
3. Расширить тесты — добавить тест на видео с ротацией.

### Questions

1. Как обрабатывать перегрузку NVENC (лимит сессий)? Runtime fallback на CPU или ошибка запроса?

---

## ccs-executor (glmt / GLM-4.7 Thinking)

### Critical Issues

1. **VAAPI device path хардкод** `/dev/dri/renderD128` — нужен discovery.
2. **Missing error handling в `FFmpegDecoder.read_frame()`** для crashed decoder mid-stream.
3. **Нет frame size consistency check** между ffprobe metadata и actual decoded frames.
4. **Resource leak если `write_frame()` exception** — stdin pipe left open.
5. **Нет early validation для `VIDEO_HW_ACCEL=nvidia|amd`** при отсутствии hardware.

### Concerns

1. **Context manager nesting race condition** — encoder crash leaves decoder running.
2. **Нет graceful shutdown** для annotation jobs с subprocess pipes.
3. **Stderr buffering deadlock risk** для decoder/encoder processes.
4. **Нет decoder/encoder pixfmt compatibility check.**
5. **Missing integration tests** для hardware detection на реальных системах.

### Suggestions

1. Добавить telemetry/metrics для GPU utilization.
2. Рассмотреть async frame read/write pipelining.
3. Добавить configuration для FFmpeg preset/RC mode.
4. Документировать fallback codec selection behavior.
5. Добавить тесты для `progress_callback`.

### Questions

1. VFR video — out of scope?
2. Video metadata passthrough (`-map_metadata`) — requirement?
3. Multi-audio-track — только первый трек, expected?
4. Memory footprint для 4K (~25MB per BGR frame) — backpressure?
5. Why `dataclass(frozen=True)` для HWAccelConfig?
