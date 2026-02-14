# Review Iteration 1 — 2026-02-15

## Источник

- Design: `docs/plans/2026-02-15-ffmpeg-hw-accel-design.md`
- Plan: `docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md`
- Review agents: codex-executor (gpt-5.3-codex), gemini-executor (gemini-3-pro-preview), ccs-executor (glmt/GLM-4.7)
- Merged output: `docs/plans/2026-02-15-ffmpeg-hw-accel-review-merged-iter-1.md`

## Замечания

### C1: Stderr deadlock в FFmpeg subprocess pipes

> stderr=PIPE без async drain → deadlock при большом stderr output (>64KB буфер заполняется, FFmpeg блокируется на stderr write, Python блокируется на stdin write).

**Источник:** Gemini (Critical #1), CCS (Concern #3), Codex (Critical #5)
**Статус:** Новое
**Ответ:** Daemon thread для async чтения stderr
**Действие:** Обновлены FFmpegDecoder и FFmpegEncoder в плане — добавлен `_drain_stderr()` daemon thread, `threading.Thread` для обоих классов

---

### C2: Read-only frame buffer из np.frombuffer

> `np.frombuffer(bytes)` возвращает read-only array. OpenCV операции (draw bbox) падают с ошибкой.

**Источник:** Codex (Critical #2)
**Статус:** Новое
**Ответ:** Добавить `.copy()` после frombuffer
**Действие:** Добавлен `.copy()` в `FFmpegDecoder.read_frame()` в плане

---

### C3: NVIDIA compose без `video` capability

> `docker-compose-nvidia.yml` имеет `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. Без `video` NVENC/NVDEC недоступны.

**Источник:** Codex (Critical #1)
**Статус:** Новое
**Ответ:** Добавить `video` в capabilities
**Действие:** Task 7 обновлён — добавлен шаг обновления docker-compose-nvidia.yml

---

### C4: Ротация видео не обрабатывается

> Телефонное видео с rotate=90° metadata → FFmpeg decode "as stored" → повёрнутые фреймы → кривой YOLO + боковое выходное видео.

**Источник:** Gemini (Critical #2)
**Статус:** Новое
**Ответ:** Auto-rotate в decoder. FFmpeg по умолчанию auto-rotate при decode, нужно убедиться что это работает с rawvideo pipe.
**Действие:** Добавлено примечание в design doc. Имплементатор должен верифицировать авто-ротацию.

---

### C5: Неполная HW-детекция — проверяет только h264

> Detection проверяет только `h264_nvenc`/`h264_vaapi`, но `VIDEO_CODEC` может быть h265/av1. При h265 на NVIDIA без `hevc_nvenc` — runtime failure.

**Источник:** Codex (Critical #3)
**Статус:** Новое
**Ответ:** Проверять encoder для configured codec
**Действие:** `detect_hw_accel()` обновлён в плане — принимает `codec` параметр, проверяет конкретный encoder

---

### C6: FFmpeg стал mandatory, но startup не проверяет

> После удаления `_merge_audio()` FFmpeg — единственный encoder. Но startup только предупреждает если FFmpeg не найден → сервис поднимается, все джобы падают.

**Источник:** Codex (Critical #6)
**Статус:** Новое
**Ответ:** Добавить hard check при startup
**Действие:** Task 6 обновлён — добавлен `shutil.which("ffmpeg")` check в lifespan

---

### I1: VAAPI device path hardcode

> `/dev/dri/renderD128` захардкожен. На других хостах может быть renderD129+.

**Источник:** CCS (Critical #2), Codex (Concern)
**Статус:** Новое
**Ответ:** Hardcode по умолчанию + env override через `VAAPI_DEVICE`
**Действие:** Добавлен `vaapi_device` в config.py и HWAccelConfig. Design doc обновлён.

---

### I2: Нет error handling в decoder при crash mid-stream

> Если FFmpeg decoder падает mid-stream, `read()` возвращает partial data → трактуется как EOF, не ошибка.

**Источник:** CCS (Critical #3), Gemini (Suggestion #1)
**Статус:** Новое
**Ответ:** Проверять `process.poll()` в read_frame
**Действие:** `read_frame()` обновлён в плане — проверяет returncode при short read

---

### I3: Frame size consistency

> ffprobe метаданные могут отличаться от реального decode output (padding при HW decode).

**Источник:** Gemini (Concern #3), CCS (Critical #4)
**Статус:** Новое
**Ответ:** Спорно, но можно добавить explicit `-s` в decoder. FFmpeg с `-pix_fmt bgr24` делает conversion.
**Действие:** Оставлено как есть — имплементатор может добавить `-s` при необходимости

---

### I4: Resource leak при exception в write_frame

> Если write_frame бросает исключение, stdin pipe не закрывается корректно.

**Источник:** CCS (Critical #5)
**Статус:** Новое
**Ответ:** `__exit__` уже вызывается при exception в with-block. Добавить try/except в close().
**Действие:** `FFmpegEncoder.close()` обновлён — try/except для stdin.close()

---

### I5: Runtime fallback не реализован

> При сбое encoder в runtime → exception, нет автоматического retry с CPU.

**Источник:** Codex (Critical #4), Gemini (Question #1)
**Статус:** Новое
**Ответ:** Startup-only fallback для v1. Runtime retry = overengineering.
**Действие:** Добавлено в Non-Goals design doc

---

### I6: AMD VAAPI игнорирует CRF

> VAAPI encoder args не содержат rate control параметров. `VIDEO_CRF` бесполезен для AMD.

**Источник:** Codex (Concern)
**Статус:** Новое
**Ответ:** Добавить `-qp` для VAAPI
**Действие:** HWAccelConfig AMD encode args обновлены — добавлен `-qp`

---

### I7: Graceful shutdown subprocess при отмене джоба

> asyncio cancellation не убивает subprocess pipes.

**Источник:** CCS (Concern #2)
**Статус:** Новое
**Ответ:** Out of scope для v1. Текущий cv2 pipeline тоже не обрабатывает.
**Действие:** Добавлено в Non-Goals design doc

---

### I9: Design/Plan расходятся по AMD packages

> Design включает `vainfo`, plan не устанавливает.

**Источник:** Codex (Concern)
**Статус:** Новое
**Ответ:** Добавить vainfo в plan
**Действие:** Task 7 обновлён — vainfo добавлен в apt-get install

---

### Q1: VFR видео

> CFR нормализация при VFR input может давать A/V desync на длинных видео.

**Источник:** Gemini (Concern #1), Codex (Concern), CCS (Question #1)
**Статус:** Новое
**Ответ:** CFR нормализация acceptable для v1
**Действие:** Добавлено в Non-Goals design doc

---

### Q2: Forced HW mode при отсутствии оборудования

> VIDEO_HW_ACCEL=nvidia но GPU нет — fail-fast или fallback?

**Источник:** CCS (Critical #1), Codex (Question)
**Статус:** Новое
**Ответ:** Fallback на CPU + warning log
**Действие:** Уже реализовано в detect_hw_accel(). Задокументировано в design doc.

---

### Q3: Metadata passthrough

> `-map_metadata 0` для сохранения глобальных тегов файла.

**Источник:** Gemini (Suggestion #2), CCS (Question #2)
**Статус:** Новое
**Ответ:** Да, добавить
**Действие:** `-map_metadata 0` добавлен в FFmpegEncoder command в плане и design doc

---

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| `docs/plans/2026-02-15-ffmpeg-hw-accel-design.md` | Обновлена детекция (проверка codec-specific encoder), VAAPI device configurable, stderr daemon thread, frame .copy(), process health check, FFmpeg mandatory, NVIDIA compose video capability, `-map_metadata 0`, Non-Goals (runtime fallback, VFR, graceful shutdown) |
| `docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md` | Task 1: vaapi_device config. Task 2: detect_hw_accel(codec, vaapi_device), AMD -qp. Task 3: stderr drain thread, .copy(), process.poll() check. Task 4: stderr drain, write_frame health check, -map_metadata. Task 6: FFmpeg mandatory check, pass codec/vaapi_device. Task 7: NVIDIA compose video capability, vainfo package |

## Статистика

- Всего замечаний: 17
- Новых: 17
- Повторов (автоответ): 0
- Пользователь сказал "стоп": Нет
- Агенты: codex-executor, gemini-executor, ccs-executor
