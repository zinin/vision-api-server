# Review Iteration 2 — 2026-02-15

## Источник

- Design: `docs/plans/2026-02-15-ffmpeg-hw-accel-design.md`
- Plan: `docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md`
- Review agents: codex-executor (gpt-5.3-codex), gemini-executor (gemini-3-pro-preview), ccs-executor (glmt/GLM-4.7)
- Merged output: `docs/plans/2026-02-15-ffmpeg-hw-accel-review-merged-iter-2.md`

## Замечания

### CRITICAL-5: Некорректный порядок аргументов FFmpeg для AMD/VAAPI

> В FFmpegEncoder аргументы из `hw_config.get_encode_args()` добавляются ПОСЛЕ `-i pipe:0`. Но для AMD `-vaapi_device` — глобальный параметр, который должен быть ПЕРЕД `-i`. FFmpeg проигнорирует устройство или выдаст ошибку.

**Источник:** Gemini (Critical #1)
**Статус:** Новое
**Ответ:** Разделить encode args на global и codec-specific
**Действие:** HWAccelConfig: добавлен `global_encode_args` property (возвращает args ПЕРЕД -i). `get_encode_args()` теперь возвращает только codec-specific args. FFmpegEncoder обновлён: `cmd += hw_config.global_encode_args` перед `-i`, `cmd += hw_config.get_encode_args()` после inputs. VAAPI `-vaapi_device` перенесён из `get_encode_args()` в `global_encode_args`.

---

### CRITICAL-7: _drain_stderr несовместим с text=False

> Popen создаётся с `text=False` (binary mode), поэтому stderr выдаёт `bytes`. Но `_drain_stderr` собирает в `list[str]` — type mismatch.

**Источник:** CCS (Critical #2)
**Статус:** Новое
**Ответ:** Изменить на `list[bytes]` (точнее `deque[bytes]`)
**Действие:** `_drain_stderr` обновлён: `collected: deque[bytes]`. Добавлена `_format_stderr()` для декодирования при формировании ошибок: `b"".join(tail).decode("utf-8", errors="replace")`.

---

### CRITICAL-8: Race condition на доступ к stderr_lines

> Daemon thread пишет в `_stderr_lines`, а `close()` читает после `join(timeout=5)`. При timeout — несинхронизированный доступ.

**Источник:** CCS (Critical #3)
**Статус:** Новое
**Ответ:** Использовать `collections.deque(maxlen=100)`
**Действие:** `_stderr_lines` изменён с `list` на `deque(maxlen=100)`. deque thread-safe для append/iteration в CPython (GIL).

---

### CRITICAL-3: ffprobe не проверяется при startup

> Pipeline зависит от ffprobe для metadata extraction, но startup check проверяет только ffmpeg. Job падает позже.

**Источник:** Codex (Critical #3)
**Статус:** Новое
**Ответ:** Добавить `shutil.which("ffprobe")` проверку в lifespan
**Действие:** Task 6 обновлён — добавлен `shutil.which("ffprobe")` check рядом с ffmpeg. Design doc обновлён: "FFmpeg and ffprobe are mandatory".

---

### CRITICAL-1: HW-детекция только статическая, нет runtime probe

> `detect_hw_accel()` не тестирует реальную работу GPU. В контейнере device может быть в списке, но не функционален.

**Источник:** Codex (Critical #1)
**Статус:** Новое
**Ответ:** Оставить статичную детекцию для v1. `-hwaccels`/`-encoders` покрывает 90% случаев. Для edge-case есть `VIDEO_HW_ACCEL=cpu`.
**Действие:** Нет изменений. Документированное ограничение.

---

### CRITICAL-2: CPU fallback не проверяет наличие энкодера

> При fallback на CPU код не проверяет, есть ли `libx264`/`libx265`/`libsvtav1` в FFmpeg. Startup OK, но job падает.

**Источник:** Codex (Critical #2)
**Статус:** Новое
**Ответ:** Добавить проверку CPU encoder при startup
**Действие:** `detect_hw_accel()` обновлён — после fallback на CPU вызывает `_has_encoder()` для configured codec. При отсутствии — `RuntimeError` с пояснением.

---

### CRITICAL-4: Баги в тестовых сниппетах плана

> Нет импортов pytest/ValidationError, BytesIO.close нельзя assert, mock encoder без poll.return_value.

**Источник:** Codex (Critical #4)
**Статус:** Новое
**Ответ:** Тесты как guidance — сниппеты в плане это направление, имплементатор адаптирует при реализации.
**Действие:** Нет изменений в плане. Имплементатор исправит при реализации.

---

### CONCERN-1: Нет `-pix_fmt yuv420p` в encoder command

> Без `-pix_fmt yuv420p` FFmpeg может выбрать yuv444p при bgr24 input, что вызовет проблемы воспроизведения на многих клиентах.

**Источник:** Codex (Concern #1)
**Статус:** Новое
**Ответ:** Добавить `-pix_fmt yuv420p`
**Действие:** Добавлен `-pix_fmt yuv420p` в `_cpu_encode_args()` и NVIDIA encode args. VAAPI уже использует `format=nv12`. Design doc обновлён с пояснением.

---

### CONCERN-3: cv2 metadata fallback может быть потерян

> План предлагает убрать cv2 "если нет других ссылок", но cv2 используется как fallback для metadata extraction.

**Источник:** Codex (Concern #3)
**Статус:** Новое
**Ответ:** Убрать cv2 fallback, ffprobe mandatory
**Действие:** ffprobe теперь mandatory (startup check). cv2 metadata fallback убирается. Это упрощает код. cv2 всё равно остаётся в зависимостях (YOLO).

---

### CONCERN-2: Task 6 worker test расплывчатый

> "Добавить комментарий/верифицировать" без конкретного assert на hw_config.

**Источник:** Codex (Concern #2)
**Статус:** Новое
**Ответ:** Оставить как guidance для имплементатора
**Действие:** Нет изменений.

---

### CONCERN-5: Memory churn от frame.tobytes()

> `frame.tobytes()` создаёт копию (~6MB/1080p) на каждый фрейм. Можно использовать `frame.data` для zero-copy.

**Источник:** Gemini (Concern #1)
**Статус:** Новое
**Ответ:** `tobytes()` для v1, оптимизация через `frame.data` — отдельный PR
**Действие:** Нет изменений.

---

### CONCERN-6: Audio всегда перекодируется в AAC

> `-c:a aac` hardcoded, даже если исходник уже AAC. `-c:a copy` эффективнее.

**Источник:** Gemini (Concern #2)
**Статус:** Новое
**Ответ:** `-c:a aac` для v1 — простота и гарантированная совместимость
**Действие:** Нет изменений.

---

### CONCERN-8: AMD VAAPI на старом FFmpeg

> `-hwaccel_device` может не работать на FFmpeg < 3.1.

**Источник:** CCS (Concern #1)
**Статус:** Новое
**Ответ:** OK для v1 — Ubuntu 24.04 имеет FFmpeg 6.x
**Действие:** Нет изменений.

---

### CONCERN-9: NVENC preset p4 захардкожен

> Не конфигурируемый preset.

**Источник:** CCS (Concern #2)
**Статус:** Новое
**Ответ:** p4 — хороший баланс для v1. Конфигурируемый preset — future enhancement.
**Действие:** Нет изменений.

---

### CONCERN-10: Mid-stream resolution changes

> FFmpegDecoder не детектирует смену разрешения в потоке.

**Источник:** CCS (Concern #3)
**Статус:** Новое
**Ответ:** Крайне редко в обычных файлах. OK для v1.
**Действие:** Нет изменений.

---

### CONCERN-11: Асимметричные таймауты

> Decoder `wait(timeout=10)` vs encoder `wait(timeout=300)`.

**Источник:** CCS (Concern #4)
**Статус:** Новое
**Ответ:** Обоснованная асимметрия — encoder финализирует (flush, trailers). OK для v1.
**Действие:** Нет изменений.

---

### SUGGESTION-13: PyAV как альтернатива subprocess

> PyAV (libav binding) для более надёжной интеграции.

**Источник:** CCS (Suggestion #6)
**Статус:** Новое
**Ответ:** Отвергнуто на этапе дизайна. Subprocess pipes — простой и надёжный подход.
**Действие:** Нет изменений.

---

### QUESTION-5: Синхронизация аудио с -map 1:a:0?

> Корректно ли синхронизируется аудио при медленной обработке?

**Источник:** Gemini (Question #1)
**Статус:** Новое
**Ответ:** Да, `-shortest` решает проблему. FFmpeg буферизует потоки. CFR output гарантирует корректный fps.
**Действие:** Нет изменений.

---

## Повторы (автоответ)

| # | Замечание | Источник повтора |
|---|-----------|-----------------|
| CONCERN-4 | Cached detect API surface | iter-1 C5 |
| SUGGESTION-1 | Active runtime probe | дубль CRITICAL-1 iter-2 |
| SUGGESTION-2 | CPU encoder validation | дубль CRITICAL-2 iter-2 |
| SUGGESTION-3 | -pix_fmt yuv420p | дубль CONCERN-1 iter-2 |
| QUESTION-1 | ffprobe mandatory? | дубль CRITICAL-3 iter-2 |
| QUESTION-2 | Target compatibility | дубль CONCERN-1 iter-2 |
| QUESTION-3 | GPU without runtime probe? | дубль CRITICAL-1 iter-2 |
| QUESTION-4 | Missing CPU encoder | дубль CRITICAL-2 iter-2 |
| CONCERN-7 | Pixel format padding | iter-1 I3 |
| SUGGESTION-5 | Restructure HWAccelConfig | дубль CRITICAL-5 iter-2 |
| SUGGESTION-6 | -init_hw_device syntax | опция CRITICAL-5 iter-2 |
| SUGGESTION-7 | Docker NVIDIA deps | iter-1 C3 |
| CRITICAL-6 | Frame size mismatch | iter-1 I3 |
| CONCERN-12 | VAAPI filter chain | iter-1 I6 |
| SUGGESTION-8 | Remove width/height | iter-1 I3 |
| SUGGESTION-9 | Validate first frame | iter-1 I3 |
| SUGGESTION-10 | Thread-safe stderr | дубль CRITICAL-8 iter-2 |
| SUGGESTION-11 | Configurable NVENC | дубль CONCERN-9 iter-2 |
| SUGGESTION-12 | Unify timeouts | дубль CONCERN-11 iter-2 |
| QUESTION-6 | Why width/height? | iter-1 I3 |
| QUESTION-7 | Validate first frame? | iter-1 I3 |
| QUESTION-8 | NVENC p4 sufficient? | дубль CONCERN-9 iter-2 |
| QUESTION-9 | Runtime fallback? | iter-1 I5 |

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| `docs/plans/2026-02-15-ffmpeg-hw-accel-design.md` | HWAccelConfig: добавлен `global_encode_args` property. FFmpegEncoder command: global args перед -i, codec args после inputs. Добавлен `-pix_fmt yuv420p`. Stderr: `deque[bytes]`. FFmpeg+ffprobe mandatory. CPU encoder validation. |
| `docs/plans/2026-02-15-ffmpeg-hw-accel-plan.md` | Task 2: `global_encode_args` property, VAAPI `-vaapi_device` перенесён, `-pix_fmt yuv420p` в CPU/NVIDIA args, CPU encoder validation в `detect_hw_accel()`. Task 3: `deque[bytes]` + `_format_stderr()`. Task 4: `deque[bytes]`, `global_encode_args` перед `-i`, `_format_stderr()`. Task 6: добавлена проверка `ffprobe`. |

## Статистика

- Всего замечаний: 41
- Новых: 18
- Повторов (автоответ): 23
- Пользователь сказал "стоп": Нет
- Агенты: codex-executor, gemini-executor, ccs-executor
