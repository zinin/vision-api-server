# Review Iteration 1 — 2026-02-15

## Источник

- Design: `docs/plans/2026-02-15-source-codec-matching-design.md`
- Plan: `docs/plans/2026-02-15-source-codec-matching-plan.md`
- Review agents: codex-executor (gpt-5.3-codex), gemini-executor (gemini-3-pro-preview), ccs-executor (glmt/glm-4.7)
- Merged output: `docs/plans/2026-02-15-source-codec-matching-review-merged-iter-1.md`

## Замечания

### #1 Breaking change: VIDEO_CODEC=auto дефолт

> Изменение дефолта с h264 на auto — breaking change. Клиент, загрузивший HEVC, раньше получал H.264, теперь получит HEVC.

**Источник:** Codex (Critical), Gemini (Important), CCS (Critical)
**Статус:** Новое
**Ответ:** Оставить auto. Осознанное решение из brainstorming. Сервер внутренний. Migration note (#14) покрывает тех, кому нужен h264.
**Действие:** Нет изменений.

---

### #2 Битрейт: нет валидации (0, negative, huge)

> bit_rate из ffprobe может быть 0, отрицательным или гигантским, что даст некорректный encode или DoS.

**Источник:** Codex (Critical), CCS (Critical)
**Статус:** Новое
**Ответ:** Справедливо. Добавить MIN_BITRATE=100_000, MAX_BITRATE=200_000_000. Вне диапазона → None → fallback CRF.
**Действие:** Обновлён план Task 4 и дизайн.

---

### #3 NVENC/VAAPI: -b:v без -maxrate/-bufsize

> -b:v без -maxrate/-bufsize может давать непредсказуемое поведение hw энкодеров.

**Источник:** CCS (Critical), Gemini (Minor)
**Статус:** Новое
**Ответ:** Отложить. Допустимо для первой итерации. NVENC defaults to VBR при -b:v.
**Действие:** Нет изменений.

---

### #4 Startup: проверяется только h264 encoder

> При auto могут понадобиться h265/av1, но они не валидированы при старте.

**Источник:** Codex (Critical), CCS (Important)
**Статус:** Новое
**Ответ:** Оставить как есть. get_encode_args() уже имеет fallback на CPU encoder для недоступных GPU кодеков.
**Действие:** Нет изменений.

---

### #5 CRF: "ignored in auto" vs self.crf в fallback

> Дизайн говорит "CRF ignored in auto", но план использует self.crf при отсутствии битрейта.

**Источник:** Codex (Important)
**Статус:** Новое
**Ответ:** Дизайн правильный. В auto mode CRF всегда 18 (hardcoded). Исправить план.
**Действие:** Обновлён план Task 4: effective_crf = 18 вместо self.crf.

---

### #6 Конструктор: дизайн говорит codec="auto", решение — "h264"

> Противоречие между design component 4 и Key Decision #4.

**Источник:** Codex (Important)
**Статус:** Новое
**Ответ:** Ложное срабатывание. План корректно использует "h264".
**Действие:** Нет изменений.

---

### #7 ffprobe: empty streams, broken r_frame_rate

> data["streams"][0] упадёт на пустом streams, r_frame_rate может быть не num/den.

**Источник:** Codex (Important)
**Статус:** Новое
**Ответ:** Pre-existing issue, но несложно исправить заодно. Добавить проверку empty streams и try/except для r_frame_rate.
**Действие:** Обновлён план Task 4.

---

### #8 Нет теста для startup wiring

> Нет теста проверяющего что detect_hw_accel получает codec="h264" при auto.

**Источник:** Codex (Important)
**Статус:** Новое
**Ответ:** Справедливо. Добавить unit-тест в Task 5.
**Действие:** Обновлён план Task 5.

---

### #9 Fallback отбрасывает bitrate для unsupported кодеков

> vp9+8Mbps → h264+CRF18, хотя bitrate известен.

**Источник:** Codex (Important)
**Статус:** Новое
**Ответ:** Ложное срабатывание. Осознанное решение из brainstorming (битрейт vp9 ≠ битрейт h264).
**Действие:** Нет изменений.

---

### #10 Маппинг кодеков: avc1, hvc1 алиасы

> ffprobe может возвращать avc1, hvc1.

**Источник:** CCS (Important), Codex (Minor)
**Статус:** Новое
**Ответ:** Ложное срабатывание. ffprobe codec_name возвращает "h264", "hevc", "av1" (не avc1/hvc1).
**Действие:** Нет изменений.

---

### #11 Нет логирования resolved codec

> Нет logging для resolved codec/bitrate.

**Источник:** Codex (Minor), Gemini (Minor), CCS (Important)
**Статус:** Новое
**Ответ:** Ложное срабатывание. Уже в плане Task 4: logger.info(f"Auto codec: source=...").
**Действие:** Нет изменений.

---

### #12 CRF/bitrate mutual exclusion

> Нужна взаимоисключающая логика.

**Источник:** Gemini (Important)
**Статус:** Новое
**Ответ:** Ложное срабатывание. План использует if/else.
**Действие:** Нет изменений.

---

### #13 ffprobe bit_rate parsing (N/A, absent)

> bit_rate может быть N/A или отсутствовать.

**Источник:** Gemini (Important)
**Статус:** Новое
**Ответ:** Ложное срабатывание. Уже обработано в плане через try/except.
**Действие:** Нет изменений.

---

### #14 Migration note в CLAUDE.md

> Нет инструкции как вернуть старое поведение.

**Источник:** Codex (Minor)
**Статус:** Новое
**Ответ:** Справедливо. Добавить комментарий.
**Действие:** Обновлён план Task 6.

---

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| `docs/plans/2026-02-15-source-codec-matching-design.md` | Добавлена валидация битрейта (MIN/MAX_BITRATE) |
| `docs/plans/2026-02-15-source-codec-matching-plan.md` | Task 4: битрейт валидация, CRF=18 hardcoded, ffprobe hardening. Task 5: startup wiring тест. Task 6: migration note. |

## Статистика

- Всего замечаний: 14
- Новых: 14
- Повторов (автоответ): 0
- Пользователь сказал "стоп": Нет
- Агенты: codex-executor, gemini-executor, ccs-executor

### Решения

- Справедливо: 3 (#2, #8, #14) — все приняты
- Спорно: 5 (#1, #3, #4, #5, #7)
  - #1: оставить auto (осознанное решение)
  - #3: отложить (первая итерация)
  - #4: оставить (fallback в get_encode_args достаточен)
  - #5: исправить план (CRF=18 hardcoded в auto)
  - #7: исправить заодно (несложно)
- Ложное срабатывание: 6 (#6, #9, #10, #11, #12, #13)
