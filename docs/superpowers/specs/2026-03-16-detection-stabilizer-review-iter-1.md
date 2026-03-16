# Review Iteration 1 — 2026-03-16

## Источник

- Design: `docs/superpowers/specs/2026-03-16-detection-stabilizer-design.md`
- Plan: `docs/superpowers/plans/2026-03-16-detection-stabilizer.md`
- Review agents: codex-executor (gpt-5.4), gemini-executor, ccs-executor (glm, albb-glm, albb-qwen, albb-kimi, albb-minimax)
- Merged output: `docs/superpowers/specs/2026-03-16-detection-stabilizer-review-merged-iter-1.md`

## Замечания

### [ALGO-STALENESS] Ghost tracks — нет ограничения на возраст трека

> Если объект ушёл на 10-й секунде, а на 5-й минуте в тех же координатах появился другой — они склеятся в один трек с интерполяцией через всё видео.

**Источник:** gemini, codex, albb-glm
**Статус:** Новое
**Ответ:** Добавить max_staleness параметр (STABILIZER_MAX_STALENESS_SEC, дефолт 5с). Трек исключается из matching если последняя детекция старше этого порога.
**Действие:** Добавлен `max_staleness_sec` в StabilizerConfig и описание staleness window в Step 1 алгоритма.

---

### [PERF-CACHE] Кэш сырых кадров на диск — 10 ГБ/мин

> Raw BGR24 frames на диск требует гигантских объёмов. 50 GB лимит ограничивает видео ~5 минутами.

**Источник:** gemini, codex, kimi, qwen, albb-minimax
**Статус:** Новое
**Ответ:** Two-pass decode (без кэша). Pass 1: FFmpeg decode + YOLO (кадры не сохраняются). Pass 2: FFmpeg decode заново + render. Нулевой диск.
**Действие:** Полностью переписана секция Frame Cache → Two-Pass Decode. Убран STABILIZER_MAX_CACHE_GB. Обновлена диаграмма pipeline.

---

### [ALGO-BACKWARD] Backward extension перепрыгивает пустоты + mutable unmatched_weak

> Алгоритм может привязать weak detections через разрывы. Мутация unmatched_weak создаёт недетерминистичность.

**Источник:** gemini, glm, kimi, albb-glm
**Статус:** Новое
**Ответ:** Строгий шаг назад по detect_every. Если на целевом кадре нет match — стоп. Deep copy unmatched_weak.
**Действие:** Переписана секция Backward extension в Step 3.

---

### [ALGO-CLASSFILTER] Ранняя фильтрация классов снижает эффективность

> Фильтр до стабилизатора убивает детекции "wrong class" до голосования, снижая качество.

**Источник:** gemini
**Статус:** Новое
**Ответ:** Фильтровать после стабилизации по stable_class, а не до.
**Действие:** Переписана секция Class Filtering. Обновлён pass 1 (все классы) и pass 2 (фильтр по stable_class).

---

### [API-STATS] Семантика total_detections и tracked_frames неоднозначна

> total_detections считает сырые YOLO hits, tracked_frames по detect_every, не по реальному происхождению.

**Источник:** codex, glm
**Статус:** Новое
**Ответ:** Пересчитать обе метрики по стабилизированным результатам.
**Действие:** Обновлена секция AnnotationStats: total_detections = сумма stabilized boxes, tracked_frames = non-detection frames с активными треками.

---

### [API-CONF] Confidence от другого класса

> confidence берётся от ближайшей детекции, но class от голосования — семантическое несоответствие.

**Источник:** codex
**Статус:** Новое
**Ответ:** confidence = max confidence среди детекций победившего класса в треке.
**Действие:** Обновлена секция Step 4 Generate StabilizedFrames.

---

### [MINOR-FIXES] Мелкие исправления

**Источник:** albb-glm, glm, albb-minimax
**Статус:** Новое
**Ответ:** Применить все.
**Действие:**
- workers=1 ограничение задокументировано
- id(track)→track_id — будет исправлено в плане
- Batch YOLO inference добавлен в Out of Scope
- Track merging: добавлено предупреждение о визуальных артефактах

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| design spec | Frame Cache → Two-Pass Decode (без кэша на диск) |
| design spec | Добавлен max_staleness_sec в StabilizerConfig |
| design spec | Step 1: staleness window для active tracks |
| design spec | Step 3: строгий step-back в backward extension + deep copy |
| design spec | Class Filtering: после стабилизации, не до |
| design spec | Step 4: confidence = max conf победившего класса |
| design spec | AnnotationStats: обе метрики по stabilized output |
| design spec | Design Decisions: обновлено |
| design spec | Out of Scope: batch inference, track merging artifacts |
| design spec | Убран STABILIZER_MAX_CACHE_GB, добавлен STABILIZER_MAX_STALENESS |
| plan | НЕ обновлён в этой итерации (требует отдельного прохода) |

## Статистика

- Всего замечаний: 7 (6 дискуссионных + 1 пакет мелких)
- Новых: 7
- Повторов (автоответ): 0
- Пользователь сказал "стоп": Нет
- Агенты: codex-executor, gemini-executor, ccs-executor (glm, albb-glm, albb-qwen, albb-kimi, albb-minimax)
