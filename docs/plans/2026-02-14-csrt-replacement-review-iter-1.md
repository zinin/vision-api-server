# Review Iteration 1 — 2026-02-14 22:51

## Источник

- Design: `docs/plans/2026-02-14-csrt-replacement-design.md`
- Plan: `docs/plans/2026-02-14-csrt-replacement-plan.md`
- Review agents: codex-executor (gpt-5.3-codex, xhigh), gemini-executor (gemini-3-pro-preview), ccs-executor (glm-4.7, glmt)
- Merged output: `docs/plans/2026-02-14-csrt-replacement-review-merged-iter-1.md`

## Замечания

### #1 model.predict() thread-safety

> `model.predict()` не является "stateless" в shared-сценарии — переиспользует `self.predictor` и обновляет `args`.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Спорно — pre-existing issue, не введено этим PR. Существующий код уже вызывает predict() из воркера и API на общем ThreadPoolExecutor. Создать отдельную задачу на thread-safety, не блокирует этот PR.
**Действие:** Нет

---

### #2 Семантика tracked_frames

> Поле `tracked_frames` больше не отражает CSRT-трекинг, а означает "hold-reused frames". Нет docstring обновления.

**Источник:** codex-executor, gemini-executor, ccs-executor
**Статус:** Новое
**Ответ:** Справедливо
**Действие:** Добавлен docstring в AnnotationStats в плане (Task 1, Step 6a). Обновлена секция API Impact в дизайн-документе.

---

### #3 opencv-contrib → opencv-python-headless

> После удаления CSRT, contrib-пакет не нужен. Стандартный opencv-python-headless содержит все используемые модули.

**Источник:** codex-executor, gemini-executor
**Статус:** Новое
**Ответ:** Справедливо
**Действие:** Добавлен шаг замены в плане (Task 1, Step 8). Обновлена таблица Files to Modify и секция Design Decisions в дизайн-документе.

---

### #4 Визуальный лаг/залипание боксов

> Быстро движущиеся объекты покажут видимый лаг при detect_every=5.

**Источник:** gemini-executor
**Статус:** Новое
**Ответ:** Ложное срабатывание — это задокументированный trade-off дизайна. Hold mode выбран осознанно.
**Действие:** Нет

---

### #5 Оптимистичные fps-оценки

> processing_time_ms не включает merge_audio.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Ложное срабатывание — fps-оценки about frame processing, не e2e. processing_time_ms всегда измерял только frame loop.
**Действие:** Нет

---

### #6 Нет теста: пустые детекции на detection-кадре

> Новые тесты не проверяют сценарий: detection frame возвращает 0 объектов → hold-кадры тоже пустые.

**Источник:** codex-executor, gemini-executor
**Статус:** Новое
**Ответ:** Справедливо
**Действие:** Добавлен тест `test_hold_clears_on_empty_detection` в план (Task 3, Step 3). Обновлён expected test count: 72 (было 71).

---

### #7 Нет rollback-стратегии

> CSRT удаляется без feature-flag.

**Источник:** codex-executor
**Статус:** Новое
**Ответ:** Ложное срабатывание — feature branch с git history, rollback = git revert.
**Действие:** Нет

---

### #8 test_full_pipeline update не детализирован

> План говорит "Update" но не указывает как.

**Источник:** ccs-executor
**Статус:** Новое
**Ответ:** Ложное срабатывание — Plan Task 2 Step 5 явно описывает: удалить mock_tracker, убрать patch _create_csrt_tracker.
**Действие:** Нет

---

### #9-12 Minor issues

**#9** tracked_frames alias (codex) — ложное срабатывание, alias = API change.
**#10** Unused imports (gemini) — ложное срабатывание, уже в плане.
**#11** Hold mode logging (ccs) — спорно, добавляет код вместо удаления.
**#12** Update docstrings (gemini, ccs) — ложное срабатывание, уже в плане Step 6.

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| `docs/plans/2026-02-14-csrt-replacement-design.md` | Добавлена секция Design Decision #5 (opencv-contrib replacement), обновлена таблица Files to Modify, расширена секция API Impact |
| `docs/plans/2026-02-14-csrt-replacement-plan.md` | Task 1: добавлены Step 6a (AnnotationStats docstring) и Step 8 (requirements.txt). Task 3: добавлен Step 3 (empty detections test). Task 4: test count 71→72 |

## Статистика

- Всего замечаний: 12
- Новых: 12
- Повторов (автоответ): 0
- Справедливых: 3
- Ложных срабатываний: 7
- Спорных: 2
- Пользователь сказал "стоп": Нет
- Агенты: codex-executor (gpt-5.3-codex), gemini-executor (gemini-3-pro-preview), ccs-executor (glm-4.7)
