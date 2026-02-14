# Merged Design Review — Iteration 1

## codex-executor (gpt-5.3-codex, reasoning: xhigh)

### Критические проблемы

1. **Неверная предпосылка: `model.predict()` не является "stateless" в shared-сценарии**
Проблема: в дизайне указано, что `model.predict()` "stateless — no conflicts", но у Ultralytics это не так: `predict()` переиспользует `self.predictor` и обновляет его `args`; официально есть отдельный гайд про thread-safe inference.
Где: `docs/plans/2026-02-14-csrt-replacement-design.md:30`, `app/main.py:191`, `app/main.py:404`, `app/video_annotator.py:134`.
Почему важно: после перехода на hold частота `predict()` в worker резко вырастет, и гонки между `/detect` и worker станут более вероятными (смешение `conf/imgsz/max_det`, нестабильные результаты).
Предложенное исправление: перед внедрением добавить синхронизацию вызовов модели (lock на модель) или выделить отдельный инстанс модели для worker. Формулировку в дизайне изменить с "stateless" на "без tracker-state, но требует thread-safety контроля".
Ссылки: https://docs.ultralytics.com/guides/yolo-thread-safe-inference/

### Важные замечания

1. **`API Impact: None` сейчас не полностью корректно**
Проблема: смысл `tracked_frames` меняется, но контракт/документация всё ещё говорит "tracker only".
Где: `docs/plans/2026-02-14-csrt-replacement-design.md:65`, `app/models.py:173`.

2. **Оценка производительности выглядит оптимистичной для end-to-end**
Проблема: метрика `processing_time_ms` фиксируется до `_merge_audio()`, а финальный FFmpeg merge/transcode идёт после.
Где: `app/video_annotator.py:175`, `app/video_annotator.py:186`.

3. **Недостаточное покрытие тестами для hold edge-cases**
Проблема: новые тесты не проверяют очистку hold при пустой детекции на detection-кадре.

4. **В плане нет rollback-стратегии**
Проблема: CSRT удаляется полностью без безопасного отката.

### Незначительные замечания

1. **Неточная формулировка про зависимость `opencv-contrib-python-headless`**
2. **Название `tracked_frames` теперь вводит в заблуждение** — предложен alias `hold_frames`.

### Положительные моменты

1. Радикальное упрощение pipeline.
2. Сохранена внешняя форма API.
3. Явное удаление мёртвого кода.
4. Выбор `model.predict()` вместо `model.track()` логичен.

---

## gemini-executor (gemini-3-pro-preview)

### Критические проблемы
Не обнаружено.

### Важные замечания

1. **Визуальный эффект "залипания" (Lag/Jitter)** — при detect_every=5 на 30fps боксы обновляются раз в ~166ms. Быстро движущиеся объекты покажут видимый лаг.
2. **Семантика `tracked_frames`** — поле больше не отражает реальный трекинг.
3. **`opencv-contrib-python-headless` dependency** — стандартный `opencv-python-headless` (без contrib) уже содержит VideoCapture/VideoWriter.

### Незначительные замечания

1. Удаление неиспользуемых импортов.
2. Тест на пустые детекции на кадре 0.
3. Обновление docstrings.

### Положительные моменты

1. Thread Safety — правильное решение с `model.predict`.
2. KISS — Hold mode прагматичен.
3. Четкий план миграции.

---

## ccs-executor (GLM-4.7, glmt)

### Критические проблемы
Отсутствуют.

### Важные замечания

1. **`current_detections` initialization** — если первый кадр не detection frame (frame_num != 0), hold будет показывать 0 объектов. Но 0 % N == 0 всегда, так что non-issue.
2. **`test_full_pipeline` update not specified** — план говорит "Update" но не детализирует.
3. **`AnnotationStats.tracked_frames` semantics** — нет запланированного обновления docstring.

### Незначительные замечания

1. `current_detections` is read-only в hold mode — добавить комментарий.
2. `test_detect_every_1` — уточнить что это edge case.
3. Добавить периодическое debug-логирование для hold mode.

### Положительные моменты

1. Отличный бенчмарк.
2. Правильный выбор `model.predict()`.
3. Hold mode — лучшее решение.
4. API backward compatibility.
5. Покрытие тестами.
6. Производительность реалистична.
7. Читабельный код.
