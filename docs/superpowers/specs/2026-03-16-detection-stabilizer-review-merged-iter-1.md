# Merged Design Review — Iteration 1

## codex-executor (gpt-5.4)

8 issues: 1 Critical, 5 Major, 2 Minor

- EDGE-2 (Critical): STABILIZER_MAX_CACHE_GB не является реальным hard limit — ffprobe может ошибаться
- ALGO-1 (Major): Неопределённое окно matching для track'ов — track'и матчатся через слишком длинные разрывы
- PERF-3 (Major): Полная materialization stabilized по всем кадрам расходует RAM
- IMPL-4 (Major): При ошибке большие временные файлы остаются до TTL cleanup
- API-5 (Major): Подсчёт tracked_frames в плане не соответствует новой семантике
- API-6 (Major): Семантика total_detections стала неоднозначной
- TEST-7 (Minor): Не хватает тестов на failure paths и wiring
- API-8 (Minor): Confidence у stabilized box может не соответствовать отображаемому class

---

## gemini-executor

6 issues: 2 Critical, 2 Major, 2 Minor

- PERF-1 (Critical): Raw frame disk caching causes massive I/O — предлагает two-pass decode вместо кэша
- ALGO-1 (Critical): Ghost tracks из-за неограниченного окна "памяти" (no staleness limit)
- ALGO-2 (Major): Backward extension нарушает непрерывность (перепрыгивает пустоты)
- ALGO-3 (Major): Ранняя фильтрация классов снижает эффективность стабилизации
- IMPL-1 (Minor): Риск переполнения max_det при заниженном пороге уверенности
- TEST-1 (Minor): Неполное тестовое покрытие backward extension

---

## ccs-executor glm (GLM-4.7)

20 issues: 2 Critical, 10 Major, 8 Minor

- ALGO-1 (Critical): Mutable unmatched_weak side-effect в backward extension
- IMPL-1 (Critical): Overlapping weak detections across tracks
- ALGO-2 (Major): Greedy matching использует только latest detection
- ALGO-3 (Major): Grace period зависит только от последней bbox
- PERF-1 (Major): O(frames*tracks) iteration в stabilize()
- EDGE-1 (Major): detect_every не кратен total_frames
- EDGE-2 (Major): fps=0 handling
- IMPL-2 (Major): Dict modification ordering
- IMPL-4 (Major): Stats counting mismatch
- TEST-1 (Major): Нет тестов для positive grace periods
- TEST-2 (Major): Нет тестов для multi-track competition
- и 8 Minor

---

## ccs-executor albb-glm (glm-5)

12 issues: 1 Critical, 2 Major, 9 Minor

- EDGE-2 (Critical): Race condition при workers > 1 — frame cache path не изолирован
- ALGO-1 (Major): Backward extension bbox drift under noise
- ALGO-2 (Major): Нет re-identification для длинных окклюзий
- 9 Minor (greedy matching, import paths, VFR edge case, etc.)

---

## ccs-executor albb-qwen (qwen3-coder-plus)

15 issues: 4 Critical, 9 Major, 2 Minor

- ALGO-1 (Critical): Backward extension logic
- ALGO-2 (Critical): Infinite loop risk in gap filling
- PERF-1 (Critical): Memory consumption with frame cache
- EDGE-1 (Critical): Disk space exhaustion handling
- и 11 Major/Minor (API toggle, FPS division by zero, cleanup, naming, tests)

---

## ccs-executor albb-kimi (kimi-k2.5)

12 issues: 2 Critical, 6 Major, 3 Minor, 1 Trivial

- ALGO-1 (Critical): unmatched_weak собирается в forward pass, backward extension невозможен
- PERF-1 (Critical): Concurrent jobs могут потребить 500GB суммарно
- ALGO-2 (Major): Weak detections создают noisy tracks
- ALGO-3 (Major): Backward grace period undefined
- PERF-2 (Major): Two-pass doubles decode time
- EDGE-1 (Major): No fallback to single-pass on ENOSPC
- EDGE-2 (Major): Unknown frame count handling
- IMPL-1 (Major): Missing AnnotationStats update

---

## ccs-executor albb-minimax (MiniMax-M2.5)

11 issues: 1 Critical, 4 Major, 6 Minor

- ALGO-1 (Critical): Backward extension пропускает высокоточные обнаружения без IoU
- PERF-2 (Major): Не указан тип исключения при ENOSPC
- IMPL-3 (Major): Нет атомарной обработки файлов между проходами
- API-4 (Major): План не обновляет все места создания VideoAnnotator
- EDGE-5 (Minor): Backward grace period может создать ложные bbox
- и 6 Minor
