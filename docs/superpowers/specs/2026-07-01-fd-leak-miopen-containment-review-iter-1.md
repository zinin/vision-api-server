# Review Iteration 1 — 2026-07-02 00:30

## Источник

- Design: `docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md`
- Plan: `docs/superpowers/plans/2026-07-01-fd-leak-miopen-containment.md`
- Review agents: claude-self (fable), codex (gpt-5.5, reasoning xhigh), ext-claude alibaba/qwen (Qwen 3.7 Plus), ext-claude ollama/minimax (MiniMax M3). Отказали (стрим оборвался без результата): zai/glm, deepseek/v4-pro, ollama/kimi.
- Merged output: `docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-review-merged-iter-1.md`
- Дедупликация: 68 сырых пунктов → 36 уникальных позиций (агент claude-mesh:review-discussion перепроверял спорные факты по репозиторию).

## Замечания

### [CRITICAL-1] Допущение «cache hit не ликает» не проверено (upstream указывает на hipModuleLoad)

> Upstream MIOpen #2223 описывает лик в hipModuleLoad/hipModuleUnload, а не в компиляции. Если ликает load-path, тёплый кэш не устраняет утечку при рестарте (~1–2k FD на lifetime процесса, но уже не `(deleted)`), а rollout-критерий «(deleted) → ~0» даст ложноположительный успех.

**Источник:** claude-self (#1, Q15; смежный Q16 про внутренний потолок HIP-модулей)
**Статус:** Обсуждено с пользователем
**Ответ:** Вариант B — без пре-деплойного эксперимента; допущение зафиксировано в спеке как открытое, rollout-мониторинг расширен на общий FD-рост (не только deleted); прод сам даст ответ на фактической версии ROCm. Q16 зафиксирован как known-unknown (сигнатура: «shared object initialization failed» при FD far below limit).
**Действие:** Спека §2 — абзац «Open assumption»; Rollout шаг 4 — двухосевой критерий; план Post-merge шаг 4 — команды для обеих осей.

---

### [CRITICAL-2] `/health` отдаёт 500 при EMFILE до вывода FD-статистики

> `_fd_stats()` вызывался после `VideoFrameExtractor()`, чей `_verify_ffmpeg` не ловит `OSError` (Errno 24) — в терминальной фазе лика эндпоинт падает 500 и новые поля не попадают ни в ответ, ни в лог (ровно прод-симптом).

**Источник:** codex (#1), claude-self (#5), qwen (#3, #8), частично minimax (#1)
**Статус:** Автоисправлено
**Ответ:** `_fd_stats()` — первой строкой `health()`; ffmpeg-проба ловит `(RuntimeError, OSError)`; тест EMFILE-пути (`patch VideoFrameExtractor, side_effect=OSError(24)` → 200 + поля).
**Действие:** Спека §3 (ordering requirement + sketch); план Task 1 Step 2 (тест), Step 4 (код).

---

### [CRITICAL-3] Task 1 Step 1 описывает несуществующее состояние (~71 uncommitted строка)

> `tests/test_endpoints.py` полностью чист; безусловный `git checkout --` при ложной предпосылке опасен и сбивает агента-исполнителя.

**Источник:** все четыре ревьюера
**Статус:** Автоисправлено
**Ответ:** Шаг переписан как верификация («убедиться, что файл чист; если грязный — сначала `git diff`, стирать только stub»).
**Действие:** План Task 1 Step 1 + Architecture-абзац; спека §5 Housekeeping.

---

### [CRITICAL-4] Пререквизит плана ложен: `.venv` существует, но битая

> `.venv/bin/python → /usr/bin/python3.13`, которого нет (системный 3.12); `source activate` проходит, первый `pytest` умирает.

**Источник:** codex (#4); подтверждено review-discussion
**Статус:** Автоисправлено
**Ответ:** Prerequisites дополнены health-check'ом (`python --version`) и инструкцией пересборки venv.
**Действие:** План Prerequisites.

---

### [CONCERN-1] Goal 1 переобещает; warm-up ~1000 FD — не верхняя граница; дисковое давление замолчано

**Источник:** codex (#2, Q2), claude-self (#4)
**Статус:** Автоисправлено
**Ответ:** Goal 1 смягчён («push out of practical reach»); §1 дополнен worst-case оценкой (adversarial imgsz-sweep — десятки тысяч компиляций, порядок самого потолка) и дисковым эффектом (997 FD ≈ 1.7 GB writable layer).
**Действие:** Спека Goals + §1.

---

### [CONCERN-2] `import resource` на уровне модуля — Unix-only, Windows ломает импорт `main.py`

**Источник:** qwen (#1), codex, claude-self (#7)
**Статус:** Автоисправлено
**Ответ:** Guarded import (`try/except ImportError: resource = None`), `_fd_stats()` возвращает `(None, None, 0)` без модуля. QUESTION-1 (нужна ли Windows-поддержка) снят этим фиксом.
**Действие:** Спека §3; план Task 1 Step 4.

---

### [CONCERN-3] Механика включения `MIOPEN_FIND_MODE` вводит в заблуждение (выглядит как .env-переменная)

**Источник:** codex, claude-self (Q17)
**Статус:** Автоисправлено
**Ответ:** Явные формулировки «включается раскомментированием строки в docker-compose-amd.yml, НЕ через .env»; в спеку внесено обоснование отказа от `${MIOPEN_FIND_MODE:-}`-прокидки (пустая env → неизвестный парсинг MIOpen).
**Действие:** Спека §4; план Task 3 Steps 2–3 (CLAUDE.md-строка, .env.example).

---

### [CONCERN-4] Цена `MIOPEN_FIND_MODE=FAST` не оцифрована

**Источник:** minimax (#7)
**Статус:** Автоисправлено
**Ответ:** Ссылка на MIOpen "Find modes" docs + качественная оценка (heuristics вместо exhaustive tuning; не бенчмаркалось).
**Действие:** Спека §4.

---

### [CONCERN-5] Операционный lifecycle volumes не описан (down -v, размер, шаринг, апгрейд)

**Источник:** qwen (#6, Q13, Q14), minimax (#8, Q17)
**Статус:** Автоисправлено
**Ответ:** Блок «Volume lifecycle / limitations» в §2: предупреждение про `down -v`, MB-scale размер и накопление версий, single-container assumption (lock contention), `volume rm` при мажорном апгрейде.
**Действие:** Спека §2; план Post-merge шаг 1.

---

### [CONCERN-6] Спам WARNING после порога (~2880/сутки) + пассивность сигнала

**Источник:** claude-self (#6, #8), minimax (S11), qwen (Q12)
**Статус:** Автоисправлено
**Ответ:** Rate-limit ≤1/час (module-level `time.monotonic()` stamp, тест на второй хит); готовая curl+jq команда в rollout/post-merge как ручная проба.
**Действие:** Спека §3 + Rollout; план Task 1 (код+тест) + Post-merge.

---

### [CONCERN-7] Битая ссылка на первичную диагностику (`.superpowers/sdd/diagnosis-findings.md`)

**Источник:** claude-self (#3)
**Статус:** Автоисправлено
**Ответ:** Ссылка заменена признанием: рабочий файл не был закоммичен, раздел Problem — консолидированная сохранившаяся запись.
**Действие:** Спека Problem.

---

### [CONCERN-8] Compose-валидация одноразовая — рефактор молча снесёт ulimits

**Источник:** claude-self (S9, S10), codex
**Статус:** Автоисправлено
**Ответ:** Одноразовый heredoc-скрипт заменён постоянным `tests/test_compose.py` (9 параметризованных тестов; `pytest.importorskip("yaml")`). `docker compose config -q` не добавлен (docker на dev-машинах не гарантирован; YAML-инварианты покрывают регрессию containment).
**Действие:** План Task 2 Step 4 (полный файл теста), Step 5 (git add); спека Testing + Files touched.

---

### [CONCERN-9] Пробелы тест-покрытия `/health` (EMFILE-путь, None-ветка)

**Источник:** codex, claude-self (S11), minimax (#9)
**Статус:** Автоисправлено
**Ответ:** +тесты: `test_health_survives_emfile_in_ffmpeg_check`, `test_health_handles_missing_procfs`, `test_health_warning_rate_limited`, `test_health_passes_deleted_count_through` (итого 7 в классе).
**Действие:** План Task 1 Step 2; спека Testing.

---

### [CONCERN-10] Выбор 65536 не обоснован через peak-usage приложения

**Источник:** minimax (#5)
**Статус:** Автоисправлено
**Ответ:** Абзац «Sizing rationale» в §1: легитимный пик — сотни FD, потолок на ~2 порядка выше, маскировку компенсирует 80%-warning.
**Действие:** Спека §1.

---

### [SUGGESTION-2] Счётчик deleted-FD (`fd_deleted`) в `/health`

**Источник:** claude-self (S12), minimax (S13), codex (Q3)
**Статус:** Обсуждено с пользователем
**Ответ:** Принято (Вариант A): `_fd_stats()` → тройка `(open_fds, fd_deleted, soft_limit)`; поле в ответе; различает compile-path vs load-path лик удалённо — операционализирует выбранный в CRITICAL-1 мониторинг.
**Действие:** Спека §3 (поля, sketch, Testing, Rollout); план Task 1 (Interfaces, тесты, код), Task 3 (api.md), Post-merge.

---

### [SUGGESTION-5] Опциональный ручной restart после стабилизации кэша

**Источник:** codex
**Статус:** Автоисправлено
**Ответ:** Строка в Rollout/Post-merge (один осознанный restart обнуляет warm-up-утечку, кэш сохраняется; не autoheal).
**Действие:** Спека Rollout шаг 5; план Post-merge шаг 5.

---

### [SUGGESTION-6] Напоминание о PR-workflow (git rm docs/superpowers/)

**Источник:** claude-self (S13)
**Статус:** Автоисправлено
**Ответ:** Раздел «Before opening the PR» в конце плана с командами.
**Действие:** План (новый раздел).

---

### [SUGGESTION-8] Микро-уточнения api.md (snapshot, host-dependent значения)

**Источник:** qwen (#3, #5)
**Статус:** Автоисправлено
**Действие:** План Task 3 Step 1 (текст примечания к /health).

---

### [SUGGESTION-10] Робастность плановых инструкций (assert единственности restart-строки, logger-константа, место импорта)

**Источник:** minimax (S12, S15), qwen (#7)
**Статус:** Автоисправлено
**Действие:** План Task 2 Step 1 (grep -c проверка), Task 1 Step 2 (MAIN_LOGGER константа + комментарий почему «main»), Step 4 (примечание про место guarded import и существующий `import time`).

---

### [QUESTION-2] Варьируют ли клиенты imgsz → нужна ли нормализация?

**Источник:** codex (Q2)
**Статус:** Обсуждено с пользователем
**Ответ:** Трафик фиксированный — активная защита (allowlist/нормализация) не нужна; worst-case остаётся задокументированным в §1.
**Действие:** Нет (документы уже соответствуют).

---

### [QUESTION-4] `/tmp` прод-контейнера — writable layer или tmpfs?

**Источник:** claude-self (Q18)
**Статус:** Авто-разрешено проверкой
**Ответ:** `grep -r tmpfs docker/` пуст → tmpfs не настроен, `/tmp` в writable layer (overlayfs); формулировка §1 про дисковое давление уже корректна.
**Действие:** Нет.

---

### [QUESTION-5] Скрипт `scripts/verify_fd_containment.sh` для post-merge проверок?

**Источник:** minimax (Q16)
**Статус:** Авто-решено после анализа (отклонено)
**Ответ:** Не добавлять: проверки — четыре однострочника, приведённые в плане дословно, выполняются один раз; постоянный мониторинг покрыт `curl /health | jq`. Скрипт — лишняя сущность, протухнет при смене имён контейнера/порта.

---

### [QUESTION-6] Градация порогов (50% INFO / 80% WARNING / 95% ERROR)?

**Источник:** minimax (Q18)
**Статус:** Авто-решено после анализа (отклонено)
**Ответ:** Один порог WARNING 80% достаточен: при 65536 и ~52 FD/день порог достигается через ~2.7 года — форы месяцы; INFO/ERROR-уровни не добавляют actionable-сигнала, но плодят ветки кода и тесты.

---

### [SUGGESTION-7] Переименовать ветку `fix/upload-fd-leak` → `fix/fd-leak-containment`

**Источник:** claude-self (S14)
**Статус:** Обсуждено с пользователем
**Ответ:** Оставить как есть; честное имя получит сам PR (имя ветки эфемерно).
**Действие:** Нет.

---

### [FP-1] «ultralytics rect=True — технически неверно, в default.yaml rect: False» (minimax Critical 2)

**Статус:** Отклонено (ложное срабатывание)
**Ответ:** Опровергнуто проверкой кода ultralytics 8.4.14: `Model.predict()` задаёт `rect=True` как **method default** (`engine/model.py`, `custom = {..., "rect": True}`), перекрывая cfg-default (тот про train/val). Приложение вызывает `model.predict()` → rect=True → aspect-ratio buckets. Дизайн прав.
**Действие (прививка):** Уточнение «method default, not cfg default» внесено в Problem-раздел спеки, чтобы будущие ревьюеры не спотыкались (учтено как авто-фикс).

---

### [FP-2] «caplog + basicConfig не пересекутся при LOG_LEVEL=ERROR» (minimax Critical 3)

**Статус:** Отклонено (ложное срабатывание)
**Ответ:** `caplog.at_level(..., logger="main")` явно выставляет уровень логгера на время блока; propagated-записи не фильтруются уровнем root-логгера (только уровнями handlers). Тест устойчив к LOG_LEVEL. Подтверждено двумя ревьюерами независимо.

---

### [FP-3] «healthcheck сам накапливает ~14 400 FD/день» (minimax Critical 1, часть)

**Статус:** Отклонено (ложное срабатывание)
**Ответ:** Pipe-FD `subprocess.run` транзиентны и закрываются по завершении вызова — накопления нет. Валидные зёрна пункта учтены: порядок замера — CRITICAL-2, дешёвая проверка — SUGGESTION-3 (отклонена).

---

### [SUGGESTION-1] Поле `fd_usage_pct` / `fd_warning_threshold` в `/health`

**Источник:** codex, qwen (S10)
**Статус:** Отклонено
**Ответ:** Производные значения в API не кладём: ratio тривиально вычисляется из двух уже отдаваемых полей, порог применяется сервером в WARNING. Минимализм API.

---

### [SUGGESTION-3] Кешировать/удешевить ffmpeg-проверку в `/health`

**Источник:** minimax (S10), qwen (S8-смежное)
**Статус:** Отклонено
**Ответ:** После CRITICAL-2 (замер до пробы) transient-шум на замер не влияет; 2 коротких subprocess раз в 30 с — ничтожная нагрузка; кеш лишил бы health живой проверки ffmpeg. Мотивировавшее «накопление FD» — FP-3.

---

### [SUGGESTION-4] One-shot prewarm-контейнер для прогрева кэша

**Источник:** codex
**Статус:** Отклонено
**Ответ:** Volume-кэш даёт тот же эффект со второго старта без новой операционной сущности; prewarm экономит только самый первый прогрев. При load-path-лике (CRITICAL-1) prewarm вовсе бессилен. Противоречит принятому минимализму (no orchestration).

---

### [SUGGESTION-9] `len(os.listdir("/proc/self/fd"))` завышен на 1 (FD самого listdir)

**Источник:** qwen (#4)
**Статус:** Отклонено
**Ответ:** Погрешность +1 несущественна на любом реалистичном пороге (80% от 65536); вычитание добавляет ложную точность.

---

### [SUGGESTION-11] Мокать `resource.getrlimit`/`os.listdir` вместо `_fd_stats`

**Источник:** minimax (S14)
**Статус:** Отклонено
**Ответ:** Happy-path тест уже проходит через реальную `_fd_stats` (покрытие есть); патч собственного тонкого helper'а — нормальная практика, патчить stdlib-зависимости — хрупче.

---

### [QUESTION-1] Нужна ли поддержка Windows?

**Источник:** codex (Q1)
**Статус:** Отклонено (поглощено)
**Ответ:** Вопрос снят авто-фиксом CONCERN-2: guarded import делает код безопасным на Windows без отдельного решения о поддержке.

## Изменения в документах

| Файл | Изменение |
|------|-----------|
| `specs/...-design.md` | Problem: ссылка на findings заменена, rect=True уточнён как method default; Goals: Goal 1 смягчён; §1: sizing rationale + adversarial worst case + диск; §2: Open assumption (hipModuleLoad) + Volume lifecycle; §3: три поля (`open_fds`/`fd_deleted`/`fd_soft_limit`), guarded import, ordering requirement, rate-limit, новый sketch; §4: цена FAST + механика включения (не .env); §5: revert→verify; Testing: 7 тестов + tests/test_compose.py; Rollout: 7 шагов (curl-проба, first-start паттерн, двухосевой мониторинг, ручной restart, rollback, down -v); Files touched обновлён |
| `plans/...-containment.md` | Architecture и File Structure синхронизированы; Prerequisites: venv health-check; Task 1: Step 1 verify-only, Step 2 — 7 тестов + MAIN_LOGGER, Step 3 — red-ожидания ×7, Step 4 — guarded import + тройка `_fd_stats` + rate-limit + OSError в пробе, Step 5 — 7 PASSED; Task 2: grep-проверка restart-строки, Step 4 — постоянный tests/test_compose.py, Step 5 — git add теста; Task 3: api.md с fd_deleted и примечаниями, CLAUDE.md/.env.example — механика включения FIND_MODE; Post-merge: 7 шагов (обе оси, rollback, restart, down -v); новый раздел «Before opening the PR» |

## Статистика

- Всего замечаний: 36 (дедупликация 68 сырых от 4 ревьюеров)
- Автоисправлено (без обсуждения): 20
- Авто-применено после анализа: 3 (QUESTION-4 разрешён проверкой, QUESTION-5 и QUESTION-6 отклонены с обоснованием)
- Обсуждено с пользователем: 4 (CRITICAL-1 → мониторинг вместо эксперимента; SUGGESTION-2 → принят; QUESTION-2 → не менять; SUGGESTION-7 → не переименовывать)
- Отклонено: 9 (FP-1, FP-2, FP-3, SUGGESTION-1, SUGGESTION-3, SUGGESTION-4, SUGGESTION-9, SUGGESTION-11, QUESTION-1)
- Повторов (автоответ): 0 (первая итерация)
- Пользователь сказал «стоп»: Нет
- Агенты: claude-self, codex (gpt-5.5 xhigh), alibaba/qwen, ollama/minimax (7 запрошено, 3 отказа: zai/glm, deepseek/v4-pro, ollama/kimi)
