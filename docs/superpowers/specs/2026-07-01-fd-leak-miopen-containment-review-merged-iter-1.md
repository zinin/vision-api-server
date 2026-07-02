# Merged Design Review — Iteration 1

- Design: `docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md`
- Plan: `docs/superpowers/plans/2026-07-01-fd-leak-miopen-containment.md`
- Date: 2026-07-01 (запуск ревью 23:41–23:43)
- Reviewers requested: claude-self, codex (gpt-5.5, xhigh), zai/glm, alibaba/qwen, deepseek/v4-pro, ollama/kimi, ollama/minimax
- Completed: 4 of 7 (см. Failed reviewers в конце)

---

## codex-executor (gpt-5.5, reasoning xhigh)

### Critical Issues

1. `/health` всё ещё может умереть до FD-статистики. В плане `_fd_stats()` вызывается после `VideoFrameExtractor()`, а тот на инициализации запускает `ffmpeg/ffprobe` через `subprocess.run` ([app/video_utils.py](/opt/github/zinin/vision-api-server/app/video_utils.py:60)). При `EMFILE` это даст `OSError`, который не ловится ни в `_verify_ffmpeg`, ни в `/health`. То есть новый сигнал может не появиться ровно в аварийном сценарии. FD-статистику нужно собирать до ffmpeg-проверки, а ffmpeg-проверка должна ловить `OSError`.

2. Цель "Prod can no longer reach FD exhaustion" переобещает. Документ сам фиксирует, что `imgsz` клиентский и диапазон 32–2016 ([design](/opt/github/zinin/vision-api-server/docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md:33)). Оценка "~1000 FDs one-time warm-up" не учитывает худший случай по множеству `imgsz × aspect buckets × models`. Persistent cache убирает повторный прогрев после recreate, но не предотвращает накопление новых leaked FDs в долгоживущем процессе.

3. План содержит опасный rollback: `git checkout -- tests/test_endpoints.py` ([plan](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-07-01-fd-leak-miopen-containment.md:60)). Сейчас файл чистый, а утверждение про "~71 uncommitted lines" уже не соответствует рабочему дереву. Такой шаг может стереть чужие изменения. Нужно заменить на "проверить diff и удалить только конкретный superseded stub".

4. Предусловие про `.venv` неверно для текущего checkout. `.venv/bin/python` указывает на отсутствующий `/usr/bin/python3.13`, при этом системный Python здесь 3.12.3. План говорит "если `.venv` отсутствует", но она присутствует и сломана ([plan](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-07-01-fd-leak-miopen-containment.md:23)). Нужна проверка работоспособности `.venv/bin/python --version` и явный rebuild venv при поломанной ссылке.

### Concerns

- `resource` Unix-only. Дизайн говорит про non-Linux dev machines, но `import resource` на Windows сломает импорт приложения раньше, чем `/proc` fallback сработает ([plan](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-07-01-fd-leak-miopen-containment.md:137)).

- Тесты `/health` не мокают `VideoFrameExtractor`, хотя проектный контекст говорит, что FFmpeg в тестах мокается. Это делает новые health-тесты зависимыми от локального ffmpeg и не покрывает `OSError: EMFILE`.

- `MIOPEN_FIND_MODE` документируется в `CLAUDE.md` как env-настройка, но compose её не пробрасывает, пока строка в AMD compose закомментирована. Это может создать ложное ожидание, что достаточно добавить `MIOPEN_FIND_MODE=FAST` в `.env`.

- Compose validation проверяет только YAML-структуру, но не `docker compose config` и не фактическую применимость `ulimits` Docker-движком ([plan](/opt/github/zinin/vision-api-server/docs/superpowers/plans/2026-07-01-fd-leak-miopen-containment.md:364)).

### Suggestions

- Перенести FD-сбор в начало `/health`, возвращать `open_fds` даже если ffmpeg-проверка падает, и добавить тест с `patch("main.VideoFrameExtractor", side_effect=OSError(24, "Too many open files"))`.

- Добавить отдельное поле `fd_usage_ratio` или хотя бы `fd_warning_threshold`, чтобы оператору не приходилось вычислять 80% лимита руками.

- Рассмотреть one-shot prewarm-контейнер/команду, которая шарит `miopen-cache` volumes, прогревает типовые формы и завершается. Тогда leaked FDs остаются в одноразовом процессе, а основной сервис стартует с тёплым cache и низким FD baseline.

- В rollout добавить опциональный ручной restart после стабилизации cache volume: cache сохранится, а leaked FDs текущего процесса обнулятся. Это не cron и не autoheal.

- Для `MIOPEN_FIND_MODE` либо явно писать "нужно редактировать compose", либо аккуратно сделать opt-in через compose env interpolation, если пустая переменная не меняет поведение MIOpen.

### Questions

- Нужно ли поддерживать запуск тестов/приложения на Windows, или достаточно Linux/macOS? От этого зависит, насколько строго чинить `resource`.

- Есть ли реальные клиенты, которые активно варьируют `imgsz`, или прод фактически живёт на одном-двух значениях? Это определяет, достаточно ли текущего `65536`, или нужен allowlist/нормализация `imgsz`.

- Хотим ли мы видеть именно `(deleted)` FD count в `/health`, а не только общий `open_fds`? Для этой аварии это был бы более точный диагностический сигнал.

---

## claude-self (fable, in-session subagent)

Ревью дизайна и плана fd-leak-miopen-containment (claude self-review, iter 1). Проверял утверждения документов против фактического кода (`app/main.py`, `tests/`, все 6 compose-файлов, `.claude/rules/api.md`, `.env.example`) и против upstream-треда MIOpen.

**Общий вердикт:** дизайн здравый, план исполним и необычно аккуратен (все old-блоки для правок сверены с фактическими файлами и совпадают; `patch("main._fd_stats")` корректен, потому что `tests/conftest.py` добавляет `app/` в `sys.path` и модуль импортируется как `main`; caplog сработает — `logging.basicConfig` + propagate по умолчанию; ожидания red-фазы в Task 1 Step 3 верны). Но есть одно непроверенное допущение в ядре дизайна и несколько рассинхронов с реальностью репозитория.

### Critical Issues

1. **Ключевое допущение §2 не проверено: «cache hit не ликает». Upstream-тред описывает лик не в компиляции, а в `hipModuleLoad`/`hipModuleUnload`** ([MIOpen #2223](https://github.com/ROCm/MIOpen/issues/2223), смежный [ROCm #2290](https://github.com/ROCm/ROCm/issues/2290)). Загрузка kernel в GPU нужна и при попадании в дисковый кэш — разница лишь в том, что при компиляции грузится O_TMPFILE (отсюда `(deleted)` в проде), а при hit — персистентный файл из `/root/.cache/miopen`. Если ликает сам load-path, то после каждого рестарта контейнера с тёплым кэшем процесс всё равно быстро утечёт ~1–2 тыс. FD (по числу уникальных модулей), только теперь FD будут указывать на живые файлы кэша, а не на `(deleted)`. Прод-данные это допущение ни подтверждают, ни опровергают: кэш там всегда был пуст (ephemeral layer), модели preloaded — наблюдений warm-hit просто не существует. Последствия, если допущение ложно: (а) заявленный эффект §2 («leak shrinks to a one-time warm-up… no longer restarted by container recreation») неверен — утечка возобновляется при каждом деплое, просто быстрее и в другом обличье; (б) rollout-критерий «`(deleted)` падает до ~0» даст ложноположительный успех, потому что считает только deleted. Containment при этом всё равно выдержит (~1–2k ≈ 3% от 65536 на жизнь процесса — ulimit спасает), но проверить надо до деплоя, а не через 2–3 дня прод-наблюдения. Проверка дешёвая: локальная репродукция уже есть (`.venv` + yolo26n на ROCm-машине) — прогнать скрипт, дождаться прогрева кэша, запустить второй процесс с тёплым `~/.cache/miopen` и сравнить прирост FD (всех, не только deleted). Заодно это уточнит и ценность аварийного рычага `MIOPEN_FIND_MODE=FAST` (при load-path-лике он тоже не помогает).

2. **Task 1 Step 1 плана описывает состояние, которого нет.** План утверждает: «The working tree has ~71 uncommitted lines in `tests/test_endpoints.py`». Фактически рабочее дерево чистое (`git status` по файлу пуст), стаба нет ни в дереве, ни в коммитах ветки (diff `master...fix/upload-fd-leak` — только 8 `.md`-файлов), grep по «fd/leak/EMFILE» в файле пуст. `git checkout --` идемпотентен и не сломается, но для агент-исполняемого плана ложная предпосылка — реальный источник замешательства («я не в том дереве? состояние уже испорчено?»). Переписать шаг как верификацию («убедиться, что файл чист; если нет — discard») и поправить §5 Housekeeping в дизайне.

### Concerns

3. **Битая ссылка на полную диагностику.** Дизайн ссылается на «full record: `.superpowers/sdd/diagnosis-findings.md`» — этого пути не существует ни в рабочем дереве, ни в истории ветки. С учётом workflow пользователя (`docs/superpowers/` вычищается перед PR и живёт только в истории ветки) первичная запись диагностики (997 FD, ~52/день, version sweep fastapi 0.128–0.139) рискует пропасть безвозвратно. Либо закоммитить findings рядом со спекой, либо убрать ссылку и признать, что раздел Problem — единственная сохранившаяся запись.

4. **Оценка warm-up «~1000 FD ≈ 1.5% лимита» — это органический трафик, не верхняя граница.** `imgsz` клиентский (32–2016 по `.claude/rules/api.md`, ultralytics округляет к кратным 32 → до ~63 значений) × aspect-buckets × 2+ моделей — разнообразный или адверсарный клиент может раздуть одноразовое исследование пространства шейпов до тысяч–десятков тысяч компиляций. 65536, скорее всего, хватит, но границу стоит проговорить в спеке. Смежное: каждый утёкший FD удерживает deleted-файл 1.4–2.0 MB на writable layer — 997 FD ≈ ~1.7 GB невидимо занятого диска; при агрессивном сканировании шейпов это заметное дисковое давление, о котором дизайн молчит.

5. **При реальном EMFILE `/health` отдаёт 500 до того, как fd-статистика попадёт в ответ.** `VideoFrameExtractor._verify_ffmpeg` (app/video_utils.py:62) ловит только `CalledProcessError`/`FileNotFoundError`; `OSError` Errno 24 из `subprocess.run` пролетает, а в `/health` (app/main.py:411-414) стоит `except RuntimeError` — итог: 500, поля `open_fds` в ответе нет именно в терминальной фазе. Это ровно прод-симптом. Не блокер (80%-warning сработает за месяцы до), но дешёвая правка — вызвать `_fd_stats()` **до** ffmpeg-проверки, чтобы WARNING гарантированно писался в лог даже когда сам эндпоинт падает.

6. **Лог-спам после пересечения порога.** Healthcheck дёргает `/health` каждые 30 с → ~2880 одинаковых WARNING в сутки, пока порог превышен. Для сигнала с горизонтом «месяцы» терпимо, но однострочный дедуп (логировать при пересечении порога или не чаще раза в час) снял бы шум.

7. **`import resource` на уровне модуля — Unix-only.** macOS покрыт (модуль есть, `/proc` нет → ветка `None` — в дизайне учтено), но на Windows упадёт сам импорт `main.py`, то есть всё приложение и все тесты. Стек проекта линуксовый, так что это скорее строка в доке, чем правка, — но формулировка дизайна «non-Linux dev machines» обещает больше, чем даёт.

8. **Пассивность сигнала (принятое решение, фиксирую следствие).** WARNING в docker logs и поле в `/health` без какого-либо алертинга — сигнал увидят, только если посмотрят. Решение «без autoheal/оркестрации» принято; тогда единственный систематический потребитель — человек. Одна строка в post-merge-разделе с готовой командой для периодической проверки (`curl -s :3001/health | jq .open_fds`) сделала бы «пассивную» наблюдаемость чуть более используемой.

### Suggestions

9. **Закоммитить постоянный тест compose-инвариантов** вместо одноразового скрипта в Task 2 Step 4 (тот же код, оформленный как `tests/test_compose.py`: 6 файлов, `ulimits.nofile` 65536, miopen-тома в AMD-паре). PyYAML доступен транзитивно через ultralytics (в `requirements*.txt` он не заявлен), docker для теста не нужен. Иначе следующий рефактор компоузов молча снесёт ulimits — и регрессию заметят через 19 дней по EMFILE.
10. **Дополнить валидацию `docker compose -f <file> config -q`** там, где docker доступен — PyYAML-скрипт проверяет значения, но не схему compose.
11. **Тест на ветку `open_fds is None`** (patch `(None, 1000)`: warning не логируется, в JSON `null`) — сейчас единственная нетривиальная ветка `_fd_stats` не покрыта; это 5 строк.
12. **Опционально: отдельное поле `deleted_fds`** (подсчёт по `os.readlink` в `/proc/self/fd`) — это точная сигнатура именно этого лика, в отличие от общего счётчика; при штатных ~100 FD стоимость ничтожна. С учётом пункта 1 полезно различать deleted- и не-deleted-рост. Понимаю аргумент простоты — на усмотрение.
13. **Добавить в конец плана напоминание про PR-workflow пользователя**: перед созданием PR `git rm` всего `docs/superpowers/` (правило из глобального CLAUDE.md). План агент-исполняемый, а этот шаг в нём не упомянут — исполнитель дойдёт до PR и забудет.
14. **Косметика: имя ветки `fix/upload-fd-leak` увековечивает опровергнутую гипотезу.** Переименование в духе `fix/fd-leak-containment` сделает PR честнее; решать пользователю.

### Questions

15. Проверялось ли в локальной репродукции поведение при тёплом **дисковом** кэше (новый процесс, кэш заполнен, компиляций нет)? Если да — это снимает пункт 1 и стоит зафиксировать в спеке одной строкой; если нет — см. пункт 1.
16. В пересказах upstream-треда упоминаются отказы `hipModuleLoad` («shared object initialization failed») **до** достижения ulimit — то есть у HIP, возможно, есть собственный внутренний потолок числа загруженных модулей, на который поднятие `nofile` не влияет. Прод-инцидент был классическим EMFILE на 1024, так что наблюдавшийся режим ulimit закрывает, — но проверялось ли, что при ~1–2k утёкших модулей HIP не упирается во внутренний лимит?
17. Блок в `.env.example` — чисто справочный: переменная не прокинута как `${MIOPEN_FIND_MODE:-}` в compose, включение рычага требует правки файла под git. Это осознанно? (Прокидка удобнее в аварии, но всегда задаёт пустую переменную окружения, и как MIOpen парсит пустую строку — неизвестно; текущее решение безопаснее. Стоит зафиксировать обоснование в спеке, чтобы следующий ревьюер не «улучшил».)
18. Куда смонтирован `/tmp` прод-контейнера — writable layer или tmpfs? Влияет только на то, что именно испытывает давление от удерживаемых deleted-файлов (диск против RAM, см. пункт 4) — для полноты доки.

Sources:
- [MIOpen #2223 — hipModuleLoad/hipModuleUnload is leaking file descriptor](https://github.com/ROCm/MIOpen/issues/2223)
- [ROCm #2290 — Shared Object Initialization Failed Exception in MIOpen](https://github.com/ROCm/ROCm/issues/2290)

---

## ext-claude-executor (alibaba/qwen — Qwen 3.7 Plus)

# Ревью дизайна и плана по сдерживанию MIOpen FD-утечки

Документы прочитаны, кодовая база изучена (compose-файлы, `app/main.py`, `tests/test_endpoints.py`, `app/video_utils.py`). Дизайн в целом крепкий: трёхуровневая защита (persistence → ulimit → observability), осторожный подход к `MIOPEN_FIND_MODE`, правильный отказ от смены геометрии инференса. Ниже — конкретные находки.

## Critical Issues

**1. `import resource` на уровне модуля сломает импорт на Windows.**
В плане (Task 1, Step 4) предлагается добавить `import resource` рядом с `import os` в `app/main.py`. Модуль `resource` — Unix-only; на Windows он отсутствует (`ImportError: No module named 'resource'`). Если разработчик запускает `pytest` локально на Windows (а такая dev-машина в проекте возможна), импорт `main.py` упадёт, и **вся тестовая база станет невозможна** — не только новые тесты, но и существующие.

В `api.md` прямо говорится: "`open_fds` counts `/proc/self/fd` entries (`null` where `/proc` is unavailable, e.g. non-Linux dev)". Дизайнерское намерение — сделать код работающим на non-Linux — но `resource.getrlimit` защищён не будет. Нужно либо:
- `try: import resource except ImportError: resource = None`, и в `_fd_stats()` возвращать `(None, 0)` при отсутствии модуля,
- либо перенести `import resource` внутрь `_fd_stats()` с тем же fallback.

**2. Шаг откатки stub'а описывает несуществующее действие.**
Task 1, Step 1: *"The working tree has ~71 uncommitted lines in `tests/test_endpoints.py` from the superseded plan… Discard them: `git checkout -- tests/test_endpoints.py`"*. Проверяю — `tests/test_endpoints.py` **полностью чист** (`git status` пуст, `git diff HEAD` пуст, последний коммит `c96690e`). Откатывать нечего. Step 1 превращается в no-op, но реализатор будет тратить время на поиск несуществующих 71 строк. Нужно просто удалить этот шаг или переформулировать как *"убедиться, что файл чист; если есть незакоммиченные изменения — откатить"*.

## Concerns

**3. `/health` временно раздувает FD-счётчик в момент замера.**
Текущая реализация `health()` на каждый вызов конструирует `VideoFrameExtractor()`, который внутри `_verify_ffmpeg()` делает **два `subprocess.run`** (`ffmpeg -version` и `ffprobe -version`). Каждый spawn создаёт ~6–12 временных FD (pipe-концы для stdin/stdout/stderr дочерних процессов). Новый `_fd_stats()` снимает мерку *после* этого, но при высокой нагрузке или совпадении timing'ов измерение может включать эти временные FD.

При пределе 65536 и threshold 80% (52428) это несущественно. Но стоит:
- упомянуть в api.md, что `open_fds` — моментальный снэпшот и может включать transient FD от самого healthcheck;
- в идеале — **поменять порядок**: сначала `_fd_stats()`, потом `VideoFrameExtractor()`, чтобы healthcheck-шум не влиял на собственное измерение.

**4. `len(os.listdir("/proc/self/fd"))` считает и FD самой директории.**
`listdir` открывает FD для `/proc/self/fd`, который живёт во время `len(...)`. Результат завышен ровно на 1. Мелочь (1 из 100–1000), но для точности можно вычитать 1, либо использовать `os.scandir` с явным закрытием. Опционально, но раз уж делаем observability — сделаем аккуратно.

**5. `resource.getrlimit` на macOS возвращает неинтуитивные значения.**
На macOS default soft limit часто 256 (или `RLIM_INFINITY` = -1 как большое беззнаковое). На Linux — 1024. Тест `test_health_reports_fd_stats` проверяет только `data["fd_soft_limit"] > 0`, что пройдёт везде, но разработчик на Mac увидит в `/health` непривычные числа. Это не ошибка, но в api.md стоит явно сказать: "значения зависят от хостового `RLIMIT_NOFILE`; в docker-контейнере — то, что выставлено в `ulimits`".

**6. `docker compose down -v` уничтожает MIOpen volumes.**
Флаг `-v` у `down` удаляет именованные volumes. Если оператор случайно дёрнет `docker compose down -v`, прогретый кеш MIOpen потеряется, и утечка начнётся заново (с одним разом на прогрев). Стоит упомянуть это в `CLAUDE.md` или rollout-секции: *"Не используйте `docker compose down -v` — это удалит MIOpen cache volumes"*. Одна строчка, но спасает от неожиданного отката.

**7. Тест `test_health_warns_when_fd_usage_high` проверяет `logger="main"`.**
Это правильно (модуль `app/main.py` импортируется как `main` в тестах), но хрупко: если кто-то переименует модуль или перенесёт `_fd_stats` в отдельный файл, тест молча перестанет ловить warning. Это мелочь, но стоит либо захардкодить имя логгера в тесте как константу, либо использовать `logger="app.main"`.

## Suggestions

**8. Порядок вычислений в `health()` — сначала `_fd_stats()`.**
Связано с п. 3. Если поставить `_fd_stats()` первой строкой, измерение будет чище:

```python
open_fds, fd_soft_limit = _fd_stats()   # снимок до healthcheck-шума
ffmpeg_available = True
try:
    VideoFrameExtractor()
except RuntimeError:
    ffmpeg_available = False
...
```

Это не меняет семантику для клиента, но делает `open_fds` воспроизводимее.

**9. Рассмотреть `psutil.Process().num_fds()` как альтернативу.**
Текущий подход (`os.listdir("/proc/self/fd")`) работает только на Linux. `psutil` кросс-платформенен. Но `psutil` — внешняя зависимость, и в проекте её нет. Текущий подход проще и достаточен для Linux-контейнера — не рекомендую менять, но стоит задокументировать, что это Linux-специфичный счётчик (в api.md уже сказано).

**10. Подумать о метрике `fd_usage_ratio` в ответе.**
Клиенту (и мониторингу) удобнее сразу видеть процент, чем считать `open_fds / fd_soft_limit` самостоятельно. Опциональное поле `fd_usage_pct: round(open_fds / fd_soft_limit * 100, 1)` упрощает Prometheus scrape и Grafana-алерты. Мелочь, но делает API self-describing.

**11. Альтернатива для volumes — `MIOPEN_CUSTOM_CACHE_DIR`.**
MIOpen поддерживает `MIOPEN_CUSTOM_CACHE_DIR` (env var), который переопределяет путь кеша вне `$HOME`. Это дало бы гибкость (можно класть кеш в `/data/miopen` вместо `/root/.cache/miopen`), но усложнило бы конфигурацию и потребовало бы отдельного env в compose. Текущий подход (bind mount стандартных путей) проще и предсказуемее — **не рекомендую менять**.

## Questions

**12. Как пользователь будет мониторить warning?**
Если есть внешний сбор логов (Loki, CloudWatch, journald с алёртами) — `WARNING`-записи достаточно. Если нет — warning в логах контейнера легко пропустить. Стоит ли добавить более агрессивную эскалацию (например, повторный warning каждые N секунд, пока FD не освободятся)? Или достаточно одного warning на вызов /health?

**13. Какой ожидаемый размер MIOpen cache volume?**
Дизайн говорит "MB scale", но конкретные цифры не приведены. Для планирования disk quota и бэкапов полезно знать: ~50 MB? ~500 MB? Несколько GB? Если у автора есть данные с прода после эксперимента с `du -sh /root/.cache/miopen`, стоит добавить в rollout-секцию.

**14. Что если cache volume расшарен между несколькими контейнерами?**
В prod-конфигурации один контейнер = один volume. Но если когда-нибудь будет N реплик, они будут concurrently писать в один MIOpen cache. MIOpen использует lock-файлы (упомянуты в плане: "find-db + lock files"), но это создаёт конкуренцию за inode. Сейчас это не проблема (single worker, single container), но стоит упомянуть ограничение.

## Итог

Дизайн зрелый и тщательно продуманный; критичных архитектурных изъянов нет. Два critical issue (Windows-совместимость `resource` и несуществующий stub) исправляются за 5 минут каждый. Остальные concerns — вопросы чистоты измерения и operational awareness, не блокирующие реализацию.

**Рекомендация:** исправить critical issues #1 и #2, учесть suggestion #9 (порядок вычислений), добавить примечание про `docker compose down -v` — и можно начинать реализацию.

---

## ext-claude-executor (ollama/minimax — MiniMax M3)

# Ревью дизайна и плана: ROCm/MIOpen FD-leak Containment

Я прочитал оба документа и изучил ключевые части кодовой базы (`app/main.py`, `app/inference_utils.py`, `app/video_utils.py`, `tests/test_endpoints.py`, `tests/conftest.py`, `docker/`), а также версии зависимостей (ultralytics 8.3+, `default.yaml` имеет `rect: False`). Ниже — критика по запрошенным разделам.

## Critical Issues

### 1. Архитектурное противоречие в `/health`: endpoint, который должен быть FD-пробой, сам пожирает FDs

Это **не упомянуто в дизайне** и **не исправлено планом**, но это главная причина, по которой 19-дневный leak удалось заметить только когда он стал критическим.

В `app/main.py:412` тело `/health` создаёт `VideoFrameExtractor()`, чей `__init__` (см. `app/video_utils.py:60-79`) вызывает `_verify_ffmpeg()`, который делает **два** `subprocess.run(["ffmpeg", "-version"], ...)` + `["ffprobe", "-version"]`. Compose healthcheck (строка 22 всех AMD/CPU/NVIDIA compose) бьёт `/health` каждые 30 секунд. То есть:

- Каждые 30 секунд — минимум 4 FDs на pipe'ы к двум subprocess'ам + пара FDs на fork/exec.
- При приближении к soft=1024 именно этот healthcheck **первым** ловит `EMFILE` (а не пользовательская нагрузка), помечает контейнер unhealthy — но в ту же секунду пытается породить ещё больше субпроцессов для следующего healthcheck'а.

**Прямое следствие для плана:** наблюдаемость через `/health` (дизайн §3) **надёжна ровно до тех пор, пока новый soft limit 65536 не сожмётся обратно к нулю** — а `VideoFrameExtractor` расходует FDs на каждый hit. Этот фоновый расход ~10 FDs/мин ≈ 14 400/день, что сопоставимо с leak rate (52/день), но в худшую сторону. Дизайн должен либо (a) закешировать результат ffmpeg-проверки (lifespan + module-level boolean), либо (b) заменить проверку на дешёвую (`shutil.which("ffmpeg") is not None`, без subprocess). Без этого FD budget при soft=1024 заполняется healthcheck'ом за ~100 минут — что согласуется с наблюдением "каждые 30 секунд".

### 2. Утверждение "ultralytics predict defaults to rect=True" — **технически неверно**

В `ultralytics/cfg/default.yaml:31` дефолт — `rect: False`. В `predictor.py:194-200` `auto=True` для `LetterBox` поднимается только при `same_shapes=True and self.args.rect=True and (model.pt or ...)`. В коде приложения (`inference_utils.py:86-93`) `rect` нигде не передаётся. То есть:

- Если `rect=False` (default), то входной shape **всегда квадратный** `(imgsz, imgsz)` — aspect ratio не должен давать разные shape buckets.
- Альтернативная гипотеза: источник разных shape — это **диапазон imgsz** (32–2016, шаг 32, ~62 значения) × imgsz из video/annotator пути (там есть ещё варианты). Это даёт сопоставимое количество buckets, но **другой** триггер.

**Прямое следствие для дизайна:** формулировка "16:9 → 576×1024, 4:3 → 768×1024, …" в §1 дизайна **противоречит дефолту `rect=False` в ultralytics**. Эмпирически триггер воспроизводится (на это ссылается текст), но механизм описан неточно. Если механизм — `imgsz` бакеты, а не aspect-ratio бакеты, то предсказание "shape space is bounded (~64 buckets per model per imgsz)" **неверно как формулировка** — должно быть "~64 buckets per model" (imgsz-пространство само и есть бакет-пространство). Это влияет на пост-merge валидацию: ожидаемый "warm-up leak ~1000 FDs" предполагает, что ~64 buckets × N слоёв/моделей = ~1000. Если реальное число другое, warm-up либо меньше, либо больше ожидаемого, и тревожный порог 80% от 65536 может быть отложенным.

### 3. План валит на нём же: `caplog.at_level(logging.WARNING, logger="main")` + `basicConfig` могут не пересечься

В `app/main.py:47-50` `logging.basicConfig(...)` настраивает **root logger** с level из env. В `app/main.py:54` `logger = logging.getLogger(__name__)` создаёт логгер `"main"`. В `caplog.at_level(logging.WARNING, logger="main")` уровень поднимается **только** на `"main"`, но basicConfig уже настроил root на `INFO` (или что в env). Если `LOG_LEVEL=ERROR` (для подавления шума), то `logger.warning` **не дойдёт** до root (WARNING < ERROR) и **caplog не поймает запись**.

Тест `test_health_warns_when_fd_usage_high` молча становится flaky при `LOG_LEVEL=ERROR`. Рекомендация: либо хардкодить уровень `logger.warning` через явный `logger.setLevel(logging.WARNING)`, либо в тесте использовать `caplog.set_level(logging.WARNING, logger="main")` (что эквивалентно), но явно проверить, что root logger не поднимет уровень выше. Эта зависимость от внешнего env не отражена в плане.

## Concerns

### 4. План "реверсит" stub в `tests/test_endpoints.py` командой `git checkout --` — но без верификации содержимого

Шаг 1 Task 1: `git checkout -- tests/test_endpoints.py` стирает ~71 строку неоткоммиченных правок. Шаг 2 не говорит, что делать, если в stub'е были **другие** изменения, не относящиеся к FD-leak. Инструкция была бы надёжнее, если бы явно требовала `git diff tests/test_endpoints.py` **до** checkout, а не после.

### 5. `ulimit` 65536 — необъяснённый выбор, без анализа альтернатив

Дизайн обосновывает 65536 арифметикой time-to-EMFILE (~3.4 года), но не через реальный peak FD-расход приложения под полной нагрузкой (10 одновременных видео-аннотаций, MAX_EXECUTOR_WORKERS=4, FFmpeg пайпы). 65536 — это на 3 порядка больше необходимого, что маскирует будущие баги (а не выявляет их).

### 6. `_fd_stats()` — `os.listdir("/proc/self/fd")` на каждом hit без rate-limit

При тысячах записей listdir начинает занимать заметное время; лишний syscall на каждом healthcheck. Кеширование или ограничение частоты в плане не предложено.

### 7. `MIOPEN_FIND_MODE` lever — нигде не объяснена реальная стоимость

Нет количественной оценки inference-performance penalty от FAST-mode и нет ссылки на документацию MIOpen, где этот mode описан. Без числа оператор не знает, какую цену платит, если его включит.

### 8. Volumes: lifecycle, рост, sharing

- Дизайн не упоминает требования к FS (read-only volume → MIOpen упадёт при первой записи).
- MIOpen не очищает старые версии из cache — за 3-5 ROCm-апгрейдов volume вырастет до нескольких GB. Нет инструкции по cleanup/rotation.
- `miopen-config` содержит lock files — если volume общий между контейнерами, возможен lock contention. Стоит зафиксировать в design limitations: "single container per host; do not share miopen-config between vision-api instances".

### 9. Отсутствие теста для случая `/proc/self/fd` недоступен (macOS/Windows dev machine)

Тест `test_health_reports_fd_stats` покрывает happy path, но не `open_fds=None` (нет procfs). Заявленное покрытие плана не полностью соответствует написанным тестам.

## Suggestions

### 10. Сделать проверку ffmpeg в `VideoFrameExtractor` дешёвой (объединить с #1)

Заменить два `subprocess.run(... "-version")` на `shutil.which(...)` — убирает ~4 FDs на каждый `/health` hit. Можно сделать в этом же плане (Task 1.5) либо отдельным коммитом.

### 11. Rate-limit на FD WARNING: раз в N секунд, не на каждом hit

При превышении порога каждый healthcheck будет логировать WARNING часами. Модульная переменная `_last_fd_warning_at` + rate-limit до 1 раза в 5 минут.

### 12. Self-review для замены `restart:` в compose — добавить assert на единственность строки

"Every file has exactly one line `restart: unless-stopped`" — проверено, так и есть во всех 6 файлах. Но стоило добавить в план явный assert: "если в файле более одной строки `restart:`, остановись".

### 13. Добавить `fd_leaked_estimate` в `/health` — считать именно (deleted) FDs

Только `open_fds`/`fd_soft_limit` не различают "всё растёт" vs "растут именно leaked MIOpen FDs". Подсчёт deleted-симлинков в `/proc/self/fd` работает в Python без subprocess.

### 14. Тест `test_health_no_warning_at_normal_usage` мокает саму `_fd_stats` — anti-pattern

Лучше патчить на уровне зависимостей (`resource.getrlimit`, `os.listdir`), иначе теряется coverage самой функции.

### 15. Self-review плана: мало type/format/style проверок

Не проверяется место `import resource` (сортировка), %-style у `logger.warning`, хрупкость assert `"fd usage" in caplog.text.lower()` к изменению формата.

## Questions

### 16. Где post-merge validation script?

Ручные команды (`docker exec ... ulimit -n`, `ls -l /proc/1/fd`) не автоматизированы. Стоит ли добавить `scripts/verify_fd_containment.sh`?

### 17. Что произойдёт при ROCm-апгрейде base image с сохранённым find-db?

Binary cache versioned (сосуществование ок), но find-db в `miopen-config` при смене MIOpen-версии может стать несовместимым → пересоздание/потеря решений. Стоит явно отметить в дизайне: "при смене MIOpen версии содержимое miopen-config может быть устаревшим; рекомендуется docker volume rm перед пересборкой".

### 18. Почему один порог `0.8 * fd_soft_limit`, а не несколько?

Может быть, два-три порога: 50% — INFO, 80% — WARNING, 95% — ERROR?

### 19. Production rollback план?

Спека описывает forward deploy, но не rollback (например: MIOpen не может прочитать старый find-db → revert compose или `docker volume rm miopen-cache miopen-config` + redeploy). Дизайн об этом умалчивает.

### 20. Ожидаемый паттерн первого запуска после миграции не описан

Docker создаёт volume пустым → на первом старте warm-up leak ~1000 FDs, и только последующие рестарты дают ~0/день. Оператор может интерпретировать первый скачок как "план не сработал". В спеке стоит явно написать: "expected pattern: first start = ~1000 FDs leak (warm-up); subsequent restarts = ~0 FDs/day".

---

## Резюме приоритетов

| # | Категория | Серьёзность | Рекомендация |
|---|-----------|-------------|--------------|
| 1 | `/health` сам ест FDs (VideoFrameExtractor) | Critical | Отдельная задача — заменить на `shutil.which` |
| 2 | Утверждение "rect=True default" | Critical | Поправить формулировку в спеке (imgsz vs aspect ratio) |
| 3 | caplog + basicConfig race | Concern | Захардкодить `logger.setLevel` или `caplog.set_level` правильно |
| 4 | `git checkout --` без diff | Concern | `git diff` до checkout |
| 5 | ulimit 65536 без анализа | Concern | Обосновать через peak concurrent FD-usage |
| 6 | `os.listdir` без rate-limit | Concern | Кеш или rate-limit |
| 7 | `MIOPEN_FIND_MODE` без цифр | Concern | Ссылка на MIOpen docs + оценка perf penalty |
| 8 | Volumes multi-container | Concern | Зафиксировать single-container assumption |
| 9 | Тест без /proc | Concern | `test_health_handles_no_proc` |
| 10 | Дешёвая ffmpeg-проверка | Suggestion | Объединить с #1 |
| 11 | Rate-limit WARNING | Suggestion | `_last_warning_at` |
| 12 | Assert на `restart:` | Suggestion | `grep -c 'restart:'` |
| 13 | `fd_leaked_estimate` | Suggestion | Подсчёт (deleted) FDs |
| 14 | Mock anti-pattern | Suggestion | Патчить зависимости |
| 15 | Self-review completeness | Suggestion | Больше проверок |
| 16 | Validation script | Question | `scripts/verify_fd_containment.sh`? |
| 17 | ROCm upgrade + find-db | Question | Compatibility caveat |
| 18 | Несколько порогов | Question | INFO 50% / WARNING 80% / ERROR 95%? |
| 19 | Rollback plan | Question | Не описан |
| 20 | Первый запуск warm-up | Question | Описать ожидаемый pattern |

**Общая оценка ревьюера:** план добросовестный, conservative и well-scoped. Спека хорошо обосновывает решение (contain, not fix). Однако критические проблемы 1 и 2 снижают надёжность дизайна: первая делает наблюдаемость менее полезной, чем кажется; вторая делает post-merge валидацию менее точной. Обе стоит решить до merge, потому что они дешёвые (5-10 строк), а эффект большой.

---

Примечание от исполнителя (rev-minimax): пункт 2 ревьюера основан на локальном `.venv` (ultralytics с `rect: False` в default.yaml) — в сессионном контексте указано, что rect=True в predict был верифицирован эмпирически на prod-версии; стоит перепроверить версию ultralytics в prod-образе, прежде чем принимать/отклонять этот пункт.

---

## Failed reviewers (стрим оборвался без финального ответа, ревью не получено)

| Reviewer | Модель | Обрыв | Лог |
|----------|--------|-------|-----|
| zai/glm | glm-5.2 | ~5 мин работы, оборвался при исследовании кода | `runs/ext-claude/zai/glm/2026-07-01-23-42-16-1158973-…` |
| deepseek/v4-pro | deepseek-v4-pro[1m] | ~33 с, оборвался на чтении compose-файлов | `runs/ext-claude/deepseek/v4-pro/2026-07-01-23-42-42-1160647-…` |
| ollama/kimi | kimi-k2.7-code:cloud | ~35 с, ни одного текстового блока | `runs/ext-claude/ollama/kimi/2026-07-01-23-42-59-1162279-…` |
