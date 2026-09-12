# Отбор кадров по движению в `/extract/frames` и `/detect/video`

Дата: 2026-09-11. Ветка `feat/motion-frame-selection`.

## 1. Зачем

Текущий отбор кадров использует переменную `scene` фильтра `select` в ffmpeg. Это средняя разность соседних кадров по всему полю, и на записях Frigate порог 0.05 пробивают только ключевые кадры кодека. На корпусе из 207 сегментов эндпоинт отдал два кадра (нулевой и сотый) в 197 случаях и поймал 8 из 17 движущихся событий эталона. Подробности, методика и таблица стратегий — в отчёте `docs/research/frame-selection/README.md` (вне git).

Отчёт предлагал добавить режим `motion` рядом со старым ради совместимости. На brainstorming решено иначе: оба проекта (vision-api-server и frigate-analyzer) принадлежат одному человеку, других пользователей нет, поэтому старый алгоритм удаляется целиком, а новый становится единственным. Параметра `mode` нет.

Новый алгоритм на корпусе ловит 15 из 17 движущихся событий при 4.7 кадра на сегмент вместо 2.07. Извлечение дорожает с 0.7 с до 1.7 с на 16-секундный сегмент 2880×1620, нагрузка на детекцию и трафик растут пропорционально числу кадров, примерно в 2.3 раза.

## 2. Принятые решения

| Решение | Обоснование |
|---|---|
| Один алгоритм, `scene` удаляется вместе с `scene_threshold`, select-фильтром, записью JPEG на диск и интервальным фолбэком | Старый алгоритм слеп к движению по природе метрики; держать мёртвый путь незачем. Откат — через тег образа |
| Оба эндпоинта получают новый отбор | Экстрактор общий |
| `max_frames` становится капом со значением по умолчанию 6 | Отдельный `motion_frames` не нужен, если старый режим не сохраняется |
| Таймстемпы настоящие, `video_duration` из ffprobe, `get_video_info` через ffprobe JSON | Закрывает тикеты К3 и Ж2 из `docs/deep-research-review-report.md`; модуль и так переписывается |
| `frame_number` — номер кадра в записи, 0-based | Полезнее для отладки; делает проверку по корпусу точной |
| Новое поле `frames[].reason` | Отладка и проверка; клиент терпит неизвестные поля (`FAIL_ON_UNKNOWN_PROPERTIES=false`) |
| Порог шторма 0.02 — константа, не параметр | В отчёте не варьировался |
| Шаг сетки в шторм не меняется | Сетка — гарантия покрытия, не зависящая от метрики; признак шторма даёт ложные срабатывания (переключение ИК) |
| Кадры в памяти в BGR, без конвертаций | `/detect` уже отдаёт YOLO BGR; `/detect/video` сегодня отдаёт RGB, каналы перепутаны |
| Тесты только на синтетике `lavfi` | Репозиторий публичный, корпус — записи домашних камер |

## 3. API

### 3.1. Параметры

Оба эндпоинта, `POST /extract/frames` и `POST /detect/video`, принимают одинаковые параметры отбора. Все необязательны.

| Параметр | Тип | Диапазон | По умолчанию | Смысл |
|---|---|---|---|---|
| `max_gap` | float | 0.5–30.0 | 4.0 | шаг сетки, секунды |
| `motion_threshold` | float | 0.0001–0.1 | 0.001 | порог метрики `blob`, доля площади кадра |
| `min_interval` | float | 0.1–30.0 | 1.0 | минимальная дистанция между любыми двумя выбранными кадрами, секунды |
| `max_frames` | int | 1–200 | 6 | кап на число кадров в ответе |
| `quality` | int | 1–100 | 85 | JPEG, только `/extract/frames` |

Параметры детекции `/detect/video` (`conf`, `imgsz`, `max_det`, `model`) не меняются.

`scene_threshold` удаляется. FastAPI игнорирует незнакомые query-параметры, поэтому текущий frigate-analyzer, который шлёт `scene_threshold=0.05&min_interval=1.0&max_frames=50&quality=85`, продолжает работать. До обновления клиента `max_frames=50` означает, что в ветреный день сервер отдаст до 16 кадров на 16-секундный сегмент (пики не чаще `min_interval`); при выкате клиенту задаётся `DETECT_MAX_FRAMES=6`.

### 3.2. Ответ

`FrameExtractionResponse` и `VideoDetectionResponse` сохраняют набор полей верхнего уровня. Меняется содержимое `frames[]`:

| Поле | Было | Стало |
|---|---|---|
| `frame_number` | порядковый номер в списке, с 1 | номер кадра в записи, с 0 (переменная `n` фильтра `select`) |
| `timestamp` | порядковый номер минус один | pts кадра в секундах, из `showinfo` |
| `reason` | — | `first`, `grid` или `motion` |
| `video_duration` | таймстемп последнего кадра | длительность из ffprobe |

`ExtractedFrameData` (`/extract/frames`) и `FrameDetection` (`/detect/video`) получают поле `reason: Literal["first", "grid", "motion"]`. Модель `VideoDetectionSettings` в `app/models.py` нигде не используется и удаляется.

Список кадров на успешном ответе никогда не пуст: кадр 0 входит всегда.

Пример `frames[]` из `/extract/frames`:

```json
[
  {"frame_number": 0,   "timestamp": 0.0,   "reason": "first",  "image_base64": "...", "width": 2880, "height": 1620},
  {"frame_number": 31,  "timestamp": 2.48,  "reason": "motion", "image_base64": "...", "width": 2880, "height": 1620},
  {"frame_number": 50,  "timestamp": 4.0,   "reason": "grid",   "image_base64": "...", "width": 2880, "height": 1620},
  {"frame_number": 100, "timestamp": 8.0,   "reason": "grid",   "image_base64": "...", "width": 2880, "height": 1620}
]
```

### 3.3. Коды ошибок

Контракт кодов сохраняется. Клиент по 400, 413 и 422 помечает запись битой навсегда, по 500 ретраит до своего таймаута.

| Код | Когда |
|---|---|
| 400 | недопустимое расширение файла |
| 413 | файл больше 500 МБ |
| 422 | ffprobe не смог прочитать файл, в файле нет видеопотока (ValueError из экстрактора), невалидные query-параметры |
| 500 | ffmpeg упал, не дал ни одного кадра, вывод второго прохода не сошёлся с ожиданием, истёк дедлайн (RuntimeError из экстрактора) |

Ветка «no frames could be extracted → 400» в обоих эндпоинтах удаляется: экстрактор гарантирует непустой список или бросает RuntimeError.

### 3.4. Версия

Версия приложения поднимается до 3.0.0 в `FastAPI(version=...)` и в ответе `GET /` (сейчас там 2.3.0 и 2.2.0): удаление параметра — ломающее изменение.

## 4. Архитектура

### 4.1. Модули

**`app/frame_selection.py`, новый.** Чистая логика, зависит только от numpy и cv2, ffmpeg не вызывает.

```python
SCAN_WIDTH = 640            # ширина кадра для метрики
DIFF_THRESHOLD = 20         # уровней яркости
STORM_MEDIAN_BLOB = 0.02    # медиана blob по сегменту, выше — шторм
PTS_TOLERANCE = 1e-3        # допуск при сравнении времён, секунды

@dataclass(frozen=True)
class SelectionParams:
    max_gap: float = 4.0
    motion_threshold: float = 0.001
    min_interval: float = 1.0
    max_frames: int = 6

@dataclass(frozen=True)
class SelectedFrame:
    index: int
    reason: Literal["first", "grid", "motion"]

def prepare_frame(gray: np.ndarray) -> np.ndarray:
    """cv2.blur 3×3; результат кэшируется вызывающим кодом для следующей пары."""

def blob_area(prev: np.ndarray, cur: np.ndarray) -> float:
    """Метрика по двум подготовленным кадрам, доля площади кадра, см. раздел 5."""

def select_frames(pts: Sequence[float], blob: Sequence[float], params: SelectionParams) -> list[SelectedFrame]:
    """Правило отбора, см. раздел 6. Результат отсортирован по index."""
```

**`app/video_utils.py`, переписывается.**

```python
@dataclass
class VideoInfo:
    duration: float     # секунды, из ffprobe; 0.0, если ffprobe не знает
    width: int          # размеры ПОСЛЕ поворота по метаданным
    height: int
    fps: float          # только для логов
    codec: str
    rotation: int       # 0, 90, 180, 270

@dataclass
class ExtractedFrame:
    image: np.ndarray   # BGR uint8, H×W×3
    timestamp: float
    frame_number: int
    reason: str

@dataclass
class SelectionStats:
    total_frames: int
    median_blob: float
    storm: bool
    counts: dict[str, int]      # по причинам
    pass1_seconds: float
    pass2_seconds: float

@dataclass
class ScanResult:
    selected: list[SelectedFrame]
    pts: list[float]            # все кадры первого прохода
    blob: list[float]
    info: VideoInfo
    stats: SelectionStats       # pass2_seconds = 0.0 до второго прохода

@dataclass
class ExtractionResult:
    frames: list[ExtractedFrame]
    info: VideoInfo
    stats: SelectionStats

class VideoFrameExtractor:
    def __init__(self, ffmpeg_path="ffmpeg", ffprobe_path="ffprobe", timeout: float = 300.0)
    def get_video_info(self, video_path: str) -> VideoInfo          # ValueError
    def scan(self, video_path: str, params: SelectionParams) -> ScanResult   # проход 1 + отбор
    def extract_frames(self, video_path: str, params: SelectionParams) -> ExtractionResult  # scan + проход 2

async def extract_frames_from_video(video_data: bytes, params: SelectionParams) -> ExtractionResult
```

Конструктор без аргументов и `_verify_ffmpeg` сохраняются: их использует `GET /health`. `_get_duration`, `_get_dimensions`, `_extract_frames_fallback`, `_parse_ffmpeg_timestamps`, `_load_frames` удаляются. Метод `scan` публичный, потому что его использует проверочный скрипт на корпусе (раздел 12) и тесты.

**`app/models.py`.** Поле `reason` в `ExtractedFrameData` и `FrameDetection`, описания `frame_number` и `timestamp` обновляются, `VideoDetectionSettings` удаляется.

**`app/main.py`.** Типы `MaxGapQuery` и `MotionThresholdQuery` добавляются, `SceneThresholdQuery` удаляется, у `MaxFramesQuery` значение по умолчанию 6 в обоих эндпоинтах. Эндпоинты собирают `SelectionParams`, вызывают `extract_frames_from_video(video_data, params)`, берут `video_duration` из `result.info.duration`, а `video_resolution` из формы первого кадра. `/extract/frames` кодирует `frame.image` в JPEG напрямую, конвертация RGB→BGR удаляется. `/detect/video` передаёт `frame.image` в `run_inference` как есть и пишет `reason` в `FrameDetection`.

### 4.2. Поток данных одного запроса

1. Тело запроса читается в память и пишется во временный файл, как сейчас. Дедлайн извлечения: `time.monotonic() + timeout`.
2. `get_video_info`: ffprobe JSON. При нулевой ширине или высоте — ValueError «no video stream».
3. Проход 1 (`scan`): ffmpeg декодирует ролик в серые кадры шириной 640 и отдаёт их потоком через pipe; для каждого кадра считается `blob` относительно предыдущего; pts берутся из `showinfo` в stderr. В памяти только предыдущий подготовленный кадр и два списка чисел.
4. `select_frames(pts, blob, params)`.
5. Проход 2: ffmpeg декодирует ролик заново, `select` пропускает только выбранные кадры, они выходят через pipe в bgr24; `-frames:v K` останавливает ffmpeg после последнего нужного кадра. Вывод режется по размеру кадра и сопоставляется с выбранными индексами по pts из `showinfo`.
6. Временный файл удаляется в `finally`.

## 5. Метрика `blob`

Метрика воспроизводит `mask_stats` из `docs/research/frame-selection/scripts/metrics.py`, на которой получены пороги отчёта. Каждый серый кадр после масштабирования размывается один раз (`cv2.blur`, 3×3) — это `prepare_frame`; результат хранится для следующей пары. Для пары подготовленных кадров:

1. `d = cv2.absdiff(prev, cur)`;
2. `d = cv2.blur(d, (3, 3))`;
3. маска `m = (d > 20)` как uint8;
4. `m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ones(3, 3))`;
5. `m = cv2.dilate(m, ones(7, 7))`;
6. `cv2.connectedComponentsWithStats(m, connectivity=8)`; `blob` = площадь крупнейшей компоненты, кроме фона, делённая на число пикселей кадра; 0.0, если компонент нет.

Стоимость около 0.5 мс на кадр 640×360. Значение 0.001 соответствует пятну примерно 15×15 px на кадре 640×360, около 70×70 px в оригинале 2880×1620. Шумовой фон в тихих сегментах не выше 0.0014.

Размер кадра первого прохода задаётся явно: ширина 640, высота `round(h × 640 / w / 2) × 2`, где `w` и `h` — размеры после поворота. Это та же формула, что в `metrics.py`, и она снимает неоднозначность округления `scale=640:-2`.

## 6. Правило отбора

Вход: `pts[i]` — время кадра `i`, `blob[i]` — метрика между кадрами `i−1` и `i`, `blob[0] = 0`, длины равны `n ≥ 1`. Функция детерминирована.

1. **Первый кадр.** Индекс 0 всегда, причина `first`.
2. **Сетка.** Шаг `step = max(max_gap, min_interval)`. Кадры перебираются по порядку; кадр `i` входит в сетку, если `pts[i] − pts[last] ≥ step − PTS_TOLERANCE`, где `last` — последний выбранный. Причина `grid`.
3. **Прореживание.** Последовательность `S` = первый кадр плюс сетка, длина `L`. Прореживание до `M` кадров при `L > M`: при `M = 1` остаётся только кадр 0; иначе остаются позиции `round_half_up(k × (L − 1) / (M − 1))` для `k = 0 … M − 1`. Первый и последний элементы `S` сохраняются. Сетка прореживается до `max_frames − 1`, а не до `max_frames`: один слот всегда остаётся под пик, поэтому запись любой длины получает хотя бы один кадр по движению. Если ни один пик не подошёл, результат — сетка, прореженная до `max_frames`, и придержанный кадр не пропадает впустую.
4. **Шторм.** Если `n ≥ 2` и медиана `blob[1:]` больше `STORM_MEDIAN_BLOB`, отбор заканчивается сеткой, прореженной до `max_frames`; проверка идёт до шага 5.
5. **Пики.** Кандидаты — индексы с `blob[i] > motion_threshold`, ещё не выбранные, в порядке убывания `blob`, при равенстве по возрастанию индекса. Кандидат берётся, если `|pts[i] − pts[j]| ≥ min_interval − PTS_TOLERANCE` для всех уже выбранных `j` и число выбранных меньше `max_frames`. Причина `motion`. Перебор заканчивается, когда кандидаты исчерпаны или бюджет выбран.
6. Результат сортируется по индексу.

Правило предполагает неубывающие pts. Экстрактор пишет предупреждение в лог, если pts первого прохода не монотонны, и передаёт значения как есть.

Отличия от эталонного `s_adaptive` в `evaluate.py`: медиана считается без `blob[0]`; прореживание сетки в эталоне отсутствует, потому что сегменты корпуса не длиннее 46 с. На результатах корпуса это не сказывается.

## 7. Конвейер ffmpeg

### 7.1. ffprobe

```
ffprobe -v error -print_format json -show_streams -show_format -select_streams v:0 <path>
```

Таймаут 30 с. Ненулевой код возврата, невалидный JSON, отсутствие потоков или нулевые размеры — ValueError с текущими текстами сообщений («could not be read as a valid video», «no video stream»). Разбор:

- `width`, `height` из `streams[0]`; `rotation` из `streams[0].side_data_list[*].rotation` (первый элемент с таким ключом), иначе из `streams[0].tags.rotate`, иначе 0; нормализуется в `{0, 90, 180, 270}` по модулю, знак не важен, потому что значение нужно только для перестановки ширины и высоты при 90 и 270. ffmpeg поворачивает кадры сам (`-autorotate` по умолчанию), и размеры сырого потока на выходе — размеры после поворота;
- `duration`: `format.duration`, иначе `streams[0].duration`, иначе 0.0. Если 0.0, экстрактор после первого прохода подставляет pts последнего декодированного кадра и пишет предупреждение;
- `fps`: `avg_frame_rate`, при нулевом знаменателе `r_frame_rate`, иначе 0.0; `codec`: `codec_name`, иначе `unknown`.

### 7.2. Проход 1

```
ffmpeg -hide_banner -nostats -loglevel info -an -i <path>
       -vf scale=640:<H>,showinfo -fps_mode passthrough
       -f rawvideo -pix_fmt gray pipe:1
```

`-fps_mode passthrough` обязателен: без него ffmpeg дублирует кадры до постоянной частоты при неверно угаданном `r_frame_rate`. `showinfo` пишет в stderr только при `-loglevel info`; `-nostats` убирает строки прогресса.

Чтение: `Popen` со stdout и stderr в pipe. Поток-демон вычитывает stderr построчно (образец: `_drain_stderr` в `app/ffmpeg_pipe.py`), собирает `pts_time` каждой строки `showinfo` в список чисел и держит последние 50 строк для сообщений об ошибках. Регулярное выражение для pts: `pts_time:\s*(-?[0-9.]+)`. Основной поток читает stdout по `640 × H` байт (`BufferedReader.read` возвращает меньше только на EOF), вызывает `prepare_frame`, считает `blob_area` относительно предыдущего кадра (для кадра 0 — 0.0) и перед каждым чтением проверяет дедлайн.

После EOF: `wait`, `join` потока stderr. Число pts и число кадров сверяются; при расхождении берётся минимум и пишется предупреждение. Ноль кадров — RuntimeError. Ненулевой код возврата при ненулевом числе кадров — предупреждение, работа продолжается: так выглядят обрезанные сегменты Frigate.

### 7.3. Проход 2

```
ffmpeg -hide_banner -nostats -loglevel info -an -i <path>
       -vf select='eq(n\,a)+eq(n\,b)+…',showinfo -fps_mode passthrough
       -frames:v <K> -f rawvideo -pix_fmt bgr24 pipe:1
```

`K` — число выбранных кадров. Переменная `n` в `select` считает кадры графа фильтров, при `passthrough` она совпадает с номером кадра первого прохода. `communicate(timeout=остаток дедлайна)`; при истечении — `kill`, RuntimeError.

Разбор вывода: из строк `showinfo` берутся `pts_time` и размер `s:(\d+)x(\d+)`; stdout режется по `W × H × 3`, где `W` и `H` из `VideoInfo`. Проверки, каждая при провале даёт RuntimeError: размер в `showinfo` равен ожидаемому; длина stdout кратна размеру кадра; число кадров равно числу строк `showinfo` и равно `K`; pts каждого выходного кадра совпадает с pts выбранного индекса из первого прохода с допуском `PTS_TOLERANCE`. Сопоставление идёт по pts, а не по позиции, чтобы пропавший кадр не сдвинул таймстемпы остальных. Массивы копируются из буфера (`.copy()`), чтобы быть записываемыми.

Память второго прохода: не больше `max_frames` полных кадров, при значениях по умолчанию 6 × 14 МБ для 2880×1620.

### 7.4. Дедлайн

Один дедлайн на извлечение, по умолчанию 300 с, покрывает оба прохода и расчёт метрики. Таймаут ffprobe отдельный, 30 с.

## 8. Логирование

Одна строка INFO на извлечение:

```
Frame selection: 5 frames (first 1, grid 3, motion 1) of 200, storm=False, median_blob=0.0004, pass1=0.82s, pass2=0.61s
```

Предупреждения: ffmpeg завершился с ошибкой после N кадров; число pts не совпало с числом кадров; pts не монотонны; ffprobe не дал длительность. Команды ffmpeg — на уровне DEBUG.

## 9. Тесты

Тесты запускаются в `.venv`: `.venv/bin/python -m pytest tests/ -v`. CI pytest не гоняет; интеграционные тесты пропускаются, если `shutil.which("ffmpeg")` пуст.

### 9.1. `tests/test_frame_selection.py`

Правило на массивах:

- один кадр → `[SelectedFrame(0, "first")]`;
- статичный ролик 10 с при 10 fps, `max_gap=4` → индексы 0, 40, 80 с причинами `first`, `grid`, `grid`;
- `min_interval=5 > max_gap=4` → шаг сетки 5;
- 40 с при `max_gap=4`, `max_frames=6` → 6 кадров, первый 0, последний 360 (pts 36.0); `max_frames=1` → только 0;
- шторм: медиана `blob` выше 0.02 при наличии пиков → только `first` и `grid`;
- пики: два пика с разной метрикой → берётся больший первым; пик ближе `min_interval` к кадру сетки не берётся; при `max_frames` бюджет ограничивает пики;
- равные метрики → меньший индекс первым;
- дрожание pts в пределах 1 мс не ломает сетку.

Метрика на массивах: одинаковые кадры → 0.0; белый квадрат 40×40 на чёрном фоне 640×360 в одном из кадров → `blob` в диапазоне, соответствующем квадрату после дилатации (примерно (40+6)² / (640×360) с допуском 20 %); гауссов шум с амплитудой ниже порога → 0.0.

### 9.2. `tests/test_video_utils.py`

- парсер ffprobe JSON на мокнутом `subprocess.run`: 64×48 длительностью 150 с даёт правильные поля; кодеки `h264` и `av1` распознаются; `rotation: -90` в `side_data_list` меняет ширину и высоту местами; отсутствие `format.duration` даёт 0.0; нет потоков → ValueError «no video stream»; ненулевой код → ValueError «could not be read as a valid video» (существующие тесты адаптируются);
- парсер `showinfo` на зафиксированном образце stderr: pts, размер, отрицательный pts, строки без `showinfo` игнорируются.

### 9.3. `tests/test_video_extraction_integration.py`

Клипы генерируются `ffmpeg -f lavfi` в `tmp_path` (фикстуры уровня модуля), кодек `mpeg4 -q:v 2`, контейнер mp4:

| Клип | Источник | Что проверяется |
|---|---|---|
| статичный, 10 с, 10 fps, 320×240 | `color=c=gray:s=320x240:r=10:d=10` | кадры 0, 40, 80; причины; `timestamp` 0/4/8 ± 0.11; `video_duration` 10 ± 0.1; форма кадров (240, 320, 3) |
| движение | тот же фон плюс `drawbox=x='40+100*t':y=100:w=30:h=30:color=white:t=fill:enable='between(t,1,3)'` | есть кадры `motion`, все внутри 0.9–3.2 с, попарно не ближе `min_interval`; кадр 0 и сетка на месте; всего не больше `max_frames` |
| шторм | `nullsrc=s=320x240:r=10:d=10,geq=lum='random(1)*255':cb=128:cr=128` | `stats.storm` истинен, причин `motion` нет |
| длинный, 40 с | статичный | 6 кадров, первый 0, последний с pts 36 ± 0.11 |
| поворот | статичный клип, перепакованный с `-display_rotation 90 -c copy` | `VideoInfo` 240×320, кадры формы (320, 240, 3); пропускается, если ffmpeg не знает `-display_rotation` |

Эндпоинт `/extract/frames` через `TestClient` на статичном клипе: код 200, поля `frame_number`, `timestamp`, `reason`, `video_duration`; `image_base64` раскодируется в JPEG 320×240; лишний `scene_threshold=0.05` в запросе не мешает; `max_gap=0.1` даёт 422.

## 10. Документация

- `.claude/rules/api.md`: таблицы параметров обоих эндпоинтов, описание алгоритма вместо «Frame Extraction Algorithm», пример ответа с `reason` и настоящими номерами кадров.
- `CLAUDE.md`: пункт «Smart Frames» в Key Patterns заменяется описанием отбора по движению; таблица эндпоинтов; строка про `app/frame_selection.py` в таблице архитектуры.
- Докстринги `/detect/video` и `/extract/frames`.

## 11. Вне области

- Потоковое чтение аплоада, лимиты памяти, отдельный executor (К4).
- JPEG-кодирование в event loop (В7).
- Изменение frigate-analyzer: убрать `scene_threshold`, задать `max_frames=6`, при желании добавить `max_gap` и `motion_threshold`. Отдельная задача в другом репозитории после выката сервера.
- Каскад на `yolo26n` как метрика для GPU-серверов.

## 12. Проверка после реализации

1. `pytest` в `.venv` зелёный.
2. Скрипт вне репозитория, рядом с корпусом в `~/vision-api-research/frame-selection/`: импортирует `app/` из репозитория, вызывает `VideoFrameExtractor().scan(path, SelectionParams())` для каждого из 207 сегментов и считает метрики функциями `evaluate.py` (`load`, сопоставление событий по индексам кадров). Ожидание: 15 из 17 движущихся событий с допуском в одно событие, около 4.7 кадра на сегмент, на пустых сегментах около 4.7.
3. Замер `extract_frames` на 16-секундном сегменте 2880×1620: около 1.7 с.
4. Ручной smoke: локальный uvicorn, `curl -F file=@seg.mp4 'http://127.0.0.1:8000/extract/frames'`, проверка полей ответа.
