# Video Annotation Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ускорить `/detect/video/visualize` примерно в 2.6 раза на NVIDIA (эталонный ролик 2560×1920, yolo26x@1024: 86 → 33 с) и не сломать AMD и CPU.

**Architecture:** Два прохода через процессы ffmpeg остаются. В проходе 1 поток-читатель (`ThreadedFrameReader`) читает кадры BGR и уменьшает их до размера инференса ровно как `LetterBox` в Ultralytics, а основной поток гонит YOLO батчами (`BatchDetector`, FP16 и размер батча по настройкам `VIDEO_FP16`/`VIDEO_BATCH_SIZE`). В проходе 2 кадры идут в yuv420p от декодера до энкодера, рамки рисуются прямо на плоскостях Y/U/V, запись идёт в потоке-писателе (`ThreadedFrameWriter`). Видео-задачи получают собственный экземпляр модели (`ModelManager.get_video_model`).

**Tech Stack:** Python 3.13 (`.venv`), FastAPI, Ultralytics YOLO ≥ 8.4.150, PyTorch, OpenCV (`cv2`), numpy, ffmpeg/ffprobe через `subprocess`, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-24-video-annotation-speedup-design.md`

## Global Constraints

- Тесты только в `.venv`: `.venv/bin/python -m pytest tests/ -v` из корня репозитория. Каждый шаг «Run» ниже подразумевает этот интерпретатор.
- Модули в `app/` импортируют друг друга по голым именам (`from frame_threads import ...`), потому что `app/` лежит в `sys.path` (см. `tests/conftest.py`). В `app/` нет `__init__.py`, так и оставить.
- Комментарии и докстринги в коде на английском, как во всём `app/`. Коммиты по образцу репозитория: `feat(video): ...`, `test(video): ...`, `docs: ...`, `chore: ...`.
- `ultralytics>=8.4.150,<9.0.0`: 8.4.150 — первая версия с параметром `predict(quantize=16)`, в 8.4.149 его нет. Устаревший `half=True` пишет WARNING на каждом вызове, его не использовать.
- `VIDEO_FP16`: `auto` | `true` | `false`. `VIDEO_BATCH_SIZE`: `auto` | целое 1–64. Регистр не важен, пустое значение означает `auto`. `auto`: NVIDIA (устройство модели `cuda` и `torch.version.hip is None`) — FP16 и батч 8; ROCm и CPU — FP32 и батч 1. `VIDEO_FP16=true` на CPU — WARNING и FP32.
- Кадр для инференса уменьшает только `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` до `inference_size()`. Скейлеры ffmpeg в проходе 1 запрещены: они меняют детекции (на 400 кадрах слабых детекций 287 вместо 485).
- Проход 2: `yuv420p` от декодера до энкодера. Цвет рамок — BT.601 limited range, как в текущем энкодере.
- Не меняются: API и ответы, `AnnotationStats`, разбивка прогресса (проход 1 — 0–80 %, проход 2 — 80–99 %), политика кодека и битрейта, откат NVENC → CPU, семантика отмены. Энкодер при отмене не убивается.
- Видео-задачи берут модель только через `ModelManager.get_video_model`, `/detect` и `/detect/video` — через `get_model`.
- Никаких записей камер в git: репозиторий публичный. Тестовые клипы генерируются `ffmpeg -f lavfi` во временный каталог, интеграционные тесты пропускаются без `ffmpeg`/`ffprobe`.
- `tests/test_supervisor.py::test_smoke_killpg_reaches_a_grandchild` падает внутри контейнера NVIDIA-образа и без этих изменений (окружение контейнера). На хосте он проходит. Ориентир — прогон на хосте.
- Перед созданием PR все файлы из `docs/superpowers/` удаляются из ветки через `git rm` и коммитятся (Task 12). Плановые документы не должны попасть в диф PR.
- Субагентам не назначать модель haiku.
- Задачи 10–12 выполняет основная сессия вместе с владельцем, а не субагент: они трогают GPU владельца, его хосты и PR.

## Review Focus

Пять классов входа, которые спека подразумевает, а модульные тесты на моках не проверяют. Каждый пункт закрыт тестом в задаче-владельце:

1. **Запись с IP-камеры в полном диапазоне `yuvj420p`** (так пишут многие камеры) — проход 2 просит у ffmpeg `yuv420p`. Рамка должна появиться своим цветом, число кадров не должно измениться. Тест `test_full_range_source`, Task 8.
2. **Звук короче видео** — `-shortest` завершает энкодер раньше, чем кончились кадры. `annotate()` должен закончиться успешно, без зависания и падения потока-писателя, результат обрезан по звуку. Тест `test_audio_shorter_than_video_ends_the_output_early`, Task 8.
3. **Переменный fps** — проходы 1 и 2 должны насчитать одни и те же кадры, иначе рамки съедут. Рамка должна быть на последнем кадре, число кадров результата равно числу кадров прохода 1. Тест `test_variable_frame_rate_keeps_passes_aligned`, Task 8.
4. **Запись без звука** — в результате нет аудио, ошибки нет. Тест `test_clip_without_audio`, Task 8.
5. **Клип короче одного батча** (кадров меньше `VIDEO_BATCH_SIZE`) — один вызов `predict` со всеми кадрами, каждый кадр размечен и записан. Тест `test_clip_shorter_than_one_batch`, Task 7.

---

## Карта файлов

| Файл | Ответственность | Задачи |
|---|---|---|
| `app/config.py` | `video_fp16`, `video_batch_size` и их валидаторы | 1 |
| `docker/docker-compose-*.yml`, `docker/deploy/docker-compose-*.yml` | проброс `VIDEO_FP16`, `VIDEO_BATCH_SIZE` | 1 |
| `requirements.txt` | `ultralytics>=8.4.150` | 1 |
| `app/ffmpeg_pipe.py` | `frame_shape()`, `_grow_pipe()`, `pix_fmt`, `read_into`, `abort`, запись без копии | 2 |
| `app/frame_threads.py` (новый) | `ThreadedFrameReader`, `ThreadedFrameWriter` | 3 |
| `app/batch_inference.py` (новый) | `InferenceMode`, `resolve_inference_mode`, `model_stride`, `inference_size`, `resize_for_inference`, `extract_detections`, `BatchDetector` | 4 |
| `app/visualization.py` | BT.601-помощники, `yuv420_planes`, `draw_detection_yuv420`, общая геометрия подписи | 5 |
| `app/model_manager.py` | `get_video_model`, выгрузка по TTL и при остановке | 6 |
| `app/video_annotator.py` | оба прохода на новых компонентах, `fp16`/`batch_size`, логи | 7 |
| `app/main.py` | worker: `get_video_model` + `resolve_inference_mode` | 9 |
| `tests/test_config.py`, `tests/test_compose.py` | настройки, проброс | 1 |
| `tests/test_ffmpeg_pipe.py` | новые возможности декодера и энкодера | 2 |
| `tests/test_frame_threads.py` (новый) | потоки чтения и записи | 3 |
| `tests/test_batch_inference.py` (новый) | ресайз против `LetterBox`, режим, батчи, OOM | 4 |
| `tests/test_visualization.py` | отрисовка в YUV | 5 |
| `tests/test_model_manager.py` (новый) | `get_video_model` | 6 |
| `tests/test_video_annotator.py` | моки на `read_into`, новые тесты прохода 1 | 7 |
| `tests/test_video_annotation_integration.py` (новый) | `annotate()` с настоящим ffmpeg | 8 |
| `tests/test_worker.py` | `get_video_model`, режим инференса | 9 |
| `CLAUDE.md`, `.claude/rules/api.md`, `.claude/rules/docker.md`, `docker/deploy/.env.example` | документация | 1, 3, 4, 7, 8 |

---

### Task 1: Настройки `VIDEO_FP16` и `VIDEO_BATCH_SIZE`, проброс в compose, версия ultralytics

✅ Done — see commit(s): `d92d1fb`

---

### Task 2: `ffmpeg_pipe` — форматы кадра, чтение в готовый буфер, запись без копии, `abort()`

✅ Done — see commit(s): `0116444`

---

### Task 3: Потоки чтения и записи `frame_threads.py`

✅ Done — see commit(s): `90515b1`

---

### Task 4: Батчевый инференс `batch_inference.py`

✅ Done — see commit(s): `d546358`

---

### Task 5: Отрисовка на кадре yuv420p

✅ Done — see commit(s): `63d2a18`

---

### Task 6: Отдельный экземпляр модели для видео-задач

✅ Done — see commit(s): `7030658`

---

### Task 7: `VideoAnnotator` на новых компонентах

✅ Done — see commit(s): `fb7f70b`

---

### Task 8: Интеграционный тест `annotate()` с настоящим ffmpeg

✅ Done — see commit(s): `8ef62f3`, `3a9c1d7`, `ba2e711`, `9d0ef5b`

---

### Task 9: Worker берёт свою модель и режим инференса

✅ Done — see commit(s): `0e597ec`, `b8c5b3d`

---

### Task 10: Проверка на RTX 3090

Выполняет основная сессия вместе с владельцем. Код не меняется, результаты идут в описание PR.

**Files:** нет.

- [ ] **Step 1: Собрать NVIDIA-образ из ветки**

Run: `docker build -f docker/nvidia/Dockerfile -t vision-api-server:speedup .`
Expected: образ собран. Первый раз скачиваются колёса torch cu130, это долго.

- [ ] **Step 2: Запустить отдельный контейнер рядом с рабочим**

```bash
docker run -d --name vas-speedup --gpus all -p 3002:8000 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e YOLO_MODELS='{"yolo26s.pt":"cuda:0","yolo26x.pt":"cuda:0"}' -e YOLO_DEVICE=cuda:0 \
  -e MODELS_DIR=/models -e LOG_LEVEL=INFO -e VIDEO_HW_ACCEL=nvidia \
  -v deploy_models:/models vision-api-server:speedup
until curl -sf localhost:3002/health >/dev/null; do sleep 5; done; curl -s localhost:3002/health
```

Рабочий контейнер `deploy-vision-api-1` на порту 3001 не трогать.

- [ ] **Step 3: Прогнать ролик с параметрами frigate-analyzer**

Ролик: копия результата задачи `09a599cd4b74` (2560×1920 HEVC, 1601 кадр). Если её уже нет в scratchpad сессии брейншторма, попросить у владельца свежую двухминутную запись той же камеры. Записи в репозиторий не класть.

```bash
VIDEO=/path/to/video.mp4
JOB=$(curl -s -X POST "localhost:3002/detect/video/visualize?conf=0.6&imgsz=1024&max_det=100&detect_every=1&line_width=2&show_labels=true&show_conf=true&model=yolo26x.pt&classes=person,car,motorcycle,truck,bicycle,cat,dog,bird,backpack,horse,sheep,cow,bear,elephant,zebra,giraffe" \
  -F "file=@$VIDEO" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
while :; do S=$(curl -s localhost:3002/jobs/$JOB); echo "$S" | grep -q '"status":"completed"\|"status":"failed"' && break; sleep 3; done; echo "$S"
docker logs vas-speedup 2>&1 | grep -E "Starting annotation|Frame processing complete"
```

Expected:
- `"status":"completed"`;
- в логе `fp16=True, batch=8`;
- `processing_time_ms` около 30–35 тыс.; эталон на продакшене — 76.8 с (исходный файл той же задачи);
- `pass2` около 10 с.

- [ ] **Step 4: Посмотреть результат и отзывчивость**

Во время повторного прогона шага 3 в соседнем терминале:
- `for i in $(seq 10); do /usr/bin/time -f %e curl -s -o /dev/null localhost:3002/health; sleep 2; done` — каждый ответ не дольше доли секунды;
- `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1` — во время прохода 1 загрузка около 90–100 %.

После прогона скачать результат и посмотреть кадр с рамкой (глазами, через Read):

```bash
curl -s -o /tmp/annotated.mp4 localhost:3002/jobs/$JOB/download
ffmpeg -hide_banner -loglevel error -y -i /tmp/annotated.mp4 -vf "select=eq(n\,800),scale=960:-1" -frames:v 1 /tmp/annotated_800.png
```

- [ ] **Step 5: Убрать контейнер и записать результаты**

Run: `docker rm -f vas-speedup`

Записать в черновик описания PR: время до и после, fps проходов, загрузку GPU и наблюдение по кадру.

---

### Task 11: Проверка на AMD и CPU

Выполняет основная сессия. Адреса хостов и доступ даёт владелец; без них задачу не начинать, а спросить.

**Files:** нет.

- [ ] **Step 1: Получить от владельца хосты AMD и CPU**

Спросить адреса, способ доступа и какой ролик использовать (тот же, что в Task 10, или запись с этого хоста).

- [ ] **Step 2: AMD, режим `auto`**

На AMD-хосте в клоне ветки собрать и запустить образ на свободном порту, с теми же переменными, что у рабочего контейнера хоста (`HSA_OVERRIDE_GFX_VERSION` и т. д.):
`docker build -f docker/amd/Dockerfile -t vision-api-server:speedup-amd .`

Прогнать ролик, как в Task 10, Step 3. Проверить:
- в логе `fp16=False, batch=1`: форма тензоров и тип прежние, новых компиляций MIOpen быть не должно;
- время против текущего образа хоста на том же ролике;
- `curl -s localhost:<port>/health`: `fd_deleted` не растёт от задачи к задаче (сигнатура утечки MIOpen).

- [ ] **Step 3: AMD, эксперимент FP16 и батч 8**

Перезапустить контейнер с `-e VIDEO_FP16=true -e VIDEO_BATCH_SIZE=8`, прогнать тот же ролик дважды. Записать время обоих прогонов (первый включает компиляцию MIOpen) и динамику `open_fds`/`fd_deleted`. По итогам предложить владельцу, менять ли `auto` для AMD. Это отдельная задача, в этой ветке `auto` не трогать.

- [ ] **Step 4: CPU**

На CPU-хосте собрать `docker build -f docker/cpu/Dockerfile -t vision-api-server:speedup-cpu .`, прогнать ролик. Убедиться, что задача завершается, в логе `fp16=False, batch=1`, а рамки на месте. Сравнить время с текущим CPU-образом.

- [ ] **Step 5: Убрать временные контейнеры и записать результаты в черновик PR**

---

### Task 12: Подготовка ветки к PR

Выполняет основная сессия.

**Files:**
- Delete: `docs/superpowers/specs/2026-09-24-video-annotation-speedup-design.md`, `docs/superpowers/plans/2026-09-24-video-annotation-speedup.md`

- [ ] **Step 1: Прогнать весь набор тестов**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS целиком.

- [ ] **Step 2: Убрать плановые документы из ветки**

```bash
git rm docs/superpowers/specs/2026-09-24-video-annotation-speedup-design.md \
  docs/superpowers/plans/2026-09-24-video-annotation-speedup.md
git commit -m "chore: remove design and plan documents before PR"
```

Документы остаются в истории ветки.

- [ ] **Step 3: Завершить ветку**

Использовать superpowers:finishing-a-development-branch. PR создаётся только после явного согласия владельца; в описание PR войдут результаты задач 10 и 11.
