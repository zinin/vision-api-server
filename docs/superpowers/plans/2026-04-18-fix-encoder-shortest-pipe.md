# Fix FFmpeg encoder `-shortest` BrokenPipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перестать падать с `RuntimeError: FFmpeg encoder pipe broken`, когда encoder корректно завершил работу (`rc == 0`) из-за флага `-shortest` и audio-stream, заканчивающегося раньше video-pipe.

**Architecture:** Локальная правка в `FFmpegEncoder.write_frame` (файл `app/ffmpeg_pipe.py`). При `BrokenPipeError` или при `poll() == 0` вызываем `wait(timeout=5)`; если `rc == 0` — устанавливаем флаг `_eof = True` и молча прекращаем писать (encoder уже корректно финализовал файл). При `rc != 0` поведение не меняется — поднимаем `RuntimeError` с stderr. При зависании — `kill` + `RuntimeError`.

**Tech Stack:** Python 3.12, `subprocess.Popen`, `pytest`, `unittest.mock.MagicMock`.

---

## File Structure

- **Modify:** `app/ffmpeg_pipe.py` — добавить поле `_eof` в `FFmpegEncoder.__init__`, переписать `write_frame` с обработкой BrokenPipe + `rc == 0` как штатного завершения.
- **Modify:** `tests/test_ffmpeg_pipe.py` — добавить 4 новых теста в класс `TestFFmpegEncoder`, оставить существующие без изменений.
- **No CLAUDE.md update** — текущая секция «Key Patterns» покрывает поведение, явная заметка про `-shortest` избыточна.

---

## Task 1: Baseline — verify current test suite passes

**Files:**
- Read: `tests/test_ffmpeg_pipe.py`

- [ ] **Step 1: Activate venv and run existing ffmpeg_pipe tests**

Run:
```bash
source .venv/bin/activate
python -m pytest tests/test_ffmpeg_pipe.py -v
```

Expected: все тесты проходят (PASS). Если есть venv — активируем; если нет — создаём:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

- [ ] **Step 2: Verify commit state clean**

Run:
```bash
git status
```

Expected: `On branch fix/encoder-shortest-pipe`, working tree clean (кроме самого plan-документа).

---

## Task 2: Failing test — encoder exits with rc=0 via `-shortest`, BrokenPipe on write

**Files:**
- Test: `tests/test_ffmpeg_pipe.py` (добавить в класс `TestFFmpegEncoder`, после существующего `test_write_frame_after_crash_raises`).

- [ ] **Step 1: Write the failing test**

Добавить в `tests/test_ffmpeg_pipe.py` внутри класса `TestFFmpegEncoder` (после метода `test_write_frame_after_crash_raises`):

```python
    def test_write_frame_graceful_eof_on_shortest(self):
        """write_frame MUST NOT raise when encoder exits cleanly (rc=0) via -shortest.

        Reproduces the real-world scenario where ffmpeg closes pipe:0 after
        audio EOF (shorter than video), Python gets BrokenPipeError, but the
        encoder process finishes normally. The output file is valid — we just
        need to stop writing further frames.
        """
        mock_proc = self._make_mock_process(returncode=0)
        # poll() returns None at the moment of the write (process still alive).
        mock_proc.poll.return_value = None
        # stdin.write raises BrokenPipeError (ffmpeg closed its stdin fd).
        mock_proc.stdin.write.side_effect = BrokenPipeError(32, "Broken pipe")
        # wait() inside the BrokenPipe handler returns 0 — clean exit.
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0

        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                crf=18,
            ) as encoder:
                # First write triggers the graceful-eof path, must not raise.
                encoder.write_frame(frame)
                # Subsequent writes are silent no-ops (no additional stdin writes).
                encoder.write_frame(frame)
                encoder.write_frame(frame)

        # stdin.write called exactly once (the one that raised BrokenPipe).
        assert mock_proc.stdin.write.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_ffmpeg_pipe.py::TestFFmpegEncoder::test_write_frame_graceful_eof_on_shortest -v
```

Expected: FAIL с `RuntimeError: FFmpeg encoder pipe broken: [Errno 32] Broken pipe`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_ffmpeg_pipe.py
git commit -m "test(ffmpeg_pipe): add failing test for -shortest graceful EOF"
```

---

## Task 3: Minimal implementation — graceful handling of BrokenPipe + rc=0

**Files:**
- Modify: `app/ffmpeg_pipe.py` (FFmpegEncoder.__init__ и write_frame).

- [ ] **Step 1: Add `_eof` flag to `FFmpegEncoder.__init__`**

Найти в `app/ffmpeg_pipe.py` строку `self._stderr_lines: deque[bytes] = deque(maxlen=100)` в методе `__init__` класса `FFmpegEncoder` (строка 126) и сразу после неё добавить новое поле. Полный изменённый фрагмент:

```python
    def __init__(
        self,
        original_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        hw_config: HWAccelConfig,
        codec: str,
        crf: int | None = None,
        bitrate: int | None = None,
    ):
        self._stderr_lines: deque[bytes] = deque(maxlen=100)
        # True after the encoder cleanly exits (rc=0) while we still had
        # frames to write — e.g. FFmpeg's -shortest closes pipe:0 when the
        # audio stream ends before the piped raw video. Subsequent
        # write_frame() calls become silent no-ops.
        self._eof = False
```

- [ ] **Step 2: Rewrite `write_frame` with graceful EOF path**

Заменить в `app/ffmpeg_pipe.py` текущий метод `write_frame` (строки 151-165) на:

```python
    def write_frame(self, frame: np.ndarray) -> None:
        """Write one BGR24 frame to the encoder.

        Raises RuntimeError if the process crashed (rc != 0). A clean
        early exit (rc == 0) is treated as EOF — the frame is silently
        dropped and further calls are no-ops. This covers FFmpeg's
        -shortest behaviour: when the audio stream ends before the
        piped raw video, ffmpeg closes pipe:0 from its side, the
        output file is already fully written, and there's nothing
        left for Python to do.
        """
        if self._eof:
            return
        rc = self._process.poll()
        if rc is not None:
            if rc == 0:
                self._eof = True
                return
            raise RuntimeError(
                f"FFmpeg encoder crashed (rc={rc}): "
                f"{_format_stderr(self._stderr_lines)}"
            )
        try:
            self._process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            # The pipe closed mid-write. Most often this means the
            # encoder just finalised the output (e.g. -shortest on an
            # audio stream shorter than the video pipe). Give it a
            # moment to reap, then distinguish clean exit vs crash.
            try:
                rc = self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
                raise RuntimeError(
                    f"FFmpeg encoder hung after pipe break: {e}. "
                    f"stderr: {_format_stderr(self._stderr_lines)}"
                ) from e
            if rc == 0:
                self._eof = True
                return
            raise RuntimeError(
                f"FFmpeg encoder pipe broken (rc={rc}): {e}. "
                f"stderr: {_format_stderr(self._stderr_lines)}"
            ) from e
```

- [ ] **Step 3: Run the Task 2 test — must now pass**

Run:
```bash
python -m pytest tests/test_ffmpeg_pipe.py::TestFFmpegEncoder::test_write_frame_graceful_eof_on_shortest -v
```

Expected: PASS.

- [ ] **Step 4: Run all ffmpeg_pipe tests — no regressions**

Run:
```bash
python -m pytest tests/test_ffmpeg_pipe.py -v
```

Expected: все тесты проходят, включая существующий `test_write_frame_after_crash_raises`.

- [ ] **Step 5: Commit implementation**

```bash
git add app/ffmpeg_pipe.py
git commit -m "fix(ffmpeg_pipe): treat encoder rc=0 after BrokenPipe as clean EOF"
```

---

## Task 4: Test — real crash after pipe break still raises

**Files:**
- Test: `tests/test_ffmpeg_pipe.py` (класс `TestFFmpegEncoder`).

Этот тест гарантирует, что реальная ошибка (encoder упал с ненулевым кодом) по-прежнему вызывает `RuntimeError`. Это защищает от регрессий, когда Task 3 мог бы замолчать настоящие падения.

- [ ] **Step 1: Write the test**

Добавить в `tests/test_ffmpeg_pipe.py` в класс `TestFFmpegEncoder`, после `test_write_frame_graceful_eof_on_shortest`:

```python
    def test_write_frame_pipe_broken_with_nonzero_rc_raises(self):
        """BrokenPipe with encoder exiting rc != 0 must still raise."""
        mock_proc = self._make_mock_process(returncode=1)
        mock_proc.poll.return_value = None
        mock_proc.stdin.write.side_effect = BrokenPipeError(32, "Broken pipe")
        mock_proc.wait.return_value = 1
        mock_proc.returncode = 1

        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="FFmpeg encoder pipe broken"):
                with FFmpegEncoder(
                    original_path="input.mp4",
                    output_path="output.mp4",
                    width=640,
                    height=480,
                    fps=30.0,
                    hw_config=config,
                    codec="h264",
                    crf=18,
                ) as encoder:
                    encoder.write_frame(frame)
```

- [ ] **Step 2: Run the test**

Run:
```bash
python -m pytest tests/test_ffmpeg_pipe.py::TestFFmpegEncoder::test_write_frame_pipe_broken_with_nonzero_rc_raises -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ffmpeg_pipe.py
git commit -m "test(ffmpeg_pipe): keep raising when encoder crashed after pipe break"
```

---

## Task 5: Test — encoder hang after pipe break escalates to kill + raise

**Files:**
- Test: `tests/test_ffmpeg_pipe.py` (класс `TestFFmpegEncoder`).

- [ ] **Step 1: Write the test**

Добавить в `tests/test_ffmpeg_pipe.py` в класс `TestFFmpegEncoder`, после `test_write_frame_pipe_broken_with_nonzero_rc_raises`:

```python
    def test_write_frame_pipe_broken_with_hang_kills_and_raises(self):
        """BrokenPipe + wait() timeout must kill the process and raise."""
        mock_proc = self._make_mock_process()
        mock_proc.poll.return_value = None
        mock_proc.stdin.write.side_effect = BrokenPipeError(32, "Broken pipe")
        # First wait() (inside BrokenPipe handler) times out, second (after kill) returns.
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5), -9]

        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="FFmpeg encoder hung after pipe break"):
                with FFmpegEncoder(
                    original_path="input.mp4",
                    output_path="output.mp4",
                    width=640,
                    height=480,
                    fps=30.0,
                    hw_config=config,
                    codec="h264",
                    crf=18,
                ) as encoder:
                    encoder.write_frame(frame)

        mock_proc.kill.assert_called_once()
```

- [ ] **Step 2: Run the test**

Run:
```bash
python -m pytest tests/test_ffmpeg_pipe.py::TestFFmpegEncoder::test_write_frame_pipe_broken_with_hang_kills_and_raises -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ffmpeg_pipe.py
git commit -m "test(ffmpeg_pipe): escalate to kill when encoder hangs after pipe break"
```

---

## Task 6: Test — encoder exits rc=0 before write (poll path) is graceful

**Files:**
- Test: `tests/test_ffmpeg_pipe.py` (класс `TestFFmpegEncoder`).

Этот тест покрывает второй путь graceful-EOF: `poll()` уже вернул `0` ДО `stdin.write()`. Существующий `test_write_frame_after_crash_raises` покрывает `poll() → 1`. Нужен зеркальный тест на `poll() → 0`.

- [ ] **Step 1: Write the test**

Добавить в `tests/test_ffmpeg_pipe.py` в класс `TestFFmpegEncoder`, после `test_write_frame_pipe_broken_with_hang_kills_and_raises`:

```python
    def test_write_frame_after_clean_exit_is_silent(self):
        """If poll() reports rc=0 before the write, treat as EOF (no raise)."""
        mock_proc = self._make_mock_process(returncode=0)
        mock_proc.poll.return_value = 0  # encoder already exited cleanly
        mock_proc.returncode = 0

        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                crf=18,
            ) as encoder:
                encoder.write_frame(frame)  # no raise
                encoder.write_frame(frame)  # no raise, silent no-op

        # stdin.write was never called because poll() short-circuited first.
        mock_proc.stdin.write.assert_not_called()
```

- [ ] **Step 2: Run the test**

Run:
```bash
python -m pytest tests/test_ffmpeg_pipe.py::TestFFmpegEncoder::test_write_frame_after_clean_exit_is_silent -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ffmpeg_pipe.py
git commit -m "test(ffmpeg_pipe): poll()=0 before write is also graceful EOF"
```

---

## Task 7: Full test suite + lint

**Files:** no changes — проверка регрессий.

- [ ] **Step 1: Run the full test suite**

Run:
```bash
python -m pytest tests/ -v
```

Expected: все тесты проходят. Особое внимание `test_video_annotator.py` — Pass 2 render должен работать как раньше.

- [ ] **Step 2: Run any lint/type-check pipeline that exists**

Run:
```bash
# Если в проекте есть Makefile, pre-commit, mypy, ruff — запустить. Если нет — пропустить.
ls -la Makefile pyproject.toml .pre-commit-config.yaml 2>/dev/null || echo "no standard lint config"
```

Expected: нет новых ошибок. Если lint-команды отсутствуют — шаг выполнен (нечего запускать).

- [ ] **Step 3: No new commits at this step (verification only)**

---

## Task 8: Remove plan document before PR

**Files:**
- Delete: `docs/superpowers/plans/2026-04-18-fix-encoder-shortest-pipe.md`

Согласно глобальной инструкции (`~/.claude/CLAUDE.md`): план не должен попадать в PR diff. Документ остаётся доступным через историю ветки.

- [ ] **Step 1: Remove plan document**

Run:
```bash
git rm docs/superpowers/plans/2026-04-18-fix-encoder-shortest-pipe.md
```

Also remove empty parent directories if they're not used:
```bash
rmdir docs/superpowers/plans docs/superpowers 2>/dev/null || true
```

- [ ] **Step 2: Commit removal**

```bash
git commit -m "chore: remove plan doc (kept in branch history)"
```

- [ ] **Step 3: Verify branch state**

Run:
```bash
git log --oneline master..HEAD
git diff master..HEAD --stat
```

Expected в `git log --oneline`:
```
<hash> chore: remove plan doc (kept in branch history)
<hash> test(ffmpeg_pipe): poll()=0 before write is also graceful EOF
<hash> test(ffmpeg_pipe): escalate to kill when encoder hangs after pipe break
<hash> test(ffmpeg_pipe): keep raising when encoder crashed after pipe break
<hash> fix(ffmpeg_pipe): treat encoder rc=0 after BrokenPipe as clean EOF
<hash> test(ffmpeg_pipe): add failing test for -shortest graceful EOF
<hash> docs: plan for encoder -shortest pipe fix
```

Expected в `git diff --stat`:
```
 app/ffmpeg_pipe.py              | ~30 lines
 tests/test_ffmpeg_pipe.py       | ~80 lines
```

Никаких `docs/` в diff быть не должно.

---

## Task 9 (optional): Manual smoke test on AMD host

**Files:** none.

Не автоматизируется — воспроизвести можно только на сервере с AMD VAAPI и тестовым видео (`51.13.mp4`). По желанию пользователя.

- [ ] **Step 1: Build image and redeploy**

```bash
cd docker
docker compose -f docker-compose-amd.yml build
./deploy/deploy-up-amd.sh
```

- [ ] **Step 2: Submit the same video that failed in the bug report**

```bash
curl -X POST http://localhost:3001/detect/video/visualize \
  -F "file=@51.13.mp4"
# -> получим job_id, потом
curl http://localhost:3001/jobs/<job_id>
```

Expected (по завершении):
- `status: "completed"`
- `progress: 100`
- `download_url` не null
- `curl http://localhost:3001/jobs/<job_id>/download -o out.mp4` → валидный mp4 с аудио и bbox'ами.

- [ ] **Step 3: Inspect output duration**

```bash
ffprobe -v error -show_format out.mp4 | grep duration
```

Expected: длительность совпадает с `min(video_duration, audio_duration)` исходного файла (может быть на ~40–80 мс короче оригинального video track — это штатное поведение `-shortest`).

---

## Self-Review

**1. Spec coverage:**
- Обработка `BrokenPipeError` при `rc == 0` → Task 3.
- Сохранение поведения при `rc != 0` → Task 4.
- Безопасный kill при зависании → Task 5.
- Обработка `poll() == 0` → Task 6.
- Регрессии → Task 7.
- Manual check → Task 9.

Все пункты варианта A из обсуждения покрыты.

**2. Placeholder scan:** No TBD / TODO / "add appropriate error handling". Все блоки кода полные.

**3. Type consistency:**
- Поле называется `self._eof` во всех задачах.
- Импорт `subprocess.TimeoutExpired` уже есть в `ffmpeg_pipe.py` (`subprocess.Popen` используется) — `subprocess.TimeoutExpired` доступен через тот же модуль.
- `BrokenPipeError` — встроенный Python exception, импорт не нужен.
- Mock setup одинаков между тестами (`_make_mock_process` + override полей).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-18-fix-encoder-shortest-pipe.md`. Two execution options:

**1. Subagent-Driven (recommended)** — я диспатчу fresh subagent на каждую задачу, review между задачами, быстрая итерация.

**2. Inline Execution** — выполняю задачи в этой сессии через executing-plans, батчево с чекпоинтами для review.

Which approach?
