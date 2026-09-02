# Process Watchdog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A stdlib-only supervisor process inside the image restarts the container when `/health` stops answering, so a hung GPU costs about two minutes of downtime instead of days.

**Architecture:** `app/supervisor.py` becomes the container's command. It starts uvicorn as a child in its own process group, polls `/health` with a timeout, and after three consecutive failures (or a startup that never becomes healthy) SIGKILLs the process group and exits with code 3; `restart: unless-stopped` then recreates the container. A stateless short-uptime cooldown throttles restart loops, and an optional SMTP notifier mails every watchdog action. Compose files get `init: true` so tini is PID 1.

**Tech Stack:** Python 3.12+ standard library only (`subprocess`, `urllib`, `smtplib`, `signal`, `logging`); pytest with fakes (no network, no sleeping) plus two real-subprocess smoke tests; PyYAML for compose invariants.

**Spec:** `docs/superpowers/specs/2026-09-02-process-watchdog-design.md`

## Global Constraints

- `app/supervisor.py` imports **only the standard library**. Never `torch`, never anything from `app/` (`config`, `main`, …). Must run on Python 3.12 (NVIDIA/AMD images) and 3.14 (CPU image); the local venv is 3.13.
- Exit-code contract: `3` watchdog-initiated restart; child's code (`128 + N` when killed by signal `N`) when the child exited or was stopped by SIGTERM/SIGINT; `2` usage error; `127` `execvp` failed with `WATCHDOG_ENABLED=false`; `1` unexpected exception in the supervisor.
- Defaults (verbatim from the spec): `WATCHDOG_ENABLED=true`, `WATCHDOG_HEALTH_URL=http://127.0.0.1:8000/health`, `WATCHDOG_INTERVAL=30`, `WATCHDOG_TIMEOUT=10`, `WATCHDOG_FAILURES=3`, `WATCHDOG_STARTUP_TIMEOUT=600`, `WATCHDOG_MIN_UPTIME=600`, `WATCHDOG_FLAP_COOLDOWN=900`, `WATCHDOG_STOP_GRACE=8`, `WATCHDOG_MAIL_TO=` (empty), `WATCHDOG_MAIL_FROM=vision-api@<hostname>`, `WATCHDOG_SMTP_HOST=` (empty), `WATCHDOG_SMTP_PORT=587`, `WATCHDOG_SMTP_USER=`, `WATCHDOG_SMTP_PASSWORD=`, `WATCHDOG_SMTP_STARTTLS=true`.
- A malformed `WATCHDOG_*` value logs `ERROR` and falls back to the default; it never prevents startup. Empty value = unset.
- Only HTTP 200 counts as a healthy probe.
- Supervisor log lines go to stderr with the format `%(asctime)s supervisor: %(levelname)s %(message)s`; the level follows `LOG_LEVEL` when valid, else `INFO`.
- Dockerfile command in all three images, exactly: `CMD ["python3", "supervisor.py", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`.
- All six compose files get `init: true`; the three files in `docker/deploy/` additionally pass through exactly these nine variables as `${VAR:-}`: `WATCHDOG_ENABLED`, `WATCHDOG_STARTUP_TIMEOUT`, `WATCHDOG_FLAP_COOLDOWN`, `WATCHDOG_MAIL_TO`, `WATCHDOG_MAIL_FROM`, `WATCHDOG_SMTP_HOST`, `WATCHDOG_SMTP_PORT`, `WATCHDOG_SMTP_USER`, `WATCHDOG_SMTP_PASSWORD`.
- `app/config.py` is not modified.
- Tests run inside the venv: `.venv/bin/python -m pytest …` from the repo root (`tests/conftest.py` puts `app/` on `sys.path`, so tests `import supervisor`). From the main session, delegate test runs to the `claude-forge:build-runner` agent.
- Every commit message ends with the trailer line `Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68`.
- Work happens on branch `feat/process-watchdog` (already exists, contains the spec).

## File Structure

| File | Responsibility |
|---|---|
| `app/supervisor.py` (new) | One module, four units: `Config` (env parsing), `HealthProbe` (one HTTP GET), `Notifier` (`NullNotifier`/`SmtpNotifier`), `Supervisor` (state machine + `ChildProcess` wrapper), plus `main()`. Kept in one file because it must be dependency-free and copied as-is into three images. |
| `tests/test_supervisor.py` (new) | Fakes (`FakeClock`, `FakeChild`, `ScriptedProbe`, `RecordingNotifier`) and all supervisor tests, grouped by unit in the same order as the tasks. |
| `docker/{amd,cpu,nvidia}/Dockerfile` | `CMD` runs the supervisor. |
| `docker/docker-compose-{amd,cpu,nvidia}.yml` | `init: true`. |
| `docker/deploy/docker-compose-{amd,cpu,nvidia}.yml` | `init: true` + nine `WATCHDOG_*` pass-throughs. |
| `tests/test_compose.py` | Deployment invariants for the above. |
| `CLAUDE.md`, `README.md`, `.claude/rules/docker.md`, `docker/deploy/.env.example` | Operator documentation. |

---

### Task 1: Config — environment parsing with safe fallbacks

**Files:**
- Create: `app/supervisor.py`
- Create: `tests/test_supervisor.py`

**Interfaces:**
- Produces: `Config` (frozen dataclass) with fields `enabled: bool`, `health_url: str`, `interval: float`, `timeout: float`, `failures: int`, `startup_timeout: float`, `min_uptime: float`, `flap_cooldown: float`, `stop_grace: float`, `mail_to: str`, `mail_from: str`, `smtp_host: str`, `smtp_port: int`, `smtp_user: str`, `smtp_password: str`, `smtp_starttls: bool`; property `mail_enabled -> bool`; classmethod `Config.from_env(environ: Mapping[str, str] | None = None) -> Config`.
- Produces: module constants `EXIT_RESTART = 3`, `EXIT_USAGE = 2`, `EXIT_EXEC_FAILED = 127`, `EXIT_SUPERVISOR_BUG = 1`; module logger `log = logging.getLogger("supervisor")`.
- Produces (tests): helper `make_config(**overrides) -> sv.Config` with fast defaults used by every later task.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_supervisor.py`:

```python
"""Tests for app/supervisor.py — the process watchdog.

The state machine is exercised with a scripted probe, a fake child, and a fake clock, so no
test sleeps or touches the network, except the HealthProbe tests (local http.server) and the
two smoke tests at the end (a real child process).
"""
import logging
import socket

import pytest

import supervisor as sv


def make_config(**overrides) -> sv.Config:
    """Fast defaults for state-machine tests; every value overridable."""
    base = dict(
        interval=1.0,
        timeout=1.0,
        failures=3,
        startup_timeout=10.0,
        min_uptime=100.0,
        flap_cooldown=50.0,
        stop_grace=2.0,
    )
    base.update(overrides)
    return sv.Config(**base)


# --------------------------------------------------------------------------- Config


def test_config_defaults_when_environment_is_empty():
    cfg = sv.Config.from_env({})
    assert cfg.enabled is True
    assert cfg.health_url == "http://127.0.0.1:8000/health"
    assert cfg.interval == 30.0
    assert cfg.timeout == 10.0
    assert cfg.failures == 3
    assert cfg.startup_timeout == 600.0
    assert cfg.min_uptime == 600.0
    assert cfg.flap_cooldown == 900.0
    assert cfg.stop_grace == 8.0
    assert cfg.mail_to == ""
    assert cfg.mail_from == f"vision-api@{socket.gethostname()}"
    assert cfg.smtp_host == ""
    assert cfg.smtp_port == 587
    assert cfg.smtp_user == ""
    assert cfg.smtp_password == ""
    assert cfg.smtp_starttls is True
    assert cfg.mail_enabled is False


def test_config_reads_every_variable():
    env = {
        "WATCHDOG_ENABLED": "false",
        "WATCHDOG_HEALTH_URL": "http://127.0.0.1:9000/health",
        "WATCHDOG_INTERVAL": "5",
        "WATCHDOG_TIMEOUT": "2.5",
        "WATCHDOG_FAILURES": "2",
        "WATCHDOG_STARTUP_TIMEOUT": "120",
        "WATCHDOG_MIN_UPTIME": "0",
        "WATCHDOG_FLAP_COOLDOWN": "0",
        "WATCHDOG_STOP_GRACE": "3",
        "WATCHDOG_MAIL_TO": "ops@example.com",
        "WATCHDOG_MAIL_FROM": "bot@example.com",
        "WATCHDOG_SMTP_HOST": "smtp.example.com",
        "WATCHDOG_SMTP_PORT": "2525",
        "WATCHDOG_SMTP_USER": "user",
        "WATCHDOG_SMTP_PASSWORD": "secret",
        "WATCHDOG_SMTP_STARTTLS": "no",
    }
    cfg = sv.Config.from_env(env)
    assert cfg.enabled is False
    assert cfg.health_url == "http://127.0.0.1:9000/health"
    assert cfg.interval == 5.0
    assert cfg.timeout == 2.5
    assert cfg.failures == 2
    assert cfg.startup_timeout == 120.0
    assert cfg.min_uptime == 0.0
    assert cfg.flap_cooldown == 0.0
    assert cfg.stop_grace == 3.0
    assert cfg.mail_to == "ops@example.com"
    assert cfg.mail_from == "bot@example.com"
    assert cfg.smtp_host == "smtp.example.com"
    assert cfg.smtp_port == 2525
    assert cfg.smtp_user == "user"
    assert cfg.smtp_password == "secret"
    assert cfg.smtp_starttls is False
    assert cfg.mail_enabled is True


@pytest.mark.parametrize(
    "name,value",
    [
        ("WATCHDOG_INTERVAL", "abc"),
        ("WATCHDOG_INTERVAL", "0"),
        ("WATCHDOG_TIMEOUT", "-1"),
        ("WATCHDOG_FAILURES", "0"),
        ("WATCHDOG_FAILURES", "1.5"),
        ("WATCHDOG_STARTUP_TIMEOUT", "0"),
        ("WATCHDOG_MIN_UPTIME", "-5"),
        ("WATCHDOG_FLAP_COOLDOWN", "x"),
        ("WATCHDOG_STOP_GRACE", "-0.1"),
        ("WATCHDOG_SMTP_PORT", "70000"),
        ("WATCHDOG_SMTP_PORT", "0"),
        ("WATCHDOG_ENABLED", "maybe"),
        ("WATCHDOG_SMTP_STARTTLS", "2"),
    ],
)
def test_config_invalid_value_logs_error_and_keeps_default(caplog, name, value):
    with caplog.at_level(logging.ERROR, logger="supervisor"):
        cfg = sv.Config.from_env({name: value})
    assert cfg == sv.Config.from_env({}), (name, value)
    assert name in caplog.text
    assert "using default" in caplog.text


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("1", True), ("yes", True), ("ON", True),
    ("false", False), ("0", False), ("No", False), ("off", False),
    ("", True), ("   ", True),
])
def test_config_boolean_spellings(raw, expected):
    assert sv.Config.from_env({"WATCHDOG_ENABLED": raw}).enabled is expected


def test_config_mail_requires_smtp_host(caplog):
    with caplog.at_level(logging.ERROR, logger="supervisor"):
        cfg = sv.Config.from_env({"WATCHDOG_MAIL_TO": "ops@example.com"})
    assert cfg.mail_to == "ops@example.com"
    assert cfg.mail_enabled is False
    assert "WATCHDOG_SMTP_HOST is required" in caplog.text


def test_config_empty_values_mean_unset():
    cfg = sv.Config.from_env({"WATCHDOG_INTERVAL": "", "WATCHDOG_HEALTH_URL": "  ", "WATCHDOG_MAIL_TO": ""})
    assert cfg == sv.Config.from_env({})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'supervisor'`.

- [ ] **Step 3: Write the module with `Config`**

Create `app/supervisor.py`:

```python
"""Process watchdog for the Vision API container.

Why a separate process
----------------------
On 2026-08-30 the production container hung for 2.5 days. The ROCm runtime
busy-spun inside a ThreadPoolExecutor worker that held the GIL, so nothing in
the uvicorn process could run: not the event loop, not ``/health``, not a
timeout check. Docker's healthcheck failed 5630 times in a row, but
``restart: unless-stopped`` only fires when the process *exits*, and a frozen
process never exits.

Three rules follow, and this module is built on them:

* The watchdog is a separate **process**. A thread, an asyncio task, or a
  Python signal handler cannot run while another thread holds the GIL.
* uvicorn must **not be PID 1**. The kernel ignores SIGKILL sent to PID 1
  from inside its own PID namespace (``pid_namespaces(7)``), so the
  supervisor owns uvicorn as a child and kills the child instead.
* **Standard library only, no application imports.** An application problem
  must not be able to take the watchdog down with it.

Usage::

    python3 supervisor.py uvicorn main:app --host 0.0.0.0 --port 8000

Every argument after the script name is the child command. Configuration is
read from ``WATCHDOG_*`` environment variables (see ``Config``).

Exit codes
----------
* ``3``   - watchdog-initiated restart (hung ``/health`` or startup timeout);
            Docker's restart policy recreates the container
* child's code - the child exited on its own or after SIGTERM/SIGINT;
            ``128 + N`` when the child died from signal ``N``
* ``2``   - usage error (no child command)
* ``127`` - ``WATCHDOG_ENABLED=false`` and ``execvp`` of the child failed
* ``1``   - unexpected exception inside the supervisor
"""

import dataclasses
import logging
import os
import socket
from collections.abc import Mapping

log = logging.getLogger("supervisor")

EXIT_RESTART = 3
EXIT_USAGE = 2
EXIT_EXEC_FAILED = 127
EXIT_SUPERVISOR_BUG = 1

_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})


def _parse_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    if raw.lower() in _TRUE:
        return True
    if raw.lower() in _FALSE:
        return False
    log.error("%s=%r is not a boolean; using default %r", name, raw, default)
    return default


def _parse_number(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    integer: bool = False,
    minimum: float = 0,
    exclusive: bool = False,
    maximum: float | None = None,
) -> float:
    """Parse a numeric variable; log and fall back to ``default`` on anything invalid."""
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        kind = "an integer" if integer else "a number"
        log.error("%s=%r is not %s; using default %r", name, raw, kind, default)
        return default
    if value < minimum or (exclusive and value == minimum):
        bound = ">" if exclusive else ">="
        log.error("%s=%r must be %s %s; using default %r", name, raw, bound, minimum, default)
        return default
    if maximum is not None and value > maximum:
        log.error("%s=%r must be <= %s; using default %r", name, raw, maximum, default)
        return default
    return value


@dataclasses.dataclass(frozen=True)
class Config:
    """Watchdog settings. ``from_env`` is the only constructor used in production."""

    enabled: bool = True
    health_url: str = "http://127.0.0.1:8000/health"
    interval: float = 30.0          # seconds between probes
    timeout: float = 10.0           # HTTP timeout of one probe
    failures: int = 3               # consecutive failures before reacting
    startup_timeout: float = 600.0  # wait for the first healthy answer (cold MIOpen compile)
    min_uptime: float = 600.0       # a hang before this uptime counts as flapping
    flap_cooldown: float = 900.0    # delay before restarting a flapping container; 0 = off
    stop_grace: float = 8.0         # seconds to wait after SIGTERM/SIGKILL before giving up
    mail_to: str = ""               # non-empty enables e-mail notification
    mail_from: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True

    @property
    def mail_enabled(self) -> bool:
        return bool(self.mail_to and self.smtp_host)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Config":
        env = os.environ if environ is None else environ
        d = cls()
        mail_to = env.get("WATCHDOG_MAIL_TO", "").strip()
        smtp_host = env.get("WATCHDOG_SMTP_HOST", "").strip()
        if mail_to and not smtp_host:
            log.error("mail disabled: WATCHDOG_SMTP_HOST is required when WATCHDOG_MAIL_TO is set")
        return cls(
            enabled=_parse_bool(env, "WATCHDOG_ENABLED", d.enabled),
            health_url=env.get("WATCHDOG_HEALTH_URL", "").strip() or d.health_url,
            interval=_parse_number(env, "WATCHDOG_INTERVAL", d.interval, exclusive=True),
            timeout=_parse_number(env, "WATCHDOG_TIMEOUT", d.timeout, exclusive=True),
            failures=int(_parse_number(env, "WATCHDOG_FAILURES", d.failures, integer=True, minimum=1)),
            startup_timeout=_parse_number(env, "WATCHDOG_STARTUP_TIMEOUT", d.startup_timeout, exclusive=True),
            min_uptime=_parse_number(env, "WATCHDOG_MIN_UPTIME", d.min_uptime),
            flap_cooldown=_parse_number(env, "WATCHDOG_FLAP_COOLDOWN", d.flap_cooldown),
            stop_grace=_parse_number(env, "WATCHDOG_STOP_GRACE", d.stop_grace),
            mail_to=mail_to,
            mail_from=env.get("WATCHDOG_MAIL_FROM", "").strip() or f"vision-api@{socket.gethostname()}",
            smtp_host=smtp_host,
            smtp_port=int(_parse_number(env, "WATCHDOG_SMTP_PORT", d.smtp_port, integer=True, minimum=1, maximum=65535)),
            smtp_user=env.get("WATCHDOG_SMTP_USER", "").strip(),
            smtp_password=env.get("WATCHDOG_SMTP_PASSWORD", ""),
            smtp_starttls=_parse_bool(env, "WATCHDOG_SMTP_STARTTLS", d.smtp_starttls),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -v`
Expected: all Config tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/supervisor.py tests/test_supervisor.py
git commit -F - <<'EOF'
feat(supervisor): add watchdog Config with safe env parsing

Standard-library-only module skeleton for the process watchdog. Every
WATCHDOG_* variable falls back to its default on a malformed value with an
ERROR log line, so a typo cannot keep the service from starting.

Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68
EOF
```

---

### Task 2: HealthProbe — one HTTP GET, True only for 200

**Files:**
- Modify: `app/supervisor.py` (imports + new class after `Config`)
- Modify: `tests/test_supervisor.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks besides `log`.
- Produces: `HealthProbe(url: str, timeout: float)`, callable `probe() -> bool`; attribute `last_failure: str` describing the most recent `False` (empty after a success).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_supervisor.py`. Add these imports at the top of the file (keep the existing ones):

```python
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
```

Then append:

```python
# --------------------------------------------------------------------------- HealthProbe


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/ok":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        elif self.path == "/slow":
            time.sleep(1.0)
            self.send_response(200)
            self.end_headers()
        elif self.path == "/created":
            self.send_response(201)
            self.end_headers()
        else:
            self.send_response(500)
            self.end_headers()

    def log_message(self, *args):  # silence the test output
        pass


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # the /slow handler hits a closed socket
        pass


@pytest.fixture
def http_base_url():
    server = _QuietServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_probe_returns_true_for_200(http_base_url):
    probe = sv.HealthProbe(f"{http_base_url}/ok", timeout=2.0)
    assert probe() is True
    assert probe.last_failure == ""


def test_probe_returns_false_for_500(http_base_url):
    probe = sv.HealthProbe(f"{http_base_url}/error", timeout=2.0)
    assert probe() is False
    assert probe.last_failure == "HTTP 500"


def test_probe_returns_false_for_non_200_success_codes(http_base_url):
    probe = sv.HealthProbe(f"{http_base_url}/created", timeout=2.0)
    assert probe() is False
    assert probe.last_failure == "HTTP 201"


def test_probe_returns_false_when_nothing_listens():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    probe = sv.HealthProbe(f"http://127.0.0.1:{free_port}/health", timeout=2.0)
    assert probe() is False
    assert probe.last_failure != ""


def test_probe_returns_false_on_timeout(http_base_url):
    probe = sv.HealthProbe(f"{http_base_url}/slow", timeout=0.2)
    started = time.monotonic()
    assert probe() is False
    assert time.monotonic() - started < 1.0
    assert probe.last_failure == "timed out after 0.2s"


def test_probe_success_clears_last_failure(http_base_url):
    probe = sv.HealthProbe(f"{http_base_url}/error", timeout=2.0)
    assert probe() is False
    probe.url = f"{http_base_url}/ok"
    assert probe() is True
    assert probe.last_failure == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -k probe -v`
Expected: FAIL with `AttributeError: module 'supervisor' has no attribute 'HealthProbe'`.

- [ ] **Step 3: Implement `HealthProbe`**

In `app/supervisor.py`, extend the imports:

```python
import dataclasses
import http.client
import logging
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping
```

Append after the `Config` class:

```python
class HealthProbe:
    """One HTTP GET of the health URL. ``True`` only for HTTP 200.

    ``last_failure`` holds a short reason for the most recent ``False`` (empty
    after a success) so the supervisor can log it.
    """

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self.last_failure = ""

    def __call__(self) -> bool:
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
                if response.status == 200:
                    self.last_failure = ""
                    return True
                self.last_failure = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            self.last_failure = f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            # A connect timeout arrives wrapped in URLError(reason=TimeoutError); a read
            # timeout (the hang signature: connection accepted, no answer) arrives bare.
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                self.last_failure = f"timed out after {self.timeout:g}s"
            else:
                self.last_failure = f"{type(reason).__name__}: {reason}"
        except Exception as exc:  # noqa: BLE001 - a probe must never propagate
            log.warning("health probe raised %s: %s", type(exc).__name__, exc)
            self.last_failure = f"{type(exc).__name__}: {exc}"
        return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -v`
Expected: all PASS, including the six probe tests.

- [ ] **Step 5: Commit**

```bash
git add app/supervisor.py tests/test_supervisor.py
git commit -F - <<'EOF'
feat(supervisor): add HealthProbe (HTTP 200 or failure with reason)

Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68
EOF
```

---

### Task 3: Notifier — RestartEvent, NullNotifier, SmtpNotifier

**Files:**
- Modify: `app/supervisor.py` (imports + new classes after `HealthProbe`)
- Modify: `tests/test_supervisor.py` (append)

**Interfaces:**
- Consumes: `Config` (fields `mail_to`, `mail_from`, `smtp_host`, `smtp_port`, `smtp_user`, `smtp_password`, `smtp_starttls`, `health_url`).
- Produces: `RestartEvent(reason: str, uptime: float, failures: int, exit_code: int, detail: str = "")` frozen dataclass; `NullNotifier().notify(event) -> None`; `SmtpNotifier(config, *, child_cmd: list[str], hostname: str | None = None, smtp_factory=smtplib.SMTP)` with `notify(event) -> None` (never raises) and `build_message(event) -> EmailMessage`; constant `SMTP_TIMEOUT = 10.0`.
- Produces (tests): `RecordingNotifier` and `RaisingNotifier` fakes used by Task 4+.

- [ ] **Step 1: Write the failing tests**

Add to the imports at the top of `tests/test_supervisor.py`:

```python
from unittest.mock import MagicMock
```

Append:

```python
# --------------------------------------------------------------------------- Notifier


class RecordingNotifier:
    def __init__(self):
        self.events: list[sv.RestartEvent] = []

    def notify(self, event):
        self.events.append(event)


class RaisingNotifier:
    def notify(self, event):
        raise RuntimeError("smtp exploded")


def _event(**overrides) -> sv.RestartEvent:
    base = dict(reason="health_failed", uptime=3600.0, failures=3, exit_code=3, detail="timed out after 10s")
    base.update(overrides)
    return sv.RestartEvent(**base)


def _smtp_double():
    """A MagicMock usable as ``with factory(...) as smtp:``."""
    smtp = MagicMock()
    smtp.__enter__.return_value = smtp
    factory = MagicMock(return_value=smtp)
    return factory, smtp


def test_null_notifier_does_nothing():
    sv.NullNotifier().notify(_event())  # must not raise


def test_smtp_notifier_sends_message_with_starttls_and_login():
    cfg = make_config(
        health_url="http://127.0.0.1:8000/health",
        mail_to="ops@example.com", mail_from="bot@example.com",
        smtp_host="smtp.example.com", smtp_port=2525,
        smtp_user="user", smtp_password="secret", smtp_starttls=True,
    )
    factory, smtp = _smtp_double()
    notifier = sv.SmtpNotifier(cfg, child_cmd=["uvicorn", "main:app"], hostname="c0ffee", smtp_factory=factory)

    notifier.notify(_event())

    factory.assert_called_once_with("smtp.example.com", 2525, timeout=sv.SMTP_TIMEOUT)
    smtp.starttls.assert_called_once_with()
    smtp.login.assert_called_once_with("user", "secret")
    smtp.send_message.assert_called_once()
    message = smtp.send_message.call_args.args[0]
    assert message["Subject"] == "[vision-api] restart: health_failed (c0ffee)"
    assert message["From"] == "bot@example.com"
    assert message["To"] == "ops@example.com"
    body = message.get_content()
    assert "reason: health_failed" in body
    assert "uptime: 3600s" in body
    assert "consecutive failures: 3" in body
    assert "detail: timed out after 10s" in body
    assert "health URL: http://127.0.0.1:8000/health" in body
    assert "child command: uvicorn main:app" in body
    assert "exit code: 3" in body
    assert "RestartCount" in body


def test_smtp_notifier_skips_starttls_and_login_when_not_configured():
    cfg = make_config(mail_to="ops@example.com", smtp_host="smtp.example.com", smtp_starttls=False)
    factory, smtp = _smtp_double()
    notifier = sv.SmtpNotifier(cfg, child_cmd=["uvicorn"], hostname="c0ffee", smtp_factory=factory)

    notifier.notify(_event(reason="startup_timeout"))

    smtp.starttls.assert_not_called()
    smtp.login.assert_not_called()
    message = smtp.send_message.call_args.args[0]
    assert message["Subject"] == "[vision-api] restart: startup_timeout (c0ffee)"
    assert message["From"] == "vision-api@c0ffee"  # mail_from empty -> hostname fallback


def test_smtp_notifier_swallows_errors_and_logs_warning(caplog):
    cfg = make_config(mail_to="ops@example.com", smtp_host="smtp.example.com")
    factory = MagicMock(side_effect=OSError("connection refused"))
    notifier = sv.SmtpNotifier(cfg, child_cmd=["uvicorn"], hostname="c0ffee", smtp_factory=factory)

    with caplog.at_level(logging.WARNING, logger="supervisor"):
        notifier.notify(_event())  # must not raise

    assert "notification failed" in caplog.text
    assert "connection refused" in caplog.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -k notifier -v`
Expected: FAIL with `AttributeError: module 'supervisor' has no attribute 'RestartEvent'` (or `NullNotifier`).

- [ ] **Step 3: Implement the notifier**

Extend the imports in `app/supervisor.py`:

```python
import dataclasses
import http.client
import logging
import os
import smtplib
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from email.message import EmailMessage
```

Add the constant next to the exit codes:

```python
SMTP_TIMEOUT = 10.0
```

Append after `HealthProbe`:

```python
@dataclasses.dataclass(frozen=True)
class RestartEvent:
    """What the supervisor is about to do and why; handed to the notifier."""

    reason: str        # "health_failed" | "startup_timeout" | "child_exited"
    uptime: float      # seconds since the child was started
    failures: int      # consecutive failed probes at the time of the event
    exit_code: int     # code the supervisor is about to exit with
    detail: str = ""   # last probe failure, or the child's exit code


class NullNotifier:
    """Used when ``WATCHDOG_MAIL_TO`` is empty."""

    def notify(self, event: RestartEvent) -> None:
        return None


class SmtpNotifier:
    """Mails one message per watchdog action. Never raises: a broken relay must not delay a restart."""

    def __init__(
        self,
        config: Config,
        *,
        child_cmd: list[str],
        hostname: str | None = None,
        smtp_factory=smtplib.SMTP,
    ) -> None:
        self._cfg = config
        self._child_cmd = list(child_cmd)
        self._hostname = hostname or socket.gethostname()
        self._smtp_factory = smtp_factory

    def notify(self, event: RestartEvent) -> None:
        try:
            message = self.build_message(event)
            with self._smtp_factory(self._cfg.smtp_host, self._cfg.smtp_port, timeout=SMTP_TIMEOUT) as smtp:
                if self._cfg.smtp_starttls:
                    smtp.starttls()
                if self._cfg.smtp_user:
                    smtp.login(self._cfg.smtp_user, self._cfg.smtp_password)
                smtp.send_message(message)
            log.info("notification sent to %s", self._cfg.mail_to)
        except Exception as exc:  # noqa: BLE001 - notification must never block the restart
            log.warning("notification failed: %s: %s", type(exc).__name__, exc)

    def build_message(self, event: RestartEvent) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = f"[vision-api] restart: {event.reason} ({self._hostname})"
        message["From"] = self._cfg.mail_from or f"vision-api@{self._hostname}"
        message["To"] = self._cfg.mail_to
        message.set_content(
            "\n".join(
                [
                    f"The vision-api supervisor is restarting the container on {self._hostname}.",
                    "",
                    f"reason: {event.reason}",
                    f"time (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
                    f"uptime: {event.uptime:.0f}s",
                    f"consecutive failures: {event.failures}",
                    f"detail: {event.detail or '-'}",
                    f"health URL: {self._cfg.health_url}",
                    f"child command: {' '.join(self._child_cmd)}",
                    f"exit code: {event.exit_code}",
                    "",
                    "Check `docker logs <container>` for the supervisor lines and",
                    "`docker inspect -f '{{.RestartCount}}' <container>` for the restart count.",
                ]
            )
        )
        return message
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/supervisor.py tests/test_supervisor.py
git commit -F - <<'EOF'
feat(supervisor): add RestartEvent and SMTP/Null notifiers

Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68
EOF
```

---

### Task 4: Supervisor core — child lifecycle, STARTING/RUNNING, restart, stop handling

**Files:**
- Modify: `app/supervisor.py` (imports + `ChildProcess`, `State`, `normalize_exit_code`, `Supervisor`)
- Modify: `tests/test_supervisor.py` (append fakes + tests)

**Interfaces:**
- Consumes: `Config`, `RestartEvent`, `EXIT_RESTART`, `log`; probe protocol `Callable[[], bool]` with optional `last_failure: str`; notifier protocol `notify(RestartEvent) -> None`.
- Produces: `ChildProcess(cmd: list[str])` with `pid: int`, `wait(timeout: float) -> int | None`, `send_signal(signum: int) -> None`, `kill_group() -> None`; `State` enum (`STARTING`, `RUNNING`, `PENDING_RESTART`); `normalize_exit_code(returncode: int) -> int`; `Supervisor(config, child_cmd, *, probe, notifier, clock=time.monotonic, child_factory=ChildProcess, tick=1.0)` with `run() -> int`, `request_stop(signum: int) -> None`, `kill_child() -> None`, property `child_pid -> int | None`.
- Produces (tests): `FakeClock`, `FakeChild`, `ScriptedProbe`, `run_until_stopped(...)`.
- Note: in this task `_on_probe` has no PENDING_RESTART branch; Task 5 replaces the method.

- [ ] **Step 1: Write the failing tests**

Add to the imports at the top of `tests/test_supervisor.py`:

```python
import signal
```

Append:

```python
# --------------------------------------------------------------------------- Supervisor fakes


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeChild:
    """Scripted child. Exits with ``exit_code`` once the fake clock reaches ``exit_at``.

    ``send_signal`` schedules a clean exit (code 0) at once unless ``ignores_signals``;
    ``kill_group`` schedules exit code -9 (Popen's "killed by signal 9") unless ``unkillable``.
    """

    def __init__(self, clock: FakeClock, *, exit_code: int | None = None, exit_at: float | None = None,
                 ignores_signals: bool = False, unkillable: bool = False):
        self.pid = 4242
        self.clock = clock
        self.exit_code = exit_code
        self.exit_at = exit_at
        self.ignores_signals = ignores_signals
        self.unkillable = unkillable
        self.signals: list[int] = []
        self.group_killed = False
        self._returncode: int | None = None

    def _exited(self) -> bool:
        return self.exit_at is not None and self.exit_at <= self.clock.now

    def wait(self, timeout: float) -> int | None:
        if self._returncode is not None:
            return self._returncode
        if not self._exited():
            self.clock.advance(timeout)
        if self._exited():
            self._returncode = self.exit_code
            return self._returncode
        return None

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)
        if not self.ignores_signals and self._returncode is None:
            self.exit_code, self.exit_at = 0, self.clock.now

    def kill_group(self) -> None:
        self.group_killed = True
        if not self.unkillable and self._returncode is None:
            self.exit_code, self.exit_at = -9, self.clock.now


class ScriptedProbe:
    """Returns ``results`` in order (the last one repeats); ``hooks`` maps call number -> callable."""

    def __init__(self, results, hooks=None):
        self.results = list(results)
        self.hooks = hooks or {}
        self.calls = 0
        self.last_failure = "scripted failure"

    def __call__(self) -> bool:
        self.calls += 1
        hook = self.hooks.get(self.calls)
        if hook is not None:
            hook()
        return self.results[min(self.calls - 1, len(self.results) - 1)]


def make_supervisor(cfg, child, probe, *, clock, notifier=None):
    return sv.Supervisor(
        cfg, ["child", "--flag"],
        probe=probe, notifier=notifier or RecordingNotifier(), clock=clock,
        child_factory=lambda cmd: child, tick=1.0,
    )


def run_until_stopped(cfg, clock, child, results, stop_at_call, notifier=None):
    """Run with a scripted probe that requests a graceful stop at probe call ``stop_at_call``."""
    holder = {}
    probe = ScriptedProbe(results, hooks={stop_at_call: lambda: holder["sup"].request_stop(signal.SIGTERM)})
    sup = make_supervisor(cfg, child, probe, clock=clock, notifier=notifier)
    holder["sup"] = sup
    return sup.run(), probe


# --------------------------------------------------------------------------- Supervisor: core


def test_healthy_service_is_left_alone():
    clock, notifier = FakeClock(), RecordingNotifier()
    child = FakeChild(clock)
    code, probe = run_until_stopped(make_config(), clock, child, [True], stop_at_call=50, notifier=notifier)
    assert code == 0
    assert probe.calls == 50
    assert child.group_killed is False
    assert child.signals == [signal.SIGTERM]
    assert notifier.events == []


def test_starting_ignores_failures_until_first_success(caplog):
    clock = FakeClock()
    child = FakeChild(clock)
    with caplog.at_level(logging.INFO, logger="supervisor"):
        code, _ = run_until_stopped(make_config(), clock, child, [False] * 5 + [True], stop_at_call=8)
    assert code == 0
    assert child.group_killed is False
    assert "healthy after 6s" in caplog.text


def test_startup_timeout_restarts(caplog):
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    notifier = RecordingNotifier()
    sup = make_supervisor(make_config(startup_timeout=10.0), child, ScriptedProbe([False]), clock=clock, notifier=notifier)
    with caplog.at_level(logging.ERROR, logger="supervisor"):
        code = sup.run()
    assert code == sv.EXIT_RESTART == 3
    assert child.group_killed is True
    assert clock.now == pytest.approx(1010.0)
    assert [e.reason for e in notifier.events] == ["startup_timeout"]
    assert notifier.events[0].exit_code == 3
    assert notifier.events[0].uptime == pytest.approx(10.0)
    assert "reason=startup_timeout" in caplog.text


def test_running_success_resets_failure_counter():
    clock = FakeClock()
    child = FakeChild(clock)
    notifier = RecordingNotifier()
    results = [True, False, False, True, False, False, True]
    code, _ = run_until_stopped(make_config(min_uptime=0.0), clock, child, results, stop_at_call=10, notifier=notifier)
    assert code == 0
    assert child.group_killed is False
    assert notifier.events == []


def test_three_failures_at_long_uptime_restart_immediately(caplog):
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    notifier = RecordingNotifier()
    probe = ScriptedProbe([True, False, False, False])
    sup = make_supervisor(make_config(min_uptime=0.0), child, probe, clock=clock, notifier=notifier)
    with caplog.at_level(logging.WARNING, logger="supervisor"):
        code = sup.run()
    assert code == 3
    assert child.group_killed is True
    assert clock.now == pytest.approx(1004.0)
    assert [e.reason for e in notifier.events] == ["health_failed"]
    assert notifier.events[0].failures == 3
    assert notifier.events[0].detail == "scripted failure"
    assert "health probe failed (3/3): scripted failure" in caplog.text
    assert "consecutive_failures=3" in caplog.text


def test_child_exit_code_is_propagated_and_notified():
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock, exit_code=7, exit_at=1005.0)
    notifier = RecordingNotifier()
    sup = make_supervisor(make_config(), child, ScriptedProbe([True]), clock=clock, notifier=notifier)
    assert sup.run() == 7
    assert child.group_killed is False
    assert [e.reason for e in notifier.events] == ["child_exited"]
    assert notifier.events[0].exit_code == 7
    assert notifier.events[0].detail == "child exit code 7"


def test_child_killed_by_signal_maps_to_128_plus_signum():
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock, exit_code=-9, exit_at=1005.0)
    sup = make_supervisor(make_config(), child, ScriptedProbe([True]), clock=clock)
    assert sup.run() == 137


def test_normalize_exit_code():
    assert sv.normalize_exit_code(0) == 0
    assert sv.normalize_exit_code(7) == 7
    assert sv.normalize_exit_code(-9) == 137
    assert sv.normalize_exit_code(-15) == 143


def test_sigterm_is_forwarded_to_child_only():
    clock = FakeClock()
    child = FakeChild(clock)
    notifier = RecordingNotifier()
    code, _ = run_until_stopped(make_config(), clock, child, [True], stop_at_call=3, notifier=notifier)
    assert code == 0
    assert child.signals == [signal.SIGTERM]
    assert child.group_killed is False
    assert notifier.events == []


def test_child_ignoring_sigterm_is_killed_after_grace(caplog):
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock, ignores_signals=True)
    notifier = RecordingNotifier()
    with caplog.at_level(logging.WARNING, logger="supervisor"):
        code, _ = run_until_stopped(make_config(stop_grace=2.0), clock, child, [True], stop_at_call=3, notifier=notifier)
    assert code == 137
    assert child.signals == [signal.SIGTERM]
    assert child.group_killed is True
    assert clock.now == pytest.approx(1005.0)  # stop requested at 1003, grace 2s
    assert notifier.events == []
    assert "did not exit 2s after signal" in caplog.text


def test_second_signal_kills_group_immediately():
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock, ignores_signals=True)
    holder = {}

    def twice():
        holder["sup"].request_stop(signal.SIGTERM)
        holder["sup"].request_stop(signal.SIGINT)

    probe = ScriptedProbe([True], hooks={3: twice})
    sup = make_supervisor(make_config(stop_grace=60.0), child, probe, clock=clock)
    holder["sup"] = sup
    assert sup.run() == 137
    assert child.group_killed is True
    assert clock.now == pytest.approx(1003.0)


def test_raising_notifier_does_not_prevent_restart(caplog):
    clock = FakeClock()
    child = FakeChild(clock)
    probe = ScriptedProbe([True, False, False, False])
    sup = make_supervisor(make_config(min_uptime=0.0), child, probe, clock=clock, notifier=RaisingNotifier())
    with caplog.at_level(logging.WARNING, logger="supervisor"):
        assert sup.run() == 3
    assert child.group_killed is True
    assert "notifier raised RuntimeError: smtp exploded" in caplog.text


def test_unkillable_child_still_exits_with_restart_code(caplog):
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock, unkillable=True)
    probe = ScriptedProbe([True, False, False, False])
    sup = make_supervisor(make_config(min_uptime=0.0, stop_grace=2.0), child, probe, clock=clock)
    with caplog.at_level(logging.ERROR, logger="supervisor"):
        assert sup.run() == 3
    assert child.group_killed is True
    assert "still alive 2s after SIGKILL" in caplog.text


def test_child_pid_property():
    clock = FakeClock()
    child = FakeChild(clock)
    holder = {}
    probe = ScriptedProbe([True], hooks={1: lambda: holder["sup"].request_stop(signal.SIGTERM)})
    sup = make_supervisor(make_config(), child, probe, clock=clock)
    holder["sup"] = sup
    assert sup.child_pid is None
    sup.run()
    assert sup.child_pid == 4242
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -v`
Expected: every new test in this task FAILS with `AttributeError: module 'supervisor' has no attribute 'Supervisor'` (or `'normalize_exit_code'`); the Config, probe and notifier tests from earlier tasks still PASS.

- [ ] **Step 3: Implement `ChildProcess`, `State`, `normalize_exit_code`, `Supervisor`**

Extend the imports in `app/supervisor.py`:

```python
import dataclasses
import enum
import http.client
import logging
import os
import signal
import smtplib
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from email.message import EmailMessage
```

Append after `SmtpNotifier`:

```python
class ChildProcess:
    """The real child: a thin wrapper around ``subprocess.Popen``.

    The child gets its own session so that ``kill_group`` reaches uvicorn and
    every ffmpeg it spawned in one call. Unit tests replace this class with a
    scripted fake exposing the same four members.
    """

    def __init__(self, cmd: list[str]) -> None:
        self._proc = subprocess.Popen(cmd, start_new_session=True)
        self.pid = self._proc.pid

    def wait(self, timeout: float) -> int | None:
        """Exit status, or ``None`` if the child is still running after ``timeout`` seconds."""
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def send_signal(self, signum: int) -> None:
        try:
            self._proc.send_signal(signum)
        except ProcessLookupError:
            pass

    def kill_group(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class State(enum.Enum):
    STARTING = "starting"                # until the first HTTP 200
    RUNNING = "running"                  # healthy, counting consecutive failures
    PENDING_RESTART = "pending_restart"  # hang detected too soon after start; cooling down


def normalize_exit_code(returncode: int) -> int:
    """Map Popen's negative "killed by signal N" to the shell convention 128 + N."""
    return returncode if returncode >= 0 else 128 - returncode


class Supervisor:
    """Runs the child, probes ``/health``, and decides when the container must restart.

    Everything with side effects (child, probe, notifier, clock) is injected so the state
    machine can be tested with fakes and a virtual clock.
    """

    def __init__(
        self,
        config: Config,
        child_cmd: list[str],
        *,
        probe: Callable[[], bool],
        notifier,
        clock: Callable[[], float] = time.monotonic,
        child_factory: Callable[[list[str]], ChildProcess] = ChildProcess,
        tick: float = 1.0,
    ) -> None:
        self._cfg = config
        self._cmd = list(child_cmd)
        self._probe = probe
        self._notifier = notifier
        self._clock = clock
        self._child_factory = child_factory
        self._tick = tick  # granularity of the idle wait: child exits and signals are noticed within one tick
        self._child: ChildProcess | None = None
        self._state = State.STARTING
        self._failures = 0
        self._started_at = 0.0
        self._next_probe_at = 0.0
        self._stop_signal: int | None = None
        self._stop_deadline = 0.0
        self._group_killed = False

    @property
    def child_pid(self) -> int | None:
        return None if self._child is None else self._child.pid

    # -- public control ------------------------------------------------------------------------

    def request_stop(self, signum: int) -> None:
        """Forward ``signum`` to the child; a second call kills the process group at once.

        Called from the SIGTERM/SIGINT handlers installed by ``main()`` and directly by tests.
        """
        if self._stop_signal is not None:
            log.warning("second signal %d received; killing the process group now", signum)
            self.kill_child()
            return
        self._stop_signal = signum
        self._stop_deadline = self._clock() + self._cfg.stop_grace
        if self._child is None:
            return  # run() forwards the signal as soon as the child exists
        log.info("forwarding signal %d to child pid %d", signum, self._child.pid)
        self._child.send_signal(signum)

    def kill_child(self) -> None:
        if self._child is not None and not self._group_killed:
            self._group_killed = True
            self._child.kill_group()

    def run(self) -> int:
        """Block until the container must exit; return the exit code."""
        self._child = self._child_factory(self._cmd)
        self._started_at = self._clock()
        self._next_probe_at = self._started_at + self._cfg.interval
        log.info("started child pid %d: %s", self._child.pid, " ".join(self._cmd))
        if self._stop_signal is not None:  # a signal arrived while the child was being spawned
            self._child.send_signal(self._stop_signal)
        while True:
            returncode = self._child.wait(timeout=self._tick)
            if returncode is not None:
                return self._on_child_exit(returncode)
            now = self._clock()
            if self._stop_signal is not None:
                if now >= self._stop_deadline and not self._group_killed:
                    log.warning(
                        "child did not exit %.0fs after signal %d; killing the process group",
                        self._cfg.stop_grace, self._stop_signal,
                    )
                    self.kill_child()
                continue  # no probing while stopping
            if now >= self._next_probe_at:
                self._next_probe_at = now + self._cfg.interval
                reason = self._on_probe(self._probe(), now)
                if reason is not None:
                    return self._restart(reason)

    # -- state machine -------------------------------------------------------------------------

    def _on_probe(self, healthy: bool, now: float) -> str | None:
        """Advance the state machine by one probe result; return a restart reason or ``None``."""
        uptime = now - self._started_at
        cfg = self._cfg
        if self._state is State.STARTING:
            if healthy:
                self._state = State.RUNNING
                self._failures = 0
                log.info("healthy after %.0fs", uptime)
                return None
            log.warning(
                "waiting for the first healthy response (%.0fs of %.0fs): %s",
                uptime, cfg.startup_timeout, self._last_failure(),
            )
            return "startup_timeout" if uptime >= cfg.startup_timeout else None
        if healthy:
            self._failures = 0
            return None
        self._failures += 1
        log.warning("health probe failed (%d/%d): %s", self._failures, cfg.failures, self._last_failure())
        if self._failures < cfg.failures:
            return None
        return "health_failed"

    def _last_failure(self) -> str:
        return getattr(self._probe, "last_failure", "") or "probe failed"

    # -- reactions -----------------------------------------------------------------------------

    def _restart(self, reason: str) -> int:
        uptime = self._clock() - self._started_at
        detail = self._last_failure()
        log.error(
            "restarting the container: reason=%s uptime=%.0fs consecutive_failures=%d (%s)",
            reason, uptime, self._failures, detail,
        )
        self._notify(RestartEvent(reason=reason, uptime=uptime, failures=self._failures,
                                  exit_code=EXIT_RESTART, detail=detail))
        self.kill_child()
        returncode = self._child.wait(timeout=self._cfg.stop_grace)
        if returncode is None:
            log.error(
                "child pid %d is still alive %.0fs after SIGKILL; exiting anyway",
                self._child.pid, self._cfg.stop_grace,
            )
        else:
            log.info("child exited with status %d after SIGKILL", returncode)
        log.error("exiting with code %d so that Docker restarts the container", EXIT_RESTART)
        return EXIT_RESTART

    def _on_child_exit(self, returncode: int) -> int:
        code = normalize_exit_code(returncode)
        uptime = self._clock() - self._started_at
        if self._stop_signal is not None:
            log.info("child exited with code %d after signal %d", code, self._stop_signal)
            return code
        log.error("child exited on its own with code %d after %.0fs", code, uptime)
        self._notify(RestartEvent(reason="child_exited", uptime=uptime, failures=self._failures,
                                  exit_code=code, detail=f"child exit code {code}"))
        return code

    def _notify(self, event: RestartEvent) -> None:
        try:
            self._notifier.notify(event)
        except Exception as exc:  # noqa: BLE001 - a broken notifier must not stop the restart
            log.warning("notifier raised %s: %s", type(exc).__name__, exc)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/supervisor.py tests/test_supervisor.py
git commit -F - <<'EOF'
feat(supervisor): add Supervisor state machine and child wrapper

STARTING -> RUNNING on the first HTTP 200; three consecutive failures or
a startup that never becomes healthy SIGKILL the child's process group and
exit 3 so restart: unless-stopped recreates the container. SIGTERM/SIGINT
are forwarded to the child, with SIGKILL after the grace period.

Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68
EOF
```

---

### Task 5: Anti-flapping — PENDING_RESTART cooldown for short-lived containers

**Files:**
- Modify: `app/supervisor.py` (constant, replace `Supervisor.__init__` fields and `_on_probe`)
- Modify: `tests/test_supervisor.py` (append)

**Interfaces:**
- Consumes: `Supervisor`, `State.PENDING_RESTART`, `Config.min_uptime`, `Config.flap_cooldown`.
- Produces: constant `PENDING_LOG_INTERVAL = 60.0`; the behaviour "hang at uptime < `min_uptime` waits `flap_cooldown` while still probing; a success cancels".

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_supervisor.py`:

```python
# --------------------------------------------------------------------------- Supervisor: anti-flapping


def test_hang_at_short_uptime_waits_for_cooldown_before_restart(caplog):
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    notifier = RecordingNotifier()
    probe = ScriptedProbe([True, False])  # healthy once, then failing forever
    # cooldown 70 s (> 60 s) so the once-a-minute progress line fires exactly once
    sup = make_supervisor(make_config(min_uptime=100.0, flap_cooldown=70.0), child, probe, clock=clock, notifier=notifier)
    with caplog.at_level(logging.INFO, logger="supervisor"):
        code = sup.run()
    assert code == 3
    assert child.group_killed is True
    # 3 failures at t+4 -> cooldown until t+74 -> kill at the probe of t+74
    assert clock.now == pytest.approx(1074.0)
    assert [e.reason for e in notifier.events] == ["health_failed"]
    assert notifier.events[0].uptime == pytest.approx(74.0)
    assert "flapping suspected, restart delayed by 70s" in caplog.text
    assert "still unhealthy; restart in 10s" in caplog.text  # logged once, at t+64


def test_recovery_during_cooldown_cancels_restart(caplog):
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    notifier = RecordingNotifier()
    results = [True, False, False, False] + [True] * 20
    with caplog.at_level(logging.INFO, logger="supervisor"):
        code, probe = run_until_stopped(make_config(min_uptime=100.0, flap_cooldown=50.0), clock, child, results,
                                        stop_at_call=20, notifier=notifier)
    assert code == 0
    assert child.group_killed is False
    assert notifier.events == []
    assert "pending restart cancelled" in caplog.text


def test_after_cancelled_cooldown_counter_starts_from_zero():
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    notifier = RecordingNotifier()
    # hang -> pending -> recover -> two failures -> recover: never three in a row again
    results = [True, False, False, False, True, False, False, True]
    code, _ = run_until_stopped(make_config(min_uptime=100.0, flap_cooldown=50.0), clock, child, results,
                                stop_at_call=12, notifier=notifier)
    assert code == 0
    assert child.group_killed is False
    assert notifier.events == []


def test_zero_cooldown_disables_flapping_rule():
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    probe = ScriptedProbe([True, False, False, False])
    sup = make_supervisor(make_config(min_uptime=100.0, flap_cooldown=0.0), child, probe, clock=clock)
    assert sup.run() == 3
    assert clock.now == pytest.approx(1004.0)


def test_long_uptime_is_not_flapping():
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    probe = ScriptedProbe([True] * 200 + [False])
    sup = make_supervisor(make_config(min_uptime=100.0, flap_cooldown=50.0), child, probe, clock=clock)
    assert sup.run() == 3
    assert clock.now == pytest.approx(1203.0)  # failures at t+201..203, no cooldown
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -k "flapping or cooldown or uptime_is_not" -v`
Expected: `test_hang_at_short_uptime_waits_for_cooldown_before_restart` FAILS (`clock.now == 1004`, no cooldown), `test_recovery_during_cooldown_cancels_restart` FAILS (code 3 instead of 0), `test_after_cancelled_cooldown_counter_starts_from_zero` FAILS (code 3 instead of 0); `test_zero_cooldown_disables_flapping_rule` and `test_long_uptime_is_not_flapping` PASS already.

- [ ] **Step 3: Implement the cooldown**

In `app/supervisor.py` add the constant next to `SMTP_TIMEOUT`:

```python
PENDING_LOG_INTERVAL = 60.0  # seconds between "still unhealthy" lines while cooling down
```

In `Supervisor.__init__`, add two fields after `self._next_probe_at = 0.0`:

```python
        self._pending_deadline = 0.0
        self._last_pending_log = 0.0
```

Replace the whole `_on_probe` method with:

```python
    def _on_probe(self, healthy: bool, now: float) -> str | None:
        """Advance the state machine by one probe result; return a restart reason or ``None``."""
        uptime = now - self._started_at
        cfg = self._cfg
        if self._state is State.STARTING:
            if healthy:
                self._state = State.RUNNING
                self._failures = 0
                log.info("healthy after %.0fs", uptime)
                return None
            log.warning(
                "waiting for the first healthy response (%.0fs of %.0fs): %s",
                uptime, cfg.startup_timeout, self._last_failure(),
            )
            return "startup_timeout" if uptime >= cfg.startup_timeout else None
        if healthy:
            if self._state is State.PENDING_RESTART:
                log.info("healthy again; pending restart cancelled")
            self._state = State.RUNNING
            self._failures = 0
            return None
        self._failures += 1
        if self._state is State.RUNNING:
            log.warning("health probe failed (%d/%d): %s", self._failures, cfg.failures, self._last_failure())
            if self._failures < cfg.failures:
                return None
            if cfg.flap_cooldown > 0 and uptime < cfg.min_uptime:
                # A hang this soon after start smells like a dead GPU: restarting every two
                # minutes would only churn. Wait, keep probing, and restart only if it stays hung.
                self._state = State.PENDING_RESTART
                self._pending_deadline = now + cfg.flap_cooldown
                self._last_pending_log = now
                log.warning(
                    "hang detected only %.0fs after start (< %.0fs); flapping suspected, restart delayed by %.0fs",
                    uptime, cfg.min_uptime, cfg.flap_cooldown,
                )
                return None
            return "health_failed"
        # PENDING_RESTART and still failing
        if now >= self._pending_deadline:
            return "health_failed"
        if now - self._last_pending_log >= PENDING_LOG_INTERVAL:
            self._last_pending_log = now
            log.info("still unhealthy; restart in %.0fs", self._pending_deadline - now)
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/supervisor.py tests/test_supervisor.py
git commit -F - <<'EOF'
feat(supervisor): delay restart when the hang comes right after start

Stateless anti-flapping: a hang at uptime below WATCHDOG_MIN_UPTIME waits
WATCHDOG_FLAP_COOLDOWN while still probing; a healthy answer cancels the
restart. A dead GPU now costs one restart per ~17 minutes instead of ~2.

Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68
EOF
```

---

### Task 6: `main()` — argv, logging, kill switch, signal wiring, smoke tests

**Files:**
- Modify: `app/supervisor.py` (imports + `_configure_logging`, `main`, `__main__` guard)
- Modify: `tests/test_supervisor.py` (append)

**Interfaces:**
- Consumes: `Config.from_env`, `HealthProbe`, `SmtpNotifier`, `NullNotifier`, `Supervisor`, exit-code constants.
- Produces: `main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int`; `_configure_logging(level_name: str | None) -> None`.

- [ ] **Step 1: Write the failing tests**

Add to the imports at the top of `tests/test_supervisor.py`:

```python
import os
import sys
```

Append:

```python
# --------------------------------------------------------------------------- main()


def test_main_without_command_prints_usage(capsys):
    assert sv.main([], {}) == sv.EXIT_USAGE == 2
    assert "usage: supervisor.py <command>" in capsys.readouterr().err


def test_main_disabled_execs_child_without_supervision(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(sv.os, "execvp", lambda file, args: calls.append((file, args)))
    with caplog.at_level(logging.WARNING, logger="supervisor"):
        code = sv.main(["uvicorn", "main:app"], {"WATCHDOG_ENABLED": "false"})
    assert calls == [("uvicorn", ["uvicorn", "main:app"])]
    assert code == sv.EXIT_EXEC_FAILED == 127  # execvp only returns on failure
    assert "WATCHDOG_ENABLED=false" in caplog.text


def test_main_disabled_reports_execvp_failure(monkeypatch, caplog):
    def failing_execvp(file, args):
        raise FileNotFoundError(f"no such file: {file}")

    monkeypatch.setattr(sv.os, "execvp", failing_execvp)
    with caplog.at_level(logging.ERROR, logger="supervisor"):
        code = sv.main(["nonexistent-command"], {"WATCHDOG_ENABLED": "0"})
    assert code == 127
    assert "execvp nonexistent-command failed" in caplog.text


def test_configure_logging_honours_log_level(monkeypatch):
    captured = {}
    monkeypatch.setattr(sv.logging, "basicConfig", lambda **kwargs: captured.update(kwargs))
    sv._configure_logging("debug")
    assert captured["level"] == logging.DEBUG
    sv._configure_logging("NOT_A_LEVEL")
    assert captured["level"] == logging.INFO
    sv._configure_logging(None)
    assert captured["level"] == logging.INFO
    assert captured["format"] == "%(asctime)s supervisor: %(levelname)s %(message)s"


# --------------------------------------------------------------------------- smoke: a real child

_STUBBORN_CHILD = [sys.executable, "-c",
                   "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals and process groups")
def test_smoke_hung_child_is_killed_and_exit_code_is_3():
    cfg = make_config(interval=0.1, timeout=0.1, failures=2, startup_timeout=0.5,
                      min_uptime=0.0, flap_cooldown=0.0, stop_grace=2.0)
    sup = sv.Supervisor(cfg, _STUBBORN_CHILD, probe=lambda: False, notifier=sv.NullNotifier(), tick=0.05)
    started = time.monotonic()
    code = sup.run()
    assert code == 3
    assert time.monotonic() - started < 10
    with pytest.raises(ProcessLookupError):
        os.kill(sup.child_pid, 0)  # reaped: the pid is gone


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals and process groups")
def test_smoke_stop_request_escalates_to_sigkill():
    cfg = make_config(interval=0.1, timeout=0.1, stop_grace=0.5)
    sup = sv.Supervisor(cfg, _STUBBORN_CHILD, probe=lambda: True, notifier=sv.NullNotifier(), tick=0.05)
    threading.Timer(0.3, lambda: sup.request_stop(signal.SIGTERM)).start()
    started = time.monotonic()
    code = sup.run()
    assert code == 137  # SIGTERM ignored -> SIGKILL after 0.5s -> 128 + 9
    assert time.monotonic() - started < 10
    with pytest.raises(ProcessLookupError):
        os.kill(sup.child_pid, 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -v`
Expected: the four `main`/`logging` tests FAIL with `AttributeError: module 'supervisor' has no attribute 'main'` / `'_configure_logging'`; the two smoke tests PASS already (they only need `Supervisor` and `ChildProcess`); everything else still PASSES.

- [ ] **Step 3: Implement `main()`**

Extend the imports in `app/supervisor.py` with `sys`:

```python
import dataclasses
import enum
import http.client
import logging
import os
import signal
import smtplib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from email.message import EmailMessage
```

Append at the end of the module:

```python
def _configure_logging(level_name: str | None) -> None:
    """Stderr, prefixed ``supervisor:``; level from ``LOG_LEVEL`` when valid, else INFO."""
    level = logging.INFO
    if level_name:
        candidate = logging.getLevelName(level_name.strip().upper())
        if isinstance(candidate, int):
            level = candidate
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s supervisor: %(levelname)s %(message)s",
    )


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    if not args:
        print("usage: supervisor.py <command> [args...]", file=sys.stderr)
        return EXIT_USAGE
    _configure_logging(env.get("LOG_LEVEL"))
    cfg = Config.from_env(env)
    if not cfg.enabled:
        log.warning("WATCHDOG_ENABLED=false: running %s without supervision", args[0])
        try:
            os.execvp(args[0], args)
        except OSError as exc:
            log.error("execvp %s failed: %s", args[0], exc)
        return EXIT_EXEC_FAILED  # execvp only comes back on failure
    probe = HealthProbe(cfg.health_url, cfg.timeout)
    notifier = SmtpNotifier(cfg, child_cmd=args) if cfg.mail_enabled else NullNotifier()
    supervisor = Supervisor(cfg, args, probe=probe, notifier=notifier)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, _frame: supervisor.request_stop(received))
    log.info(
        "watchdog armed: url=%s interval=%gs timeout=%gs failures=%d startup_timeout=%gs "
        "min_uptime=%gs flap_cooldown=%gs stop_grace=%gs mail=%s",
        cfg.health_url, cfg.interval, cfg.timeout, cfg.failures, cfg.startup_timeout,
        cfg.min_uptime, cfg.flap_cooldown, cfg.stop_grace, cfg.mail_to or "off",
    )
    try:
        return supervisor.run()
    except Exception:  # noqa: BLE001 - last resort: never leave an orphaned child behind
        log.exception("supervisor crashed; killing the child")
        supervisor.kill_child()
        return EXIT_SUPERVISOR_BUG


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_supervisor.py -v`
Expected: all PASS (smoke tests take about 1–2 s each).

Then a manual end-to-end check of the real entry point (a stand-in child that serves `/health`):

```bash
(cd app && WATCHDOG_HEALTH_URL=http://127.0.0.1:9/health WATCHDOG_INTERVAL=1 WATCHDOG_TIMEOUT=1 WATCHDOG_STARTUP_TIMEOUT=3 \
  timeout 20 ../.venv/bin/python supervisor.py python3 -c 'import time; time.sleep(30)'; echo "exit=$?")
```
Expected: a `watchdog armed:` line, three `waiting for the first healthy response` lines (port 9 refuses connections), then `restarting the container: reason=startup_timeout`, and `exit=3`.

```bash
(cd app && ../.venv/bin/python supervisor.py; echo "exit=$?")
```
Expected: `usage: supervisor.py <command> [args...]` and `exit=2`.

- [ ] **Step 5: Commit**

```bash
git add app/supervisor.py tests/test_supervisor.py
git commit -F - <<'EOF'
feat(supervisor): add main() entry point, kill switch and signal wiring

WATCHDOG_ENABLED=false execvp's the child so the container behaves exactly
as before. Two smoke tests drive a real child that ignores SIGTERM.

Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68
EOF
```

---

### Task 7: Images and compose — run the supervisor, `init: true`, pass-throughs, invariants

**Files:**
- Modify: `docker/amd/Dockerfile:35`, `docker/cpu/Dockerfile:23`, `docker/nvidia/Dockerfile:39` (the `CMD` line)
- Modify: `docker/docker-compose-amd.yml`, `docker/docker-compose-cpu.yml`, `docker/docker-compose-nvidia.yml` (`init: true`)
- Modify: `docker/deploy/docker-compose-amd.yml`, `docker/deploy/docker-compose-cpu.yml`, `docker/deploy/docker-compose-nvidia.yml` (`init: true` + nine env pass-throughs)
- Modify: `tests/test_compose.py`

**Interfaces:**
- Consumes: the `CMD` string and the nine variable names from Global Constraints.
- Produces: nothing for later tasks; Task 8 documents what this task wires.

- [ ] **Step 1: Write the failing invariant tests**

In `tests/test_compose.py`, extend the module docstring by appending this paragraph before the closing `"""`:

```
Also guards the process-watchdog wiring (design doc
docs/superpowers/specs/2026-09-02-process-watchdog-design.md, likewise removed before the PR; it
lives in the feat/process-watchdog branch history): `init: true` in every compose file, the
`WATCHDOG_*` pass-throughs in the deployment files, and the supervisor `CMD` in every Dockerfile.
```

After the `AMD_FILES = ...` line add:

```python
DEPLOY_FILES = [p for p in COMPOSE_FILES if p.parent.name == "deploy"]
DOCKERFILES = sorted((REPO_ROOT / "docker").glob("*/Dockerfile"))

# Variables docker/deploy/*.yml must forward from .env into the container (see .env.example).
WATCHDOG_PASSTHROUGH = (
    "WATCHDOG_ENABLED",
    "WATCHDOG_STARTUP_TIMEOUT",
    "WATCHDOG_FLAP_COOLDOWN",
    "WATCHDOG_MAIL_TO",
    "WATCHDOG_MAIL_FROM",
    "WATCHDOG_SMTP_HOST",
    "WATCHDOG_SMTP_PORT",
    "WATCHDOG_SMTP_USER",
    "WATCHDOG_SMTP_PASSWORD",
)
SUPERVISOR_CMD = 'CMD ["python3", "supervisor.py", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]'
```

Append at the end of the file:

```python
def test_expected_deploy_and_dockerfile_counts():
    assert len(DEPLOY_FILES) == 3, DEPLOY_FILES
    assert len(DOCKERFILES) == 3, DOCKERFILES


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_init_true_so_tini_is_pid1(path):
    # supervisor.py kills uvicorn as a child; tini as PID 1 keeps signal and zombie semantics standard
    svc = _service(yaml.safe_load(path.read_text()))
    assert svc.get("init") is True, path


@pytest.mark.parametrize("path", DEPLOY_FILES, ids=lambda p: p.name)
def test_deploy_compose_forwards_watchdog_variables(path):
    env = _service(yaml.safe_load(path.read_text()))["environment"]
    assert isinstance(env, dict), f"{path}: environment must be a mapping"
    for key in WATCHDOG_PASSTHROUGH:
        assert env.get(key) == "${%s:-}" % key, (path, key)


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.parent.name)
def test_dockerfile_runs_supervisor(path):
    cmd_lines = [line.strip() for line in path.read_text().splitlines() if line.strip().startswith("CMD")]
    assert cmd_lines == [SUPERVISOR_CMD], path
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_compose.py -v`
Expected: the six `init` tests, three pass-through tests and three Dockerfile tests FAIL; the existing ulimit/MIOpen tests still PASS.

- [ ] **Step 3: Change the three Dockerfiles**

In each of `docker/amd/Dockerfile`, `docker/cpu/Dockerfile`, `docker/nvidia/Dockerfile` replace the line

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

with

```dockerfile
# supervisor.py runs uvicorn as a child and restarts the container when /health hangs
# (see app/supervisor.py). WATCHDOG_ENABLED=false makes it exec uvicorn directly.
CMD ["python3", "supervisor.py", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker/nvidia/Dockerfile` has no trailing newline after `CMD`; add one.

- [ ] **Step 4: Add `init: true` to the six compose files**

Insert directly after the `restart: unless-stopped` line of the service in every file (`docker/docker-compose-amd.yml`, `docker/docker-compose-cpu.yml`, `docker/docker-compose-nvidia.yml`, `docker/deploy/docker-compose-amd.yml`, `docker/deploy/docker-compose-cpu.yml`, `docker/deploy/docker-compose-nvidia.yml`), with the same four-space indentation as `restart:`:

```yaml
    restart: unless-stopped
    init: true  # tini as PID 1; supervisor.py and uvicorn are ordinary children it can signal
```

- [ ] **Step 5: Add the pass-throughs to the three deploy compose files**

In `docker/deploy/docker-compose-amd.yml`, append to the `environment:` mapping after the `# MIOPEN_FIND_MODE: FAST` comment line:

```yaml
      # Process watchdog (supervisor.py): restarts the container when /health hangs.
      # Empty value = built-in default; see .env.example.
      WATCHDOG_ENABLED: ${WATCHDOG_ENABLED:-}
      WATCHDOG_STARTUP_TIMEOUT: ${WATCHDOG_STARTUP_TIMEOUT:-}
      WATCHDOG_FLAP_COOLDOWN: ${WATCHDOG_FLAP_COOLDOWN:-}
      WATCHDOG_MAIL_TO: ${WATCHDOG_MAIL_TO:-}
      WATCHDOG_MAIL_FROM: ${WATCHDOG_MAIL_FROM:-}
      WATCHDOG_SMTP_HOST: ${WATCHDOG_SMTP_HOST:-}
      WATCHDOG_SMTP_PORT: ${WATCHDOG_SMTP_PORT:-}
      WATCHDOG_SMTP_USER: ${WATCHDOG_SMTP_USER:-}
      WATCHDOG_SMTP_PASSWORD: ${WATCHDOG_SMTP_PASSWORD:-}
```

In `docker/deploy/docker-compose-cpu.yml` append the same eleven lines after `VIDEO_HW_ACCEL: cpu`. In `docker/deploy/docker-compose-nvidia.yml` append them after `NVIDIA_DRIVER_CAPABILITIES: compute,utility,video`.

- [ ] **Step 6: Run the tests to verify they pass, and validate the compose files**

Run: `.venv/bin/python -m pytest tests/test_compose.py -v`
Expected: all PASS.

Run (compose syntax check, no containers started; skip if `docker` is unavailable on this machine):

```bash
for f in docker/deploy/docker-compose-amd.yml docker/deploy/docker-compose-cpu.yml docker/deploy/docker-compose-nvidia.yml \
         docker/docker-compose-amd.yml docker/docker-compose-cpu.yml docker/docker-compose-nvidia.yml; do
  docker compose -f "$f" config --quiet && echo "OK $f"
done
```
Expected: `OK` for each file.

- [ ] **Step 7: Commit**

```bash
git add docker/ tests/test_compose.py
git commit -F - <<'EOF'
chore(docker): run uvicorn under supervisor.py, init: true, WATCHDOG_* pass-through

Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68
EOF
```

---

### Task 8: Operator documentation

**Files:**
- Modify: `CLAUDE.md` (Architecture table, Configuration block, Testing sentence, Key Patterns)
- Modify: `README.md` (Configuration table, Key Design Decisions, Project Structure)
- Modify: `.claude/rules/docker.md` (Dockerfile Overview step 7, Health Check, Troubleshooting)
- Modify: `docker/deploy/.env.example`

**Interfaces:**
- Consumes: variable names, defaults and exit codes from Global Constraints. No code.

- [ ] **Step 1: CLAUDE.md**

In the Architecture table, after the `app/detection_stabilizer.py` row add:

```markdown
| `app/supervisor.py` | Process watchdog: runs uvicorn as a child, restarts the container when `/health` hangs |
```

In the Configuration block, after the `STABILIZER_MAX_STALENESS=5.0` line and before the `# MIOPEN_FIND_MODE=FAST` comment, add:

```
WATCHDOG_ENABLED=true           # false = run uvicorn without the watchdog (emergency lever)
WATCHDOG_HEALTH_URL=http://127.0.0.1:8000/health
WATCHDOG_INTERVAL=30            # seconds between /health probes
WATCHDOG_TIMEOUT=10             # probe HTTP timeout
WATCHDOG_FAILURES=3             # consecutive failures -> SIGKILL uvicorn, exit 3, Docker restarts
WATCHDOG_STARTUP_TIMEOUT=600    # wait for the first healthy answer (cold MIOpen compile)
WATCHDOG_MIN_UPTIME=600         # a hang before this uptime counts as flapping
WATCHDOG_FLAP_COOLDOWN=900      # delay before restarting a flapping container; 0 = off
WATCHDOG_STOP_GRACE=8           # seconds after SIGTERM before SIGKILL
WATCHDOG_MAIL_TO=               # non-empty enables e-mail on every watchdog restart
WATCHDOG_MAIL_FROM=vision-api@<hostname>
WATCHDOG_SMTP_HOST=             # required for mail; port 587 + STARTTLS by default
WATCHDOG_SMTP_PORT=587
WATCHDOG_SMTP_USER=             # login only when set
WATCHDOG_SMTP_PASSWORD=
WATCHDOG_SMTP_STARTTLS=true
```

In the Testing section, change the sentence `Tests cover config, Pydantic models, JobManager, and VideoAnnotator (mocked YOLO/FFmpeg).` to:

```markdown
Tests cover config, Pydantic models, JobManager, VideoAnnotator (mocked YOLO/FFmpeg), the process watchdog (`tests/test_supervisor.py`, fake child + fake clock, two real-subprocess smoke tests) and deployment invariants of the compose files and Dockerfiles (`tests/test_compose.py`).
```

In Key Patterns, after the `**NVENC CPU fallback**` paragraph add:

```markdown
**Process Watchdog**: `app/supervisor.py` is the container command; uvicorn is its child in its own process group, tini is PID 1 (`init: true`). The supervisor polls `/health` (30 s interval, 10 s timeout); 3 consecutive failures — or no healthy answer within 600 s of start — SIGKILL the process group and exit with code 3, and `restart: unless-stopped` recreates the container. Why a process and not a thread: the 2026-08-30 incident was a ROCm busy-spin holding the GIL, so nothing inside the uvicorn process could run for 2.5 days. Why uvicorn must not be PID 1: the kernel ignores SIGKILL sent to PID 1 from inside its own PID namespace. Anti-flapping is stateless: a hang less than 600 s after start waits 900 s (still probing, a healthy answer cancels) before killing. Standard library only, no application imports. `WATCHDOG_ENABLED=false` exec's uvicorn directly.
```

- [ ] **Step 2: README.md**

In the Configuration table, after the `VAAPI_DEVICE` row add:

```markdown
| `WATCHDOG_ENABLED` | `true` | Process watchdog; `false` runs uvicorn without it |
| `WATCHDOG_INTERVAL` / `WATCHDOG_TIMEOUT` | `30` / `10` | `/health` probe period and HTTP timeout (seconds) |
| `WATCHDOG_FAILURES` | `3` | Consecutive failures before the container is restarted |
| `WATCHDOG_STARTUP_TIMEOUT` | `600` | Seconds to wait for the first healthy answer |
| `WATCHDOG_MIN_UPTIME` / `WATCHDOG_FLAP_COOLDOWN` | `600` / `900` | A hang earlier than `MIN_UPTIME` after start waits `FLAP_COOLDOWN` before restarting; `0` disables |
| `WATCHDOG_STOP_GRACE` | `8` | Seconds after SIGTERM before SIGKILL |
| `WATCHDOG_MAIL_TO` | (empty) | E-mail every watchdog restart; needs `WATCHDOG_SMTP_HOST` (`_PORT` 587, `_USER`, `_PASSWORD`, `_STARTTLS` true, `WATCHDOG_MAIL_FROM`) |
```

In Key Design Decisions add a bullet:

```markdown
- **Process watchdog** — `supervisor.py` runs uvicorn as a child and polls `/health` from outside the Python process; a GPU hang that freezes the interpreter (GIL held) ends in a SIGKILL and a container restart instead of an indefinite outage
```

In Project Structure, change `└── dependencies.py      # FastAPI dependency injection` to `├── dependencies.py      # FastAPI dependency injection` and add below it:

```
└── supervisor.py        # Process watchdog: container command, restarts on hung /health
```

- [ ] **Step 3: .claude/rules/docker.md**

In Dockerfile Overview replace `7. Run uvicorn` with:

```markdown
7. Run `supervisor.py`, which runs uvicorn as a child and restarts the container when `/health` hangs (see Health Check)
```

In the Health Check section, after the YAML block add:

```markdown
The healthcheck only *reports* status; plain Docker never acts on `unhealthy`. Inside the container `supervisor.py` polls the same endpoint with the same thresholds and, after 3 consecutive failures (or no healthy answer within 600 s of start), SIGKILLs uvicorn's process group and exits with code 3, so `restart: unless-stopped` recreates the container. All compose files set `init: true` (tini as PID 1). Tune with `WATCHDOG_*` (see `deploy/.env.example`); `WATCHDOG_ENABLED=false` disables the watchdog. On the host, `docker inspect -f '{{.RestartCount}}' <container>` counts watchdog restarts.
```

In Troubleshooting append:

```markdown
**Container restarts every few minutes:**
- The watchdog is firing: `docker logs <container> 2>&1 | grep supervisor:` shows `restarting the container: reason=...`
- `reason=startup_timeout` — model preload took longer than `WATCHDOG_STARTUP_TIMEOUT` (600 s); raise it or warm the MIOpen cache volume
- `reason=health_failed` shortly after start — the GPU is probably hung: check `rocm-smi` / `nvidia-smi` and `dmesg`; reboot the host if the GPU never comes back
- Emergency: `WATCHDOG_ENABLED=false` in `.env` and `docker compose up -d`
```

- [ ] **Step 4: docker/deploy/.env.example**

Append at the end of the file:

```
# Process watchdog (supervisor.py). Restarts the container when /health stops answering.
# Log lines are always written; e-mail is optional. Empty value = built-in default.
# WATCHDOG_ENABLED=true              # false = run uvicorn without supervision (emergency lever)
# WATCHDOG_STARTUP_TIMEOUT=600       # seconds to wait for the first healthy /health (cold MIOpen compile)
# WATCHDOG_FLAP_COOLDOWN=900         # delay before restarting a container that hung <600 s after start; 0 = off

# Optional e-mail on every watchdog restart. Requires WATCHDOG_SMTP_HOST; port 587 with STARTTLS
# by default. The password ends up in the container environment (visible in `docker inspect`).
# WATCHDOG_MAIL_TO=ops@example.com
# WATCHDOG_MAIL_FROM=vision-api@example.com
# WATCHDOG_SMTP_HOST=smtp.example.com
# WATCHDOG_SMTP_PORT=587
# WATCHDOG_SMTP_USER=
# WATCHDOG_SMTP_PASSWORD=
```

- [ ] **Step 5: Check the docs render and nothing references the old command**

Run: `grep -rn 'CMD \["uvicorn"' docker/ README.md CLAUDE.md .claude/rules/ ; grep -n "Run uvicorn$" .claude/rules/docker.md`
Expected: no output.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PASS (docs do not affect tests; this confirms the tree is green before committing).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md .claude/rules/docker.md docker/deploy/.env.example
git commit -F - <<'EOF'
docs: describe the process watchdog and its WATCHDOG_* settings

Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68
EOF
```

---

### Task 9: Final verification and branch clean-up before the PR

**Files:**
- Delete (tracked): `docs/superpowers/specs/2026-09-02-process-watchdog-design.md`, `docs/superpowers/plans/2026-09-02-process-watchdog.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all PASS; the output includes `tests/test_supervisor.py` and the new `tests/test_compose.py` cases.

- [ ] **Step 2: Run the supervisor module the way the image will**

Run: `(cd app && WATCHDOG_ENABLED=false ../.venv/bin/python supervisor.py python3 -c 'print("child ran")'; echo "exit=$?")`
Expected: a `WATCHDOG_ENABLED=false: running python3 without supervision` line, `child ran`, `exit=0` (execvp replaced the supervisor with the child).

- [ ] **Step 3: Review the diff against master**

Run: `git diff master --stat`
Expected: exactly these paths — `app/supervisor.py`, `tests/test_supervisor.py`, `tests/test_compose.py`, three Dockerfiles, six compose files, `CLAUDE.md`, `README.md`, `.claude/rules/docker.md`, `docker/deploy/.env.example`, and the two `docs/superpowers/` files (removed in the next step). `app/config.py` must not appear.

- [ ] **Step 4: Drop the working documents from the branch (user rule: they must not appear in the PR diff)**

```bash
git rm docs/superpowers/specs/2026-09-02-process-watchdog-design.md docs/superpowers/plans/2026-09-02-process-watchdog.md
git commit -F - <<'EOF'
docs: drop superpowers working documents before PR

The design spec and plan stay reachable in this branch's history.

Claude-Session: https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68
EOF
```

`docs/superpowers/plans/` also holds two untracked continuation prompts from earlier work and `docs/` holds an untracked session-transfer note; leave them alone.

- [ ] **Step 5: Hand over**

Use the `superpowers:finishing-a-development-branch` skill to open the PR from `feat/process-watchdog` to `master`. The PR description ends with `https://claude.ai/code/session_01QtFsmaMi3a3728dDCNvh68`. Rollout after merge (from the spec): tag `v0.7.0`, wait for CI (AMD build up to two hours), on the server pull `docker/deploy/` and run `deploy-up-detach-amd.sh`, then the fire drill:

```bash
docker top deploy-vision-api-1 -o pid,cmd          # host PID of "uvicorn main:app"
sudo kill -STOP <uvicorn host pid>
# about two minutes later:
docker logs --since 5m deploy-vision-api-1 2>&1 | grep supervisor:
docker inspect -f '{{.RestartCount}} {{.State.Health.Status}}' deploy-vision-api-1
curl -sf http://localhost:3001/health
```

Expected: a `restarting the container: reason=health_failed` line, `RestartCount` 1, status `healthy`, `/health` answering.
