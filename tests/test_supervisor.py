"""Tests for app/supervisor.py — the process watchdog.

The state machine is exercised with a scripted probe, a fake child, and a fake clock, so no
test sleeps or touches the network, except the HealthProbe tests (local http.server) and the
two smoke tests at the end (a real child process).
"""
import logging
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

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
    assert sv.NullNotifier().notify(_event()) is None  # must not raise, returns nothing


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


def test_stop_request_during_final_failing_probe_does_not_restart():
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    notifier = RecordingNotifier()
    holder = {}
    # The third failure would trigger a restart; SIGTERM lands while that probe is in flight.
    probe = ScriptedProbe([True, False, False, False],
                          hooks={4: lambda: holder["sup"].request_stop(signal.SIGTERM)})
    sup = make_supervisor(make_config(min_uptime=0.0), child, probe, clock=clock, notifier=notifier)
    holder["sup"] = sup
    assert sup.run() == 0
    assert child.group_killed is False
    assert child.signals == [signal.SIGTERM]
    assert notifier.events == []


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
