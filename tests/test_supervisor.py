"""Tests for app/supervisor.py — the process watchdog.

The state machine is exercised with a scripted probe, a fake child, and a fake clock, so no
test sleeps or touches the network, except the HealthProbe tests (local http.server) and the
two smoke tests at the end (a real child process).
"""
import logging
import os
import signal
import socket
import ssl
import sys
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
        ("WATCHDOG_INTERVAL", "nan"),
        ("WATCHDOG_INTERVAL", "inf"),
        ("WATCHDOG_TIMEOUT", "1e400"),
        ("WATCHDOG_FLAP_COOLDOWN", "-inf"),
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


def test_probe_ignores_proxy_environment_variables(http_base_url, monkeypatch):
    """A proxied loopback probe never reaches uvicorn: the watchdog would restart forever."""
    # urlopen caches its module-level opener; drop it so the pre-fix code really reads the env.
    monkeypatch.setattr(sv.urllib.request, "_opener", None)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    probe = sv.HealthProbe(f"{http_base_url}/ok", timeout=2.0)
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


class OrderRecordingNotifier:
    """Records, for every notification, whether the process group had already been killed."""

    def __init__(self, child):
        self.child = child
        self.killed_when_notified: list[bool] = []

    def notify(self, event):
        self.killed_when_notified.append(self.child.group_killed)


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
    smtp.starttls.assert_called_once()
    # Without an explicit context smtplib uses an unverified one and login() leaks the password.
    context = smtp.starttls.call_args.kwargs["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
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
        self.wait_timeouts: list[float] = []
        self._returncode: int | None = None

    def _exited(self) -> bool:
        return self.exit_at is not None and self.exit_at <= self.clock.now

    def wait(self, timeout: float) -> int | None:
        self.wait_timeouts.append(timeout)
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


def test_child_is_killed_before_the_notification_is_sent():
    """A stalled relay must not delay the SIGKILL by several SMTP timeouts."""
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    notifier = OrderRecordingNotifier(child)
    probe = ScriptedProbe([True, False, False, False])
    sup = make_supervisor(make_config(min_uptime=0.0), child, probe, clock=clock, notifier=notifier)
    assert sup.run() == 3
    assert notifier.killed_when_notified == [True]


def test_unkillable_child_still_exits_with_restart_code(caplog):
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock, unkillable=True)
    probe = ScriptedProbe([True, False, False, False])
    sup = make_supervisor(make_config(min_uptime=0.0, stop_grace=2.0), child, probe, clock=clock)
    with caplog.at_level(logging.ERROR, logger="supervisor"):
        assert sup.run() == 3
    assert child.group_killed is True
    assert "still alive 2s after SIGKILL" in caplog.text


def test_zero_stop_grace_still_confirms_the_kill_before_complaining(caplog):
    """wait(timeout=0) is a single WNOHANG poll: the child is not reaped yet and was not "alive"."""
    clock = FakeClock(start=1000.0)
    child = FakeChild(clock)
    probe = ScriptedProbe([True, False, False, False])
    sup = make_supervisor(make_config(min_uptime=0.0, stop_grace=0.0), child, probe, clock=clock)
    with caplog.at_level(logging.ERROR, logger="supervisor"):
        assert sup.run() == 3
    assert sv.KILL_CONFIRM_MIN_WAIT == 1.0
    assert child.wait_timeouts[-1] == pytest.approx(1.0)
    assert "still alive" not in caplog.text


def test_child_that_cannot_be_started_exits_127_without_mail(caplog):
    """A missing binary is a configuration error, not a hang: no restart code, no mail."""
    def failing_factory(cmd):
        raise FileNotFoundError(f"no such file: {cmd[0]}")

    notifier = RecordingNotifier()
    sup = sv.Supervisor(
        make_config(), ["child", "--flag"],
        probe=ScriptedProbe([True]), notifier=notifier, clock=FakeClock(),
        child_factory=failing_factory, tick=1.0,
    )
    with caplog.at_level(logging.ERROR, logger="supervisor"):
        assert sup.run() == sv.EXIT_EXEC_FAILED == 127
    assert notifier.events == []
    assert "cannot start child command child --flag" in caplog.text
    assert "no such file: child" in caplog.text


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


_FAKE_SUPERVISORS: list["FakeSupervisor"] = []


class FakeSupervisor:
    """Stands in for the real Supervisor so main()'s wiring can be inspected without a child."""

    def __init__(self, config, child_cmd, *, probe, notifier, **kwargs):
        self.config = config
        self.child_cmd = child_cmd
        self.probe = probe
        self.notifier = notifier
        self.kwargs = kwargs
        self.stopped_with: list[int] = []
        self.killed = False
        _FAKE_SUPERVISORS.append(self)

    def run(self) -> int:
        return 42

    def request_stop(self, signum: int) -> None:
        self.stopped_with.append(signum)

    def kill_child(self) -> None:
        self.killed = True


class CrashingSupervisor(FakeSupervisor):
    def run(self) -> int:
        raise RuntimeError("state machine exploded")


def test_main_wires_probe_notifier_and_signal_handlers(monkeypatch):
    _FAKE_SUPERVISORS.clear()
    handlers: list[tuple[int, object]] = []
    monkeypatch.setattr(sv, "Supervisor", FakeSupervisor)
    # Replaces the real signal.signal: the test process must keep its own handlers.
    monkeypatch.setattr(sv.signal, "signal", lambda signum, handler: handlers.append((signum, handler)))

    code = sv.main(["uvicorn", "main:app"],
                   {"WATCHDOG_MAIL_TO": "ops@example.com", "WATCHDOG_SMTP_HOST": "smtp.example.com"})

    assert code == 42
    [supervisor] = _FAKE_SUPERVISORS
    assert supervisor.child_cmd == ["uvicorn", "main:app"]
    assert isinstance(supervisor.notifier, sv.SmtpNotifier)
    assert isinstance(supervisor.probe, sv.HealthProbe)
    assert supervisor.probe.url == "http://127.0.0.1:8000/health"
    assert [signum for signum, _ in handlers] == [signal.SIGTERM, signal.SIGINT]
    for signum, handler in handlers:
        handler(signum, None)
    assert supervisor.stopped_with == [signal.SIGTERM, signal.SIGINT]
    assert supervisor.killed is False


def test_main_without_mail_settings_uses_the_null_notifier(monkeypatch):
    _FAKE_SUPERVISORS.clear()
    monkeypatch.setattr(sv, "Supervisor", FakeSupervisor)
    monkeypatch.setattr(sv.signal, "signal", lambda signum, handler: None)

    assert sv.main(["uvicorn", "main:app"], {}) == 42
    assert isinstance(_FAKE_SUPERVISORS[0].notifier, sv.NullNotifier)


def test_main_kills_the_child_when_the_supervisor_crashes(monkeypatch, caplog):
    _FAKE_SUPERVISORS.clear()
    monkeypatch.setattr(sv, "Supervisor", CrashingSupervisor)
    monkeypatch.setattr(sv.signal, "signal", lambda signum, handler: None)

    with caplog.at_level(logging.ERROR, logger="supervisor"):
        assert sv.main(["uvicorn", "main:app"], {}) == sv.EXIT_SUPERVISOR_BUG == 1
    assert _FAKE_SUPERVISORS[0].killed is True
    assert "supervisor crashed" in caplog.text


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

# Ignores SIGTERM, spawns a plain-Popen grandchild (an ffmpeg stand-in), reports its pid, sleeps.
_CHILD_WITH_GRANDCHILD = (
    "import signal, subprocess, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "grandchild = subprocess.Popen([sys.executable, '-c',\n"
    "    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
    "open(sys.argv[1], 'w').write(str(grandchild.pid))\n"
    "time.sleep(60)\n"
)


def _wait_until_gone(pid: int, timeout: float) -> bool:
    """Poll until ``pid`` disappears; an orphan stays a zombie until init reaps it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


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
def test_smoke_zero_stop_grace_does_not_claim_the_child_survived(caplog):
    cfg = make_config(interval=0.1, timeout=0.1, failures=2, startup_timeout=0.5,
                      min_uptime=0.0, flap_cooldown=0.0, stop_grace=0.0)
    sup = sv.Supervisor(cfg, _STUBBORN_CHILD, probe=lambda: False, notifier=sv.NullNotifier(), tick=0.05)
    with caplog.at_level(logging.ERROR, logger="supervisor"):
        code = sup.run()
    assert code == 3
    assert "still alive" not in caplog.text
    with pytest.raises(ProcessLookupError):
        os.kill(sup.child_pid, 0)


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


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals and process groups")
def test_smoke_killpg_reaches_a_grandchild(tmp_path):
    """Every ffmpeg is a grandchild of the supervisor; the group kill must reach it too."""
    pid_file = tmp_path / "grandchild.pid"
    cfg = make_config(interval=0.1, timeout=0.1, failures=2, startup_timeout=1.5,
                      min_uptime=0.0, flap_cooldown=0.0, stop_grace=1.0)
    cmd = [sys.executable, "-c", _CHILD_WITH_GRANDCHILD, str(pid_file)]
    sup = sv.Supervisor(cfg, cmd, probe=lambda: False, notifier=sv.NullNotifier(), tick=0.05)

    code = sup.run()

    assert code == 3
    grandchild_pid = int(pid_file.read_text())
    try:
        assert _wait_until_gone(grandchild_pid, timeout=3.0), "the grandchild survived killpg"
    finally:
        try:
            os.kill(grandchild_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    with pytest.raises(ProcessLookupError):
        os.kill(sup.child_pid, 0)
