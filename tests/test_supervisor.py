"""Tests for app/supervisor.py — the process watchdog.

The state machine is exercised with a scripted probe, a fake child, and a fake clock, so no
test sleeps or touches the network, except the HealthProbe tests (local http.server) and the
two smoke tests at the end (a real child process).
"""
import logging
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
