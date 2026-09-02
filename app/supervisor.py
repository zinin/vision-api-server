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
import enum
import http.client
import logging
import os
import signal
import smtplib
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from email.message import EmailMessage

log = logging.getLogger("supervisor")

EXIT_RESTART = 3
EXIT_USAGE = 2
EXIT_EXEC_FAILED = 127
EXIT_SUPERVISOR_BUG = 1
SMTP_TIMEOUT = 10.0
PENDING_LOG_INTERVAL = 60.0  # seconds between "still unhealthy" lines while cooling down

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


class HealthProbe:
    """One HTTP GET of the health URL. ``True`` only for HTTP 200.

    ``last_failure`` holds a short reason for the most recent ``False`` (empty
    after a success) so the supervisor can log it.
    """

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self.last_failure = ""
        # An opener with proxies switched off: urlopen's default one honours http_proxy /
        # HTTP_PROXY and has no loopback exemption, so on a host where Docker injects proxy
        # variables every probe of 127.0.0.1 would be sent to the proxy, fail, and make the
        # watchdog restart a perfectly healthy container.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self) -> bool:
        try:
            with self._opener.open(self.url, timeout=self.timeout) as response:
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
                    # An explicit default context verifies the certificate and the hostname;
                    # smtplib's own default does neither, which exposes the password below.
                    smtp.starttls(context=ssl.create_default_context())
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
        self._pending_deadline = 0.0
        self._last_pending_log = 0.0
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
                # A stop that arrived while the probe was in flight wins over the probe result.
                if reason is not None and self._stop_signal is None:
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
