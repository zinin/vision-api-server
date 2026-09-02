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
import http.client
import logging
import os
import socket
import urllib.error
import urllib.request
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
