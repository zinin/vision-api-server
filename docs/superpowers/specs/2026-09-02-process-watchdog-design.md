# Design: Process watchdog — auto-restart the container when the service hangs

- **Date:** 2026-09-02
- **Status:** Approved (ready for implementation plan)
- **Topic:** A supervisor process inside the image restarts the container when `/health` stops
  answering, so a hung GPU costs minutes of downtime instead of days

## Problem / Root cause

On 2026-08-30 07:20 UTC the production container `deploy-vision-api-1` (AMD/ROCm image on an
integrated Radeon 680M, `HSA_OVERRIDE_GFX_VERSION=10.3.0`) hung. It stayed hung for 2 days 15 hours,
burning one CPU core at 100%, until the operator noticed the process in `top` by chance.

Diagnosis (py-spy, perf, /proc, rocm-smi — collected on the server, consolidated in the session
transfer notes) established the mechanism with high confidence:

1. The GPU command queue stalled. The HSA runtime (`libhsa-runtime64`) went into a user-space
   busy-spin waiting for a completion signal that never came (97% of samples in the runtime,
   `syscall = running`, voluntary context switches frozen, `GPU% = 100` with `VRAM% = 0`).
2. The spinning thread was `inference__1`, a `ThreadPoolExecutor` worker inside
   `ultralytics ... make_anchors` — Python code — so **it held the GIL**.
3. With the GIL held, nothing else in the process could run: the event loop, `/health`, log
   writes, the `timeout=300` check in `video_utils.py:274` (the ffmpeg child became a zombie).

The Docker healthcheck did its job: it failed 5630 times in a row. Nothing reacted to it.
`restart: unless-stopped` only fires when the process exits; a frozen process never exits, and
plain Docker (unlike Swarm/Kubernetes) takes no action on `unhealthy`. `RestartCount` was 0.

Two consequences shape the design:

- **The watchdog must live outside the Python process.** A thread, `asyncio.wait_for`, or a
  Python signal handler cannot run while the GIL is held.
- **The hang looks alive.** The PID exists, the port accepts TCP connections, nothing answers. Only
  a request with a timeout detects it.

## Scope

**In scope**

- Detect the hang from a separate process and restart the container automatically.
- Log every watchdog action where the operator can see it from the host.
- Optional e-mail notification.
- Throttle restarts when the GPU is dead for good (anti-flapping).
- Apply identically to the CPU, NVIDIA, and AMD images and to all six compose files.

**Out of scope** (decided during brainstorming; separate work if ever)

- The GPU hang itself. ROCm does not support gfx1035; the override is a workaround. Nothing in
  this repository can fix that.
- Isolating inference in a killable worker process and wiring up the phantom `INFERENCE_TIMEOUT`
  setting (`app/config.py:23`, documented in `README.md` and `CLAUDE.md`, used nowhere). This
  would contain a hang to one request instead of one container restart. It is a large refactor of
  `model_manager`, `inference_utils`, `video_annotator` (which calls `model.predict` directly at
  line 357) and the job progress callbacks. The watchdog stays useful after that refactor as the
  last line of defence.
- Reducing `max_workers` or enabling `MIOPEN_FIND_MODE=FAST`. No evidence links either to the hang.
- Anything on the host or affecting other containers (GitLab runs on the same host). The feature
  lives entirely in the vision-api image and its compose files.

## Decision

**A supervisor process inside the image.** `app/supervisor.py` starts uvicorn as a child, polls
`/health` with a timeout, and on repeated failure SIGKILLs the child and exits non-zero. The
existing `restart: unless-stopped` policy then recreates the container.

Alternatives rejected:

| Option | Why not |
|---|---|
| systemd timer + script on the host | Violates the "everything inside the vision-api compose deployment" constraint. |
| `willfarrell/autoheal` sidecar | Needs `/var/run/docker.sock`, which is root on a host with a public IP and GitLab. No loop protection, webhook-only notification. |
| Own watchdog sidecar with `docker.sock` | Same socket risk plus a second image to build and publish. |
| Watchdog thread / `asyncio.wait_for` inside the app | Cannot run while the GIL is held (see Root cause). |
| "Watchdog does `kill -9 1`" | The kernel ignores SIGKILL sent to PID 1 from inside its own PID namespace (`pid_namespaces(7)`); uvicorn is PID 1 today. The supervisor must own the child instead. |

## Design

### Process tree

```
PID 1  docker-init (tini)                                   # compose: init: true
  └─ python3 supervisor.py uvicorn main:app --host 0.0.0.0 --port 8000
       └─ uvicorn main:app ...                              # the service, unchanged
```

- The child command is the supervisor's argv. The Dockerfile `CMD` stays readable, and a compose
  `command:` override keeps working.
- The child runs in its own session/process group (`start_new_session=True`) so one
  `killpg(SIGKILL)` takes uvicorn and any ffmpeg children with it.
- The child inherits stdout/stderr; `docker logs` output is unchanged. The supervisor writes its
  own lines to stderr, prefixed `supervisor:`.
- `init: true` makes tini PID 1 in every compose file: standard zombie reaping and signal
  forwarding. The supervisor also works as PID 1 (plain `docker run` without `--init`), because it
  installs handlers for SIGTERM/SIGINT and its child is never PID 1.

### Supervisor internals

`app/supervisor.py` is one file (~250 lines), **standard library only** (`subprocess`, `urllib`,
`signal`, `smtplib`, `email`, `logging`, `os`, `sys`, `time`, `socket`). It imports nothing from
the application and never imports torch, so an application-level problem cannot take the watchdog
down with it. Four units with explicit boundaries:

| Unit | Responsibility | Depends on |
|---|---|---|
| `Config.from_env(environ)` | Parse `WATCHDOG_*`, apply defaults, log `ERROR` on bad values | nothing |
| `HealthProbe(url, timeout)` | One HTTP GET; returns `True` only for HTTP 200 | `urllib` |
| `Notifier` (`NullNotifier`, `SmtpNotifier`) | Deliver a restart event; never raises | `smtplib` |
| `Supervisor` | State machine + child process wrapper; `run() -> exit code` | probe, notifier, clock, child factory injected via constructor |

`main()` parses argv, builds `Config`, and either `os.execvp`s the child command (watchdog
disabled) or runs `Supervisor` and `sys.exit`s with its result. Any unexpected exception in the
supervisor is logged with a traceback, the child is killed, and the exit code is 1.

### Detection

Every `WATCHDOG_INTERVAL` seconds (default 30) the supervisor GETs `WATCHDOG_HEALTH_URL`
(default `http://127.0.0.1:8000/health`) with `WATCHDOG_TIMEOUT` (default 10). Only HTTP 200
counts as success; a timeout, connection refused, any other status, or any exception counts as
failure (unexpected exception types are logged at `WARNING` with their class name). The
defaults match the Docker healthcheck (30s / 10s / 3 retries) which has run in production for
months without a false positive; the Docker healthcheck itself stays in place for `docker ps`.

The idle wait between probes is a loop of 1-second `child.wait()` calls, so a child exit is
noticed within a second and signal flags are checked at least once per second.

States:

- **STARTING** — from child start until the first HTTP 200. Failures are not counted (model
  preload, cold MIOpen kernel compilation). Bounded by `WATCHDOG_STARTUP_TIMEOUT` (default 600
  s); exceeding it is a hang with reason `startup_timeout`. Each failed probe logs one `WARNING`
  with the elapsed time; the first success logs `INFO healthy after N s`.
- **RUNNING** — `WATCHDOG_FAILURES` (default 3) consecutive failures mean a hang. A success
  resets the counter. Each failure logs `WARNING` with the count; successful probes are silent.
- **PENDING_RESTART** — the anti-flapping delay (below). Probing continues; a success cancels
  the pending restart and returns to RUNNING with the counter reset; one `INFO` line per minute
  reports the remaining delay. Entering PENDING_RESTART logs `WARNING` and sends no mail; the
  mail goes out with the kill.

### Reaction

`restart(reason)`:

1. Log one line: reason, uptime (monotonic seconds since the child was started), consecutive
   failures.
2. `os.killpg(child.pid, SIGKILL)` (`ProcessLookupError` means the child already died — fine).
3. `child.wait(timeout=max(WATCHDOG_STOP_GRACE, 1 s))` and log the outcome. The one-second floor
   exists because `wait(timeout=0)` is a single `WNOHANG` poll: a child killed microseconds
   earlier is not reaped yet and would be reported as surviving. If the child really is still
   alive after the wait (a process stuck in an uninterruptible kernel state cannot die yet), log
   `ERROR` and continue — but the container will not come back on its own either:
   `zap_pid_ns_processes` blocks the exiting PID-namespace init until every task in the namespace
   has gone, so the container stays in "exiting" and only a host reboot recovers it.
4. Notify (see Notifications). Bounded by the SMTP timeout; failure is logged and ignored. The
   kill deliberately comes first: a stalled relay would otherwise hold SIGKILL back by up to
   several SMTP timeouts, and the mail still leaves before the process exits.
5. Log the exit line and return exit code **3**.

Exit code 3 is reserved for watchdog-initiated restarts so it is distinguishable in `docker
events` / `docker inspect` from crashes. `restart: unless-stopped` restarts the container
regardless of the code. Time from hang to restart with defaults: about 2 minutes (3 × (30 + 10)
s) plus startup.

### Anti-flapping (stateless)

Memory across container lives would need a volume and a state file. Instead one rule uses the
current life's uptime:

> If a hang is detected in RUNNING and uptime < `WATCHDOG_MIN_UPTIME` (default 600 s), enter
> PENDING_RESTART and wait `WATCHDOG_FLAP_COOLDOWN` (default 900 s) before killing. Keep probing
> meanwhile; a success cancels the restart. `WATCHDOG_FLAP_COOLDOWN=0` disables the rule.

Effect when the GPU is dead for good:

| Scenario | Without the rule | With the rule |
|---|---|---|
| Hang during model preload | one restart per 10 min (startup timeout) | same |
| Healthy start, first request hangs | restart every ~2 min | one per ~17 min |
| Hang after hours of normal work (the incident) | immediate restart | immediate restart |

Known limitation: the supervisor cannot count "N restarts per hour" or send a "recovered" notice.
`docker inspect -f '{{.RestartCount}}'` on the host provides the count for free.

### Notifications

**Log (always).** Watchdog actions produce distinct `supervisor:` lines on stderr, visible in
`docker logs` and in the container's json-file log on the host. The supervisor's log level follows
`LOG_LEVEL` if it is a valid level name, else `INFO`.

**E-mail (optional).** Enabled when `WATCHDOG_MAIL_TO` is non-empty. `SmtpNotifier` builds an
`email.message.EmailMessage` and sends it with `smtplib.SMTP(host, port, timeout=10)`, STARTTLS
with a verifying `ssl.create_default_context()` when `WATCHDOG_SMTP_STARTTLS` is true (a relay
whose certificate does not match its hostname therefore fails), `login()` when a user is set. The
mail goes out after the kill, not before it (Reaction step 4). Every exception is caught and
logged at `WARNING`; the restart proceeds regardless. If `WATCHDOG_MAIL_TO` is set but
`WATCHDOG_SMTP_HOST` is empty, `Config` logs `ERROR mail disabled: WATCHDOG_SMTP_HOST is required`
and uses `NullNotifier`.

Events that send mail: `health_failed` (RUNNING or after the cooldown), `startup_timeout`,
`child_exited` (the child ended on its own, any code). A `docker stop` / SIGTERM shutdown sends
nothing. Subject: `[vision-api] restart: <reason> (<container hostname>)`. Body: reason, UTC
timestamp, uptime, consecutive failures, health URL, child command, exit code, and a hint to run
`docker logs` and check `RestartCount`.

### Signals and exit codes

The signal handlers only call `Supervisor.request_stop(signum)`; tests call the same method.

- **SIGTERM / SIGINT** (`docker stop`, Ctrl-C in `compose up`): forward the same signal to the
  child process only (not the group — today Docker signals only uvicorn, and the app's shutdown
  handles its ffmpeg children). Wait `WATCHDOG_STOP_GRACE` (default 8 s, inside Docker's 10 s
  grace); if the child is still alive, `killpg(SIGKILL)`. A second signal while stopping kills the
  group immediately. Exit with the child's normalized code. No notification.
- **Child exits on its own:** exit with the child's normalized code (negative `returncode` −N
  becomes 128 + N, e.g. 137 for SIGKILL). Notify `child_exited` unless a shutdown was requested.
- **Usage error** (no child command in argv): print usage to stderr, exit 2.
- **Child command cannot be started**: `execvp` failure when disabled, or a spawn that raises
  `OSError` under supervision (missing binary): log, exit 127, no mail.
- **Supervisor bug** (unexpected exception): traceback logged, child killed, exit 1.

### Configuration

All variables are read from the environment by the supervisor itself (not via `config.py`). A
malformed value — including a non-finite one such as `nan` or `inf` — logs `ERROR` and falls back
to the default: a typo in a watchdog setting must not keep the service from starting. Booleans accept `true/1/yes/on` and `false/0/no/off`,
case-insensitive; an empty value means "unset". The supervisor's own log level follows the
application's `LOG_LEVEL` when it is a valid level name, else `INFO`.

| Variable | Default | Meaning / validation |
|---|---|---|
| `WATCHDOG_ENABLED` | `true` | `false` → `execvp` the child command, no supervision (emergency lever) |
| `WATCHDOG_HEALTH_URL` | `http://127.0.0.1:8000/health` | Must be changed together with a `command:` port override |
| `WATCHDOG_INTERVAL` | `30` | seconds between probes, > 0 |
| `WATCHDOG_TIMEOUT` | `10` | HTTP timeout in seconds, > 0 |
| `WATCHDOG_FAILURES` | `3` | consecutive failures before reacting, integer ≥ 1 |
| `WATCHDOG_STARTUP_TIMEOUT` | `600` | seconds to wait for the first 200, > 0 |
| `WATCHDOG_MIN_UPTIME` | `600` | uptime below which a hang is "flapping", ≥ 0 |
| `WATCHDOG_FLAP_COOLDOWN` | `900` | delay before killing a flapping container, ≥ 0; 0 disables |
| `WATCHDOG_STOP_GRACE` | `8` | seconds after SIGTERM before SIGKILL, ≥ 0; keep below the compose `stop_grace_period` (10 s). The wait that confirms the SIGKILL is at least 1 s |
| `WATCHDOG_MAIL_TO` | empty | recipient; non-empty enables mail |
| `WATCHDOG_MAIL_FROM` | `vision-api@<hostname>` | sender |
| `WATCHDOG_SMTP_HOST` | empty | required when mail is enabled |
| `WATCHDOG_SMTP_PORT` | `587` | integer 1–65535 |
| `WATCHDOG_SMTP_USER` / `WATCHDOG_SMTP_PASSWORD` | empty | `login()` only when user is set |
| `WATCHDOG_SMTP_STARTTLS` | `true` | implicit TLS on port 465 is not supported |

The three deployment compose files (`docker/deploy/docker-compose-{amd,cpu,nvidia}.yml`) pass
through `WATCHDOG_ENABLED`, `WATCHDOG_STARTUP_TIMEOUT`, `WATCHDOG_FLAP_COOLDOWN`,
`WATCHDOG_MAIL_TO`, `WATCHDOG_MAIL_FROM`, `WATCHDOG_SMTP_HOST`, `WATCHDOG_SMTP_PORT`,
`WATCHDOG_SMTP_USER`, `WATCHDOG_SMTP_PASSWORD` as `${VAR:-}`. An empty interpolation is harmless
here because the supervisor treats empty as unset (this is our own parser, unlike the MIOpen case
documented in `.env.example`). The remaining tuning variables are documented and can be added to
a compose file when needed. Values live in `docker/deploy/.env`, which is git-ignored. The three
development compose files in `docker/` get `init: true` only.

## Changes by file

| File | Change |
|---|---|
| `app/supervisor.py` | **New.** Supervisor as designed above. Module docstring records why it is a process (GIL), why uvicorn must not stay PID 1 (namespace signal rule), and the exit-code contract. |
| `tests/test_supervisor.py` | **New.** See Testing. |
| `docker/amd/Dockerfile`, `docker/cpu/Dockerfile`, `docker/nvidia/Dockerfile` | `CMD ["python3", "supervisor.py", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`. `COPY app/ .` already ships the file. |
| `docker/docker-compose-{amd,cpu,nvidia}.yml` | `init: true` on the service. |
| `docker/deploy/docker-compose-{amd,cpu,nvidia}.yml` | `init: true` + the nine `WATCHDOG_*` pass-throughs. |
| `tests/test_compose.py` | New invariants: `init: true` in all six compose files; the nine pass-through keys in the three deploy files; the exact `CMD` line in the three Dockerfiles. Docstring gains a paragraph pointing at this spec's branch history. |
| `CLAUDE.md` | Architecture table row for `app/supervisor.py`; `WATCHDOG_*` block in Configuration; "Process watchdog" paragraph in Key Patterns. |
| `README.md` | `WATCHDOG_*` rows in the Configuration table; supervisor in Architecture / Project Structure. |
| `.claude/rules/docker.md` | Dockerfile overview step 7 → "Run `supervisor.py`, which runs uvicorn"; note under Health Check that the supervisor restarts the container on failure; Troubleshooting entry "container restarts every few minutes". |
| `docker/deploy/.env.example` | Commented `WATCHDOG_*` block with one-line explanations, including that the SMTP password is visible in `docker inspect`. |

`app/config.py` is untouched.

## Testing

All tests run under pytest inside the venv (via the build-runner agent). The state machine is
exercised with an injected probe, a fake child, and a fake clock — no sleeping, no network, no
real processes — except two smoke tests.

**Config**

- Defaults when the environment is empty.
- Every variable overridable.
- Garbage in a numeric variable → `ERROR` logged (`caplog`), default used.
- Boolean spellings for `WATCHDOG_ENABLED`.
- `WATCHDOG_MAIL_TO` without `WATCHDOG_SMTP_HOST` → `ERROR` logged, `NullNotifier`.

**State machine** (fake child: scripted `wait()` results; fake clock advanced by each wait)

1. Healthy forever → no kill, no notification.
2. STARTING: failures before the first 200 are ignored; the first 200 switches to RUNNING.
3. STARTING: startup timeout exceeded → killpg, notify `startup_timeout`, return 3.
4. RUNNING: two failures then a success → counter reset, no kill.
5. RUNNING: three failures at uptime ≥ `MIN_UPTIME` → immediate killpg, notify `health_failed`, return 3.
6. RUNNING: three failures at uptime < `MIN_UPTIME` → PENDING_RESTART; no kill before the cooldown elapses; kill after it.
7. PENDING_RESTART: a success cancels the restart; state is RUNNING with counter 0.
8. `FLAP_COOLDOWN=0` → immediate kill even at short uptime.
9. Child exits with code 7 → return 7, notify `child_exited`.
10. Child killed by signal 9 → return 137.
11. SIGTERM requested: signal forwarded to the child only; child exits 0 → return 0, no notification.
12. SIGTERM requested, child ignores it → killpg after `STOP_GRACE`, return 137, no notification.
13. Notifier that raises → restart still completes; `WARNING` logged.

**Notifier**

- `NullNotifier` does nothing.
- `SmtpNotifier` (mocked `smtplib.SMTP`): subject/body contents, STARTTLS and `login()` calls
  depending on config, exception swallowed and logged.

**HealthProbe** (a local `http.server` in a thread)

- 200 → `True`; 500 → `False`; closed port → `False`; handler slower than the timeout → `False`.

**Smoke** (real `subprocess`, POSIX)

- Child `sys.executable -c "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"`, probe always `False`, intervals of 0.1 s, `STARTUP_TIMEOUT` 0.5 s: `run()` returns 3 within a few seconds and the child is dead.
- Same child, probe always `True`, a timer thread requests a stop: the child ignores SIGTERM, dies by SIGKILL after `STOP_GRACE` 0.5 s, `run()` returns 137.

**Deployment invariants** (`tests/test_compose.py`) as listed in Changes by file.

## Rollout and verification

1. Merge the PR into `master`; tag the next minor version (`v0.7.0`). CI builds the three images
   and promotes `amd-latest` (the AMD build takes up to two hours).
2. On the server: pull `docker/deploy/` from git (compose files changed), optionally add the
   `WATCHDOG_*` mail settings to `.env`, run `deploy-up-detach-amd.sh` (it pulls and
   force-recreates).
3. Verify: `docker logs` shows the supervisor start line and `healthy after N s`;
   `docker top deploy-vision-api-1` shows tini → `python3 supervisor.py` → uvicorn.
4. Fire drill — the exact signature of the incident (process alive, port accepting, no answer):

   ```bash
   docker top deploy-vision-api-1 -o pid,cmd          # host PID of "uvicorn main:app"
   sudo kill -STOP <uvicorn host pid>
   # about two minutes later:
   docker logs --since 5m deploy-vision-api-1 | grep supervisor:
   docker inspect -f '{{.RestartCount}} {{.State.Health.Status}}' deploy-vision-api-1
   curl -sf http://localhost:3001/health
   ```

   Expected: a `health_failed` line, `RestartCount` 1, status `healthy`, `/health` answering.
5. Rollback: `WATCHDOG_ENABLED=false` in `.env` and `docker compose up -d`, or `IMAGE_TAG=0.6.0`.

If protection is needed before CI finishes, the same `supervisor.py` can be bind-mounted into the
current image via `volumes:` and `command:`. This is a manual stop-gap, not part of the
deliverable.

## Risks

- **False positive under load.** Three consecutive 10-second timeouts of `/health` within 90 s
  would restart a healthy container. The same thresholds have driven the Docker healthcheck in
  production without a false `unhealthy`; `WATCHDOG_ENABLED=false` is the kill switch.
- **Cold start longer than 10 minutes.** A first start with an empty MIOpen cache that compiles
  many kernels would loop with a 10-minute period. Raise `WATCHDOG_STARTUP_TIMEOUT`; the
  `miopen-cache` volume makes later starts fast.
- **In-memory jobs are lost on restart.** Already true for any restart; the job API is documented
  as in-memory.
- **Uninterruptible child.** Covered in Reaction step 3.

## Resolved questions from the incident notes

| Question | Decision |
|---|---|
| Notifications? | Log always; e-mail optional via SMTP settings. |
| Anti-flapping? | Stateless short-uptime cooldown (default 15 min); startup timeout bounds the preload loop. |
| Only vision-api or all containers? | Only vision-api; the feature lives inside its image and compose files. |
| Fix the root cause now? | No. Inference process isolation is a separate design. |
| Lower `max_workers` / `MIOPEN_FIND_MODE=FAST`? | No; no evidence ties them to the hang. |
