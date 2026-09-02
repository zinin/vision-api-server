"""Deployment-config invariants for the MIOpen FD-leak containment.

Guards docker-compose files against refactors that would silently drop the
`nofile` ulimits or the AMD MIOpen cache volumes. The design doc
(docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md) was
removed from the tree before the PR; it lives in the fix/upload-fd-leak
branch history.

Also guards the process-watchdog wiring (design doc
docs/superpowers/specs/2026-09-02-process-watchdog-design.md, likewise removed before the PR; it
lives in the feat/process-watchdog branch history): `init: true` and the `WATCHDOG_ENABLED`
kill switch in every compose file, the full `WATCHDOG_*` pass-through in the deployment files, and
the supervisor `CMD` in every Dockerfile.
"""
from pathlib import Path

import pytest
import yaml  # explicit dev dependency: these guards must fail loudly, never skip

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILES = sorted((REPO_ROOT / "docker").glob("docker-compose-*.yml")) + sorted(
    (REPO_ROOT / "docker" / "deploy").glob("docker-compose-*.yml")
)
AMD_FILES = [p for p in COMPOSE_FILES if "amd" in p.name]
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


def _service(data):
    return next(iter(data["services"].values()))


def test_expected_compose_files_present():
    assert len(COMPOSE_FILES) == 6, COMPOSE_FILES
    assert len(AMD_FILES) == 2, AMD_FILES


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_nofile_ulimit_is_65536(path):
    svc = _service(yaml.safe_load(path.read_text()))
    nofile = (svc.get("ulimits") or {}).get("nofile") or {}
    assert nofile.get("soft") == 65536, path
    assert nofile.get("hard") == 65536, path


@pytest.mark.parametrize("path", AMD_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_amd_miopen_cache_volumes(path):
    data = yaml.safe_load(path.read_text())
    svc = _service(data)
    mounts = svc.get("volumes") or []
    assert "miopen-cache:/root/.cache/miopen" in mounts, path
    assert "miopen-config:/root/.config/miopen" in mounts, path
    declared = data.get("volumes") or {}
    assert {"miopen-cache", "miopen-config"} <= set(declared), path


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


def _environment(svc):
    """The service environment as a mapping; the CPU dev file uses the list form (KEY=VALUE)."""
    env = svc.get("environment") or {}
    if isinstance(env, list):
        env = dict(item.partition("=")[::2] for item in env)
    return env


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_compose_forwards_the_watchdog_kill_switch(path):
    # WATCHDOG_ENABLED=false in .env is the documented emergency lever for the dev and deploy workflows alike
    env = _environment(_service(yaml.safe_load(path.read_text())))
    assert env.get("WATCHDOG_ENABLED") == "${WATCHDOG_ENABLED:-}", path


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.parent.name)
def test_dockerfile_runs_supervisor(path):
    cmd_lines = [line.strip() for line in path.read_text().splitlines() if line.strip().startswith("CMD")]
    assert cmd_lines == [SUPERVISOR_CMD], path
