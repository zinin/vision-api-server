"""Deployment-config invariants for the MIOpen FD-leak containment.

Guards docker-compose files against refactors that would silently drop the
`nofile` ulimits or the AMD MIOpen cache volumes (see
docs/superpowers/specs/2026-07-01-fd-leak-miopen-containment-design.md).
"""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")  # PyYAML arrives transitively via ultralytics

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILES = sorted((REPO_ROOT / "docker").glob("docker-compose-*.yml")) + sorted(
    (REPO_ROOT / "docker" / "deploy").glob("docker-compose-*.yml")
)
AMD_FILES = [p for p in COMPOSE_FILES if "amd" in p.name]


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
