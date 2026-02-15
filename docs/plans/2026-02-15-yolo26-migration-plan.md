# YOLO26 Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate from YOLO11 to YOLO26 by updating ultralytics dependency and replacing all model name references.

**Architecture:** Minimal migration — YOLO26 uses identical `predict()` API. Only dependency version and documentation/config model names change. No code logic changes needed.

**Tech Stack:** ultralytics >=8.4.0, Python, FastAPI, Docker

---

### Task 1: Update ultralytics dependency

**Files:**
- Modify: `requirements.txt:4`

**Step 1: Update version constraint**

Change line 4 from:
```
ultralytics>=8.3.250,<9.0.0
```
to:
```
ultralytics>=8.4.0,<9.0.0
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: bump ultralytics to >=8.4.0 for YOLO26 support (VAS-5)"
```

---

### Task 2: Update app source code references

**Files:**
- Modify: `app/main.py:313,387,467,756`
- Modify: `app/model_manager.py:107,128`
- Modify: `app/models.py:84`

**Step 1: Update app/main.py**

Replace all `yolo11s.pt` references in docstrings/descriptions (4 occurrences):
- Line 313: `ModelQuery` description — `yolo11s.pt` → `yolo26s.pt`
- Line 387: endpoint docstring — `yolo11s.pt` → `yolo26s.pt`
- Line 467: endpoint docstring — `yolo11s.pt` → `yolo26s.pt`
- Line 756: endpoint docstring — `yolo11s.pt` → `yolo26s.pt`

**Step 2: Update app/model_manager.py**

Replace docstring references (2 occurrences):
- Line 107: `{"yolo11s.pt": "cpu"}` → `{"yolo26s.pt": "cpu"}`
- Line 128: `'yolo11s.pt'` → `'yolo26s.pt'`

**Step 3: Update app/models.py**

- Line 84: example `"model": "yolo11s.pt"` → `"model": "yolo26s.pt"`

**Step 4: Run tests to verify nothing broke**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass (they use mocks, not real model names).

**Step 5: Commit**

```bash
git add app/main.py app/model_manager.py app/models.py
git commit -m "docs: update YOLO model references to yolo26 in app code (VAS-5)"
```

---

### Task 3: Update test fixtures

**Files:**
- Modify: `tests/test_endpoints.py:38`
- Modify: `tests/test_worker.py:74`

**Step 1: Update test_endpoints.py**

- Line 38: `"yolo11s.pt"` → `"yolo26s.pt"`

**Step 2: Update test_worker.py**

- Line 74: `"yolo11s.pt"` → `"yolo26s.pt"`

**Step 3: Run tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

**Step 4: Commit**

```bash
git add tests/test_endpoints.py tests/test_worker.py
git commit -m "test: update model name fixtures to yolo26s (VAS-5)"
```

---

### Task 4: Update Docker compose files

**Files:**
- Modify: `docker/docker-compose-nvidia.yml:9`
- Modify: `docker/docker-compose-cpu.yml:9`
- Modify: `docker/docker-compose-amd.yml:9`
- Modify: `docker/deploy/docker-compose-nvidia.yml:9`
- Modify: `docker/deploy/docker-compose-cpu.yml:7`
- Modify: `docker/deploy/docker-compose-amd.yml:10`
- Modify: `docker/deploy/.env.example:5,7`

**Step 1: Update docker/docker-compose-nvidia.yml**

```yaml
YOLO_MODELS: ${YOLO_MODELS:-'{"yolo26s.pt":"cuda:0","yolo26x.pt":"cuda:0"}'}
```

**Step 2: Update docker/docker-compose-cpu.yml**

```yaml
- YOLO_MODELS=${YOLO_MODELS:-'{"yolo26s.pt":"cpu","yolo26x.pt":"cpu"}'}
```

**Step 3: Update docker/docker-compose-amd.yml**

```yaml
YOLO_MODELS: ${YOLO_MODELS:-'{"yolo26s.pt":"cuda:0","yolo26x.pt":"cuda:0"}'}
```

**Step 4: Update docker/deploy/ compose files**

Same pattern for all three deploy compose files — replace `yolo11s` → `yolo26s` and `yolo11x` → `yolo26x`.

**Step 5: Update docker/deploy/.env.example**

```
# Available: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt

# YOLO_MODELS={"yolo26s.pt":"<device>","yolo26x.pt":"<device>"}
```

**Step 6: Commit**

```bash
git add docker/
git commit -m "chore: update Docker configs to YOLO26 model names (VAS-5)"
```

---

### Task 5: Update project documentation

**Files:**
- Modify: `CLAUDE.md:55`
- Modify: `.claude/rules/api.md:20,37,112,113,150,183-187`
- Modify: `.claude/rules/docker.md:54,174,182`

**Step 1: Update CLAUDE.md**

- Line 55: `yolo11s.pt` → `yolo26s.pt`

**Step 2: Update .claude/rules/api.md**

Replace all `yolo11` references:
- Line 20: `yolo11s.pt` → `yolo26s.pt`
- Line 37: `yolo11s.pt` → `yolo26s.pt`
- Line 112: `yolo11s.pt` → `yolo26s.pt`
- Line 113: `yolo11m.pt` → `yolo26m.pt`
- Line 150: `yolo11m.pt` → `yolo26m.pt`
- Lines 183-187: `yolo11n/s/m/l/x.pt` → `yolo26n/s/m/l/x.pt`

**Step 3: Update .claude/rules/docker.md**

- Line 54: `yolo11s.pt` → `yolo26s.pt`
- Line 174: `yolo11n.pt, yolo11s.pt` → `yolo26n.pt, yolo26s.pt`
- Line 182: `yolo11s.pt` → `yolo26s.pt`

**Step 4: Commit**

```bash
git add CLAUDE.md .claude/rules/
git commit -m "docs: update all documentation to reference YOLO26 models (VAS-5)"
```

---

### Task 6: Final verification

**Step 1: Grep for any remaining yolo11 references**

```bash
grep -r "yolo11" --include="*.py" --include="*.md" --include="*.yml" --include="*.yaml" --include="*.txt" --include="*.env*" .
```

Expected: Only the design document (`docs/plans/2026-02-15-yolo26-migration-design.md`) should reference `yolo11`.

**Step 2: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.
