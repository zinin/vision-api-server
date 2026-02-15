# YOLO26 Migration Design

**Date:** 2026-02-15
**Ticket:** VAS-5
**Approach:** Minimal migration (update deps + docs)

## Context

Current project uses `ultralytics>=8.3.250,<9.0.0` with YOLO11 models. YOLO26 was released January 14, 2026 (ultralytics v8.4.0) with full backward-compatible `predict()` API.

## YOLO26 Benefits

- **NMS-free end-to-end inference** — predictions without NMS post-processing, lower latency
- **Up to 43% faster CPU inference**
- **ProgLoss + STAL** — improved small object detection
- **DFL removal** — simpler model, better edge export
- **MuSGD optimizer** — better training stability

## What Changes

### 1. requirements.txt
- `ultralytics>=8.3.250,<9.0.0` → `ultralytics>=8.4.0,<9.0.0`

### 2. Documentation / Examples
Replace `yolo11s.pt` → `yolo26s.pt` in:
- `CLAUDE.md` — config examples
- `.env` / Docker env files
- `.claude/rules/api.md` (if references model names)
- Any README or doc referencing model names

### 3. Tests
- Update any hardcoded model names in test fixtures
- Existing mock-based tests should pass without changes (they mock YOLO results, not real models)

## What Does NOT Change

- `app/model_manager.py` — `YOLO()`, `model.to()`, `model.names`, `model.predict()` API identical
- `app/inference_utils.py` — `model.predict(source, conf, imgsz, max_det, verbose)` unchanged
- `app/video_annotator.py` — same detection extraction pattern
- `app/visualization.py` — `result.boxes.xyxy/cls/conf` unchanged
- All API endpoints — no changes
- Docker build process — `pip install` picks up new version from requirements.txt

## API Compatibility Verification

All YOLO APIs used in the project are stable across YOLO11 → YOLO26:

| API | Status |
|-----|--------|
| `YOLO(model_name)` | Compatible |
| `model.to(device)` | Compatible |
| `model.predict(source, conf, imgsz, max_det, verbose)` | Compatible |
| `model.names` | Compatible |
| `model.model.parameters()` | Compatible |
| `result.boxes.xyxy` | Compatible |
| `result.boxes.cls` | Compatible |
| `result.boxes.conf` | Compatible |

## Risks

- **Low:** API fully backward compatible, no breaking changes
- **Docker:** ultralytics 8.4.x must be compatible with PyTorch versions in CUDA/ROCm images — verify during build

## Model Naming

Default model in examples: `yolo26s.pt` (Small — best speed/accuracy balance for API server)
