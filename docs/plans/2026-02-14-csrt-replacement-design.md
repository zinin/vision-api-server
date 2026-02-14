# Design: Replace CSRT Tracker with YOLO + Hold Mode

**Date:** 2026-02-14
**Status:** Draft
**Task:** VAS-2 (Video Annotation performance optimization)

## Problem

Video annotation pipeline (`VideoAnnotator`) uses OpenCV CSRT tracker for intermediate frames between YOLO detections. CSRT runs on CPU at ~220ms/frame (OpenCV 4.10+ regression), resulting in ~5.4 fps while GPU sits idle 99% of the time.

Benchmark (2-minute video, 1651 frames, yolo11x@1024, detect_every=150):

| Operation | Frames | Time/frame | Total |
|-----------|--------|------------|-------|
| YOLO (GPU) | 12 | ~50ms | ~0.6s |
| CSRT tracking (CPU) | 1639 | ~220ms | ~360s |
| Read/Write/Draw | 1651 | ~1ms | ~1.6s |

**Root cause:** CSRT is CPU-only, no GPU support in Python OpenCV. Became even slower in OpenCV 4.10+.

## Solution

Remove CSRT entirely. Replace with:

- **Detection frames** (`frame_num % detect_every == 0`): run `model.predict()` on GPU
- **Intermediate frames**: hold (freeze) last known detections — zero CPU/GPU cost, just redraw cached bboxes

## Design Decisions

### 1. `model.predict()` over `model.track()`

`model.track()` stores tracker state in `model.predictor.trackers`. The model instance is shared between the annotation worker and API endpoints (`/detect`, `/detect/visualize`). Concurrent access would corrupt tracker state. `model.predict()` is stateless — no conflicts.

Additionally, `model.track()` doesn't solve class flipping (class comes from per-frame YOLO anyway), and track IDs are not displayed in the current UI (bboxes show class_name + confidence only).

### 2. Hold mode over alternative trackers

| Option | Speed | Quality | Complexity |
|--------|-------|---------|------------|
| **Hold (chosen)** | ~0ms/frame | Static bboxes between detections | Minimal — remove code |
| MOSSE tracker | ~1ms/frame | Low accuracy, loses objects | Keep tracker infra |
| KCF tracker | ~3-5ms/frame | Medium accuracy | Keep tracker infra |
| Empty frames | ~0ms/frame | Bboxes flicker/disappear | Minimal |

Hold is the simplest option that gives acceptable visual quality. At detect_every=5 (default), bboxes refresh every ~170ms at 30fps — imperceptible in most cases.

### 3. Keep `default_detect_every=5`

GPU economy by default. Users can set `detect_every=1` for maximum quality when GPU resources are not a concern.

### 4. Remove CSRT entirely

No replacement tracker. `_create_csrt_tracker()`, `_init_trackers()`, `_update_trackers()` become dead code. `opencv-contrib-python-headless` stays in requirements — still needed for `cv2.VideoWriter`, `cv2.VideoCapture`, etc.

## Expected Performance

| detect_every | Approach | fps (estimated) |
|-------------|----------|----------------|
| 5 (default) | Current CSRT | ~5.4 fps |
| 5 (default) | **New hold mode** | **~90+ fps** |
| 1 | New (YOLO every frame) | ~20-33 fps |

## API Impact

None. `detect_every` parameter unchanged (range 1-300, default 5). Same endpoints, same response format. `AnnotationStats.tracked_frames` semantics shift from "CSRT-tracked" to "hold-reused", but field name stays the same.

## Files to Modify

| File | Change |
|------|--------|
| `app/video_annotator.py` | Remove CSRT functions/methods, simplify frame loop |
| `tests/test_video_annotator.py` | Remove CSRT tests, add hold mode tests |

## Files Unchanged

- `app/config.py` — `default_detect_every=5` stays
- `app/main.py` — `_annotation_worker()` unchanged
- `app/visualization.py` — `draw_detection()` unchanged
- `app/models.py`, `app/job_manager.py`, `app/dependencies.py` — unchanged
