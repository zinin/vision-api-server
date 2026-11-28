"""Concurrent requests on one model instance keep their own conf, imgsz and max_det."""
import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import httpx
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes

from config import Settings, get_settings
from dependencies import get_model_manager
from inference_utils import run_inference, shutdown_executor
from main import app
from model_manager import ModelEntry

IMAGE = np.zeros((32, 32, 3), dtype=np.uint8)


class SharedPredictorModel:
    """Takes conf, imgsz and max_det the way Ultralytics before 8.4.209 does.

    There ``Model.predict()`` writes the call's arguments to the one cached
    predictor first and only then takes the predictor's lock; the run reads
    them from the predictor while it computes. The detections returned here
    spell out what the run read: ``max_det`` boxes with confidence ``conf``
    and x1 ``imgsz``.
    """

    names = {0: "person"}

    def __init__(self):
        self._args = None
        self._lock = threading.Lock()  # the predictor's own lock

    def predict(self, source, conf, imgsz, max_det, **_):
        self._args = (conf, imgsz, max_det)
        with self._lock:
            time.sleep(0.1)  # inference; the other requests arrive meanwhile
            conf, imgsz, max_det = self._args
        box = [float(imgsz), 0.0, float(imgsz) + 1.0, 1.0, conf, 0.0]
        return [SimpleNamespace(boxes=Boxes(torch.tensor([box] * max_det), orig_shape=(32, 32)))]


@pytest.fixture(autouse=True)
def _fresh_executor():
    """A pool of the default size, whatever an earlier test created."""
    shutdown_executor()
    yield
    shutdown_executor()


@pytest.fixture
def entry():
    visualizer = MagicMock()
    visualizer.draw_yolo_results.side_effect = lambda image, **_: image
    return ModelEntry(model=SharedPredictorModel(), visualizer=visualizer, model_name="m.pt", device="cpu")


@pytest.fixture
async def http(entry, tmp_path):
    manager = MagicMock()
    manager.get_model = AsyncMock(return_value=entry)
    app.dependency_overrides[get_model_manager] = lambda: manager
    # max_executor_workers pinned: a pool of one thread from MAX_EXECUTOR_WORKERS=1 never overlaps
    # the calls, and the race test would pass without the lock
    app.dependency_overrides[get_settings] = lambda: Settings(
        yolo_models="{}", video_jobs_dir=str(tmp_path), max_executor_workers=4
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


def _query(conf, imgsz, max_det):
    return {"model": "m.pt", "conf": conf, "imgsz": imgsz, "max_det": max_det}


def _image_upload():
    encoded, png = cv2.imencode(".png", IMAGE)
    assert encoded
    return {"file": ("frame.png", png.tobytes(), "image/png")}


async def _detect(http, conf, imgsz, max_det):
    """(confidence, x1, number of boxes) that /detect answered with."""
    response = await http.post("/detect", params=_query(conf, imgsz, max_det), files=_image_upload())
    response.raise_for_status()
    detections = response.json()["detections"]
    return detections[0]["confidence"], detections[0]["bbox"]["x1"], len(detections)


async def _detect_video(http, conf, imgsz, max_det):
    """The same triple for every frame /detect/video answered with."""
    response = await http.post(
        "/detect/video",
        params=_query(conf, imgsz, max_det),
        files={"file": ("clip.mp4", b"not decoded", "video/mp4")},
    )
    response.raise_for_status()
    return [
        (round(frame["detections"][0]["confidence"], 2), frame["detections"][0]["bbox"]["x1"], frame["count"])
        for frame in response.json()["frames"]
    ]


async def _visualize(http, conf, imgsz, max_det):
    """/detect/visualize answers with a picture: only the box count shows."""
    response = await http.post("/detect/visualize", params=_query(conf, imgsz, max_det), files=_image_upload())
    response.raise_for_status()
    return int(response.headers["X-Detections-Count"])


async def test_concurrent_requests_keep_their_own_parameters(http):
    extraction = SimpleNamespace(
        frames=[SimpleNamespace(frame_number=n, timestamp=n / 10, reason="grid", image=IMAGE) for n in (0, 10)],
        info=SimpleNamespace(duration=2.0),
    )
    with patch("main.extract_frames_from_video", AsyncMock(return_value=extraction)):
        image, video, visualized, second_image = await asyncio.gather(
            _detect(http, 0.11, 320, 1),
            _detect_video(http, 0.22, 640, 2),
            _visualize(http, 0.33, 960, 3),
            _detect(http, 0.44, 1280, 4),
        )

    assert image == (0.11, 320.0, 1)
    assert video == [(0.22, 640.0, 2)] * 2
    assert visualized == 3
    assert second_image == (0.44, 1280.0, 4)


async def test_other_instances_run_in_parallel():
    meeting = threading.Barrier(2, timeout=5)

    class MeetingModel:
        def predict(self, **_):
            meeting.wait()  # passes only while both instances are inside predict()
            return []

    entries = [
        ModelEntry(model=MeetingModel(), visualizer=None, model_name=name, device="cpu")
        for name in ("a.pt", "b.pt")
    ]

    await asyncio.gather(*(run_inference(entry, IMAGE, 0.5, 640, 100) for entry in entries))


async def test_cancelled_request_keeps_the_instance_until_predict_returns():
    entered = threading.Event()
    finish = threading.Event()
    log = []

    class SlowModel:
        def predict(self, conf, **_):
            log.append(f"start {conf}")
            entered.set()
            finish.wait(5)
            log.append(f"end {conf}")
            return []

    entry = ModelEntry(model=SlowModel(), visualizer=None, model_name="m.pt", device="cpu")
    first = asyncio.create_task(run_inference(entry, IMAGE, 0.1, 640, 100))
    assert await asyncio.to_thread(entered.wait, 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert entry.predict_lock.locked(), "cancelling the request released the instance while predict() still runs"

    second = asyncio.create_task(run_inference(entry, IMAGE, 0.2, 640, 100))
    await asyncio.sleep(0.2)  # long enough for the second call to get in, if nothing stops it
    assert log == ["start 0.1"], "the second call entered predict() while the first was still inside"

    finish.set()
    await second
    assert log == ["start 0.1", "end 0.1", "start 0.2", "end 0.2"]


async def test_failed_predict_releases_the_instance():
    calls = []

    class FailingOnceModel:
        def predict(self, conf, **_):
            calls.append(conf)
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return []

    entry = ModelEntry(model=FailingOnceModel(), visualizer=None, model_name="m.pt", device="cpu")
    with pytest.raises(RuntimeError, match="out of memory"):
        await run_inference(entry, IMAGE, 0.1, 640, 100)
    assert not entry.predict_lock.locked(), "a failed predict() left the instance locked"

    assert await run_inference(entry, IMAGE, 0.2, 640, 100) == []
    assert calls == [0.1, 0.2]
