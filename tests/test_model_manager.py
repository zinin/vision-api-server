import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from model_manager import ModelEntry, ModelManager


def _entry(name: str, device: str) -> ModelEntry:
    return ModelEntry(model=MagicMock(), visualizer=MagicMock(), model_name=name, device=device)


@pytest.fixture
def manager():
    mm = ModelManager(default_device="cpu", ttl_seconds=60)
    mm._preloaded["yolo26s.pt"] = _entry("yolo26s.pt", "cuda:0")
    mm._preloaded["yolo26x.pt"] = _entry("yolo26x.pt", "cuda:0")
    return mm


def _loader(delay: float = 0.0):
    """Stand-in for _load_model_sync: a fresh entry per call, optionally slow."""
    def load(model_name, device):
        if delay:
            time.sleep(delay)
        return _entry(model_name, device)
    return MagicMock(side_effect=load)


class TestGetVideoModel:
    async def test_separate_instance_on_the_preloaded_device(self, manager):
        loader = _loader()
        with patch.object(manager, "_load_model_sync", loader):
            entry = await manager.get_video_model("yolo26x.pt")

        loader.assert_called_once_with("yolo26x.pt", "cuda:0")
        assert entry is not manager._preloaded["yolo26x.pt"]
        assert entry.model is not manager._preloaded["yolo26x.pt"].model
        assert entry.device == "cuda:0"

    async def test_none_means_the_default_model(self, manager):
        loader = _loader()
        with patch.object(manager, "_load_model_sync", loader):
            entry = await manager.get_video_model(None)
        assert entry.model_name == "yolo26s.pt"

    async def test_no_default_model_raises_value_error(self):
        with pytest.raises(ValueError):
            await ModelManager(default_device="cpu").get_video_model(None)

    async def test_not_preloaded_model_uses_the_default_device(self, manager):
        loader = _loader()
        with patch.object(manager, "_load_model_sync", loader):
            await manager.get_video_model("yolo26m.pt")
        loader.assert_called_once_with("yolo26m.pt", "cpu")

    async def test_cached_between_calls(self, manager):
        loader = _loader()
        with patch.object(manager, "_load_model_sync", loader):
            first = await manager.get_video_model("yolo26x.pt")
            second = await manager.get_video_model("yolo26x.pt")
        assert first is second
        assert loader.call_count == 1

    async def test_concurrent_calls_load_once(self, manager):
        loader = _loader(delay=0.1)
        with patch.object(manager, "_load_model_sync", loader):
            first, second = await asyncio.gather(
                manager.get_video_model("yolo26x.pt"),
                manager.get_video_model("yolo26x.pt"),
            )
        assert first is second
        assert loader.call_count == 1

    async def test_load_failure_raises_runtime_error(self, manager):
        loader = MagicMock(side_effect=OSError("file is corrupt"))
        with patch.object(manager, "_load_model_sync", loader):
            with pytest.raises(RuntimeError, match="file is corrupt"):
                await manager.get_video_model("yolo26x.pt")
        assert "yolo26x.pt" not in manager._video_models

    async def test_idle_video_model_is_evicted(self, manager):
        with patch.object(manager, "_load_model_sync", _loader()):
            await manager.get_video_model("yolo26x.pt")
        manager._video_models["yolo26x.pt"].last_used_at -= 120

        assert await manager.cleanup_expired() == 1
        assert "yolo26x.pt" not in manager._video_models
        assert "yolo26x.pt" in manager._preloaded

    async def test_recently_used_video_model_is_kept(self, manager):
        with patch.object(manager, "_load_model_sync", _loader()):
            await manager.get_video_model("yolo26x.pt")
        assert await manager.cleanup_expired() == 0
        assert "yolo26x.pt" in manager._video_models

    async def test_shutdown_clears_video_models(self, manager):
        with patch.object(manager, "_load_model_sync", _loader()):
            await manager.get_video_model("yolo26x.pt")
        await manager.shutdown()
        assert manager._video_models == {}
