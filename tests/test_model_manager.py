import asyncio
import gc
import time
import weakref
from unittest.mock import MagicMock, patch

import pytest

from model_manager import CachedModelEntry, ModelEntry, ModelManager


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


class TestCleanupExpired:
    @pytest.mark.parametrize("device", ["cuda:0", "cpu"])
    @pytest.mark.parametrize("tier", ["_cached", "_video_models"])
    async def test_eviction_frees_a_model_in_a_reference_cycle(self, tier, device):
        """A YOLO object that has run predict() sits in reference cycles, so
        dropping it frees nothing until the cyclic GC runs. The CUDA cache
        can only give the memory back after that collection."""
        mm = ModelManager(default_device=device, ttl_seconds=60)

        class CyclicModel:
            def __init__(self):
                self.itself = self

        model = CyclicModel()
        model_ref = weakref.ref(model)
        getattr(mm, tier)["yolo26x.pt"] = CachedModelEntry(
            entry=ModelEntry(model=model, visualizer=MagicMock(), model_name="yolo26x.pt", device=device),
            last_used_at=time.time() - 120,
        )
        del model
        freed_when_emptied = []

        gc.disable()  # no automatic collection: only cleanup_expired() may free the cycle
        try:
            with (
                patch("model_manager.torch.cuda.is_available", return_value=True),
                patch("model_manager.torch.cuda.empty_cache",
                      side_effect=lambda: freed_when_emptied.append(model_ref() is None)),
            ):
                assert await mm.cleanup_expired() == 1
            assert model_ref() is None, "the evicted model is still alive"
        finally:
            gc.enable()
        assert freed_when_emptied == [True]

    @pytest.mark.parametrize("default_device", ["cuda:0", "cpu"])
    async def test_eviction_empties_the_cuda_cache_whatever_the_default_device(self, default_device):
        """A video model lives on its preloaded model's device, which need not
        be the default one (YOLO_DEVICE=cpu with yolo26x.pt preloaded on
        cuda:0): its eviction must still hand the memory back."""
        mm = ModelManager(default_device=default_device, ttl_seconds=60)
        mm._video_models["yolo26x.pt"] = CachedModelEntry(
            entry=_entry("yolo26x.pt", "cuda:0"), last_used_at=time.time() - 120,
        )
        with (
            patch("model_manager.torch.cuda.is_available", return_value=True),
            patch("model_manager.torch.cuda.empty_cache") as empty_cache,
        ):
            assert await mm.cleanup_expired() == 1
        empty_cache.assert_called_once_with()

    async def test_no_collection_without_an_eviction(self, manager):
        with patch("gc.collect") as collect:
            assert await manager.cleanup_expired() == 0
        collect.assert_not_called()
