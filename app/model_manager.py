import asyncio
import gc
import logging
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen

import torch
from ultralytics import YOLO

from visualization import DetectionVisualizer

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 30  # socket timeout in seconds (per read operation)
DOWNLOAD_RETRIES = 3
ULTRALYTICS_ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


@dataclass
class ModelEntry:
    """Container for a loaded YOLO model and its resources."""
    model: YOLO
    visualizer: DetectionVisualizer
    model_name: str
    device: str


@dataclass
class CachedModelEntry:
    """Model entry with TTL tracking for cache eviction."""
    entry: ModelEntry
    last_used_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used_at = time.time()

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired based on TTL."""
        return (time.time() - self.last_used_at) > ttl_seconds


class ModelManager:
    """
    Manages loading, caching, and unloading of YOLO models.

    - Preloaded models: loaded at startup, never evicted
    - Cached models: loaded on-demand, evicted after TTL
    """

    def __init__(self, default_device: str, ttl_seconds: int = 900, models_dir: str = ""):
        self.default_device = default_device
        self.ttl_seconds = ttl_seconds
        self.models_dir = models_dir

        self._preloaded: dict[str, ModelEntry] = {}
        self._cached: dict[str, CachedModelEntry] = {}
        # Instances of their own for video annotation jobs, see get_video_model()
        self._video_models: dict[str, CachedModelEntry] = {}
        self._loading_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        self._cleanup_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    @property
    def default_model(self) -> str | None:
        """Return the first preloaded model name as default."""
        if self._preloaded:
            return next(iter(self._preloaded.keys()))
        return None

    def is_preloaded(self, model_name: str) -> bool:
        """True for a model loaded at startup from YOLO_MODELS."""
        return model_name in self._preloaded

    @staticmethod
    def _is_valid_model_file(path: Path) -> bool:
        """Check if model file is a valid zip archive (PyTorch model format)."""
        try:
            with zipfile.ZipFile(path) as zf:
                zf.namelist()
            return True
        except (zipfile.BadZipFile, OSError):
            return False

    @staticmethod
    def _download_model(model_name: str, dest: Path) -> None:
        """Download model from GitHub with timeout and retry."""
        url = f"{ULTRALYTICS_ASSETS_URL}/{model_name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")

        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                logger.info(f"Downloading {model_name} (attempt {attempt}/{DOWNLOAD_RETRIES})")
                resp = urlopen(url, timeout=DOWNLOAD_TIMEOUT)
                total = int(resp.getheader("Content-Length", 0))
                downloaded = 0

                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                if total and downloaded != total:
                    logger.warning(f"Incomplete download: {downloaded}/{total} bytes")
                    tmp.unlink(missing_ok=True)
                    continue

                if not ModelManager._is_valid_model_file(tmp):
                    logger.warning(f"Downloaded file failed integrity check")
                    tmp.unlink(missing_ok=True)
                    continue

                tmp.rename(dest)
                logger.info(f"Model {model_name} downloaded ({downloaded / 1048576:.1f} MB)")
                return

            except Exception as e:
                logger.warning(f"Download attempt {attempt} failed: {e}")
                tmp.unlink(missing_ok=True)

        raise RuntimeError(
            f"Failed to download {model_name} after {DOWNLOAD_RETRIES} attempts"
        )

    def _ensure_model_file(self, model_name: str, model_path: str) -> None:
        """Ensure model file exists and is valid, downloading if needed."""
        path = Path(model_path)
        if path.exists():
            if self._is_valid_model_file(path):
                return
            logger.warning(f"Corrupt model file: {model_path}, removing for re-download")
            path.unlink()
        self._download_model(model_name, path)

    @staticmethod
    def _safe_model_path(models_dir: str, model_name: str) -> str:
        """Resolve model path ensuring it stays within models_dir (path traversal protection)."""
        resolved = Path(models_dir).resolve() / Path(model_name).name
        return str(resolved)

    def _load_model_sync(self, model_name: str, device: str) -> ModelEntry:
        """Synchronously load a YOLO model on specified device (blocking)."""
        logger.info(f"Loading model: {model_name} on device: {device}")
        model_path = self._safe_model_path(self.models_dir, model_name) if self.models_dir else model_name

        if self.models_dir:
            self._ensure_model_file(model_name, model_path)

        try:
            model = YOLO(model_path)
            model.to(device)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                f"Failed to load {model_name} on {device}: CUDA out of memory. "
                f"Try a smaller model or different device."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name} on {device}: {e}")

        visualizer = DetectionVisualizer(model.names)

        device_info = next(model.model.parameters()).device
        logger.info(f"Model {model_name} loaded successfully on device: {device_info}")

        return ModelEntry(
            model=model,
            visualizer=visualizer,
            model_name=model_name,
            device=device
        )

    async def _load_model_async(self, model_name: str, device: str) -> ModelEntry:
        """Load model in thread pool to avoid blocking event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._load_model_sync, model_name, device
        )

    async def preload_models(self, model_device_map: dict[str, str]) -> None:
        """Preload models at startup. These models are never evicted.

        Args:
            model_device_map: Mapping of model name to device (e.g. {"yolo26s.pt": "cpu"})
        """
        for model_name, device in model_device_map.items():
            if model_name in self._preloaded:
                logger.debug(f"Model {model_name} already preloaded, skipping")
                continue

            try:
                entry = await self._load_model_async(model_name, device)
                self._preloaded[model_name] = entry
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")
                raise RuntimeError(f"Failed to preload model {model_name}: {e}") from e

        logger.info(f"Preloaded {len(self._preloaded)} models: {list(self._preloaded.keys())}")

    async def get_model(self, model_name: str | None = None) -> ModelEntry:
        """
        Get a model by name. Loads on-demand if not already loaded.

        Args:
            model_name: Model name (e.g. 'yolo26s.pt'). If None, returns default model.

        Returns:
            ModelEntry with the loaded model and visualizer.

        Raises:
            ValueError: If no model name provided and no default model available.
            RuntimeError: If model loading fails.
        """
        if model_name is None:
            # Try to use first preloaded model as default
            if self._preloaded:
                model_name = next(iter(self._preloaded.keys()))
            else:
                raise ValueError("No model specified and no default model available")

        # Check preloaded models first
        if model_name in self._preloaded:
            return self._preloaded[model_name]

        # Check cached models
        if model_name in self._cached:
            cached = self._cached[model_name]
            cached.touch()
            return cached.entry

        # Need to load the model - get or create lock for this model
        async with self._global_lock:
            if model_name not in self._loading_locks:
                self._loading_locks[model_name] = asyncio.Lock()
            lock = self._loading_locks[model_name]

        # Load with model-specific lock to prevent duplicate loads
        async with lock:
            # Double-check after acquiring lock
            if model_name in self._cached:
                cached = self._cached[model_name]
                cached.touch()
                return cached.entry

            # Load the model on default device
            try:
                entry = await self._load_model_async(model_name, self.default_device)
                self._cached[model_name] = CachedModelEntry(entry=entry)
                logger.info(f"Model {model_name} loaded on-demand on {self.default_device} and cached (TTL: {self.ttl_seconds}s)")
                return entry
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise RuntimeError(f"Failed to load model {model_name}: {e}") from e
            finally:
                # Clean up lock
                async with self._global_lock:
                    self._loading_locks.pop(model_name, None)

    async def get_video_model(self, model_name: str | None = None) -> ModelEntry:
        """A model instance of its own for video annotation jobs.

        An Ultralytics model predicts through one cached predictor that holds
        its own copy of the weights (FP16 under ``quantize=16``) and the
        arguments of the latest call. On the instance ``/detect`` uses, the
        two would rebuild that predictor (deep copy, fuse, warm-up) whenever
        ``quantize`` alternates between their calls, and a call on one thread
        would replace the ``conf`` and ``imgsz`` that a run on the other
        reads. The instance is loaded from the same file onto the same device
        as ``get_model`` would use, cached, and evicted after ``ttl_seconds``
        without a job.

        Raises:
            ValueError: If no model name provided and no default model available.
            RuntimeError: If model loading fails.
        """
        if model_name is None:
            if not self._preloaded:
                raise ValueError("No model specified and no default model available")
            model_name = next(iter(self._preloaded))

        cached = self._video_models.get(model_name)
        if cached is not None:
            cached.touch()
            return cached.entry

        lock_key = f"video:{model_name}"
        async with self._global_lock:
            if lock_key not in self._loading_locks:
                self._loading_locks[lock_key] = asyncio.Lock()
            lock = self._loading_locks[lock_key]

        async with lock:
            cached = self._video_models.get(model_name)
            if cached is not None:
                cached.touch()
                return cached.entry

            preloaded = self._preloaded.get(model_name)
            device = preloaded.device if preloaded is not None else self.default_device
            try:
                entry = await self._load_model_async(model_name, device)
                self._video_models[model_name] = CachedModelEntry(entry=entry)
                logger.info(f"Video model {model_name} loaded on {device} (TTL: {self.ttl_seconds}s)")
                return entry
            except Exception as e:
                logger.error(f"Failed to load video model {model_name}: {e}")
                raise RuntimeError(f"Failed to load model {model_name}: {e}") from e
            finally:
                async with self._global_lock:
                    self._loading_locks.pop(lock_key, None)

    async def cleanup_expired(self) -> int:
        """Remove expired models from cache. Returns count of evicted models."""
        evicted = 0
        expired_keys = [
            name for name, cached in self._cached.items()
            if cached.is_expired(self.ttl_seconds)
        ]

        for model_name in expired_keys:
            cached = self._cached.pop(model_name, None)
            if cached:
                logger.info(f"Evicting expired model from cache: {model_name}")
                # Help garbage collector
                del cached.entry.model
                del cached.entry.visualizer
                evicted += 1

        # A job longer than the TTL keeps its own reference to the model, so
        # evicting the entry mid-job frees nothing here: the annotation worker
        # collects the instance once the job lets go of it.
        expired_video = [
            name for name, cached in self._video_models.items()
            if cached.is_expired(self.ttl_seconds)
        ]
        for model_name in expired_video:
            cached = self._video_models.pop(model_name, None)
            if cached:
                logger.info(f"Evicting idle video model: {model_name}")
                del cached.entry.model
                del cached.entry.visualizer
                evicted += 1

        if evicted > 0:
            # A YOLO object that has run predict() sits in reference cycles:
            # the dels above free nothing until the cyclic GC runs, and a full
            # collection may not come for a long time. Collect first, so that
            # empty_cache() can hand the evicted models' memory back.
            gc.collect()
            # Not gated on the default device: a video model lives on its
            # preloaded model's device (YOLO_DEVICE=cpu with a model preloaded
            # on cuda:0), and predict() without device= runs on the GPU even
            # for a model loaded on the CPU. empty_cache() does nothing while
            # CUDA is uninitialised.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared after model eviction")

        return evicted

    async def _cleanup_loop(self, interval_seconds: int = 60) -> None:
        """Background task that periodically cleans up expired models."""
        logger.info(f"Starting model cleanup task (interval: {interval_seconds}s, TTL: {self.ttl_seconds}s)")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval_seconds
                )
                # If we get here, shutdown was signaled
                break
            except asyncio.TimeoutError:
                # Normal timeout - run cleanup
                evicted = await self.cleanup_expired()
                if evicted > 0:
                    logger.info(f"Cleanup task evicted {evicted} expired model(s)")

        logger.info("Model cleanup task stopped")

    def start_cleanup_task(self, interval_seconds: int = 60) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._shutdown_event.clear()
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(interval_seconds)
            )

    async def shutdown(self) -> None:
        """Gracefully shutdown the model manager."""
        logger.info("Shutting down ModelManager...")

        # Stop cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._cleanup_task.cancel()
                logger.warning("Cleanup task did not stop gracefully, cancelled")

        # Clear cached models
        for model_name, cached in list(self._cached.items()):
            logger.debug(f"Unloading cached model: {model_name}")
            del cached.entry.model
            del cached.entry.visualizer
        self._cached.clear()

        for model_name, cached in list(self._video_models.items()):
            logger.debug(f"Unloading video model: {model_name}")
            del cached.entry.model
            del cached.entry.visualizer
        self._video_models.clear()

        # Clear preloaded models
        for model_name, entry in list(self._preloaded.items()):
            logger.debug(f"Unloading preloaded model: {model_name}")
            del entry.model
            del entry.visualizer
        self._preloaded.clear()

        # Clear CUDA cache
        if self.default_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("ModelManager shutdown complete")

    def get_status(self) -> dict:
        """Get current status of loaded models."""
        now = time.time()

        preloaded_info = [
            {"name": name, "device": entry.device}
            for name, entry in self._preloaded.items()
        ]

        def with_ttl(entries: dict[str, CachedModelEntry]) -> list[dict]:
            return [
                {
                    "name": name,
                    "device": cached.entry.device,
                    "expires_in_seconds": int(max(0, self.ttl_seconds - (now - cached.last_used_at)))
                }
                for name, cached in entries.items()
            ]

        return {
            "preloaded": preloaded_info,
            "cached": with_ttl(self._cached),
            # Video jobs' own instances: a second copy of the weights until eviction
            "video": with_ttl(self._video_models),
            "default_device": self.default_device,
            "ttl_seconds": self.ttl_seconds
        }
