"""Background threads that keep the ffmpeg pipes busy while the main thread computes.

Pass 1 of the video annotator reads frames on a ``ThreadedFrameReader`` thread
while the main thread runs YOLO; pass 2 writes frames on a
``ThreadedFrameWriter`` thread while the main thread reads and draws the next
one. Pipe I/O, cv2 and torch release the GIL, so the threads really overlap.
"""
import logging
import queue
import threading
from typing import Any, Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

_POLL_SECONDS = 0.1
_JOIN_SECONDS = 10.0
_END = object()


class _Failure:
    """An exception raised on a background thread, handed to the main thread."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


class ThreadedFrameReader:
    """Read decoder frames on a background thread.

    The thread reads each frame into one reusable buffer of
    ``decoder.frame_shape``, calls ``transform(frame_num, frame)`` and queues
    ``(frame_num, result)``. The buffer is overwritten by the next read, so
    ``transform`` must return a new object, or None to only count the frame.

    Iterate over the reader to consume the frames; an exception from the
    thread is re-raised by the iterator. Leaving the ``with`` block before the
    end kills the decoder through ``decoder.abort()`` so that a read blocked in
    the pipe returns, then joins the thread.
    """

    def __init__(
        self,
        decoder: Any,
        transform: Callable[[int, np.ndarray], Any],
        queue_size: int,
    ):
        self._decoder = decoder
        self._transform = transform
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._finished = False  # the consumer got the end marker or the failure
        self._thread = threading.Thread(target=self._run, name="frame-reader", daemon=True)
        self._thread.start()

    def _put(self, item: Any) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=_POLL_SECONDS)
                return True
            except queue.Full:
                pass
        return False

    def _run(self) -> None:
        try:
            buf = np.empty(self._decoder.frame_shape, dtype=np.uint8)
            frame_num = 0
            while not self._stop.is_set() and self._decoder.read_into(buf):
                if not self._put((frame_num, self._transform(frame_num, buf))):
                    return
                frame_num += 1
            self._put(_END)
        except BaseException as exc:  # handed to the consumer, re-raised there
            self._put(_Failure(exc))

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        while True:
            try:
                item = self._queue.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    self._finished = True
                    raise RuntimeError("Frame reader thread stopped without an end marker")
                continue
            if item is _END:
                self._finished = True
                return
            if isinstance(item, _Failure):
                self._finished = True
                raise item.exc
            yield item

    def close(self) -> None:
        if not self._finished:
            self._stop.set()
            self._decoder.abort()
        self._thread.join(timeout=_JOIN_SECONDS)
        if self._thread.is_alive():
            logger.warning(f"Frame reader thread did not stop within {_JOIN_SECONDS:.0f}s")

    def __enter__(self) -> "ThreadedFrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ThreadedFrameWriter:
    """Write frames to an encoder on a background thread.

    ``acquire()`` hands out a buffer of ``frame_shape`` from a fixed pool,
    ``submit()`` queues it, and the thread returns it to the pool once
    ``encoder.write_frame`` is done with it. ``submit()`` returns False after
    the encoder finished early (``write_frame`` returned False, e.g. on
    ``-shortest``); callers stop feeding frames then.

    An exception from ``write_frame`` is re-raised by the next ``acquire()``,
    ``submit()`` or clean exit from the ``with`` block. Leaving the block with
    an exception drops the frames still queued; a clean exit writes them all.
    """

    def __init__(
        self,
        encoder: Any,
        frame_shape: tuple[int, ...],
        pool_size: int = 4,
        queue_size: int = 2,
    ):
        self._encoder = encoder
        self._pool: queue.Queue = queue.Queue()
        for _ in range(pool_size):
            self._pool.put(np.empty(frame_shape, dtype=np.uint8))
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._drop = threading.Event()
        self._eof = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="frame-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            buf = self._queue.get()
            if buf is _END:
                return
            try:
                if self._error is None and not self._drop.is_set() and not self._eof.is_set():
                    if not self._encoder.write_frame(buf):
                        self._eof.set()
            except BaseException as exc:  # re-raised on the main thread
                self._error = exc
            finally:
                self._pool.put(buf)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def acquire(self) -> np.ndarray:
        """A free buffer from the pool; blocks while all of them are queued."""
        self._raise_if_failed()
        return self._pool.get()

    def release(self, buf: np.ndarray) -> None:
        """Return a buffer that will not be submitted."""
        self._pool.put(buf)

    def submit(self, buf: np.ndarray) -> bool:
        """Queue ``buf`` for writing. False once the encoder has finished."""
        self._raise_if_failed()
        if self._eof.is_set():
            self._pool.put(buf)
            return False
        self._queue.put(buf)
        return True

    def close(self, drop: bool = False) -> None:
        if drop:
            self._drop.set()
        self._queue.put(_END)
        # No timeout: the encoder closes its stdin right after this returns, so
        # the thread must be done with it. A slow CPU encoder can take seconds
        # per 4K frame; a hung one blocked the old single-threaded loop too.
        self._thread.join()

    def __enter__(self) -> "ThreadedFrameWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close(drop=exc_type is not None)
        if exc_type is None:
            self._raise_if_failed()
