import threading
import time

import numpy as np
import pytest

from frame_threads import ThreadedFrameReader, ThreadedFrameWriter


class FakeDecoder:
    """read_into() fills the buffer with the frame number.

    ``fail_at`` raises on that read; ``block_at`` blocks that read until
    abort() is called, like a pipe read that only killing ffmpeg ends.
    """

    def __init__(self, frames=5, shape=(2, 3), fail_at=None, block_at=None):
        self.frame_shape = shape
        self._frames = frames
        self._fail_at = fail_at
        self._block_at = block_at
        self.reads = 0
        self.aborted = threading.Event()

    def read_into(self, buf):
        n = self.reads
        if n == self._block_at:
            self.aborted.wait(timeout=5)
            raise RuntimeError("FFmpeg decoder crashed (killed by signal 9)")
        if n == self._fail_at:
            raise RuntimeError("FFmpeg decoder crashed (rc=1)")
        if n >= self._frames:
            return False
        buf.fill(n)
        self.reads += 1
        return True

    def abort(self):
        self.aborted.set()


def _copy(frame_num, frame):
    return frame.copy()


class TestThreadedFrameReader:
    def test_yields_transformed_frames_in_order(self):
        decoder = FakeDecoder(frames=5)
        with ThreadedFrameReader(decoder, _copy, queue_size=2) as frames:
            items = list(frames)
        assert [n for n, _ in items] == [0, 1, 2, 3, 4]
        # The buffer is reused, so each payload must be the copy made for its frame.
        assert [int(payload[0, 0]) for _, payload in items] == [0, 1, 2, 3, 4]

    def test_none_from_transform_still_counts_the_frame(self):
        decoder = FakeDecoder(frames=4)

        def even_only(frame_num, frame):
            return frame.copy() if frame_num % 2 == 0 else None

        with ThreadedFrameReader(decoder, even_only, queue_size=2) as frames:
            items = list(frames)
        assert [(n, payload is None) for n, payload in items] == [
            (0, False), (1, True), (2, False), (3, True),
        ]

    def test_decoder_error_is_raised_in_the_consumer(self):
        decoder = FakeDecoder(frames=5, fail_at=2)
        received = []
        with pytest.raises(RuntimeError, match="decoder crashed"):
            with ThreadedFrameReader(decoder, _copy, queue_size=2) as frames:
                for n, _ in frames:
                    received.append(n)
        assert received == [0, 1]

    def test_transform_error_is_raised_in_the_consumer(self):
        def broken(frame_num, frame):
            raise ValueError("bad frame")

        with pytest.raises(ValueError, match="bad frame"):
            with ThreadedFrameReader(FakeDecoder(frames=3), broken, queue_size=2) as frames:
                list(frames)

    def test_leaving_early_aborts_a_blocked_read(self):
        decoder = FakeDecoder(frames=10, block_at=1)
        started = time.monotonic()
        with pytest.raises(KeyError):
            with ThreadedFrameReader(decoder, _copy, queue_size=2) as frames:
                for _ in frames:
                    raise KeyError("consumer gave up")
        assert decoder.aborted.is_set()
        assert time.monotonic() - started < 2

    def test_leaving_early_with_a_full_queue_does_not_hang(self):
        decoder = FakeDecoder(frames=1000)
        started = time.monotonic()
        with ThreadedFrameReader(decoder, _copy, queue_size=1) as frames:
            next(iter(frames))
            time.sleep(0.2)  # let the reader fill the queue and block on put()
        assert decoder.aborted.is_set()
        assert time.monotonic() - started < 2

    def test_reaching_the_end_does_not_abort(self):
        decoder = FakeDecoder(frames=3)
        with ThreadedFrameReader(decoder, _copy, queue_size=2) as frames:
            list(frames)
        assert not decoder.aborted.is_set()


class FakeEncoder:
    """Records a copy of every frame; ``eof_after`` / ``fail_at`` / ``delay`` shape it."""

    def __init__(self, eof_after=None, fail_at=None, delay=0.0):
        self.frames = []
        self._eof_after = eof_after
        self._fail_at = fail_at
        self._delay = delay

    def write_frame(self, frame):
        n = len(self.frames)
        if n == self._fail_at:
            raise RuntimeError("[hevc_nvenc] InitializeEncoder failed: out of memory (10)")
        if self._eof_after is not None and n >= self._eof_after:
            return False
        if self._delay:
            time.sleep(self._delay)
        self.frames.append(frame.copy())
        return True


SHAPE = (4,)


class TestThreadedFrameWriter:
    def test_writes_all_frames_in_order(self):
        encoder = FakeEncoder()
        with ThreadedFrameWriter(encoder, SHAPE, pool_size=3, queue_size=2) as writer:
            for n in range(10):
                buf = writer.acquire()
                buf.fill(n)
                assert writer.submit(buf) is True
        assert [int(f[0]) for f in encoder.frames] == list(range(10))

    def test_clean_exit_waits_for_a_slow_encoder(self):
        encoder = FakeEncoder(delay=0.05)
        with ThreadedFrameWriter(encoder, SHAPE, pool_size=4, queue_size=2) as writer:
            for n in range(4):
                buf = writer.acquire()
                buf.fill(n)
                writer.submit(buf)
        assert len(encoder.frames) == 4

    def test_submit_returns_false_after_encoder_eof(self):
        encoder = FakeEncoder(eof_after=2)
        results = []
        with ThreadedFrameWriter(encoder, SHAPE, pool_size=2, queue_size=1) as writer:
            for n in range(20):
                buf = writer.acquire()
                buf.fill(n)
                ok = writer.submit(buf)
                results.append(ok)
                if not ok:
                    break
        assert results[-1] is False
        assert len(encoder.frames) == 2

    def test_encoder_error_is_raised_with_its_message(self):
        encoder = FakeEncoder(fail_at=0)
        with pytest.raises(RuntimeError, match="out of memory"):
            with ThreadedFrameWriter(encoder, SHAPE, pool_size=2, queue_size=1) as writer:
                for n in range(5):
                    buf = writer.acquire()
                    writer.submit(buf)

    def test_exception_in_the_body_drops_queued_frames(self):
        encoder = FakeEncoder(delay=0.1)
        with pytest.raises(KeyError):
            with ThreadedFrameWriter(encoder, SHAPE, pool_size=4, queue_size=3) as writer:
                for n in range(4):
                    buf = writer.acquire()
                    buf.fill(n)
                    writer.submit(buf)
                raise KeyError("cancelled")
        assert len(encoder.frames) < 4

    def test_release_returns_an_unused_buffer(self):
        encoder = FakeEncoder()
        with ThreadedFrameWriter(encoder, SHAPE, pool_size=1, queue_size=1) as writer:
            buf = writer.acquire()
            writer.release(buf)
            again = writer.acquire()  # would block forever if release() lost it
            writer.release(again)
        assert encoder.frames == []

    def test_buffers_have_the_requested_shape(self):
        with ThreadedFrameWriter(FakeEncoder(), (3, 5, 3), pool_size=1) as writer:
            buf = writer.acquire()
            assert buf.shape == (3, 5, 3)
            assert buf.dtype == np.uint8
            writer.release(buf)
