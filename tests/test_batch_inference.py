import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.checks import check_imgsz

from batch_inference import (
    BatchDetector,
    InferenceMode,
    extract_detections,
    inference_size,
    model_stride,
    resize_for_inference,
    resolve_inference_mode,
)


def _result(boxes_data):
    """YOLO result double. Each tuple: (x1, y1, x2, y2, class_id, confidence)."""
    result = MagicMock()
    if not boxes_data:
        result.boxes = None
        return result
    boxes = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([b[:4] for b in boxes_data], np.float32)
    boxes.cls.cpu.return_value.numpy.return_value = np.array([b[4] for b in boxes_data], np.float32)
    boxes.conf.cpu.return_value.numpy.return_value = np.array([b[5] for b in boxes_data], np.float32)
    boxes.__len__ = lambda self: len(boxes_data)
    result.boxes = boxes
    return result


class TestInferenceSize:
    @pytest.mark.parametrize("width,height,imgsz,expected", [
        (2560, 1920, 1024, (1024, 768)),
        (1920, 1080, 1024, (1024, 576)),
        (1080, 1920, 1024, (576, 1024)),
        (320, 240, 1024, (1024, 768)),  # LetterBox scales small frames up
        (2560, 1920, 1000, (1024, 768)),  # imgsz is rounded up to a multiple of 32
    ])
    def test_matches_letterbox_arithmetic(self, width, height, imgsz, expected):
        assert inference_size(width, height, imgsz, 32) == expected

    @pytest.mark.parametrize("width,height,imgsz", [
        (2560, 1920, 1024),
        (1920, 1080, 1024),
        (1080, 1920, 1024),
        (1001, 757, 640),
        (320, 240, 1024),
        (2688, 1520, 1000),
        (640, 480, 640),
    ])
    def test_pre_resized_frame_letterboxes_to_the_same_tensor(self, width, height, imgsz):
        """Byte-for-byte guard: if Ultralytics changes how LetterBox resizes,
        the pre-resize on the reader thread would silently change detections."""
        rng = np.random.default_rng(width * height)
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        size = inference_size(width, height, imgsz, 32)
        letterbox = LetterBox(check_imgsz(imgsz, stride=32, min_dim=2), auto=True, stride=32)
        np.testing.assert_array_equal(
            letterbox(image=resize_for_inference(frame, size)),
            letterbox(image=frame),
        )

    def test_same_size_returns_a_copy(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        out = resize_for_inference(frame, (640, 480))
        assert out is not frame
        np.testing.assert_array_equal(out, frame)


class TestModelStride:
    def test_reads_the_largest_stride(self):
        model = MagicMock()
        model.model.stride = torch.tensor([8.0, 16.0, 32.0])
        assert model_stride(model) == 32

    def test_keeps_a_larger_stride(self):
        model = MagicMock()
        model.model.stride = torch.tensor([8.0, 16.0, 32.0, 64.0])
        assert model_stride(model) == 64

    def test_falls_back_to_32(self):
        assert model_stride(MagicMock()) == 32
        assert model_stride(object()) == 32


class TestResolveInferenceMode:
    def test_auto_on_nvidia_is_fp16_batch_8(self):
        with patch("batch_inference.torch.version.hip", None):
            assert resolve_inference_mode("auto", "auto", "cuda:0") == InferenceMode(True, 8)

    def test_auto_on_rocm_is_fp32_batch_1(self):
        with patch("batch_inference.torch.version.hip", "7.2.0"):
            assert resolve_inference_mode("auto", "auto", "cuda:0") == InferenceMode(False, 1)

    def test_auto_on_cpu_is_fp32_batch_1(self):
        assert resolve_inference_mode("auto", "auto", "cpu") == InferenceMode(False, 1)

    def test_explicit_values_override_auto(self):
        with patch("batch_inference.torch.version.hip", "7.2.0"):
            assert resolve_inference_mode("true", "8", "cuda:0") == InferenceMode(True, 8)
        with patch("batch_inference.torch.version.hip", None):
            assert resolve_inference_mode("false", "2", "cuda") == InferenceMode(False, 2)

    def test_fp16_on_cpu_is_refused_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="batch_inference"):
            mode = resolve_inference_mode("true", "4", "cpu")
        assert mode == InferenceMode(False, 4)
        assert "VIDEO_FP16=true ignored" in caplog.text


class TestExtractDetections:
    def test_single_detection(self):
        dets = extract_detections(_result([(10, 20, 100, 200, 0, 0.9)]), 5, {0: "person"})
        assert len(dets) == 1
        assert dets[0].frame_num == 5
        assert dets[0].class_name == "person"
        assert dets[0].confidence == pytest.approx(0.9)
        assert dets[0].bbox == (10, 20, 100, 200)

    def test_scale_maps_boxes_to_the_full_frame(self):
        dets = extract_detections(_result([(10, 20, 100, 200, 0, 0.9)]), 0, {0: "person"}, (2.5, 2.5))
        assert dets[0].bbox == (25, 50, 250, 500)

    def test_unknown_class_id(self):
        dets = extract_detections(_result([(1, 2, 3, 4, 7, 0.5)]), 0, {0: "person"})
        assert dets[0].class_name == "class_7"

    def test_empty_boxes(self):
        assert extract_detections(_result([]), 0, {0: "person"}) == []


def _model(boxes_data=((10, 20, 100, 200, 0, 0.9),), oom_above=None):
    """predict() double answering one result per frame; ``oom_above`` makes
    batches larger than that raise CUDA out of memory."""
    model = MagicMock()

    def predict(source, **kwargs):
        if oom_above is not None and len(source) > oom_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return [_result(list(boxes_data)) for _ in source]

    model.predict.side_effect = predict
    return model


def _detector(model, batch_size=3, fp16=False, scale=(1.0, 1.0)):
    return BatchDetector(
        model, {0: "person"}, conf=0.24, imgsz=640, max_det=100,
        fp16=fp16, batch_size=batch_size, scale=scale,
    )


def _frames(n):
    return [np.full((4, 4, 3), i, np.uint8) for i in range(n)]


class TestBatchDetector:
    def test_batches_frames_and_flushes_the_rest(self):
        model = _model()
        detector = _detector(model, batch_size=3)
        for n, frame in enumerate(_frames(7)):
            detector.add(n * 2, frame)
        detector.flush()

        assert [len(c.kwargs["source"]) for c in model.predict.call_args_list] == [3, 3, 1]
        assert sorted(detector.detections) == [0, 2, 4, 6, 8, 10, 12]
        assert detector.detected_frames == 7

    def test_frames_without_boxes_are_counted_but_not_stored(self):
        detector = _detector(_model(boxes_data=()), batch_size=2)
        for n, frame in enumerate(_frames(3)):
            detector.add(n, frame)
        detector.flush()
        assert detector.detections == {}
        assert detector.detected_frames == 3

    def test_predict_arguments(self):
        model = _model()
        detector = _detector(model, batch_size=1)
        detector.add(0, _frames(1)[0])
        kwargs = model.predict.call_args.kwargs
        assert kwargs["conf"] == 0.24
        assert kwargs["imgsz"] == 640
        assert kwargs["max_det"] == 100
        assert kwargs["verbose"] is False
        assert "quantize" not in kwargs

    def test_fp16_passes_quantize_16(self):
        model = _model()
        detector = _detector(model, batch_size=1, fp16=True)
        detector.add(0, _frames(1)[0])
        assert model.predict.call_args.kwargs["quantize"] == 16

    def test_scale_applies_to_stored_boxes(self):
        detector = _detector(_model(), batch_size=1, scale=(2.0, 2.0))
        detector.add(0, _frames(1)[0])
        assert detector.detections[0][0].bbox == (20, 40, 200, 400)

    def test_out_of_memory_halves_the_batch_and_loses_nothing(self, caplog):
        model = _model(oom_above=2)
        detector = _detector(model, batch_size=8)
        with caplog.at_level(logging.WARNING, logger="batch_inference"):
            for n, frame in enumerate(_frames(8)):
                detector.add(n, frame)
            detector.flush()

        assert detector.batch_size == 2
        assert sorted(detector.detections) == list(range(8))
        assert detector.detected_frames == 8
        assert "out of memory at batch 8" in caplog.text
        assert "out of memory at batch 4" in caplog.text

    def test_out_of_memory_at_batch_1_is_raised(self):
        detector = _detector(_model(oom_above=0), batch_size=1)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            detector.add(0, _frames(1)[0])

    def test_result_count_mismatch_is_an_error(self):
        model = MagicMock()
        model.predict.return_value = [_result([(1, 2, 3, 4, 0, 0.9)])]
        detector = _detector(model, batch_size=2)
        with pytest.raises(RuntimeError, match="1 results for 2 frames"):
            for n, frame in enumerate(_frames(2)):
                detector.add(n, frame)
