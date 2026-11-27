"""What run_inference() asks of predict()."""
from unittest.mock import MagicMock

import numpy as np

from inference_utils import run_inference
from model_manager import ModelEntry


async def test_predict_keeps_the_nms_iou_of_0_7():
    """Ultralytics 8.4.208 lowered predict's default NMS IoU to 0.5, which
    changes which overlapping boxes survive; the API keeps 0.7."""
    model = MagicMock()
    model.predict.return_value = []
    entry = ModelEntry(model=model, visualizer=MagicMock(), model_name="m.pt", device="cpu")

    await run_inference(entry, np.zeros((8, 8, 3), np.uint8), conf=0.5, imgsz=64, max_det=10)

    assert model.predict.call_args.kwargs["iou"] == 0.7
