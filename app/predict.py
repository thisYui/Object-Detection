from pathlib import Path
from time import perf_counter

try:
    from .model_loader import get_faster_rcnn_model
except ImportError:
    from model_loader import get_faster_rcnn_model

from src.inference.predict_faster_rcnn import (
    draw_detections,
    filter_predictions,
    load_image,
    run_faster_rcnn_prediction,
)


def detect_image(
    image_path,
    output_path,
    weights_path,
    threshold=0.5,
    device=None,
):
    """
    Run cached Faster R-CNN inference for the web app.
    """
    started_at = perf_counter()

    model, id2label, resolved_device = get_faster_rcnn_model(
        weights_path=weights_path,
        device=device,
    )

    image = load_image(str(image_path))
    prediction = run_faster_rcnn_prediction(model, image, resolved_device)
    detections = filter_predictions(prediction, id2label, threshold)

    rendered = draw_detections(image, detections, color="#566b4d")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    elapsed = perf_counter() - started_at

    return {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "detections": detections,
        "threshold": threshold,
        "device": str(resolved_device),
        "inference_seconds": elapsed,
    }
