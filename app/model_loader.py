from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict_faster_rcnn import (  # noqa: E402
    DEFAULT_CLASS_NAMES,
    load_faster_rcnn_model,
)


_MODEL_CACHE = {}


def resolve_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_faster_rcnn_model(weights_path, device=None, class_names=None):
    """
    Load Faster R-CNN once and reuse it across requests.
    """
    resolved_device = resolve_device(device)
    names = class_names or DEFAULT_CLASS_NAMES
    cache_key = (str(Path(weights_path).resolve()), str(resolved_device), tuple(names))

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_faster_rcnn_model(
            weights_path=str(weights_path),
            device=resolved_device,
            class_names=names,
        )

    return _MODEL_CACHE[cache_key]
