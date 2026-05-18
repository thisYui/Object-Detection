"""
evaluate_faster_rcnn.py

Evaluate a torchvision Faster R-CNN object detection model on a COCO-format dataset.

Expected project usage:
    python src/evaluation/evaluate_faster_rcnn.py \
        --weights models/faster_rcnn/best.pth \
        --data configs/dataset.yaml \
        --annotations data/processed/test/annotations.json \
        --split test \
        --output-dir experiments/faster_rcnn/evaluation

Requirements:
    pip install torch torchvision pyyaml pandas pycocotools tqdm

On Windows, if pycocotools fails:
    pip install pycocotools-windows

The script saves:
    - metrics_summary.json
    - metrics_summary.csv
    - per_class_metrics.csv
    - faster_rcnn_predictions_coco.json
    - optional benchmark_fps.json / benchmark_fps.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise ImportError(
        "[ERROR] Missing pycocotools. Install it with: pip install pycocotools\n"
        "On Windows, if this fails, try: pip install pycocotools-windows"
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────────────────────
# 1. YAML / Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Dataset YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"[ERROR] Invalid YAML content: {path}")
    return data


def normalize_class_names(names: Any) -> List[str]:
    """Accept YOLO names as list or dict and return an ordered list."""
    if names is None:
        return []

    if isinstance(names, list):
        return [str(x) for x in names]

    if isinstance(names, dict):
        def key_to_int(k: Any) -> int:
            try:
                return int(k)
            except Exception:
                return 10**9

        return [str(names[k]) for k in sorted(names.keys(), key=key_to_int)]

    raise ValueError("[ERROR] `names` in dataset YAML must be a list or dictionary.")


def resolve_split_images_dir(data_yaml_path: str | Path, data_cfg: Dict[str, Any], split: str) -> Optional[Path]:
    """
    Resolve image directory for train/val/test from YOLO-style data.yaml.

    Supports:
        path: data/processed
        train: train/images
        val: val/images
        test: test/images
    """
    split_value = data_cfg.get(split)
    if not split_value:
        return None

    split_path = Path(str(split_value))

    if split_path.is_absolute():
        return split_path

    root = data_cfg.get("path")
    if root:
        root_path = Path(str(root))
        if not root_path.is_absolute():
            candidate = (Path.cwd() / root_path / split_path).resolve()
            if candidate.exists():
                return candidate
            return (Path(data_yaml_path).resolve().parent / root_path / split_path).resolve()
        return (root_path / split_path).resolve()

    return (Path(data_yaml_path).resolve().parent / split_path).resolve()


def infer_coco_annotation_path(images_dir: Path, split: str) -> Optional[Path]:
    """
    Try common COCO annotation locations when --annotations is not provided.
    This is only a convenience fallback. Passing --annotations is safer.
    """
    candidates = [
        images_dir.parent / "annotations.json",
        images_dir.parent / f"{split}_annotations.json",
        images_dir.parent / f"instances_{split}.json",
        images_dir.parent / f"instances_{split}2017.json",
        images_dir.parents[1] / "annotations" / f"instances_{split}.json" if len(images_dir.parents) > 1 else None,
        images_dir.parents[1] / "annotations" / f"instances_{split}2017.json" if len(images_dir.parents) > 1 else None,
        images_dir.parents[1] / f"{split}_annotations.json" if len(images_dir.parents) > 1 else None,
    ]

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()

    return None


def iter_images_from_path(path: Path) -> Iterable[Path]:
    """Yield image paths from a directory."""
    if not path.exists():
        return
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class COCODetectionEvalDataset(CocoDetection):
    """COCO dataset wrapper for Faster R-CNN evaluation."""

    def __init__(self, image_dir: str | Path, annotation_file: str | Path):
        super().__init__(str(image_dir), str(annotation_file))

    def __getitem__(self, index: int):
        image, annotations = super().__getitem__(index)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_id = self.ids[index]
        width, height = image.size
        image_tensor = F.to_tensor(image)

        target = {
            "image_id": int(image_id),
            "width": int(width),
            "height": int(height),
            "annotations": annotations,
        }
        return image_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_faster_rcnn_model(num_object_classes: int) -> torch.nn.Module:
    """
    Build Faster R-CNN ResNet50-FPN.

    num_object_classes excludes the background class.
    torchvision Faster R-CNN needs: num_classes = object classes + 1 background.
    """
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_object_classes + 1)
    return model


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = value
    return cleaned


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Support common checkpoint formats."""
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise ValueError("[ERROR] Unsupported checkpoint format.")

    for key in ["model_state_dict", "state_dict", "model"]:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    Infer total Faster R-CNN classes from predictor shape.
    Return total classes including background.
    """
    possible_keys = [
        "roi_heads.box_predictor.cls_score.weight",
        "module.roi_heads.box_predictor.cls_score.weight",
    ]
    for key in possible_keys:
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[0])
    return None


def load_checkpoint(weights_path: str | Path, device: torch.device, num_object_classes: int) -> torch.nn.Module:
    weights_path = Path(weights_path)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model.to(device)
        return model

    state_dict = clean_state_dict_keys(extract_state_dict(checkpoint))

    total_classes = infer_num_classes_from_state_dict(state_dict)
    if total_classes is not None:
        inferred_object_classes = max(total_classes - 1, 1)
        if inferred_object_classes != num_object_classes:
            print(
                "[Warn] Class count from YAML does not match checkpoint. "
                f"YAML object classes={num_object_classes}, checkpoint object classes={inferred_object_classes}. "
                "Using checkpoint class count."
            )
            num_object_classes = inferred_object_classes

    model = build_faster_rcnn_model(num_object_classes)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print("[Warn] Missing keys while loading checkpoint:")
        for key in missing_keys[:20]:
            print(f"  - {key}")
        if len(missing_keys) > 20:
            print(f"  ... and {len(missing_keys) - 20} more")

    if unexpected_keys:
        print("[Warn] Unexpected keys while loading checkpoint:")
        for key in unexpected_keys[:20]:
            print(f"  - {key}")
        if len(unexpected_keys) > 20:
            print(f"  ... and {len(unexpected_keys) - 20} more")

    model.to(device)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4. Prediction conversion / metrics
# ──────────────────────────────────────────────────────────────────────────────

def xyxy_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def clip_box_xyxy(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))
    return [x1, y1, x2, y2]


def convert_predictions_to_coco_format(
    outputs: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, Any]],
    conf: float,
    category_id_offset: int,
) -> List[Dict[str, Any]]:
    """
    Convert torchvision Faster R-CNN outputs to COCO detection JSON format.

    torchvision labels usually are:
        0 = background
        1 = first object class
        2 = second object class

    If your COCO annotations use category_id 0..N-1, use --category-id-offset -1.
    If your COCO annotations use category_id 1..N, use --category-id-offset 0.
    """
    rows: List[Dict[str, Any]] = []

    for output, target in zip(outputs, targets):
        image_id = int(target["image_id"])
        width = int(target["width"])
        height = int(target["height"])

        boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu()
        scores = output.get("scores", torch.empty((0,))).detach().cpu()
        labels = output.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()

        for box, score, label in zip(boxes, scores, labels):
            score_value = float(score.item())
            if score_value < conf:
                continue

            clipped = clip_box_xyxy(box.tolist(), width, height)
            x, y, w, h = xyxy_to_xywh(clipped)
            if w <= 0 or h <= 0:
                continue

            category_id = int(label.item()) + int(category_id_offset)

            rows.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "score": score_value,
                }
            )

    return rows


def summarize_coco_metrics(coco_eval: COCOeval) -> Dict[str, Any]:
    stats = coco_eval.stats

    summary = {
        "precision_mean": safe_float(stats[0]),       # COCO AP@[.50:.95], closest precision summary
        "recall_mean": safe_float(stats[8]),          # AR@100
        "mAP50": safe_float(stats[1]),
        "mAP50_95": safe_float(stats[0]),
        "mAP75": safe_float(stats[2]),
        "mAP_small": safe_float(stats[3]),
        "mAP_medium": safe_float(stats[4]),
        "mAP_large": safe_float(stats[5]),
        "AR_1": safe_float(stats[6]),
        "AR_10": safe_float(stats[7]),
        "AR_100": safe_float(stats[8]),
        "AR_small": safe_float(stats[9]),
        "AR_medium": safe_float(stats[10]),
        "AR_large": safe_float(stats[11]),
    }

    p = summary["precision_mean"]
    r = summary["recall_mean"]
    summary["f1_mean"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return summary


def extract_per_class_metrics(coco_eval: COCOeval, coco_gt: COCO, class_names: List[str]) -> pd.DataFrame:
    """
    Extract per-class AP from COCOeval.

    COCO precision shape:
        [IoU thresholds, recall thresholds, categories, area ranges, max detections]
    """
    precisions = coco_eval.eval.get("precision")
    recalls = coco_eval.eval.get("recall")
    if precisions is None:
        return pd.DataFrame()

    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)

    rows: List[Dict[str, Any]] = []
    for idx, cat in enumerate(cats):
        cat_id = int(cat["id"])
        class_name = cat.get("name") or (class_names[idx] if idx < len(class_names) else f"class_{cat_id}")

        # all area index 0, maxDets=100 index 2
        precision_all = precisions[:, :, idx, 0, 2]
        valid_precision_all = precision_all[precision_all > -1]
        ap50_95 = float(valid_precision_all.mean()) if valid_precision_all.size else None

        precision_50 = precisions[0, :, idx, 0, 2]
        valid_precision_50 = precision_50[precision_50 > -1]
        ap50 = float(valid_precision_50.mean()) if valid_precision_50.size else None

        recall_value = None
        if recalls is not None:
            # recall shape: [IoU thresholds, categories, area ranges, max detections]
            recall_all = recalls[:, idx, 0, 2]
            valid_recall = recall_all[recall_all > -1]
            recall_value = float(valid_recall.mean()) if valid_recall.size else None

        f1 = None
        if ap50_95 is not None and recall_value is not None and (ap50_95 + recall_value) > 0:
            f1 = 2 * ap50_95 * recall_value / (ap50_95 + recall_value)

        rows.append(
            {
                "class_id": idx,
                "category_id": cat_id,
                "class_name": str(class_name),
                "precision": ap50_95,
                "recall": recall_value,
                "f1_score": f1,
                "mAP50": ap50,
                "mAP50_95": ap50_95,
                # Backward-compatible aliases.
                "AP50": ap50,
                "AP50_95": ap50_95,
            }
        )

    return pd.DataFrame(rows)


def get_file_size_mb(path: str | Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def load_image_as_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return F.to_tensor(image)


def benchmark_fps(
    model: torch.nn.Module,
    image_paths: List[Path],
    device: torch.device,
    max_images: int,
    warmup: int,
) -> Dict[str, Any]:
    selected = image_paths[: max_images if max_images > 0 else len(image_paths)]
    if not selected:
        return {
            "benchmark_images": 0,
            "avg_seconds_per_image": None,
            "fps_end_to_end": None,
            "note": "No images found for benchmarking.",
        }

    model.eval()

    warmup_items = selected[: min(warmup, len(selected))]
    with torch.no_grad():
        for img_path in warmup_items:
            image = load_image_as_tensor(img_path).to(device)
            _ = model([image])

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for img_path in selected:
            image = load_image_as_tensor(img_path).to(device)
            _ = model([image])

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg = elapsed / len(selected)
    fps = len(selected) / elapsed if elapsed > 0 else None

    return {
        "benchmark_images": len(selected),
        "total_seconds": elapsed,
        "avg_seconds_per_image": avg,
        "avg_ms_per_image": avg * 1000,
        "fps": fps,
        "fps_end_to_end": fps,
        "note": "End-to-end timing includes image loading and model forward pass.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN object detection model")

    parser.add_argument("--weights", type=str, default="models/faster_rcnn/best.pth", help="Path to Faster R-CNN .pth checkpoint")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Path to dataset YAML for split paths and class names")
    parser.add_argument("--annotations", type=str, default='data/processed/annotations/test.json', help="Path to COCO annotation JSON for the selected split")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--output-dir", type=str, default="experiments/faster_rcnn/evaluation", help="Directory to save evaluation outputs")

    parser.add_argument("--batch", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cpu, cuda, cuda:0. Leave empty for auto")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for exported COCO predictions")
    parser.add_argument("--category-id-offset", type=int, default=-1, help="Use -1 for 0-based COCO category_id, 0 for 1-based COCO category_id")

    parser.add_argument("--benchmark", action="store_true", help="Run a simple FPS benchmark after validation")
    parser.add_argument("--benchmark-count", type=int, default=100, help="Max number of images for FPS benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup images before FPS benchmark")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    data_yaml_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"[ERROR] Weight file not found: {weights_path}")

    data_cfg = load_yaml(data_yaml_path)
    class_names = normalize_class_names(data_cfg.get("names"))
    if not class_names:
        raise ValueError("[ERROR] No class names found in dataset YAML under `names`.")

    images_dir = resolve_split_images_dir(data_yaml_path, data_cfg, args.split)
    if images_dir is None or not images_dir.exists():
        raise FileNotFoundError(
            f"[ERROR] Could not resolve image directory for split `{args.split}` from: {data_yaml_path}"
        )

    annotation_path = Path(args.annotations).resolve() if args.annotations else infer_coco_annotation_path(images_dir, args.split)
    if annotation_path is None or not annotation_path.exists():
        raise FileNotFoundError(
            "[ERROR] COCO annotation JSON not found. Pass it explicitly, for example:\n"
            f"  --annotations data/processed/{args.split}/annotations.json\n"
            "or put it in a common location such as data/processed/test/annotations.json."
        )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("[Info] Loading Faster R-CNN model")
    print(f"       Weights     : {weights_path}")
    print(f"       Data        : {data_yaml_path}")
    print(f"       Split       : {args.split}")
    print(f"       Images      : {images_dir}")
    print(f"       Annotations : {annotation_path}")
    print(f"       Device      : {device}")
    print(f"       Classes     : {len(class_names)}")

    model = load_checkpoint(weights_path, device, len(class_names))
    model.eval()

    dataset = COCODetectionEvalDataset(images_dir, annotation_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    print("\n[Info] Running evaluation inference...")
    all_predictions: List[Dict[str, Any]] = []
    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, total=len(dataloader)):
            images = [img.to(device) for img in images]

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            total_inference_time += elapsed
            total_images += len(images)

            batch_predictions = convert_predictions_to_coco_format(
                outputs=outputs,
                targets=targets,
                conf=args.conf,
                category_id_offset=args.category_id_offset,
            )
            all_predictions.extend(batch_predictions)

    prediction_json_path = output_dir / "faster_rcnn_predictions_coco.json"
    save_json(prediction_json_path, all_predictions)

    avg_seconds = total_inference_time / max(total_images, 1)
    fps = total_images / total_inference_time if total_inference_time > 0 else 0.0

    speed_summary = {
        "total_images": int(total_images),
        "total_inference_time_seconds": float(total_inference_time),
        "avg_seconds_per_image": float(avg_seconds),
        "avg_ms_per_image": float(avg_seconds * 1000),
        "fps": float(fps),
        # Backward-compatible alias.
        "fps_from_forward_time": float(fps),
        "model_size_mb": float(get_file_size_mb(weights_path)),
    }

    if not all_predictions:
        print("\n[Warn] No predictions were generated. COCO evaluation is skipped.")
        summary = {
            "precision_mean": 0.0,
            "recall_mean": 0.0,
            "f1_mean": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            **speed_summary,
            "weights": str(weights_path),
            "data_yaml": str(data_yaml_path),
            "annotations": str(annotation_path),
            "split": args.split,
            "batch": args.batch,
            "device": args.device or "auto",
            "num_classes": len(class_names),
            "class_names": class_names,
            "conf": args.conf,
            "category_id_offset": args.category_id_offset,
        }
        save_json(output_dir / "metrics_summary.json", summary)
        pd.DataFrame([summary]).drop(columns=["class_names"], errors="ignore").to_csv(output_dir / "metrics_summary.csv", index=False)
        print("\n[Done] Faster R-CNN evaluation complete.")
        return

    print("\n[Info] Running COCO evaluation...")
    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(str(prediction_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    summary = summarize_coco_metrics(coco_eval)
    summary.update(
        {
            **speed_summary,
            "weights": str(weights_path),
            "data_yaml": str(data_yaml_path),
            "annotations": str(annotation_path),
            "split": args.split,
            "batch": args.batch,
            "device": args.device or "auto",
            "num_classes": len(class_names),
            "class_names": class_names,
            "conf": args.conf,
            "category_id_offset": args.category_id_offset,
        }
    )

    per_class_df = extract_per_class_metrics(coco_eval, coco_gt, class_names)

    summary_json_path = output_dir / "metrics_summary.json"
    summary_csv_path = output_dir / "metrics_summary.csv"
    per_class_csv_path = output_dir / "per_class_metrics.csv"

    save_json(summary_json_path, summary)
    pd.DataFrame([summary]).drop(columns=["class_names"], errors="ignore").to_csv(summary_csv_path, index=False)
    if not per_class_df.empty:
        per_class_df.to_csv(per_class_csv_path, index=False)

    print("\n[Result] Summary metrics")
    print(f"  Precision mean : {summary['precision_mean']:.4f}")
    print(f"  Recall mean    : {summary['recall_mean']:.4f}")
    print(f"  F1 mean        : {summary['f1_mean']:.4f}")
    print(f"  mAP@0.5        : {summary['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95   : {summary['mAP50_95']:.4f}")
    print(f"  FPS estimate   : {summary['fps']:.2f}")

    print("\n[Saved]")
    print(f"  {summary_json_path}")
    print(f"  {summary_csv_path}")
    if not per_class_df.empty:
        print(f"  {per_class_csv_path}")
    print(f"  {prediction_json_path}")

    if args.benchmark:
        image_paths = list(iter_images_from_path(images_dir))
        bench = benchmark_fps(
            model=model,
            image_paths=image_paths,
            device=device,
            max_images=args.benchmark_count,
            warmup=args.warmup,
        )
        bench_json_path = output_dir / "benchmark_fps.json"
        bench_csv_path = output_dir / "benchmark_fps.csv"
        save_json(bench_json_path, bench)
        pd.DataFrame([bench]).to_csv(bench_csv_path, index=False)

        print("\n[Benchmark]")
        if bench.get("fps_end_to_end") is not None:
            print(f"  Images         : {bench['benchmark_images']}")
            print(f"  Avg ms/image   : {bench['avg_ms_per_image']:.2f}")
            print(f"  FPS end-to-end : {bench['fps_end_to_end']:.2f}")
        else:
            print(f"  {bench.get('note')}")
        print(f"  {bench_json_path}")
        print(f"  {bench_csv_path}")

    print("\n[Done] Faster R-CNN evaluation complete.")


if __name__ == "__main__":
    main()
