import os
import json
import csv
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

try:
    from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
except ImportError:
    raise ImportError(
        "[ERROR] Missing required libraries. Install them with:\n"
        "pip install transformers timm safetensors"
    )

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    raise ImportError(
        "[ERROR] Missing pycocotools. Install it with:\n"
        "pip install pycocotools\n\n"
        "On Windows, if this fails, try:\n"
        "pip install pycocotools-windows"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Small IO Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, data: Any) -> None:
    """Save JSON with a unified signature: save_json(path, data)."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_dict_to_csv(data: Dict[str, Any], path: str | Path) -> None:
    """Save a metrics dictionary as one CSV row, matching YOLOv8/Faster R-CNN."""
    pd.DataFrame([data]).drop(columns=["class_names"], errors="ignore").to_csv(path, index=False)


def save_rows_to_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    """Save per-class rows with pandas, matching YOLOv8/Faster R-CNN."""
    pd.DataFrame(rows).to_csv(path, index=False)


def get_model_size_mb(weights_path: str) -> float:
    path = Path(weights_path)

    if path.is_file():
        return path.stat().st_size / (1024 * 1024)

    if path.is_dir():
        total_size = 0

        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size / (1024 * 1024)

    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. Dataset Config Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_class_names_from_yaml(data_yaml: Optional[str]) -> List[str]:
    if data_yaml is None or not os.path.exists(data_yaml):
        return DEFAULT_CLASS_NAMES

    data = load_yaml(data_yaml)
    names = data.get("names", None)

    if names is None:
        return DEFAULT_CLASS_NAMES

    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]

    if isinstance(names, list):
        return names

    return DEFAULT_CLASS_NAMES


def resolve_images_dir(data_yaml: str, split: str) -> str:
    """
    Resolve image directory from YOLO-style dataset.yaml.

    Example:
        path: data/processed
        train: train/images
        val: val/images
        test: test/images
    """

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"[ERROR] Dataset YAML not found: {data_yaml}")

    data = load_yaml(data_yaml)

    root = data.get("path", "")
    split_value = data.get(split, None)

    if split_value is None:
        raise ValueError(
            f"[ERROR] Split '{split}' not found in {data_yaml}. "
            f"Expected key: train, val, or test."
        )

    split_path = Path(split_value)

    if split_path.is_absolute():
        return str(split_path)

    # Prefer path from YAML relative to project root/current working directory.
    if root:
        return str(Path(root) / split_path)

    # If no root is declared, resolve relative to YAML file location.
    return str(Path(data_yaml).parent / split_path)


def infer_annotations_path(data_yaml: str, split: str) -> str:
    """
    Infer COCO annotation file from your current project layout:

        data/processed/annotations/test.json
        data/processed/annotations/val.json
        data/processed/annotations/train.json
    """

    data = load_yaml(data_yaml)
    root = data.get("path", "")

    if not root:
        root = str(Path(data_yaml).parent)

    return str(Path(root) / "annotations" / f"{split}.json")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class COCODetectionEvalDataset(CocoDetection):
    def __init__(self, image_dir: str, annotation_file: str):
        super().__init__(image_dir, annotation_file)

    def __getitem__(self, index: int):
        image, annotations = super().__getitem__(index)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_id = self.ids[index]
        width, height = image.size

        target = {
            "image_id": image_id,
            "width": width,
            "height": height,
            "annotations": annotations,
        }

        return image, target


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    return images, targets


# ──────────────────────────────────────────────────────────────────────────────
# 5. COCO Conversion
# ──────────────────────────────────────────────────────────────────────────────

def xyxy_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box

    return [
        float(x1),
        float(y1),
        float(x2 - x1),
        float(y2 - y1),
    ]


def clip_box_xyxy(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box

    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))

    return [x1, y1, x2, y2]


def convert_outputs_to_coco_predictions(
    processed_outputs: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, Any]],
    score_threshold: float,
    category_id_offset: int,
) -> List[Dict[str, Any]]:
    """
    Convert Hugging Face post-processed Deformable DETR output to COCO detection format.

    In your training file, id2label starts at 0:
        0: person, 1: bicycle, ...

    So for your current dataset, category_id_offset should normally be 0.

    If your COCO annotations use category_id from 1, use:
        --category-id-offset 1
    """

    coco_predictions = []

    for output, target in zip(processed_outputs, targets):
        image_id = int(target["image_id"])
        width = int(target["width"])
        height = int(target["height"])

        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            score = float(score)

            if score < score_threshold:
                continue

            box = clip_box_xyxy(box.tolist(), width, height)
            x, y, w, h = xyxy_to_xywh(box)

            if w <= 0 or h <= 0:
                continue

            category_id = int(label) + int(category_id_offset)

            coco_predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "score": score,
            })

    return coco_predictions


# ──────────────────────────────────────────────────────────────────────────────
# 6. COCO Metrics
# ──────────────────────────────────────────────────────────────────────────────

def summarize_coco_stats(coco_eval: COCOeval) -> Dict[str, float]:
    stats = coco_eval.stats

    precision_mean = float(stats[0])  # COCO AP@[.50:.95], used as the closest precision summary.
    recall_mean = float(stats[8])     # AR@100, closest summary recall from COCOeval.
    f1_mean = (2 * precision_mean * recall_mean / (precision_mean + recall_mean)) if (precision_mean + recall_mean) > 0 else 0.0

    return {
        "precision_mean": precision_mean,
        "recall_mean": recall_mean,
        "f1_mean": f1_mean,
        "mAP50": float(stats[1]),
        "mAP50_95": float(stats[0]),
        "mAP75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }


def extract_per_class_metrics(
    coco_eval: COCOeval,
    coco_gt: COCO,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Save per-class AP in a file named per_class_metrics.csv,
    matching the output style of evaluate_yolov8.py.

    COCOeval precision shape:
        [T, R, K, A, M]
    """

    precisions = coco_eval.eval["precision"]
    recalls = coco_eval.eval["recall"]

    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)

    rows = []

    for idx, cat in enumerate(cats):
        category_id = int(cat["id"])
        class_name = cat.get("name", None)

        if class_name is None:
            class_index = category_id

            if 0 <= class_index < len(class_names):
                class_name = class_names[class_index]
            else:
                class_name = f"class_{category_id}"

        # Area = all, maxDets = 100
        precision = precisions[:, :, idx, 0, 2]
        precision_valid = precision[precision > -1]

        precision_50 = precisions[0, :, idx, 0, 2]
        precision_50_valid = precision_50[precision_50 > -1]

        recall = recalls[:, idx, 0, 2]
        recall_valid = recall[recall > -1]

        ap_50_95 = float(np.mean(precision_valid)) if precision_valid.size else float("nan")
        ap_50 = float(np.mean(precision_50_valid)) if precision_50_valid.size else float("nan")
        ar_100 = float(np.mean(recall_valid)) if recall_valid.size else float("nan")

        f1 = (2 * ap_50_95 * ar_100 / (ap_50_95 + ar_100)) if (ap_50_95 + ar_100) > 0 else 0.0

        rows.append({
            "class_id": category_id,
            "category_id": category_id,
            "class_name": class_name,
            "precision": ap_50_95,
            "recall": ar_100,
            "f1_score": f1,
            "mAP50": ap_50,
            "mAP50_95": ap_50_95,
            # Backward-compatible aliases.
            "AP50": ap_50,
            "AP50_95": ap_50_95,
        })

    return rows


def run_coco_evaluation(
    annotation_file: str,
    prediction_json_path: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], COCOeval]:
    coco_gt = COCO(annotation_file)
    coco_dt = coco_gt.loadRes(prediction_json_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = summarize_coco_stats(coco_eval)

    return metrics, [], coco_eval


# ──────────────────────────────────────────────────────────────────────────────
# 7. Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_model(
    model,
    processor,
    dataloader,
    device: torch.device,
    warmup_batches: int = 3,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    total_images = 0
    total_time = 0.0

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="[Benchmark]")):
            encoding = processor(images=images, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)
            pixel_mask = encoding["pixel_mask"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            if batch_idx >= warmup_batches:
                total_time += end - start
                total_images += len(images)

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    avg_time = total_time / max(total_images, 1)
    fps = total_images / total_time if total_time > 0 else 0.0

    return {
        "benchmark_images": int(total_images),
        "total_seconds": float(total_time),
        "avg_seconds_per_image": float(avg_time),
        "avg_ms_per_image": float(avg_time * 1000),
        "fps": float(fps),
        # Backward-compatible aliases.
        "fps_from_forward_time": float(fps),
        "note": "Forward-pass timing excludes preprocessing and postprocessing.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Main Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    images_dir = args.images if args.images else resolve_images_dir(args.data, args.split)
    annotations_path = args.annotations if args.annotations else infer_annotations_path(args.data, args.split)
    class_names = load_class_names_from_yaml(args.data)

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"[ERROR] Images directory not found: {images_dir}")

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"[ERROR] Annotation file not found: {annotations_path}")

    print(f"[Info] Device: {device}")
    print(f"[Info] Weights: {args.weights}")
    print(f"[Info] Images: {images_dir}")
    print(f"[Info] Annotations: {annotations_path}")
    print(f"[Info] Split: {args.split}")
    print(f"[Info] Output dir: {output_dir}")
    print(f"[Info] Number of classes: {len(class_names)}")
    print(f"[Info] Class names: {class_names}")
    print(f"[Info] Score threshold: {args.threshold}")
    print(f"[Info] Category ID offset: {args.category_id_offset}")

    dataset = COCODetectionEvalDataset(
        image_dir=images_dir,
        annotation_file=annotations_path,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    print("\n[Info] Loading Deformable DETR model...")

    processor = DeformableDetrImageProcessor.from_pretrained(args.weights)
    model = DeformableDetrForObjectDetection.from_pretrained(args.weights)
    model.to(device)
    model.eval()

    all_predictions = []
    total_images = 0
    total_inference_time = 0.0

    print("\n[Info] Running evaluation inference...")

    with torch.no_grad():
        for images, targets in tqdm(dataloader, total=len(dataloader)):
            encoding = processor(images=images, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)
            pixel_mask = encoding["pixel_mask"].to(device)

            target_sizes = torch.tensor(
                [[target["height"], target["width"]] for target in targets],
                dtype=torch.long,
                device=device,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            total_inference_time += end_time - start_time
            total_images += len(images)

            processed_outputs = processor.post_process_object_detection(
                outputs,
                threshold=args.threshold,
                target_sizes=target_sizes,
            )

            batch_predictions = convert_outputs_to_coco_predictions(
                processed_outputs=processed_outputs,
                targets=targets,
                score_threshold=args.threshold,
                category_id_offset=args.category_id_offset,
            )

            all_predictions.extend(batch_predictions)

    predictions_path = output_dir / "deformable_detr_predictions_coco.json"
    save_json(predictions_path, all_predictions)

    avg_inference_time = total_inference_time / max(total_images, 1)
    fps = total_images / total_inference_time if total_inference_time > 0 else 0.0
    model_size_mb = get_model_size_mb(args.weights)

    speed_summary = {
        "total_images": int(total_images),
        "total_inference_time_seconds": float(total_inference_time),
        "avg_seconds_per_image": float(avg_inference_time),
        "avg_ms_per_image": float(avg_inference_time * 1000),
        "fps": float(fps),
        # Backward-compatible aliases.
        "fps_from_forward_time": float(fps),
        "avg_inference_time_seconds_per_image": float(avg_inference_time),
        "avg_inference_time_ms_per_image": float(avg_inference_time * 1000),
        "model_size_mb": float(model_size_mb),
    }

    if len(all_predictions) == 0:
        print("\n[Warning] No predictions were generated. Metrics will be set to 0.")

        metrics_summary = {
            "precision_mean": 0.0,
            "recall_mean": 0.0,
            "f1_mean": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "mAP75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
            **speed_summary,
            "weights": args.weights,
            "data": args.data,
            "split": args.split,
            "images": images_dir,
            "annotations": annotations_path,
            "threshold": args.threshold,
            "category_id_offset": args.category_id_offset,
        }

        save_json(output_dir / "metrics_summary.json", metrics_summary)
        save_dict_to_csv(metrics_summary, str(output_dir / "metrics_summary.csv"))
        save_rows_to_csv([], str(output_dir / "per_class_metrics.csv"))

        return

    print("\n[Info] Running COCO evaluation...")

    coco_gt = COCO(annotations_path)
    coco_dt = coco_gt.loadRes(str(predictions_path))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_metrics = summarize_coco_stats(coco_eval)
    per_class_metrics = extract_per_class_metrics(
        coco_eval=coco_eval,
        coco_gt=coco_gt,
        class_names=class_names,
    )

    metrics_summary = {
        **coco_metrics,
        **speed_summary,
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "images": images_dir,
        "annotations": annotations_path,
        "threshold": args.threshold,
        "category_id_offset": args.category_id_offset,
    }

    save_json(output_dir / "metrics_summary.json", metrics_summary)
    save_dict_to_csv(metrics_summary, str(output_dir / "metrics_summary.csv"))
    save_rows_to_csv(per_class_metrics, str(output_dir / "per_class_metrics.csv"))

    if args.benchmark:
        print("\n[Info] Running separate FPS benchmark...")

        benchmark_results = benchmark_model(
            model=model,
            processor=processor,
            dataloader=dataloader,
            device=device,
            warmup_batches=args.warmup,
            max_batches=args.benchmark_count,
        )

        save_json(output_dir / "benchmark_fps.json", benchmark_results)
        save_dict_to_csv(benchmark_results, str(output_dir / "benchmark_fps.csv"))

    print("\n[Done] Evaluation finished.")
    print(f"[Info] Results saved to: {output_dir}")

    print("\nSummary:")
    print(f"  Precision mean  : {metrics_summary['precision_mean']:.4f}")
    print(f"  Recall mean     : {metrics_summary['recall_mean']:.4f}")
    print(f"  F1 mean         : {metrics_summary['f1_mean']:.4f}")
    print(f"  mAP@0.5         : {metrics_summary['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95    : {metrics_summary['mAP50_95']:.4f}")
    print(f"  FPS             : {metrics_summary['fps']:.2f}")
    print(f"  Time/Image      : {metrics_summary['avg_ms_per_image']:.2f} ms")
    print(f"  Model Size      : {metrics_summary['model_size_mb']:.2f} MB")


# ──────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Deformable DETR on COCO-format object detection dataset"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="models/deformable_detr/best",
        help="Path to Hugging Face Deformable DETR model folder, e.g. models/deformable_detr/best",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset.yaml",
    )

    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Optional images directory. If omitted, resolved from dataset.yaml and --split.",
    )

    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help=(
            "Optional COCO annotation JSON. If omitted, inferred as "
            "data/processed/annotations/<split>.json from dataset.yaml path."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/deformable_detr/evaluation",
        help="Directory to save evaluation results.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help=(
            "Confidence threshold for exported predictions. "
            "For COCO mAP, keep this low, e.g. 0.001 or 0.05."
        ),
    )

    parser.add_argument(
        "--category-id-offset",
        type=int,
        default=0,
        help=(
            "Offset applied to Deformable DETR predicted labels before COCO eval. "
            "Use 0 if your COCO category_id is 0-based. "
            "Use 1 if your COCO category_id is 1-based."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda or cpu. Default: auto.",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a separate FPS benchmark after evaluation.",
    )

    parser.add_argument(
        "--warmup",
        "--warmup-batches",
        dest="warmup",
        type=int,
        default=3,
        help="Number of warmup batches for benchmark. Alias: --warmup-batches.",
    )

    parser.add_argument(
        "--benchmark-count",
        "--benchmark-batches",
        dest="benchmark_count",
        type=int,
        default=None,
        help="Maximum number of batches for benchmark. Alias: --benchmark-batches.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
