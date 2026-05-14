"""
evaluate_yolov8.py

Evaluate a YOLOv8 object detection model on a YOLO-format dataset.

Expected project usage:
    python src/evaluation/evaluate_yolov8.py \
        --weights models/yolov8/best.pt \
        --data configs/dataset.yaml \
        --split test \
        --output-dir experiments/yolov8/evaluation

Requirements:
    pip install ultralytics pyyaml pandas

The script saves:
    - metrics_summary.json
    - metrics_summary.csv
    - per_class_metrics.csv
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
import yaml

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "[ERROR] Missing ultralytics. Install it with: pip install ultralytics"
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
        # YAML keys can be int or str. Sort numerically where possible.
        def key_to_int(k: Any) -> int:
            try:
                return int(k)
            except Exception:
                return 10**9

        return [str(names[k]) for k in sorted(names.keys(), key=key_to_int)]

    raise ValueError("[ERROR] `names` in dataset YAML must be a list or dictionary.")


def resolve_split_images_dir(data_yaml_path: str | Path, data_cfg: Dict[str, Any], split: str) -> Optional[Path]:
    """
    Resolve image directory for train/val/test from YOLO data.yaml.

    Supports common YOLO format:
        path: data/processed
        train: train/images
        val: val/images
        test: test/images
    """
    split_value = data_cfg.get(split)
    if not split_value:
        return None

    # If split is a list file, return the list file path. Caller handles it.
    split_path = Path(str(split_value))

    if split_path.is_absolute():
        return split_path

    root = data_cfg.get("path")
    if root:
        root_path = Path(str(root))
        if not root_path.is_absolute():
            # Resolve relative path from current working directory first.
            candidate = (Path.cwd() / root_path / split_path).resolve()
            if candidate.exists():
                return candidate
            # Also support path relative to the YAML location.
            return (Path(data_yaml_path).resolve().parent / root_path / split_path).resolve()
        return (root_path / split_path).resolve()

    return (Path(data_yaml_path).resolve().parent / split_path).resolve()


def iter_images_from_path(path: Path) -> Iterable[Path]:
    """Yield image paths from either a directory or a YOLO image-list txt file."""
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p
    elif path.is_file() and path.suffix.lower() == ".txt":
        base = path.parent
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if not item:
                    continue
                p = Path(item)
                if not p.is_absolute():
                    p = (base / p).resolve()
                if p.exists() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    yield p


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def extract_summary_metrics(metrics: Any) -> Dict[str, Any]:
    """Extract common detection metrics from Ultralytics metric object."""
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", None)

    summary = {
        "precision_mean": safe_float(getattr(box, "mp", 0.0)),
        "recall_mean": safe_float(getattr(box, "mr", 0.0)),
        "mAP50": safe_float(getattr(box, "map50", 0.0)),
        "mAP50_95": safe_float(getattr(box, "map", 0.0)),
    }

    p = summary["precision_mean"]
    r = summary["recall_mean"]
    summary["f1_mean"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    if isinstance(speed, dict):
        # Usually contains preprocess, inference, loss, postprocess in ms/image.
        for key, value in speed.items():
            summary[f"speed_{key}_ms_per_image"] = safe_float(value)

        inference_ms = safe_float(speed.get("inference", 0.0))
        if inference_ms > 0:
            summary["fps_from_ultralytics_inference_time"] = 1000.0 / inference_ms

    return summary

def values_to_list(value):
    """Convert Ultralytics metric values safely to a Python list."""
    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        return value

    return [value]

def extract_per_class_metrics(metrics: Any, class_names: List[str]) -> pd.DataFrame:
    """Extract per-class precision/recall/AP when available."""
    box = getattr(metrics, "box", None)
    if box is None:
        return pd.DataFrame()

    # Ultralytics usually exposes arrays: p, r, ap50, maps.
    p_values = values_to_list(getattr(box, "p", None))
    r_values = values_to_list(getattr(box, "r", None))
    ap50_values = values_to_list(getattr(box, "ap50", None))
    map_values = values_to_list(getattr(box, "maps", None))

    n = max(len(class_names), len(p_values), len(r_values), len(ap50_values), len(map_values))
    rows = []
    for i in range(n):
        rows.append(
            {
                "class_id": i,
                "class_name": class_names[i] if i < len(class_names) else f"class_{i}",
                "precision": safe_float(p_values[i]) if i < len(p_values) else None,
                "recall": safe_float(r_values[i]) if i < len(r_values) else None,
                "AP50": safe_float(ap50_values[i]) if i < len(ap50_values) else None,
                "AP50_95": safe_float(map_values[i]) if i < len(map_values) else None,
            }
        )

    return pd.DataFrame(rows)


def benchmark_fps(
    model: YOLO,
    image_paths: List[Path],
    imgsz: int,
    conf: float,
    iou: float,
    device: Optional[str],
    max_images: int,
    warmup: int,
) -> Dict[str, Any]:
    """Simple end-to-end predict benchmark using image files."""
    selected = image_paths[: max_images if max_images > 0 else len(image_paths)]
    if not selected:
        return {
            "benchmark_images": 0,
            "avg_seconds_per_image": None,
            "fps_end_to_end": None,
            "note": "No images found for benchmarking.",
        }

    # Warmup on a few images.
    warmup_items = selected[: min(warmup, len(selected))]
    for img in warmup_items:
        _ = model.predict(
            source=str(img), imgsz=imgsz, conf=conf, iou=iou, device=device,
            verbose=False, save=False
        )

    start = time.perf_counter()
    for img in selected:
        _ = model.predict(
            source=str(img), imgsz=imgsz, conf=conf, iou=iou, device=device,
            verbose=False, save=False
        )
    elapsed = time.perf_counter() - start

    avg = elapsed / len(selected)
    fps = len(selected) / elapsed if elapsed > 0 else None

    return {
        "benchmark_images": len(selected),
        "total_seconds": elapsed,
        "avg_seconds_per_image": avg,
        "avg_ms_per_image": avg * 1000,
        "fps_end_to_end": fps,
        "note": "End-to-end timing includes model.predict overhead and image loading.",
    }


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 object detection model")

    parser.add_argument("--weights", type=str, default="models/yolov8/best.pt", help="Path to YOLOv8 weight file")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Path to YOLO dataset YAML")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--output-dir", type=str, default="experiments/yolov8/evaluation", help="Directory to save evaluation outputs")

    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. 0, cpu, cuda:0. Leave empty for auto")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for validation")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS/validation")

    parser.add_argument("--save-json", action="store_true", help="Ask Ultralytics to save COCO-style prediction JSON")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence scores in prediction labels where supported")
    parser.add_argument("--plots", action="store_true", help="Save validation plots such as PR curve/confusion matrix")

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

    print("[Info] Loading YOLO model")
    print(f"       Weights : {weights_path}")
    print(f"       Data    : {data_yaml_path}")
    print(f"       Split   : {args.split}")

    model = YOLO(str(weights_path))

    print("\n[Info] Running evaluation...")
    metrics = model.val(
        data=str(data_yaml_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        save_conf=args.save_conf,
        plots=args.plots,
        project=str(output_dir),
        name="ultralytics_val",
        exist_ok=True,
        verbose=False,
    )

    summary = extract_summary_metrics(metrics)
    summary.update(
        {
            "weights": str(weights_path),
            "data_yaml": str(data_yaml_path),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device or "auto",
            "num_classes": len(class_names),
            "class_names": class_names,
        }
    )

    per_class_df = extract_per_class_metrics(metrics, class_names)

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

    if "fps_from_ultralytics_inference_time" in summary:
        print(f"  FPS estimate   : {summary['fps_from_ultralytics_inference_time']:.2f}")

    print("\n[Saved]")
    print(f"  {summary_json_path}")
    print(f"  {summary_csv_path}")
    if not per_class_df.empty:
        print(f"  {per_class_csv_path}")

    if args.benchmark:
        split_path = resolve_split_images_dir(data_yaml_path, data_cfg, args.split)
        if split_path is None and args.split == "test":
            print("\n[Warn] `test` split is missing in data YAML. Falling back to `val` for FPS benchmark.")
            split_path = resolve_split_images_dir(data_yaml_path, data_cfg, "val")

        image_paths = list(iter_images_from_path(split_path)) if split_path else []
        bench = benchmark_fps(
            model=model,
            image_paths=image_paths,
            imgsz=args.imgsz,
            conf=max(args.conf, 0.25),
            iou=args.iou,
            device=args.device,
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

    print("\n[Done] YOLOv8 evaluation complete.")


if __name__ == "__main__":
    main()
