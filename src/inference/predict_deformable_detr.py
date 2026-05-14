import os
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from transformers import (
        DeformableDetrForObjectDetection,
        DeformableDetrImageProcessor
    )
except ImportError:
    raise ImportError(
        "[ERROR] Missing required libraries. Run: pip install transformers timm"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Label Mapping
# ──────────────────────────────────────────────────────────────────────────────
ID2LABEL = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "traffic light",
    7: "stop sign"
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Utility Functions
# ──────────────────────────────────────────────────────────────────────────────
def load_model_and_processor(model_dir: str, device: torch.device):
    """
    Load Deformable DETR model and processor from a Hugging Face save_pretrained directory.

    Expected directory structure:
        models/deformable_detr/best/
            ├── config.json
            ├── model.safetensors or pytorch_model.bin
            └── preprocessor_config.json
    """

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"[ERROR] Model directory not found: {model_dir}")

    print(f"[Info] Loading processor from: {model_dir}")
    processor = DeformableDetrImageProcessor.from_pretrained(model_dir)

    print(f"[Info] Loading Deformable DETR model from: {model_dir}")
    model = DeformableDetrForObjectDetection.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    return model, processor


def load_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Input image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load image: {e}")

    return image


def get_font(font_size: int = 16):
    """
    Try to load a common font. Fall back to default PIL font if unavailable.
    """

    font_candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def draw_predictions(image, results, threshold: float):
    """
    Draw bounding boxes, class names and confidence scores on image.
    """

    draw = ImageDraw.Draw(image)
    font = get_font(font_size=16)

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    detected_count = 0

    for box, score, label in zip(boxes, scores, labels):
        score = float(score)

        if score < threshold:
            continue

        detected_count += 1

        label_id = int(label)
        label_name = ID2LABEL.get(label_id, f"Unknown_{label_id}")

        x_min, y_min, x_max, y_max = box.tolist()
        x_min = round(x_min, 2)
        y_min = round(y_min, 2)
        x_max = round(x_max, 2)
        y_max = round(y_max, 2)

        text = f"{label_name}: {score:.2f}"

        print(
            f"Detected: {label_name} | "
            f"Confidence: {score:.4f} | "
            f"Box: [{x_min}, {y_min}, {x_max}, {y_max}]"
        )

        # Draw bounding box
        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline="red",
            width=3
        )

        # Draw text background
        text_bbox = draw.textbbox((x_min, y_min), text, font=font)
        text_bg = [
            text_bbox[0],
            text_bbox[1],
            text_bbox[2] + 4,
            text_bbox[3] + 4
        ]

        draw.rectangle(text_bg, fill="red")

        # Draw text
        draw.text(
            (x_min + 2, y_min + 2),
            text,
            fill="white",
            font=font
        )

    print(f"[Info] Number of detections after threshold: {detected_count}")

    return image


# ──────────────────────────────────────────────────────────────────────────────
# 3. Prediction Function
# ──────────────────────────────────────────────────────────────────────────────
def predict_deformable_detr(
    model_dir: str,
    image_path: str,
    output_path: str,
    threshold: float
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Info] Using device: {device}")
    print(f"[Info] Confidence threshold: {threshold}")

    model, processor = load_model_and_processor(model_dir, device)
    image = load_image(image_path)

    original_width, original_height = image.size
    print(f"[Info] Input image size: {original_width}x{original_height}")

    # Processor prepares image tensor
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert raw outputs to boxes in original image size
    target_sizes = torch.tensor(
        [[original_height, original_width]],
        device=device
    )

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )[0]

    print(f"[Info] Raw detections after post-process: {len(results['scores'])}")

    # Move result tensors to CPU
    results = {
        key: value.cpu()
        for key, value in results.items()
    }

    output_image = image.copy()
    output_image = draw_predictions(
        output_image,
        results,
        threshold=threshold
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_image.save(output_path)
    print(f"[Done] Prediction result saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for Deformable DETR"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/deformable_detr/best",
        help="Path to the saved Deformable DETR model directory"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/predictions/deformable_detr_result.jpg",
        help="Path to save the output image"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for filtering predictions"
    )

    args = parser.parse_args()

    predict_deformable_detr(
        model_dir=args.model_dir,
        image_path=args.image_path,
        output_path=args.output_path,
        threshold=args.threshold
    )