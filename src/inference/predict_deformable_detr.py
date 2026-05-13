import os
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor


# Giữ đúng mapping class như file train_deformable_detr.py
ID2LABEL = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "traffic light",
    7: "stop sign",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_and_processor(weights: str, device: torch.device):
    """
    Hỗ trợ 2 kiểu weight:
    1. Hugging Face folder: models/deformable_detr/best/
       - Có config.json, model.safetensors/pytorch_model.bin, preprocessor_config.json
       - Đây là kiểu được tạo bởi model.save_pretrained(...)

    2. File .pt/.pth: models/deformable_detr/best.pt
       - Script sẽ khởi tạo base model rồi load state_dict từ file.
    """
    weights_path = Path(weights)

    if weights_path.is_dir():
        processor = DeformableDetrImageProcessor.from_pretrained(str(weights_path))
        model = DeformableDetrForObjectDetection.from_pretrained(str(weights_path))
        model.to(device)
        model.eval()
        return model, processor

    processor = DeformableDetrImageProcessor.from_pretrained(
        "SenseTime/deformable-detr",
        size={"shortest_edge": 480, "longest_edge": 800},
        max_size=800,
    )

    model = DeformableDetrForObjectDetection.from_pretrained(
        "SenseTime/deformable-detr",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    checkpoint = torch.load(str(weights_path), map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format. Expected a state_dict-like .pt/.pth file.")

    # Xử lý checkpoint được lưu từ DataParallel/DDP nếu có prefix module.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model, processor


def draw_predictions(image: Image.Image, results: dict, score_threshold: float) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = results["labels"].tolist()

    for box, score, label_id in zip(boxes, scores, labels):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box
        label_name = ID2LABEL.get(int(label_id), str(label_id))
        text = f"{label_name}: {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        y_text = max(0, y1 - text_height - 4)
        draw.rectangle([x1, y_text, x1 + text_width + 6, y_text + text_height + 4], fill="red")
        draw.text((x1 + 3, y_text + 2), text, fill="white", font=font)

    return output


@torch.no_grad()
def predict_image(weights: str, image_path: str, output_path: str, threshold: float, device_arg: str):
    device = get_device(device_arg)
    print(f"[Info] Using device: {device}")

    model, processor = load_model_and_processor(weights, device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)  # (height, width)
    processed_results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    print("\n[Predictions]")
    if len(processed_results["scores"]) == 0:
        print("No object detected.")
    else:
        for score, label, box in zip(
            processed_results["scores"],
            processed_results["labels"],
            processed_results["boxes"],
        ):
            label_name = ID2LABEL.get(int(label.item()), str(int(label.item())))
            box_list = [round(x, 2) for x in box.tolist()]
            print(f"- {label_name:<15} score={score.item():.3f} box={box_list}")

    output_img = draw_predictions(image, processed_results, threshold)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output_img.save(output_path)
    print(f"\n[Done] Saved result to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict with Deformable DETR")
    parser.add_argument("--weights", type=str, default="models/deformable_detr/best.pt", help="Path to best folder, last folder, .pt, or .pth")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="experiments/deformable_detr/sample_predictions/result.jpg", help="Path to save output image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu")
    args = parser.parse_args()

    predict_image(
        weights=args.weights,
        image_path=args.image,
        output_path=args.output,
        threshold=args.threshold,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
