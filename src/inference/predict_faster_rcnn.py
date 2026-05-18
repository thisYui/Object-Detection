import argparse
import os

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_id2label(class_names=None):
    """
    Faster R-CNN reserves label 0 for background, so object classes start at 1.
    """
    names = class_names or DEFAULT_CLASS_NAMES
    return {idx + 1: name for idx, name in enumerate(names)}


def load_faster_rcnn_model(weights_path, device=None, class_names=None):
    """
    Load a Faster R-CNN model from a .pth state_dict checkpoint.

    Returns:
        tuple: (model, id2label, device)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    id2label = build_id2label(class_names)
    num_classes = len(id2label) + 1

    model = get_model(num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, id2label, device


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to load image: {image_path}") from exc


def run_faster_rcnn_prediction(model, image, device):
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        return model([image_tensor])[0]


def filter_predictions(prediction, id2label, threshold):
    boxes = prediction["boxes"].detach().cpu().tolist()
    scores = prediction["scores"].detach().cpu().tolist()
    labels = prediction["labels"].detach().cpu().tolist()

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue

        class_id = int(label)
        detections.append(
            {
                "class_id": class_id,
                "class_name": id2label.get(class_id, f"Unknown_{class_id}"),
                "confidence": round(float(score), 4),
                "bbox": [round(float(value), 2) for value in box],
            }
        )

    return detections


def get_font(font_size=15):
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        return ImageFont.load_default()


def draw_detections(image, detections, color="green"):
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    font = get_font()

    for detection in detections:
        box = detection["bbox"]
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"

        draw.rectangle(box, outline=color, width=3)
        text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box[0], box[1]), label, fill="white", font=font)

    return output_image


def predict_faster_rcnn(
    image_path,
    weights_path="models/faster_rcnn/best.pth",
    output_path="experiments/faster_rcnn/sample_predictions/faster_rcnn_result.jpg",
    threshold=0.5,
    device=None,
    class_names=None,
    save_image=True,
):
    """
    Run Faster R-CNN inference on one image and optionally save a visualization.

    Args:
        image_path (str): Input image path.
        weights_path (str): Path to Faster R-CNN .pth weights.
        output_path (str): Path to save the rendered result image.
        threshold (float): Minimum confidence score.
        device: Optional torch.device or device string.
        class_names (list[str] | None): Object class names without background.
        save_image (bool): Save rendered output image when True.

    Returns:
        dict: Inference result with output path and detections.
    """
    if device is not None:
        device = torch.device(device)

    model, id2label, device = load_faster_rcnn_model(weights_path, device, class_names)
    image = load_image(image_path)
    prediction = run_faster_rcnn_prediction(model, image, device)
    detections = filter_predictions(prediction, id2label, threshold)

    saved_output_path = None
    if save_image:
        output_image = draw_detections(image, detections)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_image.save(output_path)
        saved_output_path = output_path

    return {
        "image_path": image_path,
        "weights_path": weights_path,
        "output_path": saved_output_path,
        "threshold": threshold,
        "device": str(device),
        "detections": detections,
    }


def predict_and_visualize(model_path, image_path, output_path, threshold):
    """
    Backward-compatible wrapper for the old function name.
    """
    return predict_faster_rcnn(
        image_path=image_path,
        weights_path=model_path,
        output_path=output_path,
        threshold=threshold,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Faster R-CNN")
    parser.add_argument(
        "--weights",
        "--model_path",
        dest="weights_path",
        type=str,
        default="models/faster_rcnn/best.pth",
        help="Path to the saved best.pth weights file",
    )
    parser.add_argument(
        "--image",
        "--image_path",
        dest="image_path",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--output",
        "--output_path",
        dest="output_path",
        type=str,
        default="experiments/faster_rcnn/sample_predictions/faster_rcnn_result.jpg",
        help="Path to save the output image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for filtering predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on, for example cpu, cuda, or cuda:0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = predict_faster_rcnn(
        image_path=args.image_path,
        weights_path=args.weights_path,
        output_path=args.output_path,
        threshold=args.threshold,
        device=args.device,
    )

    print(f"Using device: {result['device']}")
    print(f"Found {len(result['detections'])} detections with threshold >= {args.threshold}")
    for detection in result["detections"]:
        print(
            "Detected: "
            f"{detection['class_name']} | "
            f"Confidence: {detection['confidence']:.2f} | "
            f"Box: {detection['bbox']}"
        )
    print(f"Prediction saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
