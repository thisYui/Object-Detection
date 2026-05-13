import os
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Thay thế Classification Head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def predict_and_visualize(model_path, image_path, output_path, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    # Khởi tạo Label Mapping (Faster R-CNN dành class 0 cho Background)
    # Các class giao thông bắt đầu từ 1
    id2label = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
        5: 'bus', 6: 'truck', 7: 'traffic light', 8: 'stop sign'
    }
    num_classes = len(id2label) + 1  # 8 class + 1 background = 9

    # Nạp mô hình và trọng số
    print(f"Loading Faster R-CNN model from: {model_path}")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Tải và xử lý ảnh đầu vào
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}")
        return

    try:
        # Bắt buộc ép kiểu RGB để tránh lỗi kênh màu
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load image. Error: {e}")
        return

    # Chuyển đổi ảnh thành Tensor [C, H, W] chuẩn bị cho PyTorch
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(original_image).to(device)


    # Suy luận (Inference)
    with torch.no_grad():
        # Torchvision Faster R-CNN tự động áp dụng NMS (Non-Maximum Suppression) trong chế độ eval()
        prediction = model([image_tensor])[0]

    # Khởi tạo công cụ vẽ
    draw = ImageDraw.Draw(original_image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()

    print(f"Found {len(boxes)} raw predictions. Filtering with threshold >= {threshold}...")

    # Vẽ kết quả thỏa mãn ngưỡng độ tin cậy
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            box = [round(i, 2) for i in box]
            score_val = round(score, 2)
            label_name = id2label.get(label, f"Unknown_{label}")

            print(f"Detected: {label_name} | Confidence: {score_val} | Box: {box}")

            # Vẽ khung chữ nhật (đổi màu xanh lá để phân biệt với DETR màu đỏ)
            draw.rectangle(box, outline="green", width=3)
            
            # Vẽ nền text và chữ
            text = f"{label_name}: {score_val}"
            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="green")
            draw.text((box[0], box[1]), text, fill="white", font=font)

    # Lưu ảnh kết quả
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    original_image.save(output_path)
    print(f"Prediction successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Faster R-CNN")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/faster_rcnn/best.pth", 
        help="Path to the saved best.pth weights file"
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
        default="data/predictions/faster_rcnn_result.jpg", 
        help="Path to save the output image"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5, 
        help="Confidence threshold for filtering predictions"
    )
    
    args = parser.parse_args()
    predict_and_visualize(args.model_path, args.image_path, args.output_path, args.threshold)