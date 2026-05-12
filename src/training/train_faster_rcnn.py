import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 1. Custom Dataset: Đọc ảnh và nhãn định dạng YOLO, chuyển sang định dạng Faster R-CNN
class YOLOTrafficDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.imgs = [img for img in sorted(os.listdir(img_dir)) if img.endswith('.jpg')]

    def __getitem__(self, idx):
        # Tải ảnh
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        height, width, _ = image.shape
        image /= 255.0 # Chuẩn hóa ảnh [0, 1]
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Tải nhãn
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    
                    # Chuyển đổi tọa độ YOLO (chuẩn hóa) sang Faster R-CNN (pixel tuyệt đối x1, y1, x2, y2)
                    x1 = (x_center - w / 2) * width
                    y1 = (y_center - h / 2) * height
                    x2 = (x_center + w / 2) * width
                    y2 = (y_center + h / 2) * height
                    
                    boxes.append([x1, y1, x2, y2])
                    # Faster R-CNN: Lớp 0 luôn là Background, nên ID vật thể phải cộng thêm 1
                    labels.append(int(class_id) + 1) 

        # Nếu ảnh không có nhãn (Background image)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        return image, target

    def __len__(self):
        return len(self.imgs)

# Hàm hỗ trợ gom batch cho PyTorch DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    # Load mô hình pre-trained ResNet50
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Lấy số lượng features của layer phân loại cuối cùng
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Thay thế Head cũ bằng Head mới với số lớp của bài toán (bao gồm background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_faster_rcnn(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Bắt đầu huấn luyện Faster R-CNN trên: {device}")

    # Chuẩn bị Dataset và DataLoader
    train_dataset = YOLOTrafficDataset(config['dataset']['train_images'], config['dataset']['train_labels'])
    train_data_loader = DataLoader(
        train_dataset, batch_size=config['model']['batch_size'], shuffle=True, 
        num_workers=2, collate_fn=collate_fn
    )

    # Khởi tạo mô hình và Optimizer
    model = get_model(config['model']['num_classes'])
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config['model']['learning_rate'], momentum=0.9, weight_decay=0.0005)

    num_epochs = config['model']['epochs']
    os.makedirs(config['save_dir'], exist_ok=True)

    # Vòng lặp huấn luyện (Training Loop)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass: Faster R-CNN tự động tính toán Total Loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Total Loss: {epoch_loss/len(train_data_loader):.4f}")

        # Lưu checkpoint cuối cùng (last.pth)
        torch.save(model.state_dict(), os.path.join(config['save_dir'], 'last.pth'))

    # Lưu best.pth (Trong ví dụ này lưu epoch cuối làm best để tiết kiệm code validation)
    torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best.pth'))
    print(f"Huấn luyện hoàn tất! Trọng số lưu tại: {config['save_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN")
    parser.add_argument('--config', type=str, default='configs/faster_rcnn_config.yaml')
    args = parser.parse_args()
    train_faster_rcnn(args.config)