import os
import argparse
import torch
from ultralytics import YOLO

def train_yolov8(config_path):
    print(f"Starting YOLOv8 training with config: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Check GPU availability dynamically
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load pre-trained model
    model = YOLO('yolov8n.pt')

    # Execute training with advanced parameters
    results = model.train(
        data=config_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project='models/yolov8',
        name='train_results',
        exist_ok=True,
        patience=20,
        lr0=0.01,
        lrf=0.1,  
        optimizer='AdamW',
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,

        # Augmentation
        hsv_h=0.015,        # color jitter
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,         # flip horizontaly (car, human are both valid)
        mosaic=1.0,         # mosaic augmentation — very effective for YOLO
        mixup=0.1,          # mixup
        copy_paste=0.1,     # help classs which has less sample like bicycle, stop sign

        save=True,
        save_period=10,
        val=True,
        plots=True,
        workers=4,         
    )

    # Explicitly load the best weights for final evaluation
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    print(f"Loading best model from: {best_model_path}")
    best_model = YOLO(best_model_path)
    
    metrics = best_model.val()
    print(f"Validation metrics: {metrics}")
    
    print("Training completed successfully.")
    print(f"All outputs saved to: {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Object Detection Model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/yolov8_config.yaml',
        help='Path to the dataset configuration YAML file'
    )
    args = parser.parse_args()
    train_yolov8(args.config)