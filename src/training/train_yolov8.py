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
        imgsz=416,
        batch=16,
        device=device,
        project='models/yolov8',
        name='train_results',
        exist_ok=True,
        patience=10,      # Early stopping mechanism
        save=True,        # Save weights
        save_period=5,    # Save checkpoint every 5 epochs
        val=True,         # Perform validation after each epoch
        plots=True        # Generate visualization plots
    )

    # Explicitly load the best weights for final evaluation
    best_model_path = os.path.join('models/yolov8', 'train_results', 'weights', 'best.pt')
    best_model = YOLO(best_model_path)
    
    # Evaluate the best model
    metrics = best_model.val()
    print(f"Validation metrics: {metrics}")
    
    print("Training completed successfully.")
    print(f"Weights and logs are saved in: {os.path.dirname(best_model_path)}")

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