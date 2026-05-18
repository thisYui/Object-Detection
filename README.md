# Object Detection Application

## 1. Overview

This project implements an object detection pipeline for traffic-related objects. It includes dataset preparation, model training, model evaluation, result comparison, single-image inference, and a Flask web application.

The project follows a **research + application** workflow:

- **Research:** train and evaluate multiple object detection architectures on the same dataset.
- **Application:** select the best model based on experiment results and deploy it in a web application.

The final deployed model is **Faster R-CNN**, selected because it achieved the highest detection accuracy on the test set.

---

## 2. Dataset

The dataset is built from a traffic-related subset of COCO. It contains 8 object classes:

```text
0. person
1. bicycle
2. car
3. motorcycle
4. bus
5. truck
6. traffic light
7. stop sign
```

The processed dataset is split using a **70/15/15** ratio:

| Split | Images | Purpose |
|---|---:|---|
| Train | 3500 | Model training |
| Validation | 749 | Validation during training |
| Test | 750 | Final evaluation and model comparison |

The dataset is stored in both YOLO-style and COCO-style formats:

- YOLO labels: used by YOLOv8.
- COCO JSON annotations: used by Faster R-CNN and Deformable DETR.

Dataset structure:

```text
data/processed/
+-- train/
|   +-- images/
|   +-- labels/
+-- val/
|   +-- images/
|   +-- labels/
+-- test/
|   +-- images/
|   +-- labels/
+-- annotations/
    +-- train.json
    +-- val.json
    +-- test.json
```

Dataset configuration:

```text
configs/dataset.yaml
```

---

## 3. Models

Three object detection architectures were trained and evaluated:

| Model | Type | Main characteristic |
|---|---|---|
| Faster R-CNN | Two-stage CNN detector | Highest accuracy in this project |
| YOLOv8 | One-stage YOLO detector | Fastest inference speed |
| Deformable DETR | Transformer-based detector | Modern architecture, harder to tune |

### 3.1. Faster R-CNN

Faster R-CNN is a two-stage detector. It first proposes candidate regions using a Region Proposal Network, then classifies and refines bounding boxes.

Advantages:

- Best accuracy in the current experiments.
- Strong and stable baseline for object detection.
- Good bounding box localization.

Limitations:

- Slow inference on CPU.
- Larger model size.
- More complex training and deployment than YOLOv8.

### 3.2. YOLOv8

YOLOv8 is a one-stage detector that predicts boxes and classes in a single forward pass.

Advantages:

- Fastest inference speed.
- Small model size.
- Easy to train and deploy.

Limitations:

- Lower mAP than Faster R-CNN in this project.
- May be less stable for small or occluded objects.

### 3.3. Deformable DETR

Deformable DETR is a Transformer-based object detector using deformable attention.

Advantages:

- Represents a modern Transformer-based approach.
- Can model global relationships in an image.

Limitations:

- Lowest accuracy in the current experiments.
- More difficult to train and tune.
- Requires more data, epochs, and compute to become competitive.

---

## 4. Current Project Structure

```text
Object-Detection/
+-- README.md
+-- requirements.txt
+-- configs/
|   +-- dataset.yaml
|   +-- faster_rcnn_config.yaml
|   +-- yolov8_config.yaml
|   +-- deformable_detr_config.yaml
+-- data/
|   +-- raw/
|   +-- processed/
+-- src/
|   +-- data/
|   |   +-- download_dataset.py
|   |   +-- split_dataset.py
|   |   +-- convert_to_yolo.py
|   |   +-- convert_to_coco.py
|   +-- training/
|   |   +-- train_faster_rcnn.py
|   |   +-- train_yolov8.py
|   |   +-- train_deformable_detr.py
|   +-- evaluation/
|   |   +-- evaluate_faster_rcnn.py
|   |   +-- evaluate_yolov8.py
|   |   +-- evaluate_deformable_detr.py
|   |   +-- compare_models.py
|   +-- inference/
|       +-- predict_faster_rcnn.py
|       +-- predict_yolov8.py
|       +-- predict_deformable_detr.py
+-- models/
|   +-- faster_rcnn/
|   +-- yolov8/
|   +-- deformable_detr/
+-- experiments/
|   +-- faster_rcnn/
|   +-- yolov8/
|   +-- deformable_detr/
|   +-- model_comparison/
+-- app/
    +-- main.py
    +-- model_loader.py
    +-- predict.py
    +-- static/
    |   +-- css/
    |   +-- js/
    |   +-- uploads/
    |   +-- results/
    +-- templates/
        +-- base.html
        +-- index.html
        +-- result.html
```

---

## 5. Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The project has been tested with the local virtual environment:

```powershell
venv\Scripts\python.exe
```

---

## 6. Training

### 6.1. Faster R-CNN

```powershell
python src\training\train_faster_rcnn.py --config configs\faster_rcnn_config.yaml
```

Weights are saved to:

```text
models/faster_rcnn/
+-- best.pth
+-- last.pth
```

### 6.2. YOLOv8

```powershell
python src\training\train_yolov8.py --config configs\yolov8_config.yaml
```

Weights are saved to:

```text
models/yolov8/
+-- best.pt
+-- last.pt
```

### 6.3. Deformable DETR

```powershell
python src\training\train_deformable_detr.py --config configs\deformable_detr_config.yaml
```

Weights are saved to:

```text
models/deformable_detr/
+-- best/
+-- last/
```

---

## 7. Evaluation

All models were evaluated on the same test split. FPS values were measured on **CPU**.

### 7.1. Faster R-CNN

```powershell
python src\evaluation\evaluate_faster_rcnn.py --weights models\faster_rcnn\best.pth --data configs\dataset.yaml --annotations data\processed\annotations\test.json --split test --output-dir experiments\faster_rcnn\evaluation --batch 4 --device cpu --workers 2 --conf 0.001 --category-id-offset -1 --benchmark --benchmark-count 100 --warmup 5
```

### 7.2. YOLOv8

```powershell
python src\evaluation\evaluate_yolov8.py --weights models\yolov8\best.pt --data configs\dataset.yaml --split test --output-dir experiments\yolov8\evaluation --imgsz 640 --batch 16 --device cpu --workers 2 --conf 0.001 --iou 0.6 --save-json --plots --benchmark --benchmark-count 100 --warmup 5
```

### 7.3. Deformable DETR

```powershell
python src\evaluation\evaluate_deformable_detr.py --weights models\deformable_detr\best --data configs\dataset.yaml --annotations data\processed\annotations\test.json --split test --output-dir experiments\deformable_detr\evaluation --threshold 0.001 --category-id-offset 0 --batch-size 4 --device cpu --num-workers 2 --benchmark --benchmark-count 100 --warmup 3
```

---

## 8. Model Comparison

Run:

```powershell
python src\evaluation\compare_models.py
```

Generated files:

```text
experiments/model_comparison/
+-- comparison_summary.csv
+-- model_ranking.csv
+-- per_class_comparison.csv
+-- comparison_report.md
```

Current comparison summary:

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | FPS | Model size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 0.3727 | 0.4718 | 0.4164 | 0.6062 | 0.3727 | 0.37 | 158.17 MB |
| YOLOv8 | 0.4492 | 0.3828 | 0.4133 | 0.3764 | 0.2406 | 23.81 | 5.94 MB |
| Deformable DETR | 0.0744 | 0.1823 | 0.1057 | 0.1604 | 0.0744 | 0.62 | 155.81 MB |

### Final model choice

The selected deployment model is **Faster R-CNN**.

Reason:

- It achieved the highest mAP@0.5.
- It achieved the highest mAP@0.5:0.95.
- It produced the best overall detection accuracy on the test set.

Trade-off:

- Faster R-CNN is slow on CPU, around 0.37 FPS.
- For real-time deployment, GPU acceleration or model optimization is recommended.

---

## 9. Inference

### 9.1. Faster R-CNN single-image inference

```powershell
venv\Scripts\python.exe src\inference\predict_faster_rcnn.py --image test.jpg --weights models\faster_rcnn\best.pth --output experiments\faster_rcnn\sample_predictions\test_result.jpg --threshold 0.5 --device cpu
```

The Faster R-CNN inference module exposes reusable functions for the web app:

- `load_faster_rcnn_model`
- `predict_faster_rcnn`
- `draw_detections`
- `filter_predictions`

---

## 10. Web Application

The web application is built with Flask and uses the selected **Faster R-CNN** model.

Main features:

- Upload an image and run object detection.
- Preview the selected image before detection.
- View detection result with bounding boxes, class names, confidence scores, and bbox coordinates.
- View the list of 8 supported traffic classes.
- Use webcam input.
- Auto-detect webcam frames every 2 seconds.
- Draw bounding boxes directly on the webcam preview.
- Start/stop webcam.
- Start/stop webcam auto detection.
- Mirrored webcam preview and mirrored captured frame.

### 10.1. Run the app

```powershell
venv\Scripts\python.exe app\main.py
```

Open:

```text
http://127.0.0.1:5000/
```

### 10.2. Stop the app

Find the process using port 5000:

```powershell
Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort,State,OwningProcess
```

Stop the process:

```powershell
Stop-Process -Id <OwningProcess>
```

### 10.3. Web app flow

Image upload flow:

```text
User selects image
        |
        v
Preview image in browser
        |
        v
Upload image to Flask backend
        |
        v
Run Faster R-CNN inference
        |
        v
Save result image to app/static/results/
        |
        v
Display result page
```

Webcam flow:

```text
User starts webcam
        |
        v
Browser captures a frame every 2 seconds
        |
        v
Frame is sent to /api/webcam-detect
        |
        v
Backend runs Faster R-CNN inference
        |
        v
Frontend draws bounding boxes on webcam overlay
```

---

## 11. Important Output Files

Evaluation outputs:

```text
experiments/faster_rcnn/evaluation/
experiments/yolov8/evaluation/
experiments/deformable_detr/evaluation/
```

Model comparison outputs:

```text
experiments/model_comparison/
```

Trained weights:

```text
models/faster_rcnn/best.pth
models/yolov8/best.pt
models/deformable_detr/best/
```

Web app runtime outputs:

```text
app/static/uploads/
app/static/results/
```

---

## 12. Technology Stack

```text
Python
PyTorch
TorchVision
Ultralytics YOLO
Hugging Face Transformers
Pandas
PyCOCOTools
Flask
HTML/CSS/JavaScript
Pillow
OpenCV
```

---

## 13. Summary Pipeline

```text
Prepare COCO traffic subset
        |
        v
Convert annotations to YOLO and COCO formats
        |
        v
Split data into train/validation/test using 70/15/15
        |
        v
Train Faster R-CNN, YOLOv8, and Deformable DETR
        |
        v
Evaluate all models on the same test set
        |
        v
Compare accuracy, speed, model size, and deployment trade-offs
        |
        v
Select Faster R-CNN based on highest accuracy
        |
        v
Deploy Faster R-CNN in Flask web application
```
