# Object Detection Application

## 1. Giới thiệu

Đây là đồ án cuối kỳ cho bài toán **Object Detection**. Mục tiêu của project là xây dựng một hệ thống nhận dạng đối tượng trên ảnh, trong đó nhóm thực hiện huấn luyện, đánh giá và so sánh nhiều kiến trúc mô hình khác nhau, sau đó chọn mô hình tốt nhất để triển khai thành ứng dụng web.

Project được xây dựng theo hướng **research + application**:

- Phần research: huấn luyện và đánh giá nhiều mô hình object detection trên cùng một dataset.
- Phần application: chọn mô hình tốt nhất để xây dựng web application cho phép upload ảnh và trả về kết quả nhận diện đối tượng.

---

## 2. Mục tiêu project

Project gồm 2 mục tiêu chính:

### 2.1. Research / Experiment

- Xây dựng bộ dữ liệu cho bài toán object detection.
- Dataset có tối thiểu 5 lớp đối tượng.
- Chia dữ liệu thành các tập `train`, `validation`, `test`.
- Huấn luyện và đánh giá 3 mô hình:
  - Faster R-CNN
  - YOLOv8
  - Deformable DETR
- So sánh kết quả giữa các mô hình.
- Phân tích ưu điểm, nhược điểm của từng mô hình.
- Chọn mô hình tốt nhất dựa trên kết quả thực nghiệm.

### 2.2. Application

- Xây dựng web application.
- Cho phép người dùng upload một ảnh.
- Sử dụng mô hình tốt nhất để phát hiện đối tượng trong ảnh.
- Trả về ảnh kết quả có bounding box, class name và confidence score.

---

## 3. Các mô hình sử dụng

Project sử dụng 3 kiến trúc object detection đại diện cho 3 hướng tiếp cận khác nhau:

```text
1. Faster R-CNN      - CNN-based / Two-stage detector
2. YOLOv8            - YOLO-based / One-stage detector
3. Deformable DETR   - Transformer-based detector
```

---

### 3.1. Faster R-CNN

Faster R-CNN là mô hình object detection thuộc nhóm **two-stage detector**.

Mô hình gồm hai giai đoạn chính:

1. Đề xuất vùng có khả năng chứa đối tượng bằng Region Proposal Network.
2. Phân loại đối tượng và tinh chỉnh bounding box.

Ưu điểm:

- Độ chính xác tốt.
- Phù hợp làm baseline để so sánh.
- Kiến trúc rõ ràng, dễ phân tích trong báo cáo.

Nhược điểm:

- Tốc độ inference chậm hơn YOLO.
- Khó triển khai real-time hơn.
- Thời gian huấn luyện và suy luận tương đối lớn.

---

### 3.2. YOLOv8

YOLOv8 là mô hình thuộc họ **YOLO - You Only Look Once**, là nhóm mô hình **one-stage detector**. Mô hình này có khả năng phát hiện đối tượng nhanh và phù hợp với các ứng dụng thực tế.

Ưu điểm:

- Tốc độ inference nhanh.
- Dễ huấn luyện với custom dataset.
- Dễ triển khai trong web application.
- Phù hợp với bài toán yêu cầu xử lý gần real-time.

Nhược điểm:

- Có thể kém ổn định hơn Faster R-CNN trong một số trường hợp có object nhỏ hoặc bị che khuất.
- Kết quả phụ thuộc nhiều vào chất lượng dataset và quá trình annotation.

---

### 3.3. Deformable DETR

Deformable DETR là mô hình object detection dựa trên **Transformer**, được cải tiến từ DETR. Mô hình này sử dụng cơ chế attention hiệu quả hơn để cải thiện tốc độ hội tụ và khả năng phát hiện object nhỏ.

Ưu điểm:

- Đại diện cho hướng tiếp cận hiện đại dựa trên Transformer.
- Có khả năng học quan hệ toàn cục trong ảnh.
- Cải thiện tốc độ hội tụ so với DETR gốc.

Nhược điểm:

- Cấu trúc phức tạp.
- Yêu cầu tài nguyên huấn luyện cao hơn.
- Khó triển khai và tinh chỉnh hơn YOLOv8.

---

## 4. Cấu trúc thư mục

```text
Object-Detection-Application/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── dataset.yaml
│   ├── faster_rcnn_config.yaml
│   ├── yolov8_config.yaml
│   └── deformable_detr_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── annotations/
│   │
│   ├── processed/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   │
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_dataset_preprocessing.ipynb
│   ├── 03_visualize_annotations.ipynb
│   └── 04_compare_results.ipynb
│
├── src/
│   ├── data/
│   │   ├── split_dataset.py
│   │   ├── convert_to_yolo.py
│   │   ├── convert_to_coco.py
│   │   └── check_dataset.py
│   │
│   ├── training/
│   │   ├── train_faster_rcnn.py
│   │   ├── train_yolov8.py
│   │   └── train_deformable_detr.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_faster_rcnn.py
│   │   ├── evaluate_yolov8.py
│   │   ├── evaluate_deformable_detr.py
│   │   ├── compare_models.py
│   │   └── metrics.py
│   │
│   ├── inference/
│   │   ├── predict_faster_rcnn.py
│   │   ├── predict_yolov8.py
│   │   ├── predict_deformable_detr.py
│   │   └── visualize_prediction.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── seed.py
│       └── visualization.py
│
├── models/
│   ├── faster_rcnn/
│   │   ├── best.pth
│   │   └── last.pth
│   │
│   ├── yolov8/
│   │   ├── best.pt
│   │   └── last.pt
│   │
│   └── deformable_detr/
│       ├── best.pth
│       └── last.pth
│
├── experiments/
│   ├── faster_rcnn/
│   │   ├── logs/
│   │   ├── results.csv
│   │   ├── confusion_matrix.png
│   │   └── sample_predictions/
│   │
│   ├── yolov8/
│   │   ├── logs/
│   │   ├── results.csv
│   │   ├── confusion_matrix.png
│   │   └── sample_predictions/
│   │
│   └── deformable_detr/
│       ├── logs/
│       ├── results.csv
│       ├── confusion_matrix.png
│       └── sample_predictions/
│
├── app/
│   ├── main.py
│   ├── model_loader.py
│   ├── predict.py
│   ├── static/
│   │   ├── uploads/
│   │   └── results/
│   │
│   └── templates/
│       ├── index.html
│       └── result.html
│
├── report/
│   ├── figures/
│   ├── tables/
│   ├── final_report.docx
│   └── final_report.pdf
│
└── scripts/
    ├── run_train_all.sh
    ├── run_evaluate_all.sh
    └── run_app.sh
```

---

## 5. Dataset

Dataset được sử dụng cho bài toán **Object Detection**. Bộ dữ liệu cần có tối thiểu **5 lớp đối tượng** và đủ số lượng ảnh để quá trình huấn luyện, đánh giá mô hình có ý nghĩa.

### 5.1. Danh sách lớp đối tượng

Ví dụ:

```text
Class 0: class_1
Class 1: class_2
Class 2: class_3
Class 3: class_4
Class 4: class_5
```

> Lưu ý: Khi triển khai thực tế, thay `class_1`, `class_2`, ... bằng tên các lớp thật trong dataset của nhóm.

---

### 5.2. Chia dữ liệu

Dataset được chia thành 3 tập:

```text
Train: 70%
Validation: 20%
Test: 10%
```

Hoặc có thể chia theo tỷ lệ:

```text
Train: 80%
Validation: 10%
Test: 10%
```

Trong đó:

- `train`: dùng để huấn luyện mô hình.
- `val`: dùng để đánh giá mô hình trong quá trình huấn luyện.
- `test`: dùng để đánh giá kết quả cuối cùng của mô hình sau khi huấn luyện.

---

### 5.3. Cấu trúc dataset

```text
data/
├── raw/
│   ├── images/
│   └── annotations/
│
└── processed/
    ├── train/
    │   ├── images/
    │   └── labels/
    │
    ├── val/
    │   ├── images/
    │   └── labels/
    │
    └── test/
        ├── images/
        └── labels/
```

Trong đó:

- `data/raw/images/`: chứa ảnh gốc.
- `data/raw/annotations/`: chứa annotation gốc.
- `data/processed/train/`: chứa dữ liệu huấn luyện.
- `data/processed/val/`: chứa dữ liệu validation.
- `data/processed/test/`: chứa dữ liệu test.

---

## 6. Cài đặt môi trường

### 6.1. Clone project

```bash
git clone https://github.com/your-username/Object-Detection-Application.git
cd Object-Detection-Application
```

### 6.2. Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
venv\Scripts\activate
```

Kích hoạt môi trường ảo trên macOS/Linux:

```bash
source venv/bin/activate
```

### 6.3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 7. Cấu hình dataset

Ví dụ file `configs/dataset.yaml`:

```yaml
path: data/processed
train: train/images
val: val/images
test: test/images

names:
  0: class_1
  1: class_2
  2: class_3
  3: class_4
  4: class_5
```

File này được sử dụng để khai báo đường dẫn dataset và danh sách class cho quá trình huấn luyện, đặc biệt là với YOLOv8.

---

## 8. Huấn luyện mô hình

Project thực hiện huấn luyện 3 mô hình:

```text
1. Faster R-CNN
2. YOLOv8
3. Deformable DETR
```

Mỗi mô hình được huấn luyện trên cùng một dataset để đảm bảo việc so sánh là công bằng.

---

### 8.1. Train Faster R-CNN

```bash
python src/training/train_faster_rcnn.py --config configs/faster_rcnn_config.yaml
```

Weight sau khi huấn luyện được lưu tại:

```text
models/faster_rcnn/
```

Ví dụ:

```text
models/faster_rcnn/
├── best.pth
└── last.pth
```

---

### 8.2. Train YOLOv8

```bash
python src/training/train_yolov8.py --config configs/yolov8_config.yaml
```

Weight sau khi huấn luyện được lưu tại:

```text
models/yolov8/
```

Ví dụ:

```text
models/yolov8/
├── best.pt
└── last.pt
```

---

### 8.3. Train Deformable DETR

```bash
python src/training/train_deformable_detr.py --config configs/deformable_detr_config.yaml
```

Weight sau khi huấn luyện được lưu tại:

```text
models/deformable_detr/
```

Ví dụ:

```text
models/deformable_detr/
├── best.pth
└── last.pth
```

---

## 9. Đánh giá mô hình

Sau khi huấn luyện, cả 3 mô hình được đánh giá trên cùng tập `test`.

### 9.1. Evaluate Faster R-CNN

```bash
python src/evaluation/evaluate_faster_rcnn.py --weights models/faster_rcnn/best.pth --data configs/dataset.yaml --annotations data/processed/annotations/test.json --split test --output-dir experiments/faster_rcnn/evaluation --batch 4 --device cpu --workers 2 --conf 0.001 --category-id-offset -1 --benchmark --benchmark-count 100 --warmup 5
```

### 9.2. Evaluate YOLOv8

```bash
python src/evaluation/evaluate_yolov8.py --weights models/yolov8/best.pt --data configs/dataset.yaml --split test --output-dir experiments/yolov8/evaluation --imgsz 640 --batch 16 --device cpu --workers 2 --conf 0.001 --iou 0.6 --save-json --plots --benchmark --benchmark-count 100 --warmup 5
```

### 9.3. Evaluate Deformable DETR

```bash
python src/evaluation/evaluate_deformable_detr.py --weights models/deformable_detr/best --data configs/dataset.yaml --annotations data/processed/annotations/test.json --split test --output-dir experiments/deformable_detr/evaluation --threshold 0.001 --category-id-offset 0 --batch-size 4 --device cpu --num-workers 2 --benchmark --benchmark-count 100 --warmup 3
```

---

## 10. Metrics đánh giá

Các mô hình được đánh giá bằng các chỉ số sau:

| Metric | Ý nghĩa |
|---|---|
| Precision | Tỷ lệ dự đoán đúng trên tổng số dự đoán |
| Recall | Tỷ lệ object thật được mô hình phát hiện đúng |
| F1-score | Trung bình điều hòa giữa Precision và Recall |
| mAP@0.5 | Mean Average Precision tại ngưỡng IoU = 0.5 |
| mAP@0.5:0.95 | Mean Average Precision trung bình trên nhiều ngưỡng IoU |
| Inference Time | Thời gian xử lý một ảnh |
| FPS | Số ảnh/frame xử lý được mỗi giây |
| Model Size | Kích thước file weight |
| Training Time | Thời gian huấn luyện |

---

## 11. So sánh kết quả

Sau khi đánh giá từng mô hình, chạy script tổng hợp kết quả:

```bash
python src/evaluation/compare_models.py
```

Kết quả so sánh được lưu tại:

```text
experiments/comparison_results.csv
```

Bảng so sánh mẫu:

| Model | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 | FPS | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | ... | ... | ... | ... | ... | ... | ... |
| YOLOv8 | ... | ... | ... | ... | ... | ... | ... |
| Deformable DETR | ... | ... | ... | ... | ... | ... | ... |

---

## 12. Chọn mô hình tốt nhất

Sau khi so sánh kết quả, nhóm chọn mô hình tốt nhất dựa trên các tiêu chí:

```text
- Độ chính xác
- Tốc độ inference
- Khả năng triển khai thực tế
- Độ phức tạp khi huấn luyện
- Độ phức tạp khi tích hợp vào web application
- Kích thước mô hình
```

Mô hình được chọn để triển khai ứng dụng là mô hình có sự cân bằng tốt nhất giữa **độ chính xác** và **tốc độ xử lý**.

Ví dụ, nếu kết quả thực nghiệm cho thấy YOLOv8 có tốc độ nhanh, độ chính xác tốt và dễ triển khai, nhóm có thể chọn:

```text
YOLOv8
```

Lý do chọn YOLOv8:

```text
- Tốc độ xử lý nhanh
- Dễ triển khai trong web application
- Kích thước model phù hợp
- Có thể chạy tốt trên CPU/GPU
- Phù hợp với bài toán thực tế
```

---

## 13. Chạy inference

Chạy thử dự đoán trên một ảnh:

```bash
python src/inference/predict_yolov8.py \
    --weights models/yolov8/best.pt \
    --image data/processed/test/images/sample.jpg
```

Ảnh kết quả được lưu tại:

```text
experiments/yolov8/sample_predictions/
```

Ví dụ kết quả:

```text
experiments/yolov8/sample_predictions/
├── sample_001_result.jpg
├── sample_002_result.jpg
└── sample_003_result.jpg
```

---

## 14. Web Application

Web application được xây dựng để demo mô hình tốt nhất sau quá trình thực nghiệm.

Ứng dụng cho phép người dùng:

```text
- Upload một ảnh từ máy tính
- Chạy object detection bằng mô hình đã chọn
- Hiển thị ảnh kết quả có bounding box
- Hiển thị tên lớp đối tượng
- Hiển thị confidence score
```

---

### 14.1. Chạy ứng dụng

```bash
python app/main.py
```

Sau đó mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

---

### 14.2. Luồng xử lý của ứng dụng

```text
User upload image
        ↓
Lưu ảnh vào app/static/uploads/
        ↓
Load best model
        ↓
Chạy inference
        ↓
Vẽ bounding box lên ảnh
        ↓
Lưu ảnh kết quả vào app/static/results/
        ↓
Hiển thị kết quả trên giao diện web
```

---

### 14.3. Cấu trúc thư mục app

```text
app/
├── main.py
├── model_loader.py
├── predict.py
├── static/
│   ├── uploads/
│   └── results/
│
└── templates/
    ├── index.html
    └── result.html
```

Trong đó:

- `main.py`: file chạy chính của web application.
- `model_loader.py`: load mô hình tốt nhất.
- `predict.py`: xử lý inference.
- `static/uploads/`: lưu ảnh người dùng upload.
- `static/results/`: lưu ảnh sau khi detect.
- `templates/index.html`: giao diện upload ảnh.
- `templates/result.html`: giao diện hiển thị kết quả.

---

## 15. Kết quả thực nghiệm

Kết quả thực nghiệm của từng mô hình được lưu trong thư mục:

```text
experiments/
├── faster_rcnn/
├── yolov8/
└── deformable_detr/
```

Mỗi thư mục mô hình gồm:

```text
logs/
results.csv
confusion_matrix.png
sample_predictions/
```

Trong đó:

- `logs/`: lưu log quá trình huấn luyện.
- `results.csv`: lưu kết quả đánh giá mô hình.
- `confusion_matrix.png`: ma trận nhầm lẫn.
- `sample_predictions/`: lưu một số ảnh dự đoán mẫu.

---

## 16. Công nghệ sử dụng

Project sử dụng các công nghệ và thư viện sau:

```text
- Python
- PyTorch
- TorchVision
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Flask
- HTML/CSS
```

---

## 17. Thành viên nhóm

| MSSV | Họ tên | Nhiệm vụ |
|---|---|---|
| ... | ... | Thu thập dữ liệu, annotation |
| ... | ... | Xử lý dataset, chia train/val/test |
| ... | ... | Train và đánh giá Faster R-CNN |
| ... | ... | Train và đánh giá YOLOv8 |
| ... | ... | Train và đánh giá Deformable DETR |
| ... | ... | Xây dựng web application |
| ... | ... | Viết báo cáo, tổng hợp kết quả |

---

## 18. Cách nộp bài

Nhóm nộp các thành phần sau:

```text
1. Source code
2. Dataset hoặc link Google Drive chứa dataset
3. Trained weights của các mô hình
4. Kết quả thực nghiệm
5. Báo cáo final_report.pdf
6. Hướng dẫn chạy project
```

Nếu dataset hoặc weight có kích thước lớn, nhóm có thể upload lên Google Drive và đính kèm link trong README hoặc báo cáo.

Ví dụ:

```text
Dataset: https://drive.google.com/...
Weights: https://drive.google.com/...
```

---

## 19. Tóm tắt pipeline

```text
Collect Dataset
        ↓
Annotate Images
        ↓
Split Train / Val / Test
        ↓
Train Faster R-CNN
        ↓
Train YOLOv8
        ↓
Train Deformable DETR
        ↓
Evaluate 3 Models
        ↓
Compare Results
        ↓
Select Best Model
        ↓
Build Web Application
        ↓
Final Report
```
