import os
import argparse
import cv2
from ultralytics import YOLO

def run_inference(image_path, model_path, output_dir):
    """
    Run object detection inference on a single image using a trained YOLOv8 model.
    """
    print(f"Bắt đầu nhận diện trên ảnh: {image_path}")

    # 1. Kiểm tra đầu vào
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file trọng số tại {model_path}")

    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # 2. Tải mô hình YOLOv8 đã được huấn luyện
    print(f"Đang tải mô hình từ: {model_path}...")
    model = YOLO(model_path)

    # 3. Thực thi dự đoán
    # save=False vì chúng ta sẽ tự cấu hình lại việc lưu ảnh bằng OpenCV cho linh hoạt
    results = model.predict(source=image_path, conf=0.25, save=False)

    # 4. Trích xuất và vẽ kết quả lên ảnh gốc
    # results là một danh sách, do ta chỉ đưa vào 1 ảnh nên lấy phần tử đầu tiên [0]
    result = results[0]
    
    # Lấy mảng ảnh (numpy array) nguyên bản
    img = result.orig_img
    
    # Duyệt qua từng đối tượng được phát hiện
    boxes = result.boxes
    for box in boxes:
        # Lấy tọa độ bounding box [x1, y1, x2, y2]
        b = box.xyxy[0].cpu().numpy().astype(int)
        
        # Lấy nhãn (class id) và điểm tự tin (confidence score)
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        
        # Lấy tên của lớp đối tượng từ metadata của mô hình
        class_name = model.names[cls_id]

        # Định dạng text hiển thị trên ảnh
        label = f"{class_name} {conf:.2f}"

        # Vẽ hình chữ nhật (Bounding Box)
        # Sử dụng màu xanh lá (0, 255, 0)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        
        # Thêm khung nền cho text để dễ đọc hơn
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (b[0], b[1] - 20), (b[0] + w, b[1]), (0, 255, 0), -1)
        
        # Ghi tên class và điểm conf lên ảnh
        cv2.putText(img, label, (b[0], b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 5. Lưu ảnh kết quả
    base_name = os.path.basename(image_path)
    file_name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{file_name}_result{ext}")
    
    cv2.imwrite(output_path, img)
    print(f"Hoàn tất! Ảnh kết quả đã được lưu tại: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Object Detection with YOLOv8")
    
    # Đường dẫn đến file best.pt mà bạn vừa tải về
    parser.add_argument('--weights', type=str, default='models/yolov8/best.pt', help='Đường dẫn tới file trọng số mô hình')
    
    # Đường dẫn đến bức ảnh cần test
    parser.add_argument('--image', type=str, required=True, help='Đường dẫn tới ảnh cần nhận diện')
    
    # Nơi lưu kết quả
    parser.add_argument('--output', type=str, default='experiments/yolov8/sample_predictions', help='Thư mục lưu ảnh kết quả')
    
    args = parser.parse_args()
    run_inference(args.image, args.weights, args.output)