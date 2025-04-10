import os
from ultralytics import YOLO

# Tải mô hình cơ sở
model = YOLO('yolov8n.pt')  # hoặc 's', 'm', 'l', 'x' tùy theo kích thước mong muốn

data = os.path.abspath('custom_yolo_dataset/data.yaml')

# Huấn luyện mô hình với dữ liệu đã chuyển đổi
results = model.train(
    data=data,
    epochs=300,  # Số epoch huấn luyện
    imgsz=1280,   # Kích thước ảnh đầu vào
    batch=5,    # Kích thước batch
    name='custom_model',  # Tên thư mục lưu kết quả,
    pretrained=False,  # Không sử dụng mô hình đã huấn luyện trước
)

# Mô hình đã huấn luyện sẽ được lưu tại: runs/detect/custom_model/weights/best.pt