import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Thư mục chứa dữ liệu đã lọc
INPUT_DIR = "IDD_Segmentation_custom"
# Thư mục đầu ra cho định dạng YOLO
OUTPUT_DIR = "custom_yolo_dataset"

# Tạo cấu trúc thư mục YOLO
os.makedirs(os.path.join(OUTPUT_DIR, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "labels"), exist_ok=True)

# Tìm tất cả các file JSON
json_files = []
for root, _, files in os.walk(os.path.join(INPUT_DIR, "gtFine")):
    for file in files:
        if file.endswith("_polygons.json"):
            json_files.append(os.path.join(root, file))

# Tỷ lệ phân chia train/val
TRAIN_RATIO = 0.8
np.random.shuffle(json_files)
split_idx = int(len(json_files) * TRAIN_RATIO)
train_files = json_files[:split_idx]
val_files = json_files[split_idx:]

# Ánh xạ nhãn
class_map = {"motorcycle": 0, "bicycle": 1}

# Hàm tính bounding box từ polygon
def polygon_to_bbox(polygon):
    # Chuyển polygon thành numpy array
    points = np.array(polygon)
    # Tìm tọa độ min, max
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    return [x_min, y_min, x_max, y_max]

# Xử lý từng file
def process_files(file_list, split_name):
    for json_file in tqdm(file_list, desc=f"Xử lý {split_name}"):
        # Đọc file JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Tìm tên file ảnh tương ứng
        base_name = os.path.basename(json_file).split('_gtFine_polygons.json')[0]
        rel_dir = os.path.relpath(os.path.dirname(json_file), os.path.join(INPUT_DIR, "gtFine"))
        img_name = f"{base_name}_leftImg8bit.png"
        img_path = os.path.join(INPUT_DIR, "leftImg8bit", rel_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Không tìm thấy ảnh: {img_path}")
            continue
        
        # Đọc kích thước ảnh
        img_width = data['imgWidth']
        img_height = data['imgHeight']
        
        # Tạo file label YOLO
        yolo_labels = []
        
        for obj in data['objects']:
            label = obj['label'].lower()
            
            # Chỉ xử lý motorcycle và bicycle
            if label not in class_map:
                continue
                
            class_id = class_map[label]
            polygon = obj['polygon']
            
            # Chuyển polygon thành bbox
            x_min, y_min, x_max, y_max = polygon_to_bbox(polygon)
            
            # Chuyển sang định dạng YOLO (chuẩn hóa)
            x_center = (x_min + x_max) / (2 * img_width)
            y_center = (y_min + y_max) / (2 * img_height)
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Giới hạn giá trị trong khoảng [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        # Nếu có nhãn hợp lệ
        if yolo_labels:
            # Tạo tên file mới
            new_img_name = f"{base_name}.png"
            new_label_name = f"{base_name}.txt"
            
            # Đường dẫn đầu ra
            output_img_path = os.path.join(OUTPUT_DIR, split_name, "images", new_img_name)
            output_label_path = os.path.join(OUTPUT_DIR, split_name, "labels", new_label_name)
            
            # Sao chép file ảnh
            shutil.copy(img_path, output_img_path)
            
            # Ghi file nhãn
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

# Xử lý files
process_files(train_files, "train")
process_files(val_files, "val")

# Tạo file data.yaml
yaml_content = f"""# Dataset for motorcycle detection
train: {os.path.join(OUTPUT_DIR, 'train')}
val: {os.path.join(OUTPUT_DIR, 'val')}

# Classes
nc: {len(class_map)}
names: {list(class_map.keys())}
"""

with open(os.path.join(OUTPUT_DIR, "data.yaml"), 'w') as f:
    f.write(yaml_content)

print(f"Đã chuyển đổi dữ liệu sang định dạng YOLO trong thư mục {OUTPUT_DIR}")