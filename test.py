import os
import json
import glob
import shutil
import base64
import hashlib
from tqdm import tqdm

# Thư mục chứa dữ liệu
BASE_DIR = "IDD_Segmentation"
GTFINE_DIR = os.path.join(BASE_DIR, "gtFine")
LEFTIMG_DIR = os.path.join(BASE_DIR, "leftImg8bit")  # Thêm đường dẫn tới thư mục ảnh
OUTPUT_DIR = "IDD_Segmentation_custom"

# Các lớp cần trích xuất
TARGET_CLASSES = ["motorcycle", "bicycle"]

# Tạo thư mục đầu ra
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tìm tất cả các file JSON
json_files = glob.glob(f"{GTFINE_DIR}/**/*_polygons.json", recursive=True)

# Thống kê
stats = {cls: 0 for cls in TARGET_CLASSES}
total_files = 0
files_with_targets = 0

# Dictionary lưu ánh xạ giữa md5 và tên file gốc
md5_to_original = {}

for json_file in tqdm(json_files):
    # Đọc file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Tìm đối tượng thuộc lớp cần trích xuất
    found_objects = []
    for obj in data['objects']:
        if obj['label'].lower() in TARGET_CLASSES:
            found_objects.append(obj)
            stats[obj['label'].lower()] += 1
    
    # Nếu tìm thấy đối tượng thuộc lớp cần trích xuất
    if found_objects:
        files_with_targets += 1
        
        # Tạo file JSON mới chỉ với các đối tượng đã lọc
        output_data = {
            'imgHeight': data['imgHeight'],
            'imgWidth': data['imgWidth'],
            'objects': found_objects
        }
        
        # Lấy phần tương đối từ thư mục gtFine
        rel_dir = os.path.dirname(os.path.relpath(json_file, GTFINE_DIR))
        
        # Lấy tên file cơ sở không có _gtFine_polygons.json
        base_name = os.path.basename(json_file).split('_gtFine_polygons.json')[0]
        
        # Tạo tên file ảnh đầy đủ
        img_name = f"{base_name}_leftImg8bit.png"
        img_file = os.path.join(LEFTIMG_DIR, rel_dir, img_name)
        
        # Kiểm tra xem file ảnh có tồn tại không
        if os.path.exists(img_file):
            # Đọc nội dung file ảnh
            with open(img_file, 'rb') as img_f:
                img_content = img_f.read()
            
            # Mã hóa nội dung thành base64
            img_base64 = base64.b64encode(img_content)
            
            # Tạo MD5 của chuỗi base64
            md5_hash = hashlib.md5(img_base64).hexdigest()
            
            # Lưu ánh xạ
            md5_to_original[md5_hash] = base_name
            
            # Tạo tên file mới
            new_json_name = f"{md5_hash}_gtFine_polygons.json"
            new_img_name = f"{md5_hash}_leftImg8bit.png"
            
            # Lưu file JSON mới
            json_output = os.path.join(OUTPUT_DIR, "gtFine", rel_dir, new_json_name)
            os.makedirs(os.path.dirname(json_output), exist_ok=True)
            with open(json_output, 'w') as f:
                json.dump(output_data, f, indent=4)
            
            # Lưu file ảnh mới
            img_output = os.path.join(OUTPUT_DIR, "leftImg8bit", rel_dir, new_img_name)
            os.makedirs(os.path.dirname(img_output), exist_ok=True)
            shutil.copy(img_file, img_output)
        else:
            print(f"Không tìm thấy file ảnh: {img_file}")
    
    total_files += 1

# Lưu ánh xạ md5 -> tên file gốc
mapping_file = os.path.join(OUTPUT_DIR, "md5_mapping.json")
with open(mapping_file, 'w') as f:
    json.dump(md5_to_original, f, indent=4)

# In thống kê
print(f"\nĐã xử lý {total_files} files")
print(f"Tìm thấy {files_with_targets} files có đối tượng thuộc lớp cần trích xuất")
for cls, count in stats.items():
    print(f"    - {cls}: {count} đối tượng")
print(f"Ánh xạ MD5 đã được lưu vào {mapping_file}")