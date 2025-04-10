import os
import numpy as np
import cv2
import time
import sys

from core.rtsp import RTSPReader
from skimage.metrics import mean_squared_error as ssim
from ultralytics import YOLO

if getattr(sys, 'frozen', False):  # Kiểm tra nếu đang chạy từ file .exe
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# Tải model YOLO
model = YOLO("yolov8n.pt")
CONFIDENCE = 0.25
blank = None
old_frame = None
activity_count = 0
start_frames = 3
thresh = 350
yolo_count = 0
labels = open(os.path.join(base_path, "coco.names")).read().strip().split("\n")

# Dictionary để theo dõi hướng di chuyển
track_memory = {}

def process_yolo(frame):
    """Phát hiện và theo dõi đối tượng trong frame bằng YOLOv8 với ByteTrack tích hợp"""
    global CONFIDENCE

    # Sử dụng track() thay cho predict() với ByteTrack tích hợp sẵn
    results = model.track(frame, conf=CONFIDENCE, classes=[1, 3], verbose=False, persist=True, tracker="botsort.yaml")[0]
    
    if len(results) == 0 or results.boxes is None:
        return False

    # Theo dõi đối tượng
    boxes = results.boxes
    object_found = len(boxes) > 0

    # Xử lý các đối tượng được theo dõi
    for box in boxes:
        # Chỉ xử lý các box có track ID
        if box.id is None:
            continue
            
        # Lấy ID và class
        track_id = int(box.id.item())
        class_id = int(box.cls.item())
        
        # print(f"Class ID: {class_id}")
        
        # Lấy tọa độ bounding box
        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        
        # Tính tọa độ trung tâm
        centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Khởi tạo track nếu chưa có trong memory
        if track_id not in track_memory:
            track_memory[track_id] = {
                'previous_centroid': centroid,
                'previous_direction': None
            }
            continue
            
        # Lấy thông tin trước đó
        prev_centroid = track_memory[track_id]['previous_centroid']
        
        # Tính vector chuyển động
        dx = centroid[0] - prev_centroid[0]
        dy = centroid[1] - prev_centroid[1]
        
        # Xác định hướng di chuyển với ngưỡng
        MIN_MOVEMENT = 5  # pixels
        if abs(dy) < MIN_MOVEMENT:
            # Nếu di chuyển ít, giữ hướng trước đó
            direction = track_memory[track_id]['previous_direction']
            if direction is None:  # Nếu không có hướng trước đó
                direction = "unknown"
        elif dy > 0:
            direction = "out"
        else:
            direction = "in"
            
        # if direction != "unknown":
        print(f"Track ID {track_id} Class {labels[class_id]} moving {direction}")
        
        # Cập nhật thông tin track
        track_memory[track_id]['previous_centroid'] = centroid
        track_memory[track_id]['previous_direction'] = direction

    return object_found

def display_rtsp_with_pyav(rtsp_url, window_name="RTSP Stream"):
    # Khai báo sử dụng biến toàn cục
    global blank, old_frame, thresh, activity_count, start_frames, yolo_count

    # Initialize the RTSP reader
    reader = RTSPReader(rtsp_url)
    
    # Connect to the RTSP stream
    if not reader.connect():
        print("Failed to connect to RTSP stream. Exiting.")
        return
    
    # Start reading frames in the background
    reader.start()
    
    # Create a window for display
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("Press 'q' to exit")
    
    try:
        while True:
            # Get the latest frame
            frame = reader.get_frame()
            
            if frame is not None:
                ssim_val = 0

                if frame.shape[1]/frame.shape[0] > 1.55:
                    res = (256,144) # 16:9
                else:
                    res = (216,162) # 4:3

                # Đo thời gian xử lý
                start_time = time.time()

                # Resize image, make it grayscale, then blur it
                if blank is None:
                    blank = np.zeros((res[1], res[0]), np.uint8)

                resized_frame = cv2.resize(frame, res)
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                final_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)

                # Calculate difference between current and previous frame, then get ssim value
                if old_frame is None:
                    old_frame = final_frame
                else:
                    diff = cv2.absdiff(final_frame, old_frame)
                    result = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
                    ssim_val = int(ssim(result, blank))
                    old_frame = final_frame


                if ssim_val > thresh:
                    process_yolo(frame.copy())
                #     activity_count += 1
                #     if activity_count >= start_frames:
                #         if process_yolo(frame.copy()):
                #             yolo_count += 1
                #         else:
                #             yolo_count = 0

                #         if yolo_count > 1:
                #             activity_count = 0
                #             yolo_count = 0
                    
                # else:
                #     activity_count = 0
                #     yolo_count = 0

                # Kết thúc đo thời gian
                end_time = time.time()
                process_time = (end_time - start_time) * 1000  # Chuyển sang milliseconds
                # print(f"Processing time: {process_time:.2f} ms")

                # Display the frame
                cv2.imshow(window_name, frame.copy())
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # No frame available yet
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Clean up
        reader.stop()
        cv2.destroyAllWindows()
        print("Resources released")


if __name__ == "__main__":
    # Get RTSP URL from command line or user input
    if len(sys.argv) > 1:
        rtsp_url = sys.argv[1]
    else:
        # rtsp_url = 'rtsp://admin:JDQOPY@cam110.ddns.net:554/ch1/main'
        rtsp_url = 'rtsp://admin:Tsicongnghe@123@dongbacdh.autoddns.com:554/profile1'
        # rtsp_url = input("Enter RTSP URL (e.g., rtsp://username:password@ip:port/path): ")
    
    display_rtsp_with_pyav(rtsp_url)