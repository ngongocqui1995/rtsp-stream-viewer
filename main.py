import numpy as np
import cv2
import time
import sys

from core.rtsp import RTSPReader
from skimage.metrics import mean_squared_error as ssim
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
CONFIDENCE = 0.5
blank = None
old_frame = None
thresh = 500

def process_yolo(frame):
    # Khai báo sử dụng biến toàn cục
    global CONFIDENCE

    print("Processing YOLO...")
    results = model.predict(frame, conf=CONFIDENCE, classes=(1, 3), verbose=False)[0]
    object_found = False

    # Loop over the detections
    for data in results.boxes.data.tolist():
        # Get the bounding box coordinates, confidence, and class id
        xmin, ymin, xmax, ymax, confidence, class_id = data

        # Converting the coordinates and the class id to integers
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        class_id = int(class_id)

        print(f"Class ID: {class_id}")

    #     if labels[class_id] in yolo_list:
    #         object_found = True

    #     if labels[class_id] == "person":
    #         person_count += 1

    #     # Draw a bounding box rectangle and label on the image
    #     color = [int(c) for c in colors[class_id]]
    #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
    #     text = f"{labels[class_id]}: {confidence:.2f}"
    #     # Calculate text width & height to draw the transparent boxes as background of the text
    #     (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
    #     text_offset_x = xmin
    #     text_offset_y = ymin - 5
    #     box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
    #     overlay = img.copy()
    #     cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
    #     # Add opacity (transparency to the box)
    #     img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    #     # Now put the text (label: confidence %)
    #     cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #         fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    # frame_height, frame_width, _ = img.shape
    # cv2.putText(img, f"People Count: {person_count}", (frame_width - 270, 30), cv2.FONT_HERSHEY_SIMPLEX,
    #         fontScale=1, color=(0, 255, 0), thickness=2)
    return object_found

def display_rtsp_with_pyav(rtsp_url, window_name="RTSP Stream"):
    # Khai báo sử dụng biến toàn cục
    global blank, old_frame, thresh

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
                    process_yolo(frame)

                    # Kết thúc đo thời gian
                    end_time = time.time()
                    process_time = (end_time - start_time) * 1000  # Chuyển sang milliseconds
                    print(f"Processing time: {process_time:.2f} ms")


                # Display the frame
                cv2.imshow(window_name, frame)
                
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
        rtsp_url = 'rtsp://admin:JDQOPY@cam110.ddns.net:554/ch1/main'
        # rtsp_url = 'rtsp://admin:Tsicongnghe@123@dongbacdh.autoddns.com:554/profile1'
        # rtsp_url = input("Enter RTSP URL (e.g., rtsp://username:password@ip:port/path): ")
    
    display_rtsp_with_pyav(rtsp_url)