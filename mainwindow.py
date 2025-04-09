# This Python file uses the following encoding: utf-8
import sys
import os
import cv2
import numpy as np
import json
import requests
import threading
import queue

from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from skimage.metrics import mean_squared_error as ssim
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QComboBox, QWidget, QGraphicsView, QGraphicsScene, QSizePolicy, QCheckBox, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer, Signal

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_MainWindow

if getattr(sys, 'frozen', False):  # Kiểm tra nếu đang chạy từ file .exe
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
    
class MainWindow(QMainWindow):
    frame_ready = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.center_window()
        self.setFixedSize(self.size())
        self.frame_ready.connect(self.update_ui)
        self.labels = open(os.path.join(base_path, "coco.names")).read().strip().split("\n")
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.old_frame = None
        self.thresh = 350

        # Định nghĩa đường chéo cho đếm đối tượng
        self.line_start = None  # Sẽ được cập nhật khi có frame đầu tiên
        self.line_end = None    # Sẽ được cập nhật khi có frame đầu tiên
        
        # Định nghĩa tỷ lệ để tính toán đường chéo
        self.line_start_x_offset = 400  # offset từ giữa màn hình
        self.line_start_y = 0           # y của điểm bắt đầu
        self.line_end_x = 0             # x của điểm kết thúc
        self.line_end_y_offset = 200    # offset từ đáy màn hình

        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=30)
        self.counted_tracks = {}
        self.worker_counters = {1: 0, 2: 0}

        self.ai_cbb = self.findChild(QComboBox, "ai_cbb")
        self.ai_cbb.currentTextChanged.connect(self.update_model)

        self.etd_input = self.findChild(QLineEdit, "etd_input")
        self.etd_input.textChanged.connect(self.update_etd)

        self.confidence_cbb = self.findChild(QComboBox, "confidence_cbb")
        self.confidence_cbb.currentTextChanged.connect(self.update_confidence)
        self.confidence = float(self.confidence_cbb.currentText())

        # Initialize UI components
        self.start_btn = self.findChild(QPushButton, "start_btn")
        self.stop_btn = self.findChild(QPushButton, "stop_btn")
        self.host_input = self.findChild(QLineEdit, "host_input")
        self.port_input = self.findChild(QLineEdit, "port_input")
        self.user_input = self.findChild(QLineEdit, "user_input")
        self.pass_input = self.findChild(QLineEdit, "pass_input")
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.protocol_cbb = self.findChild(QComboBox, "protocol_cbb")
        self.stream_path_input = self.findChild(QLineEdit, "stream_path_input")
        self.api_input = self.findChild(QLineEdit, "api_input")
        self.token_input = self.findChild(QLineEdit, "token_input")
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.yolo_label = self.findChild(QLabel, "yolo_label")
        self.read_label = self.findChild(QLabel, "read_label")
        self.worker_1_label = self.findChild(QLabel, "worker_1_label")
        self.worker_2_label = self.findChild(QLabel, "worker_2_label")
        self.reset_detection_btn = self.findChild(QPushButton, "reset_detection_btn")
        self.reset_detection_btn.clicked.connect(self.reset_detection)
        self.frame_cbb = self.findChild(QComboBox, "frame_cbb")
        self.checkboxes = []
        for label in self.labels:
            key = label.replace(" ", "_") + "_chkbox"
            self.checkboxes.append(self.findChild(QCheckBox, key))
        self.graphics_scene = QGraphicsScene()
        self.graphics_view = self.findChild(QGraphicsView, "graphics_view")
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Connect actions to buttons
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)

        # Timer update
        self.timer_api = QTimer()
        self.timer_api.timeout.connect(self.sendApi)

        self.cap = None # Initialize camera variable

        # Load settings
        self.load_settings()

    def sendApi(self):
        """Send API request"""
        url = self.api_input.text().strip()
        token = self.token_input.text().strip()

        if url:
            try:
                headers = {"Authorization": f"Bearer {token}"}
                checked = [checkbox.objectName().replace("_chkbox", "").replace("_", " ") for checkbox in self.checkboxes if checkbox.isChecked()]
                body = {key.replace(" ", "_"): value for key, value in self.label_counts.items() if key in checked}
                requests.post(url, json=body, headers=headers)
                self.label_counts = { label: 0 for label in self.labels }
            except requests.exceptions.RequestException as e:
                print(f"API request failed: {e}")
            
    def start_stream(self):
        """Start stream from RTSP URL"""
        if self.cap:
            self.cap.release()  # Release the previous capture if it exists

        self.host = self.host_input.text().strip()
        self.port = self.port_input.text().strip()
        self.username = self.user_input.text().strip()
        self.password = self.pass_input.text().strip()
        self.protocol = self.protocol_cbb.currentText()
        self.stream_path = self.stream_path_input.text().strip()

        if self.username and self.password:
            self.rtsp_url = f"{self.protocol}{self.username}:{self.password}@{self.host}:{self.port}/{self.stream_path}"
        else:
            self.rtsp_url = f"{self.protocol}{self.host}:{self.port}/{self.stream_path}"

        if not self.host:
            self.graphics_scene.addText("Please enter host!")
            return

        if not self.port:
            self.graphics_scene.addText("Please enter port!")
            return
        
        if not self.stream_path:
            self.graphics_scene.addText("Please enter stream path!")
            return
        
        if not self.protocol:
            self.graphics_scene.addText("Please select protocol!")
            return

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.graphics_scene.addText("Not connected to stream!")
            return
        
        # Tạo hàng đợi để lưu trữ khung hình
        self.frame_queue = queue.Queue(maxsize=500)

        # Hàng đợi để lưu trữ các khung hình cần xử lý
        self.yolo_queue = queue.Queue(maxsize=500)

        # ThreadPoolExecutor với 2 worker để xử lý YOLO song song
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Khởi chạy luồng đọc khung hình
        self.running = True
        self.read_thread = threading.Thread(target=self.read_frames, daemon=True)
        self.read_thread.start()

        # Khởi chạy 2 worker để xử lý YOLO song song
        self.executor.submit(self.process_yolo_worker, 1)  # Worker 1
        self.executor.submit(self.process_yolo_worker, 2)  # Worker 2

        # Khởi chạy luồng xử lý khung hình
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.process_thread.start()

        # Set up
        self.timer_api.start(int(self.etd_input.text().strip()) if self.etd_input.text().strip() else 1000)
        self.start_btn.setText("Start")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.host_input.setEnabled(False)
        self.port_input.setEnabled(False)
        self.user_input.setEnabled(False)
        self.pass_input.setEnabled(False)
        self.protocol_cbb.setEnabled(False)
        self.stream_path_input.setEnabled(False)
        self.reset_detection_btn.setEnabled(False)
        for checkbox in self.checkboxes:
            checkbox.setEnabled(False)
        self.update_model(self.ai_cbb.currentText())
        self.setWindowTitle(self.host)

    def read_frames(self):
        """Read frames from the camera and put them into the queue"""
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Đưa khung hình vào hàng đợi (nếu hàng đợi đầy, bỏ qua khung hình)
                try:
                    self.frame_queue.put(frame, timeout=1)
                    self.yolo_label.setText(f"Yolo: {self.yolo_queue.qsize()}")
                except queue.Full:
                    pass
            else:
                self.retry_connection()
                break

    def stop_stream(self):
        """Stop stream"""
        self.running = False  # Dừng luồng đọc và xử lý khung hình
        if hasattr(self, 'read_thread') and self.read_thread.is_alive():
            self.read_thread.join()  # Chờ luồng đọc kết thúc
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join()  # Chờ luồng xử lý kết thúc

        # Shutdown ThreadPoolExecutor
        self.executor.shutdown(wait=True)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.timer_api.stop()
        self.graphics_scene.clear()
        self.graphics_scene.addText("Stream stopped!")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.host_input.setEnabled(True)
        self.port_input.setEnabled(True)
        self.user_input.setEnabled(True)
        self.pass_input.setEnabled(True)
        self.protocol_cbb.setEnabled(True)
        self.stream_path_input.setEnabled(True)
        self.reset_detection_btn.setEnabled(True)
        for checkbox in self.checkboxes:
            checkbox.setEnabled(True)
            checkbox.setText(checkbox.objectName().replace("_chkbox", "").replace("_", " "))
        self.setWindowTitle("RTSP Stream Viewer")

    def process_frame(self, frame):
        """Process a single frame and update the UI"""
        try:
            # if frame.shape[1]/frame.shape[0] > 1.55:
            #     res = (256,144) # 16:9
            # else:
            #     res = (216,162) # 4:3

            # blank = np.zeros((res[1],res[0]), np.uint8)
            # resized_frame = cv2.resize(frame, res)
            # gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            # final_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)

            # if self.old_frame is None:
            #     self.old_frame = final_frame
            #     return

            # # Calculate difference between current and previous frame
            # diff = cv2.absdiff(final_frame, self.old_frame)
            # result = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
            # ssim_val = int(ssim(result, blank))
            # self.old_frame = final_frame

            # if ssim_val > self.thresh:
            #     # Đưa khung hình đã được tăng cường vào queue để YOLO xử lý
            #     self.yolo_queue.put(frame, timeout=1)
            
            # Hiển thị lên giao diện người dùng
            self.read_label.setText(f"Read: {self.frame_queue.qsize()}")
            self.frame_ready.emit(frame)  # Hiển thị khung hình đã được tăng cường
        except queue.Full:
            pass

    def update_ui(self, frame):
        """Update the UI with the processed frame"""
        h, w, ch = frame.shape
        
        # Cập nhật tọa độ đường chéo theo kích thước frame
        self.line_start = (w // 2 + self.line_start_x_offset, self.line_start_y)
        self.line_end = (self.line_end_x, h - self.line_end_y_offset)
        
        # Vẽ đường chéo để dễ quan sát
        cv2.line(frame, self.line_start, self.line_end, (0, 255, 0), 2)

        # # Vẽ hình chữ nhật ROI lên frame để debug (chỉ để kiểm tra)
        # if self.min_x and self.min_y and self.max_x and self.max_y:
        #     cv2.rectangle(frame, (self.min_x, self.min_y), (self.max_x, self.max_y), (0, 0, 255), 2)

        for label, count in self.label_counts.items():
            self.checkboxes[self.labels.index(label)].setText(f"({count}){label}")

        self.worker_1_label.setText(f"Worker 1: {self.worker_counters[1]}")
        self.worker_2_label.setText(f"Worker 2: {self.worker_counters[2]}")

        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(qimg)

        self.graphics_scene.clear()
        self.graphics_scene.addPixmap(pixmap)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.IgnoreAspectRatio)

    def closeEvent(self, event):
        """Close event handler"""
        self.save_settings()  # Save settings before closing
        self.stop_stream()
        event.accept()

    def retry_connection(self):
        """Retry connection to the stream"""
        self.stop_stream()
        self.start_btn.setText("Auto Reconnect")
        self.start_btn.setEnabled(False)
        QTimer.singleShot(3000, self.start_stream)

    def center_window(self):
        """Center the window on the screen"""
        screen = self.screen().geometry()
        window_size = self.geometry()

        center_x = (screen.width() - window_size.width()) // 2
        center_y = (screen.height() - window_size.height()) // 2

        self.move(center_x, center_y)

    def process_frames(self):
        """Process frames from the queue"""
        while self.running:
            try:
                # Lấy khung hình từ hàng đợi
                frame = self.frame_queue.get(timeout=1)
                self.process_frame(frame)
            except queue.Empty:
                pass
            
    def process_yolo_worker(self, worker_id):
        """Worker function to process frames from the YOLO queue in batches"""
        batch = []
        original_frames = []

        while self.running:
            try:
                batch_size = int(self.frame_cbb.currentText())
                
                # Lấy khung hình từ hàng đợi với timeout
                frame = self.yolo_queue.get(timeout=1)

                self.worker_counters[worker_id] = len(batch) + 1

                # Lấy kích thước frame
                h, w, _ = frame.shape
                
                # Cập nhật tọa độ đường chéo
                self.line_start = (w // 2 + self.line_start_x_offset, self.line_start_y)
                self.line_end = (self.line_end_x, h - self.line_end_y_offset)
                
                # Tính toán vùng 30% xung quanh đường chéo
                margin = int(w * 0.15)  # 15% mỗi bên của đường chéo, tổng 30%
                
                # Tạo vùng hình chữ nhật bao quanh đường chéo
                min_x = max(0, min(self.line_start[0], self.line_end[0]) - margin)
                max_x = min(w, max(self.line_start[0], self.line_end[0]) + margin)
                min_y = max(0, min(self.line_start[1], self.line_end[1]) - margin)
                max_y = min(h, max(self.line_start[1], self.line_end[1]) + margin)
                self.min_x = min_x
                self.min_y = min_y
                self.max_x = max_x
                self.max_y = max_y
                
                # Cắt vùng trung tâm theo đường chéo
                roi = frame[min_y:max_y, min_x:max_x]
                
                # Kiểm tra ROI hợp lệ
                if roi.size == 0 or roi is None:
                    continue
                    
                # Thêm vùng trung tâm vào batch
                batch.append(roi)
                original_frames.append(frame)

                # Phần code còn lại giữ nguyên...
                if len(batch) == batch_size:
                    try:
                        checkedList = [checkbox.objectName().replace("_chkbox", "").replace("_", " ") 
                                     for checkbox in self.checkboxes if checkbox.isChecked()]
                        classes = [self.labels.index(label) for label in checkedList if label in self.labels]
                        
                        # Thêm timeout cho predict bằng cách sử dụng một thread riêng
                        predict_result = [None]
                        predict_error = [None]
                        
                        def predict_with_timeout():
                            try:
                                predict_result[0] = self.model.predict(
                                    batch, conf=self.confidence, classes=classes, verbose=False)
                            except Exception as e:
                                predict_error[0] = e
                        
                        predict_thread = threading.Thread(target=predict_with_timeout)
                        predict_thread.daemon = True
                        predict_thread.start()
                        
                        # Chờ tối đa batch_size giây
                        predict_thread.join(batch_size)
                        
                        if predict_thread.is_alive():
                            # Nếu thread vẫn đang chạy sau batch_size giây, coi như bị timeout
                            print("YOLO prediction timeout, skipping batch")
                            batch = []
                            original_frames = []
                            continue
                        
                        if predict_error[0] is not None:
                            # Có lỗi xảy ra
                            print(f"Error in YOLO prediction: {predict_error[0]}")
                            batch = []
                            original_frames = []
                            continue
                            
                        results = predict_result[0]
                        
                        # Xử lý kết quả YOLO
                        for roi, result, original_frame in zip(batch, results, original_frames):
                            try:
                                self.process_yolo_result(original_frame, result, (min_x, min_y))
                            except Exception as e:
                                print(f"Error processing YOLO result: {e}")
                                
                        batch = []  # Reset batch sau khi xử lý
                        original_frames = []
                        
                    except Exception as e:
                        print(f"Error preparing YOLO prediction: {e}")
                        batch = []  # Reset batch khi có lỗi
                        original_frames = []

                self.worker_counters[worker_id] = len(batch) + 1
                        
            except queue.Empty:
                # Xử lý các khung hình còn lại trong batch nếu đã chờ quá lâu
                if batch and len(batch) >= 3:  # Xử lý nếu có ít nhất 3 frame trong batch
                    try:
                        checkedList = [checkbox.objectName().replace("_chkbox", "").replace("_", " ") 
                                    for checkbox in self.checkboxes if checkbox.isChecked()]
                        classes = [self.labels.index(label) for label in checkedList if label in self.labels]
                        results = self.model.predict(batch, conf=self.confidence, classes=classes, verbose=False)
                    
                        # Xử lý kết quả YOLO
                        for roi, result, original_frame in zip(batch, results, original_frames):
                            self.process_yolo_result(original_frame, result, (min_x, min_y))
                            
                    except Exception as e:
                        print(f"Error processing remaining batch: {e}")
                    
                    batch = []  # Reset batch
                    original_frames = []
                pass
            except Exception as e:
                print(f"Unexpected error in YOLO worker: {e}")
                # Reset batch khi có lỗi không xác định
                batch = []
                original_frames = []

        # Xử lý các khung hình còn lại trong batch khi dừng
        if batch:
            try:
                results = self.model.predict(batch, conf=self.confidence, verbose=False)
                for roi, result, original_frame in zip(batch, results, original_frames):
                    self.process_yolo_result(original_frame, result, (min_x, min_y))
            except Exception as e:
                print(f"Error processing final batch: {e}")
        
        self.worker_counters[worker_id] = len(batch) + 1

    def process_yolo_result(self, frame, result, offset):
        """Process YOLO detection result for a single frame"""
        offset_x, offset_y = offset  # Tọa độ gốc của vùng trung tâm

        # Lấy kích thước frame
        h, w = frame.shape[:2]
        
        # Cập nhật tọa độ đường chéo
        self.line_start = (w // 2 + self.line_start_x_offset, self.line_start_y)
        self.line_end = (self.line_end_x, h - self.line_end_y_offset)
        
        # Phần còn lại giữ nguyên cho đến phần xử lý track
        checkedList = [checkbox.objectName().replace("_chkbox", "").replace("_", " ") 
                      for checkbox in self.checkboxes if checkbox.isChecked()]
        
        detections = []
        
        if result is None:
            return
        
        is_obb_model = "obb" in self.ai_cbb.currentText()
        
        try:
            if is_obb_model:
                # Xử lý kết quả từ mô hình OBB
                if hasattr(result, 'obb') and result.obb is not None and hasattr(result.obb, 'data'):
                    for data in result.obb.data.tolist():
                        # OBB format: [cx, cy, w, h, angle, conf, cls]
                        if len(data) >= 7:
                            cx, cy, w, h, angle, confidence, class_id = data[:7]
                            class_id = int(class_id)
                            
                            # Chuyển đổi tọa độ tâm thành góc trên bên trái để tương thích với DeepSORT
                            xmin = int(cx - w/2) + offset_x
                            ymin = int(cy - h/2) + offset_y
                            width = int(w)
                            height = int(h)
                            
                            # Chỉ xử lý các đối tượng đã được chọn
                            if class_id < len(self.labels):
                                class_name = self.labels[class_id]
                                if class_name in checkedList:
                                    detections.append(([xmin, ymin, width, height], confidence, class_name))
            else:
                # Xử lý kết quả từ mô hình YOLOv8 tiêu chuẩn
                if hasattr(result, 'boxes') and result.boxes is not None and hasattr(result.boxes, 'data'):
                    for data in result.boxes.data.tolist():
                        # Get the bounding box coordinates, confidence, and class id
                        xmin, ymin, xmax, ymax, confidence, class_id = data
                        
                        # Chuyển đổi tọa độ từ vùng trung tâm sang khung hình gốc
                        xmin += offset_x
                        xmax += offset_x
                        ymin += offset_y
                        ymax += offset_y
                        
                        # Converting the coordinates and the class id to integers
                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                        class_id = int(class_id)
                        
                        # Chỉ xử lý các đối tượng đã được chọn
                        if class_id < len(self.labels):
                            class_name = self.labels[class_id]
                            if class_name in checkedList:
                                detections.append(([xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_name))
                                
        except Exception as e:
            print(f"Error parsing YOLO results: {e}")
            return
            
        # Nếu không có detections, không cần xử lý thêm
        if not detections:
            return
            
        # Cập nhật tracks
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Lấy danh sách track_id hiện tại
        current_track_ids = {track.track_id for track in tracks if track.is_confirmed()}
        
        # Cập nhật hoặc thêm thời gian tồn tại cho tracks
        for track_id in list(self.counted_tracks.keys()):
            if track_id in current_track_ids:
                # Reset bộ đếm nếu track vẫn xuất hiện
                if 'frames_missing' in self.counted_tracks[track_id]:
                    self.counted_tracks[track_id]['frames_missing'] = 0
            else:
                # Tăng bộ đếm nếu track không xuất hiện
                if 'frames_missing' not in self.counted_tracks[track_id]:
                    self.counted_tracks[track_id]['frames_missing'] = 1
                else:
                    self.counted_tracks[track_id]['frames_missing'] += 1
                
                # Chỉ xóa khi track mất đi quá số frame quy định
                if self.counted_tracks[track_id]['frames_missing'] > 200:  # Giữ track trong 30 frame
                    del self.counted_tracks[track_id]
        
        # Xử lý từng track
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            # Lấy track_id để xác định đối tượng duy nhất
            track_id = track.track_id
            
            # Lấy tọa độ bounding box
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = map(int, ltrb)
            
            # Tính tọa độ tâm (cần cả x và y)
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            
            # Kiểm tra xem track đã được đếm chưa
            if track_id not in self.counted_tracks:
                self.counted_tracks[track_id] = {'counted': False, 'prev_x': center_x, 'prev_y': center_y}
            
            # Lấy vị trí trước đó
            prev_x = self.counted_tracks[track_id]['prev_x']
            prev_y = self.counted_tracks[track_id]['prev_y']
            counted = self.counted_tracks[track_id]['counted']
            
            # Hàm kiểm tra điểm có nằm bên trái đường chéo không
            def is_on_left_side(x, y):
                """Kiểm tra điểm (x,y) có nằm bên trái đường thẳng không"""
                x1, y1 = self.line_start
                x2, y2 = self.line_end
                return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) < 0
            
            # Xác định vị trí trước và hiện tại so với đường chéo
            was_on_left = is_on_left_side(prev_x, prev_y)
            now_on_left = is_on_left_side(center_x, center_y)
            
            # Nếu vị trí trước và hiện tại khác nhau, tức là đã vượt qua đường chéo
            if not counted and was_on_left != now_on_left:
                # Lấy tên lớp của đối tượng
                class_name = track.get_det_class()
                
                # Tăng biến đếm cho loại đối tượng
                if class_name in self.label_counts:
                    self.label_counts[class_name] += 1
                
                # Đánh dấu đã đếm
                self.counted_tracks[track_id]['counted'] = True
            
            # Cập nhật vị trí trước đó
            self.counted_tracks[track_id]['prev_x'] = center_x
            self.counted_tracks[track_id]['prev_y'] = center_y
            
        # Xóa các counted_tracks quá cũ
        self.counted_tracks = {k: v for k, v in self.counted_tracks.items() if k in current_track_ids}

    def save_settings(self):
        """Save settings to a JSON file."""
        settings = {
            "host": self.host_input.text().strip(),
            "port": self.port_input.text().strip(),
            "username": self.user_input.text().strip(),
            "password": self.pass_input.text().strip(),
            "protocol": self.protocol_cbb.currentText(),
            "stream_path": self.stream_path_input.text().strip(),
            "api": self.api_input.text().strip(),
            "token": self.token_input.text().strip(),
            "etd": self.etd_input.text().strip(),
            "ai": self.ai_cbb.currentText(),
            "checked_labels": [checkbox.objectName().replace("_chkbox", "").replace("_", " ") for checkbox in self.checkboxes if checkbox.isChecked()],
            "label_counts": {label: self.label_counts[label] for label in self.labels if label in self.label_counts},
            "confidence": self.confidence_cbb.currentText(),
            "frame": self.frame_cbb.currentText()
        }

        with open(os.path.join(base_path, "settings.json"), "w") as file:
            json.dump(settings, file, indent=4)
        print("Settings saved!")

    def load_settings(self):
        """Load settings from a JSON file."""
        try:
            with open(os.path.join(base_path, "settings.json"), "r") as file:
                settings = json.load(file)

            self.host_input.setText(settings.get("host", ""))
            self.port_input.setText(settings.get("port", ""))
            self.user_input.setText(settings.get("username", ""))
            self.pass_input.setText(settings.get("password", ""))
            self.protocol_cbb.setCurrentText(settings.get("protocol", ""))
            self.stream_path_input.setText(settings.get("stream_path", ""))
            self.api_input.setText(settings.get("api", ""))
            self.ai_cbb.setCurrentText(settings.get("ai", "yolov8n.pt"))
            self.token_input.setText(settings.get("token", ""))
            self.etd_input.setText(settings.get("etd", "5000"))
            self.confidence_cbb.setCurrentText(settings.get("confidence", "0.3"))
            self.frame_cbb.setCurrentText(settings.get("frame", "5"))
            self.label_counts = settings.get("label_counts", {label: 0 for label in self.labels})
            checked_labels = settings.get("checked_labels", [])
            for checkbox in self.checkboxes:
                label = checkbox.objectName().replace("_chkbox", "").replace("_", " ")
                checkbox.setChecked(label in checked_labels)
            print("Settings loaded!")
        except FileNotFoundError:
            print("Settings file not found. Using default values.")

    def reset_detection(self):
        """Reset detection counts"""
        self.label_counts = {label: 0 for label in self.labels}
        for checkbox in self.checkboxes:
            checkbox.setText(checkbox.objectName().replace("_chkbox", "").replace("_", " "))
        print("Detection counts reset!")

    def update_confidence(self, confidence):
        """Update confidence when selection changes"""
        self.confidence = float(confidence)
        print(f"Confidence changed to {confidence}")

    def update_etd(self, etd):
        """Update ETD when input changes"""
        self.timer_api.setInterval(int(etd) if etd else 1000)
        print(f"ETD changed to {etd}")

    def update_model(self, model_name):
        """Update model when selection changes"""
        self.model = YOLO(model_name)
        # self.model = YOLO("yolov8x-bike-motorcycle.pt")
        print(f"Model changed to {model_name}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
