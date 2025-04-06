# This Python file uses the following encoding: utf-8
import sys
import os
import cv2
import numpy as np
import json
import requests

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from skimage.metrics import mean_squared_error as ssim
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QComboBox, QWidget, QGraphicsView, QGraphicsScene, QSizePolicy, QCheckBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.center_window()
        self.setFixedSize(self.size())
        self.thresh = 350
        self.start_frames = 3
        self.activity_count = 0
        self.yolo_count = 0
        self.confidence = 0.5
        self.labels = open(os.path.join(base_path, "coco.names")).read().strip().split("\n")
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.model = YOLO(os.path.join(base_path, "yolov8n.pt"))
        self.font_scale = 1
        self.thickness = 1

        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)

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
        self.margin_input = self.findChild(QLineEdit, "margin_input")
        self.token_input = self.findChild(QLineEdit, "token_input")
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.etd_input = self.findChild(QLineEdit, "etd_input")
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
        self.timer_frame = QTimer()
        self.timer_frame.timeout.connect(self.update_frame)
        self.timer_api = QTimer()
        self.timer_api.timeout.connect(self.sendApi)

        self.cap = None # Initialize camera variable
        self.old_frame = None # Initialize old frame variable

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
        self.margin = int(self.margin_input.text().strip()) if self.margin_input.text().strip() else 70
        self.rtsp_url = self.protocol + self.username + ":" + self.password + "@" + self.host + ":" + self.port + "/" + self.stream_path

        if not self.host:
            self.graphics_scene.addText("Please enter host!")
            return

        if not self.port:
            self.graphics_scene.addText("Please enter port!")
            return
        
        if not self.username:
            self.graphics_scene.addText("Please enter username!")
            return
        
        if not self.password:
            self.graphics_scene.addText("Please enter password!")
            return
        
        if not self.stream_path:
            self.graphics_scene.addText("Please enter stream path!")
            return
        
        if not self.protocol:
            self.graphics_scene.addText("Please select protocol!")
            return

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not self.cap.isOpened():
            self.graphics_scene.addText("Not connected to stream!")
            return

        # Set up
        self.timer_frame.start(30)
        self.timer_api.start(int(self.etd_input.text().strip()) if self.etd_input.text().strip() else 1000)
        self.label_counts = { label: 0 for label in self.labels }
        self.start_btn.setText("Start")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.setWindowTitle(self.host)

    def stop_stream(self):
        """Stop stream"""
        if self.cap:
            self.cap.release()
            self.cap = None

        self.timer_frame.stop()
        self.timer_api.stop()
        self.graphics_scene.clear()
        self.graphics_scene.addText("Stream stopped!")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.setWindowTitle("RTSP Stream Viewer")

    def update_frame(self):
        """Update frame from camera"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                h, w, ch = frame.shape
                aspect_ratio = w / h
                if (aspect_ratio > 1.5):
                    res = (256, 144)
                else:
                    res = (200, 200)

                # Resize image, make it grayscale, then blur it
                self.blank = np.zeros((res[1], res[0]), np.uint8)
                resized_frame = cv2.resize(frame, res)
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                final_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)


                # Initialize ssim_val with a default value
                ssim_val = 0

                # Calculate difference between current and previous frame, then get ssim value
                if self.old_frame is not None:
                    diff = cv2.absdiff(final_frame, self.old_frame)
                    result = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
                    ssim_val = int(ssim(result, self.blank))
                self.old_frame = final_frame

                # Check if the ssim value is above the threshold
                if ssim_val > self.thresh:
                    self.activity_count += 1
                    if self.activity_count >= self.start_frames:
                        if self.process_yolo(frame):
                            self.yolo_count += 1
                        else:
                            self.yolo_count = 0

                        if self.yolo_count > 1:
                            self.activity_count = 0
                            self.yolo_count = 0
                else:
                    self.activity_count = 0
                    self.yolo_count = 0                

                # Process YOLO detection
                y_offset = 30
                for label, count in self.label_counts.items():
                    if count > 0:
                        cv2.putText(frame, f"{label} Count: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
                        y_offset += 20

                # Convert the frame to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Convert QImage to QPixmap
                pixmap = QPixmap.fromImage(qimg)

                # Remove the previous pixmap and add the new one
                self.graphics_scene.clear()
                self.graphics_scene.addPixmap(pixmap)

                # Fit the pixmap into the graphics view without scrollbars
                self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.IgnoreAspectRatio)
            else:
                self.graphics_scene.addText("Failed to read frame!")
                self.retry_connection()

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

    def process_yolo(self, frame):
        """Process YOLO detection"""
        results = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        detections = []

        # Virtual lines (x-coordinates)
        line_left = self.margin
        line_right = frame.shape[1] - self.margin

        # Vẽ hai đường ảo
        cv2.line(frame, (line_left, 0), (line_left, frame.shape[0]), (0, 255, 0), 2)
        cv2.line(frame, (line_right, 0), (line_right, frame.shape[0]), (0, 255, 0), 2)

        # Loop over the detections
        for data in results.boxes.data.tolist():
            # Get the bounding box coordinates, confidence, and class id
            xmin, ymin, xmax, ymax, confidence, class_id = data

            # Converting the coordinates and the class id to integers
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            class_id = int(class_id)

            # Check if the person crosses the virtual lines
            checkedList = [checkbox.objectName().replace("_chkbox", "").replace("_", " ") for checkbox in self.checkboxes if checkbox.isChecked()]
            if self.labels[class_id] in checkedList:
                detections.append(([xmin, ymin, xmax - xmin, ymax - ymin], confidence, self.labels[class_id]))

            # Draw a bounding box rectangle and label on the image
            color = [int(c) for c in self.colors[class_id]]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=color, thickness=self.thickness)
            text = f"{self.labels[class_id]}: {confidence:.2f}"

            # Calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.font_scale, thickness=self.thickness)[0]
            text_offset_x = xmin
            text_offset_y = ymin - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = frame.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)

            # Add opacity (transparency to the box)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Now put the text (label: confidence %)
            cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.font_scale, color=(0, 0, 0), thickness=self.thickness)

        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Loop over tracked objects
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            center_x = (x1 + x2) // 2

            if line_left - 5 < center_x < line_left + 5:
                self.label_counts[track.get_det_class()] += 1
            elif line_right - 5 < center_x < line_right + 5:
                self.label_counts[track.get_det_class()] += 1

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return len(tracks) > 0

    def save_settings(self):
        """Save settings to a JSON file."""
        settings = {
            "host": self.host_input.text().strip(),
            "port": self.port_input.text().strip(),
            "username": self.user_input.text().strip(),
            "password": self.pass_input.text().strip(),
            "protocol": self.protocol_cbb.currentText(),
            "stream_path": self.stream_path_input.text().strip(),
            "margin": self.margin_input.text().strip(),
            "api": self.api_input.text().strip(),
            "token": self.token_input.text().strip(),
            "etd": self.etd_input.text().strip(),
            "checked_labels": [checkbox.objectName().replace("_chkbox", "").replace("_", " ") for checkbox in self.checkboxes if checkbox.isChecked()],
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
            self.margin_input.setText(settings.get("margin", "50"))
            self.api_input.setText(settings.get("api", ""))
            self.token_input.setText(settings.get("token", ""))
            checked_labels = settings.get("checked_labels", [])
            self.etd_input.setText(settings.get("etd", "5000"))
            for checkbox in self.checkboxes:
                label = checkbox.objectName().replace("_chkbox", "").replace("_", " ")
                checkbox.setChecked(label in checked_labels)
            print("Settings loaded!")
        except FileNotFoundError:
            print("Settings file not found. Using default values.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
