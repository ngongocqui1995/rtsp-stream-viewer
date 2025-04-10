import time
import av
from av.error import InvalidDataError

from threading import Thread, Event

class RTSPReader:
    def __init__(self, rtsp_url, options=None):
        self.rtsp_url = rtsp_url
        self.options = options or {}
        self.container = None
        self.video_stream = None
        self.stop_event = Event()
        self.frame = None
        self.connected = False
        self.reconnect_delay = 5
        
    def connect(self):
        # Thử kết nối với UDP trước, sau đó TCP nếu thất bại
        return self._try_connect('tcp') or self._try_connect('udp')

    def _try_connect(self, transport_protocol):
        try:
            print(f"Trying to connect with {transport_protocol.upper()}...")
            
            # Set up connection options for RTSP
            input_options = {
                'rtsp_transport': transport_protocol,
                'fflags': 'nobuffer+discardcorrupt',
                'flags': 'low_delay',
                'analyzeduration': '0',
                'probesize': '1024',
                'stimeout': '5000000',  # Tăng timeout lên 5 giây
                'timeout': '5000000',
                'max_delay': '500000',
                'reorder_queue_size': '0',
                'buffer_size': '16384'
            }
            
            # Update with any user-provided options
            input_options.update(self.options)
            
            # Open the RTSP stream
            self.container = av.open(self.rtsp_url, options=input_options)
            
            # Get the video stream
            streams = [stream for stream in self.container.streams if stream.type == 'video']
            if not streams:
                raise ValueError("No video stream found in the RTSP source")
            
            self.video_stream = streams[0]
            self.connected = True
            print(f"Successfully connected to {self.rtsp_url} using {transport_protocol.upper()}")
            return True
            
        except Exception as e:
            print(f"Connection error with {transport_protocol.upper()}: {e}")
            if self.container:
                self.container.close()
                self.container = None
            self.connected = False
            return False
    
    def start(self):
        """Start reading frames in a background thread"""
        self.stop_event.clear()
        Thread(target=self._read_frames, daemon=True).start()
    
    def _read_frames(self):
        """Background thread to continuously read frames"""
        while not self.stop_event.is_set():
            if not self.connected:
                print(f"Not connected. Attempting to reconnect in {self.reconnect_delay} seconds...")
                time.sleep(self.reconnect_delay)
                if self.connect():
                    continue
                else:
                    # Increase reconnect delay (with a maximum)
                    self.reconnect_delay = min(self.reconnect_delay * 1.5, 30)
                    continue
            
            try:
                # Read frames with a timeout
                for frame in self.container.decode(video=0):
                    if self.stop_event.is_set():
                        break
                    
                    # Convert PyAV frame to numpy array
                    img = frame.to_ndarray(format='bgr24')
                    self.frame = img
            
            # Cập nhật các loại exception cần bắt
            except (InvalidDataError, StopIteration, EOFError, ConnectionError, OSError) as e:
                print(f"Error reading frame: {e}")
                self.connected = False
                # Try to reconnect
                if self.container:
                    try:
                        self.container.close()
                    except:
                        pass  # Ignore errors when closing
                    self.container = None
                # Thêm một chút delay trước khi thử lại
                time.sleep(0.5)
    
    def get_frame(self):
        """Get the latest frame"""
        return self.frame
    
    def stop(self):
        """Stop the reader thread"""
        self.stop_event.set()
        if self.container:
            self.container.close()
            self.container = None
        self.connected = False