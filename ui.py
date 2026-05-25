import sys
import os
import cv2
import time
import numpy as np
from collections import deque, Counter
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize, QPropertyAnimation, QSequentialAnimationGroup
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, 
    QHBoxLayout, QFrame, QComboBox, QGraphicsOpacityEffect
)

# Typography families using Inter and standard sans-serif system fallbacks
FONT_FAMILY = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
FONT_TITLE_FAMILY = "Inter, Trebuchet MS, sans-serif"

class VideoWorker(QThread):
    """
    Asynchronously grabs frames from OpenCV, runs the detector,
    resizes frames inside the background thread using highly optimized cv2.resize,
    and emits raw landmarks and pre-scaled RGB frames directly to the main UI.
    """
    frame_ready = Signal(object, str, float, object, float, dict) # QImage, gesture_name, confidence, landmarks, detector_fps, scores
    camera_error = Signal(str)

    def __init__(self, detector, camera_index=0):
        super().__init__()
        self.detector = detector
        self.camera_index = camera_index
        self.running = True
        self.cap = None

    def run(self):
        """Webcam capture loop running in background thread with 30 FPS throttling."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Optimize camera capture resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.camera_error.emit(f"Failed to open webcam at index {self.camera_index}")
            self.running = False
            return

        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.msleep(30)
                continue
                
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process frame using detector (applies exposure/LUT adjustments)
            # Returns RGB frame directly from detector to bypass rgbSwapped copies
            processed_rgb, gesture_name, confidence, landmarks, fps, scores = self.detector.process_frame(frame)
            
            # Thread-Level Optimization: Scale the frame directly in OpenCV (compiled C++) 
            # to a highly optimized 600x450 viewport canvas, completely saving CPU cycles in Qt's paint thread.
            resized_rgb = cv2.resize(processed_rgb, (600, 450), interpolation=cv2.INTER_LINEAR)
            
            # Thread-Safe QImage construction: Build format_RGB888 QImage directly inside background thread!
            # Deep copy .copy() ensures the buffer is completely self-contained and safe across thread boundaries.
            height, width, channel = resized_rgb.shape
            bytes_per_line = channel * width
            q_img = QImage(resized_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            # Emit pre-built QImage and gesture metrics safely to the GUI thread
            self.frame_ready.emit(q_img, gesture_name, confidence, landmarks, fps, scores)
            
            # Throttle the loop to a steady 30 FPS ceiling to prevent signal queue congestion
            elapsed = time.time() - start_time
            sleep_ms = max(1, int((0.0333 - elapsed) * 1000))
            self.msleep(sleep_ms)

        # Release capture resources
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def stop(self):
        """Stops the worker thread."""
        self.running = False
        self.wait()


class WebcamLabel(QWidget):
    """
    A cinematic webcam feed viewport that draws premium anti-aliased landmarks
    and corner guides natively in a single paintEvent pass.
    """
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.landmarks = None
        
        # Performance Recovery: Pre-allocate and cache drawing pens and brushes 
        # to avoid allocation garbage collection spikes during 30 FPS draw loops.
        self.guide_pen = QPen(QColor(255, 255, 255, 55))
        self.guide_pen.setWidth(2)
        
        # Enhanced visibility white skeletal pen
        self.connection_pen = QPen(QColor(255, 255, 255, 85)) 
        self.connection_pen.setWidth(1)
        
        self.joint_brush = QBrush(QColor(255, 255, 255, 195)) # Sharp bright joints
        self.tip_inner_brush = QBrush(QColor(255, 255, 255, 255))
        
        self.accent_color = QColor(229, 169, 60, 200) # Soft gold default
        
        # Pre-compile skeletal ligament mappings
        self.connections = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (5,9), (9,10), (10,11), (11,12),
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (17,18), (18,19), (19,20),
            (0,17)
        ]

    def set_image(self, q_img):
        """Saves current frame and schedules a single paintEvent refresh."""
        self.current_image = q_img
        self.update()

    def set_landmarks(self, landmarks, accent_color_hex="#E5A93C"):
        """Stores landmarks and schedules redraw."""
        self.landmarks = landmarks
        # Parse hex and set translucent alpha
        q_color = QColor(accent_color_hex)
        q_color.setAlpha(200) # High-contrast outer fingertip glow
        self.accent_color = q_color
        self.accent_brush = QBrush(q_color) # Pre-cached fingertip glow brush
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 1. Render the underlying pre-scaled camera image directly
        if self.current_image:
            # Centering offset math
            img_w = self.current_image.width()
            img_h = self.current_image.height()
            
            # Map coordinates to fit viewport widget rect cleanly
            target_rect = self.rect()
            painter.drawImage(target_rect, self.current_image)
        else:
            painter.fillRect(self.rect(), QColor(13, 14, 20, 100))
            
        # 2. Draw modern, elegant corner guides
        painter.setPen(self.guide_pen)
        g_len = 15
        # Top-Left
        painter.drawLine(15, 15, 15 + g_len, 15)
        painter.drawLine(15, 15, 15, 15 + g_len)
        # Top-Right
        painter.drawLine(w - 15, 15, w - 15 - g_len, 15)
        painter.drawLine(w - 15, 15, w - 15, 15 + g_len)
        # Bottom-Left
        painter.drawLine(15, h - 15, 15 + g_len, h - 15)
        painter.drawLine(15, h - 15, 15, h - 15 - g_len)
        # Bottom-Right
        painter.drawLine(w - 15, h - 15, w - 15 - g_len, h - 15)
        painter.drawLine(w - 15, h - 15, w - 15, h - 15 - g_len)

        # 3. Draw landmarks skeletal overlays dynamically if coordinates exist
        if not self.landmarks or not self.current_image:
            return
            
        img_w = self.current_image.width()
        img_h = self.current_image.height()
        
        # Check if landmarks represents multiple hands (nested lists)
        # We determine this by checking if the first element is a list or similar
        has_multiple = False
        if len(self.landmarks) > 0:
            first_elem = self.landmarks[0]
            if isinstance(first_elem, list) or (hasattr(first_elem, '__len__') and not hasattr(first_elem, 'x')):
                has_multiple = True
                
        hands_list = self.landmarks if has_multiple else [self.landmarks]
        
        for hand in hands_list:
            if hand is None or len(hand) < 21:
                continue
                
            points = []
            for lm in hand:
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append((px, py))
                
            # Draw connections using pre-allocated connection pen
            painter.setPen(self.connection_pen)
            for start, end in self.connections:
                if start < len(points) and end < len(points):
                    painter.drawLine(points[start][0], points[start][1], points[end][0], points[end][1])
                    
            # Draw joints as tiny, soft white nodes and finger tips with colored highlights
            painter.setPen(Qt.PenStyle.NoPen)
            for i, (px, py) in enumerate(points):
                if i in [4, 8, 12, 16, 20]: # Tips get a glowing colored aura
                    painter.setBrush(getattr(self, 'accent_brush', self.joint_brush))
                    painter.drawEllipse(px - 6, py - 6, 12, 12)
                    painter.setBrush(self.tip_inner_brush)
                    painter.drawEllipse(px - 3, py - 3, 6, 6)
                else: # Standard joints get small translucent white nodes
                    painter.setBrush(self.joint_brush)
                    painter.drawEllipse(px - 3, py - 3, 6, 6)


class ElegantPanel(QFrame):
    """A reusable QFrame styled with selective glassmorphism and rounded corners."""
    def __init__(self, bg_color="rgba(18, 20, 28, 0.60)", border_color="rgba(255, 255, 255, 0.07)"):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 16px;
            }}
        """)


class AnimeSystemUI(QMainWindow):
    """
    Stage 3: Premium Anime Companion UI with optimized cv2.resize offloading,
    cached QPainter landmarks, center-cropped aspect ratios, and rolling gesture stabilization.
    """
    def __init__(self, detector, anime_controller):
        super().__init__()
        self.detector = detector
        self.anime_controller = anime_controller
        self.video_thread = None
        self.current_camera_index = 0
        self.current_active_gesture = "None"
        
        # Optimized 6-frame rolling buffer for responsive confirmations (<200ms)
        self.gesture_buffer = deque(maxlen=6)
        
        # Debounce Cooldown Lock: locked for 300ms after triggers to allow smooth responsive fades
        self.cooldown_until = 0.0
        
        # Performance Recovery: Caching variables to prevent layout thrashing and CSS parsing on every frame
        self.cached_gesture_text = ""
        self.cached_gesture_style = ""
        self.cached_conf_text = ""
        self.cached_fps_text = ""
        self.fps_update_counter = 0
        
        # EMA Confidence Damping
        self.smooth_confidence = 0.0
        
        # Gesture State Persistence & Adaptive Hysteresis Counters
        self.none_state_counter = 0  # visible hand unrecognized frames
        
        # State variable for cinematic signature pose tracking
        self.signature_active_pose = "None"
        self.absence_counter = 0     # hand completely absent frames
        self.peak_confidence = 0.0   # peak confidence for adaptive windows
        
        # Load local Inter Google Fonts dynamically
        self.load_fonts()
        
        # Setup ambient shifting background system
        self.init_ambient_background()
        
        # Setup dashboard visual grid
        self.init_ui()
        self.start_camera()

    def load_fonts(self):
        """Attempts to register cached Inter fonts into the Qt database."""
        fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
        reg_path = os.path.join(fonts_dir, "Inter-Regular.ttf")
        bold_path = os.path.join(fonts_dir, "Inter-Bold.ttf")
        
        if os.path.exists(reg_path) and os.path.exists(bold_path):
            reg_id = QFontDatabase.addApplicationFont(reg_path)
            bold_id = QFontDatabase.addApplicationFont(bold_path)
            print(f"Loaded premium Inter typography successfully! (Reg: {reg_id}, Bold: {bold_id})")
        else:
            print("Using default system typography fallback.")

    def init_ambient_background(self):
        """Creates stacked dual layers for PS5-style ambient shifting blurs."""
        self.bg_label_bottom = QLabel(self)
        self.bg_label_top = QLabel(self)
        
        # Darkening overlay to ensure text remains perfectly readable
        self.bg_overlay = QFrame(self)
        self.bg_overlay.setStyleSheet("background-color: rgba(7, 8, 11, 0.84);")
        
        # Enable opacity effects on the top layer to crossfade blurs
        self.bg_opacity_effect = QGraphicsOpacityEffect(self.bg_label_top)
        self.bg_label_top.setGraphicsEffect(self.bg_opacity_effect)
        self.bg_opacity_effect.setOpacity(0.0)

    def resizeEvent(self, event):
        """Ensures the background layers expand dynamically to cover any screen resize."""
        super().resizeEvent(event)
        self.bg_label_bottom.setGeometry(self.rect())
        self.bg_label_top.setGeometry(self.rect())
        self.bg_overlay.setGeometry(self.rect())

    def init_ui(self):
        # Window properties
        self.setWindowTitle("Antigravity 2.0 - Premium Anime Companion")
        self.resize(1280, 780)
        self.setMinimumSize(1024, 680)
        
        # Base container with transparent background to show ambient shifting blurs
        central_widget = QWidget()
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(25, 20, 25, 25)
        main_layout.setSpacing(25)

        # ==================== LEFT COLUMN: CAMERA & METRICS ====================
        left_layout = QVBoxLayout()
        left_layout.setSpacing(20)
        # Layout Balance: Shift left column stretch from 16 to 13 to enlarge right panel
        main_layout.addLayout(left_layout, stretch=13)

        # 1. Top Header Area (Minimal layout, breathing AI status pulse)
        header_widget = QWidget()
        header_widget.setFixedHeight(50)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel("ANTIGRAVITY 2.0")
        title_font = QFont()
        title_font.setFamily(FONT_TITLE_FAMILY)
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #F5F6F8; letter-spacing: 3px; background: transparent;")
        
        mode_label = QLabel("AI COMPANION INTERFACE")
        mode_font = QFont()
        mode_font.setFamily(FONT_FAMILY)
        mode_font.setPointSize(9)
        mode_font.setBold(True)
        mode_label.setFont(mode_font)
        mode_label.setStyleSheet("color: #6A6D7C; letter-spacing: 2px; background: transparent;")
        
        self.status_tag = QLabel("● SYSTEM ACTIVE")
        status_font = QFont()
        status_font.setFamily(FONT_FAMILY)
        status_font.setPointSize(9)
        status_font.setBold(True)
        self.status_tag.setFont(status_font)
        self.status_tag.setStyleSheet("color: #39FF14; letter-spacing: 1px; background: transparent;")
        
        # Create subtle breathing loop on AI status tag
        self.init_breathing_status()
        
        header_layout.addWidget(title_label)
        header_layout.addSpacing(15)
        header_layout.addWidget(mode_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_tag)
        left_layout.addWidget(header_widget)

        # 2. Camera Viewport Panel (WebcamLabel integrates custom white skeletons)
        self.cam_panel = ElegantPanel("rgba(13, 14, 20, 0.35)", "rgba(255, 255, 255, 0.04)")
        cam_layout = QVBoxLayout(self.cam_panel)
        cam_layout.setContentsMargins(8, 8, 8, 8)
        
        self.cam_feed_label = WebcamLabel()
        cam_layout.addWidget(self.cam_feed_label)
        left_layout.addWidget(self.cam_panel, stretch=1)

        # 3. Telemetry Footer (Simplified, floating glass pills)
        telemetry_widget = QWidget()
        telemetry_widget.setFixedHeight(85)
        telemetry_layout = QHBoxLayout(telemetry_widget)
        telemetry_layout.setContentsMargins(0, 0, 0, 0)
        telemetry_layout.setSpacing(15)

        # Pill 1: Active Gesture
        pill_gesture = ElegantPanel()
        pill_g_layout = QVBoxLayout(pill_gesture)
        pill_g_layout.setContentsMargins(20, 12, 20, 12)
        pill_g_layout.setSpacing(2)
        lbl_g_title = QLabel("GESTURE")
        g_title_font = QFont()
        g_title_font.setFamily(FONT_FAMILY)
        g_title_font.setPointSize(8)
        g_title_font.setBold(True)
        lbl_g_title.setFont(g_title_font)
        lbl_g_title.setStyleSheet("color: #6A6D7C; letter-spacing: 1px;")
        
        self.gesture_val_label = QLabel("NONE")
        g_val_font = QFont()
        g_val_font.setFamily(FONT_TITLE_FAMILY)
        g_val_font.setPointSize(15)
        g_val_font.setBold(True)
        self.gesture_val_label.setFont(g_val_font)
        self.gesture_val_label.setStyleSheet("color: #F5F6F8;")
        pill_g_layout.addWidget(lbl_g_title)
        pill_g_layout.addWidget(self.gesture_val_label)
        telemetry_layout.addWidget(pill_gesture, stretch=1)

        # Pill 2: Confidence
        pill_conf = ElegantPanel()
        pill_c_layout = QVBoxLayout(pill_conf)
        pill_c_layout.setContentsMargins(20, 12, 20, 12)
        pill_c_layout.setSpacing(2)
        lbl_c_title = QLabel("MATCH")
        c_title_font = QFont()
        c_title_font.setFamily(FONT_FAMILY)
        c_title_font.setPointSize(8)
        c_title_font.setBold(True)
        lbl_c_title.setFont(c_title_font)
        lbl_c_title.setStyleSheet("color: #6A6D7C; letter-spacing: 1px;")
        
        self.conf_val_label = QLabel("0.0%")
        c_val_font = QFont()
        c_val_font.setFamily(FONT_TITLE_FAMILY)
        c_val_font.setPointSize(15)
        c_val_font.setBold(True)
        self.conf_val_label.setFont(c_val_font)
        self.conf_val_label.setStyleSheet("color: #F5F6F8;")
        pill_c_layout.addWidget(lbl_c_title)
        pill_c_layout.addWidget(self.conf_val_label)
        telemetry_layout.addWidget(pill_conf, stretch=1)

        # Pill 3: Tracker FPS
        pill_fps = ElegantPanel()
        pill_f_layout = QVBoxLayout(pill_fps)
        pill_f_layout.setContentsMargins(20, 12, 20, 12)
        pill_f_layout.setSpacing(2)
        lbl_f_title = QLabel("PERFORMANCE")
        f_title_font = QFont()
        f_title_font.setFamily(FONT_FAMILY)
        f_title_font.setPointSize(8)
        f_title_font.setBold(True)
        lbl_f_title.setFont(f_title_font)
        lbl_f_title.setStyleSheet("color: #6A6D7C; letter-spacing: 1px;")
        
        self.fps_val_label = QLabel("0.0 FPS")
        f_val_font = QFont()
        f_val_font.setFamily(FONT_TITLE_FAMILY)
        f_val_font.setPointSize(15)
        f_val_font.setBold(True)
        self.fps_val_label.setFont(f_val_font)
        self.fps_val_label.setStyleSheet("color: #E5A93C;") # Warm soft gold tone
        pill_f_layout.addWidget(lbl_f_title)
        pill_f_layout.addWidget(self.fps_val_label)
        telemetry_layout.addWidget(pill_fps, stretch=1)

        # Pill 4: Camera Source (Minimal Selector)
        pill_cam = ElegantPanel()
        pill_cam_layout = QVBoxLayout(pill_cam)
        pill_cam_layout.setContentsMargins(20, 12, 20, 12)
        pill_cam_layout.setSpacing(2)
        lbl_cam_title = QLabel("CAMERA")
        cam_title_font = QFont()
        cam_title_font.setFamily(FONT_FAMILY)
        cam_title_font.setPointSize(8)
        cam_title_font.setBold(True)
        lbl_cam_title.setFont(cam_title_font)
        lbl_cam_title.setStyleSheet("color: #6A6D7C; letter-spacing: 1px;")
        
        self.camera_box = QComboBox()
        self.camera_box.addItems(["CAM 0", "CAM 1", "CAM 2"])
        self.camera_box.currentIndexChanged.connect(self.change_camera)
        self.camera_box.setStyleSheet("""
            QComboBox {
                background-color: transparent;
                color: #F5F6F8;
                border: none;
                font-family: Inter, sans-serif;
                font-size: 14px;
                font-weight: bold;
                padding: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #0C0D14;
                color: #F5F6F8;
                selection-background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)
        pill_cam_layout.addWidget(lbl_cam_title)
        pill_cam_layout.addWidget(self.camera_box)
        telemetry_layout.addWidget(pill_cam, stretch=1)

        left_layout.addWidget(telemetry_widget)

        # ==================== RIGHT COLUMN: DYNAMIC ANIME DISPLAY ====================
        # Visual Balance: enlargedRight panel (stretch=11) for elegant anime key visuals
        self.right_panel = ElegantPanel("rgba(18, 20, 28, 0.65)", "rgba(255, 255, 255, 0.05)")
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(25, 25, 25, 25)
        right_layout.setSpacing(20)
        main_layout.addWidget(self.right_panel, stretch=11)

        # Container for stacked crossfading posters
        # Poster Size enlarged to 340x420 (tall cinematic portrait format)
        self.poster_container = QWidget()
        self.poster_container.setFixedSize(340, 420)
        self.poster_container.setStyleSheet("background: transparent;")
        
        # Dual stacked labels
        self.poster_label_bottom = QLabel(self.poster_container)
        self.poster_label_bottom.setFixedSize(340, 420)
        self.poster_label_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.poster_label_bottom.setStyleSheet("border-radius: 12px; background: rgba(0,0,0,0.2);")
        
        self.poster_label_top = QLabel(self.poster_container)
        self.poster_label_top.setFixedSize(340, 420)
        self.poster_label_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.poster_label_top.setStyleSheet("border-radius: 12px; background: transparent;")
        
        # Apply Opacity effect for transitions
        self.poster_opacity_effect = QGraphicsOpacityEffect(self.poster_label_top)
        self.poster_label_top.setGraphicsEffect(self.poster_opacity_effect)
        self.poster_opacity_effect.setOpacity(0.0)

        # Default placeholder on bottom
        self.poster_label_bottom.setText("Companion Idle\nShow a Hand Gesture")
        poster_font = QFont()
        poster_font.setFamily(FONT_FAMILY)
        poster_font.setPointSize(12)
        poster_font.setBold(True)
        self.poster_label_bottom.setFont(poster_font)
        self.poster_label_bottom.setStyleSheet("""
            color: #6A6D7C; 
            border: 1px dashed rgba(255, 255, 255, 0.1); 
            border-radius: 12px; 
            background-color: rgba(0, 0, 0, 0.2);
        """)

        right_layout.addWidget(self.poster_container, alignment=Qt.AlignmentFlag.AlignCenter)

        # Text Details Group
        text_layout = QVBoxLayout()
        text_layout.setSpacing(5)
        
        self.anime_title_label = QLabel("Companion Ready")
        anime_title_font = QFont()
        anime_title_font.setFamily(FONT_TITLE_FAMILY)
        anime_title_font.setPointSize(22)
        anime_title_font.setBold(True)
        self.anime_title_label.setFont(anime_title_font)
        self.anime_title_label.setStyleSheet("color: #F5F6F8; letter-spacing: 1px;")
        self.anime_title_label.setWordWrap(True)
        text_layout.addWidget(self.anime_title_label)

        self.char_name_label = QLabel("Awaiting live gesture input...")
        char_name_font = QFont()
        char_name_font.setFamily(FONT_FAMILY)
        char_name_font.setPointSize(13)
        self.char_name_label.setFont(char_name_font)
        self.char_name_label.setStyleSheet("color: #E5A93C;") # Soft gold tone
        text_layout.addWidget(self.char_name_label)
        right_layout.addLayout(text_layout)

        # Divider line
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: rgba(255, 255, 255, 0.06); max-height: 1px; border: none;")
        right_layout.addWidget(divider)

        # Description Layout
        self.description_label = QLabel(
            "Show an iconic anime signature pose to summon your favorite companion:\n\n"
            "• ⚔️ Shadow Clone Jutsu (Crossed Hands) → Naruto Uzumaki\n"
            "• ✌ Peace Sign → Monkey D. Luffy\n"
            "• ❤️ Scout Salute (Fist over Heart) → Eren Yeager\n"
            "• 🧘 Prayer Focus (Palms Pressed Flat) → Tanjiro Kamado\n"
            "• 🔮 Domain Expansion (Overlapping Index/Middle) → Gojo Satoru"
        )
        desc_font = QFont()
        desc_font.setFamily(FONT_FAMILY)
        desc_font.setPointSize(10)
        self.description_label.setFont(desc_font)
        self.description_label.setStyleSheet("color: #A2A5B3; line-height: 18px;")
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_layout.addWidget(self.description_label, stretch=1)

        # Translucent Quote Panel
        self.quote_panel = QFrame()
        self.quote_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.03);
                border-left: 3px solid rgba(255, 255, 255, 0.15);
                border-radius: 4px;
            }
        """)
        self.quote_panel.setFixedHeight(65)
        quote_layout = QVBoxLayout(self.quote_panel)
        quote_layout.setContentsMargins(15, 5, 15, 5)
        
        self.quote_label = QLabel('"Standby..."')
        quote_font = QFont()
        quote_font.setFamily(FONT_FAMILY)
        quote_font.setPointSize(10)
        quote_font.setItalic(True)
        self.quote_label.setFont(quote_font)
        self.quote_label.setStyleSheet("color: #F5F6F8;")
        self.quote_label.setWordWrap(True)
        quote_layout.addWidget(self.quote_label)
        
        right_layout.addWidget(self.quote_panel)

    def init_breathing_status(self):
        """Fades AI active light continually between 0.45 and 1.0 opacity."""
        self.status_opacity_effect = QGraphicsOpacityEffect(self.status_tag)
        self.status_tag.setGraphicsEffect(self.status_opacity_effect)
        self.status_opacity_effect.setOpacity(1.0)
        
        # Build loop animations
        anim_fade_out = QPropertyAnimation(self.status_opacity_effect, b"opacity")
        anim_fade_out.setDuration(1200)
        anim_fade_out.setStartValue(1.0)
        anim_fade_out.setEndValue(0.45)
        
        anim_fade_in = QPropertyAnimation(self.status_opacity_effect, b"opacity")
        anim_fade_in.setDuration(1200)
        anim_fade_in.setStartValue(0.45)
        anim_fade_in.setEndValue(1.0)
        
        self.pulse_group = QSequentialAnimationGroup(self)
        self.pulse_group.addAnimation(anim_fade_out)
        self.pulse_group.addAnimation(anim_fade_in)
        self.pulse_group.setLoopCount(-1) # Infinite looping
        self.pulse_group.start()

    def start_camera(self):
        """Initializes and runs the camera thread."""
        self.stop_camera()
        self.video_thread = VideoWorker(self.detector, self.current_camera_index)
        self.video_thread.frame_ready.connect(self.on_frame_ready)
        self.video_thread.camera_error.connect(self.on_camera_error)
        self.video_thread.start()

    def stop_camera(self):
        """Gracefully halts the camera thread."""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

    def center_crop_pixmap(self, pixmap, target_width=340, target_height=420):
        """Scales image keeping aspect ratio by expanding and crops a perfect centered box."""
        target_size = QSize(target_width, target_height)
        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )
        # Crop centered box to avoid stretching
        x = (scaled_pixmap.width() - target_width) // 2
        y = (scaled_pixmap.height() - target_height) // 2
        return scaled_pixmap.copy(x, y, target_width, target_height)

    @Slot(object, str, float, object, float, dict)
    def on_frame_ready(self, frame, gesture_name, confidence, landmarks, fps, scores):
        """Fires whenever the worker thread has a processed, pre-built QImage frame."""
        # 1. Throttle tracker FPS updates to prevent layout thrashing and flicker
        self.fps_update_counter += 1
        if self.fps_update_counter >= 15:
            self.fps_update_counter = 0
            fps_text = f"{fps:.1f} FPS"
            if self.cached_fps_text != fps_text:
                self.fps_val_label.setText(fps_text)
                self.cached_fps_text = fps_text
        
        # 2. Geometry-Change Auto-Invalidation (Immediate Reset on Shape Collapse)
        # Bypasses sticky state persistence if the hand deforms strongly away from active pose
        if self.current_active_gesture != "None" and landmarks is not None:
            active_score = scores.get(self.current_active_gesture, 0.0)
            if active_score < 0.22:
                # Shape collapsed: drop state immediately and skip retention frames!
                self.current_active_gesture = "None"
                self.cooldown_until = time.time() + 0.30
                self.peak_confidence = 0.0
                self.reset_to_standby()
        
        # 3. Track hand presence and absence counters
        if landmarks is not None:
            self.absence_counter = 0 # Reset hand absence frame counter since hand is visible
            
            # Instant Switch on Confident New Gesture: bypasses buffer/queue latency for intentional transitions
            # If the user confidently shows a new gesture, it triggers instantly (0ms latency!)
            if gesture_name != "None" and gesture_name != self.current_active_gesture and confidence >= 0.72:
                confirmed_gesture = gesture_name
                # Fast-clear gesture buffer and pre-fill to synchronize with the new active state
                self.gesture_buffer.clear()
                for _ in range(6):
                    self.gesture_buffer.append(gesture_name)
            else:
                # Push current frame result to our optimized 6-frame rolling queue
                self.gesture_buffer.append(gesture_name)
                counts = Counter(self.gesture_buffer)
                dominant_gesture, count = counts.most_common(1)[0]
                
                # Threshold: gesture category must dominate at least 4 out of 6 frames (approx 66%)
                confirmed_gesture = "None"
                if count >= 4 and dominant_gesture != "None":
                    confirmed_gesture = dominant_gesture
        else:
            # Hand is completely absent: clear queue immediately to prepare for next entrance
            confirmed_gesture = "None"
            self.gesture_buffer.clear()
            self.absence_counter += 1
            
        current_time = time.time()
        
        # Determine tracking visual accents
        accent_color_hex = "#E5A93C"
        
        # Decide transitions with State Persistence Engine
        if confirmed_gesture != "None":
            # Update peak confidence to enable adaptive visible-hand persistence later
            if confidence > self.peak_confidence:
                self.peak_confidence = confidence
                
            # Reset the visible unrecognized counter since a valid gesture is confirmed
            self.none_state_counter = 0
            
            # Transition immediately if switching directly between two valid gestures
            if self.current_active_gesture != "None" and self.current_active_gesture != confirmed_gesture:
                if current_time >= self.cooldown_until:
                    self.current_active_gesture = confirmed_gesture
                    # 300ms responsive cooldown (immediate yet controlled)
                    self.cooldown_until = current_time + 0.30
                    self.peak_confidence = confidence
                    
                    # Confidence Crossfade: Reset starting confidence to the new gesture's raw score
                    # This allows the new gesture's confidence to rise rapidly instead of carrying over the old high value!
                    self.smooth_confidence = confidence
                    
                    anime_data = self.anime_controller.get_anime_data(confirmed_gesture)
                    if anime_data:
                        self.update_anime_details(anime_data)
            
            # Or if transitioning from standby NONE to a valid gesture
            elif self.current_active_gesture == "None":
                if current_time >= self.cooldown_until:
                    self.current_active_gesture = confirmed_gesture
                    self.cooldown_until = current_time + 0.30
                    self.peak_confidence = confidence
                    
                    # Starting confidence initialized
                    self.smooth_confidence = confidence
                    
                    anime_data = self.anime_controller.get_anime_data(confirmed_gesture)
                    if anime_data:
                        self.update_anime_details(anime_data)
                        
            # Determine visual accent from active state
            anime_data = self.anime_controller.get_anime_data(self.current_active_gesture)
            if anime_data:
                accent_color_hex = anime_data.get("accent_color", "#E5A93C")
        else:
            # Current frame returns None. Implement State Persistence & Hysteresis windows.
            if self.current_active_gesture != "None":
                if landmarks is not None:
                    # Hand is visible but gesture is unrecognized: increment visible counter
                    self.none_state_counter += 1
                    
                    # Adaptive visible-hand persistence window: 25 frames if peak confidence was high (>0.85), else 15
                    max_retention_frames = 25 if self.peak_confidence > 0.85 else 15
                    
                    # Drop back to standby only if unrecognized frames exceed the retention window
                    if self.none_state_counter >= max_retention_frames:
                        if current_time >= self.cooldown_until:
                            self.current_active_gesture = "None"
                            self.cooldown_until = current_time + 0.30
                            self.peak_confidence = 0.0
                            self.reset_to_standby()
                else:
                    # Hand is completely absent: check 10-frame absence grace window to absorb exits
                    if self.absence_counter >= 10:
                        if current_time >= self.cooldown_until:
                            self.current_active_gesture = "None"
                            self.cooldown_until = current_time + 0.30
                            self.peak_confidence = 0.0
                            self.reset_to_standby()
                            
                # Keep current companion accent color active during retention
                anime_data = self.anime_controller.get_anime_data(self.current_active_gesture)
                if anime_data:
                    accent_color_hex = anime_data.get("accent_color", "#E5A93C")

        # Draw custom white coordinate vectors on the pre-scaled frame
        self.cam_feed_label.set_landmarks(landmarks, accent_color_hex)
        
        # 3. Smart Confidence Fallback & Soft Damping
        alpha_ema = 0.25
        if self.current_active_gesture != "None":
            # Use active state's continuous score from gesture detector as confidence tracking
            active_score = scores.get(self.current_active_gesture, 0.0)
            if gesture_name == self.current_active_gesture:
                self.smooth_confidence = alpha_ema * active_score + (1 - alpha_ema) * self.smooth_confidence
            else:
                # Differentiate intentional gesture change vs. same-gesture movement
                if gesture_name != "None":
                    # Intentional transition: decay old confidence rapidly (15% per frame)
                    self.smooth_confidence *= 0.85
                else:
                    # Same-gesture movement/unclassified dip: decay slowly (4% per frame) to maintain stability
                    self.smooth_confidence *= 0.96
        else:
            self.smooth_confidence = alpha_ema * 0.0 + (1 - alpha_ema) * self.smooth_confidence
        
        # 4. Anime Signature Pose Verification & Activation Pulse
        active_signature = "None"
        if self.current_active_gesture in ["Shadow Clone Jutsu", "Domain Expansion", "Scout Salute", "Prayer Focus"]:
            active_signature = self.current_active_gesture
            
        # Trigger real-time visual activation transitions only on signature state changes (prevents thrashing)
        if self.signature_active_pose != active_signature:
            self.signature_active_pose = active_signature
            anime_data = self.anime_controller.get_anime_data(self.current_active_gesture)
            theme_color = anime_data.get("accent_color", "#E5A93C") if anime_data else "#E5A93C"
            
            if active_signature != "None":
                # 1. Apply thick glowing border to the right information panel
                self.right_panel.setStyleSheet(f"""
                    QFrame {{
                        background-color: rgba(18, 20, 28, 0.78);
                        border: 2.5px solid {theme_color};
                        border-radius: 16px;
                    }}
                """)
                # 2. Update character vocal technique quotes
                if active_signature == "Shadow Clone Jutsu":
                    self.quote_label.setText('"Kage Bunshin no Jutsu!"')
                elif active_signature == "Domain Expansion":
                    self.quote_label.setText('"Ryoiki Tenkai: Muryokusho!"')
                elif active_signature == "Scout Salute":
                    self.quote_label.setText('"Shinzou wo Sasageyo!"')
                elif active_signature == "Prayer Focus":
                    self.quote_label.setText('"Mizu no Kokyu..."')
            else:
                # Restore standard companion stylesheet and standard quote
                self.right_panel.setStyleSheet(f"""
                    QFrame {{
                        background-color: rgba(18, 20, 28, 0.65);
                        border: 1px solid rgba(255, 255, 255, 0.05);
                        border-radius: 16px;
                    }}
                """)
                if anime_data:
                    self.quote_label.setText(f'"{anime_data["quote"]}"')
                else:
                    self.quote_label.setText('"Standby..."')
                    
        # 5. Prevent Layout Thrashing: Only set text/style if values actually change
        if self.current_active_gesture != "None":
            if active_signature != "None":
                gesture_text = active_signature.upper()
            else:
                gesture_text = self.current_active_gesture.upper()
            style_text = f"color: {accent_color_hex};"
            conf_text = f"{self.smooth_confidence * 100:.1f}%"
        else:
            gesture_text = "NONE"
            style_text = "color: #F5F6F8;"
            conf_text = "0.0%"

        # Execute layout thrashing checks
        if self.cached_gesture_text != gesture_text:
            self.gesture_val_label.setText(gesture_text)
            self.cached_gesture_text = gesture_text
            
        if self.cached_gesture_style != style_text:
            self.gesture_val_label.setStyleSheet(style_text)
            self.cached_gesture_style = style_text
            
        if self.cached_conf_text != conf_text:
            self.conf_val_label.setText(conf_text)
            self.cached_conf_text = conf_text

        # Draw the pre-built QImage directly inside the custom QWidget repaint (Zero-Copy!)
        self.cam_feed_label.set_image(frame)

    def reset_to_standby(self):
        """Restores default standby visual onboarding descriptions."""
        self.anime_title_label.setText("Companion Ready")
        self.char_name_label.setText("Awaiting live gesture input...")
        self.quote_label.setText('"Standby..."')
        self.char_name_label.setStyleSheet("color: #E5A93C;")
        self.description_label.setText(
            "Show an iconic anime signature pose to summon your favorite companion:\n\n"
            "• ⚔️ Shadow Clone Jutsu (Crossed Hands) → Naruto Uzumaki\n"
            "• ✌ Peace Sign → Monkey D. Luffy\n"
            "• ❤️ Scout Salute (Fist over Heart) → Eren Yeager\n"
            "• 🧘 Prayer Focus (Palms Pressed Flat) → Tanjiro Kamado\n"
            "• 🔮 Domain Expansion (Overlapping Index/Middle) → Gojo Satoru"
        )
        self.quote_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.03);
                border-left: 3px solid rgba(255, 255, 255, 0.15);
                border-radius: 4px;
            }
        """)

    def update_anime_details(self, data):
        """Updates display text and triggers ambient shifting background and poster crossfades."""
        self.anime_title_label.setText(data["title"])
        self.char_name_label.setText(data["character"])
        self.quote_label.setText(f'"{data["quote"]}"')
        self.description_label.setText(data["theme"])
        
        # Update themed color details
        theme_color = data.get("accent_color", "#E5A93C")
        self.char_name_label.setStyleSheet(f"color: {theme_color}; font-weight: bold;")
        self.quote_panel.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(255, 255, 255, 0.02);
                border-left: 3px solid {theme_color};
                border-radius: 4px;
            }}
        """)
        
        # 1. Trigger PS5-style ambient backdrop shifting crossfade
        if data["ambient_path"] and os.path.exists(data["ambient_path"]):
            self.bg_label_bottom.setPixmap(self.bg_label_top.pixmap())
            
            pixmap = QPixmap(data["ambient_path"])
            # Performance Optimization: Scale blurred background using FastTransformation
            # Since the image is pre-blurred (80px radius), there is zero visual difference,
            # but scaling is 50x faster, completely eliminating transition stutter!
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.FastTransformation
            )
            self.bg_label_top.setPixmap(scaled_pixmap)
            
            # Animate the blur layer dissolve
            self.bg_opacity_effect.setOpacity(0.0)
            self.bg_anim = QPropertyAnimation(self.bg_opacity_effect, b"opacity")
            self.bg_anim.setDuration(400)
            self.bg_anim.setStartValue(0.0)
            self.bg_anim.setEndValue(1.0)
            self.bg_anim.start()

        # 2. Trigger stacked poster card dissolve crossfade
        if data["image_path"] and os.path.exists(data["image_path"]):
            # Move the top label pixmap to the bottom label as a stable fallback
            self.poster_label_bottom.setPixmap(self.poster_label_top.pixmap())
            self.poster_label_bottom.setStyleSheet("border: none; background: transparent;")
            
            # Set the new poster image to the top label after center-cropping
            pixmap = QPixmap(data["image_path"])
            cropped_pixmap = self.center_crop_pixmap(pixmap, 340, 420)
            self.poster_label_top.setPixmap(cropped_pixmap)
            
            # Animate the poster dissolve
            self.poster_opacity_effect.setOpacity(0.0)
            self.poster_anim = QPropertyAnimation(self.poster_opacity_effect, b"opacity")
            self.poster_anim.setDuration(350)
            self.poster_anim.setStartValue(0.0)
            self.poster_anim.setEndValue(1.0)
            self.poster_anim.start()

    @Slot(str)
    def on_camera_error(self, err_msg):
        """Displays error when camera fails to open."""
        self.cam_feed_label.setText(f"Camera Initialization Failed\n{err_msg}")
        self.cam_feed_label.setStyleSheet("color: #FF4D4D; font-weight: bold;")

    def change_camera(self, index):
        """Changes webcam feed source."""
        if self.current_camera_index != index:
            self.current_camera_index = index
            self.start_camera()

    def closeEvent(self, event):
        """Overrides window close handler to ensure proper camera thread shutdown."""
        self.stop_camera()
        self.detector.release()
        event.accept()


def run_app(detector, anime_controller):
    app = QApplication(sys.argv)
    window = AnimeSystemUI(detector, anime_controller)
    window.show()
    sys.exit(app.exec())
