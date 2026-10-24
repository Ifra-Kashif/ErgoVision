import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading
import customtkinter as ctk
from PIL import Image, ImageTk

# Set appearance and theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def play_alert_sound():
    def sound_worker():
        winsound.Beep(880, 250)
    threading.Thread(target=sound_worker, daemon=True).start()

def calculate_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:
        return 90.0
    return np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

def get_ear(landmarks, eye_indices, img_w, img_h):
    coords = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = coords
    v1 = np.linalg.norm(np.array(p2) - np.array(p6))
    v2 = np.linalg.norm(np.array(p3) - np.array(p5))
    h = np.linalg.norm(np.array(p1) - np.array(p4))
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

class ErgoVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ErgoVision - AI Ergonomics & Fatigue Monitor")
        self.geometry("980x640")
        self.resizable(False, False)

        # Video capture & tracking states
        self.cap = None
        self.is_running = False
        self.pose = None
        self.face_mesh = None

        # Analytics & Thresholds
        self.POSTURE_THRESHOLD = 8.0
        self.SLOUCH_TIME_LIMIT = 3.0
        self.slouch_start_time = None
        self.last_alert_time = 0
        
        # Optimized Blink Detector State
        self.EAR_THRESHOLD = 0.235  # Tuned for natural desktop distance
        self.eye_closed = False     # State flag for instantaneous blink capture
        self.blink_count = 0
        self.session_start = time.time()

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left: Video Frame
        self.video_frame = ctk.CTkFrame(self, corner_radius=12)
        self.video_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.video_frame, text="Click 'Start Monitoring' to Begin", font=("Segoe UI", 16))
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Right: Dashboard Control & Metrics
        self.dash_frame = ctk.CTkFrame(self, corner_radius=12)
        self.dash_frame.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")

        ctk.CTkLabel(self.dash_frame, text="ErgoVision", font=("Segoe UI", 24, "bold")).pack(pady=(15, 5))
        ctk.CTkLabel(self.dash_frame, text="Real-Time Telemetry", font=("Segoe UI", 12), text_color="gray").pack(pady=(0, 15))

        self.card_posture = self._create_metric_card(self.dash_frame, "POSTURE STATUS", "Idle", "#333333")
        self.card_tilt = self._create_metric_card(self.dash_frame, "SHOULDER TILT", "0.0°", "#333333")
        self.card_ear = self._create_metric_card(self.dash_frame, "LIVE EAR", "0.00", "#333333")
        self.card_blinks = self._create_metric_card(self.dash_frame, "TOTAL BLINKS", "0", "#333333")
        self.card_bpm = self._create_metric_card(self.dash_frame, "BLINK RATE", "0 /min", "#333333")

        self.btn_toggle = ctk.CTkButton(
            self.dash_frame,
            text="Start Monitoring",
            command=self.toggle_monitoring,
            font=("Segoe UI", 14, "bold"),
            height=40,
            fg_color="#1f6aa5"
        )
        self.btn_toggle.pack(side="bottom", fill="x", padx=15, pady=20)

    def _create_metric_card(self, parent, title, initial_val, bg_color):
        frame = ctk.CTkFrame(parent, corner_radius=8, fg_color=bg_color)
        frame.pack(fill="x", padx=15, pady=5)
        
        lbl_title = ctk.CTkLabel(frame, text=title, font=("Segoe UI", 10, "bold"), text_color="#aaaaaa")
        lbl_title.pack(anchor="w", padx=12, pady=(6, 0))
        
        lbl_val = ctk.CTkLabel(frame, text=initial_val, font=("Segoe UI", 18, "bold"))
        lbl_val.pack(anchor="w", padx=12, pady=(0, 6))
        return lbl_val

    def toggle_monitoring(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

            self.is_running = True
            self.session_start = time.time()
            self.blink_count = 0
            self.eye_closed = False
            self.btn_toggle.configure(text="Stop Monitoring", fg_color="#a83232")
            self.update_feed()
        else:
            self.is_running = False
            self.btn_toggle.configure(text="Start Monitoring", fg_color="#1f6aa5")
            if self.cap:
                self.cap.release()
            self.video_label.configure(image="", text="Session Paused")

    def update_feed(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            curr_time = time.time()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            pose_res = self.pose.process(rgb_frame)
            face_res = self.face_mesh.process(rgb_frame)

            # --- 1. Posture Tracking ---
            shoulder_angle = 0.0
            is_slouching = False

            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                p1 = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
                p2 = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
                shoulder_angle = calculate_angle(p1, p2)
                is_slouching = shoulder_angle > self.POSTURE_THRESHOLD

                if is_slouching:
                    if self.slouch_start_time is None:
                        self.slouch_start_time = curr_time
                    elif curr_time - self.slouch_start_time >= self.SLOUCH_TIME_LIMIT:
                        if curr_time - self.last_alert_time > 4.0:
                            play_alert_sound()
                            self.last_alert_time = curr_time
                else:
                    self.slouch_start_time = None

                line_color = (220, 50, 50) if is_slouching else (50, 220, 50)
                cv2.line(frame, p1, p2, line_color, 3)

            # --- 2. Robust State-Machine Blink Tracking ---
            avg_ear = 0.0
            if face_res.multi_face_landmarks:
                f_lm = face_res.multi_face_landmarks[0].landmark
                left_ear = get_ear(f_lm, LEFT_EYE, w, h)
                right_ear = get_ear(f_lm, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # State machine: tracks the exact transition from closed -> open
                if avg_ear < self.EAR_THRESHOLD:
                    self.eye_closed = True
                    dot_color = (0, 0, 255)  # Red dots when eyes register closed
                else:
                    if self.eye_closed:
                        self.blink_count += 1
                        self.eye_closed = False
                    dot_color = (0, 255, 0)  # Green dots when eyes are open

                # Visual landmark feedback on eyelids
                for idx in LEFT_EYE + RIGHT_EYE:
                    pt = (int(f_lm[idx].x * w), int(f_lm[idx].y * h))
                    cv2.circle(frame, pt, 2, dot_color, -1)

            # --- 3. Telemetry Updates ---
            posture_text = "SLOUCHING" if is_slouching else "OPTIMAL"
            posture_color = "#e63946" if is_slouching else "#2a9d8f"
            self.card_posture.configure(text=posture_text, text_color=posture_color)
            self.card_tilt.configure(text=f"{shoulder_angle:.1f}°")
            self.card_ear.configure(text=f"{avg_ear:.2f}")
            self.card_blinks.configure(text=str(self.blink_count))

            elapsed = (curr_time - self.session_start) / 60.0
            bpm = int(self.blink_count / elapsed) if elapsed > 0.05 else self.blink_count
            self.card_bpm.configure(text=f"{bpm} /min")

            # Convert frame for CustomTkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
            self.video_label.configure(image=img_tk, text="")

        # Poll faster (15 ms) for smooth tracking
        self.after(15, self.update_feed)

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = ErgoVisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()