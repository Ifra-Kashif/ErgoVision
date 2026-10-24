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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def play_alert_sound():
    def sound_worker():
        winsound.Beep(900, 300)
    threading.Thread(target=sound_worker, daemon=True).start()

def get_horizontal_angle(p1, p2):
    """Calculates absolute angle of line relative to horizontal (0 deg = level shoulders)."""
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:
        return 90.0
    return np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

def get_vertical_angle(p_bottom, p_top):
    """Calculates angle of line relative to the vertical axis (0 deg = upright)."""
    dx = p_top[0] - p_bottom[0]
    dy = p_bottom[1] - p_top[1]  # Invert because image y-axis points down
    if dy <= 0:
        return 90.0
    return np.degrees(np.arctan2(abs(dx), dy))

def get_ear(landmarks, eye_indices, img_w, img_h):
    coords = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = coords
    v1 = np.linalg.norm(np.array(p2) - np.array(p6))
    v2 = np.linalg.norm(np.array(p3) - np.array(p5))
    h = np.linalg.norm(np.array(p1) - np.array(p4))
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

class ErgoVisionProApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ErgoVision Pro - AI Ergonomics & Fatigue Telemetry")
        self.geometry("1060x700")
        self.resizable(False, False)

        self.cap = None
        self.is_running = False
        self.pose = None
        self.face_mesh = None

        # Thresholds
        self.SHOULDER_TILT_THRESHOLD = 8.0  # degrees from horizontal
        self.NECK_SLOUCH_THRESHOLD = 32.0   # degrees from vertical
        self.PROXIMITY_THRESHOLD = 175.0    # inter-ocular pixel distance
        self.DROWSY_EAR_THRESHOLD = 0.23    # half-closed eyelids
        self.BLINK_EAR_THRESHOLD = 0.20     # full blink closure

        # Telemetry State
        self.posture_slouch_start = None
        self.drowsy_start = None
        self.last_alert_time = 0
        self.eye_closed = False
        self.blink_count = 0
        self.session_start = time.time()

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Video Viewport
        self.video_frame = ctk.CTkFrame(self, corner_radius=12)
        self.video_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.video_frame, text="Click 'Start Monitoring' to Begin", font=("Segoe UI", 16))
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Dashboard Sidebar
        self.dash_frame = ctk.CTkFrame(self, corner_radius=12)
        self.dash_frame.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")

        ctk.CTkLabel(self.dash_frame, text="ErgoVision Pro", font=("Segoe UI", 24, "bold")).pack(pady=(12, 2))
        ctk.CTkLabel(self.dash_frame, text="Real-Time Health Telemetry", font=("Segoe UI", 11), text_color="gray").pack(pady=(0, 8))

        # Metric Cards
        self.card_alert = self._create_metric_card(self.dash_frame, "ACTIVE ADVISORY", "System Idle", "#242424")
        self.card_shoulder = self._create_metric_card(self.dash_frame, "SHOULDER ALIGNMENT", "0.0° (Optimal)", "#242424")
        self.card_neck = self._create_metric_card(self.dash_frame, "FORWARD HEAD / NECK", "0.0° (Optimal)", "#242424")
        self.card_dist = self._create_metric_card(self.dash_frame, "SCREEN PROXIMITY", "Safe Distance", "#242424")
        self.card_drowsy = self._create_metric_card(self.dash_frame, "EYE ENGAGEMENT", "Attentive", "#242424")
        self.card_bpm = self._create_metric_card(self.dash_frame, "BLINK RATE / STATUS", "0 /min", "#242424")

        self.btn_toggle = ctk.CTkButton(
            self.dash_frame,
            text="Start Monitoring",
            command=self.toggle_monitoring,
            font=("Segoe UI", 14, "bold"),
            height=38,
            fg_color="#1f6aa5"
        )
        self.btn_toggle.pack(side="bottom", fill="x", padx=15, pady=15)

    def _create_metric_card(self, parent, title, initial_val, bg_color):
        frame = ctk.CTkFrame(parent, corner_radius=8, fg_color=bg_color)
        frame.pack(fill="x", padx=15, pady=3)
        lbl_title = ctk.CTkLabel(frame, text=title, font=("Segoe UI", 10, "bold"), text_color="#aaaaaa")
        lbl_title.pack(anchor="w", padx=10, pady=(4, 0))
        lbl_val = ctk.CTkLabel(frame, text=initial_val, font=("Segoe UI", 14, "bold"))
        lbl_val.pack(anchor="w", padx=10, pady=(0, 5))
        return lbl_val

    def trigger_advisory(self, message, curr_time):
        self.card_alert.configure(text=message, text_color="#e63946")
        if curr_time - self.last_alert_time > 4.0:
            play_alert_sound()
            self.last_alert_time = curr_time

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

            primary_advisory = "Normal / Good Posture"
            advisory_active = False

            # --- 1. Skeletal Tracking (Shoulders & Tech-Neck) ---
            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                l_sh = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
                r_sh = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
                r_ear = (int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h))

                # Shoulder tilt
                shoulder_angle = get_horizontal_angle(l_sh, r_sh)
                shoulder_tilted = shoulder_angle > self.SHOULDER_TILT_THRESHOLD

                # Neck slouch angle
                neck_angle = get_vertical_angle(r_sh, r_ear)
                neck_slouching = neck_angle > self.NECK_SLOUCH_THRESHOLD

                # Visual skeleton lines
                sh_color = (0, 0, 255) if shoulder_tilted else (0, 255, 0)
                neck_color = (0, 0, 255) if neck_slouching else (0, 255, 0)

                cv2.line(frame, l_sh, r_sh, sh_color, 3)
                cv2.circle(frame, l_sh, 5, (255, 0, 0), -1)
                cv2.circle(frame, r_sh, 5, (255, 0, 0), -1)

                cv2.line(frame, r_sh, r_ear, neck_color, 3)
                cv2.circle(frame, r_ear, 5, (255, 0, 0), -1)

                # Posture status evaluations
                if shoulder_tilted:
                    self.card_shoulder.configure(text=f"{shoulder_angle:.1f}° (Tilted)", text_color="#e63946")
                    primary_advisory = "Shoulders uneven! Level your posture"
                    advisory_active = True
                else:
                    self.card_shoulder.configure(text=f"{shoulder_angle:.1f}° (Optimal)", text_color="#2a9d8f")

                if neck_slouching:
                    if self.posture_slouch_start is None:
                        self.posture_slouch_start = curr_time
                    elif curr_time - self.posture_slouch_start >= 2.5:
                        primary_advisory = "Chin up! Tech neck detected"
                        advisory_active = True
                        self.card_neck.configure(text=f"{neck_angle:.1f}° (Slouching)", text_color="#e63946")
                else:
                    self.posture_slouch_start = None
                    self.card_neck.configure(text=f"{neck_angle:.1f}° (Optimal)", text_color="#2a9d8f")

            # --- 2. Facial Mesh, Screen Proximity & Eye Tracking ---
            if face_res.multi_face_landmarks:
                f_lm = face_res.multi_face_landmarks[0].landmark

                # Proximity Check (Inter-ocular distance)
                left_outer = (f_lm[33].x * w, f_lm[33].y * h)
                right_outer = (f_lm[263].x * w, f_lm[263].y * h)
                face_pixel_dist = np.linalg.norm(np.array(left_outer) - np.array(right_outer))

                if face_pixel_dist > self.PROXIMITY_THRESHOLD:
                    primary_advisory = "Too close to screen! Sit back"
                    advisory_active = True
                    self.card_dist.configure(text=f"{int(face_pixel_dist)} px (Too Close)", text_color="#e63946")
                else:
                    self.card_dist.configure(text=f"{int(face_pixel_dist)} px (Safe Distance)", text_color="#2a9d8f")

                # Eye Aspect Ratio calculation
                avg_ear = (get_ear(f_lm, LEFT_EYE, w, h) + get_ear(f_lm, RIGHT_EYE, w, h)) / 2.0

                # Drowsy vs Normal state
                is_drowsy = avg_ear < self.DROWSY_EAR_THRESHOLD
                if is_drowsy:
                    if self.drowsy_start is None:
                        self.drowsy_start = curr_time
                    elif curr_time - self.drowsy_start >= 1.8:
                        primary_advisory = "Eye strain / Heavy eyelids!"
                        advisory_active = True
                        self.card_drowsy.configure(text=f"EAR {avg_ear:.2f} (Drowsy)", text_color="#e63946")
                else:
                    self.drowsy_start = None
                    self.card_drowsy.configure(text=f"EAR {avg_ear:.2f} (Attentive)", text_color="#2a9d8f")

                # Blink detection state machine
                if avg_ear < self.BLINK_EAR_THRESHOLD:
                    self.eye_closed = True
                    eye_dot_color = (0, 0, 255)  # Red dots when eyes close
                else:
                    if self.eye_closed:
                        self.blink_count += 1
                        self.eye_closed = False
                    eye_dot_color = (0, 255, 0)  # Green dots when eyes are open

                # Render reactive eye landmark dots
                for idx in LEFT_EYE + RIGHT_EYE:
                    pt = (int(f_lm[idx].x * w), int(f_lm[idx].y * h))
                    cv2.circle(frame, pt, 2, eye_dot_color, -1)

            # --- 3. Blink Rate Analytics ---
            elapsed = (curr_time - self.session_start) / 60.0
            bpm = int(self.blink_count / elapsed) if elapsed > 0.1 else self.blink_count

            if elapsed > 0.5 and bpm < 8:
                primary_advisory = "Dry Eye Warning! Blink frequently"
                advisory_active = True
                self.card_bpm.configure(text=f"{bpm} /min (Low)", text_color="#e63946")
            else:
                self.card_bpm.configure(text=f"{bpm} /min (Normal)", text_color="#2a9d8f")

            # --- 4. Alert Routing & Top Banner ---
            if advisory_active:
                self.trigger_advisory(primary_advisory, curr_time)
                cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 220), -1)
                cv2.putText(frame, f"! {primary_advisory} !", (w // 2 - 210, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                self.card_alert.configure(text="All Metrics Optimal", text_color="#2a9d8f")

            # Update Frame in GUI
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
            self.video_label.configure(image=img_tk, text="")

        self.after(15, self.update_feed)

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = ErgoVisionProApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()