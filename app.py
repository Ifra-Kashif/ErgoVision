import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading
from datetime import datetime
import customtkinter as ctk
from PIL import Image, ImageTk

# Import our dedicated local database module
import database

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
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:
        return 90.0
    return np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

def get_vertical_angle(p_bottom, p_top):
    dx = p_top[0] - p_bottom[0]
    dy = p_bottom[1] - p_top[1]
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

        self.title("ErgoVision Pro - AI Ergonomics & Clinical Telemetry")
        
        # Clean, natural desktop window size
        self.geometry("980x640")
        self.resizable(False, False)

        # Initialize SQLite database silently
        database.init_db()

        self.cap = None
        self.is_running = False
        self.pose = None
        self.face_mesh = None

        # Calibration Defaults
        self.baseline_neck_angle = 15.0
        self.baseline_face_width = 130.0
        self.baseline_shoulder_angle = 2.0
        self.is_calibrated = False

        self.is_calibrating = False
        self.calibration_start = 0
        self.calib_neck_samples = []
        self.calib_dist_samples = []
        self.calib_shoulder_samples = []

        # Thresholds
        self.NECK_SLOUCH_DELTA = 15.0
        self.PROXIMITY_RATIO = 1.30
        self.SHOULDER_TILT_MAX = 8.0
        self.DROWSY_EAR_THRESHOLD = 0.23
        self.BLINK_EAR_THRESHOLD = 0.20

        # Telemetry State
        self.posture_slouch_start = None
        self.drowsy_start = None
        self.last_alert_time = 0
        self.eye_closed = False
        self.blink_count = 0
        self.session_start = 0

        # Session Stats
        self.slouch_warning_count = 0
        self.proximity_warning_count = 0
        self.fatigue_warning_count = 0
        self.total_frames_tracked = 0
        self.good_frames_tracked = 0

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # 1. Left Fixed-Ratio Video Viewport
        self.video_frame = ctk.CTkFrame(self, corner_radius=12)
        self.video_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.video_frame, text="Click 'Start Monitoring' to Begin", font=("Segoe UI", 16))
        self.video_label.pack(expand=True, padx=10, pady=10)

        # 2. Right Scrollable Sidebar (All buttons & cards cleanly accessible)
        self.dash_frame = ctk.CTkScrollableFrame(self, corner_radius=12, width=300)
        self.dash_frame.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")

        ctk.CTkLabel(self.dash_frame, text="ErgoVision Pro", font=("Segoe UI", 22, "bold")).pack(pady=(6, 2))
        self.lbl_calib_status = ctk.CTkLabel(self.dash_frame, text="Calibration: Default (Not Calibrated)", font=("Segoe UI", 11), text_color="#f4a261")
        self.lbl_calib_status.pack(pady=(0, 8))

        # Metric Cards
        self.card_alert = self._create_metric_card(self.dash_frame, "ACTIVE ADVISORY", "System Idle", "#242424")
        self.card_shoulder = self._create_metric_card(self.dash_frame, "SHOULDER ALIGNMENT", "0.0° (Optimal)", "#242424")
        self.card_neck = self._create_metric_card(self.dash_frame, "FORWARD HEAD / NECK", "0.0° (Optimal)", "#242424")
        self.card_dist = self._create_metric_card(self.dash_frame, "SCREEN PROXIMITY", "Safe Distance", "#242424")
        self.card_drowsy = self._create_metric_card(self.dash_frame, "EYE ENGAGEMENT", "Attentive", "#242424")
        self.card_bpm = self._create_metric_card(self.dash_frame, "BLINK RATE / STATUS", "0 /min", "#242424")

        # Action Buttons
        self.btn_toggle = ctk.CTkButton(
            self.dash_frame,
            text="Start Monitoring",
            command=self.toggle_monitoring,
            font=("Segoe UI", 13, "bold"),
            height=36,
            fg_color="#1f6aa5"
        )
        self.btn_toggle.pack(fill="x", padx=10, pady=(14, 4))

        self.btn_calibrate = ctk.CTkButton(
            self.dash_frame,
            text="Calibrate Ideal Posture (3s)",
            command=self.start_calibration,
            font=("Segoe UI", 12, "bold"),
            height=32,
            fg_color="#457b9d"
        )
        self.btn_calibrate.pack(fill="x", padx=10, pady=3)

        self.btn_report = ctk.CTkButton(
            self.dash_frame,
            text="Monthly Health Report",
            command=self.open_monthly_report,
            font=("Segoe UI", 12, "bold"),
            height=32,
            fg_color="#2b2d42"
        )
        self.btn_report.pack(fill="x", padx=10, pady=(3, 10))

    def _create_metric_card(self, parent, title, initial_val, bg_color):
        frame = ctk.CTkFrame(parent, corner_radius=8, fg_color=bg_color)
        frame.pack(fill="x", padx=10, pady=2)
        lbl_title = ctk.CTkLabel(frame, text=title, font=("Segoe UI", 9, "bold"), text_color="#aaaaaa")
        lbl_title.pack(anchor="w", padx=10, pady=(3, 0))
        lbl_val = ctk.CTkLabel(frame, text=initial_val, font=("Segoe UI", 13, "bold"))
        lbl_val.pack(anchor="w", padx=10, pady=(0, 4))
        return lbl_val

    def start_calibration(self):
        if not self.is_running:
            return
        self.is_calibrating = True
        self.calibration_start = time.time()
        self.calib_neck_samples.clear()
        self.calib_dist_samples.clear()
        self.calib_shoulder_samples.clear()
        self.btn_calibrate.configure(text="Calibrating... Sit Upright", fg_color="#e76f51")

    def trigger_advisory(self, message, curr_time, category="posture"):
        self.card_alert.configure(text=message, text_color="#e63946")
        if curr_time - self.last_alert_time > 4.0:
            play_alert_sound()
            self.last_alert_time = curr_time
            if category == "posture":
                self.slouch_warning_count += 1
            elif category == "proximity":
                self.proximity_warning_count += 1
            elif category == "fatigue":
                self.fatigue_warning_count += 1

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
            self.slouch_warning_count = 0
            self.proximity_warning_count = 0
            self.fatigue_warning_count = 0
            self.total_frames_tracked = 0
            self.good_frames_tracked = 0

            self.btn_toggle.configure(text="Stop Monitoring", fg_color="#a83232")
            self.update_feed()
        else:
            self.is_running = False
            self.btn_toggle.configure(text="Start Monitoring", fg_color="#1f6aa5")
            if self.cap:
                self.cap.release()
            self.video_label.configure(image="", text="Session Complete")
            self._save_and_display_summary()

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
            advisory_category = "posture"

            current_neck_angle = None
            current_shoulder_angle = None
            current_face_dist = None

            # --- 1. Skeletal Tracking ---
            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                l_sh = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
                r_sh = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
                r_ear = (int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h))

                current_shoulder_angle = get_horizontal_angle(l_sh, r_sh)
                current_neck_angle = get_vertical_angle(r_sh, r_ear)

                neck_limit = self.baseline_neck_angle + self.NECK_SLOUCH_DELTA
                neck_slouching = current_neck_angle > neck_limit
                shoulder_tilted = current_shoulder_angle > self.SHOULDER_TILT_MAX

                sh_color = (0, 0, 255) if shoulder_tilted else (0, 255, 0)
                neck_color = (0, 0, 255) if neck_slouching else (0, 255, 0)

                cv2.line(frame, l_sh, r_sh, sh_color, 3)
                cv2.circle(frame, l_sh, 5, (255, 0, 0), -1)
                cv2.circle(frame, r_sh, 5, (255, 0, 0), -1)

                cv2.line(frame, r_sh, r_ear, neck_color, 3)
                cv2.circle(frame, r_ear, 5, (255, 0, 0), -1)

                if shoulder_tilted:
                    self.card_shoulder.configure(text=f"{current_shoulder_angle:.1f}° (Tilted)", text_color="#e63946")
                    primary_advisory = "Shoulders uneven! Level your posture"
                    advisory_active = True
                    advisory_category = "posture"
                else:
                    self.card_shoulder.configure(text=f"{current_shoulder_angle:.1f}° (Optimal)", text_color="#2a9d8f")

                if neck_slouching:
                    if self.posture_slouch_start is None:
                        self.posture_slouch_start = curr_time
                    elif curr_time - self.posture_slouch_start >= 2.5:
                        primary_advisory = "Chin up! Tech neck detected"
                        advisory_active = True
                        advisory_category = "posture"
                        self.card_neck.configure(text=f"{current_neck_angle:.1f}° (Slouching)", text_color="#e63946")
                else:
                    self.posture_slouch_start = None
                    self.card_neck.configure(text=f"{current_neck_angle:.1f}° (Optimal)", text_color="#2a9d8f")

            # --- 2. Facial Mesh & Proximity ---
            if face_res.multi_face_landmarks:
                f_lm = face_res.multi_face_landmarks[0].landmark

                left_outer = (f_lm[33].x * w, f_lm[33].y * h)
                right_outer = (f_lm[263].x * w, f_lm[263].y * h)
                current_face_dist = np.linalg.norm(np.array(left_outer) - np.array(right_outer))

                max_safe_dist = self.baseline_face_width * self.PROXIMITY_RATIO
                is_too_close = current_face_dist > max_safe_dist

                if is_too_close:
                    primary_advisory = "Too close to screen! Sit back"
                    advisory_active = True
                    advisory_category = "proximity"
                    self.card_dist.configure(text=f"{int(current_face_dist)} px (Too Close)", text_color="#e63946")
                else:
                    self.card_dist.configure(text=f"{int(current_face_dist)} px (Safe Distance)", text_color="#2a9d8f")

                avg_ear = (get_ear(f_lm, LEFT_EYE, w, h) + get_ear(f_lm, RIGHT_EYE, w, h)) / 2.0

                if avg_ear < self.DROWSY_EAR_THRESHOLD:
                    if self.drowsy_start is None:
                        self.drowsy_start = curr_time
                    elif curr_time - self.drowsy_start >= 1.8:
                        primary_advisory = "Eye strain / Heavy eyelids!"
                        advisory_active = True
                        advisory_category = "fatigue"
                        self.card_drowsy.configure(text=f"EAR {avg_ear:.2f} (Drowsy)", text_color="#e63946")
                else:
                    self.drowsy_start = None
                    self.card_drowsy.configure(text=f"EAR {avg_ear:.2f} (Attentive)", text_color="#2a9d8f")

                if avg_ear < self.BLINK_EAR_THRESHOLD:
                    self.eye_closed = True
                    eye_dot_color = (0, 0, 255)
                else:
                    if self.eye_closed:
                        self.blink_count += 1
                        self.eye_closed = False
                    eye_dot_color = (0, 255, 0)

                for idx in LEFT_EYE + RIGHT_EYE:
                    pt = (int(f_lm[idx].x * w), int(f_lm[idx].y * h))
                    cv2.circle(frame, pt, 2, eye_dot_color, -1)

            if not self.is_calibrating:
                self.total_frames_tracked += 1
                if not advisory_active:
                    self.good_frames_tracked += 1

            # --- 3. Calibration Sampling Loop ---
            if self.is_calibrating:
                calib_elapsed = curr_time - self.calibration_start
                if current_neck_angle:
                    self.calib_neck_samples.append(current_neck_angle)
                if current_face_dist:
                    self.calib_dist_samples.append(current_face_dist)
                if current_shoulder_angle:
                    self.calib_shoulder_samples.append(current_shoulder_angle)

                remaining = max(0, 3 - int(calib_elapsed))
                cv2.rectangle(frame, (0, 0), (w, 45), (200, 100, 0), -1)
                cv2.putText(frame, f"CALIBRATING... HOLD UPRIGHT ({remaining}s)", (w // 2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if calib_elapsed >= 3.0:
                    self.is_calibrating = False
                    if self.calib_neck_samples:
                        self.baseline_neck_angle = float(np.median(self.calib_neck_samples))
                    if self.calib_dist_samples:
                        self.baseline_face_width = float(np.median(self.calib_dist_samples))
                    if self.calib_shoulder_samples:
                        self.baseline_shoulder_angle = float(np.median(self.calib_shoulder_samples))

                    self.is_calibrated = True
                    self.lbl_calib_status.configure(text=f"Calibrated: Neck {self.baseline_neck_angle:.1f}° | Dist {int(self.baseline_face_width)}px", text_color="#2a9d8f")
                    self.btn_calibrate.configure(text="Recalibrate Posture", fg_color="#457b9d")

            # --- 4. Blink Rate Analytics ---
            elapsed = (curr_time - self.session_start) / 60.0
            bpm = int(self.blink_count / elapsed) if elapsed > 0.1 else self.blink_count

            if elapsed > 0.5 and bpm < 8:
                primary_advisory = "Dry Eye Warning! Blink frequently"
                advisory_active = True
                advisory_category = "fatigue"
                self.card_bpm.configure(text=f"{bpm} /min (Low)", text_color="#e63946")
            else:
                self.card_bpm.configure(text=f"{bpm} /min (Normal)", text_color="#2a9d8f")

            # --- 5. Alert Routing ---
            if advisory_active and not self.is_calibrating:
                self.trigger_advisory(primary_advisory, curr_time, category=advisory_category)
                cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 220), -1)
                cv2.putText(frame, f"! {primary_advisory} !", (w // 2 - 210, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif not self.is_calibrating:
                self.card_alert.configure(text="All Metrics Optimal", text_color="#2a9d8f")

            # Original clean 1:1 natural camera aspect (no stretching or zooming)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
            self.video_label.configure(image=img_tk, text="")

        self.after(15, self.update_feed)

    def _save_and_display_summary(self):
        duration_sec = int(time.time() - self.session_start)
        if duration_sec < 5:
            return

        compliance_ratio = (self.good_frames_tracked / self.total_frames_tracked) if self.total_frames_tracked > 0 else 1.0
        ergo_score = max(0, min(100, int(compliance_ratio * 100)))
        duration_min = round(duration_sec / 60.0, 1)
        avg_bpm = int(self.blink_count / (duration_sec / 60.0)) if duration_sec > 10 else self.blink_count

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": duration_min,
            "ergonomic_score": ergo_score,
            "total_blinks": self.blink_count,
            "avg_blink_rate": avg_bpm,
            "slouch_warnings": self.slouch_warning_count,
            "proximity_warnings": self.proximity_warning_count,
            "fatigue_warnings": self.fatigue_warning_count
        }

        # Save to SQLite Database
        database.log_session(record)

        # Show Session Modal
        self._show_modal_summary(record)

    def _show_modal_summary(self, record):
        modal = ctk.CTkToplevel(self)
        modal.title("Session Complete")
        modal.geometry("380x420")
        modal.resizable(False, False)
        modal.attributes("-topmost", True)

        ctk.CTkLabel(modal, text="Session Summary", font=("Segoe UI", 20, "bold")).pack(pady=(20, 5))
        ctk.CTkLabel(modal, text=f"Saved to Database: {record['timestamp']}", font=("Segoe UI", 11), text_color="gray").pack(pady=(0, 15))

        score_color = "#2a9d8f" if record["ergonomic_score"] >= 80 else "#e76f51" if record["ergonomic_score"] >= 60 else "#e63946"
        ctk.CTkLabel(modal, text=f"{record['ergonomic_score']} / 100", font=("Segoe UI", 36, "bold"), text_color=score_color).pack(pady=5)
        ctk.CTkLabel(modal, text="Ergonomic Health Score", font=("Segoe UI", 12)).pack(pady=(0, 20))

        info_frame = ctk.CTkFrame(modal, corner_radius=8, fg_color="#242424")
        info_frame.pack(fill="x", padx=25, pady=5)

        stats = [
            ("Duration", f"{record['duration_minutes']} min"),
            ("Total Blinks", str(record["total_blinks"])),
            ("Average Blink Rate", f"{record['avg_blink_rate']} /min"),
            ("Posture Slouches", str(record["slouch_warnings"])),
            ("Proximity Warnings", str(record["proximity_warnings"]))
        ]

        for label, val in stats:
            row = ctk.CTkFrame(info_frame, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=4)
            ctk.CTkLabel(row, text=label, font=("Segoe UI", 12), text_color="#aaaaaa").pack(side="left")
            ctk.CTkLabel(row, text=val, font=("Segoe UI", 12, "bold")).pack(side="right")

        ctk.CTkButton(modal, text="Done", command=modal.destroy, font=("Segoe UI", 13, "bold"), height=35).pack(pady=20)

    def open_monthly_report(self):
        """Displays aggregated SQL analytics and clinical remarks."""
        report = database.generate_monthly_report()
        
        modal = ctk.CTkToplevel(self)
        modal.title("Client Health & Ergonomics Report")
        modal.geometry("450x520")
        modal.resizable(False, False)
        modal.attributes("-topmost", True)

        ctk.CTkLabel(modal, text="Aggregated Health Report", font=("Segoe UI", 20, "bold")).pack(pady=(18, 4))
        ctk.CTkLabel(modal, text="Workstation Telemetry & Behavioral Analytics", font=("Segoe UI", 11), text_color="gray").pack(pady=(0, 12))

        if not report:
            ctk.CTkLabel(modal, text="No completed sessions logged yet in database.", font=("Segoe UI", 13)).pack(pady=40)
            ctk.CTkButton(modal, text="Close", command=modal.destroy, width=100).pack()
            return

        summary_card = ctk.CTkFrame(modal, corner_radius=8, fg_color="#242424")
        summary_card.pack(fill="x", padx=20, pady=5)

        metrics = [
            ("Total Monitored Sessions", str(report["total_sessions"])),
            ("Cumulative Time Monitored", f"{report['total_hours']} hours"),
            ("Average Ergonomic Score", f"{report['avg_score']} / 100"),
            ("Average Blink Frequency", f"{report['avg_blink']} /min"),
            ("Total Slouch Incidents", str(report["total_slouch"])),
            ("Total Proximity Violations", str(report["total_proximity"]))
        ]

        for label, val in metrics:
            row = ctk.CTkFrame(summary_card, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=3)
            ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color="#aaaaaa").pack(side="left")
            ctk.CTkLabel(row, text=val, font=("Segoe UI", 11, "bold")).pack(side="right")

        ctk.CTkLabel(modal, text="Clinical & Behavioral Remarks", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=25, pady=(15, 5))
        
        remarks_frame = ctk.CTkFrame(modal, corner_radius=8, fg_color="#1e1e24")
        remarks_frame.pack(fill="x", padx=20, pady=5)

        for remark in report["remarks"]:
            lbl = ctk.CTkLabel(remarks_frame, text=f"• {remark}", font=("Segoe UI", 11), wraplength=380, justify="left", text_color="#dcdcdc")
            lbl.pack(anchor="w", padx=12, pady=4)

        ctk.CTkButton(modal, text="Export / Done", command=modal.destroy, font=("Segoe UI", 13, "bold"), height=35).pack(pady=15)

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = ErgoVisionProApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()