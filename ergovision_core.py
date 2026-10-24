import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize MediaPipe tools
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def play_alert_sound():
    """Plays an alert beep in a non-blocking background thread."""
    def sound_worker():
        winsound.Beep(880, 250)  # 880 Hz for 250 ms
    threading.Thread(target=sound_worker, daemon=True).start()

def calculate_angle(p1, p2):
    """Calculates angle of line (p1 -> p2) relative to horizontal."""
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:
        return 90.0
    return np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

def get_ear(landmarks, eye_indices, img_w, img_h):
    """Calculates Eye Aspect Ratio (EAR)."""
    coords = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = coords

    v1 = np.linalg.norm(np.array(p2) - np.array(p6))
    v2 = np.linalg.norm(np.array(p3) - np.array(p5))
    h = np.linalg.norm(np.array(p1) - np.array(p4))

    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

def run_ergovision():
    cap = cv2.VideoCapture(1)  # Index 1 verified for your hardware
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Posture thresholding
    POSTURE_ANGLE_THRESHOLD = 8.0
    SLOUCH_TIME_LIMIT = 3.0  # seconds before alert triggers
    slouch_start_time = None
    posture_alert_active = False

    # Fatigue / Blink parameters
    EAR_THRESHOLD = 0.21
    CONSECUTIVE_FRAMES = 2
    frame_counter = 0
    blink_count = 0
    session_start_time = time.time()
    last_alert_time = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        print("ErgoVision Core running. Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            current_time = time.time()

            # Process inference in RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            pose_results = pose.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)

            rgb_frame.flags.writeable = True

            # ----------------- 1. Posture Processing -----------------
            shoulder_angle = 0.0
            is_slouching = False

            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark
                l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                p1 = (int(l_sh.x * w), int(l_sh.y * h))
                p2 = (int(r_sh.x * w), int(r_sh.y * h))
                shoulder_angle = calculate_angle(p1, p2)

                is_slouching = shoulder_angle > POSTURE_ANGLE_THRESHOLD

                # Posture duration filter
                if is_slouching:
                    if slouch_start_time is None:
                        slouch_start_time = current_time
                    elif current_time - slouch_start_time >= SLOUCH_TIME_LIMIT:
                        posture_alert_active = True
                        if current_time - last_alert_time > 4.0:
                            play_alert_sound()
                            last_alert_time = current_time
                else:
                    slouch_start_time = None
                    posture_alert_active = False

                # Draw shoulder indicator
                line_color = (0, 0, 255) if is_slouching else (0, 255, 0)
                cv2.line(frame, p1, p2, line_color, 3)
                cv2.circle(frame, p1, 5, (255, 0, 0), -1)
                cv2.circle(frame, p2, 5, (255, 0, 0), -1)

            # ----------------- 2. Eye & Blink Processing -----------------
            avg_ear = 0.0
            if face_results.multi_face_landmarks:
                f_lm = face_results.multi_face_landmarks[0].landmark
                l_ear = get_ear(f_lm, LEFT_EYE, w, h)
                r_ear = get_ear(f_lm, RIGHT_EYE, w, h)
                avg_ear = (l_ear + r_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    frame_counter += 1
                else:
                    if frame_counter >= CONSECUTIVE_FRAMES:
                        blink_count += 1
                    frame_counter = 0

            # ----------------- 3. HUD & Visual Feedback -----------------
            elapsed_min = (current_time - session_start_time) / 60.0
            bpm = int(blink_count / elapsed_min) if elapsed_min > 0.1 else blink_count

            # Semi-transparent HUD overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 140), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

            # Metrics
            p_status = "TILTED" if is_slouching else "GOOD"
            p_color = (0, 0, 255) if is_slouching else (0, 255, 0)
            cv2.putText(frame, f"Posture: {p_status} ({shoulder_angle:.1f} deg)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Total Blinks: {blink_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Blink Rate: ~{bpm} /min", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Warning banner across top if slouching persists
            if posture_alert_active:
                cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 220), -1)
                cv2.putText(frame, "WARNING: PLEASE SIT UPRIGHT!", (w // 2 - 190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("ErgoVision - Unified System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_ergovision()