import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Landmark indices for Left and Right Eyes (MediaPipe standard)
# Format: [p1 (outer), p2 (top-left), p3 (top-right), p4 (inner), p5 (bottom-right), p6 (bottom-left)]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    """Calculates 2D Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_ear(landmarks, eye_indices, img_w, img_h):
    """Calculates the Eye Aspect Ratio (EAR) for a given set of eye landmarks."""
    # Convert normalized landmark coordinates to pixel coordinates
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * img_w), int(lm.y * img_h)))

    p1, p2, p3, p4, p5, p6 = coords

    # Compute vertical distances
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)

    # Compute horizontal distance
    h = euclidean_distance(p1, p4)

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear

def run_fatigue_tracker():
    cap = cv2.VideoCapture(1)  # Index 1 verified for your webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Blink detection state variables
    EAR_THRESHOLD = 0.21
    CONSECUTIVE_FRAMES = 2
    frame_counter = 0
    blink_count = 0

    # Timer for Blinks Per Minute (BPM)
    start_time = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        print("Fatigue Engine running. Blink to test counter. Press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if results.multi_face_landmarks:
                mesh_landmarks = results.multi_face_landmarks[0].landmark

                # Calculate EAR for both eyes and take the average
                left_ear = get_ear(mesh_landmarks, LEFT_EYE, w, h)
                right_ear = get_ear(mesh_landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # Blink detection logic
                if avg_ear < EAR_THRESHOLD:
                    frame_counter += 1
                else:
                    if frame_counter >= CONSECUTIVE_FRAMES:
                        blink_count += 1
                    frame_counter = 0

                # Calculate elapsed time and approximate Blinks Per Minute
                elapsed_minutes = (time.time() - start_time) / 60.0
                bpm = int(blink_count / elapsed_minutes) if elapsed_minutes > 0.1 else blink_count

                # Visual feedback on eye state
                eye_color = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)
                eye_state = "Closed" if avg_ear < EAR_THRESHOLD else "Open"

                # Draw minimal eye circles for visual confirmation
                for idx in LEFT_EYE + RIGHT_EYE:
                    lm = mesh_landmarks[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, eye_color, -1)

                # HUD Overlay
                cv2.putText(
                    frame,
                    f"EAR: {avg_ear:.2f} ({eye_state})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    eye_color,
                    2
                )
                cv2.putText(
                    frame,
                    f"Blinks: {blink_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"Blinks/Min: ~{bpm}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            cv2.imshow("ErgoVision - Eye Fatigue Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_fatigue_tracker()