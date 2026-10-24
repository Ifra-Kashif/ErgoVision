import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose tools
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(p1, p2):
    """
    Calculates the absolute angle (in degrees) of the line 
    connecting p1 and p2 relative to the horizontal axis.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    if x2 - x1 == 0:
        return 90.0

    radians = np.arctan2(abs(y2 - y1), abs(x2 - x1))
    angle = np.degrees(radians)
    return angle

def run_posture_tracker():
    # Use index 1 (your working physical camera)
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera on index 1.")
        return

    # Initialize Pose model
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        print("Posture tracking active. Press 'q' in the video window to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert to RGB for MediaPipe inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            results = pose.process(rgb_frame)

            # Back to BGR for OpenCV
            rgb_frame.flags.writeable = True

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract Left Shoulder (11) and Right Shoulder (12)
                l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                # Convert normalized [0.0, 1.0] to pixel coordinates
                l_pos = (int(l_shoulder.x * w), int(l_shoulder.y * h))
                r_pos = (int(r_shoulder.x * w), int(r_shoulder.y * h))

                # Calculate tilt angle
                shoulder_angle = calculate_angle(l_pos, r_pos)

                # Posture health logic
                if shoulder_angle <= 8.0:
                    status = "Good Posture"
                    color = (0, 255, 0)  # Green
                else:
                    status = "Slouching / Tilted"
                    color = (0, 0, 255)  # Red

                # Draw landmarks and skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Connect shoulders with indicator line
                cv2.line(frame, l_pos, r_pos, color, 3)
                cv2.circle(frame, l_pos, 6, (255, 0, 0), -1)
                cv2.circle(frame, r_pos, 6, (255, 0, 0), -1)

                # HUD Overlay
                cv2.putText(
                    frame,
                    f"Status: {status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )
                cv2.putText(
                    frame,
                    f"Shoulder Tilt: {shoulder_angle:.1f} deg",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            cv2.imshow("ErgoVision - Posture Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_posture_tracker()