# Python script to capture ESP32-CAM MJPEG stream,
# perform pose estimation with MediaPipe, and analyze basic posture & gestures.

import cv2
import mediapipe as mp
import numpy as np

# -- Configuration nur temporär, ändert sich -----------------------------------------------------------
STREAM_URL = 'http://192.168.2.197/stream'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_facing(landmarks, w, h):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    
    delta_x = right_shoulder[0] - left_shoulder[0]
    delta_y = right_shoulder[1] - left_shoulder[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    if angle > 15:
        return 'facing right'
    if angle < -15:
        return 'facing left'
    return 'facing forward'

def detect_hand_raise(landmarks, w, h):
    results = {'left': False, 'right': False}
    for side in ['LEFT', 'RIGHT']:
        shoulder = landmarks[getattr(mp_pose.PoseLandmark, f'{side}_SHOULDER').value]
        wrist = landmarks[getattr(mp_pose.PoseLandmark, f'{side}_WRIST').value]

        if wrist.y * h < shoulder.y * h - 20:
            results[side.lower()] = True
    return results

# Start video capture
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"Error: Could not open stream {STREAM_URL}")
    exit(1)

cv2.namedWindow('Pose & Gesture Analysis', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose & Gesture Analysis', 640, 480)
cv2.startWindowThread()

print("Starting posture & gesture analysis. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # 1) Facing direction
        facing = get_facing(landmarks, w, h)

        # 2) Hand raise detection
        hands = detect_hand_raise(landmarks, w, h)
        status = []
        if hands['left']:
            status.append('Left hand raised')
        if hands['right']:
            status.append('Right hand raised')
        if not status:
            status.append('No hands raised')

        mp_drawing.draw_landmarks(output,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))

        cv2.putText(output, f"Orientation: {facing}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(output, f"Status: {', '.join(status)}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('Pose & Gesture Analysis', output)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
