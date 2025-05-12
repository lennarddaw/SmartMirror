import cv2
import mediapipe as mp
import numpy as np
from collections import deque

STREAM_URL = "http://10.5.16.33/stream"
squat_count = 0
stage = None
angle_buffer = deque(maxlen=5)

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

def smooth_angle(angle):
    angle_buffer.append(angle)
    return sum(angle_buffer) / len(angle_buffer)
# ——————————————————————————————————————————————————————————————

def detect_squat_combined(landmarks):
    global squat_count, stage

    hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    ]
    knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    ]
    ankle = [
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    ]
    shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    ]

    hip_angle   = calculate_angle(shoulder, hip, knee)
    knee_angle  = calculate_angle(hip, knee, ankle)

    hip_angle   = smooth_angle(hip_angle)
    knee_angle  = smooth_angle(knee_angle)


    is_down = knee_angle < 90
    is_up   = knee_angle > 160

    if is_down and stage != "down":
        stage = "down"
    elif is_up and stage == "down":
        stage = "up"
        squat_count += 1

# ——————————————————————————————————————————————————————————————
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"Error: Could not open stream {STREAM_URL}")
    exit(1)

cv2.namedWindow('Pose & Gesture Analysis', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose & Gesture Analysis', 640, 480)

print("Starting posture & gesture analysis. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        detect_squat_combined(landmarks)


        mp_drawing.draw_landmarks(output,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)

        cv2.putText(output, f"Squats: {squat_count}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('Pose & Gesture Analysis', output)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
