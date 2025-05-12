import cv2
import mediapipe as mp
import numpy as np
from collections import deque

STREAM_URL = 'http://10.5.16.33/stream'

pushup_count     = 0
stage            = None
angle_buffer     = deque(maxlen=5)
is_not_straight  = False

# Mediapipe Setup
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(min_detection_confidence=0.5,
                         min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def smoothed_angle(current_angle):
    angle_buffer.append(current_angle)
    return sum(angle_buffer) / len(angle_buffer)

def set_alarm(output, text="Oberkoerper gerade du fauler Sack!"):

    cv2.putText(output, text, (10,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

def detect_pushup_combined(landmarks, output):
    global pushup_count, stage, is_not_straight

    elbow_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
    )

    body_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
    )

    body_angle = smoothed_angle(body_angle)
    elbow_angle = smoothed_angle(elbow_angle)

    is_not_straight = body_angle < 130
    is_straight     = body_angle > 165
    is_level        = abs(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y -
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    ) < 0.10
    is_down         = elbow_angle < 80
    is_up           = elbow_angle > 100

    if is_not_straight:
        set_alarm(output)

    if is_down and is_straight and is_level:
        stage = "down"
    elif is_up and stage == "down" and is_straight and is_level:
        stage = "up"
        pushup_count += 1

# — Haupt-Loop —
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"Error: Could not open stream {STREAM_URL}")
    exit(1)

cv2.namedWindow('Pose & Gesture Analysis', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose & Gesture Analysis', 640, 480)

print("Starting posture & gesture analysis. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    output  = frame.copy()

    if results.pose_landmarks:
        detect_pushup_combined(results.pose_landmarks.landmark, output)

        mp_drawing.draw_landmarks(output,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec((0,255,0),2,2),
                                  mp_drawing.DrawingSpec((0,0,255),2,2))
        cv2.putText(output, f"Push-ups: {pushup_count}", (10,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('Pose & Gesture Analysis', output)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
