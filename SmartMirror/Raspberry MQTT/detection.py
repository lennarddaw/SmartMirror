import cv2
import mediapipe as mp
import numpy as np
from collections import deque


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

class PushUpCounter:
    def __init__(self, buffer_size=30):
        self.count = 0
        self.stage = None
        self.angle_buffer = deque(maxlen=buffer_size)

    def update(self, landmarks, w, h):
        mp_pose = mp.solutions.pose
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

        angle = calculate_angle(shoulder, elbow, wrist)
        self.angle_buffer.append(angle)

        # Dynamische Thresholds
        min_a = min(self.angle_buffer)
        max_a = max(self.angle_buffer)
        range_a = max_a - min_a if max_a > min_a else 1.0
        down_thr = min_a + 0.2 * range_a
        up_thr = max_a - 0.2 * range_a

        if angle < down_thr:
            self.stage = 'down'
        if angle > up_thr and self.stage == 'down':
            self.stage = 'up'
            self.count += 1
        return angle

class SquatCounter:
    def __init__(self, buffer_size=30):
        self.count = 0
        self.stage = None
        self.angle_buffer = deque(maxlen=buffer_size)

    def update(self, landmarks, w, h):
        mp_pose = mp.solutions.pose
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

        angle = calculate_angle(hip, knee, ankle)
        self.angle_buffer.append(angle)

        # Dynamische Thresholds
        min_a = min(self.angle_buffer)
        max_a = max(self.angle_buffer)
        range_a = max_a - min_a if max_a > min_a else 1.0
        down_thr = min_a + 0.2 * range_a
        up_thr = max_a - 0.2 * range_a

        if angle < down_thr:
            self.stage = 'down'
        if angle > up_thr and self.stage == 'down':
            self.stage = 'up'
            self.count += 1
        return angle

# Übungserkennung per Winkel

def detect_exercise_type(elbow_angle, knee_angle):
    if elbow_angle < 150 and knee_angle > 160:
        return 'squat'
    if knee_angle < 150 and elbow_angle > 160:
        return 'push_up'
    return None

# Geste: Swipe-Right mit linkem Handgelenk

def detect_swipe_right(landmarks, w, h, buffer, threshold=100):
    mp_pose = mp.solutions.pose
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    x = int(lw.x * w)
    y = int(lw.y * h)
    buffer.append((x, y))
    if len(buffer) == buffer.maxlen:
        x_start, y_start = buffer[0]
        x_end, y_end = buffer[-1]
        buffer.clear()
        if x_end - x_start > threshold and abs(y_end - y_start) < threshold / 2:
            return True
    return False

STREAM_URL = 'http://192.168.2.197/stream'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

gesture_buffer = deque(maxlen=10)
pushup_counter = PushUpCounter()
squat_counter = SquatCounter()

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

        # Winkel anzeigen
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        knee_angle = calculate_angle(hip, knee, ankle)

        cv2.putText(output, f"Elbow: {elbow_angle:.1f}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(output, f"Knee: {knee_angle:.1f}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Reset-Geste
        if detect_swipe_right(landmarks, w, h, gesture_buffer):
            pushup_counter.count = 0
            pushup_counter.stage = None
            squat_counter.count = 0
            squat_counter.stage = None
            cv2.putText(output, "Counters reset", (int(w/2)-100, int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # Übungserkennung & Zählen
            exercise = detect_exercise_type(elbow_angle, knee_angle)
            if exercise == 'push_up':
                pushup_counter.update(landmarks, w, h)
            elif exercise == 'squat':
                squat_counter.update(landmarks, w, h)

        # Landmarks & Zähler-Display
        mp_drawing.draw_landmarks(output,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)
        cv2.putText(output, f"Push-ups: {pushup_counter.count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output, f"Squats: {squat_counter.count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Pose & Gesture Analysis', output)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
