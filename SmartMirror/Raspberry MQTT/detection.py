import cv2
import mediapipe as mp

from SmartMirror.detection_models.push_up_model import PushUpCounter
from SmartMirror.detection_models.squat_model import SquatCounter


STREAM_URL = 'http://192.168.2.197/stream'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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

        pushup_counter.update(landmarks, w, h)
        squat_counter.update(landmarks, w, h)

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
