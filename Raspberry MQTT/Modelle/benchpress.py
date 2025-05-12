import mediapipe as mp
import cv2
import numpy as np
from collections import deque

benchpress_count = 0
stage = None
angle_buffer = deque(maxlen=5)

STREAM_URL = "http://"

# Mediapipe Setup
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(min_detection_confidence=0.5,
                         min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a,b,c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def smoothed_angle(current_angle):
    angle_buffer.append(current_angle)
    return sum(angle_buffer) / len(angle_buffer)

def detect_benchpress_combined(landmarks, output):
        global stage, benchpress_count
        elbow_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
    )
        elbow_angle = smoothed_angle(elbow_angle)

        is_down = elbow_angle < 90
        is_up = elbow_angle > 100

        if is_down:
             stage = "down"
        elif is_up:
             stage = "up"
             benchpress_count += 1

             cap = cv2.VideoCapture(STREAM_URL)
             if not cap.isOpened():
                  print(f"Error: Unable to open video stream {STREAM_URL}")
                  exit(1)

        
            
        
    