import numpy as np
import mediapipe as mp

class PushUpCounter:
    def __init__(self):
        self.count = 0
        self.stage = None

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def update(self, landmarks, w, h):
        mp_pose = mp.solutions.pose
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

        angle = self.calculate_angle(shoulder, elbow, wrist)

        if angle < 90:
            self.stage = "down"
        if angle > 160 and self.stage == "down":
            self.stage = "up"
            self.count += 1
