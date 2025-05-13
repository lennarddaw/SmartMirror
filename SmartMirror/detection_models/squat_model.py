import numpy as np
import mediapipe as mp

class SquatCounter:
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
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

        angle = self.calculate_angle(hip, knee, ankle)

        if angle < 90:
            self.stage = "down"
        if angle > 160 and self.stage == "down":
            self.stage = "up"
            self.count += 1
