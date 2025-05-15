import numpy as np
import mediapipe as mp 


class CalfRaisesCounter:

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

        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h,
                ]