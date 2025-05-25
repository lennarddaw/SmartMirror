import os
import numpy as np
import cv2
import mediapipe as mp
from labeling_tool import VideoSegmenter


class LabelAugmenter:
    """
    Augments binary labels by expanding rep-start markers in a temporal window.
    """
    def __init__(self, fps: int = 30, margin_sec: float = 0.1):
        self.margin = int(fps * margin_sec)

    def augment(self, labels: np.ndarray) -> np.ndarray:
        aug = labels.copy()
        ones = np.where(labels == 1.0)[0]
        for idx in ones:
            start = max(0, idx - self.margin)
            end = min(len(labels), idx + self.margin + 1)
            aug[start:end] = 1.0
        return aug


class FeatureExtractor:
    """
    Uses MediaPipe to extract landmark-based features (angles) from frames.
    """
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)
        
        self.relevant_ids = [
            2,   # left eye
            5,   # right eye
            11, 12, 13, 14, 15, 16,         # shoulders, elbows, wrists
            17, 18, 19, 20, 21, 22,         # fingers and thumbs
            23, 24, 25, 26, 27, 28,         # hips, knees, ankles
            29, 30, 31, 32                  # heels, foot indices
        ]

        self.angle_triplets = [
            # --- Head alignment (neck area approximation) ---
            (2, 0, 3),       # left shoulder - left eye - right shoulder
            (14, 2, 0),      # left hip - left shoulder - left eye
            (15, 3, 1),      # right hip - right shoulder - right eye

            # --- Shoulders ---
            (14, 2, 4),      # left hip - left shoulder - left elbow
            (15, 3, 5),      # right hip - right shoulder - right elbow

            # --- Elbows ---
            (2, 4, 6),       # left shoulder - left elbow - left wrist
            (3, 5, 7),       # right shoulder - right elbow - right wrist

            # --- Wrists (flexion/extension) ---
            (4, 6, 10),      # left elbow - wrist - index
            (5, 7, 11),      # right elbow - wrist - index
            (4, 6, 12),      # left elbow - wrist - thumb
            (5, 7, 13),      # right elbow - wrist - thumb
            (4, 6, 8),       # left elbow - wrist - pinky
            (5, 7, 9),       # right elbow - wrist - pinky

            # --- Spine & Hip connection ---
            (2, 14, 15),     # left shoulder - left hip - right hip
            (3, 15, 14),     # right shoulder - right hip - left hip

            # --- Hips ---
            (2, 14, 16),     # left shoulder - left hip - left knee
            (3, 15, 17),     # right shoulder - right hip - right knee

            # --- Knees ---
            (14, 16, 18),    # left hip - knee - ankle
            (15, 17, 19),    # right hip - knee - ankle

            # --- Ankles / Feet ---
            (16, 18, 22),    # left knee - ankle - foot index
            (17, 19, 23),    # right knee - ankle - foot index
            (18, 20, 22),    # left ankle - heel - foot index
            (19, 21, 23),    # right ankle - heel - foot index
            (20, 18, 16),    # heel - ankle - knee
            (21, 19, 17),    # heel - ankle - knee
        ]
        
        # Filtered landmark indices and names for joint angles:
        #  0 - left eye
        #  1 - right eye
        #  2 - left shoulder
        #  3 - right shoulder
        #  4 - left elbow
        #  5 - right elbow
        #  6 - left wrist
        #  7 - right wrist
        #  8 - left pinky
        #  9 - right pinky
        # 10 - left index
        # 11 - right index
        # 12 - left thumb
        # 13 - right thumb
        # 14 - left hip
        # 15 - right hip
        # 16 - left knee
        # 17 - right knee
        # 18 - left ankle
        # 19 - right ankle
        # 20 - left heel
        # 21 - right heel
        # 22 - left foot index
        # 23 - right foot index

    def extract_angles(self, frame: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        lm_map = [res.pose_landmarks.landmark[idx] for idx in self.relevant_ids]
        angles = []
        for a_id, b_id, c_id in self.angle_triplets:
            a = lm_map[a_id]
            b = lm_map[b_id]
            c = lm_map[c_id]
            ab = np.hypot(a.x - b.x, a.y - b.y)
            bc = np.hypot(b.x - c.x, b.y - c.y)
            ac = np.hypot(a.x - c.x, a.y - c.y)
            if ab == 0 or bc == 0:
                angle = 0.0
            else:
                cos_val = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
                angle = float(np.arccos(np.clip(cos_val, -1.0, 1.0))) / np.pi  # Normalize to [0, 1]
            angles.append(angle)
        return np.array(angles) if angles else None


class DatasetBuilderAll:
    """
    Builds a dataset of features for all frames from all videos in one go.
    """
    def __init__(self, videos_dir: str = "videos", fps: int = 30):
        self.videos_dir = videos_dir
        self.augmenter = LabelAugmenter(fps)
        self.extractor = FeatureExtractor()
        self.segmenter = VideoSegmenter()

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        X_all, y_all = [], []
        for file in os.listdir(self.videos_dir):
            if not file.endswith('.mp4'):
                continue
            # Load labeled frames
            labels, frames = zip(*self.segmenter.get_labeled_frames(file))
            labels = self.augmenter.augment(np.array(labels))

            for idx, frame in enumerate(frames):
                feat = self.extractor.extract_angles(frame)
                if feat is None:
                    continue
                X_all.append(feat)
                y_all.append(labels[idx])

        X = np.stack(X_all)  # shape: (total_frames, num_features)
        y = np.array(y_all)  # shape: (total_frames,)
        print(f"Built features for {X.shape[0]} frames with {X.shape[1]} features each")
        return X, y

    def save(self, X: np.ndarray, y: np.ndarray, path: str = 'dataset.npz'):
        np.savez(path, X=X, y=y)
        print(f"Saved all-frame dataset to {path}")


if __name__ == '__main__':
    builder = DatasetBuilderAll()
    X, y = builder.build()
    builder.save(X, y)
