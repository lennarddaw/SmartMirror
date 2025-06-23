#!/usr/bin/env python3
"""
Data Processing Module

Handles feature extraction, dataset building, and data augmentation for both exercise repetition 
segmentation and exercise type classification.
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from video_labeler import VideoSegmenter


class LabelAugmenter:
    """
    Augments binary labels by expanding rep-start markers in a temporal window.
    
    This helps the model learn the temporal context around repetition starts.
    """
    
    def __init__(self, fps: int = 30, margin_sec: float = 0.1):
        """
        Initialize the label augmenter.
        
        Args:
            fps: Frames per second of the videos
            margin_sec: Time margin in seconds to expand around labels
        """
        self.margin = int(fps * margin_sec)

    def augment(self, labels: np.ndarray) -> np.ndarray:
        """
        Expand binary labels by adding margin around positive samples.
        
        Args:
            labels: Binary array where 1 indicates repetition start
            
        Returns:
            Augmented labels with expanded positive regions
        """
        aug = labels.copy()
        ones = np.where(labels == 1.0)[0]
        
        for idx in ones:
            start = max(0, idx - self.margin)
            end = min(len(labels), idx + self.margin + 1)
            aug[start:end] = 1.0
            
        return aug


class FeatureExtractor:
    """
    Extracts pose-based features from video frames using MediaPipe.
    
    Calculates 25 joint angles from pose landmarks to represent body posture.
    """
    
    def __init__(self):
        """Initialize MediaPipe pose detector and define joint angle calculations."""
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)
        
        # Relevant landmark indices for angle calculation
        self.relevant_ids = [
            2,   # left eye
            5,   # right eye
            11, 12, 13, 14, 15, 16,         # shoulders, elbows, wrists
            17, 18, 19, 20, 21, 22,         # fingers and thumbs
            23, 24, 25, 26, 27, 28,         # hips, knees, ankles
            29, 30, 31, 32                  # heels, foot indices
        ]

        # Angle triplets: (a, b, c) where angle is at point b between lines (a-b) and (b-c)
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

    def extract_angles(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Extract joint angles from a video frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Array of 25 normalized joint angles or None if no pose detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        
        if not res.pose_landmarks:
            return None
            
        # Extract relevant landmarks
        lm_map = [res.pose_landmarks.landmark[idx] for idx in self.relevant_ids]
        
        # Calculate angles for each triplet
        angles = []
        for a_id, b_id, c_id in self.angle_triplets:
            a = lm_map[a_id]
            b = lm_map[b_id]
            c = lm_map[c_id]
            
            # Calculate distances
            ab = np.hypot(a.x - b.x, a.y - b.y)
            bc = np.hypot(b.x - c.x, b.y - c.y)
            ac = np.hypot(a.x - c.x, a.y - c.y)
            
            # Calculate angle using cosine law
            if ab == 0 or bc == 0:
                angle = 0.0
            else:
                cos_val = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
                angle = float(np.arccos(np.clip(cos_val, -1.0, 1.0))) / np.pi  # Normalize to [0, 1]
                
            angles.append(angle)
            
        return np.array(angles) if angles else None

    def get_feature_dimension(self) -> int:
        """Get the number of features extracted per frame."""
        return len(self.angle_triplets)


class SegmentationDatasetBuilder:
    """
    Builds training datasets for exercise repetition segmentation.
    
    Processes videos, extracts features, and creates labeled datasets for training
    exercise-specific repetition detection models.
    """
    
    def __init__(self, videos_dir: str = "data/raw", labels_dir: str = "data/labels", fps: int = 30):
        """
        Initialize the segmentation dataset builder.
        
        Args:
            videos_dir: Directory containing training videos
            labels_dir: Directory containing manual labels
            fps: Frames per second for temporal calculations
        """
        self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        self.augmenter = LabelAugmenter(fps)
        self.extractor = FeatureExtractor()
        self.segmenter = VideoSegmenter(videos_dir=videos_dir, labels_dir=labels_dir)

    def build(self, exercise_type: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Build segmentation dataset from labeled videos for a specific exercise.
        
        Args:
            exercise_type: Specific exercise type to process
            
        Returns:
            Tuple of (features, labels) arrays
        """
        X_all, y_all = [], []
        
        # Get video directory for this exercise
        video_dir = os.path.join(self.videos_dir, exercise_type)
        if not os.path.exists(video_dir):
            raise ValueError(f"Video directory not found: {video_dir}")
        
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"Processing {len(video_files)} videos for {exercise_type}...")
        
        for video_file in video_files:
            try:
                # Load labeled frames
                labels, frames = zip(*self.segmenter.get_labeled_frames(f"{exercise_type}/{video_file}"))
                labels = self.augmenter.augment(np.array(labels))

                # Extract features for each frame
                for idx, frame in enumerate(frames):
                    feat = self.extractor.extract_angles(frame)
                    if feat is not None:
                        X_all.append(feat)
                        y_all.append(labels[idx])

                print(f"Processed {video_file}: {len(frames)} frames")
                
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

        if not X_all:
            raise ValueError("No valid features extracted from videos")

        X = np.stack(X_all)  # shape: (total_frames, num_features)
        y = np.array(y_all)  # shape: (total_frames,)
        
        print(f"Built segmentation dataset: {X.shape[0]} frames with {X.shape[1]} features each")
        print(f"Positive samples: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
        
        return X, y

    def save(self, X: np.ndarray, y: np.ndarray, path: str = 'dataset.npz'):
        """
        Save dataset to NPZ file.
        
        Args:
            X: Feature array
            y: Label array
            path: Output file path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        np.savez(path, X=X, y=y)
        print(f"Segmentation dataset saved to {path}")


class ClassificationDatasetBuilder:
    """
    Builds training datasets for exercise type classification.
    
    Processes all videos from all exercise types to create a unified classification dataset.
    """
    
    def __init__(self, videos_dir: str = "data/raw", fps: int = 30):
        """
        Initialize the classification dataset builder.
        
        Args:
            videos_dir: Directory containing training videos organized by exercise type
            fps: Frames per second for temporal calculations
        """
        self.videos_dir = videos_dir
        self.extractor = FeatureExtractor()

    def build(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Build classification dataset from all exercise videos.
        
        Returns:
            Tuple of (features, labels, label_mapping) where:
            - features: Array of shape (total_frames, num_features)
            - labels: Array of shape (total_frames,) with integer labels
            - label_mapping: Dict mapping exercise names to integer labels
        """
        X_all, y_all = [], []
        label_mapping = {}
        current_label = 0
        
        # Get all exercise types (subdirectories)
        exercise_types = [d for d in os.listdir(self.videos_dir) 
                         if os.path.isdir(os.path.join(self.videos_dir, d))]
        exercise_types.sort()  # Ensure consistent ordering
        
        print(f"Found {len(exercise_types)} exercise types: {exercise_types}")
        
        for exercise_type in exercise_types:
            label_mapping[exercise_type] = current_label
            video_dir = os.path.join(self.videos_dir, exercise_type)
            video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            
            print(f"Processing {len(video_files)} videos for {exercise_type} (label: {current_label})...")
            
            for video_file in video_files:
                try:
                    video_path = os.path.join(video_dir, video_file)
                    cap = cv2.VideoCapture(video_path)
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        feat = self.extractor.extract_angles(frame)
                        if feat is not None:
                            X_all.append(feat)
                            y_all.append(current_label)
                    
                    cap.release()
                    print(f"  Processed {video_file}")
                    
                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
                    continue
            
            current_label += 1

        if not X_all:
            raise ValueError("No valid features extracted from videos")

        X = np.stack(X_all)  # shape: (total_frames, num_features)
        y = np.array(y_all)  # shape: (total_frames,)
        
        print(f"Built classification dataset: {X.shape[0]} frames with {X.shape[1]} features each")
        print(f"Exercise distribution:")
        for exercise, label in label_mapping.items():
            count = np.sum(y == label)
            print(f"  {exercise}: {count} frames ({count/len(y)*100:.1f}%)")
        
        return X, y, label_mapping

    def save(self, X: np.ndarray, y: np.ndarray, label_mapping: dict, path: str = 'classification_dataset.npz'):
        """
        Save classification dataset to NPZ file.
        
        Args:
            X: Feature array
            y: Label array
            label_mapping: Mapping of exercise names to integer labels
            path: Output file path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save both data and label mapping
        np.savez(path, X=X, y=y, label_mapping=label_mapping)
        print(f"Classification dataset saved to {path}")
        print(f"Label mapping: {label_mapping}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Build training datasets for exercise detection'
    )
    parser.add_argument('--mode', type=str, required=True, choices=['segmentation', 'classification'],
                        help='Dataset mode: segmentation or classification')
    parser.add_argument('--exercise', type=str, default=None,
                        help='Exercise type for segmentation mode (e.g., push-ups, squats)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output dataset path')
    args = parser.parse_args()
    
    try:
        if args.mode == 'segmentation':
            if not args.exercise:
                raise ValueError("Exercise type required for segmentation mode")
            
            # Build segmentation dataset
            if args.output is None:
                args.output = f'data/processed/dataset_{args.exercise}.npz'
            
            builder = SegmentationDatasetBuilder()
            X, y = builder.build(exercise_type=args.exercise)
            builder.save(X, y, path=args.output)
            
        elif args.mode == 'classification':
            # Build classification dataset
            if args.output is None:
                args.output = 'data/processed/classification_dataset.npz'
            
            builder = ClassificationDatasetBuilder()
            X, y, label_mapping = builder.build()
            builder.save(X, y, label_mapping, path=args.output)
        
        print(f"Successfully created {args.mode} dataset: {args.output}")
        
    except Exception as e:
        print(f"Error building dataset: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
