#!/usr/bin/env python3
"""
Classification Demo

Reads a video, computes exercise type classification, writes annotated frames to a new video,
and generates classification confidence plots.

This demo processes video sequences and classifies the exercise type being performed.
"""

import os
import argparse
from typing import List, Tuple

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_builder import FeatureExtractor

# Output directories
ANNOTATED_DIR = 'output_videos/annotated'
PLOT_DIR = 'output_videos/plots'
for d in (ANNOTATED_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)


def load_model(path: str) -> Tuple[tf.keras.Model, List[str]]:
    """
    Load and return a compiled TensorFlow/Keras model and class names from the given path.
    """
    model = tf.keras.models.load_model(path, compile=False)
    
    # Load class names - try to get proper exercise names
    class_names_path = path.replace('.keras', '_classes.txt')
    class_names = []
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    
    # If we only have numeric classes, map them to exercise names
    if not class_names or all(c.isdigit() for c in class_names):
        # Default exercise names based on the dataset structure (alphabetical order)
        default_names = ['dips', 'pull-ups', 'push-ups', 'squats']
        
        # Get number of classes from model output shape
        try:
            if hasattr(model.output_shape, '__len__') and len(model.output_shape) > 0:
                if isinstance(model.output_shape[-1], int):
                    num_classes = model.output_shape[-1]
                else:
                    num_classes = 4  # fallback
            else:
                num_classes = 4  # fallback
        except:
            num_classes = 4  # fallback
            
        class_names = default_names[:num_classes]
        print(f"Using default exercise names: {class_names}")
    
    return model, class_names


def compute_frame_by_frame_classification(
    video_path: str,
    model: tf.keras.Model,
    extractor: FeatureExtractor,
    sequence_length: int = 140
) -> Tuple[List[int], List[float], List[List[float]]]:
    """
    Compute classification for each frame using sliding window approach.
    
    Returns:
        predicted_classes: List of predicted class indices for each frame
        confidences: List of confidence scores for each frame
        all_probabilities: List of probability distributions for each frame
    """
    cap = cv2.VideoCapture(video_path)
    features: List[np.ndarray] = []

    # Extract features from all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feat = extractor.extract_angles(frame)
        if feat is None:
            # If pose/keypoints not found, substitute a zero-vector
            feat = np.zeros(model.input_shape[-1], dtype=np.float32)
        features.append(feat)
    cap.release()

    if len(features) == 0:
        return [], [], []

    predicted_classes = []
    confidences = []
    all_probabilities = []

    # Use sliding window approach for frame-by-frame classification
    for i in range(len(features)):
        # Create sequence centered on current frame
        start_idx = max(0, i - sequence_length // 2)
        end_idx = min(len(features), start_idx + sequence_length)
        
        # Extract sequence
        sequence = features[start_idx:end_idx]
        
        # Pad if necessary
        if len(sequence) < sequence_length:
            pad_length = sequence_length - len(sequence)
            sequence.extend([np.zeros_like(sequence[0])] * pad_length)
        
        # Prepare for prediction
        sequence = np.stack(sequence, axis=0)
        sequence = sequence[None, ...]  # add batch dimension
        
        # Get prediction
        probabilities = model.predict(sequence, verbose=0)
        predicted_class = np.argmax(probabilities[0])
        confidence = float(probabilities[0][predicted_class])
        
        predicted_classes.append(predicted_class)
        confidences.append(confidence)
        all_probabilities.append(probabilities[0].tolist())

    return predicted_classes, confidences, all_probabilities


def overlay_dynamic_classification(
    input_video: str,
    output_path: str,
    predicted_classes: List[int],
    confidences: List[float],
    all_probabilities: List[List[float]],
    class_names: List[str],
    font_scale: float = 1.0,
    thickness: int = 2
) -> None:
    """
    Overlay dynamic classification results onto each frame and write to a new video.
    """
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count < len(predicted_classes):
            predicted_class = predicted_classes[frame_count]
            confidence = confidences[frame_count]
            probabilities = all_probabilities[frame_count]
            predicted_class_name = class_names[predicted_class]
        else:
            # Handle case where we have more frames than predictions
            predicted_class = predicted_classes[-1] if predicted_classes else 0
            confidence = confidences[-1] if confidences else 0.0
            probabilities = all_probabilities[-1] if all_probabilities else [0.0] * len(class_names)
            predicted_class_name = class_names[predicted_class]
        
        # Create overlay text
        overlay_text = [
            f"Exercise: {predicted_class_name}",
            f"Confidence: {confidence:.3f}",
            "",
            "All Probabilities:"
        ]
        
        # Add all class probabilities
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            marker = "✓" if class_name == predicted_class_name else " "
            overlay_text.append(f"  {marker} {class_name}: {prob:.3f}")
        
        # Draw overlay
        y_offset = 30
        for i, text in enumerate(overlay_text):
            if i == 0:  # Main prediction
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)  # Green if confident, orange if not
                font_scale_current = font_scale * 1.5
                thickness_current = thickness + 1
            elif i == 1:  # Confidence
                color = (255, 255, 255)
                font_scale_current = font_scale
                thickness_current = thickness
            elif i == 2:  # Empty line
                y_offset += 20
                continue
            elif i == 3:  # "All Probabilities" header
                color = (255, 255, 255)
                font_scale_current = font_scale
                thickness_current = thickness
            else:  # Individual probabilities
                if class_name == predicted_class_name:
                    color = (0, 255, 0)  # Green for predicted class
                else:
                    color = (200, 200, 200)  # Gray for others
                font_scale_current = font_scale * 0.8
                thickness_current = thickness - 1
            
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale_current, color, thickness_current)
            y_offset += 30
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (width - 150, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()


def plot_dynamic_classification_results(
    predicted_classes: List[int],
    confidences: List[float],
    all_probabilities: List[List[float]],
    class_names: List[str],
    output_path: str
) -> None:
    """
    Plot dynamic classification results over time.
    """
    if not predicted_classes:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Predicted class over time
    frames = range(len(predicted_classes))
    ax1.plot(frames, predicted_classes, 'b-', linewidth=2, label='Predicted Class')
    ax1.set_ylabel('Exercise Class')
    ax1.set_title('Exercise Classification Over Time')
    ax1.set_yticks(range(len(class_names)))
    ax1.set_yticklabels(class_names)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Confidence over time
    ax2.plot(frames, confidences, 'r-', linewidth=2, label='Confidence')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Classification Confidence Over Time')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the classification demo."""
    parser = argparse.ArgumentParser(
        description='Exercise Classification Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a video with default settings
  python classification_demo.py --input test_videos/push-up_1.mp4 --model models/classification/exercise_classifier.keras
  
  # Classify with custom sequence length
  python classification_demo.py --input test_videos/squat_10.mp4 --model models/classification/exercise_classifier.keras --sequence-length 200
        """
    )
    
    parser.add_argument('--input', required=True,
                       help='Input video file path')
    parser.add_argument('--model', required=True,
                       help='Path to trained classification model (.keras file)')
    parser.add_argument('--sequence-length', type=int, default=140,
                       help='Length of sequence to use for classification (default: 140)')
    parser.add_argument('--output-dir', default='output_videos',
                       help='Output directory for results (default: output_videos)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input video file not found: {args.input}")
        return 1
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train a classification model first using:")
        print("  python main.py train --mode classification")
        return 1
    
    try:
        print(f"Loading classification model: {args.model}")
        model, class_names = load_model(args.model)
        print(f"Loaded model with {len(class_names)} classes: {class_names}")
        
        print(f"Initializing feature extractor...")
        extractor = FeatureExtractor()
        
        print(f"Processing video: {args.input}")
        predicted_classes, confidences, all_probabilities = compute_frame_by_frame_classification(
            args.input, model, extractor, args.sequence_length
        )
        
        print(f"\nClassification Results:")
        for i, (predicted_class, confidence) in enumerate(zip(predicted_classes, confidences)):
            predicted_class_name = class_names[predicted_class]
            print(f"  Frame {i}: Predicted Exercise: {predicted_class_name}, Confidence: {confidence:.3f}")
        
        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        annotated_path = os.path.join(ANNOTATED_DIR, f"{base_name}_classified.mp4")
        plot_path = os.path.join(PLOT_DIR, f"{base_name}_classification.png")
        
        # Create annotated video
        print(f"\nCreating annotated video: {annotated_path}")
        overlay_dynamic_classification(
            args.input, annotated_path, predicted_classes, confidences, all_probabilities, class_names
        )
        
        # Create classification plot
        print(f"Creating classification plot: {plot_path}")
        plot_dynamic_classification_results(
            predicted_classes, confidences, all_probabilities, class_names, plot_path
        )
        
        print(f"\nClassification demo completed successfully!")
        print(f"  Annotated video: {annotated_path}")
        print(f"  Classification plot: {plot_path}")
        
    except Exception as e:
        print(f"Error during classification: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 