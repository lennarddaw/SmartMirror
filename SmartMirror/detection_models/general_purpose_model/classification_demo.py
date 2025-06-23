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
    
    # Load class names
    class_names_path = path.replace('.keras', '_classes.txt')
    class_names = []
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Default class names if file doesn't exist
        class_names = ['push-ups', 'squats', 'pull-ups', 'dips']
    
    return model, class_names


def compute_sequence_classification(
    video_path: str,
    model: tf.keras.Model,
    extractor: FeatureExtractor,
    sequence_length: int = 140
) -> Tuple[str, List[float], List[np.ndarray]]:
    """
    1) Read frames from the video.
    2) Extract feature-vectors (pose angles) for each frame.
    3) Create sequences of specified length.
    4) Return the predicted exercise type, confidence scores, and all probabilities.
    """
    cap = cv2.VideoCapture(video_path)
    features: List[np.ndarray] = []

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

    # If the video was empty or extractor never produced a feature, return default
    if len(features) == 0:
        return "unknown", [0.0, 0.0, 0.0, 0.0], []

    # Pad or truncate to sequence_length
    if len(features) < sequence_length:
        # Pad with zeros if video is too short
        pad_length = sequence_length - len(features)
        features.extend([np.zeros_like(features[0])] * pad_length)
    elif len(features) > sequence_length:
        # Truncate if video is too long (take middle portion)
        start_idx = (len(features) - sequence_length) // 2
        features = features[start_idx:start_idx + sequence_length]

    # Create sequence for prediction
    sequence = np.stack(features, axis=0)
    sequence = sequence[None, ...]  # add batch dimension
    
    # Get prediction
    probabilities = model.predict(sequence, verbose=0)
    predicted_class = np.argmax(probabilities[0])
    confidence = float(probabilities[0][predicted_class])

    return predicted_class, confidence, probabilities[0].tolist()


def plot_classification_results(
    probabilities: List[float],
    class_names: List[str],
    predicted_class: int,
    confidence: float,
    output_path: str
) -> None:
    """
    Plot the classification probabilities as a bar chart and save as PNG.
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(class_names, probabilities, color='skyblue', alpha=0.7)
    
    # Highlight the predicted class
    bars[predicted_class].set_color('red')
    bars[predicted_class].set_alpha(0.8)
    
    # Add probability values on bars
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        plt.text(i, prob + 0.01, f'{prob:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Exercise Classification Results\nPredicted: {class_names[predicted_class]} (Confidence: {confidence:.3f})')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def overlay_classification(
    input_video: str,
    output_path: str,
    predicted_class: str,
    confidence: float,
    probabilities: List[float],
    class_names: List[str],
    font_scale: float = 1.0,
    thickness: int = 2
) -> None:
    """
    Overlay classification results onto each frame and write to a new video.
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
        
        # Create overlay text
        overlay_text = [
            f"Exercise: {predicted_class}",
            f"Confidence: {confidence:.3f}",
            "",
            "All Probabilities:"
        ]
        
        # Add all class probabilities
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            marker = "✓" if class_name == predicted_class else " "
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
                if class_name == predicted_class:
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
        predicted_class_idx, confidence, probabilities = compute_sequence_classification(
            args.input, model, extractor, args.sequence_length
        )
        
        predicted_class = class_names[predicted_class_idx]
        print(f"\nClassification Results:")
        print(f"  Predicted Exercise: {predicted_class}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  All Probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            marker = "✓" if i == predicted_class_idx else " "
            print(f"    {marker} {class_name}: {prob:.3f}")
        
        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        annotated_path = os.path.join(ANNOTATED_DIR, f"{base_name}_classified.mp4")
        plot_path = os.path.join(PLOT_DIR, f"{base_name}_classification.png")
        
        # Create annotated video
        print(f"\nCreating annotated video: {annotated_path}")
        overlay_classification(
            args.input, annotated_path, predicted_class, confidence,
            probabilities, class_names
        )
        
        # Create classification plot
        print(f"Creating classification plot: {plot_path}")
        plot_classification_results(
            probabilities, class_names, predicted_class_idx, confidence, plot_path
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