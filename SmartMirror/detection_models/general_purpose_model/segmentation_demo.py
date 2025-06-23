#!/usr/bin/env python3
"""
Segmentation Test Demo

Reads a video, computes per-frame probabilities, writes annotated frames to a new video,
performs peak detection on a smoothed probability signal (using prominence + distance only),
and generates probability and peak-detection plots.

Defaults have been tuned as follows:
  • Savitzky–Golay smooth_window = 15
  • Prominence              = 0.15
  • Distance (min frames)   = 25

You can still override at runtime with --smooth, --prom, and --dist flags.
"""
import os
import argparse
from typing import List

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

from dataset_builder import FeatureExtractor

# Output directories
ANNOTATED_DIR = 'output_videos/annotated'
PLOT_DIR      = 'output_videos/plots'
PEAK_DIR      = 'output_videos/peak_detection'
for d in (ANNOTATED_DIR, PLOT_DIR, PEAK_DIR):
    os.makedirs(d, exist_ok=True)


def load_model(path: str) -> tf.keras.Model:
    """
    Load and return a compiled TensorFlow/Keras model from the given path.
    """
    return tf.keras.models.load_model(path, compile=False)


def compute_frame_probabilities(
    video_path: str,
    model: tf.keras.Model,
    extractor: FeatureExtractor,
    window_size: int = 140
) -> List[float]:
    """
    1) Read each frame from the video.
    2) Extract feature-vector (e.g. pose angles) for each frame.
    3) Pad so that each frame can be passed through the TCN with a full `window_size`.
    4) Return a list of per-frame probabilities (floats between 0 and 1).
    """
    cap = cv2.VideoCapture(video_path)
    features: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feat = extractor.extract_angles(frame)
        if feat is None:
            # If pose/keypoints not found, substitute a zero‐vector
            feat = np.zeros(model.input_shape[-1], dtype=np.float32)
        features.append(feat)
    cap.release()

    # If the video was empty or extractor never produced a feature, return empty
    if len(features) == 0:
        return []

    # Pad at the beginning so that the first real frame still sees a full window
    pad = [np.zeros_like(features[0])] * (window_size - 1)
    feats_padded = pad + features

    probs: List[float] = []
    for i in range(len(features)):
        window = np.stack(feats_padded[i : i + window_size], axis=0)
        window = window[None, ...]  # add batch dimension
        pred = model.predict(window, verbose=0)
        # assume model output shape is (1, window_size, 1) and we want the last timestep
        probs.append(float(pred[0, -1, 0]))

    return probs


def plot_probabilities(
    probabilities: List[float],
    fps: float,
    output_path: str
) -> None:
    """
    Plot the raw, per-frame probabilities over time and save as a PNG.
    """
    if len(probabilities) == 0:
        print(f"No probabilities to plot; skipping {output_path}.")
        return

    times = np.arange(len(probabilities)) / fps
    plt.figure(figsize=(10, 4))
    plt.plot(times, probabilities, label='Rep Probability')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.title('Probability over Time')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_rep_starts(
    probabilities: List[float],
    fps: float,
    base_name: str,
    distance: int = 15,
    smooth_window: int = 15,
    polyorder: int = 2,
    prominence: float = 0.15 # 0.15
) -> List[float]:
    """
    Smooth the probability trace using a Savitzky–Golay filter, then detect peaks 
    using only PROMINENCE + DISTANCE. No fixed-height threshold is used here.

    Args:
        probabilities: list of per-frame floats50 in [0,1].
        fps: frames-per-second of the video (so frame-index → time (s)).
        base_name: name used to save the peak‐detection diagnostic plot.
        distance: minimum separation, in frames, between consecutive peaks. Default 25.
        smooth_window: window length for Savitzky–Golay smoothing (must be odd). Default 15.
        polyorder: polynomial order for Savitzky–Golay smoothing. Default 2.
        prominence: minimum prominence a peak must have. Default 0.15.
    Returns:
        timestamps: a list of "rep start" times (in seconds) corresponding to each detected peak.
    """
    if len(probabilities) == 0:
        print("No probabilities—skipping peak detection.")
        return []

    # 1) Smooth the raw probability array
    smoothed = savgol_filter(probabilities, smooth_window, polyorder)

    # 2) Find peaks using ONLY distance & prominence
    peaks, properties = find_peaks(
        smoothed,
        distance=distance,
        prominence=prominence
    )

    # Convert indices → time in seconds
    timestamps = [float(idx) / fps for idx in peaks]

    # 3) Save a diagnostic plot to show smoothed curve + detected peaks
    times = np.arange(len(smoothed)) / fps
    plt.figure(figsize=(10, 4))
    plt.plot(times, smoothed, label='Smoothed Rep Probability')
    plt.plot(peaks / fps, smoothed[peaks], 'x', label='Detected Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    title_str = (
        f"Peak Detection on Smoothed Probability\n"
    )
    plt.title(title_str)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    peak_plot_path = os.path.join(PEAK_DIR, f"{base_name}.png")
    plt.savefig(peak_plot_path)
    plt.close()
    print(f"Peak-detection graph saved to: {peak_plot_path}")

    return timestamps


def overlay(
    input_video: str,
    output_path: str,
    probabilities: List[float],
    rep_timestamps: List[float],
    fps: float,
    font_scale: float = 0.8,
    thickness: int = 1
) -> None:
    """
    Overlay per-frame probability and a running rep count onto each frame, then write to a new video.
    No threshold is used to change text color—everything is drawn in white.

    Args:
        input_video: path to the original video file.
        output_path: where to save the annotated video file (MP4).
        probabilities: list of per-frame probabilities (floats in [0,1]).
        rep_timestamps: list of rep-start times (in seconds).
        fps: video frames-per-second.
        font_scale: scale of the overlay text.
        thickness: thickness of the overlay text strokes.
    """
    cap = cv2.VideoCapture(input_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    rep_count = 0
    current_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get current frame probability and time
        if frame_idx < len(probabilities):
            prob = probabilities[frame_idx]
            current_time = frame_idx / fps
        else:
            prob = 0.0

        # Count reps that have started before current time
        rep_count = sum(1 for t in rep_timestamps if t <= current_time)

        # Draw overlay text
        prob_text = f"Prob: {prob:.3f}"
        rep_text = f"Reps: {rep_count}"
        time_text = f"Time: {current_time:.1f}s"

        # Position text in top-left corner
        y_offset = 30
        cv2.putText(frame, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, rep_text, (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, time_text, (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def main():
    """Main function for segmentation testing."""
    parser = argparse.ArgumentParser(description='Test exercise repetition detection model')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--model', required=True, help='Model name (e.g., push-ups, squats)')
    parser.add_argument('--window', type=int, default=140, help='Window size for inference')
    parser.add_argument('--smooth', type=int, default=15, help='Smoothing window length')
    parser.add_argument('--prom', type=float, default=0.15, help='Peak detection prominence')
    parser.add_argument('--dist', type=int, default=25, help='Minimum distance between peaks')
    
    args = parser.parse_args()
    
    # Update model path to use new structure
    model_path = f'models/segmentation/{args.model}.keras'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Available models:")
        seg_dir = 'models/segmentation'
        if os.path.exists(seg_dir):
            for f in os.listdir(seg_dir):
                if f.endswith('.keras'):
                    print(f"  - {f.replace('.keras', '')}")
        return 1
    
    # Load model and extractor
    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    extractor = FeatureExtractor()
    
    # Get video info
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"Processing video: {args.input}")
    print(f"Video FPS: {fps}")
    
    # Compute probabilities
    print("Computing frame probabilities...")
    probabilities = compute_frame_probabilities(
        args.input, model, extractor, args.window
    )
    
    if not probabilities:
        print("No probabilities computed. Check if video contains valid poses.")
        return 1
    
    print(f"Computed probabilities for {len(probabilities)} frames")
    
    # Generate base name for output files
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # Create plots
    print("Generating plots...")
    plot_probabilities(probabilities, fps, os.path.join(PLOT_DIR, f"{base_name}.png"))
    
    # Detect repetitions
    print("Detecting repetitions...")
    rep_timestamps = calculate_rep_starts(
        probabilities, fps, base_name, args.dist, args.smooth, 2, args.prom
    )
    
    print(f"Detected {len(rep_timestamps)} repetitions")
    for i, timestamp in enumerate(rep_timestamps, 1):
        print(f"  Rep {i}: {timestamp:.2f}s")
    
    # Create annotated video
    print("Creating annotated video...")
    output_video = os.path.join(ANNOTATED_DIR, f"{base_name}_annotated.mp4")
    overlay(args.input, output_video, probabilities, rep_timestamps, fps)
    
    print(f"Results saved to:")
    print(f"  Annotated video: {output_video}")
    print(f"  Probability plot: {os.path.join(PLOT_DIR, f'{base_name}.png')}")
    print(f"  Peak detection plot: {os.path.join(PEAK_DIR, f'{base_name}.png')}")
    
    return 0


if __name__ == '__main__':
    exit(main())
 