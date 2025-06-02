#!/usr/bin/env python3
"""
test.py

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

from data_processing import FeatureExtractor

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
        timestamps: a list of “rep start” times (in seconds) corresponding to each detected peak.
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
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx >= len(probabilities):
            break

        prob = probabilities[idx]
        time_sec = idx / fps
        # Count how many reps have already started up to this time
        count = sum(1 for t in rep_timestamps if t <= time_sec)

        prob_text  = f"Rep Prob: {prob:.2f}"
        count_text = f"Reps: {count}"
        # Draw both texts in white (BGR = (255, 255, 255))
        prob_pos   = (10, 30)
        count_pos  = (width - 150, 30)
        color      = (255, 255, 255)

        cv2.putText(frame, prob_text,  prob_pos,  cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(frame, count_text, count_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        out.write(frame)
        idx += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True,
        help="Path to input video file (e.g. video.mp4)"
    )
    parser.add_argument(
        "--model", default="rep_segmenter_tcn",
        help="Model name (without .keras). Will load models/<model>.keras"
    )
    parser.add_argument(
        "--window", type=int, default=140,
        help="Window size for the TCN input. Must match how the model was trained."
    )
    parser.add_argument(
        "--smooth", type=int, default=15,
        help="Savitzky–Golay smoothing window length (must be odd). Default: 15"
    )
    parser.add_argument(
        "--prom", type=float, default=0.15,
        help="Minimum prominence for peak detection. Default: 0.15"
    )
    parser.add_argument(
        "--dist", type=int, default=25,
        help="Minimum distance (in frames) between consecutive peaks. Default: 25"
    )
    args = parser.parse_args()

    # 1) Initialize feature extractor & load the Keras model
    extractor = FeatureExtractor()
    model     = load_model(f"models/{args.model}.keras")

    # 2) Compute per-frame probabilities
    probs = compute_frame_probabilities(
        video_path  = args.input,
        model       = model,
        extractor   = extractor,
        window_size = args.window
    )

    # 3) Read the video again—just to grab fps
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    # 4) Plot & save the raw probability curve
    prob_plot_path = os.path.join(PLOT_DIR, f"{base_name}.png")
    plot_probabilities(probs, fps, prob_plot_path)
    print(f"Probability graph saved to: {prob_plot_path}")

    # 5) Perform peak detection (using only prominence + distance, with our new defaults)
    rep_timestamps = calculate_rep_starts(
        probabilities  = probs,
        fps            = fps,
        base_name      = base_name,
        distance       = args.dist,
        smooth_window  = args.smooth,
        polyorder      = 2,
        prominence     = args.prom
    )

    # 6) Overlay probabilities & rep count onto the video frames
    annotated_video_path = os.path.join(ANNOTATED_DIR, f"{base_name}.mp4")
    overlay(
        input_video     = args.input,
        output_path     = annotated_video_path,
        probabilities   = probs,
        rep_timestamps  = rep_timestamps,
        fps             = fps
    )
    print(f"Annotated video saved to: {annotated_video_path}")
