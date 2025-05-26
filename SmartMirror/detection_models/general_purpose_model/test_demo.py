#!/usr/bin/env python3
"""
test.py

Reads a video, computes per-frame probabilities, writes annotated frames to a new video, performs peak detection on a smoothed probability signal,
and generates probability and peak-detection plots.
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

THRESHOLD = 0.42  # TODO: increase this to 0.5 after the more data is added and model confidence increases

# Output directories
ANNOTATED_DIR = 'output_videos/annotated'
PLOT_DIR = 'output_videos/plots'
PEAK_DIR = 'output_videos/peak_detection'
for d in (ANNOTATED_DIR, PLOT_DIR, PEAK_DIR):
    os.makedirs(d, exist_ok=True)


def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)


def compute_frame_probabilities(
    video_path: str,
    model: tf.keras.Model,
    extractor: FeatureExtractor,
    window_size: int = 140
) -> List[float]:
    cap = cv2.VideoCapture(video_path)
    features: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feat = extractor.extract_angles(frame)
        if feat is None:
            feat = np.zeros(model.input_shape[-1], dtype=np.float32)
        features.append(feat)
    cap.release()

    # pad at start so each frame has a full window
    pad = [np.zeros_like(features[0])] * (window_size - 1)
    feats_padded = pad + features

    probs: List[float] = []
    for i in range(len(features)):
        window = np.stack(feats_padded[i: i + window_size], axis=0)
        window = window[None, ...]
        pred = model.predict(window, verbose=0)
        probs.append(float(pred[0, -1, 0]))
    return probs


def plot_probabilities(
    probabilities: List[float],
    fps: float,
    output_path: str
) -> None:
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
    threshold: float = THRESHOLD,
    distance: int = 20,
    smooth_window: int = 11,
    polyorder: int = 2
) -> List[float]:
    """
    Smooth the probability signal, detect peaks corresponding to rep starts,
    save a peak-detection plot, and return a list of timestamps (in seconds).
    """
    smoothed = savgol_filter(probabilities, smooth_window, polyorder)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=distance)
    timestamps = [float(p) / fps for p in peaks]

    # Plot smoothed signal with peaks
    times = np.arange(len(smoothed)) / fps
    plt.figure(figsize=(10, 4))
    plt.plot(times, smoothed, label='Smoothed Rep Probability')
    plt.plot(np.array(peaks) / fps, smoothed[peaks], 'x', label='Detected Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.title('Peak Detection on Smoothed Probability')
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
    threshold: float = THRESHOLD,
    font_scale: float = 0.8,
    thickness: int = 1
) -> None:
    """
    Overlay per-frame probability and a running rep count onto video frames.
    """
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx >= len(probabilities):
            break

        prob = probabilities[idx]
        time_sec = idx / fps
        count = sum(1 for t in rep_timestamps if t <= time_sec)

        # Texts
        prob_text = f"Rep Prob: {prob:.2f}"
        count_text = f"Reps: {count}"
        prob_pos = (10, 30)
        count_pos = (width - 150, 30)
        color = (0, 0, 255) if prob > threshold else (0, 255, 0)

        cv2.putText(frame, prob_text, prob_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(frame, count_text, count_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        out.write(frame)
        idx += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--model", default="rep_segmenter_tcn.keras", help="Path to model")
    parser.add_argument("--window", type=int, default=140, help="Window size for TCN input")
    args = parser.parse_args()

    extractor = FeatureExtractor()
    model = load_model(args.model)

    # Compute raw probabilities per frame
    probs = compute_frame_probabilities(args.input, model, extractor, args.window)

    # Get video's FPS
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    # Plot and save raw probability graph
    prob_plot_path = os.path.join(PLOT_DIR, f"{base_name}.png")
    plot_probabilities(probs, fps, prob_plot_path)
    print(f"Probability graph saved to: {prob_plot_path}")

    # Perform peak detection
    rep_timestamps = calculate_rep_starts(probs, fps, base_name)

    # Overlay probabilities and rep count
    annotated_video_path = os.path.join(ANNOTATED_DIR, f"{base_name}.mp4")
    overlay(args.input, annotated_video_path, probs, rep_timestamps, fps)
    print(f"Annotated video saved to: {annotated_video_path}")
