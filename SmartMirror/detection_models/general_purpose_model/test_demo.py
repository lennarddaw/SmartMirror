#!/usr/bin/env python3
"""
test.py

Reads a video, computes per-frame probabilities, writes annotated frames to a new video, and generates a probability-time graph.
"""
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List
from data_processing import FeatureExtractor
import os
import argparse


ANNOTATED_DIR = 'output_videos/annotated'
PLOT_DIR = 'output_videos/plots'
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)


def compute_frame_probabilities(video_path: str, model: tf.keras.Model, extractor: FeatureExtractor, window_size: int = 140) -> List[float]:
    cap = cv2.VideoCapture(video_path)
    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feat = extractor.extract_angles(frame)
        if feat is None:
            feat = np.zeros(model.input_shape[-1], dtype=np.float32)
        features.append(feat)
    cap.release()

    pad = [np.zeros_like(features[0])] * (window_size - 1)
    feats_padded = pad + features

    probs = []
    for i in range(len(features)):
        window = np.stack(feats_padded[i: i + window_size], axis=0)
        window = window[None, ...]
        pred = model.predict(window, verbose=0)
        probs.append(float(pred[0, -1, 0]))
    return probs


def overlay_and_validate(input_video: str, output_path: str, probabilities: List[float], position=(10, 30), font_scale=0.8, thickness=1) -> bool:
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
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
        color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)
        text = f"Rep Prob: {prob:.2f}"

        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        out.write(frame)

        idx += 1

    cap.release()
    out.release()

    cap_check = cv2.VideoCapture(output_path)
    written_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_check.release()

    return written_frames == len(probabilities)


def plot_probabilities(probabilities: List[float], fps: float, output_path: str):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--model", default="rep_segmenter_tcn.keras", help="Path to model")
    parser.add_argument("--window", type=int, default=140, help="Window size")
    args = parser.parse_args()

    extractor = FeatureExtractor()
    model = load_model(args.model)

    probs = compute_frame_probabilities(args.input, model, extractor, args.window)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    annotated_video_path = os.path.join(ANNOTATED_DIR, f"{base_name}.mp4")
    if not overlay_and_validate(args.input, annotated_video_path, probs):
        exit(1)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    plot_output_path = os.path.join(PLOT_DIR, f"{base_name}.png")
    plot_probabilities(probs, fps, plot_output_path)
    print(f"Probability graph saved to: {plot_output_path}")
