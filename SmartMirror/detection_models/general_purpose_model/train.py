"""
trainer.py

Lädt ein Frame-basiertes Dataset (dataset.npz), erstellt Sequenzen mit Slide-Windows und trainiert das Modell.
"""
import numpy as np
import os
import datetime

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from model import build_exercise_segmentation_model


def load_frame_dataset(path: str = 'dataset.npz') -> tuple[np.ndarray, np.ndarray]:
    """
    Load frame-based feature and label arrays from a .npz file.

    Returns:
        X_frames: np.ndarray of shape (total_frames, num_features)
        y_frames: np.ndarray of shape (total_frames,)
    """
    data = np.load(path)
    X = data['X']  # (total_frames, num_features)
    y = data['y']  # (total_frames,)
    return X, y


def create_sequences(
    X_frames: np.ndarray,
    y_frames: np.ndarray,
    window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide non-overlapping windows over frame data to build sequences.

    Args:
        X_frames: (F, D)
        y_frames: (F,)
        window_size: Länge der Zeitfenster

    Returns:
        X_seq: (N, window_size, D)
        y_seq: (N, window_size, 1)
    """
    num_frames, num_features = X_frames.shape
    num_windows = num_frames // window_size
    X_seq = np.array([X_frames[i*window_size:(i+1)*window_size]
                      for i in range(num_windows)])
    y_seq = np.array([y_frames[i*window_size:(i+1)*window_size].reshape(-1,1)
                      for i in range(num_windows)])
    return X_seq, y_seq


class Trainer:
    """
    Trains and saves the exercise segmentation model.
    """
    def __init__(self,
                 model_path: str = 'rep_segmenter_tcn.keras',
                 window_size: int = 140):
        self.model_path = model_path
        self.window_size = window_size

    def train(self, X_seq: np.ndarray, y_seq: np.ndarray):
        input_dim = X_seq.shape[2]
        model = build_exercise_segmentation_model(input_dim=input_dim)
        logdir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        callbacks = [
            TensorBoard(log_dir=logdir, histogram_freq=1),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        model.fit(
            X_seq,
            y_seq,
            epochs=10000,
            batch_size=16,
            validation_split=0.2,
            callbacks=callbacks
        )
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train model using frame-based dataset and sliding windows'
    )
    parser.add_argument('--input', type=str, default='dataset.npz',
                        help='Pfad zum frame-basierten NPZ-Dataset')
    parser.add_argument('--window', type=int, default=140,
                        help='Fenstergröße in Frames')
    parser.add_argument('--model', type=str, default='rep_segmenter_tcn.keras',
                        help='Pfad zum gespeicherten Modell')
    args = parser.parse_args()

    # 1. Load frames and labels
    X_frames, y_frames = load_frame_dataset(args.input)
    print(f'Loaded frame dataset: X={X_frames.shape}, y={y_frames.shape}')

    # 2. Build sequences
    X_seq, y_seq = create_sequences(X_frames, y_frames, args.window)
    print(f'Created sequences: X_seq={X_seq.shape}, y_seq={y_seq.shape}')

    # 3. Train
    trainer = Trainer(model_path=args.model, window_size=args.window)
    trainer.train(X_seq, y_seq)
