"""
Segmentation Training Module

Handles model training for exercise repetition detection using TCN architecture.
"""

import numpy as np
import os
import datetime
import argparse
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from .model import build_exercise_segmentation_model


def load_frame_dataset(path: str = 'dataset.npz') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load frame-based feature and label arrays from a .npz file.

    Args:
        path: Path to the NPZ file containing X and y arrays
        
    Returns:
        X_frames: np.ndarray of shape (total_frames, num_features)
        y_frames: np.ndarray of shape (total_frames,)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
        
    data = np.load(path)
    X = data['X']  # (total_frames, num_features)
    y = data['y']  # (total_frames,)
    
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    print(f"Positive samples: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    
    return X, y


def create_sequences(
    X_frames: np.ndarray,
    y_frames: np.ndarray,
    window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from frame data.

    Args:
        X_frames: Feature array of shape (total_frames, num_features)
        y_frames: Label array of shape (total_frames,)
        window_size: Length of time windows in frames

    Returns:
        X_seq: Sequence array of shape (num_windows, window_size, num_features)
        y_seq: Label array of shape (num_windows, window_size, 1)
    """
    num_frames, num_features = X_frames.shape
    num_windows = num_frames // window_size
    
    if num_windows == 0:
        raise ValueError(f"Video too short: {num_frames} frames, need at least {window_size}")
    
    # Create non-overlapping windows
    X_seq = np.array([X_frames[i*window_size:(i+1)*window_size]
                      for i in range(num_windows)])
    y_seq = np.array([y_frames[i*window_size:(i+1)*window_size].reshape(-1, 1)
                      for i in range(num_windows)])
    
    print(f"Created sequences: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    print(f"Positive windows: {np.sum(np.any(y_seq > 0.5, axis=1))} ({np.sum(np.any(y_seq > 0.5, axis=1))/len(y_seq)*100:.1f}%)")
    
    return X_seq, y_seq


class SegmentationTrainer:
    """
    Trains and saves the exercise segmentation model.
    """
    
    def __init__(self,
                 model_path: str = 'rep_segmenter_tcn.keras',
                 window_size: int = 140,
                 batch_size: int = 24,
                 epochs: int = 10000,
                 learning_rate: float = 5e-4,
                 validation_split: float = 0.2,
                 patience: int = 20):
        """
        Initialize the trainer.
        
        Args:
            model_path: Path to save the trained model
            window_size: Input sequence length
            batch_size: Training batch size (increased for higher capacity model)
            epochs: Maximum number of training epochs
            learning_rate: Learning rate for optimizer (increased for better convergence)
            validation_split: Fraction of data for validation
            patience: Early stopping patience (adjusted for faster convergence)
        """
        self.model_path = model_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.patience = patience

    def train(self, X_seq: np.ndarray, y_seq: np.ndarray) -> tf.keras.Model:
        """
        Train the segmentation model.
        
        Args:
            X_seq: Input sequences of shape (num_windows, window_size, num_features)
            y_seq: Target labels of shape (num_windows, window_size, 1)
            
        Returns:
            Trained model
        """
        # Determine input dimension from data
        input_dim = X_seq.shape[2]
        print(f"Building model with input_dim={input_dim}")
        
        # Build the model
        model = build_exercise_segmentation_model(
            input_dim=input_dim,
            learning_rate=self.learning_rate
        )
        
        # Configure callbacks
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        logdir = os.path.join('logs', f'segmenter_{timestamp}')
        
        callbacks = [
            TensorBoard(log_dir=logdir, histogram_freq=1),
            EarlyStopping(
                monitor='val_loss', 
                patience=self.patience, 
                restore_best_weights=True
            )
        ]
        
        print(f"Training logs will be saved to: {logdir}")
        print(f"Model will be saved to: {self.model_path}")
        
        # Train the model
        print(f"Starting training with {X_seq.shape[0]} sequences...")
        model.fit(
            X_seq,
            y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else '.', exist_ok=True)
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train exercise repetition detection model'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Exercise type (e.g., push-ups, squats)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset file (default: data/processed/dataset_{input}.npz)')
    parser.add_argument('--window', type=int, default=140,
                        help='Window size in frames')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--model-dir', type=str, default='models/segmentation',
                        help='Directory to save the model')
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.dataset is None:
        args.dataset = f'data/processed/dataset_{args.input}.npz'
    
    # Determine model path
    model_path = os.path.join(args.model_dir, f"{args.input}.keras")
    
    try:
        # 1. Load frames and labels
        print(f"Loading dataset: {args.dataset}")
        X_frames, y_frames = load_frame_dataset(args.dataset)
        
        # 2. Build sequences
        print(f"Creating sequences with window_size={args.window}")
        X_seq, y_seq = create_sequences(X_frames, y_frames, args.window)
        
        # 3. Train model
        print(f"Training model for {args.input}")
        trainer = SegmentationTrainer(
            model_path=model_path,
            window_size=args.window,
            batch_size=args.batch,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience
        )
        trainer.train(X_seq, y_seq)
        
        print(f"Training completed successfully!")
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 