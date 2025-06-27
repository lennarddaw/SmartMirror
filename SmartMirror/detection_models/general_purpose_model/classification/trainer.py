"""
Classification Training Module

Handles model training for exercise type classification using TCN architecture.
"""

import numpy as np
import os
import datetime
import argparse
from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .model import build_exercise_classification_model


def load_classification_dataset(path: str = 'classification_dataset.npz') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load classification dataset with features and exercise type labels.

    Args:
        path: Path to the NPZ file containing X, y, and class_names arrays
        
    Returns:
        X: np.ndarray of shape (num_samples, sequence_length, num_features)
        y: np.ndarray of shape (num_samples,) with class indices
        class_names: List of exercise class names
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
        
    data = np.load(path, allow_pickle=True)
    X = data['X']  # (num_samples, num_features)
    y = data['y']  # (num_samples,) - class indices
    
    # Reshape X to add sequence dimension if needed
    if len(X.shape) == 2:
        # Add sequence dimension: (num_samples, 1, num_features)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        print(f"Reshaped X from {data['X'].shape} to {X.shape}")
    
    # Handle class_names - provide defaults if not present
    if 'class_names' in data:
        class_names = data['class_names'].tolist()
    elif 'label_mapping' in data:
        # Use label_mapping if available
        label_mapping = data['label_mapping'].item()
        class_names = list(label_mapping.values())
        print(f"Using label_mapping for class names: {class_names}")
    else:
        # Default class names based on the number of unique classes
        num_classes = len(np.unique(y))
        default_names = ['push-ups', 'squats', 'pull-ups', 'dips']
        class_names = default_names[:num_classes]
        print(f"Warning: No class_names found in dataset, using defaults: {class_names}")
    
    print(f"Loaded classification dataset: X={X.shape}, y={y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    if class_names:
        print(f"Class names: {class_names}")
        for i, class_name in enumerate(class_names):
            count = np.sum(y == i)
            print(f"  {class_name}: {count} samples")
    
    return X, y, class_names


def prepare_classification_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for classification training with train/validation/test splits.

    Args:
        X: Feature array of shape (num_samples, sequence_length, num_features)
        y: Label array of shape (num_samples,)
        test_size: Fraction of data for testing
        val_size: Fraction of remaining data for validation
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Split datasets
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Data splits:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class ClassificationTrainer:
    """
    Trains and saves the exercise classification model.
    """
    
    def __init__(self,
                 model_path: str = 'exercise_classifier.keras',
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 1e-3,
                 patience: int = 15,
                 class_names: List[str] = None):
        """
        Initialize the trainer.
        
        Args:
            model_path: Path to save the trained model
            batch_size: Training batch size
            epochs: Maximum number of training epochs
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            class_names: List of exercise class names
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.class_names = class_names or ['push-ups', 'squats', 'pull-ups', 'dips']

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """
        Train the classification model.
        
        Args:
            X_train: Training sequences of shape (num_train, sequence_length, num_features)
            y_train: Training labels of shape (num_train,)
            X_val: Validation sequences of shape (num_val, sequence_length, num_features)
            y_val: Validation labels of shape (num_val,)
            
        Returns:
            Trained model
        """
        # Determine input dimension and number of classes from data
        input_dim = X_train.shape[2]
        num_classes = len(np.unique(y_train))
        print(f"Building classification model with input_dim={input_dim}, num_classes={num_classes}")
        
        # Build the model
        model = build_exercise_classification_model(
            input_dim=input_dim,
            num_classes=num_classes,
            learning_rate=self.learning_rate
        )
        
        # Configure callbacks
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        logdir = os.path.join('logs', f'classifier_{timestamp}')
        
        callbacks = [
            TensorBoard(log_dir=logdir, histogram_freq=1),
            EarlyStopping(
                monitor='val_accuracy', 
                patience=self.patience, 
                restore_best_weights=True,
                mode='max'
            ),
            ModelCheckpoint(
                filepath=self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        print(f"Training logs will be saved to: {logdir}")
        print(f"Model will be saved to: {self.model_path}")
        
        # Train the model
        print(f"Starting training with {X_train.shape[0]} training samples...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        
        # Save class names with the model
        if self.class_names:
            class_names_path = self.model_path.replace('.keras', '_classes.txt')
            with open(class_names_path, 'w') as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")
            print(f"Class names saved to: {class_names_path}")
        
        print(f"Training completed successfully!")
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        
        return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train exercise type classification model'
    )
    parser.add_argument('--dataset', type=str, default='data/processed/classification_dataset.npz',
                        help='Path to classification dataset file')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--model-dir', type=str, default='models/classification',
                        help='Directory to save the model')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for testing')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Fraction of remaining data for validation')
    
    args = parser.parse_args()
    
    # Determine model path
    model_path = os.path.join(args.model_dir, "exercise_classifier.keras")
    
    try:
        # 1. Load classification dataset
        print(f"Loading classification dataset: {args.dataset}")
        X, y, class_names = load_classification_dataset(args.dataset)
        
        # 2. Prepare data splits
        print(f"Preparing data splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_classification_data(
            X, y, test_size=args.test_size, val_size=args.val_size
        )
        
        # 3. Train model
        print(f"Training classification model...")
        trainer = ClassificationTrainer(
            model_path=model_path,
            batch_size=args.batch,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            class_names=class_names
        )
        model = trainer.train(X_train, y_train, X_val, y_val)
        
        # 4. Evaluate on test set
        print(f"Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        print(f"Training completed successfully!")
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 