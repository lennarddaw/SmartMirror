#!/usr/bin/env python3
"""
Main CLI for Exercise Detection System

Provides easy access to all system functionality including training, testing, and data processing
for both segmentation and classification models.
"""

import argparse
import sys
import os


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Exercise Detection System - Segmentation & Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build segmentation dataset
  python main.py build-dataset --mode segmentation --exercise push-ups
  
  # Build classification dataset
  python main.py build-dataset --mode classification
  
  # Train segmentation model
  python main.py train --mode segmentation --exercise push-ups
  
  # Test segmentation model
  python main.py test --mode segmentation --input test_videos/video.mp4 --model push-ups
  
  # Label videos (interactive)
  python main.py label --exercise push-ups
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build dataset command
    build_parser = subparsers.add_parser('build-dataset', help='Build training datasets')
    build_parser.add_argument('--mode', required=True, choices=['segmentation', 'classification'],
                             help='Dataset mode')
    build_parser.add_argument('--exercise', 
                             help='Exercise type for segmentation mode (e.g., push-ups, squats)')
    build_parser.add_argument('--output', 
                             help='Output dataset path')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train exercise detection models')
    train_parser.add_argument('--mode', required=True, choices=['segmentation', 'classification'],
                             help='Training mode')
    train_parser.add_argument('--exercise', 
                             help='Exercise type for segmentation mode')
    train_parser.add_argument('--dataset',
                             help='Dataset path')
    train_parser.add_argument('--window', type=int, default=140,
                             help='Window size in frames (segmentation only)')
    train_parser.add_argument('--batch', type=int, default=16,
                             help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=10000,
                             help='Maximum training epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test models on video')
    test_parser.add_argument('--mode', required=True, choices=['segmentation', 'classification'],
                            help='Test mode')
    test_parser.add_argument('--input', required=True,
                            help='Input video file path')
    test_parser.add_argument('--model', 
                            help='Model name (for segmentation mode)')
    test_parser.add_argument('--window', type=int, default=140,
                            help='Window size for inference (segmentation only)')
    test_parser.add_argument('--smooth', type=int, default=15,
                            help='Smoothing window length (segmentation only)')
    test_parser.add_argument('--prom', type=float, default=0.15,
                            help='Peak detection prominence (segmentation only)')
    test_parser.add_argument('--dist', type=int, default=25,
                            help='Minimum distance between peaks (segmentation only)')
    
    # Label command
    label_parser = subparsers.add_parser('label', help='Label videos for training')
    label_parser.add_argument('--exercise', required=True,
                             help='Exercise type to label')
    
    # Model info command
    info_parser = subparsers.add_parser('model-info', help='Show model information')
    info_parser.add_argument('--mode', required=True, choices=['segmentation', 'classification'],
                            help='Model type')
    info_parser.add_argument('--exercise', 
                            help='Exercise type (for segmentation mode)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'build-dataset':
            return build_dataset(args)
        elif args.command == 'train':
            return train_model(args)
        elif args.command == 'test':
            return test_model(args)
        elif args.command == 'label':
            return label_videos(args)
        elif args.command == 'model-info':
            return show_model_info(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1


def build_dataset(args):
    """Build training datasets."""
    from dataset_builder import SegmentationDatasetBuilder, ClassificationDatasetBuilder
    
    if args.mode == 'segmentation':
        if not args.exercise:
            raise ValueError("Exercise type required for segmentation mode")
        
        # Determine output path
        if args.output is None:
            args.output = f'data/processed/dataset_{args.exercise}.npz'
        
        print(f"Building segmentation dataset for {args.exercise}...")
        
        builder = SegmentationDatasetBuilder()
        X, y = builder.build(exercise_type=args.exercise)
        builder.save(X, y, path=args.output)
        
    elif args.mode == 'classification':
        # Determine output path
        if args.output is None:
            args.output = 'data/processed/classification_dataset.npz'
        
        print("Building classification dataset from all exercise types...")
        
        builder = ClassificationDatasetBuilder()
        X, y, label_mapping = builder.build()
        builder.save(X, y, label_mapping, path=args.output)
    
    print(f"Dataset built successfully: {args.output}")
    return 0


def train_model(args):
    """Train exercise detection models."""
    if args.mode == 'segmentation':
        if not args.exercise:
            raise ValueError("Exercise type required for segmentation mode")
        
        from segmentation.trainer import load_frame_dataset, create_sequences, SegmentationTrainer
        
        # Determine dataset path
        if args.dataset is None:
            args.dataset = f'data/processed/dataset_{args.exercise}.npz'
        
        # Determine model path
        model_path = f'models/segmentation/{args.exercise}.keras'
        
        print(f"Training segmentation model for {args.exercise}...")
        
        # Load dataset
        X_frames, y_frames = load_frame_dataset(args.dataset)
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_frames, y_frames, args.window)
        
        # Train model
        trainer = SegmentationTrainer(
            model_path=model_path,
            window_size=args.window,
            batch_size=args.batch,
            epochs=args.epochs,
            learning_rate=args.lr
        )
        trainer.train(X_seq, y_seq)
        
        print(f"Training completed: {model_path}")
        
    elif args.mode == 'classification':
        from classification.trainer import load_classification_dataset, prepare_classification_data, ClassificationTrainer
        
        # Determine dataset path
        if args.dataset is None:
            args.dataset = 'data/processed/classification_dataset.npz'
        
        # Determine model path
        model_path = 'models/classification/exercise_classifier.keras'
        
        print("Training classification model...")
        
        # Load dataset
        X, y, class_names = load_classification_dataset(args.dataset)
        
        # Prepare data splits
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_classification_data(X, y)
        
        # Train model
        trainer = ClassificationTrainer(
            model_path=model_path,
            batch_size=args.batch,
            epochs=args.epochs,
            learning_rate=args.lr,
            class_names=class_names
        )
        trainer.train(X_train, y_train, X_val, y_val)
        
        print(f"Training completed: {model_path}")
    
    return 0


def test_model(args):
    """Test models on video."""
    import subprocess
    
    if args.mode == 'segmentation':
        if not args.model:
            raise ValueError("Model name required for segmentation mode")
        
        # Check if model exists
        model_path = f'models/segmentation/{args.model}.keras'
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Available segmentation models:")
            seg_dir = 'models/segmentation'
            if os.path.exists(seg_dir):
                for f in os.listdir(seg_dir):
                    if f.endswith('.keras'):
                        print(f"  - {f.replace('.keras', '')}")
            return 1
        
        # Build test command
        cmd = [
            'python', 'segmentation_demo.py',
            '--input', args.input,
            '--model', args.model,
            '--window', str(args.window),
            '--smooth', str(args.smooth),
            '--prom', str(args.prom),
            '--dist', str(args.dist)
        ]
        
        print(f"Testing {args.model} segmentation model on {args.input}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run test
        result = subprocess.run(cmd)
        return result.returncode
        
    elif args.mode == 'classification':
        # Check if model exists
        model_path = 'models/classification/exercise_classifier.keras'
        if not os.path.exists(model_path):
            print(f"Classification model not found: {model_path}")
            print("Please train the classification model first using:")
            print("  python main.py train --mode classification")
            return 1
        
        # Build test command
        cmd = [
            'python', 'classification_demo.py',
            '--model', model_path,
            '--dataset', 'data/processed/classification_dataset.npz',
            '--save-plots'
        ]
        
        print(f"Testing classification model...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run test
        result = subprocess.run(cmd)
        return result.returncode


def label_videos(args):
    """Open labeling tool for videos."""
    import subprocess
    
    video_dir = f'data/raw/{args.exercise}'
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return 1
    
    print(f"Opening labeling tool for {args.exercise}...")
    print(f"Video directory: {video_dir}")
    
    # Run labeling tool
    cmd = ['python', 'video_labeler.py']
    result = subprocess.run(cmd)
    return result.returncode


def show_model_info(args):
    """Show information about trained models."""
    import tensorflow as tf
    
    if args.mode == 'segmentation':
        if not args.exercise:
            print("Available segmentation models:")
            seg_dir = 'models/segmentation'
            if os.path.exists(seg_dir):
                for f in os.listdir(seg_dir):
                    if f.endswith('.keras'):
                        print(f"  - {f.replace('.keras', '')}")
            return 0
        
        model_path = f'models/segmentation/{args.exercise}.keras'
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return 1
        
        print(f"Segmentation Model: {model_path}")
        print(f"File size: {os.path.getsize(model_path) / 1024:.1f} KB")
        
        # Load and show model info
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            print(f"Total parameters: {model.count_params():,}")
            
            # Show model summary
            print("\nModel Summary:")
            model.summary()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1
            
    elif args.mode == 'classification':
        model_path = 'models/classification/exercise_classifier.keras'
        if not os.path.exists(model_path):
            print(f"Classification model not found: {model_path}")
            print("Please train the classification model first using:")
            print("  python main.py train --mode classification")
            return 1
        
        print(f"Classification Model: {model_path}")
        print(f"File size: {os.path.getsize(model_path) / 1024:.1f} KB")
        
        # Load and show model info
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            print(f"Total parameters: {model.count_params():,}")
            
            # Load class names
            class_names_path = model_path.replace('.keras', '_classes.txt')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    class_names = [line.strip() for line in f.readlines()]
                print(f"Classes: {class_names}")
            
            # Show model summary
            print("\nModel Summary:")
            model.summary()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 