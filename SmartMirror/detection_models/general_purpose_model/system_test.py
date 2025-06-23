#!/usr/bin/env python3
"""
System Test Script

Tests the cleaned up exercise detection system to ensure all components work correctly.
"""

import os
import sys
import numpy as np
import tensorflow as tf


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from segmentation.model import build_exercise_segmentation_model, ResidualTCNBlock
        print("✓ Segmentation model module imported successfully")
    except Exception as e:
        print(f"✗ Segmentation model module import failed: {e}")
        return False
    
    try:
        from segmentation.trainer import SegmentationTrainer, load_frame_dataset, create_sequences
        print("✓ Segmentation trainer module imported successfully")
    except Exception as e:
        print(f"✗ Segmentation trainer module import failed: {e}")
        return False
    
    try:
        from dataset_builder import FeatureExtractor, SegmentationDatasetBuilder, ClassificationDatasetBuilder, LabelAugmenter
        print("✓ Dataset builder module imported successfully")
    except Exception as e:
        print(f"✗ Dataset builder module import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation and compilation."""
    print("\nTesting model creation...")
    
    try:
        from segmentation.model import build_exercise_segmentation_model
        
        # Create model
        model = build_exercise_segmentation_model(input_dim=25)
        
        # Test model properties
        assert model.input_shape == (None, None, 25), f"Expected input shape (None, None, 25), got {model.input_shape}"
        assert model.output_shape == (None, None, 1), f"Expected output shape (None, None, 1), got {model.output_shape}"
        
        # Test model compilation
        assert model.optimizer is not None, "Model should have optimizer"
        assert model.loss is not None, "Model should have loss function"
        assert len(model.metrics) > 0, "Model should have metrics"
        
        print("✓ Model created and compiled successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Parameters: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_feature_extractor():
    """Test feature extraction."""
    print("\nTesting feature extraction...")
    
    try:
        from dataset_builder import FeatureExtractor
        
        # Create extractor
        extractor = FeatureExtractor()
        
        # Test feature dimension
        feature_dim = extractor.get_feature_dimension()
        assert feature_dim == 25, f"Expected 25 features, got {feature_dim}"
        
        print("✓ Feature extractor created successfully")
        print(f"  Feature dimension: {feature_dim}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature extractor test failed: {e}")
        return False


def test_data_structures():
    """Test data processing structures."""
    print("\nTesting data structures...")
    
    try:
        from dataset_builder import LabelAugmenter
        
        # Test label augmentation
        augmenter = LabelAugmenter(fps=30, margin_sec=0.1)
        
        # Create test labels
        labels = np.zeros(100)
        labels[50] = 1.0  # Single positive sample
        
        # Augment labels
        augmented = augmenter.augment(labels)
        
        # Check that augmentation expanded the positive region
        margin_frames = int(30 * 0.1)  # 3 frames
        expected_ones = 2 * margin_frames + 1  # Original + margin on both sides
        
        actual_ones = np.sum(augmented)
        assert actual_ones >= expected_ones, f"Expected at least {expected_ones} ones, got {actual_ones}"
        
        print("✓ Label augmentation works correctly")
        print(f"  Original positives: {np.sum(labels)}")
        print(f"  Augmented positives: {actual_ones}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data structures test failed: {e}")
        return False


def test_sequence_creation():
    """Test sequence creation from frame data."""
    print("\nTesting sequence creation...")
    
    try:
        from segmentation.trainer import create_sequences
        
        # Create test data
        num_frames = 300
        num_features = 25
        window_size = 140
        
        X_frames = np.random.random((num_frames, num_features))
        y_frames = np.random.randint(0, 2, num_frames).astype(float)
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_frames, y_frames, window_size)
        
        # Check shapes
        expected_windows = num_frames // window_size
        assert X_seq.shape == (expected_windows, window_size, num_features), \
            f"Expected X_seq shape ({expected_windows}, {window_size}, {num_features}), got {X_seq.shape}"
        assert y_seq.shape == (expected_windows, window_size, 1), \
            f"Expected y_seq shape ({expected_windows}, {window_size}, 1), got {y_seq.shape}"
        
        print("✓ Sequence creation works correctly")
        print(f"  Input frames: {num_frames}")
        print(f"  Window size: {window_size}")
        print(f"  Output sequences: {X_seq.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sequence creation test failed: {e}")
        return False


def test_file_structure():
    """Test that the file structure is correct."""
    print("\nTesting file structure...")
    
    required_files = [
        'main.py',
        'dataset_builder.py', 
        'segmentation_demo.py',
        'video_labeler.py',
        'system_test.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'segmentation',
        'classification',
        'models/segmentation',
        'models/classification',
        'data/raw',
        'data/labels',
        'data/processed',
        'test_videos',
        'output_videos',
        'logs'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    missing_dirs = []
    for dir in required_dirs:
        if not os.path.exists(dir):
            missing_dirs.append(dir)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    
    print("✓ File structure is correct")
    print(f"  Files: {len(required_files)}")
    print(f"  Directories: {len(required_dirs)}")
    
    return True


def test_models_exist():
    """Test that trained models exist."""
    print("\nTesting model availability...")
    
    models_dir = 'models/segmentation'
    if not os.path.exists(models_dir):
        print("✗ Segmentation models directory not found")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    
    if not model_files:
        print("✗ No trained segmentation models found")
        return False
    
    print("✓ Trained segmentation models found:")
    for model in model_files:
        model_path = os.path.join(models_dir, model)
        size_kb = os.path.getsize(model_path) / 1024
        print(f"  - {model} ({size_kb:.1f} KB)")
    
    return True


def main():
    """Run all tests."""
    print("Exercise Detection System - System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_feature_extractor,
        test_data_structures,
        test_sequence_creation,
        test_file_structure,
        test_models_exist
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1


if __name__ == '__main__':
    exit(main()) 