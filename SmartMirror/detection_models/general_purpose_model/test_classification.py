#!/usr/bin/env python3
"""
Test Classification Components

Tests the classification model, trainer, and demo functionality.
"""

import numpy as np
import os
import sys
import tempfile

def test_classification_model():
    """Test the classification model creation."""
    print("Testing classification model creation...")
    
    try:
        from classification.model import build_exercise_classification_model
        
        # Create model
        model = build_exercise_classification_model(
            input_dim=25,
            num_classes=4,
            num_filters=32,
            num_layers=2
        )
        
        print(f"✓ Model created successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Parameters: {model.count_params():,}")
        
        # Test model prediction
        test_input = np.random.random((1, 100, 25))  # 1 sample, 100 frames, 25 features
        prediction = model.predict(test_input, verbose=0)
        
        print(f"✓ Model prediction works")
        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Prediction sum: {np.sum(prediction):.4f} (should be ~1.0)")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_classification_trainer():
    """Test the classification trainer."""
    print("\nTesting classification trainer...")
    
    try:
        from classification.trainer import ClassificationTrainer, prepare_classification_data
        
        # Create dummy data
        num_samples = 50
        sequence_length = 100
        num_features = 25
        num_classes = 4
        
        X = np.random.random((num_samples, sequence_length, num_features))
        y = np.random.randint(0, num_classes, num_samples)
        
        print(f"✓ Created dummy data: X={X.shape}, y={y.shape}")
        
        # Test data preparation
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_classification_data(X, y)
        
        print(f"✓ Data preparation works")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        # Test trainer creation
        trainer = ClassificationTrainer(
            model_path='test_classifier.keras',
            batch_size=8,
            epochs=2,
            learning_rate=1e-3,
            class_names=['push-ups', 'squats', 'pull-ups', 'dips']
        )
        
        print(f"✓ Trainer created successfully")
        
        # Clean up
        if os.path.exists('test_classifier.keras'):
            os.remove('test_classifier.keras')
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        return False


def test_classification_demo():
    """Test the classification demo components."""
    print("\nTesting classification demo components...")
    
    try:
        from classification_demo import load_trained_model, predict_exercise_type
        
        # Create a temporary model for testing
        from classification.model import build_exercise_classification_model
        
        model = build_exercise_classification_model(
            input_dim=25,
            num_classes=4
        )
        
        # Save model temporarily
        temp_model_path = 'temp_test_model.keras'
        model.save(temp_model_path)
        
        # Save class names
        class_names = ['push-ups', 'squats', 'pull-ups', 'dips']
        class_names_path = temp_model_path.replace('.keras', '_classes.txt')
        with open(class_names_path, 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        # Test model loading
        loaded_model, loaded_class_names = load_trained_model(temp_model_path)
        
        print(f"✓ Model loading works")
        print(f"  Loaded class names: {loaded_class_names}")
        
        # Test prediction
        test_input = np.random.random((2, 100, 25))
        predictions, probabilities = predict_exercise_type(loaded_model, test_input, loaded_class_names)
        
        print(f"✓ Prediction works")
        print(f"  Predictions: {predictions}")
        print(f"  Probabilities shape: {probabilities.shape}")
        
        # Clean up
        os.remove(temp_model_path)
        os.remove(class_names_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Demo test failed: {e}")
        return False


def main():
    """Run all classification tests."""
    print("="*60)
    print("CLASSIFICATION COMPONENT TESTS")
    print("="*60)
    
    tests = [
        test_classification_model,
        test_classification_trainer,
        test_classification_demo
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    print(f"TESTS COMPLETED: {passed}/{total} passed")
    print("="*60)
    
    if passed == total:
        print("✓ All classification components working correctly!")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit(main()) 