"""
Classification Module

Contains all functionality for exercise type classification.
This module will be implemented in future updates.
"""

from .model import (
    build_exercise_classification_model,
    get_model_summary,
    save_model_plot
)

from .trainer import (
    ClassificationTrainer,
    load_classification_dataset,
    prepare_classification_data
)

__all__ = [
    # Model functions
    'build_exercise_classification_model',
    'get_model_summary', 
    'save_model_plot',
    
    # Training classes and functions
    'ClassificationTrainer',
    'load_classification_dataset',
    'prepare_classification_data'
]

# TODO: Implement classification functionality
# - Classification model architecture
# - Classification trainer
# - Classification inference 