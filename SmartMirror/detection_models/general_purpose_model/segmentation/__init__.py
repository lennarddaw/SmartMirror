"""
Segmentation Module

Contains all functionality for exercise repetition detection using TCN models.
"""

from .model import build_exercise_segmentation_model, ResidualTCNBlock
from .trainer import SegmentationTrainer, load_frame_dataset, create_sequences

__all__ = [
    'build_exercise_segmentation_model',
    'ResidualTCNBlock', 
    'SegmentationTrainer',
    'load_frame_dataset',
    'create_sequences'
] 