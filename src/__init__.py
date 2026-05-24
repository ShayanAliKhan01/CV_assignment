"""
Pose Estimation & Activity Classification Pipeline
Package: pose_classification
"""

__version__ = "1.1.0"
__author__ = "CCP Assignment"

from .pose_estimator import PoseEstimator
from .angle_calculator import AngleCalculator
from .data_processor import DataProcessor
from .classifier import ActivityClassifier
from .evaluator import Evaluator

__all__ = [
    'PoseEstimator',
    'AngleCalculator',
    'DataProcessor',
    'ActivityClassifier',
    'Evaluator'
]
